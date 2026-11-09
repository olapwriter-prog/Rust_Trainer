//! Power- and heat-aware overnight runs.
//!
//! Training runs only when it is harmless: in the charging **Deep** stage, with
//! the battery above a set level and the temperature under the heat limit.
//! `agentd` publishes the device state (it already owns staging and the
//! chip-priority Training mode); the trainer reads it before every example and
//! pauses, with a checkpoint, the moment a condition fails. Resuming needs a
//! cooler temperature than pausing (hysteresis), so it does not flap at the limit.
//!
//! When training finishes the candidate goes through the eval gate against the
//! *current* model (base + active adapter), and is promoted or thrown away. Every
//! transition is written to the registry ledger.

use crate::adapter::{base_fingerprint, load_adapter_into, save_adapter, AdapterMeta};
use crate::eval::{gate, measure, EvalReport, GateConfig};
use crate::model::Model;
use crate::registry::{now_unix, Registry};
use crate::tokenizer::Example;
use crate::train::{Control, PrefPair, StopReason, TrainConfig, TrainData, Trainer};
use serde::{Deserialize, Serialize};
use std::io;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Active,
    Idle,
    Light,
    Deep,
}

/// Device state as published by `agentd`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeviceState {
    pub stage: Stage,
    pub charging: bool,
    pub battery_pct: f32,
    pub temp_c: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PowerPolicy {
    pub required_stage: Stage,
    pub require_charging: bool,
    pub min_battery_pct: f32,
    pub max_temp_c: f32,
    /// A paused run resumes only below `max_temp_c - resume_margin_c` and above
    /// `min_battery_pct + resume_margin_pct`.
    pub resume_margin_c: f32,
    pub resume_margin_pct: f32,
}

impl Default for PowerPolicy {
    fn default() -> Self {
        PowerPolicy {
            required_stage: Stage::Deep,
            require_charging: true,
            min_battery_pct: 50.0,
            max_temp_c: 40.0,
            resume_margin_c: 3.0,
            resume_margin_pct: 5.0,
        }
    }
}

impl PowerPolicy {
    /// `Ok(())` if training may (continue to) run; otherwise the reason it may not.
    pub fn check(&self, s: &DeviceState, running: bool) -> Result<(), String> {
        let (temp_lim, batt_lim) = if running {
            (self.max_temp_c, self.min_battery_pct)
        } else {
            (self.max_temp_c - self.resume_margin_c, self.min_battery_pct + self.resume_margin_pct)
        };
        if s.stage != self.required_stage {
            return Err(format!("stage {:?} (need {:?})", s.stage, self.required_stage));
        }
        if self.require_charging && !s.charging {
            return Err("not charging".into());
        }
        if s.battery_pct < batt_lim {
            return Err(format!("battery {:.0}% < {:.0}%", s.battery_pct, batt_lim));
        }
        if s.temp_c >= temp_lim {
            return Err(format!("temperature {:.1}°C ≥ {:.1}°C", s.temp_c, temp_lim));
        }
        Ok(())
    }
}

pub trait DeviceMonitor {
    fn state(&mut self) -> io::Result<DeviceState>;
}

/// Reads the JSON status file `agentd` keeps up to date.
pub struct FileMonitor(pub PathBuf);

impl DeviceMonitor for FileMonitor {
    fn state(&mut self) -> io::Result<DeviceState> {
        let b = std::fs::read(&self.0)?;
        serde_json::from_slice(&b).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

/// A fixed state (CLI `--force`, tests).
pub struct FixedMonitor(pub DeviceState);
impl DeviceMonitor for FixedMonitor {
    fn state(&mut self) -> io::Result<DeviceState> {
        Ok(self.0.clone())
    }
}

/// [`Control`] that consults the device state before every example.
/// An unreadable state counts as "not allowed".
pub struct PolicyControl<'a> {
    pub monitor: &'a mut dyn DeviceMonitor,
    pub policy: &'a PowerPolicy,
    pub last_reason: Option<String>,
}

impl<'a> PolicyControl<'a> {
    pub fn allowed(&mut self, running: bool) -> bool {
        let r = match self.monitor.state() {
            Ok(s) => self.policy.check(&s, running),
            Err(e) => Err(format!("device state unavailable: {e}")),
        };
        self.last_reason = r.as_ref().err().cloned();
        r.is_ok()
    }
}

impl<'a> Control for PolicyControl<'a> {
    fn should_continue(&mut self) -> bool {
        self.allowed(true)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobConfig {
    pub base_model: String,
    pub train: TrainConfig,
    pub gate: GateConfig,
    pub power: PowerPolicy,
    /// Skip the night if fewer confirmed training examples than this exist.
    pub min_train_examples: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Outcome {
    Accepted {
        id: String,
        report: EvalReport,
    },
    Rejected {
        id: String,
        report: EvalReport,
    },
    /// Conditions never allowed finishing before the deadline; the checkpoint is kept.
    Deferred {
        reason: String,
        step: u64,
        total_steps: u64,
    },
    Skipped {
        reason: String,
    },
}

/// Status file for `agentd` / the UI: `adapters/work/status.json`.
#[derive(Serialize)]
struct Status<'a> {
    ts: u64,
    phase: &'a str,
    step: u64,
    total_steps: u64,
    detail: Option<String>,
}

fn write_status(dir: &Path, phase: &str, step: u64, total: u64, detail: Option<String>) {
    let _ = std::fs::create_dir_all(dir);
    let _ = crate::adapter::write_json(dir.join("status.json"), &Status { ts: now_unix(), phase, step, total_steps: total, detail });
}

pub struct Heldout<'a> {
    pub sft: &'a [Example],
    pub pref: &'a [PrefPair],
}

/// One overnight job: wait for safe conditions, train (pausing/resuming as the
/// device state changes), then gate, swap or discard, with ledger entries.
///
/// `model` must be the base model without adapters. `wait` is called while
/// conditions do not allow training; it sleeps for a poll interval and returns
/// `false` once the run should give up (e.g. morning). `log` receives progress.
pub fn run_overnight(
    model: &mut Model,
    registry: &Registry,
    job: &JobConfig,
    data: TrainData,
    heldout: Heldout,
    monitor: &mut dyn DeviceMonitor,
    wait: &mut dyn FnMut() -> bool,
    log: &mut dyn FnMut(&str),
) -> io::Result<Outcome> {
    let work = registry.work_dir();
    std::fs::create_dir_all(&work)?;
    let ckpt = work.join("checkpoint.safetensors");

    if data.len() < job.min_train_examples {
        let reason = format!("{} confirmed examples (< {})", data.len(), job.min_train_examples);
        registry.ledger("skipped", None, serde_json::json!({"reason": reason}))?;
        return Ok(Outcome::Skipped { reason });
    }
    let n_examples = data.len();

    let mut trainer = Trainer::new(model, job.train.clone(), data);
    let resumed = if ckpt.exists() {
        match trainer.load_checkpoint(&ckpt) {
            Ok(()) => true,
            Err(e) => {
                // The confirmed data grew or the config changed: start over.
                log(&format!("checkpoint not reusable ({e}); starting fresh"));
                registry.ledger("restarted", None, serde_json::json!({"reason": e.to_string()}))?;
                std::fs::remove_file(&ckpt)?;
                // load_checkpoint only mutates state after all checks pass, so the
                // fresh trainer is intact.
                false
            }
        }
    } else {
        false
    };
    if resumed {
        registry.ledger("resumed", None, serde_json::json!({"step": trainer.state.step, "total": trainer.state.total_steps}))?;
    } else {
        registry.ledger(
            "started",
            None,
            serde_json::json!({
                "examples": n_examples,
                "total_steps": trainer.state.total_steps,
                "trainable_params": trainer.model.trainable_params(),
            }),
        )?;
    }

    let policy = &job.power;
    loop {
        // Wait for conditions that allow (re)starting.
        loop {
            let mut ctl = PolicyControl { monitor: &mut *monitor, policy, last_reason: None };
            if ctl.allowed(false) {
                break;
            }
            let reason = ctl.last_reason.unwrap_or_default();
            write_status(&work, "waiting", trainer.state.step, trainer.state.total_steps, Some(reason.clone()));
            if !wait() {
                trainer.save_checkpoint(&ckpt)?;
                registry.ledger(
                    "deferred",
                    None,
                    serde_json::json!({"reason": reason, "step": trainer.state.step, "total": trainer.state.total_steps}),
                )?;
                return Ok(Outcome::Deferred { reason, step: trainer.state.step, total_steps: trainer.state.total_steps });
            }
        }
        write_status(&work, "training", trainer.state.step, trainer.state.total_steps, None);
        let mut ctl = PolicyControl { monitor: &mut *monitor, policy, last_reason: None };
        let stop = trainer.run(&mut ctl, Some(&ckpt), |m| log(m))?;
        match stop {
            StopReason::Finished => break,
            StopReason::Paused => {
                let reason = ctl.last_reason.clone().unwrap_or_default();
                registry.ledger(
                    "paused",
                    None,
                    serde_json::json!({"reason": reason, "step": trainer.state.step, "total": trainer.state.total_steps}),
                )?;
                log(&format!("paused: {reason}"));
            }
        }
    }

    // Save the candidate.
    write_status(&work, "evaluating", trainer.state.step, trainer.state.total_steps, None);
    let id = registry.new_id();
    let final_loss = trainer.state.losses.last().copied();
    let steps = trainer.state.step;
    let model = trainer.model;
    let mut meta = AdapterMeta {
        id: id.clone(),
        base_model: job.base_model.clone(),
        base_fingerprint: base_fingerprint(model),
        rank: job.train.rank,
        alpha: job.train.alpha,
        targets: job.train.targets.clone(),
        created_unix: now_unix(),
        train_steps: steps,
        train_examples: n_examples,
        final_train_loss: final_loss,
        eval: None,
    };
    let cand_dir = registry.adapter_dir(&id);
    save_adapter(model, &cand_dir, &meta)?;

    // Eval gate: candidate vs current (base + active adapter, or base alone).
    let (cand, tokens) = measure(model, heldout.sft, heldout.pref);
    let current = match registry.active_dir() {
        Some(dir) => {
            load_adapter_into(model, &dir)?;
            measure(model, heldout.sft, heldout.pref).0
        }
        None => {
            model.set_lora_enabled(false);
            let m = measure(model, heldout.sft, heldout.pref).0;
            model.set_lora_enabled(true);
            m
        }
    };
    let report = gate(&job.gate, &current, &cand, heldout.sft.len() + heldout.pref.len(), tokens);
    let detail = serde_json::to_value(&report)?;
    meta.eval = Some(detail.clone());
    crate::adapter::write_json(cand_dir.join(crate::adapter::META_FILE), &meta)?;
    let _ = std::fs::remove_dir_all(&work);
    if report.accepted {
        registry.promote(&id, detail)?;
        log(&format!("accepted {id}: {}", report.reason));
        Ok(Outcome::Accepted { id, report })
    } else {
        registry.reject(&id, detail)?;
        log(&format!("rejected {id}: {}", report.reason));
        Ok(Outcome::Rejected { id, report })
    }
}
