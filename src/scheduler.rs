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

use crate::adapter::{base_fingerprint, save_adapter, AdapterMeta};
use crate::eval::{gate, measure, measure_sums, EvalReport, GateConfig, Sums};
use crate::model::Model;
use crate::moe::{measure_current, prompt_of, select_expert, Current, MoeConfig, GENERAL};
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

/// Train until finished, pausing and resuming with the device state.
/// Returns `Some(reason)` if the deadline passed first (checkpoint saved, ledger `deferred`).
fn drive(
    trainer: &mut Trainer,
    ckpt: &Path,
    job: &JobConfig,
    registry: &Registry,
    work: &Path,
    monitor: &mut dyn DeviceMonitor,
    wait: &mut dyn FnMut() -> bool,
    log: &mut dyn FnMut(&str),
    expert: Option<&str>,
) -> io::Result<Option<String>> {
    let policy = &job.power;
    loop {
        // Wait for conditions that allow (re)starting.
        loop {
            let mut ctl = PolicyControl { monitor: &mut *monitor, policy, last_reason: None };
            if ctl.allowed(false) {
                break;
            }
            let reason = ctl.last_reason.unwrap_or_default();
            write_status(work, "waiting", trainer.state.step, trainer.state.total_steps, Some(reason.clone()));
            if !wait() {
                trainer.save_checkpoint(ckpt)?;
                registry.ledger(
                    "deferred",
                    expert,
                    serde_json::json!({"reason": reason, "step": trainer.state.step, "total": trainer.state.total_steps}),
                )?;
                return Ok(Some(reason));
            }
        }
        let phase = match expert {
            Some(e) => format!("training {e}"),
            None => "training".to_string(),
        };
        write_status(work, &phase, trainer.state.step, trainer.state.total_steps, None);
        let mut ctl = PolicyControl { monitor: &mut *monitor, policy, last_reason: None };
        match trainer.run(&mut ctl, Some(ckpt), |m| log(m))? {
            StopReason::Finished => return Ok(None),
            StopReason::Paused => {
                let reason = ctl.last_reason.clone().unwrap_or_default();
                registry.ledger(
                    "paused",
                    expert,
                    serde_json::json!({"reason": reason, "step": trainer.state.step, "total": trainer.state.total_steps}),
                )?;
                log(&format!("paused: {reason}"));
            }
        }
    }
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

    if let Some(reason) = drive(&mut trainer, &ckpt, job, registry, &work, monitor, wait, log, None)? {
        return Ok(Outcome::Deferred { reason, step: trainer.state.step, total_steps: trainer.state.total_steps });
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
    let current_cfg = crate::moe::Current::from_dir(registry.active_dir())?;
    let current = crate::moe::measure_current(model, &current_cfg, heldout.sft, heldout.pref)?.metrics();
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

/// Data fingerprint for resuming a multi-expert run.
fn moe_fingerprint(data: &TrainData, labels: &[Option<String>], job: &JobConfig, moe: &MoeConfig) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |b: &[u8]| {
        for x in b {
            h ^= *x as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    };
    let items: Vec<&Example> = match data {
        TrainData::Sft(v) => v.iter().collect(),
        TrainData::Pref(v) => v.iter().flat_map(|p| [&p.chosen, &p.rejected]).collect(),
    };
    for e in items {
        for t in &e.tokens {
            feed(&t.to_le_bytes());
        }
        feed(&e.answer_start.to_le_bytes());
    }
    feed(serde_json::to_string(labels).unwrap_or_default().as_bytes());
    feed(serde_json::to_string(&job.train).unwrap_or_default().as_bytes());
    feed(serde_json::to_string(moe).unwrap_or_default().as_bytes());
    format!("{h:016x}")
}

fn subset(data: &TrainData, idx: &[usize]) -> TrainData {
    match data {
        TrainData::Sft(v) => TrainData::Sft(idx.iter().map(|&i| v[i].clone()).collect()),
        TrainData::Pref(v) => TrainData::Pref(idx.iter().map(|&i| v[i].clone()).collect()),
    }
}

/// Per-expert decision recorded in the eval report.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpertReport {
    pub group: String,
    pub train_examples: usize,
    pub heldout_examples: usize,
    pub expert_nll: Option<f32>,
    pub fallback_nll: Option<f32>,
    /// Whether the router sends this group to its own expert (else the fallback).
    pub kept: bool,
}

/// Overnight job producing a **mixture of LoRA experts** (see [`crate::moe`]).
///
/// `labels[i]` is the skill label of training item `i` (or `None`). Falls back to
/// [`run_overnight`] when the data forms fewer than two expert groups. Each
/// expert trains with the same pause/resume policy and its own checkpoint;
/// finished experts survive a deferred night. The routed bundle is gated against
/// the current model (base, single adapter or bundle).
pub fn run_overnight_moe(
    model: &mut Model,
    registry: &Registry,
    job: &JobConfig,
    moe: &MoeConfig,
    data: TrainData,
    labels: &[Option<String>],
    heldout: Heldout,
    monitor: &mut dyn DeviceMonitor,
    wait: &mut dyn FnMut() -> bool,
    log: &mut dyn FnMut(&str),
) -> io::Result<Outcome> {
    if data.len() < job.min_train_examples {
        let reason = format!("{} confirmed examples (< {})", data.len(), job.min_train_examples);
        registry.ledger("skipped", None, serde_json::json!({"reason": reason}))?;
        return Ok(Outcome::Skipped { reason });
    }
    assert_eq!(labels.len(), data.len(), "one label slot per training item");
    let prompts: Vec<&[u32]> = match &data {
        TrainData::Sft(v) => v.iter().map(prompt_of).collect(),
        TrainData::Pref(v) => v.iter().map(|p| prompt_of(&p.chosen)).collect(),
    };
    let plan = crate::moe::plan(&prompts, labels, moe);
    let n_experts = plan.experts().count();
    if n_experts < 2 {
        log(&format!("{n_experts} skill group(s) with ≥{} examples: training a single adapter", moe.min_expert_examples));
        return run_overnight(model, registry, job, data, heldout, monitor, wait, log);
    }

    let root = registry.work_dir();
    let work = root.join("moe");
    let bundle = work.join("bundle");
    let fp = moe_fingerprint(&data, labels, job, moe);
    let fp_file = work.join("fingerprint");
    let resumed = match std::fs::read_to_string(&fp_file) {
        Ok(old) if old == fp => true,
        Ok(_) => {
            log("data or config changed since the last night: restarting the mixture");
            registry.ledger("restarted", None, serde_json::json!({"reason": "data or config changed", "kind": "moe"}))?;
            let _ = std::fs::remove_dir_all(&work);
            false
        }
        Err(_) => false,
    };
    std::fs::create_dir_all(bundle.join(crate::moe::EXPERTS_DIR))?;
    std::fs::write(&fp_file, &fp)?;
    let groups_json: Vec<serde_json::Value> =
        plan.groups.iter().map(|g| serde_json::json!({"group": g.name, "examples": g.members.len(), "expert": g.has_expert})).collect();
    registry.ledger(
        if resumed { "resumed" } else { "started" },
        None,
        serde_json::json!({"kind": "moe", "grouping": plan.grouping, "groups": groups_json, "general": moe.general_expert, "examples": data.len()}),
    )?;

    // The general expert first, on all data; every skill expert then starts from it
    // and specialises on its own group (branch-train), so it begins no worse than
    // the general expert even though it sees only part of the data.
    let mut jobs: Vec<(String, Vec<usize>)> = Vec::new();
    if moe.general_expert {
        jobs.push((GENERAL.to_string(), (0..data.len()).collect()));
    }
    jobs.extend(plan.experts().map(|g| (g.name.clone(), g.members.clone())));
    for (name, members) in &jobs {
        let out = crate::moe::expert_dir(&bundle, name);
        if out.join(crate::adapter::META_FILE).exists() {
            log(&format!("expert {name}: already trained"));
            continue;
        }
        let ckpt = work.join(format!("ckpt-{name}.safetensors"));
        let mut trainer = Trainer::new(&mut *model, job.train.clone(), subset(&data, members));
        let general_dir = crate::moe::expert_dir(&bundle, GENERAL);
        if name != GENERAL && general_dir.join(crate::adapter::META_FILE).exists() {
            crate::adapter::load_adapter_into(trainer.model, &general_dir)?;
        }
        if ckpt.exists() && trainer.load_checkpoint(&ckpt).is_err() {
            std::fs::remove_file(&ckpt)?;
        }
        log(&format!("expert {name}: {} examples, {} steps", members.len(), trainer.state.total_steps));
        if let Some(reason) = drive(&mut trainer, &ckpt, job, registry, &root, monitor, wait, log, Some(name))? {
            return Ok(Outcome::Deferred { reason, step: trainer.state.step, total_steps: trainer.state.total_steps });
        }
        let meta = AdapterMeta {
            id: name.clone(),
            base_model: job.base_model.clone(),
            base_fingerprint: base_fingerprint(trainer.model),
            rank: job.train.rank,
            alpha: job.train.alpha,
            targets: job.train.targets.clone(),
            created_unix: now_unix(),
            train_steps: trainer.state.step,
            train_examples: members.len(),
            final_train_loss: trainer.state.losses.last().copied(),
            eval: None,
        };
        save_adapter(trainer.model, &out, &meta)?;
        let _ = std::fs::remove_file(&ckpt);
        registry.ledger(
            "expert_trained",
            Some(name),
            serde_json::json!({"examples": members.len(), "final_loss": meta.final_train_loss}),
        )?;
    }

    // Route held-out data and decide, per group, expert vs fallback.
    write_status(&root, "evaluating", 0, 0, None);
    let mut router = plan.router.clone();
    let fallback = router.fallback.clone();
    let mut by_group: Vec<(Vec<Example>, Vec<PrefPair>)> = vec![(Vec::new(), Vec::new()); router.centroids.len()];
    let mut unsure: (Vec<Example>, Vec<PrefPair>) = (Vec::new(), Vec::new());
    for ex in heldout.sft {
        match router.route(prompt_of(ex)).group {
            Some(g) => by_group[g].0.push(ex.clone()),
            None => unsure.0.push(ex.clone()),
        }
    }
    for p in heldout.pref {
        match router.route(prompt_of(&p.chosen)).group {
            Some(g) => by_group[g].1.push(p.clone()),
            None => unsure.1.push(p.clone()),
        }
    }
    let mut cand = Sums::default();
    let mut experts_report = Vec::new();
    for (gi, g) in plan.groups.iter().enumerate() {
        let (s, p) = &by_group[gi];
        let n_held = s.len() + p.len();
        select_expert(model, &bundle, fallback.as_deref())?;
        let fb = measure_sums(model, s, p);
        let (kept, ex_nll, chosen) = if g.has_expert {
            select_expert(model, &bundle, Some(&g.name))?;
            let es = measure_sums(model, s, p);
            // Keep the expert unless it is measurably worse on its own requests.
            let keep = n_held == 0 || es.metrics().nll <= fb.metrics().nll;
            (keep, (n_held > 0).then(|| es.metrics().nll), if keep { es } else { fb })
        } else {
            (false, None, fb)
        };
        if !kept {
            router.experts[gi] = None;
        }
        cand.add(&chosen);
        experts_report.push(ExpertReport {
            group: g.name.clone(),
            train_examples: g.members.len(),
            heldout_examples: n_held,
            expert_nll: ex_nll,
            fallback_nll: (n_held > 0).then(|| fb.metrics().nll),
            kept,
        });
    }
    select_expert(model, &bundle, fallback.as_deref())?;
    cand.add(&measure_sums(model, &unsure.0, &unsure.1));
    model.set_lora_enabled(true);

    let current_cfg = Current::from_dir(registry.active_dir())?;
    let current = measure_current(model, &current_cfg, heldout.sft, heldout.pref)?;
    let mut report = gate(&job.gate, &current.metrics(), &cand.metrics(), heldout.sft.len() + heldout.pref.len(), cand.tokens);
    if !router.experts.iter().any(|e| e.is_some()) && report.accepted {
        report.reason = format!("{} (no skill expert beat the fallback; bundle reduces to the general expert)", report.reason);
    }

    // Drop experts the router never selects, write the router, move into the registry.
    let used = router.used_experts();
    for (name, _) in &jobs {
        if !used.contains(name) {
            let _ = std::fs::remove_dir_all(crate::moe::expert_dir(&bundle, name));
        }
    }
    crate::adapter::write_json(bundle.join(crate::moe::ROUTER_FILE), &router)?;
    let id = registry.new_id();
    let detail = serde_json::json!({"kind": "moe", "eval": report, "experts": experts_report, "grouping": plan.grouping});
    let meta = AdapterMeta {
        id: id.clone(),
        base_model: job.base_model.clone(),
        base_fingerprint: base_fingerprint(model),
        rank: job.train.rank,
        alpha: job.train.alpha,
        targets: job.train.targets.clone(),
        created_unix: now_unix(),
        train_steps: 0,
        train_examples: data.len(),
        final_train_loss: None,
        eval: Some(detail.clone()),
    };
    crate::adapter::write_json(bundle.join(crate::adapter::META_FILE), &meta)?;
    std::fs::rename(&bundle, registry.adapter_dir(&id))?;
    let _ = std::fs::remove_dir_all(&root);
    if report.accepted {
        registry.promote(&id, detail)?;
        log(&format!("accepted mixture {id} ({} experts): {}", used.len(), report.reason));
        Ok(Outcome::Accepted { id, report })
    } else {
        registry.reject(&id, detail)?;
        log(&format!("rejected mixture {id}: {}", report.reason));
        Ok(Outcome::Rejected { id, report })
    }
}
