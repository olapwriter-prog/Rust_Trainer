//! Training loop: supervised fine-tuning on confirmed answers and DPO on
//! corrections, with resumable checkpoints and a pause hook.
//!
//! A step consumes `batch_size` examples (one sequence at a time; gradients are
//! accumulated), clips the global gradient norm and applies AdamW. Before each
//! example the [`Control`] is consulted; when it says stop, the partial step is
//! discarded and a checkpoint of the last completed step is written, so
//! unplugging pauses training and the next run resumes exactly where it was.

use crate::model::{answer_targets, Model, Target};
use crate::optim::{clip_grad_norm, param_names, AdamW, AdamWConfig, Schedule};
use crate::safetensors::{write_f32, SafeTensors};
use crate::tensor::{sigmoid, Rng};
use crate::tokenizer::Example;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io;
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Objective {
    /// Cross-entropy on answer tokens only.
    Sft,
    /// Direct Preference Optimisation against the frozen base (adapter disabled).
    Dpo { beta: f32 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    pub rank: usize,
    pub alpha: f32,
    pub targets: Vec<String>,
    pub lr: f32,
    pub min_lr_ratio: f32,
    pub warmup_steps: u64,
    pub epochs: usize,
    pub batch_size: usize,
    pub max_grad_norm: f32,
    pub weight_decay: f32,
    pub seed: u64,
    /// Write a checkpoint every N optimiser steps (and always on pause / finish).
    pub checkpoint_every: u64,
    pub objective: Objective,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            rank: 8,
            alpha: 16.0,
            targets: crate::model::DEFAULT_TARGETS.iter().map(|s| s.to_string()).collect(),
            lr: 2e-4,
            min_lr_ratio: 0.1,
            warmup_steps: 10,
            epochs: 2,
            batch_size: 4,
            max_grad_norm: 1.0,
            weight_decay: 0.0,
            seed: 1234,
            checkpoint_every: 10,
            objective: Objective::Sft,
        }
    }
}

/// A preference pair sharing one prompt.
#[derive(Clone, Debug)]
pub struct PrefPair {
    pub chosen: Example,
    pub rejected: Example,
}

pub enum TrainData {
    Sft(Vec<Example>),
    Pref(Vec<PrefPair>),
}

impl TrainData {
    pub fn len(&self) -> usize {
        match self {
            TrainData::Sft(v) => v.len(),
            TrainData::Pref(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Asked before every example whether training may continue.
pub trait Control {
    fn should_continue(&mut self) -> bool;
}

/// Never pauses.
pub struct RunToEnd;
impl Control for RunToEnd {
    fn should_continue(&mut self) -> bool {
        true
    }
}

/// Pauses after a fixed number of examples (tests, and time-boxed runs).
pub struct StopAfter(pub usize);
impl Control for StopAfter {
    fn should_continue(&mut self) -> bool {
        if self.0 == 0 {
            return false;
        }
        self.0 -= 1;
        true
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StopReason {
    Finished,
    Paused,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainState {
    pub step: u64,
    pub total_steps: u64,
    pub epoch: usize,
    /// Position in `order` of the next example.
    pub cursor: usize,
    pub order: Vec<usize>,
    pub rng: Rng,
    pub opt_t: u64,
    pub losses: Vec<f32>,
    /// DPO: cached reference log-probs `(chosen, rejected)` per pair.
    pub ref_logps: Vec<(f32, f32)>,
    pub data_fingerprint: u64,
    pub config_json: String,
}

pub struct Trainer<'m> {
    pub model: &'m mut Model,
    pub cfg: TrainConfig,
    pub data: TrainData,
    pub state: TrainState,
    pub opt: AdamW,
    pub sched: Schedule,
}

fn data_fingerprint(data: &TrainData) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |e: &Example| {
        for &t in &e.tokens {
            h ^= t as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h ^= e.answer_start as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    };
    match data {
        TrainData::Sft(v) => v.iter().for_each(&mut feed),
        TrainData::Pref(v) => v.iter().for_each(|p| {
            feed(&p.chosen);
            feed(&p.rejected)
        }),
    }
    h
}

fn sum_logp(model: &Model, ex: &Example) -> f32 {
    model.logprobs(&ex.tokens, &answer_targets(&ex.tokens, ex.answer_start)).iter().sum()
}

impl<'m> Trainer<'m> {
    /// Fresh run: attaches new LoRA adapters to `model`.
    pub fn new(model: &'m mut Model, cfg: TrainConfig, data: TrainData) -> Self {
        assert!(!data.is_empty(), "no training data");
        match (&cfg.objective, &data) {
            (Objective::Sft, TrainData::Sft(_)) | (Objective::Dpo { .. }, TrainData::Pref(_)) => {}
            _ => panic!("objective {:?} does not match the data kind", cfg.objective),
        }
        let targets: Vec<&str> = cfg.targets.iter().map(String::as_str).collect();
        let n = model.attach_lora(cfg.rank, cfg.alpha, &targets, cfg.seed);
        assert!(n > 0, "no linear matched the LoRA targets {targets:?}");
        let mut rng = Rng::new(cfg.seed ^ 0xD1CE);
        let mut order: Vec<usize> = (0..data.len()).collect();
        rng.shuffle(&mut order);
        let steps_per_epoch = data.len().div_ceil(cfg.batch_size);
        let total_steps = (steps_per_epoch * cfg.epochs) as u64;
        let opt = AdamW::new(AdamWConfig { weight_decay: cfg.weight_decay, ..Default::default() }, model);
        let sched =
            Schedule { peak_lr: cfg.lr, warmup_steps: cfg.warmup_steps.min(total_steps / 2), total_steps, min_ratio: cfg.min_lr_ratio };
        let state = TrainState {
            step: 0,
            total_steps,
            epoch: 0,
            cursor: 0,
            order,
            rng,
            opt_t: 0,
            losses: Vec::new(),
            ref_logps: Vec::new(),
            data_fingerprint: data_fingerprint(&data),
            config_json: serde_json::to_string(&cfg).unwrap(),
        };
        Trainer { model, cfg, data, state, opt, sched }
    }

    /// Resume from a checkpoint written by [`Trainer::save_checkpoint`].
    /// Fails if the data or config changed since the checkpoint.
    pub fn resume(model: &'m mut Model, cfg: TrainConfig, data: TrainData, ckpt: impl AsRef<Path>) -> io::Result<Self> {
        let mut tr = Trainer::new(model, cfg, data);
        tr.load_checkpoint(ckpt)?;
        Ok(tr)
    }

    pub fn finished(&self) -> bool {
        self.state.step >= self.state.total_steps
    }

    fn ensure_ref_logps(&mut self, ctl: &mut dyn Control) -> bool {
        let TrainData::Pref(pairs) = &self.data else { return true };
        if self.state.ref_logps.len() == pairs.len() {
            return true;
        }
        self.model.set_lora_enabled(false);
        while self.state.ref_logps.len() < pairs.len() {
            if !ctl.should_continue() {
                self.model.set_lora_enabled(true);
                return false;
            }
            let p = &pairs[self.state.ref_logps.len()];
            let r = (sum_logp(self.model, &p.chosen), sum_logp(self.model, &p.rejected));
            self.state.ref_logps.push(r);
        }
        self.model.set_lora_enabled(true);
        true
    }

    /// Accumulate the gradient of one example; returns its loss contribution.
    fn example_grad(&mut self, idx: usize, batch_tokens: usize, batch_n: usize) -> f32 {
        match &self.data {
            TrainData::Sft(v) => {
                let ex = &v[idx];
                let targets: Vec<Target> = answer_targets(&ex.tokens, ex.answer_start);
                let w = vec![1.0 / batch_tokens as f32; targets.len()];
                let lp = self.model.forward_backward(&ex.tokens, &targets, &w);
                -lp.iter().sum::<f32>() / batch_tokens as f32
            }
            TrainData::Pref(pairs) => {
                let Objective::Dpo { beta } = self.cfg.objective else { unreachable!() };
                let p = &pairs[idx];
                let (rc, rr) = self.state.ref_logps[idx];
                let pc = sum_logp(self.model, &p.chosen);
                let pr = sum_logp(self.model, &p.rejected);
                let z = beta * ((pc - rc) - (pr - rr));
                // L = -log σ(z);  dL/dpc = -β σ(-z),  dL/dpr = +β σ(-z)
                let c = beta * sigmoid(-z) / batch_n as f32;
                for (ex, coef) in [(&p.chosen, c), (&p.rejected, -c)] {
                    let targets = answer_targets(&ex.tokens, ex.answer_start);
                    let w = vec![coef; targets.len()];
                    self.model.forward_backward(&ex.tokens, &targets, &w);
                }
                // -log σ(z) = softplus(-z)
                crate::tensor::softplus(-z) / batch_n as f32
            }
        }
    }

    fn example_tokens(&self, idx: usize) -> usize {
        match &self.data {
            TrainData::Sft(v) => v[idx].answer_len(),
            TrainData::Pref(_) => 1,
        }
    }

    /// Train until finished or the control says stop. A checkpoint is written to
    /// `ckpt` (if given) every `checkpoint_every` steps, on pause and at the end.
    pub fn run(&mut self, ctl: &mut dyn Control, ckpt: Option<&Path>, mut log: impl FnMut(&str)) -> io::Result<StopReason> {
        if !self.ensure_ref_logps(ctl) {
            if let Some(p) = ckpt {
                self.save_checkpoint(p)?;
            }
            return Ok(StopReason::Paused);
        }
        let n = self.data.len();
        while !self.finished() {
            // Gather the batch indices (wrapping into the next epoch).
            let mut batch = Vec::with_capacity(self.cfg.batch_size);
            let (mut cursor, mut epoch, mut order, mut rng) =
                (self.state.cursor, self.state.epoch, self.state.order.clone(), self.state.rng.clone());
            while batch.len() < self.cfg.batch_size.min(n) {
                if cursor >= n {
                    cursor = 0;
                    epoch += 1;
                    rng.shuffle(&mut order);
                }
                batch.push(order[cursor]);
                cursor += 1;
            }
            let batch_tokens: usize = batch.iter().map(|&i| self.example_tokens(i)).sum::<usize>().max(1);

            self.model.zero_grad();
            let mut loss = 0.0;
            let mut paused = false;
            for &i in &batch {
                if !ctl.should_continue() {
                    paused = true;
                    break;
                }
                loss += self.example_grad(i, batch_tokens, batch.len());
            }
            if paused {
                self.model.zero_grad();
                if let Some(p) = ckpt {
                    self.save_checkpoint(p)?;
                }
                log(&format!("paused at step {}/{}", self.state.step, self.state.total_steps));
                return Ok(StopReason::Paused);
            }
            let gnorm = clip_grad_norm(self.model, self.cfg.max_grad_norm);
            let lr = self.sched.lr(self.state.step);
            self.opt.step(self.model, lr);
            self.state.step += 1;
            self.state.opt_t = self.opt.t;
            self.state.cursor = cursor;
            self.state.epoch = epoch;
            self.state.order = order;
            self.state.rng = rng;
            self.state.losses.push(loss);
            log(&format!(
                "step {:>5}/{} epoch {} loss {:.4} |g| {:.3} lr {:.2e}",
                self.state.step, self.state.total_steps, self.state.epoch, loss, gnorm, lr
            ));
            if let Some(p) = ckpt {
                if self.cfg.checkpoint_every > 0 && self.state.step % self.cfg.checkpoint_every == 0 {
                    self.save_checkpoint(p)?;
                }
            }
        }
        if let Some(p) = ckpt {
            self.save_checkpoint(p)?;
        }
        Ok(StopReason::Finished)
    }

    /// LoRA weights + AdamW moments + loop state in one safetensors file (atomic write).
    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let names = param_names(self.model);
        let mut tensors: Vec<(String, Vec<usize>, &[f32])> = Vec::new();
        let mut i = 0;
        for l in self.model.lora_linears() {
            let lo = l.lora.as_ref().unwrap();
            for p in [&lo.a, &lo.b] {
                let n = &names[i];
                tensors.push((n.clone(), vec![p.len()], p.as_slice()));
                tensors.push((format!("{n}.adam_m"), vec![p.len()], self.opt.m[i].as_slice()));
                tensors.push((format!("{n}.adam_v"), vec![p.len()], self.opt.v[i].as_slice()));
                i += 1;
            }
        }
        let mut md = BTreeMap::new();
        md.insert("trainer_state".to_string(), serde_json::to_string(&self.state)?);
        write_f32(path, &tensors, &md)
    }

    pub fn load_checkpoint(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        let st = SafeTensors::open(path)?;
        let state: TrainState = serde_json::from_str(
            st.metadata.get("trainer_state").ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no trainer_state"))?,
        )
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        if state.data_fingerprint != self.state.data_fingerprint {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "training data changed since checkpoint"));
        }
        if state.config_json != self.state.config_json {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "training config changed since checkpoint"));
        }
        // Read everything first so a bad checkpoint leaves the trainer untouched.
        let names = param_names(self.model);
        let sizes: Vec<usize> = self
            .model
            .lora_linears()
            .iter()
            .flat_map(|l| {
                let lo = l.lora.as_ref().unwrap();
                [lo.a.len(), lo.b.len()]
            })
            .collect();
        let mut ws = Vec::new();
        let mut ms = Vec::new();
        let mut vs = Vec::new();
        for (n, &sz) in names.iter().zip(&sizes) {
            let (w, _) = st.read_f32(n)?;
            if w.len() != sz {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{n}: shape changed")));
            }
            ws.push(w);
            ms.push(st.read_f32(&format!("{n}.adam_m"))?.0);
            vs.push(st.read_f32(&format!("{n}.adam_v"))?.0);
        }
        let mut ws = ws.into_iter();
        for l in self.model.lora_linears_mut() {
            let lo = l.lora.as_mut().unwrap();
            lo.a = ws.next().unwrap();
            lo.b = ws.next().unwrap();
        }
        self.opt.m = ms;
        self.opt.v = vs;
        self.opt.t = state.opt_t;
        self.state = state;
        Ok(())
    }
}
