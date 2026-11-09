//! AdamW, learning-rate schedule and global-norm gradient clipping over the
//! model's LoRA tensors.

use crate::model::Model;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdamWConfig {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        AdamWConfig { beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0 }
    }
}

/// Optimiser state: first/second moments for each LoRA tensor, in the order
/// given by [`param_names`] (each adapted linear contributes `A` then `B`).
#[derive(Clone, Debug)]
pub struct AdamW {
    pub cfg: AdamWConfig,
    pub t: u64,
    pub m: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

/// Stable names of the trainable tensors, e.g. `model.layers.0.self_attn.q_proj.A`.
pub fn param_names(model: &Model) -> Vec<String> {
    model.lora_linears().iter().flat_map(|l| [format!("{}.A", l.name), format!("{}.B", l.name)]).collect()
}

impl AdamW {
    pub fn new(cfg: AdamWConfig, model: &Model) -> Self {
        let mut m = Vec::new();
        for l in model.lora_linears() {
            let lo = l.lora.as_ref().unwrap();
            m.push(vec![0.0; lo.a.len()]);
            m.push(vec![0.0; lo.b.len()]);
        }
        let v = m.clone();
        AdamW { cfg, t: 0, m, v }
    }

    pub fn step(&mut self, model: &mut Model, lr: f32) {
        self.t += 1;
        let c = self.cfg.clone();
        let bc1 = 1.0 - c.beta1.powi(self.t as i32);
        let bc2 = 1.0 - c.beta2.powi(self.t as i32);
        let mut idx = 0;
        for l in model.lora_linears_mut() {
            let lo = l.lora.as_mut().unwrap();
            for (p, g) in [(&mut lo.a, &lo.ga), (&mut lo.b, &lo.gb)] {
                let (m, v) = (&mut self.m[idx], &mut self.v[idx]);
                assert_eq!(m.len(), p.len(), "optimizer state does not match parameters");
                for i in 0..p.len() {
                    m[i] = c.beta1 * m[i] + (1.0 - c.beta1) * g[i];
                    v[i] = c.beta2 * v[i] + (1.0 - c.beta2) * g[i] * g[i];
                    let mh = m[i] / bc1;
                    let vh = v[i] / bc2;
                    p[i] -= lr * (mh / (vh.sqrt() + c.eps) + c.weight_decay * p[i]);
                }
                idx += 1;
            }
        }
    }
}

/// Global L2 norm of all LoRA gradients; scales them down to `max_norm` if larger.
pub fn clip_grad_norm(model: &mut Model, max_norm: f32) -> f32 {
    let mut ss = 0.0f64;
    for l in model.lora_linears() {
        let lo = l.lora.as_ref().unwrap();
        for g in lo.ga.iter().chain(&lo.gb) {
            ss += (*g as f64) * (*g as f64);
        }
    }
    let norm = ss.sqrt() as f32;
    if max_norm > 0.0 && norm > max_norm {
        let s = max_norm / (norm + 1e-6);
        for l in model.lora_linears_mut() {
            let lo = l.lora.as_mut().unwrap();
            lo.ga.iter_mut().chain(lo.gb.iter_mut()).for_each(|g| *g *= s);
        }
    }
    norm
}

/// Linear warm-up then cosine decay to `min_ratio · peak`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Schedule {
    pub peak_lr: f32,
    pub warmup_steps: u64,
    pub total_steps: u64,
    pub min_ratio: f32,
}

impl Schedule {
    pub fn lr(&self, step: u64) -> f32 {
        if self.warmup_steps > 0 && step < self.warmup_steps {
            return self.peak_lr * (step + 1) as f32 / self.warmup_steps as f32;
        }
        let span = self.total_steps.saturating_sub(self.warmup_steps).max(1);
        let p = ((step - self.warmup_steps.min(step)) as f32 / span as f32).min(1.0);
        let cos = 0.5 * (1.0 + (std::f32::consts::PI * p).cos());
        self.peak_lr * (self.min_ratio + (1.0 - self.min_ratio) * cos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn schedule_shape() {
        let s = Schedule { peak_lr: 1.0, warmup_steps: 10, total_steps: 110, min_ratio: 0.1 };
        assert!((s.lr(0) - 0.1).abs() < 1e-6);
        assert!((s.lr(9) - 1.0).abs() < 1e-6);
        assert!((s.lr(10) - 1.0).abs() < 1e-6);
        assert!((s.lr(110) - 0.1).abs() < 1e-6);
        assert!(s.lr(60) < 0.6 && s.lr(60) > 0.5);
    }
}
