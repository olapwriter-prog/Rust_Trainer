//! Frozen linear layer with an optional trainable LoRA adapter:
//! `y = W·x + b + (α/r)·B·A·x`.

use crate::frozen::Frozen;
use crate::tensor::{matmul_nn, matmul_nt, matmul_tn_acc, Rng};

/// Trainable low-rank adapter. `a` is `[r × in]`, `b` is `[out × r]`.
#[derive(Clone, Debug)]
pub struct Lora {
    pub rank: usize,
    pub alpha: f32,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub ga: Vec<f32>,
    pub gb: Vec<f32>,
}

impl Lora {
    /// PEFT-style init: A ~ U(-1/√in, 1/√in) (kaiming-uniform with a=√5), B = 0,
    /// so a fresh adapter is an exact no-op.
    pub fn new(rank: usize, alpha: f32, inp: usize, out: usize, rng: &mut Rng) -> Self {
        let bound = 1.0 / (inp as f32).sqrt();
        let a = (0..rank * inp).map(|_| rng.uniform_sym(bound)).collect();
        Lora {
            rank,
            alpha,
            a,
            b: vec![0.0; out * rank],
            ga: vec![0.0; rank * inp],
            gb: vec![0.0; out * rank],
        }
    }
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }
    pub fn zero_grad(&mut self) {
        self.ga.iter_mut().for_each(|g| *g = 0.0);
        self.gb.iter_mut().for_each(|g| *g = 0.0);
    }
    pub fn num_params(&self) -> usize {
        self.a.len() + self.b.len()
    }
}

pub struct Linear {
    /// HF tensor name without the `.weight` suffix, e.g. `model.layers.3.self_attn.q_proj`.
    pub name: String,
    pub w: Frozen,
    pub bias: Option<Vec<f32>>,
    pub lora: Option<Lora>,
    /// When false the adapter is bypassed (reference model for DPO / eval baseline).
    pub lora_enabled: bool,
}

impl Linear {
    pub fn new(name: impl Into<String>, w: Frozen, bias: Option<Vec<f32>>) -> Self {
        if let Some(b) = &bias {
            assert_eq!(b.len(), w.out_dim());
        }
        Linear { name: name.into(), w, bias, lora: None, lora_enabled: true }
    }
    pub fn in_dim(&self) -> usize {
        self.w.in_dim()
    }
    pub fn out_dim(&self) -> usize {
        self.w.out_dim()
    }
    pub fn attach_lora(&mut self, rank: usize, alpha: f32, rng: &mut Rng) {
        self.lora = Some(Lora::new(rank, alpha, self.in_dim(), self.out_dim(), rng));
    }
    fn active_lora(&self) -> Option<&Lora> {
        if self.lora_enabled {
            self.lora.as_ref()
        } else {
            None
        }
    }

    /// `x` is `[t × in]`; returns `[t × out]`.
    pub fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let (inp, out) = (self.in_dim(), self.out_dim());
        assert_eq!(x.len(), t * inp, "{}: input shape", self.name);
        let mut y = self.w.forward(x, t);
        if let Some(b) = &self.bias {
            for row in y.chunks_mut(out) {
                for (v, bb) in row.iter_mut().zip(b) {
                    *v += bb;
                }
            }
        }
        if let Some(l) = self.active_lora() {
            let h = matmul_nt(x, &l.a, t, inp, l.rank); // [t × r]
            let d = matmul_nt(&h, &l.b, t, l.rank, out); // [t × out]
            let s = l.scale();
            for (v, dv) in y.iter_mut().zip(&d) {
                *v += s * dv;
            }
        }
        y
    }

    /// Back-propagate `dy` (`[t × out]`) given the forward input `x`.
    /// Accumulates LoRA gradients and returns `dx` (`[t × in]`).
    pub fn backward(&mut self, x: &[f32], dy: &[f32], t: usize) -> Vec<f32> {
        let (inp, out) = (self.in_dim(), self.out_dim());
        assert_eq!(dy.len(), t * out);
        let mut dx = vec![0.0; t * inp];
        self.w.backward_input(dy, t, &mut dx);
        let enabled = self.lora_enabled;
        if let (true, Some(l)) = (enabled, self.lora.as_mut()) {
            let r = l.rank;
            let s = l.scale();
            let h = matmul_nt(x, &l.a, t, inp, r); // [t × r]
            // dB[out×r] += s · dyᵀ·h
            let sdy: Vec<f32> = dy.iter().map(|v| v * s).collect();
            matmul_tn_acc(&mut l.gb, &sdy, &h, t, out, r);
            // g = s · dy·B   [t × r]
            let g = matmul_nn(&sdy, &l.b, t, out, r);
            // dA[r×in] += gᵀ·x
            matmul_tn_acc(&mut l.ga, &g, x, t, r, inp);
            // dx += g·A
            let gx = matmul_nn(&g, &l.a, t, r, inp);
            for (d, v) in dx.iter_mut().zip(&gx) {
                *d += v;
            }
        }
        dx
    }
}
