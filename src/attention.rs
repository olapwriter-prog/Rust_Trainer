//! Causal grouped-query attention with RoPE, covering both
//! * Qwen2.5: q/k/v bias, full rotary, no QK-norm, no output gate;
//! * Qwen3.5 full-attention layers: QK-RMSNorm, partial rotary and a sigmoid
//!   output gate packed into `q_proj` (`[q | gate]` per head).

use crate::linear::Linear;
use crate::ops::{rmsnorm, rmsnorm_backward, Rope};
use crate::tensor::{dot, sigmoid};
use rayon::prelude::*;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AttnConfig {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Number of leading dims per head that get rotary (== head_dim for full rotary).
    pub rot_dim: usize,
    pub rope_theta: f32,
    pub qk_norm: bool,
    pub output_gate: bool,
}

pub struct Attention {
    pub cfg: AttnConfig,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    pub eps: f32,
    pub norm_offset: f32,
}

pub struct AttnCache {
    x: Vec<f32>,
    q_raw: Vec<f32>,
    k_raw: Vec<f32>,
    q_inv: Vec<f32>,
    k_inv: Vec<f32>,
    q: Vec<f32>, // after norm + rope  [t × nh × hd]
    k: Vec<f32>, // [t × nkv × hd]
    v: Vec<f32>,
    gate: Vec<f32>, // pre-sigmoid gate [t × nh × hd] (empty if no gate)
    probs: Vec<f32>, // [nh × t × t]
    o: Vec<f32>,     // attention output before gate [t × nh × hd]
    og: Vec<f32>,    // o_proj input [t × nh × hd]
}

impl Attention {
    fn split_q(&self, qf: &[f32], t: usize) -> (Vec<f32>, Vec<f32>) {
        let (nh, hd) = (self.cfg.n_heads, self.cfg.head_dim);
        if !self.cfg.output_gate {
            return (qf.to_vec(), Vec::new());
        }
        let mut q = vec![0.0; t * nh * hd];
        let mut g = vec![0.0; t * nh * hd];
        for p in 0..t {
            for h in 0..nh {
                let src = (p * nh + h) * 2 * hd;
                let dst = (p * nh + h) * hd;
                q[dst..dst + hd].copy_from_slice(&qf[src..src + hd]);
                g[dst..dst + hd].copy_from_slice(&qf[src + hd..src + 2 * hd]);
            }
        }
        (q, g)
    }

    pub fn forward(&self, x: &[f32], t: usize, rope: &Rope) -> (Vec<f32>, AttnCache) {
        let c = &self.cfg;
        let (nh, nkv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
        let qf = self.q_proj.forward(x, t);
        let (q_raw, gate) = self.split_q(&qf, t);
        let k_raw = self.k_proj.forward(x, t);
        let v = self.v_proj.forward(x, t);

        let (mut q, q_inv) = match &self.q_norm {
            Some(w) => rmsnorm(&q_raw, w, hd, self.eps, self.norm_offset),
            None => (q_raw.clone(), Vec::new()),
        };
        let (mut k, k_inv) = match &self.k_norm {
            Some(w) => rmsnorm(&k_raw, w, hd, self.eps, self.norm_offset),
            None => (k_raw.clone(), Vec::new()),
        };
        rope.apply(&mut q, t, nh, hd, false);
        rope.apply(&mut k, t, nkv, hd, false);

        let scale = 1.0 / (hd as f32).sqrt();
        let group = nh / nkv;
        // Per-head attention, parallel over heads.
        let per_head: Vec<(Vec<f32>, Vec<f32>)> = (0..nh)
            .into_par_iter()
            .map(|h| {
                let g = h / group;
                let mut probs = vec![0.0; t * t];
                let mut out = vec![0.0; t * hd];
                for i in 0..t {
                    let qi = &q[(i * nh + h) * hd..(i * nh + h + 1) * hd];
                    let row = &mut probs[i * t..(i + 1) * t];
                    let mut m = f32::NEG_INFINITY;
                    for j in 0..=i {
                        let kj = &k[(j * nkv + g) * hd..(j * nkv + g + 1) * hd];
                        row[j] = dot(qi, kj) * scale;
                        m = m.max(row[j]);
                    }
                    let mut s = 0.0;
                    for j in 0..=i {
                        row[j] = (row[j] - m).exp();
                        s += row[j];
                    }
                    let oi = &mut out[i * hd..(i + 1) * hd];
                    for j in 0..=i {
                        row[j] /= s;
                        let vj = &v[(j * nkv + g) * hd..(j * nkv + g + 1) * hd];
                        for d in 0..hd {
                            oi[d] += row[j] * vj[d];
                        }
                    }
                }
                (probs, out)
            })
            .collect();

        let mut probs = vec![0.0; nh * t * t];
        let mut o = vec![0.0; t * nh * hd];
        for (h, (p, out)) in per_head.into_iter().enumerate() {
            probs[h * t * t..(h + 1) * t * t].copy_from_slice(&p);
            for i in 0..t {
                o[(i * nh + h) * hd..(i * nh + h + 1) * hd].copy_from_slice(&out[i * hd..(i + 1) * hd]);
            }
        }
        let og = if c.output_gate {
            o.iter().zip(&gate).map(|(a, g)| a * sigmoid(*g)).collect()
        } else {
            o.clone()
        };
        let y = self.o_proj.forward(&og, t);
        let cache = AttnCache { x: x.to_vec(), q_raw, k_raw, q_inv, k_inv, q, k, v, gate, probs, o, og };
        (y, cache)
    }

    pub fn backward(&mut self, cache: AttnCache, dy: &[f32], t: usize, rope: &Rope) -> Vec<f32> {
        let c = self.cfg.clone();
        let (nh, nkv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
        let d_og = self.o_proj.backward(&cache.og, dy, t);

        // Output gate.
        let (d_o, d_gate) = if c.output_gate {
            let mut d_o = vec![0.0; d_og.len()];
            let mut d_g = vec![0.0; d_og.len()];
            for i in 0..d_og.len() {
                let s = sigmoid(cache.gate[i]);
                d_o[i] = d_og[i] * s;
                d_g[i] = d_og[i] * cache.o[i] * s * (1.0 - s);
            }
            (d_o, d_g)
        } else {
            (d_og, Vec::new())
        };

        let scale = 1.0 / (hd as f32).sqrt();
        let group = nh / nkv;
        let (q, k, v, probs) = (&cache.q, &cache.k, &cache.v, &cache.probs);
        let per_head: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..nh)
            .into_par_iter()
            .map(|h| {
                let g = h / group;
                let p = &probs[h * t * t..(h + 1) * t * t];
                let mut dq = vec![0.0; t * hd];
                let mut dk = vec![0.0; t * hd];
                let mut dv = vec![0.0; t * hd];
                let mut dp = vec![0.0; t];
                for i in 0..t {
                    let doi = &d_o[(i * nh + h) * hd..(i * nh + h + 1) * hd];
                    let mut sum = 0.0;
                    for j in 0..=i {
                        let vj = &v[(j * nkv + g) * hd..(j * nkv + g + 1) * hd];
                        dp[j] = dot(doi, vj);
                        sum += p[i * t + j] * dp[j];
                        let pij = p[i * t + j];
                        let dvj = &mut dv[j * hd..(j + 1) * hd];
                        for d in 0..hd {
                            dvj[d] += pij * doi[d];
                        }
                    }
                    let qi = &q[(i * nh + h) * hd..(i * nh + h + 1) * hd];
                    for j in 0..=i {
                        let ds = p[i * t + j] * (dp[j] - sum) * scale;
                        if ds == 0.0 {
                            continue;
                        }
                        let kj = &k[(j * nkv + g) * hd..(j * nkv + g + 1) * hd];
                        let dqi = &mut dq[i * hd..(i + 1) * hd];
                        for d in 0..hd {
                            dqi[d] += ds * kj[d];
                        }
                        let dkj = &mut dk[j * hd..(j + 1) * hd];
                        for d in 0..hd {
                            dkj[d] += ds * qi[d];
                        }
                    }
                }
                (dq, dk, dv)
            })
            .collect();

        let mut dq = vec![0.0; t * nh * hd];
        let mut dk = vec![0.0; t * nkv * hd];
        let mut dv = vec![0.0; t * nkv * hd];
        for (h, (dqh, dkh, dvh)) in per_head.into_iter().enumerate() {
            let g = h / group;
            for i in 0..t {
                dq[(i * nh + h) * hd..(i * nh + h + 1) * hd].copy_from_slice(&dqh[i * hd..(i + 1) * hd]);
                for d in 0..hd {
                    dk[(i * nkv + g) * hd + d] += dkh[i * hd + d];
                    dv[(i * nkv + g) * hd + d] += dvh[i * hd + d];
                }
            }
        }

        // RoPE is orthogonal: backward = inverse rotation.
        rope.apply(&mut dq, t, nh, hd, true);
        rope.apply(&mut dk, t, nkv, hd, true);
        let dq_raw = match &self.q_norm {
            Some(w) => rmsnorm_backward(&cache.q_raw, w, &cache.q_inv, &dq, hd, self.norm_offset),
            None => dq,
        };
        let dk_raw = match &self.k_norm {
            Some(w) => rmsnorm_backward(&cache.k_raw, w, &cache.k_inv, &dk, hd, self.norm_offset),
            None => dk,
        };

        // Re-pack [q | gate] for q_proj.
        let dqf = if c.output_gate {
            let mut f = vec![0.0; t * nh * 2 * hd];
            for p in 0..t {
                for h in 0..nh {
                    let dst = (p * nh + h) * 2 * hd;
                    let src = (p * nh + h) * hd;
                    f[dst..dst + hd].copy_from_slice(&dq_raw[src..src + hd]);
                    f[dst + hd..dst + 2 * hd].copy_from_slice(&d_gate[src..src + hd]);
                }
            }
            f
        } else {
            dq_raw
        };

        let x = &cache.x;
        let mut dx = self.q_proj.backward(x, &dqf, t);
        let dxk = self.k_proj.backward(x, &dk_raw, t);
        let dxv = self.v_proj.backward(x, &dv, t);
        for i in 0..dx.len() {
            dx[i] += dxk[i] + dxv[i];
        }
        dx
    }

    pub fn linears_mut(&mut self) -> Vec<&mut Linear> {
        vec![&mut self.q_proj, &mut self.k_proj, &mut self.v_proj, &mut self.o_proj]
    }
    pub fn linears(&self) -> Vec<&Linear> {
        vec![&self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj]
    }
}
