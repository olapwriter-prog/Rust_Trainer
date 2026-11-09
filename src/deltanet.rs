//! Qwen3.5 Gated DeltaNet ("linear attention") layer with backpropagation
//! through time.
//!
//! Forward (per value head `h`, state `S ∈ R^{dv×dk}`, matching HF
//! `torch_recurrent_gated_delta_rule` up to the transpose of S):
//! ```text
//!   mixed = in_proj_qkv(x);  act = silu(causal_conv1d(mixed))
//!   q, k  = l2norm(act_q), l2norm(act_k);  q *= 1/√dk;   v = act_v
//!   β = σ(in_proj_b(x));  g = -exp(A_log)·softplus(in_proj_a(x) + dt_bias);  α = e^g
//!   S'  = α·S_{t-1}
//!   u   = v - S'·k
//!   S_t = S' + β·u·kᵀ
//!   o_t = S_t·q
//!   y   = out_proj( rmsnorm(o)·w ⊙ silu(in_proj_z(x)) )
//! ```
//! BPTT keeps only every `chunk`-th state and recomputes the states inside a
//! chunk during the backward pass, so memory is `O((T/chunk + chunk)·dv·dk)` per
//! head rather than `O(T·dv·dk)`.

use crate::linear::Linear;
use crate::ops::{
    causal_conv1d, causal_conv1d_backward, l2norm, l2norm_backward, rmsnorm, rmsnorm_backward, silu_backward,
    silu_vec,
};
use crate::tensor::{sigmoid, silu, silu_grad, softplus};
use rayon::prelude::*;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DeltaNetConfig {
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub k_head_dim: usize,
    pub v_head_dim: usize,
    pub conv_kernel: usize,
    /// BPTT checkpoint interval (time steps between stored states).
    #[serde(default = "default_chunk")]
    pub bptt_chunk: usize,
}

fn default_chunk() -> usize {
    16
}

impl DeltaNetConfig {
    pub fn key_dim(&self) -> usize {
        self.n_k_heads * self.k_head_dim
    }
    pub fn value_dim(&self) -> usize {
        self.n_v_heads * self.v_head_dim
    }
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
}

pub struct DeltaNet {
    pub cfg: DeltaNetConfig,
    pub in_proj_qkv: Linear,
    pub in_proj_z: Linear,
    pub in_proj_b: Linear,
    pub in_proj_a: Linear,
    /// `[conv_dim × kernel]`
    pub conv_w: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub a_log: Vec<f32>,
    /// Gated RMSNorm weight over `v_head_dim` (plain weight, no +1 offset).
    pub norm_w: Vec<f32>,
    pub out_proj: Linear,
    pub eps: f32,
}

pub struct DeltaCache {
    x: Vec<f32>,
    conv: Vec<f32>, // pre-silu conv output [t × conv_dim]
    q_hat: Vec<f32>,
    k_hat: Vec<f32>,
    q_n: Vec<f32>,
    k_n: Vec<f32>,
    q: Vec<f32>, // normalised & scaled [t × nk × dk]
    k: Vec<f32>, // normalised [t × nk × dk]
    v: Vec<f32>, // [t × nv × dv]
    z: Vec<f32>,
    a_raw: Vec<f32>, // [t × nv]
    beta: Vec<f32>,
    alpha: Vec<f32>,
    o: Vec<f32>, // [t × nv × dv]
    o_inv: Vec<f32>,
    normed: Vec<f32>,
    gated: Vec<f32>,
    /// Per v-head: states S before steps 0, chunk, 2·chunk, … each `dv·dk`.
    ckpt: Vec<Vec<f32>>,
}

/// Inputs to the recurrence for one value head.
struct HeadIo<'a> {
    t: usize,
    dk: usize,
    dv: usize,
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    nk: usize,
    nv: usize,
    kh: usize,
    h: usize,
    beta: &'a [f32],
    alpha: &'a [f32],
}

impl<'a> HeadIo<'a> {
    #[inline]
    fn qv(&self, t: usize) -> &[f32] {
        &self.q[(t * self.nk + self.kh) * self.dk..(t * self.nk + self.kh + 1) * self.dk]
    }
    #[inline]
    fn kv(&self, t: usize) -> &[f32] {
        &self.k[(t * self.nk + self.kh) * self.dk..(t * self.nk + self.kh + 1) * self.dk]
    }
    #[inline]
    fn vv(&self, t: usize) -> &[f32] {
        &self.v[(t * self.nv + self.h) * self.dv..(t * self.nv + self.h + 1) * self.dv]
    }
    #[inline]
    fn b(&self, t: usize) -> f32 {
        self.beta[t * self.nv + self.h]
    }
    #[inline]
    fn a(&self, t: usize) -> f32 {
        self.alpha[t * self.nv + self.h]
    }

    /// One recurrence step in place: `s` holds S_{t-1} on entry and S_t on exit.
    /// `u` receives `v - S'k`. Returns nothing; S' is `alpha·S_{t-1}`.
    fn step(&self, s: &mut [f32], t: usize, u: &mut [f32]) {
        let (dk, dv) = (self.dk, self.dv);
        let (al, be) = (self.a(t), self.b(t));
        let kt = self.kv(t);
        let vt = self.vv(t);
        for i in 0..dv {
            let row = &mut s[i * dk..(i + 1) * dk];
            let mut sk = 0.0;
            for j in 0..dk {
                row[j] *= al;
                sk += row[j] * kt[j];
            }
            u[i] = vt[i] - sk;
            let c = be * u[i];
            for j in 0..dk {
                row[j] += c * kt[j];
            }
        }
    }

    fn output(&self, s: &[f32], t: usize, o: &mut [f32]) {
        let qt = self.qv(t);
        for i in 0..self.dv {
            o[i] = crate::tensor::dot(&s[i * self.dk..(i + 1) * self.dk], qt);
        }
    }
}

impl DeltaNet {
    fn head_io<'a>(
        &self,
        t: usize,
        h: usize,
        q: &'a [f32],
        k: &'a [f32],
        v: &'a [f32],
        beta: &'a [f32],
        alpha: &'a [f32],
    ) -> HeadIo<'a> {
        let c = &self.cfg;
        let ratio = c.n_v_heads / c.n_k_heads;
        HeadIo {
            t,
            dk: c.k_head_dim,
            dv: c.v_head_dim,
            q,
            k,
            v,
            nk: c.n_k_heads,
            nv: c.n_v_heads,
            kh: h / ratio,
            h,
            beta,
            alpha,
        }
    }

    pub fn forward(&self, x: &[f32], t: usize) -> (Vec<f32>, DeltaCache) {
        let c = self.cfg.clone();
        let (nk, nv, dk, dv) = (c.n_k_heads, c.n_v_heads, c.k_head_dim, c.v_head_dim);
        let (kd, vd, cd) = (c.key_dim(), c.value_dim(), c.conv_dim());
        assert_eq!(nv % nk, 0, "n_v_heads must be a multiple of n_k_heads");

        let mixed = self.in_proj_qkv.forward(x, t);
        let conv = causal_conv1d(&mixed, &self.conv_w, t, cd, c.conv_kernel);
        let act = silu_vec(&conv);
        let mut q_hat = vec![0.0; t * kd];
        let mut k_hat = vec![0.0; t * kd];
        let mut v = vec![0.0; t * vd];
        for p in 0..t {
            let row = &act[p * cd..(p + 1) * cd];
            q_hat[p * kd..(p + 1) * kd].copy_from_slice(&row[..kd]);
            k_hat[p * kd..(p + 1) * kd].copy_from_slice(&row[kd..2 * kd]);
            v[p * vd..(p + 1) * vd].copy_from_slice(&row[2 * kd..]);
        }
        let (mut q, q_n) = l2norm(&q_hat, dk, 1e-6);
        let (k, k_n) = l2norm(&k_hat, dk, 1e-6);
        let qs = 1.0 / (dk as f32).sqrt();
        q.iter_mut().for_each(|v| *v *= qs);

        let z = self.in_proj_z.forward(x, t);
        let b_raw = self.in_proj_b.forward(x, t);
        let a_raw = self.in_proj_a.forward(x, t);
        let beta: Vec<f32> = b_raw.iter().map(|v| sigmoid(*v)).collect();
        let mut alpha = vec![0.0; t * nv];
        for p in 0..t {
            for h in 0..nv {
                let g = -self.a_log[h].exp() * softplus(a_raw[p * nv + h] + self.dt_bias[h]);
                alpha[p * nv + h] = g.exp();
            }
        }

        let chunk = c.bptt_chunk.max(1);
        let heads: Vec<(Vec<f32>, Vec<f32>)> = (0..nv)
            .into_par_iter()
            .map(|h| {
                let io = self.head_io(t, h, &q, &k, &v, &beta, &alpha);
                let mut s = vec![0.0; dv * dk];
                let mut u = vec![0.0; dv];
                let mut o = vec![0.0; t * dv];
                let mut ck = Vec::with_capacity(((t + chunk - 1) / chunk) * dv * dk);
                for p in 0..t {
                    if p % chunk == 0 {
                        ck.extend_from_slice(&s);
                    }
                    io.step(&mut s, p, &mut u);
                    io.output(&s, p, &mut o[p * dv..(p + 1) * dv]);
                }
                (o, ck)
            })
            .collect();
        let mut o = vec![0.0; t * vd];
        let mut ckpt = Vec::with_capacity(nv);
        for (h, (oh, ck)) in heads.into_iter().enumerate() {
            for p in 0..t {
                o[(p * nv + h) * dv..(p * nv + h + 1) * dv].copy_from_slice(&oh[p * dv..(p + 1) * dv]);
            }
            ckpt.push(ck);
        }

        let (normed, o_inv) = rmsnorm(&o, &self.norm_w, dv, self.eps, 0.0);
        let gated: Vec<f32> = normed.iter().zip(&z).map(|(n, zz)| n * silu(*zz)).collect();
        let y = self.out_proj.forward(&gated, t);
        let cache = DeltaCache {
            x: x.to_vec(),
            conv,
            q_hat,
            k_hat,
            q_n,
            k_n,
            q,
            k,
            v,
            z,
            a_raw,
            beta,
            alpha,
            o,
            o_inv,
            normed,
            gated,
            ckpt,
        };
        (y, cache)
    }

    pub fn backward(&mut self, cache: DeltaCache, dy: &[f32], t: usize) -> Vec<f32> {
        let c = self.cfg.clone();
        let (nk, nv, dk, dv) = (c.n_k_heads, c.n_v_heads, c.k_head_dim, c.v_head_dim);
        let (kd, vd, cd) = (c.key_dim(), c.value_dim(), c.conv_dim());
        let chunk = c.bptt_chunk.max(1);

        // out_proj and gated norm.
        let d_gated = self.out_proj.backward(&cache.gated, dy, t);
        let mut d_normed = vec![0.0; t * vd];
        let mut dz = vec![0.0; t * vd];
        for i in 0..t * vd {
            let zz = cache.z[i];
            d_normed[i] = d_gated[i] * silu(zz);
            dz[i] = d_gated[i] * cache.normed[i] * silu_grad(zz);
        }
        let d_o = rmsnorm_backward(&cache.o, &self.norm_w, &cache.o_inv, &d_normed, dv, 0.0);

        // BPTT, parallel over value heads.
        let (q, k, v, beta, alpha) = (&cache.q, &cache.k, &cache.v, &cache.beta, &cache.alpha);
        let this = &*self;
        let heads: Vec<HeadGrads> = (0..nv)
            .into_par_iter()
            .map(|h| {
                let io = this.head_io(t, h, q, k, v, beta, alpha);
                head_backward(&io, &cache.ckpt[h], &d_o, chunk)
            })
            .collect();

        let mut dq = vec![0.0; t * kd];
        let mut dk_ = vec![0.0; t * kd];
        let mut dv_ = vec![0.0; t * vd];
        let mut dbeta = vec![0.0; t * nv];
        let mut dalpha = vec![0.0; t * nv];
        let ratio = nv / nk;
        for (h, g) in heads.into_iter().enumerate() {
            let kh = h / ratio;
            for p in 0..t {
                for j in 0..dk {
                    dq[(p * nk + kh) * dk + j] += g.dq[p * dk + j];
                    dk_[(p * nk + kh) * dk + j] += g.dk[p * dk + j];
                }
                dv_[(p * nv + h) * dv..(p * nv + h + 1) * dv].copy_from_slice(&g.dv[p * dv..(p + 1) * dv]);
                dbeta[p * nv + h] = g.dbeta[p];
                dalpha[p * nv + h] = g.dalpha[p];
            }
        }

        // Gates: β = σ(b), α = exp(-exp(A_log)·softplus(a + dt_bias)).
        let mut db = vec![0.0; t * nv];
        let mut da = vec![0.0; t * nv];
        for p in 0..t {
            for h in 0..nv {
                let i = p * nv + h;
                let be = beta[i];
                db[i] = dbeta[i] * be * (1.0 - be);
                let dg = dalpha[i] * alpha[i];
                da[i] = dg * (-self.a_log[h].exp()) * sigmoid(cache.a_raw[i] + self.dt_bias[h]);
            }
        }

        // q scale and l2 norms.
        let qs = 1.0 / (dk as f32).sqrt();
        dq.iter_mut().for_each(|v| *v *= qs);
        let q_unit: Vec<f32> = q.iter().map(|v| v / qs).collect();
        let dq_hat = l2norm_backward(&q_unit, &cache.q_n, &dq, dk);
        let dk_hat = l2norm_backward(k, &cache.k_n, &dk_, dk);
        let _ = (&cache.q_hat, &cache.k_hat);

        // Concatenate into d_act, through silu, conv, in_proj_qkv.
        let mut d_act = vec![0.0; t * cd];
        for p in 0..t {
            let row = &mut d_act[p * cd..(p + 1) * cd];
            row[..kd].copy_from_slice(&dq_hat[p * kd..(p + 1) * kd]);
            row[kd..2 * kd].copy_from_slice(&dk_hat[p * kd..(p + 1) * kd]);
            row[2 * kd..].copy_from_slice(&dv_[p * vd..(p + 1) * vd]);
        }
        let d_conv = silu_backward(&cache.conv, &d_act);
        let d_mixed = causal_conv1d_backward(&d_conv, &self.conv_w, t, cd, c.conv_kernel);

        let x = &cache.x;
        let mut dx = self.in_proj_qkv.backward(x, &d_mixed, t);
        let dxz = self.in_proj_z.backward(x, &dz, t);
        let dxb = self.in_proj_b.backward(x, &db, t);
        let dxa = self.in_proj_a.backward(x, &da, t);
        for i in 0..dx.len() {
            dx[i] += dxz[i] + dxb[i] + dxa[i];
        }
        dx
    }

    pub fn linears_mut(&mut self) -> Vec<&mut Linear> {
        vec![&mut self.in_proj_qkv, &mut self.in_proj_z, &mut self.in_proj_b, &mut self.in_proj_a, &mut self.out_proj]
    }
    pub fn linears(&self) -> Vec<&Linear> {
        vec![&self.in_proj_qkv, &self.in_proj_z, &self.in_proj_b, &self.in_proj_a, &self.out_proj]
    }
}

struct HeadGrads {
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
    dbeta: Vec<f32>,
    dalpha: Vec<f32>,
}

/// Reverse-mode through the recurrence for one value head.
fn head_backward(io: &HeadIo, ckpt: &[f32], d_o_all: &[f32], chunk: usize) -> HeadGrads {
    let (t, dk, dv) = (io.t, io.dk, io.dv);
    let sz = dv * dk;
    let mut g = HeadGrads {
        dq: vec![0.0; t * dk],
        dk: vec![0.0; t * dk],
        dv: vec![0.0; t * dv],
        dbeta: vec![0.0; t],
        dalpha: vec![0.0; t],
    };
    let mut ds = vec![0.0f32; sz]; // dL/dS_t carried backwards
    let mut states = vec![0.0f32; chunk * sz]; // S_{t-1} for each step in the chunk
    let mut s = vec![0.0f32; sz];
    let mut sp = vec![0.0f32; sz]; // S'
    let mut u = vec![0.0f32; dv];
    let mut w = vec![0.0f32; dv];
    let n_chunks = (t + chunk - 1) / chunk;

    for ci in (0..n_chunks).rev() {
        let start = ci * chunk;
        let end = (start + chunk).min(t);
        // Recompute the states inside this chunk from its checkpoint.
        s.copy_from_slice(&ckpt[ci * sz..(ci + 1) * sz]);
        for p in start..end {
            states[(p - start) * sz..(p - start + 1) * sz].copy_from_slice(&s);
            io.step(&mut s, p, &mut u);
        }
        for p in (start..end).rev() {
            let s_prev = &states[(p - start) * sz..(p - start + 1) * sz];
            let (al, be) = (io.a(p), io.b(p));
            let kt = io.kv(p);
            let qt = io.qv(p);
            let vt = io.vv(p);
            let dop = &d_o_all[(p * io.nv + io.h) * dv..(p * io.nv + io.h + 1) * dv];
            // S' and u, then S_t (into `s`).
            for i in 0..dv {
                let mut sk = 0.0;
                for j in 0..dk {
                    sp[i * dk + j] = al * s_prev[i * dk + j];
                    sk += sp[i * dk + j] * kt[j];
                }
                u[i] = vt[i] - sk;
                for j in 0..dk {
                    s[i * dk + j] = sp[i * dk + j] + be * u[i] * kt[j];
                }
            }
            // o = S_t q  ⇒  dS += do·qᵀ, dq = S_tᵀ·do
            let dq = &mut g.dq[p * dk..(p + 1) * dk];
            for i in 0..dv {
                let doi = dop[i];
                for j in 0..dk {
                    ds[i * dk + j] += doi * qt[j];
                    dq[j] += s[i * dk + j] * doi;
                }
            }
            // w = dS·k
            for i in 0..dv {
                w[i] = crate::tensor::dot(&ds[i * dk..(i + 1) * dk], kt);
            }
            // dv = β w ; dβ = u·w
            let mut dbeta = 0.0;
            for i in 0..dv {
                g.dv[p * dv + i] = be * w[i];
                dbeta += u[i] * w[i];
            }
            g.dbeta[p] = dbeta;
            // dk = β (dSᵀu − S'ᵀw)
            let dkp = &mut g.dk[p * dk..(p + 1) * dk];
            for i in 0..dv {
                let (ui, wi) = (u[i], w[i]);
                for j in 0..dk {
                    dkp[j] += be * (ds[i * dk + j] * ui - sp[i * dk + j] * wi);
                }
            }
            // dS' = dS − β w kᵀ ; dα = <dS', S_{t-1}> ; dS_{t-1} = α dS'
            let mut dal = 0.0;
            for i in 0..dv {
                let c = be * w[i];
                for j in 0..dk {
                    let v = ds[i * dk + j] - c * kt[j];
                    dal += v * s_prev[i * dk + j];
                    ds[i * dk + j] = al * v;
                }
            }
            g.dalpha[p] = dal;
        }
    }
    g
}
