//! Frozen (base-model) weight matrices.
//!
//! The base weights never receive gradients. The only thing training needs from
//! them is the forward product `y = x·Wᵀ` and the input gradient `dx = dy·W`.
//! That is the whole contract of [`FrozenMatmul`], so the inference engine's own
//! compressed matmul kernels can be plugged in by implementing this trait. No
//! dequantised copy of the weight is ever materialised.

use crate::tensor::{axpy, dot, matmul_nt};
use rayon::prelude::*;
use std::sync::Arc;

/// A frozen `[out × in]` weight matrix.
pub trait FrozenMatmul: Send + Sync {
    fn out_dim(&self) -> usize;
    fn in_dim(&self) -> usize;
    /// `y[t×out] = x[t×in] · Wᵀ`
    fn forward(&self, x: &[f32], t: usize) -> Vec<f32>;
    /// `dx[t×in] += dy[t×out] · W`. Gradient to the *input only*; W is frozen.
    fn backward_input(&self, dy: &[f32], t: usize, dx: &mut [f32]);
    /// Dequantise one row (used for embedding lookup when the table is tied to lm_head).
    fn row(&self, r: usize, out: &mut [f32]);
    /// Bytes the weight occupies in memory.
    fn nbytes(&self) -> usize;
}

pub type Frozen = Arc<dyn FrozenMatmul>;

/// Block size of Q8_0 (same as ggml/llama.cpp).
pub const QK8_0: usize = 32;

/// Built-in frozen weight storage.
pub enum FrozenWeight {
    F32 { w: Vec<f32>, out: usize, inp: usize },
    /// Q8_0: per-32-element block an f32 scale and 32 signed bytes.
    Q8_0 { scales: Vec<f32>, qs: Vec<i8>, out: usize, inp: usize },
}

impl FrozenWeight {
    pub fn f32(w: Vec<f32>, out: usize, inp: usize) -> Self {
        assert_eq!(w.len(), out * inp, "weight shape mismatch");
        FrozenWeight::F32 { w, out, inp }
    }

    /// Quantise to Q8_0 if the row length allows it; otherwise keep f32.
    pub fn q8_0(w: &[f32], out: usize, inp: usize) -> Self {
        assert_eq!(w.len(), out * inp);
        if inp % QK8_0 != 0 {
            return Self::f32(w.to_vec(), out, inp);
        }
        let nb = w.len() / QK8_0;
        let mut scales = Vec::with_capacity(nb);
        let mut qs = Vec::with_capacity(w.len());
        for blk in w.chunks(QK8_0) {
            let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            scales.push(d);
            for v in blk {
                qs.push((v * id).round().clamp(-127.0, 127.0) as i8);
            }
        }
        FrozenWeight::Q8_0 { scales, qs, out, inp }
    }

    pub fn into_arc(self) -> Frozen {
        Arc::new(self)
    }
}

#[inline]
fn q8_dot(scales: &[f32], qs: &[i8], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (b, d) in scales.iter().enumerate() {
        let q = &qs[b * QK8_0..(b + 1) * QK8_0];
        let xv = &x[b * QK8_0..(b + 1) * QK8_0];
        let mut s = 0.0f32;
        for i in 0..QK8_0 {
            s += q[i] as f32 * xv[i];
        }
        acc += d * s;
    }
    acc
}

#[inline]
fn q8_axpy(alpha: f32, scales: &[f32], qs: &[i8], y: &mut [f32]) {
    for (b, d) in scales.iter().enumerate() {
        let a = alpha * d;
        if a == 0.0 {
            continue;
        }
        let q = &qs[b * QK8_0..(b + 1) * QK8_0];
        let yv = &mut y[b * QK8_0..(b + 1) * QK8_0];
        for i in 0..QK8_0 {
            yv[i] += a * q[i] as f32;
        }
    }
}

impl FrozenMatmul for FrozenWeight {
    fn out_dim(&self) -> usize {
        match self {
            FrozenWeight::F32 { out, .. } | FrozenWeight::Q8_0 { out, .. } => *out,
        }
    }
    fn in_dim(&self) -> usize {
        match self {
            FrozenWeight::F32 { inp, .. } | FrozenWeight::Q8_0 { inp, .. } => *inp,
        }
    }

    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        match self {
            FrozenWeight::F32 { w, out, inp } => matmul_nt(x, w, t, *inp, *out),
            FrozenWeight::Q8_0 { scales, qs, out, inp } => {
                assert_eq!(x.len(), t * inp);
                let (out, inp) = (*out, *inp);
                let bpr = inp / QK8_0;
                let mut y = vec![0.0; t * out];
                y.par_chunks_mut(out).enumerate().for_each(|(ti, yr)| {
                    let xr = &x[ti * inp..(ti + 1) * inp];
                    for (o, yv) in yr.iter_mut().enumerate() {
                        *yv = q8_dot(
                            &scales[o * bpr..(o + 1) * bpr],
                            &qs[o * inp..(o + 1) * inp],
                            xr,
                        );
                    }
                });
                y
            }
        }
    }

    fn backward_input(&self, dy: &[f32], t: usize, dx: &mut [f32]) {
        let (out, inp) = (self.out_dim(), self.in_dim());
        assert_eq!(dy.len(), t * out);
        assert_eq!(dx.len(), t * inp);
        match self {
            FrozenWeight::F32 { w, .. } => {
                dx.par_chunks_mut(inp).enumerate().for_each(|(ti, dxr)| {
                    let dyr = &dy[ti * out..(ti + 1) * out];
                    for (o, g) in dyr.iter().enumerate() {
                        if *g != 0.0 {
                            axpy(*g, &w[o * inp..(o + 1) * inp], dxr);
                        }
                    }
                });
            }
            FrozenWeight::Q8_0 { scales, qs, .. } => {
                let bpr = inp / QK8_0;
                dx.par_chunks_mut(inp).enumerate().for_each(|(ti, dxr)| {
                    let dyr = &dy[ti * out..(ti + 1) * out];
                    for (o, g) in dyr.iter().enumerate() {
                        if *g != 0.0 {
                            q8_axpy(
                                *g,
                                &scales[o * bpr..(o + 1) * bpr],
                                &qs[o * inp..(o + 1) * inp],
                                dxr,
                            );
                        }
                    }
                });
            }
        }
    }

    fn row(&self, r: usize, out_row: &mut [f32]) {
        match self {
            FrozenWeight::F32 { w, inp, .. } => out_row.copy_from_slice(&w[r * inp..(r + 1) * inp]),
            FrozenWeight::Q8_0 { scales, qs, inp, .. } => {
                let bpr = inp / QK8_0;
                out_row.iter_mut().for_each(|v| *v = 0.0);
                q8_axpy(1.0, &scales[r * bpr..(r + 1) * bpr], &qs[r * inp..(r + 1) * inp], out_row);
            }
        }
    }

    fn nbytes(&self) -> usize {
        match self {
            FrozenWeight::F32 { w, .. } => w.len() * 4,
            FrozenWeight::Q8_0 { scales, qs, .. } => scales.len() * 4 + qs.len(),
        }
    }
}

/// Compute only selected rows of `x·Wᵀ` for a single input vector — used for
/// logits at answer positions so we never build a full `[T × vocab]` matrix.
pub fn forward_one(w: &dyn FrozenMatmul, x: &[f32]) -> Vec<f32> {
    w.forward(x, 1)
}

/// Helper used in tests: dense reference `y = x·Wᵀ`.
pub fn dense_reference(w: &[f32], x: &[f32], out: usize, inp: usize) -> Vec<f32> {
    (0..out).map(|o| dot(&w[o * inp..(o + 1) * inp], x)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Rng;

    #[test]
    fn q8_close_to_f32_and_backward_consistent() {
        let mut r = Rng::new(3);
        let (out, inp, t) = (6, 64, 3);
        let w = r.vec_normal(out * inp, 0.1);
        let x = r.vec_normal(t * inp, 1.0);
        let f = FrozenWeight::f32(w.clone(), out, inp);
        let q = FrozenWeight::q8_0(&w, out, inp);
        let yf = f.forward(&x, t);
        let yq = q.forward(&x, t);
        for (a, b) in yf.iter().zip(&yq) {
            assert!((a - b).abs() < 0.02, "{a} vs {b}");
        }
        // <dy, x·Wᵀ> == <dy·W, x>
        let dy = r.vec_normal(t * out, 1.0);
        for wq in [&f, &q] {
            let y = wq.forward(&x, t);
            let mut dx = vec![0.0; t * inp];
            wq.backward_input(&dy, t, &mut dx);
            let lhs = dot(&dy, &y);
            let rhs = dot(&dx, &x);
            assert!((lhs - rhs).abs() < 1e-3 * lhs.abs().max(1.0));
        }
    }
}
