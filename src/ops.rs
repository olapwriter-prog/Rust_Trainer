//! Elementwise / per-row ops with hand-written backward passes.

use crate::tensor::{dot, silu, silu_grad};

/// RMSNorm over rows of width `d`: `y = x / rms(x) · (w + offset)`.
/// Qwen2.5 uses `offset = 0`; Qwen3.5 ("zero-centred" norm) uses `offset = 1`.
/// Returns `(y, inv_rms per row)`.
pub fn rmsnorm(x: &[f32], w: &[f32], d: usize, eps: f32, offset: f32) -> (Vec<f32>, Vec<f32>) {
    rmsnorm_ordered(x, w, d, eps, offset, false)
}

/// RMSNorm with an explicit rounding order: `(x·s)·w` (`fused_scale = false`, the
/// engine's block/final norms) or `x·(s·w)` (`true`, the engine's in-place QK-norm).
pub fn rmsnorm_ordered(x: &[f32], w: &[f32], d: usize, eps: f32, offset: f32, fused_scale: bool) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(w.len(), d);
    assert_eq!(x.len() % d, 0);
    let rows = x.len() / d;
    let mut y = vec![0.0; x.len()];
    let mut inv = vec![0.0; rows];
    for r in 0..rows {
        let xr = &x[r * d..(r + 1) * d];
        let ms = dot(xr, xr) / d as f32;
        let ir = 1.0 / (ms + eps).sqrt();
        inv[r] = ir;
        if fused_scale {
            for i in 0..d {
                y[r * d + i] = xr[i] * (ir * (w[i] + offset));
            }
        } else {
            for i in 0..d {
                y[r * d + i] = xr[i] * ir * (w[i] + offset);
            }
        }
    }
    (y, inv)
}

/// Input gradient of [`rmsnorm`] (the weight is frozen).
pub fn rmsnorm_backward(x: &[f32], w: &[f32], inv: &[f32], dy: &[f32], d: usize, offset: f32) -> Vec<f32> {
    let rows = x.len() / d;
    let mut dx = vec![0.0; x.len()];
    let mut g = vec![0.0; d];
    for r in 0..rows {
        let xr = &x[r * d..(r + 1) * d];
        let dyr = &dy[r * d..(r + 1) * d];
        for i in 0..d {
            g[i] = dyr[i] * (w[i] + offset);
        }
        let ir = inv[r];
        let c = dot(&g, xr) * ir * ir * ir / d as f32;
        for i in 0..d {
            dx[r * d + i] = g[i] * ir - xr[i] * c;
        }
    }
    dx
}

/// L2-normalise rows of width `d`: `y = x / sqrt(Σx² + eps)`. Returns `(y, norms)`.
pub fn l2norm(x: &[f32], d: usize, eps: f32) -> (Vec<f32>, Vec<f32>) {
    let rows = x.len() / d;
    let mut y = vec![0.0; x.len()];
    let mut n = vec![0.0; rows];
    for r in 0..rows {
        let xr = &x[r * d..(r + 1) * d];
        let nr = (dot(xr, xr) + eps).sqrt();
        n[r] = nr;
        for i in 0..d {
            y[r * d + i] = xr[i] / nr;
        }
    }
    (y, n)
}

/// `dx = (dy - y·(y·dy)) / n`
pub fn l2norm_backward(y: &[f32], norms: &[f32], dy: &[f32], d: usize) -> Vec<f32> {
    let rows = y.len() / d;
    let mut dx = vec![0.0; y.len()];
    for r in 0..rows {
        let yr = &y[r * d..(r + 1) * d];
        let dyr = &dy[r * d..(r + 1) * d];
        let p = dot(yr, dyr);
        for i in 0..d {
            dx[r * d + i] = (dyr[i] - yr[i] * p) / norms[r];
        }
    }
    dx
}

/// Precomputed cos/sin for rotary embedding over the first `rot_dim` dims of
/// each head. NEOX style rotates `(x[i], x[i+rot/2])` (Qwen); normal / GPT-J
/// style rotates `(x[2i], x[2i+1])` (Llama GGUF layout). Frequencies are
/// computed exactly like the inference engine (`rie_core::ops::Rope`), in f32.
///
/// It also carries the *token layout* attention must respect: each token's
/// position, and for packed batches a segment per token. Segment 0 is a prompt
/// shared by the whole batch; a token in segment `k > 0` may attend to the shared
/// prompt and to earlier tokens of its own segment only. That lets several examples
/// that start with the same prompt be computed in one pass, the prompt once.
pub struct Rope {
    pub rot_dim: usize,
    pub neox: bool,
    pub cos: Vec<f32>, // [t × rot_dim/2]
    pub sin: Vec<f32>,
    pub segments: Option<Vec<u32>>,
}

impl Rope {
    pub fn new(t: usize, rot_dim: usize, theta: f32) -> Self {
        Self::with(t, rot_dim, true, theta, 1.0, None)
    }

    pub fn with(t: usize, rot_dim: usize, neox: bool, base: f32, freq_scale: f32, factors: Option<&[f32]>) -> Self {
        let positions: Vec<usize> = (0..t).collect();
        Self::at_positions(&positions, None, rot_dim, neox, base, freq_scale, factors)
    }

    /// Tables for tokens at arbitrary `positions`, with an optional segment layout (see above).
    pub fn at_positions(
        positions: &[usize],
        segments: Option<Vec<u32>>,
        rot_dim: usize,
        neox: bool,
        base: f32,
        freq_scale: f32,
        factors: Option<&[f32]>,
    ) -> Self {
        assert!(rot_dim % 2 == 0);
        if let Some(s) = &segments {
            assert_eq!(s.len(), positions.len());
        }
        let t = positions.len();
        let half = rot_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| {
                let mut f = base.powf(-2.0 * i as f32 / rot_dim as f32) * freq_scale;
                if let Some(fac) = factors {
                    if i < fac.len() && fac[i] != 0.0 {
                        f /= fac[i];
                    }
                }
                f
            })
            .collect();
        let mut cos = vec![0.0; t * half];
        let mut sin = vec![0.0; t * half];
        for (p, &pos) in positions.iter().enumerate() {
            for (i, f) in inv_freq.iter().enumerate() {
                let (s, c) = (pos as f32 * f).sin_cos();
                cos[p * half + i] = c;
                sin[p * half + i] = s;
            }
        }
        Rope { rot_dim, neox, cos, sin, segments }
    }

    /// May token `i` attend to token `j`? Causal, and within the segment layout.
    #[inline]
    pub fn allowed(&self, i: usize, j: usize) -> bool {
        j <= i
            && match &self.segments {
                None => true,
                Some(s) => s[j] == 0 || s[j] == s[i],
            }
    }

    /// In-place rotation of `x` laid out `[t × heads × head_dim]`.
    /// `inverse = true` applies the transpose (which is the backward pass).
    pub fn apply(&self, x: &mut [f32], t: usize, heads: usize, head_dim: usize, inverse: bool) {
        let half = self.rot_dim / 2;
        let sgn = if inverse { -1.0 } else { 1.0 };
        for p in 0..t {
            for h in 0..heads {
                let base = (p * heads + h) * head_dim;
                for i in 0..half {
                    let c = self.cos[p * half + i];
                    let s = sgn * self.sin[p * half + i];
                    let (a, b) = if self.neox { (base + i, base + i + half) } else { (base + 2 * i, base + 2 * i + 1) };
                    let (x1, x2) = (x[a], x[b]);
                    x[a] = x1 * c - x2 * s;
                    x[b] = x2 * c + x1 * s;
                }
            }
        }
    }
}

/// Causal depthwise conv1d over time: `x` `[t × c]`, `w` `[c × k]` (HF layout
/// `[c, 1, k]` squeezed). `y[t, c] = Σ_j w[c, j] · x[t - k + 1 + j, c]`.
pub fn causal_conv1d(x: &[f32], w: &[f32], t: usize, c: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0; t * c];
    for ti in 0..t {
        for j in 0..k {
            let src = ti as isize - (k as isize - 1) + j as isize;
            if src < 0 {
                continue;
            }
            let src = src as usize;
            for ch in 0..c {
                y[ti * c + ch] += w[ch * k + j] * x[src * c + ch];
            }
        }
    }
    y
}

pub fn causal_conv1d_backward(dy: &[f32], w: &[f32], t: usize, c: usize, k: usize) -> Vec<f32> {
    let mut dx = vec![0.0; t * c];
    for ti in 0..t {
        for j in 0..k {
            let src = ti as isize - (k as isize - 1) + j as isize;
            if src < 0 {
                continue;
            }
            let src = src as usize;
            for ch in 0..c {
                dx[src * c + ch] += w[ch * k + j] * dy[ti * c + ch];
            }
        }
    }
    dx
}

pub fn silu_vec(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| silu(*v)).collect()
}

/// `dx = dy · silu'(x)`
pub fn silu_backward(x: &[f32], dy: &[f32]) -> Vec<f32> {
    x.iter().zip(dy).map(|(a, g)| g * silu_grad(*a)).collect()
}

/// Numerically stable log-softmax of one row, returning `(logsumexp, probs)`.
///
/// The normaliser is a sum over the whole vocabulary (152k terms for Qwen). A
/// sequential f32 sum of that many positive terms drifts by ~1e-4 relative, which
/// shifted every log-prob and scaled every softmax gradient; it is accumulated in
/// f64 instead (found by `tools/reference_check.py` against a float64 reference).
pub fn softmax_row(logits: &[f32]) -> (f32, Vec<f32>) {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = logits.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = p.iter().map(|&v| v as f64).sum();
    let inv = (1.0 / s) as f32;
    for v in p.iter_mut() {
        *v *= inv;
    }
    (m + s.ln() as f32, p)
}
