//! Minimal dense f32 kernels. Everything is row-major: a `[T × D]` activation is
//! `T` rows of `D` contiguous floats.

use rayon::prelude::*;

/// Rows per rayon task in the matmul kernels.
const ROW_CHUNK: usize = 4;

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    // 4-way unrolled accumulation: faster and slightly more accurate than a naive loop.
    let n = a.len();
    let mut s = [0.0f32; 4];
    let chunks = n / 4;
    for i in 0..chunks {
        let j = i * 4;
        s[0] += a[j] * b[j];
        s[1] += a[j + 1] * b[j + 1];
        s[2] += a[j + 2] * b[j + 2];
        s[3] += a[j + 3] * b[j + 3];
    }
    let mut tail = 0.0;
    for j in chunks * 4..n {
        tail += a[j] * b[j];
    }
    (s[0] + s[1]) + (s[2] + s[3]) + tail
}

#[inline]
pub fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}

#[inline]
pub fn add_into(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

/// `out[m×n] = a[m×k] · bᵀ` where `b` is `[n×k]` (i.e. `x · Wᵀ` for a weight `W[out×in]`).
pub fn matmul_nt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    let mut out = vec![0.0; m * n];
    if n == 0 {
        return out;
    }
    out.par_chunks_mut(n * ROW_CHUNK)
        .enumerate()
        .for_each(|(ci, oc)| {
            for (ri, orow) in oc.chunks_mut(n).enumerate() {
                let i = ci * ROW_CHUNK + ri;
                let arow = &a[i * k..(i + 1) * k];
                for (j, o) in orow.iter_mut().enumerate() {
                    *o = dot(arow, &b[j * k..(j + 1) * k]);
                }
            }
        });
    out
}

/// `out[m×n] = a[m×k] · b[k×n]`.
pub fn matmul_nn(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut out = vec![0.0; m * n];
    if n == 0 {
        return out;
    }
    out.par_chunks_mut(n * ROW_CHUNK)
        .enumerate()
        .for_each(|(ci, oc)| {
            for (ri, orow) in oc.chunks_mut(n).enumerate() {
                let i = ci * ROW_CHUNK + ri;
                for p in 0..k {
                    let av = a[i * k + p];
                    if av != 0.0 {
                        axpy(av, &b[p * n..(p + 1) * n], orow);
                    }
                }
            }
        });
    out
}

/// `out[m×n] += aᵀ · b` where `a` is `[k×m]` and `b` is `[k×n]`.
/// This is the weight-gradient shape: `dW[out×in] += dYᵀ[out×T] · X[T×in]`.
pub fn matmul_tn_acc(out: &mut [f32], a: &[f32], b: &[f32], k: usize, m: usize, n: usize) {
    assert_eq!(a.len(), k * m);
    assert_eq!(b.len(), k * n);
    assert_eq!(out.len(), m * n);
    if n == 0 {
        return;
    }
    out.par_chunks_mut(n).enumerate().for_each(|(i, orow)| {
        for p in 0..k {
            let av = a[p * m + i];
            if av != 0.0 {
                axpy(av, &b[p * n..(p + 1) * n], orow);
            }
        }
    });
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// d/dx silu(x)
#[inline]
pub fn silu_grad(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

#[inline]
pub fn softplus(x: f32) -> f32 {
    // Matches torch.nn.functional.softplus (threshold 20).
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// Small deterministic RNG (SplitMix64) so training is reproducible across
/// pause/resume without pulling in a dependency.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Rng {
    pub state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: seed ^ 0x9E37_79B9_7F4A_7C15 }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Uniform in [-bound, bound).
    pub fn uniform_sym(&mut self, bound: f32) -> f32 {
        (self.uniform() * 2.0 - 1.0) * bound
    }
    /// Standard normal (Box–Muller).
    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
    pub fn vec_normal(&mut self, n: usize, std: f32) -> Vec<f32> {
        (0..n).map(|_| self.normal() * std).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmuls_agree() {
        let mut r = Rng::new(1);
        let (m, k, n) = (5, 7, 3);
        let a = r.vec_normal(m * k, 1.0);
        let b = r.vec_normal(n * k, 1.0);
        let c = matmul_nt(&a, &b, m, k, n);
        // bᵀ as [k×n]
        let mut bt = vec![0.0; k * n];
        for j in 0..n {
            for p in 0..k {
                bt[p * n + j] = b[j * k + p];
            }
        }
        let c2 = matmul_nn(&a, &bt, m, k, n);
        for (x, y) in c.iter().zip(&c2) {
            assert!((x - y).abs() < 1e-5);
        }
        // (aᵀ)ᵀ·bt == a·bt using tn with a stored as [m×k] -> treat a as [k'=m rows]
        let mut c3 = vec![0.0; k * n];
        // out[k×n] = aᵀ[k×m]·c[m×n]
        matmul_tn_acc(&mut c3, &a, &c, m, k, n);
        let mut c4 = vec![0.0; k * n];
        for p in 0..k {
            for j in 0..n {
                for i in 0..m {
                    c4[p * n + j] += a[i * k + p] * c[i * n + j];
                }
            }
        }
        for (x, y) in c3.iter().zip(&c4) {
            assert!((x - y).abs() < 1e-4);
        }
    }
}
