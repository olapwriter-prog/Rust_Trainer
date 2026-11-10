//! Every backward pass is checked against central finite differences.
//!
//! Inputs are checked with directional derivatives (`dL/dx · d` for a random `d`),
//! LoRA parameters coordinate-wise on a random sample of entries.

use rust_trainer::model::{answer_targets, Mixer, Model, ModelConfig, Target};
use rust_trainer::ops::*;
use rust_trainer::tensor::Rng;

fn dot64(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

fn close(analytic: f64, numeric: f64, rtol: f64, atol: f64) -> bool {
    (analytic - numeric).abs() <= rtol * analytic.abs().max(numeric.abs()) + atol
}

/// Directional check of an input gradient. `f` maps x→y, `back` maps (x, dy)→dx.
fn check_input<F, B>(name: &str, x: &[f32], out_len: usize, f: F, back: B)
where
    F: Fn(&[f32]) -> Vec<f32>,
    B: Fn(&[f32], &[f32]) -> Vec<f32>,
{
    let mut rng = Rng::new(name.len() as u64 * 7919);
    let r = rng.vec_normal(out_len, 1.0);
    for trial in 0..3 {
        let d = rng.vec_normal(x.len(), 1.0);
        let dx = back(x, &r);
        let analytic = dot64(&dx, &d);
        let eps = 2e-3f32;
        let xp: Vec<f32> = x.iter().zip(&d).map(|(a, b)| a + eps * b).collect();
        let xm: Vec<f32> = x.iter().zip(&d).map(|(a, b)| a - eps * b).collect();
        let numeric = (dot64(&f(&xp), &r) - dot64(&f(&xm), &r)) / (2.0 * eps as f64);
        assert!(close(analytic, numeric, 2e-2, 2e-3), "{name} trial {trial}: analytic {analytic} vs numeric {numeric}");
    }
}

#[test]
fn rmsnorm_grad() {
    let mut rng = Rng::new(1);
    let d = 16;
    let x = rng.vec_normal(3 * d, 1.0);
    for off in [0.0, 1.0] {
        let w = rng.vec_normal(d, 0.5);
        check_input(
            &format!("rmsnorm off={off}"),
            &x,
            x.len(),
            |x| rmsnorm(x, &w, d, 1e-6, off).0,
            |x, dy| {
                let (_, inv) = rmsnorm(x, &w, d, 1e-6, off);
                rmsnorm_backward(x, &w, &inv, dy, d, off)
            },
        );
    }
}

#[test]
fn l2norm_grad() {
    let mut rng = Rng::new(2);
    let d = 8;
    let x = rng.vec_normal(4 * d, 1.0);
    check_input(
        "l2norm",
        &x,
        x.len(),
        |x| l2norm(x, d, 1e-6).0,
        |x, dy| {
            let (y, n) = l2norm(x, d, 1e-6);
            l2norm_backward(&y, &n, dy, d)
        },
    );
}

#[test]
fn conv_silu_grad() {
    let mut rng = Rng::new(3);
    let (t, c, k) = (7, 5, 4);
    let x = rng.vec_normal(t * c, 1.0);
    let w = rng.vec_normal(c * k, 0.5);
    check_input(
        "conv1d+silu",
        &x,
        t * c,
        |x| silu_vec(&causal_conv1d(x, &w, t, c, k)),
        |x, dy| {
            let conv = causal_conv1d(x, &w, t, c, k);
            causal_conv1d_backward(&silu_backward(&conv, dy), &w, t, c, k)
        },
    );
}

#[test]
fn rope_grad() {
    let mut rng = Rng::new(4);
    let (t, h, hd) = (5, 3, 8);
    let rope = Rope::new(t, 4, 10000.0);
    let x = rng.vec_normal(t * h * hd, 1.0);
    check_input(
        "rope",
        &x,
        x.len(),
        |x| {
            let mut y = x.to_vec();
            rope.apply(&mut y, t, h, hd, false);
            y
        },
        |_, dy| {
            let mut dx = dy.to_vec();
            rope.apply(&mut dx, t, h, hd, true);
            dx
        },
    );
}

fn randomize_lora_b(model: &mut Model, seed: u64) {
    let mut rng = Rng::new(seed);
    for l in model.lora_linears_mut() {
        let lo = l.lora.as_mut().unwrap();
        for v in lo.b.iter_mut() {
            *v = rng.normal() * 0.1;
        }
    }
}

/// Input gradient of every mixer type (attention variants and DeltaNet with
/// several BPTT chunk sizes, including chunk = 1 and chunk > T).
#[test]
fn mixer_input_grads() {
    let t = 7;
    for (label, cfg) in [("dense", ModelConfig::tiny_dense(50)), ("hybrid", ModelConfig::tiny_hybrid(50))] {
        for chunk in [1usize, 3, 4, 7, 20] {
            let mut cfg = cfg.clone();
            if let Some(d) = cfg.delta.as_mut() {
                d.bptt_chunk = chunk;
            } else if chunk != 1 {
                continue;
            }
            let mut model = Model::random(cfg, 11, false);
            model.attach_lora(2, 4.0, rust_trainer::model::DEFAULT_TARGETS, 5);
            randomize_lora_b(&mut model, 6);
            let d = model.hidden();
            let rope = Rope::new(t, model.cfg.attn.rot_dim, model.cfg.attn.rope_theta);
            let mut rng = Rng::new(9);
            let x = rng.vec_normal(t * d, 1.0);
            for (bi, block) in model.blocks.iter_mut().enumerate() {
                let name = format!("{label} block{bi} chunk{chunk}");
                // Mixers are borrowed mutably for backward; use a RefCell-free split.
                let mixer = &mut block.mixer;
                let ptr: *mut Mixer = mixer;
                // SAFETY: `f` only reads through the shared reborrow, and `back` runs
                // after `f` has returned; never concurrently.
                let fwd = |x: &[f32]| -> Vec<f32> {
                    match unsafe { &*ptr } {
                        Mixer::Attn(a) => a.forward(x, t, &rope).0,
                        Mixer::Delta(dn) => dn.forward(x, t).0,
                    }
                };
                let back = |x: &[f32], dy: &[f32]| -> Vec<f32> {
                    match unsafe { &mut *ptr } {
                        Mixer::Attn(a) => {
                            let (_, c) = a.forward(x, t, &rope);
                            a.backward(c, dy, t, &rope)
                        }
                        Mixer::Delta(dn) => {
                            let (_, c) = dn.forward(x, t);
                            dn.backward(c, dy, t)
                        }
                    }
                };
                check_input(&name, &x, t * d, fwd, back);
            }
        }
    }
}

/// End-to-end: LoRA parameter gradients through the whole model and the
/// answer-token loss, with and without activation recomputation.
#[test]
fn model_lora_grads() {
    for (label, cfg, quant) in [
        ("dense-f32", ModelConfig::tiny_dense(40), false),
        ("hybrid-f32", ModelConfig::tiny_hybrid(40), false),
        ("hybrid-q8", ModelConfig::tiny_hybrid(40), true),
    ] {
        let mut model = Model::random(cfg, 21, quant);
        model.attach_lora(3, 6.0, rust_trainer::model::DEFAULT_TARGETS, 22);
        randomize_lora_b(&mut model, 23);
        let mut rng = Rng::new(24);
        let tokens: Vec<u32> = (0..9).map(|_| rng.below(40) as u32).collect();
        let targets = answer_targets(&tokens, 4);
        let w = vec![1.0 / targets.len() as f32; targets.len()];

        let loss_of = |m: &Model| -> f64 {
            let lp = m.logprobs(&tokens, &targets);
            -lp.iter().zip(&w).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>()
        };

        model.checkpointing = true;
        model.zero_grad();
        model.forward_backward(&tokens, &targets, &w);
        let grads_ckpt: Vec<(Vec<f32>, Vec<f32>)> =
            model.lora_linears().iter().map(|l| (l.lora.as_ref().unwrap().ga.clone(), l.lora.as_ref().unwrap().gb.clone())).collect();
        model.checkpointing = false;
        model.zero_grad();
        model.forward_backward(&tokens, &targets, &w);
        for (l, (ga, gb)) in model.lora_linears().iter().zip(&grads_ckpt) {
            let lo = l.lora.as_ref().unwrap();
            for (a, b) in lo.ga.iter().zip(ga).chain(lo.gb.iter().zip(gb)) {
                assert!((a - b).abs() <= 1e-5 * a.abs().max(1.0), "{label}: checkpointing changed grads");
            }
        }

        // Coordinate checks on a sample of entries in every adapted linear.
        let n_lin = model.lora_linears().len();
        let mut checked = 0;
        for li in 0..n_lin {
            for which in 0..2 {
                for _ in 0..2 {
                    let (len, analytic) = {
                        let l = &model.lora_linears()[li];
                        let lo = l.lora.as_ref().unwrap();
                        let (p, g) = if which == 0 { (&lo.a, &lo.ga) } else { (&lo.b, &lo.gb) };
                        let idx = rng.below(p.len());
                        (idx, g[idx] as f64)
                    };
                    let idx = len;
                    let eps = 1e-2f32;
                    let set = |m: &mut Model, delta: f32| {
                        let mut ls = m.lora_linears_mut();
                        let lo = ls[li].lora.as_mut().unwrap();
                        let p = if which == 0 { &mut lo.a } else { &mut lo.b };
                        p[idx] += delta;
                    };
                    set(&mut model, eps);
                    let lp = loss_of(&model);
                    set(&mut model, -2.0 * eps);
                    let lm = loss_of(&model);
                    set(&mut model, eps);
                    let numeric = (lp - lm) / (2.0 * eps as f64);
                    let name = model.lora_linears()[li].name.clone();
                    assert!(
                        close(analytic, numeric, 3e-2, 3e-4),
                        "{label} {name} {} [{idx}]: analytic {analytic} vs numeric {numeric}",
                        if which == 0 { "A" } else { "B" }
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 20);
    }
}

/// The weighted-logprob loss is linear in the weights, so a DPO-style
/// coefficient vector must give exactly the weighted sum of per-target grads.
#[test]
fn weights_are_linear() {
    let mut model = Model::random(ModelConfig::tiny_hybrid(30), 31, false);
    model.attach_lora(2, 2.0, rust_trainer::model::DEFAULT_TARGETS, 32);
    randomize_lora_b(&mut model, 33);
    let tokens: Vec<u32> = vec![1, 5, 7, 2, 9, 4];
    let targets: Vec<Target> = answer_targets(&tokens, 2);
    let n = targets.len();
    let grads = |m: &mut Model, w: &[f32]| -> Vec<f32> {
        m.zero_grad();
        m.forward_backward(&tokens, &targets, w);
        m.lora_linears().iter().flat_map(|l| l.lora.as_ref().unwrap().ga.clone()).collect()
    };
    let g1 = grads(&mut model, &vec![1.0; n]);
    let g2 = grads(&mut model, &vec![-0.5; n]);
    for (a, b) in g1.iter().zip(&g2) {
        assert!((a * -0.5 - b).abs() < 1e-5 * a.abs().max(1.0));
    }
}

/// A packed batch (shared prompt computed once) gives the same log-probs and LoRA
/// gradients as running each example separately.
#[test]
fn packed_batch_equals_separate_examples() {
    use rust_trainer::model::pack;
    let mut model = Model::random(ModelConfig::tiny_dense(50), 41, false);
    model.attach_lora(3, 6.0, rust_trainer::model::DEFAULT_TARGETS, 42);
    randomize_lora_b(&mut model, 43);
    let mut rng = Rng::new(44);
    let prompt: Vec<u32> = (0..13).map(|_| rng.below(50) as u32).collect();
    // three examples sharing the prompt, different lengths; one answer starts inside the shared part
    let mut seqs: Vec<(Vec<u32>, usize)> = Vec::new();
    for (extra, start) in [(5usize, 15usize), (9, 14), (3, 12)] {
        let mut s = prompt.clone();
        s.extend((0..extra).map(|_| rng.below(50) as u32));
        seqs.push((s, start));
    }
    let grads_of = |m: &Model| -> Vec<f32> {
        m.lora_linears()
            .iter()
            .flat_map(|l| {
                let lo = l.lora.as_ref().unwrap();
                lo.ga.iter().chain(&lo.gb).copied().collect::<Vec<_>>()
            })
            .collect()
    };

    // separate
    model.zero_grad();
    let mut sep_lp = Vec::new();
    for (s, start) in &seqs {
        let t = answer_targets(s, *start);
        let w = vec![0.1; t.len()];
        sep_lp.extend(model.forward_backward(s, &t, &w));
    }
    let g_sep = grads_of(&model);

    // packed
    let refs: Vec<&[u32]> = seqs.iter().map(|(s, _)| s.as_slice()).collect();
    let p = pack(&refs);
    assert_eq!(p.shared, 13, "the whole common prompt is shared");
    let mut targets = Vec::new();
    for (k, (s, start)) in seqs.iter().enumerate() {
        targets.extend(p.map_targets(k, &answer_targets(s, *start)));
    }
    let w = vec![0.1; targets.len()];
    model.zero_grad();
    let packed_lp = model.forward_backward_packed(&p, &targets, &w, &mut || false).unwrap();
    let g_pack = grads_of(&model);

    for (a, b) in sep_lp.iter().zip(&packed_lp) {
        assert!((a - b).abs() < 1e-5, "logprob {a} vs {b}");
    }
    let scale = g_sep.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    for (a, b) in g_sep.iter().zip(&g_pack) {
        assert!((a - b).abs() <= 1e-5 * scale.max(1.0), "grad {a} vs {b}");
    }
    // a batch of one is just the sequence
    let one = pack(&refs[..1]);
    assert_eq!(one.shared, 0);
    assert_eq!(one.tokens, seqs[0].0);
}
