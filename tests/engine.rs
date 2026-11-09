//! The trainer against the Rust Inference Engine (`rie-core`).
//!
//! Synthetic GGUF models are built with the engine's own test builder, so no
//! download is needed. The engine's `CpuBackend` is the reference.
#![cfg(feature = "rie")]

use rust_trainer::adapter::{base_fingerprint, save_adapter, AdapterMeta};
use rust_trainer::data::Text;
use rust_trainer::eval::{evaluate_vs_base, GateConfig};
use rust_trainer::model::{answer_targets, Model, Target, DEFAULT_TARGETS};
use rust_trainer::rie::rie_core::gguf::GgufFile;
use rust_trainer::rie::rie_core::model::cpu::CpuBackend;
use rust_trainer::rie::rie_core::model::Backend;
use rust_trainer::rie::rie_core::quant::GgmlType;
use rust_trainer::rie::rie_core::testing::SyntheticSpec;
use rust_trainer::rie::rie_core::threadpool::{PoolConfig, ThreadPool};
use rust_trainer::rie::{load_gguf_from, read_gguf_adapter, EngineChat, GgufOptions, RieFrozen, GGUF_ADAPTER_FILE};
use rust_trainer::tensor::Rng;
use rust_trainer::tokenizer::{Example, Formatter};
use rust_trainer::train::{RunToEnd, TrainConfig, TrainData, Trainer};
use std::sync::Arc;

fn opts() -> GgufOptions {
    GgufOptions { n_threads: Some(2), pin_threads: false }
}

fn synth(arch: &str, wtype: GgmlType, tied: bool) -> Arc<GgufFile> {
    let mut s = SyntheticSpec::tiny(arch, wtype);
    s.tied_output = tied;
    Arc::new(GgufFile::from_bytes(s.build()).unwrap())
}

/// Engine logits for every position of `tokens`.
fn engine_logits(file: &Arc<GgufFile>, tokens: &[u32]) -> (Vec<f32>, usize) {
    let rm = Arc::new(rust_trainer::rie::rie_core::model::Model::load(file.clone()).unwrap());
    let nv = rm.cfg.n_vocab;
    let pool = Arc::new(ThreadPool::new(PoolConfig { n_threads: 2, cores: vec![], spin: Default::default() }));
    let mut be = CpuBackend::new(rm, pool, 128, 64);
    let mut logits = vec![0.0; tokens.len() * nv];
    be.forward(tokens, 0, true, &mut logits).unwrap();
    (logits, nv)
}

fn log_softmax(row: &[f32]) -> Vec<f32> {
    let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = m + row.iter().map(|v| (v - m).exp()).sum::<f32>().ln();
    row.iter().map(|v| v - lse).collect()
}

/// Forward parity with the engine's `CpuBackend` across architectures and
/// quantisations: **bit-exact**. The trainer runs the engine's own matmul kernels
/// (including int8 activation quantisation), reads K/V as f16 with the engine's
/// `dot_f16_f32`/`axpy_f16`, uses the engine's `dot_f32`, and matches its
/// operation order in RMSNorm, QK-norm, SiLU, RoPE and softmax. Any drift would
/// be amplified by int8 rounding of activations, so exact equality is required.
#[test]
fn logits_match_engine() {
    let mut rng = Rng::new(5);
    let tokens: Vec<u32> = (0..24).map(|_| 3 + rng.below(300) as u32).collect();
    for arch in ["llama", "qwen2", "qwen3"] {
        for wt in [GgmlType::F32, GgmlType::F16, GgmlType::Q8_0, GgmlType::Q4_0, GgmlType::Q4_K, GgmlType::Q6_K] {
            for tied in [false, true] {
                let file = synth(arch, wt, tied);
                let (ref_logits, nv) = engine_logits(&file, &tokens);
                let gm = load_gguf_from(file.clone(), &opts()).unwrap();
                assert_eq!(gm.model.cfg.tie_embeddings, tied);
                let mut diffs = Vec::new();
                for pos in [0usize, 11, 23] {
                    let reference = log_softmax(&ref_logits[pos * nv..(pos + 1) * nv]);
                    let targets: Vec<Target> = (0..nv as u32).map(|tok| Target { pos, token: tok }).collect();
                    let ours = gm.model.logprobs(&tokens, &targets);
                    diffs.extend(ours.iter().zip(&reference).map(|(a, b)| (a - b).abs()));
                }
                let worst = diffs.iter().cloned().fold(0.0f32, f32::max);
                let close = diffs.iter().filter(|d| **d < 1e-3).count() as f32 / diffs.len() as f32;
                let argmax_targets: Vec<Target> = (0..tokens.len())
                    .map(|pos| {
                        let row = &ref_logits[pos * nv..(pos + 1) * nv];
                        let tok = (0..nv).max_by(|&i, &j| row[i].total_cmp(&row[j])).unwrap() as u32;
                        Target { pos, token: tok }
                    })
                    .collect();
                let (_, hits) = gm.model.score(&tokens, &argmax_targets);
                eprintln!(
                    "{arch:>6} {wt:?} tied={tied}: max |Δ log p| {worst:.1e}, {:.1}% < 1e-3, greedy {hits}/{}",
                    close * 100.0,
                    tokens.len()
                );
                assert_eq!(worst, 0.0, "{arch} {wt:?} tied={tied}: not bit-exact ({:.1}% < 1e-3)", close * 100.0);
                assert_eq!(hits, tokens.len(), "{arch} {wt:?}: greedy agreed on {hits}/{}", tokens.len());
            }
        }
    }
}

/// The dequantised input gradient equals a dense `dy·W` for every engine format.
#[test]
fn input_gradient_matches_dense_reference() {
    let mut rng = Rng::new(9);
    for wt in [GgmlType::F32, GgmlType::F16, GgmlType::Q8_0, GgmlType::Q4_0, GgmlType::Q4_1, GgmlType::Q5_0, GgmlType::Q4_K, GgmlType::Q6_K]
    {
        let file = synth("qwen2", wt, false);
        let gm = load_gguf_from(file, &opts()).unwrap();
        let lin = &gm.model.blocks[0].mlp.down_proj;
        let (out, inp, t) = (lin.out_dim(), lin.in_dim(), 3);
        let mut w = vec![0.0; out * inp];
        for r in 0..out {
            lin.w.row(r, &mut w[r * inp..(r + 1) * inp]);
        }
        let dy = rng.vec_normal(t * out, 1.0);
        let mut dx = vec![0.0; t * inp];
        lin.w.backward_input(&dy, t, &mut dx);
        for ti in 0..t {
            for i in (0..inp).step_by(37) {
                let want: f32 = (0..out).map(|o| dy[ti * out + o] * w[o * inp + i]).sum();
                assert!((dx[ti * inp + i] - want).abs() < 1e-3 * want.abs().max(1.0), "{wt:?}");
            }
        }
        let _ = std::any::type_name::<RieFrozen>();
    }
}

/// Numerical LoRA gradients through the engine-backed model (F32 weights, so the
/// engine keeps activations in f32 and finite differences are meaningful),
/// including Llama's normal-style RoPE and Qwen3's QK-norm.
#[test]
fn lora_grads_through_engine_model() {
    for arch in ["llama", "qwen2", "qwen3"] {
        let file = synth(arch, GgmlType::F32, true);
        let mut model = load_gguf_from(file, &opts()).unwrap().model;
        // f16 KV rounding is step-wise, which defeats finite differences; it is the
        // identity in the backward pass, so check the smooth model.
        for b in &mut model.blocks {
            if let rust_trainer::model::Mixer::Attn(a) = &mut b.mixer {
                a.cfg.kv_f16 = false;
            }
        }
        let n = model.attach_lora(2, 4.0, DEFAULT_TARGETS, 1);
        assert_eq!(model.lora_linears().len(), 7 * 2, "{arch}: every projection adapted");
        assert!(n > 0);
        let mut rng = Rng::new(2);
        for l in model.lora_linears_mut() {
            l.lora.as_mut().unwrap().b.iter_mut().for_each(|v| *v = rng.normal() * 0.05);
        }
        let tokens: Vec<u32> = (0..10).map(|_| 3 + rng.below(300) as u32).collect();
        let targets = answer_targets(&tokens, 5);
        let w = vec![1.0 / targets.len() as f32; targets.len()];
        model.zero_grad();
        model.forward_backward(&tokens, &targets, &w);
        let loss = |m: &Model| -> f64 { -m.logprobs(&tokens, &targets).iter().map(|v| *v as f64).sum::<f64>() / targets.len() as f64 };
        let n_lin = model.lora_linears().len();
        for li in (0..n_lin).step_by(3) {
            for which in 0..2 {
                let (idx, analytic) = {
                    let lo = model.lora_linears()[li].lora.clone().unwrap();
                    let (p, g) = if which == 0 { (lo.a, lo.ga) } else { (lo.b, lo.gb) };
                    let i = rng.below(p.len());
                    (i, g[i] as f64)
                };
                let eps = 1e-2f32;
                let bump = |m: &mut Model, d: f32| {
                    let mut ls = m.lora_linears_mut();
                    let lo = ls[li].lora.as_mut().unwrap();
                    if which == 0 {
                        lo.a[idx] += d
                    } else {
                        lo.b[idx] += d
                    }
                };
                bump(&mut model, eps);
                let lp = loss(&model);
                bump(&mut model, -2.0 * eps);
                let lm = loss(&model);
                bump(&mut model, eps);
                let numeric = (lp - lm) / (2.0 * eps as f64);
                let name = model.lora_linears()[li].name.clone();
                assert!(
                    (analytic - numeric).abs() <= 3e-2 * analytic.abs().max(numeric.abs()) + 2e-4,
                    "{arch} {name}: analytic {analytic} vs numeric {numeric}"
                );
            }
        }
    }
}

/// Prompts are tokenised exactly as `Engine::chat` does.
#[test]
fn prompt_tokens_match_engine_chat() {
    let file = synth("qwen2", GgmlType::Q8_0, false);
    let gm = load_gguf_from(file, &opts()).unwrap();
    let fmt = EngineChat::new(gm.tokenizer.clone(), gm.template, 256).unwrap();
    let msgs = vec![
        rust_trainer::rie::rie_core::chat::Message::new(rust_trainer::rie::rie_core::chat::Role::System, "be brief"),
        rust_trainer::rie::rie_core::chat::Message::new(rust_trainer::rie::rie_core::chat::Role::User, "the phone"),
    ];
    let want = gm.tokenizer.encode(&gm.template.render(&msgs, true), true, true);
    let got = fmt.prompt_tokens(Some("be brief"), &Text::Str("the phone".into()));
    assert_eq!(got, want);
    let ex = fmt.example(Some("be brief"), &Text::Str("the phone".into()), &Text::Str("fast".into())).unwrap();
    assert_eq!(*ex.tokens.last().unwrap(), fmt.end_token);
    assert_eq!(&ex.tokens[..ex.answer_start], want.as_slice());
}

/// Training on a Q4_0 model through the engine kernels, then the GGUF adapter
/// round-trip the engine will load.
#[test]
fn train_q4_model_and_gguf_adapter_roundtrip() {
    let file = synth("qwen3", GgmlType::Q4_0, true);
    let gm = load_gguf_from(file.clone(), &opts()).unwrap();
    let fmt = EngineChat::new(gm.tokenizer.clone(), gm.template, 64).unwrap();
    let words = ["the", "a", "phone", "fast", "in", "er", "to", "on"];
    let mut train: Vec<Example> = Vec::new();
    let mut held: Vec<Example> = Vec::new();
    for (i, a) in words.iter().enumerate() {
        for (j, b) in words.iter().enumerate() {
            let ex = fmt.example(None, &Text::Str(format!("{a} {b}")), &Text::Str(format!("{b} {a}!"))).unwrap();
            if (i + j) % 5 == 0 {
                held.push(ex)
            } else {
                train.push(ex)
            }
        }
    }
    let mut model = gm.model;
    let cfg = TrainConfig { rank: 4, alpha: 8.0, lr: 4e-3, epochs: 2, batch_size: 8, warmup_steps: 2, ..Default::default() };
    let mut t = Trainer::new(&mut model, cfg, TrainData::Sft(train));
    t.run(&mut RunToEnd, None, |_| {}).unwrap();
    let l = &t.state.losses;
    assert!(l[l.len() - 1] < 0.7 * l[0], "loss {} -> {}", l[0], l[l.len() - 1]);
    let rep = evaluate_vs_base(t.model, &GateConfig { min_examples: 4, ..Default::default() }, &held, &[]);
    assert!(rep.candidate.nll < rep.current.nll, "{rep:?}");

    let dir = std::env::temp_dir().join(format!("rt-gguf-adapter-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let meta = AdapterMeta { id: "g".into(), base_fingerprint: base_fingerprint(t.model), rank: 4, alpha: 8.0, ..Default::default() };
    save_adapter(t.model, &dir, &meta).unwrap();
    let gguf_path = dir.join(GGUF_ADAPTER_FILE);
    assert!(gguf_path.exists());
    let g = GgufFile::open(&gguf_path).unwrap();
    assert_eq!(g.get_str("general.type"), Some("adapter"));
    assert_eq!(g.get_str("adapter.type"), Some("lora"));
    assert_eq!(g.get_str("general.architecture"), Some("qwen3"));
    assert_eq!(g.get_f32("adapter.lora.alpha"), Some(8.0));
    let ta = g.tensor("blk.0.attn_q.weight.lora_a").unwrap();
    assert_eq!(ta.dims[0] as usize, t.model.blocks[0].linears()[0].in_dim());
    assert_eq!(ta.dims[1], 4);

    let ex = &held[0];
    let targets = answer_targets(&ex.tokens, ex.answer_start);
    let want = t.model.logprobs(&ex.tokens, &targets);
    let mut fresh = load_gguf_from(file, &opts()).unwrap().model;
    let n = read_gguf_adapter(&mut fresh, &gguf_path).unwrap();
    assert_eq!(n, t.model.lora_linears().len());
    assert_eq!(fresh.logprobs(&ex.tokens, &targets), want);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn unsupported_architectures_are_refused() {
    for arch in ["gemma2", "phi3", "granite"] {
        let file = synth(arch, GgmlType::Q8_0, false);
        assert!(load_gguf_from(file, &opts()).is_err(), "{arch} should be refused");
    }
}

#[test]
fn f16_rounding_matches_engine_kv_cache() {
    use rust_trainer::rie::rie_core::kernels::{f32_to_f16_slice, fp16_to_f32};
    let mut rng = Rng::new(3);
    let mut xs: Vec<f32> = (0..20000).map(|i| rng.normal() * 10f32.powi(i % 9 - 6)).collect();
    xs.extend([0.0, -0.0, 65504.0, 1e-8, 6.1e-5, 2049.0, 2051.0]);
    let mut h = vec![0u16; xs.len()];
    f32_to_f16_slice(&xs, &mut h);
    for (x, hb) in xs.iter().zip(&h) {
        assert_eq!(rust_trainer::attention::round_f16(*x), fp16_to_f32(*hb), "x = {x:e}");
    }
}

/// Same bit-exactness check on a real GGUF model. Opt-in (needs a file):
/// `RT_GGUF=qwen3-0.6b-q4_0.gguf cargo test --release --test engine real_model -- --ignored`
#[test]
#[ignore]
fn real_model_logits_match_engine() {
    let Ok(path) = std::env::var("RT_GGUF") else { return };
    let file = Arc::new(GgufFile::open(&path).unwrap());
    let gm = load_gguf_from(file.clone(), &opts()).unwrap();
    let tokens = gm.tokenizer.encode("<|im_start|>user\nturn on the lamp in the kitchen<|im_end|>\n<|im_start|>assistant\n", true, true);
    let (ref_logits, nv) = engine_logits(&file, &tokens);
    let positions: Vec<usize> = (0..tokens.len()).collect();
    let ours = gm.model.logits_at(&tokens, &positions);
    let worst = ours[..tokens.len() * nv].iter().zip(&ref_logits).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("{path}: {} tokens × vocab {nv} logits, max |Δ| = {worst:e}", tokens.len());
    assert_eq!(worst, 0.0);
}
