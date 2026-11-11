//! Export a real model, a LoRA adapter, an example, and the trainer's loss and
//! analytic gradients for an independent float64 reference check
//! (`tools/reference_check.py`).
//!
//! `cargo run --release --example export_reference -- MODEL.gguf OUT.safetensors [RANK]`
//!
//! The frozen matrices are replaced by their exact f32 dequantisation, and f16 KV
//! rounding is turned off, so the trainer computes a smooth function that a
//! float64 reference can differentiate numerically. (With quantised weights the
//! engine rounds activations to int8 before every matmul, which finite differences
//! cannot see through; that path is covered by the bit-exactness tests instead.)

use rust_trainer::frozen::{FrozenMatmul, FrozenWeight};
use rust_trainer::model::{answer_targets, Mixer, DEFAULT_TARGETS};
use rust_trainer::rie::{load_gguf, GgufOptions};
use rust_trainer::safetensors::write_f32;
use rust_trainer::tensor::Rng;
use std::collections::BTreeMap;
use std::sync::Arc;

fn dequant(w: &dyn FrozenMatmul) -> Vec<f32> {
    let (o, i) = (w.out_dim(), w.in_dim());
    let mut out = vec![0.0; o * i];
    for r in 0..o {
        w.row(r, &mut out[r * i..(r + 1) * i]);
    }
    out
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    assert!(a.len() >= 2, "usage: export_reference MODEL.gguf OUT.safetensors [RANK]");
    let rank: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let gm = load_gguf(&a[0], &GgufOptions { n_threads: None, pin_threads: false }).expect("load model");
    let tok = gm.tokenizer.clone();
    let mut model = gm.model;

    // Smooth f32 model: dequantised weights, no f16 KV rounding.
    let embed = Arc::new(FrozenWeight::f32(dequant(model.embed.as_ref()), model.cfg.vocab_size, model.cfg.hidden_size));
    let tied = model.cfg.tie_embeddings;
    let lm = if tied {
        embed.clone()
    } else {
        Arc::new(FrozenWeight::f32(dequant(model.lm_head.as_ref()), model.cfg.vocab_size, model.cfg.hidden_size))
    };
    model.embed = embed;
    model.lm_head = lm;
    for b in &mut model.blocks {
        if let Mixer::Attn(at) = &mut b.mixer {
            at.cfg.kv_f16 = false;
        }
        for l in b.linears_mut() {
            let (o, i) = (l.out_dim(), l.in_dim());
            l.w = Arc::new(FrozenWeight::f32(dequant(l.w.as_ref()), o, i));
        }
    }
    model.cfg.attn.kv_f16 = false;

    // LoRA on every projection, with non-zero B so every gradient is exercised.
    model.attach_lora(rank, 2.0 * rank as f32, DEFAULT_TARGETS, 7);
    let mut rng = Rng::new(11);
    for l in model.lora_linears_mut() {
        let lo = l.lora.as_mut().unwrap();
        lo.b.iter_mut().for_each(|v| *v = rng.normal() * 0.02);
    }

    let prompt = "<|im_start|>user\nturn on the torch please<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let mut tokens = tok.encode(prompt, true, true);
    let start = tokens.len();
    tokens.extend(tok.encode("{\"tool\":\"flashlight\",\"args\":{\"on\":\"on\"}}", false, false));
    tokens.push(tok.token_id("<|im_end|>").unwrap());
    let targets = answer_targets(&tokens, start);
    let w = vec![1.0 / targets.len() as f32; targets.len()];
    model.zero_grad();
    let lp = model.forward_backward(&tokens, &targets, &w);
    let loss: f64 = -lp.iter().map(|v| *v as f64).sum::<f64>() / targets.len() as f64;
    eprintln!("{} tokens, {} answer targets, loss {loss:.6}", tokens.len(), targets.len());

    // Collect everything the reference needs.
    let c = &model.cfg;
    let mut owned: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    owned.push(("token_embd".into(), vec![c.vocab_size, c.hidden_size], dequant(model.embed.as_ref())));
    if !tied {
        owned.push(("output".into(), vec![c.vocab_size, c.hidden_size], dequant(model.lm_head.as_ref())));
    }
    owned.push(("output_norm".into(), vec![c.hidden_size], model.final_norm.clone()));
    let mut lora_names = Vec::new();
    for (i, b) in model.blocks.iter().enumerate() {
        owned.push((format!("blk.{i}.attn_norm"), vec![c.hidden_size], b.input_norm.clone()));
        owned.push((format!("blk.{i}.ffn_norm"), vec![c.hidden_size], b.post_norm.clone()));
        if let Mixer::Attn(at) = &b.mixer {
            if let Some(n) = &at.q_norm {
                owned.push((format!("blk.{i}.attn_q_norm"), vec![n.len()], n.clone()));
            }
            if let Some(n) = &at.k_norm {
                owned.push((format!("blk.{i}.attn_k_norm"), vec![n.len()], n.clone()));
            }
        }
        for l in b.linears() {
            let short = l.name.clone();
            owned.push((format!("{short}.weight"), vec![l.out_dim(), l.in_dim()], dequant(l.w.as_ref())));
            if let Some(bias) = &l.bias {
                owned.push((format!("{short}.bias"), vec![bias.len()], bias.clone()));
            }
            if let Some(lo) = &l.lora {
                owned.push((format!("{short}.lora_a"), vec![lo.rank, l.in_dim()], lo.a.clone()));
                owned.push((format!("{short}.lora_b"), vec![l.out_dim(), lo.rank], lo.b.clone()));
                owned.push((format!("{short}.grad_a"), vec![lo.rank, l.in_dim()], lo.ga.clone()));
                owned.push((format!("{short}.grad_b"), vec![l.out_dim(), lo.rank], lo.gb.clone()));
                lora_names.push(short);
            }
        }
    }
    owned.push(("tokens".into(), vec![tokens.len()], tokens.iter().map(|&t| t as f32).collect()));
    let meta = serde_json::json!({
        "arch": c.arch, "n_layer": c.n_layers(), "hidden": c.hidden_size, "vocab": c.vocab_size,
        "n_head": c.attn.n_heads, "n_kv_head": c.attn.n_kv_heads, "head_dim": c.attn.head_dim,
        "rot_dim": c.attn.rot_dim, "rope_theta": c.attn.rope_theta, "rope_neox": c.attn.rope_neox,
        "rope_freq_scale": c.attn.rope_freq_scale, "rms_eps": c.rms_eps, "tied": tied,
        "lora_scale": 2.0, "rank": rank, "answer_start": start, "loss": loss, "lora": lora_names,
    });
    let mut md = BTreeMap::new();
    md.insert("reference".to_string(), meta.to_string());
    let views: Vec<(String, Vec<usize>, &[f32])> = owned.iter().map(|(n, s, v)| (n.clone(), s.clone(), v.as_slice())).collect();
    write_f32(&a[1], &views, &md).expect("write");
    eprintln!("wrote {}", a[1]);
}
