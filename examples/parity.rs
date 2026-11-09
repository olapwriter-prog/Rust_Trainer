//! Parity harness against Hugging Face `transformers` + `peft`
//! (driven by `tools/parity.py`).
//!
//! `cargo run --release --example parity -- MODEL_DIR ADAPTER_DIR CASE.json OUT.json`
//!
//! CASE.json: `{"tokens":[...], "answer_start": N}`. Writes per-target log-probs,
//! the mean answer-token loss and the gradient of that loss for every LoRA
//! tensor (PEFT names), so autograd and our backward pass can be compared.

use rust_trainer::adapter::load_adapter_into;
use rust_trainer::hf::{load_model, LoadOptions};
use rust_trainer::model::answer_targets;
use serde_json::{json, Map, Value};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(a.len(), 4, "usage: parity MODEL_DIR ADAPTER_DIR CASE.json OUT.json");
    let mut model = load_model(&a[0], &LoadOptions { quantize_q8: false }).expect("load model");
    let case: Value = serde_json::from_slice(&std::fs::read(&a[2]).unwrap()).unwrap();
    let tokens: Vec<u32> = case["tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let start = case["answer_start"].as_u64().unwrap() as usize;
    let targets = answer_targets(&tokens, start);

    let base_lp = model.logprobs(&tokens, &targets);
    // Fingerprint check is skipped for externally produced PEFT adapters.
    let meta = rust_trainer::adapter::AdapterMeta {
        alpha: case["lora_alpha"].as_f64().unwrap() as f32,
        rank: case["r"].as_u64().unwrap() as usize,
        ..Default::default()
    };
    rust_trainer::adapter::write_json(std::path::Path::new(&a[1]).join("trainer_meta.json"), &meta).unwrap();
    load_adapter_into(&mut model, &a[1]).expect("load adapter");
    let w = vec![1.0 / targets.len() as f32; targets.len()];
    model.zero_grad();
    let lp = model.forward_backward(&tokens, &targets, &w);
    let loss = -lp.iter().sum::<f32>() / targets.len() as f32;
    let mut grads = Map::new();
    for l in model.lora_linears() {
        let lo = l.lora.as_ref().unwrap();
        grads.insert(format!("base_model.model.{}.lora_A.weight", l.name), json!(lo.ga));
        grads.insert(format!("base_model.model.{}.lora_B.weight", l.name), json!(lo.gb));
    }
    let out = json!({"base_logprobs": base_lp, "logprobs": lp, "loss": loss, "grads": grads});
    std::fs::write(&a[3], serde_json::to_vec(&out).unwrap()).unwrap();
}
