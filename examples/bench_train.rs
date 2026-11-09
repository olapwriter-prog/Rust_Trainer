//! Training throughput on real model shapes (random weights) through the engine.
//!
//! `cargo run --release --example bench_train -- PRESET|FILE.gguf [WTYPE] [SEQ] [STEPS] [THREADS]`
//! e.g. `bench_train qwen2.5-0.5b Q4_0 128 3 4`. Prints seconds per step and
//! trained tokens per second (forward + recomputation + backward).

use rust_trainer::model::{answer_targets, DEFAULT_TARGETS};
use rust_trainer::rie::rie_core::{gguf::GgufFile, quant::GgmlType, testing::SyntheticSpec};
use rust_trainer::rie::{load_gguf_from, GgufOptions};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let preset = a.first().map(String::as_str).unwrap_or("qwen2.5-0.5b");
    let wtype = match a.get(1).map(String::as_str).unwrap_or("Q4_0") {
        "Q8_0" => GgmlType::Q8_0,
        "Q4_K" => GgmlType::Q4_K,
        "F16" => GgmlType::F16,
        _ => GgmlType::Q4_0,
    };
    let seq: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    let steps: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let threads: Option<usize> = a.get(4).and_then(|s| s.parse().ok());
    let rank: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(8);
    let t0 = Instant::now();
    // A preset name builds the model in memory; a .gguf path is memory-mapped.
    let file = if preset.ends_with(".gguf") {
        Arc::new(GgufFile::open(preset).unwrap())
    } else {
        Arc::new(GgufFile::from_bytes(SyntheticSpec::preset(preset, wtype).expect("unknown preset").build()).unwrap())
    };
    let mut model = load_gguf_from(file, &GgufOptions { n_threads: threads, pin_threads: false }).unwrap().model;
    let n = model.attach_lora(rank, 2.0 * rank as f32, DEFAULT_TARGETS, 1);
    eprintln!(
        "{preset} {wtype:?}: {:.0} MB frozen, {:.2} M LoRA params (r={rank}), built in {:.1}s",
        model.frozen_bytes() as f64 / 1e6,
        n as f64 / 1e6,
        t0.elapsed().as_secs_f32()
    );
    let tokens: Vec<u32> = (0..seq as u32).map(|i| 3 + (i * 7919) % 300).collect();
    let targets = answer_targets(&tokens, seq / 2);
    let w = vec![1.0 / targets.len() as f32; targets.len()];
    let mut times = Vec::new();
    for s in 0..steps {
        let t = Instant::now();
        model.zero_grad();
        model.forward_backward(&tokens, &targets, &w);
        let dt = t.elapsed().as_secs_f64();
        eprintln!("step {s}: {dt:.2}s");
        times.push(dt);
    }
    let best = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let t = Instant::now();
    let _ = model.logprobs(&tokens, &targets);
    let fwd = t.elapsed().as_secs_f64();
    println!(
        "{preset} {wtype:?} seq={seq} rank={rank} threads={}: train step {best:.2}s = {:.0} tok/s (forward only {fwd:.2}s = {:.0} tok/s)",
        threads.map(|t| t.to_string()).unwrap_or("auto".into()),
        seq as f64 / best,
        seq as f64 / fwd
    );
}
