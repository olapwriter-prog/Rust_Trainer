//! agentd's training-job protocol (drop-in for `agentos-llm train`).
#![cfg(feature = "rie")]

use rust_trainer::agentd::{load_aglora, run, AgentdJob, Encoders};
use rust_trainer::model::answer_targets;
use rust_trainer::rie::rie_core::{gguf::GgufFile, quant::GgmlType, testing::SyntheticSpec};
use rust_trainer::rie::{load_gguf_from, GgufOptions};
use serde_json::json;
use std::sync::Arc;

fn model() -> rust_trainer::model::Model {
    let mut s = SyntheticSpec::tiny("qwen3", GgmlType::Q8_0);
    s.n_layer = 2;
    let f = Arc::new(GgufFile::from_bytes(s.build()).unwrap());
    load_gguf_from(f, &GgufOptions { n_threads: Some(2), pin_threads: false }).unwrap().model
}

fn export(dir: &std::path::Path) -> std::path::PathBuf {
    let tools = [("call", "to"), ("flashlight", "on"), ("read_calendar", "day")];
    let words = ["the", "a", "phone", "fast", "in", "on"];
    let mut lines = Vec::new();
    for i in 0..30 {
        let (tool, arg) = tools[i % 3];
        let w = words[i % 6];
        let answer = json!({"args": {arg: w}, "tool": tool}).to_string(); // agentd sorts keys
        lines.push(
            json!({"messages": [
                {"role": "system", "content": "you run the phone"},
                {"role": "user", "content": format!("{tool} {w} {i}")},
                {"role": "assistant", "content": answer}
            ]})
            .to_string(),
        );
    }
    let p = dir.join("sft.jsonl");
    std::fs::write(&p, lines.join("\n")).unwrap();
    p
}

fn job(dir: &std::path::Path, max_secs: Option<u64>) -> AgentdJob {
    AgentdJob { data: export(dir), out: dir.join("adapter.new.lora"), steps: 6, batch: 2, max_secs, max_len: 256, ..Default::default() }
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("rt-agentd-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn encoders_run(m: &mut rust_trainer::model::Model, j: &AgentdJob) -> serde_json::Value {
    // The synthetic vocabulary is SentencePiece bytes: use the engine tokenizer via a fresh load.
    let mut s = SyntheticSpec::tiny("qwen3", GgmlType::Q8_0);
    s.n_layer = 2;
    let f = Arc::new(GgufFile::from_bytes(s.build()).unwrap());
    let tok = rust_trainer::rie::rie_core::Tokenizer::from_gguf(&f).unwrap();
    let t2 = tok.encode("x", false, false); // warm
    let _ = t2;
    let tok = Arc::new(tok);
    let (a, b) = (tok.clone(), tok.clone());
    let prompt = move |s: &str| a.encode(s, true, true);
    let text = move |s: &str| b.encode(s, false, false);
    let enc = Encoders { prompt: &prompt, text: &text, im_end: 2 };
    run(m, &enc, j, &mut |l| eprintln!("{l}")).unwrap()
}

#[test]
fn chunked_job_resumes_exactly_and_writes_aglora() {
    // Uninterrupted reference.
    let d1 = tmp("full");
    let mut m1 = model();
    let full = encoders_run(&mut m1, &job(&d1, None));
    assert_eq!(full["done"], true);

    // agentd's way: a time-boxed launch that stops at once, then launches until done.
    let d2 = tmp("chunked");
    let mut m = model();
    let first = encoders_run(&mut m, &job(&d2, Some(0)));
    assert_eq!(first["done"], false, "{first}");
    assert_eq!(first["step"], 0);
    assert!(d2.join("adapter.new.ckpt").exists());
    // a kill mid-run leaves the last checkpoint; resume continues it
    let mut m = model();
    let second = encoders_run(&mut m, &job(&d2, None));
    assert_eq!(second["done"], true, "{second}");
    assert_eq!(second["base"], full["base"], "base score is carried over, not re-judged");
    assert_eq!(second["tuned"], full["tuned"]);
    assert_eq!(second["loss"], full["loss"], "resumed run must equal the uninterrupted one");
    assert!(!d2.join("adapter.new.ckpt").exists(), "checkpoint removed when finished");

    if full["kept"] == true {
        // AGLORA1 round trip reproduces the trained model exactly.
        let mut fresh = model();
        let n = load_aglora(&mut fresh, d1.join("adapter.new.lora")).unwrap();
        assert_eq!(n, 2 * 4, "q/k/v/o on both layers");
        let toks: Vec<u32> = (3..40).collect();
        let t = answer_targets(&toks, 20);
        assert_eq!(fresh.logprobs(&toks, &t), m1.logprobs(&toks, &t));
        assert!(d1.join("adapter.new.gguf").exists(), "engine-loadable GGUF beside it");
    }
    let _ = std::fs::remove_dir_all(d1);
    let _ = std::fs::remove_dir_all(d2);
}
