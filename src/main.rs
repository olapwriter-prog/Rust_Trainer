//! `rust_trainer` command line.
//!
//! ```text
//! rust_trainer toy      [--arch hybrid|dense] [--steps N]
//! rust_trainer info     --model DIR [--rank R]
//! rust_trainer train    --model DIR --data FILE --out DIR [--ckpt FILE] [options]
//! rust_trainer eval     --model DIR --adapter DIR --data FILE [options]
//! rust_trainer run      --model DIR --data FILE --status FILE [--hours H] [--poll-secs S] [options]
//! rust_trainer rollback --model DIR [--reason TEXT]
//! rust_trainer status   --model DIR
//! ```
//! Options: `--rank 8 --alpha 16 --lr 2e-4 --epochs 2 --batch 4 --max-len 512
//! --heldout 0.1 --dpo BETA --q8 --tokenizer FILE --assistant-prefix TEXT
//! --min-improvement 0.02 --min-examples 8 --min-train 50 --min-battery 50 --max-temp 40`.

use rust_trainer::adapter::{base_fingerprint, load_adapter_into, save_adapter, AdapterMeta};
use rust_trainer::data::{Dataset, Text};
use rust_trainer::eval::{evaluate_vs_base, GateConfig};
use rust_trainer::hf::{estimate, load_model, LoadOptions};
use rust_trainer::model::{Model, ModelConfig, DEFAULT_TARGETS};
use rust_trainer::registry::{now_unix, Registry};
use rust_trainer::scheduler::{run_overnight, FileMonitor, Heldout, JobConfig, PowerPolicy};
use rust_trainer::tokenizer::{ByteTokenizer, ChatFormat, Example, Tokenizer};
use rust_trainer::train::{Objective, PrefPair, RunToEnd, TrainConfig, TrainData, Trainer};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

struct Args {
    cmd: String,
    kv: HashMap<String, String>,
    flags: Vec<String>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut it = std::env::args().skip(1);
        let cmd = it.next().ok_or_else(usage)?;
        let mut kv = HashMap::new();
        let mut flags = Vec::new();
        let rest: Vec<String> = it.collect();
        let mut i = 0;
        while i < rest.len() {
            let a = &rest[i];
            let key = a.strip_prefix("--").ok_or_else(|| format!("unexpected argument {a}"))?;
            if i + 1 < rest.len() && !rest[i + 1].starts_with("--") {
                kv.insert(key.to_string(), rest[i + 1].clone());
                i += 2;
            } else {
                flags.push(key.to_string());
                i += 1;
            }
        }
        Ok(Args { cmd, kv, flags })
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.kv.get(k).map(String::as_str)
    }
    fn req(&self, k: &str) -> Result<&str, String> {
        self.get(k).ok_or_else(|| format!("missing --{k}"))
    }
    fn num<T: std::str::FromStr>(&self, k: &str, d: T) -> Result<T, String> {
        match self.get(k) {
            None => Ok(d),
            Some(v) => v.parse().map_err(|_| format!("--{k}: cannot parse {v:?}")),
        }
    }
    fn flag(&self, k: &str) -> bool {
        self.flags.iter().any(|f| f == k)
    }
}

fn usage() -> String {
    "usage: rust_trainer <toy|info|train|eval|run|rollback|status> [--options]  (see README.md)".to_string()
}

fn main() -> ExitCode {
    let args = match Args::parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let r = match args.cmd.as_str() {
        "toy" => cmd_toy(&args),
        "info" => cmd_info(&args),
        "train" => cmd_train(&args),
        "eval" => cmd_eval(&args),
        "run" => cmd_run(&args),
        "rollback" => cmd_rollback(&args),
        "status" => cmd_status(&args),
        _ => Err(usage()),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn train_config(a: &Args) -> Result<TrainConfig, String> {
    let mut c = TrainConfig {
        rank: a.num("rank", 8)?,
        alpha: a.num("alpha", 16.0)?,
        lr: a.num("lr", 2e-4)?,
        epochs: a.num("epochs", 2)?,
        batch_size: a.num("batch", 4)?,
        warmup_steps: a.num("warmup", 10)?,
        seed: a.num("seed", 1234)?,
        checkpoint_every: a.num("checkpoint-every", 10)?,
        ..Default::default()
    };
    if let Some(t) = a.get("targets") {
        c.targets = t.split(',').map(str::to_string).collect();
    }
    if let Some(b) = a.get("dpo") {
        c.objective = Objective::Dpo { beta: b.parse().map_err(|_| "--dpo: expected a number")? };
    }
    Ok(c)
}

fn gate_config(a: &Args) -> Result<GateConfig, String> {
    Ok(GateConfig {
        min_rel_improvement: a.num("min-improvement", 0.02)?,
        min_examples: a.num("min-examples", 8)?,
        max_accuracy_drop: a.num("max-accuracy-drop", 0.0)?,
    })
}

fn load(a: &Args) -> Result<Model, String> {
    let dir = a.req("model")?;
    let mut m = load_model(dir, &LoadOptions { quantize_q8: a.flag("q8") }).map_err(|e| format!("loading {dir}: {e}"))?;
    if let Some(c) = a.get("bptt-chunk") {
        let c: usize = c.parse().map_err(|_| "--bptt-chunk")?;
        for b in &mut m.blocks {
            if let rust_trainer::model::Mixer::Delta(d) = &mut b.mixer {
                d.cfg.bptt_chunk = c;
            }
        }
    }
    Ok(m)
}

/// Raw-text data needs a tokenizer; pre-tokenised (`*_ids`) data does not.
fn tokenizer(a: &Args) -> Result<Option<Box<dyn Tokenizer>>, String> {
    let path = a
        .get("tokenizer")
        .map(PathBuf::from)
        .or_else(|| a.get("model").map(|m| Path::new(m).join("tokenizer.json")).filter(|p| p.exists()));
    match path {
        None => Ok(None),
        #[cfg(feature = "hf-tokenizers")]
        Some(p) => Ok(Some(Box::new(rust_trainer::tokenizer::HfTokenizer::from_file(&p)?))),
        #[cfg(not(feature = "hf-tokenizers"))]
        Some(p) => {
            eprintln!(
                "note: {} found but built without the `hf-tokenizers` feature; only pre-tokenised data can be used",
                p.display()
            );
            Ok(None)
        }
    }
}

fn needs_tokenizer(t: &Text) -> bool {
    !matches!(t, Text::Ids(_))
}

fn build_examples(
    ds: &Dataset,
    tok: Option<&dyn Tokenizer>,
    prefix: &str,
    max_len: usize,
) -> Result<(Vec<Example>, Vec<PrefPair>), String> {
    let any_text = ds.sft.iter().any(|r| needs_tokenizer(&r.prompt) || needs_tokenizer(&r.response))
        || ds.pref.iter().any(|r| needs_tokenizer(&r.prompt) || needs_tokenizer(&r.chosen) || needs_tokenizer(&r.rejected));
    let byte = ByteTokenizer;
    let tok: &dyn Tokenizer = match tok {
        Some(t) => t,
        None if any_text => {
            return Err("data contains raw text: pass --tokenizer tokenizer.json (build with --features hf-tokenizers) or export *_ids".into())
        }
        None => &byte, // only used for the ChatML special ids, which pre-tokenised data never needs
    };
    let fmt = ChatFormat::qwen(tok, prefix, max_len).or_else(|e| {
        if any_text { Err(e) } else { ChatFormat::qwen(&byte, prefix, max_len) }
    })?;
    let sft = ds
        .sft
        .iter()
        .filter_map(|r| fmt.example(tok, r.system.as_deref(), &r.prompt, &r.response))
        .collect();
    let pref = ds
        .pref
        .iter()
        .filter_map(|r| {
            Some(PrefPair {
                chosen: fmt.example(tok, r.system.as_deref(), &r.prompt, &r.chosen)?,
                rejected: fmt.example(tok, r.system.as_deref(), &r.prompt, &r.rejected)?,
            })
        })
        .collect();
    Ok((sft, pref))
}

fn default_prefix(m: &Model) -> &'static str {
    if m.cfg.arch == "qwen3_5" {
        "<think>\n\n</think>\n\n"
    } else {
        ""
    }
}

struct Prepared {
    train: TrainData,
    held_sft: Vec<Example>,
    held_pref: Vec<PrefPair>,
    n_train: usize,
}

fn prepare(a: &Args, model: &Model, cfg: &TrainConfig) -> Result<Prepared, String> {
    let path = a.req("data")?;
    let ds = Dataset::load_jsonl(path).map_err(|e| format!("{path}: {e}"))?;
    eprintln!(
        "data: {} supervised, {} preference; dropped {} unconfirmed, {} malformed",
        ds.sft.len(),
        ds.pref.len(),
        ds.dropped_unconfirmed,
        ds.dropped_malformed
    );
    let (train, held) = ds.split(a.num("heldout", 0.1)?);
    let tok = tokenizer(a)?;
    let prefix = a.get("assistant-prefix").map(|s| s.replace("\\n", "\n")).unwrap_or_else(|| default_prefix(model).into());
    let max_len = a.num("max-len", 512)?;
    let (tr_sft, tr_pref) = build_examples(&train, tok.as_deref(), &prefix, max_len)?;
    let (held_sft, held_pref) = build_examples(&held, tok.as_deref(), &prefix, max_len)?;
    for ex in tr_sft.iter().chain(&held_sft) {
        if let Some(&t) = ex.tokens.iter().find(|&&t| t as usize >= model.cfg.vocab_size) {
            return Err(format!("token id {t} exceeds the model vocab {}", model.cfg.vocab_size));
        }
    }
    let train = match cfg.objective {
        Objective::Sft => TrainData::Sft(tr_sft),
        Objective::Dpo { .. } => TrainData::Pref(tr_pref),
    };
    let n_train = train.len();
    Ok(Prepared { train, held_sft, held_pref, n_train })
}

fn log_line(s: &str) {
    eprintln!("{s}");
}

fn cmd_train(a: &Args) -> Result<(), String> {
    let mut model = load(a)?;
    let cfg = train_config(a)?;
    let p = prepare(a, &model, &cfg)?;
    if p.n_train == 0 {
        return Err("no confirmed training examples".into());
    }
    let out = PathBuf::from(a.req("out")?);
    let ckpt = a.get("ckpt").map(PathBuf::from).unwrap_or_else(|| out.with_extension("ckpt"));
    let mut tr = Trainer::new(&mut model, cfg.clone(), p.train);
    if ckpt.exists() {
        tr.load_checkpoint(&ckpt).map_err(|e| format!("resume {}: {e}", ckpt.display()))?;
        eprintln!("resumed at step {}/{}", tr.state.step, tr.state.total_steps);
    }
    eprintln!("trainable LoRA params: {}", tr.model.trainable_params());
    let t0 = Instant::now();
    tr.run(&mut RunToEnd, Some(&ckpt), log_line).map_err(|e| e.to_string())?;
    let steps = tr.state.step;
    let loss = tr.state.losses.last().copied();
    let model = tr.model;
    let meta = AdapterMeta {
        id: out.file_name().and_then(|s| s.to_str()).unwrap_or("adapter").to_string(),
        base_model: a.req("model")?.to_string(),
        base_fingerprint: base_fingerprint(model),
        rank: cfg.rank,
        alpha: cfg.alpha,
        targets: cfg.targets.clone(),
        created_unix: now_unix(),
        train_steps: steps,
        train_examples: p.n_train,
        final_train_loss: loss,
        eval: None,
    };
    save_adapter(model, &out, &meta).map_err(|e| e.to_string())?;
    eprintln!("saved adapter to {} after {:.1}s", out.display(), t0.elapsed().as_secs_f32());
    let rep = evaluate_vs_base(model, &gate_config(a)?, &p.held_sft, &p.held_pref);
    println!("{}", serde_json::to_string_pretty(&rep).unwrap());
    Ok(())
}

fn cmd_eval(a: &Args) -> Result<(), String> {
    let mut model = load(a)?;
    let cfg = TrainConfig::default();
    let p = prepare(a, &model, &cfg)?;
    load_adapter_into(&mut model, a.req("adapter")?).map_err(|e| e.to_string())?;
    let rep = evaluate_vs_base(&mut model, &gate_config(a)?, &p.held_sft, &p.held_pref);
    println!("{}", serde_json::to_string_pretty(&rep).unwrap());
    Ok(())
}

fn cmd_run(a: &Args) -> Result<(), String> {
    let mut model = load(a)?;
    let train = train_config(a)?;
    let p = prepare(a, &model, &train)?;
    let registry = Registry::open(a.req("model")?).map_err(|e| e.to_string())?;
    let job = JobConfig {
        base_model: a.req("model")?.to_string(),
        train,
        gate: gate_config(a)?,
        power: PowerPolicy {
            min_battery_pct: a.num("min-battery", 50.0)?,
            max_temp_c: a.num("max-temp", 40.0)?,
            ..Default::default()
        },
        min_train_examples: a.num("min-train", 50)?,
    };
    let mut monitor = FileMonitor(PathBuf::from(a.req("status")?));
    let deadline = Instant::now() + Duration::from_secs_f32(a.num("hours", 8.0f32)? * 3600.0);
    let poll = Duration::from_secs(a.num("poll-secs", 60)?);
    let mut wait = || {
        if Instant::now() + poll > deadline {
            return false;
        }
        std::thread::sleep(poll);
        true
    };
    let mut log = |s: &str| eprintln!("{s}");
    let outcome = run_overnight(
        &mut model,
        &registry,
        &job,
        p.train,
        Heldout { sft: &p.held_sft, pref: &p.held_pref },
        &mut monitor,
        &mut wait,
        &mut log,
    )
    .map_err(|e| e.to_string())?;
    println!("{}", serde_json::to_string_pretty(&outcome).unwrap());
    Ok(())
}

fn cmd_rollback(a: &Args) -> Result<(), String> {
    let r = Registry::open(a.req("model")?).map_err(|e| e.to_string())?;
    let to = r.rollback(a.get("reason").unwrap_or("manual")).map_err(|e| e.to_string())?;
    println!("active adapter: {}", to.as_deref().unwrap_or("(base model)"));
    Ok(())
}

fn cmd_status(a: &Args) -> Result<(), String> {
    let r = Registry::open(a.req("model")?).map_err(|e| e.to_string())?;
    println!("active:   {}", r.active().as_deref().unwrap_or("(base model)"));
    println!("accepted: {:?}", r.accepted());
    if let Ok(s) = std::fs::read_to_string(r.work_dir().join("status.json")) {
        println!("run:      {}", s.trim());
    }
    for e in r.read_ledger().iter().rev().take(10) {
        println!("  {} {:<9} {:<14} {}", e.ts, e.event, e.adapter.as_deref().unwrap_or("-"), e.detail);
    }
    Ok(())
}

fn cmd_info(a: &Args) -> Result<(), String> {
    let dir = a.req("model")?;
    let root: serde_json::Value = serde_json::from_slice(
        &std::fs::read(Path::new(dir).join("config.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let cfg = rust_trainer::hf::parse_config(&root).map_err(|e| e.to_string())?;
    let rank = a.num("rank", 8)?;
    let e = estimate(&cfg, rank, DEFAULT_TARGETS);
    println!("arch {} | layers {} | hidden {}", cfg.arch, cfg.n_layers(), cfg.hidden_size);
    println!("base params        {:>8.2} M", e.base_params as f64 / 1e6);
    println!("LoRA params (r={rank}) {:>8.2} M", e.lora_params as f64 / 1e6);
    println!("adapter + AdamW    {:>8.1} MB (f32 weights, grads, 2 moments)", e.lora_params as f64 * 16.0 / 1e6);
    println!("forward  GFLOP/tok {:>8.2}", e.forward_flops_per_token / 1e9);
    println!("train    GFLOP/tok {:>8.2} (≈3× forward, +1 forward for recomputation)", 4.0 * e.forward_flops_per_token / 1e9);
    Ok(())
}

/// Self-contained demo on a random-weight model: shows the loss falling and the
/// eval gate accepting the adapter on held-out examples.
fn cmd_toy(a: &Args) -> Result<(), String> {
    let tok = ByteTokenizer;
    let mut cfg = if a.get("arch") == Some("dense") {
        ModelConfig::tiny_dense(tok.vocab_size())
    } else {
        ModelConfig::tiny_hybrid(tok.vocab_size())
    };
    cfg.hidden_size = 64;
    cfg.intermediate_size = 128;
    let mut model = Model::random(cfg, 7, a.flag("q8"));
    let (train, held) = toy_data();
    let fmt = ChatFormat::qwen(&tok, "", 128)?;
    let mk = |v: &[(String, String)]| -> Vec<Example> {
        v.iter()
            .filter_map(|(p, r)| fmt.example(&tok, None, &Text::Str(p.clone()), &Text::Str(r.clone())))
            .collect()
    };
    let (tr, he) = (mk(&train), mk(&held));
    let tc = TrainConfig {
        rank: 8,
        alpha: 16.0,
        lr: 3e-3,
        epochs: a.num("epochs", 4)?,
        batch_size: 4,
        warmup_steps: 5,
        ..Default::default()
    };
    let mut trainer = Trainer::new(&mut model, tc, TrainData::Sft(tr));
    eprintln!("toy: {} trainable params, {} steps", trainer.model.trainable_params(), trainer.state.total_steps);
    let t0 = Instant::now();
    trainer.run(&mut RunToEnd, None, log_line).map_err(|e| e.to_string())?;
    eprintln!("trained in {:.1}s", t0.elapsed().as_secs_f32());
    let rep = evaluate_vs_base(trainer.model, &GateConfig::default(), &he, &[]);
    println!("{}", serde_json::to_string_pretty(&rep).unwrap());
    Ok(())
}

/// "Device command → structured action" pairs; held-out uses unseen devices/rooms.
pub fn toy_data() -> (Vec<(String, String)>, Vec<(String, String)>) {
    let verbs = [("turn on", "on"), ("switch off", "off"), ("dim", "dim")];
    let things = ["lamp", "fan", "heater", "radio", "kettle", "tv"];
    let rooms = ["kitchen", "hall", "office", "garage"];
    let mut train = Vec::new();
    let mut held = Vec::new();
    for (vi, (v, act)) in verbs.iter().enumerate() {
        for (ti, t) in things.iter().enumerate() {
            for (ri, r) in rooms.iter().enumerate() {
                let p = format!("{v} the {t} in the {r}");
                let ans = format!("{{\"act\":\"{act}\",\"dev\":\"{t}\",\"room\":\"{r}\"}}");
                if (vi + ti + ri) % 5 == 0 {
                    held.push((p, ans));
                } else {
                    train.push((p, ans));
                }
            }
        }
    }
    (train, held)
}
