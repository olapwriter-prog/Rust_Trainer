//! `rust_trainer` command line.
//!
//! ```text
//! rust_trainer toy      [--arch hybrid|dense] [--steps N]
//! rust_trainer info     --model DIR [--rank R]
//! rust_trainer train    --model DIR --data FILE --out DIR [--ckpt FILE] [options]
//! rust_trainer train    -m M.gguf --data sft.jsonl --grammar G --out adapter.new.lora --steps N [--max-seconds S]
//!                       (agentd's job protocol; drop-in for `agentos-llm train`)
//! rust_trainer eval     --model DIR --adapter DIR --data FILE [options]
//! rust_trainer run      --model DIR --data FILE --status FILE [--hours H] [--poll-secs S] [options]
//! rust_trainer rollback --model DIR [--reason TEXT]
//! rust_trainer status   --model DIR
//! rust_trainer route    --model M.gguf --prompt TEXT [--adapter BUNDLE]   (mixture: which expert)
//! rust_trainer act-eval --model M.gguf --data FILE [--adapter DIR]        (exact action accuracy)
//! rust_trainer synth    --out FILE.gguf [--arch qwen3 | --preset qwen3-0.6b] [--wtype Q4_0]
//! ```
//! Options: `--rank 8 --alpha 16 --lr 2e-4 --epochs 2 --batch 4 --max-len 512
//! --heldout 0.1 --dpo BETA --q8 --tokenizer FILE --assistant-prefix TEXT
//! --min-improvement 0.02 --min-examples 8 --min-train 50 --min-battery 50 --max-temp 40`.

use rust_trainer::adapter::{base_fingerprint, load_adapter_into, save_adapter, AdapterMeta};
use rust_trainer::data::{Dataset, Text};
use rust_trainer::eval::{evaluate_vs_base, GateConfig};
use rust_trainer::hf::{estimate, load_model, LoadOptions};
use rust_trainer::model::{Model, ModelConfig, DEFAULT_TARGETS};
use rust_trainer::moe::{is_bundle, load_router, measure_current, Current, MoeConfig};
use rust_trainer::registry::{now_unix, Registry};
use rust_trainer::scheduler::{run_overnight, run_overnight_moe, FileMonitor, Heldout, JobConfig, PowerPolicy};
use rust_trainer::tokenizer::{ByteTokenizer, ChatFormat, ChatMl, Example, Formatter, Tokenizer};
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
            // agentd / agentos-llm style short options
            let a: &str = match a.as_str() {
                "-m" => "--model",
                "-t" => "--threads",
                other => other,
            };
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
    "usage: rust_trainer <toy|info|train|eval|act-eval|run|route|rollback|status|synth> [--options]  (see README.md)".to_string()
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
        "route" => cmd_route(&args),
        "act-eval" => cmd_act_eval(&args),
        #[cfg(feature = "rie")]
        "synth" => cmd_synth(&args),
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

/// A loaded base model, plus the engine's tokenizer and chat template for GGUF models.
struct Loaded {
    model: Model,
    #[cfg(feature = "rie")]
    engine: Option<(std::sync::Arc<rust_trainer::rie::rie_core::Tokenizer>, rust_trainer::rie::rie_core::chat::ChatTemplate)>,
}

fn is_gguf(path: &str) -> bool {
    path.to_ascii_lowercase().ends_with(".gguf")
}

fn load(a: &Args) -> Result<Loaded, String> {
    let path = a.req("model")?;
    if is_gguf(path) {
        #[cfg(feature = "rie")]
        {
            let opts =
                rust_trainer::rie::GgufOptions { n_threads: a.get("threads").and_then(|t| t.parse().ok()), pin_threads: !a.flag("no-pin") };
            let gm = rust_trainer::rie::load_gguf(path, &opts).map_err(|e| format!("loading {path}: {e}"))?;
            let template = match a.get("template") {
                Some(name) => {
                    rust_trainer::rie::rie_core::chat::ChatTemplate::from_name(name).ok_or_else(|| format!("unknown --template {name}"))?
                }
                None => gm.template,
            };
            eprintln!("model: {} via rie-core ({:?} template)", gm.arch_name, template);
            return Ok(Loaded { model: gm.model, engine: Some((gm.tokenizer, template)) });
        }
        #[cfg(not(feature = "rie"))]
        return Err("GGUF models need the `rie` feature".into());
    }
    let mut m = load_model(path, &LoadOptions { quantize_q8: a.flag("q8") }).map_err(|e| format!("loading {path}: {e}"))?;
    if let Some(c) = a.get("bptt-chunk") {
        let c: usize = c.parse().map_err(|_| "--bptt-chunk")?;
        for b in &mut m.blocks {
            if let rust_trainer::model::Mixer::Delta(d) = &mut b.mixer {
                d.cfg.bptt_chunk = c;
            }
        }
    }
    Ok(Loaded {
        model: m,
        #[cfg(feature = "rie")]
        engine: None,
    })
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
            eprintln!("note: {} found but built without the `hf-tokenizers` feature; only pre-tokenised data can be used", p.display());
            Ok(None)
        }
    }
}

fn needs_tokenizer(t: &Text) -> bool {
    !matches!(t, Text::Ids(_))
}

fn any_raw_text(ds: &Dataset) -> bool {
    ds.sft.iter().any(|r| needs_tokenizer(&r.prompt) || needs_tokenizer(&r.response))
        || ds.pref.iter().any(|r| needs_tokenizer(&r.prompt) || needs_tokenizer(&r.chosen) || needs_tokenizer(&r.rejected))
}

/// Tokenised examples with their skill labels (for the mixture of experts).
struct Built {
    sft: Vec<Example>,
    sft_labels: Vec<Option<String>>,
    pref: Vec<PrefPair>,
    pref_labels: Vec<Option<String>>,
}

fn build_examples(ds: &Dataset, fmt: &dyn Formatter) -> Built {
    let (sft, sft_labels) =
        ds.sft.iter().filter_map(|r| Some((fmt.example(r.system.as_deref(), &r.prompt, &r.response)?, r.label.clone()))).unzip();
    let (pref, pref_labels) = ds
        .pref
        .iter()
        .filter_map(|r| {
            let pair = PrefPair {
                chosen: fmt.example(r.system.as_deref(), &r.prompt, &r.chosen)?,
                rejected: fmt.example(r.system.as_deref(), &r.prompt, &r.rejected)?,
            };
            Some((pair, r.label.clone()))
        })
        .unzip();
    Built { sft, sft_labels, pref, pref_labels }
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
    train_labels: Vec<Option<String>>,
    held_sft: Vec<Example>,
    held_pref: Vec<PrefPair>,
    n_train: usize,
}

fn prepare(a: &Args, loaded: &Loaded, cfg: &TrainConfig) -> Result<Prepared, String> {
    let model = &loaded.model;
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
    let max_len = a.num("max-len", 512)?;
    let any_text = any_raw_text(&ds);

    // GGUF models: format exactly as the engine's chat() does.
    #[cfg(feature = "rie")]
    let engine_fmt = match &loaded.engine {
        Some((tok, template)) => Some(rust_trainer::rie::EngineChat::new(tok.clone(), *template, max_len).map_err(|e| e.to_string())?),
        None => None,
    };
    #[cfg(not(feature = "rie"))]
    let engine_fmt: Option<ChatMl<'static>> = None;

    let tok = tokenizer(a)?;
    let byte = ByteTokenizer;
    let prefix = a.get("assistant-prefix").map(|s| s.replace("\\n", "\n")).unwrap_or_else(|| default_prefix(model).into());
    let chatml;
    let fmt: &dyn Formatter = match &engine_fmt {
        Some(f) => f,
        None => {
            let t: &dyn Tokenizer = match tok.as_deref() {
                Some(t) => t,
                None if any_text => {
                    return Err("data contains raw text: pass --tokenizer tokenizer.json (build with --features hf-tokenizers), use a GGUF model, or export *_ids".into())
                }
                None => &byte, // pre-tokenised data only needs the ChatML special ids
            };
            let f =
                ChatFormat::qwen(t, &prefix, max_len)
                    .or_else(|e| if any_text { Err(e) } else { ChatFormat::qwen(&byte, &prefix, max_len) })?;
            chatml = ChatMl { fmt: f, tok: t };
            &chatml
        }
    };
    let tr = build_examples(&train, fmt);
    let he = build_examples(&held, fmt);
    let (tr_sft, tr_pref, held_sft, held_pref) = (tr.sft, tr.pref, he.sft, he.pref);
    for ex in tr_sft.iter().chain(&held_sft) {
        if let Some(&t) = ex.tokens.iter().find(|&&t| t as usize >= model.cfg.vocab_size) {
            return Err(format!("token id {t} exceeds the model vocab {}", model.cfg.vocab_size));
        }
    }
    let (train, train_labels) = match cfg.objective {
        Objective::Sft => (TrainData::Sft(tr_sft), tr.sft_labels),
        Objective::Dpo { .. } => (TrainData::Pref(tr_pref), tr.pref_labels),
    };
    let n_train = train.len();
    Ok(Prepared { train, train_labels, held_sft, held_pref, n_train })
}

/// Load an adapter directory: PEFT safetensors, or only `adapter.gguf` (e.g. from llama.cpp).
fn load_adapter_dir(model: &mut Model, dir: &str) -> Result<(), String> {
    let d = Path::new(dir);
    if d.join(rust_trainer::adapter::WEIGHTS_FILE).exists() && d.join(rust_trainer::adapter::META_FILE).exists() {
        return load_adapter_into(model, d).map(|_| ()).map_err(|e| e.to_string());
    }
    #[cfg(feature = "rie")]
    {
        let g = if d.is_file() { d.to_path_buf() } else { d.join(rust_trainer::rie::GGUF_ADAPTER_FILE) };
        if g.exists() {
            return rust_trainer::rie::read_gguf_adapter(model, &g).map(|_| ()).map_err(|e| e.to_string());
        }
    }
    Err(format!("{dir}: no adapter found"))
}

fn log_line(s: &str) {
    eprintln!("{s}");
}

/// agentd's training job (`train -m M --data sft.jsonl --grammar G --out adapter.new.lora
/// --steps N --max-seconds S`): progress on stderr, one JSON summary line on stdout.
#[cfg(feature = "rie")]
fn cmd_train_agentd(a: &Args) -> Result<(), String> {
    use rust_trainer::agentd::{run, AgentdJob, Encoders};
    let loaded = load(a)?;
    let Some((tok, _)) = &loaded.engine else { return Err("agentd jobs need a GGUF model".into()) };
    let grammar = match a.get("grammar") {
        Some(p) => Some(std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?),
        None => None,
    };
    let d = AgentdJob::default();
    let job = AgentdJob {
        data: PathBuf::from(a.req("data")?),
        out: PathBuf::from(a.req("out")?),
        grammar,
        steps: a.num("steps", d.steps)?,
        batch: a.num("batch", d.batch)?,
        rank: a.num("rank", d.rank)?,
        alpha: a.num("alpha", d.alpha)?,
        lr: a.num("lr", d.lr)?,
        holdout: a.num("holdout", d.holdout)?,
        max_secs: a.get("max-seconds").map(|s| s.parse().map_err(|_| "--max-seconds")).transpose()?,
        max_len: a.num("max-len", d.max_len)?,
    };
    let tk = tok.clone();
    let tk2 = tok.clone();
    let prompt = move |s: &str| tk.encode(s, true, true);
    let text = move |s: &str| tk2.encode(s, false, false);
    let im_end = tok.token_id("<|im_end|>").ok_or("the model's vocabulary has no <|im_end|>")?;
    let enc = Encoders { prompt: &prompt, text: &text, im_end };
    let mut model = loaded.model;
    let out = run(&mut model, &enc, &job, &mut |line| eprintln!("rust_trainer train: {line}")).map_err(|e| e.to_string())?;
    println!("{out}");
    Ok(())
}

#[cfg(not(feature = "rie"))]
fn cmd_train_agentd(_a: &Args) -> Result<(), String> {
    Err("agentd jobs need the `rie` feature".into())
}

fn cmd_train(a: &Args) -> Result<(), String> {
    // agentd launches `train -m MODEL --data … --grammar … --out … --steps N --max-seconds S`.
    if a.get("steps").is_some() || a.get("grammar").is_some() {
        return cmd_train_agentd(a);
    }
    let loaded = load(a)?;
    let cfg = train_config(a)?;
    let p = prepare(a, &loaded, &cfg)?;
    let mut model = loaded.model;
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
    let loaded = load(a)?;
    let cfg = TrainConfig::default();
    let p = prepare(a, &loaded, &cfg)?;
    let mut model = loaded.model;
    let dir = a.req("adapter")?;
    let rep = if is_bundle(dir) {
        let cur = Current::from_dir(Some(PathBuf::from(dir))).map_err(|e| e.to_string())?;
        let base = measure_current(&mut model, &Current::Base, &p.held_sft, &p.held_pref).map_err(|e| e.to_string())?;
        let cand = measure_current(&mut model, &cur, &p.held_sft, &p.held_pref).map_err(|e| e.to_string())?;
        rust_trainer::eval::gate(&gate_config(a)?, &base.metrics(), &cand.metrics(), p.held_sft.len() + p.held_pref.len(), cand.tokens)
    } else {
        load_adapter_dir(&mut model, dir)?;
        evaluate_vs_base(&mut model, &gate_config(a)?, &p.held_sft, &p.held_pref)
    };
    println!("{}", serde_json::to_string_pretty(&rep).unwrap());
    Ok(())
}

fn cmd_run(a: &Args) -> Result<(), String> {
    let loaded = load(a)?;
    let train = train_config(a)?;
    let p = prepare(a, &loaded, &train)?;
    let mut model = loaded.model;
    let registry = Registry::open(a.req("model")?).map_err(|e| e.to_string())?;
    let job = JobConfig {
        base_model: a.req("model")?.to_string(),
        train,
        gate: gate_config(a)?,
        power: PowerPolicy { min_battery_pct: a.num("min-battery", 50.0)?, max_temp_c: a.num("max-temp", 40.0)?, ..Default::default() },
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
    let heldout = Heldout { sft: &p.held_sft, pref: &p.held_pref };
    let experts: usize = a.num("experts", 1)?;
    let outcome = if experts > 1 {
        let moe = MoeConfig {
            max_experts: experts,
            min_expert_examples: a.num("min-expert-examples", 24)?,
            general_expert: !a.flag("no-general"),
            min_similarity: a.num("min-similarity", 0.1)?,
            ..Default::default()
        };
        run_overnight_moe(&mut model, &registry, &job, &moe, p.train, &p.train_labels, heldout, &mut monitor, &mut wait, &mut log)
    } else {
        run_overnight(&mut model, &registry, &job, p.train, heldout, &mut monitor, &mut wait, &mut log)
    }
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
    if let Some(dir) = r.active_dir().filter(|d| is_bundle(d)) {
        let router = load_router(&dir).map_err(|e| e.to_string())?;
        println!("mixture:  fallback = {}", router.fallback.as_deref().unwrap_or("(base model)"));
        for (g, e) in router.names.iter().zip(&router.experts) {
            println!("          {g:<16} → {}", e.as_deref().unwrap_or("fallback"));
        }
    }
    if let Ok(s) = std::fs::read_to_string(r.work_dir().join("status.json")) {
        println!("run:      {}", s.trim());
    }
    for e in r.read_ledger().iter().rev().take(10) {
        println!("  {} {:<9} {:<14} {}", e.ts, e.event, e.adapter.as_deref().unwrap_or("-"), e.detail);
    }
    Ok(())
}

/// Held-out *action* accuracy: does greedy decoding produce exactly the confirmed
/// answer, and (for JSON actions with an "intent") the right intent? Computed
/// exactly from one teacher-forced pass per example (see `eval::greedy_prefix`).
/// Compares the base model with `--adapter` (single adapter or mixture).
#[cfg(feature = "rie")]
fn cmd_act_eval(a: &Args) -> Result<(), String> {
    use rust_trainer::eval::greedy_prefix;
    let loaded = load(a)?;
    let p = prepare(a, &loaded, &TrainConfig::default())?;
    let Some((tok, _)) = &loaded.engine else { return Err("act-eval needs a GGUF model".into()) };
    let mut model = loaded.model;
    // Tokens up to and including the intent name, per example (if the answer is a JSON action).
    let intent_len = |ex: &Example| -> Option<usize> {
        let ans = tok.decode(&ex.tokens[ex.answer_start..], false);
        let v: serde_json::Value = serde_json::from_str(ans.trim()).ok()?;
        let intent = v.get("intent")?.as_str()?;
        Some(tok.encode(&format!("{{\"intent\":\"{intent}"), false, false).len())
    };
    let ilens: Vec<Option<usize>> = p.held_sft.iter().map(intent_len).collect();
    let score = |model: &Model, examples: &[(usize, &Example)]| -> (usize, usize, usize) {
        let (mut exact, mut intent, mut with_intent) = (0, 0, 0);
        for (i, ex) in examples {
            let (ok, n) = greedy_prefix(model, ex);
            exact += (ok == n) as usize;
            if let Some(l) = ilens[*i] {
                with_intent += 1;
                intent += (ok >= l.min(n)) as usize;
            }
        }
        (exact, intent, with_intent)
    };
    let all: Vec<(usize, &Example)> = p.held_sft.iter().enumerate().collect();
    let n = all.len();
    let report = |name: &str, (e, i, w): (usize, usize, usize)| {
        println!(
            "{name:<10} exact action {e:>4}/{n} = {:5.1}%   intent {i:>4}/{w} = {:5.1}%",
            100.0 * e as f32 / n.max(1) as f32,
            100.0 * i as f32 / w.max(1) as f32
        );
    };
    model.set_lora_enabled(false);
    report("base", score(&model, &all));
    if let Some(dir) = a.get("adapter") {
        if is_bundle(dir) {
            let router = load_router(dir).map_err(|e| e.to_string())?;
            let mut by: std::collections::BTreeMap<Option<String>, Vec<(usize, &Example)>> = Default::default();
            for (i, ex) in &all {
                by.entry(router.route(rust_trainer::moe::prompt_of(ex)).expert).or_default().push((*i, *ex));
            }
            let mut tot = (0, 0, 0);
            for (expert, items) in by {
                rust_trainer::moe::select_expert(&mut model, Path::new(dir), expert.as_deref()).map_err(|e| e.to_string())?;
                let (e, i, w) = score(&model, &items);
                println!("  expert {:<10} {} requests: exact {e}, intent {i}/{w}", expert.as_deref().unwrap_or("(base)"), items.len());
                tot = (tot.0 + e, tot.1 + i, tot.2 + w);
            }
            report("mixture", tot);
        } else {
            load_adapter_dir(&mut model, dir)?;
            report("adapter", score(&model, &all));
        }
    }
    Ok(())
}

#[cfg(not(feature = "rie"))]
fn cmd_act_eval(_a: &Args) -> Result<(), String> {
    Err("act-eval needs a GGUF model (build with the `rie` feature)".into())
}

/// Show which expert of a mixture a prompt is routed to (GGUF models).
#[cfg(feature = "rie")]
fn cmd_route(a: &Args) -> Result<(), String> {
    let loaded = load(a)?;
    let dir = match a.get("adapter") {
        Some(d) => PathBuf::from(d),
        None => Registry::open(a.req("model")?).map_err(|e| e.to_string())?.active_dir().ok_or("no active adapter")?,
    };
    if !is_bundle(&dir) {
        return Err(format!("{} is a single adapter, not a mixture", dir.display()));
    }
    let router = load_router(&dir).map_err(|e| e.to_string())?;
    let Some((tok, template)) = &loaded.engine else { return Err("route needs a GGUF model".into()) };
    let fmt = rust_trainer::rie::EngineChat::new(tok.clone(), *template, 1 << 20).map_err(|e| e.to_string())?;
    let tokens = fmt.prompt_tokens(a.get("system"), &Text::Str(a.req("prompt")?.to_string()));
    let r = router.route(&tokens);
    let group = r.group.map(|g| router.names[g].clone()).unwrap_or_else(|| "(unsure)".into());
    let expert = r.expert.clone().unwrap_or_else(|| "(base model)".into());
    println!("{}", serde_json::json!({"group": group, "similarity": r.similarity, "expert": expert}));
    Ok(())
}

#[cfg(not(feature = "rie"))]
fn cmd_route(_a: &Args) -> Result<(), String> {
    Err("route needs a GGUF model (build with the `rie` feature)".into())
}

fn cmd_info(a: &Args) -> Result<(), String> {
    let path = a.req("model")?;
    let (cfg, stored) = if is_gguf(path) {
        let m = load(a)?.model;
        let b = m.frozen_bytes();
        (m.cfg, Some(b))
    } else {
        let root: serde_json::Value =
            serde_json::from_slice(&std::fs::read(Path::new(path).join("config.json")).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        (rust_trainer::hf::parse_config(&root).map_err(|e| e.to_string())?, None)
    };
    let rank = a.num("rank", 8)?;
    let e = estimate(&cfg, rank, DEFAULT_TARGETS);
    println!("arch {} | layers {} | hidden {}", cfg.arch, cfg.n_layers(), cfg.hidden_size);
    println!("base params        {:>8.2} M", e.base_params as f64 / 1e6);
    if let Some(b) = stored {
        println!("frozen weights     {:>8.1} MB as stored (memory-mapped, not copied)", b as f64 / 1e6);
    }
    println!("LoRA params (r={rank}) {:>8.2} M", e.lora_params as f64 / 1e6);
    println!("adapter + AdamW    {:>8.1} MB (f32 weights, grads, 2 moments)", e.lora_params as f64 * 16.0 / 1e6);
    println!("forward  GFLOP/tok {:>8.2}", e.forward_flops_per_token / 1e9);
    println!("train    GFLOP/tok {:>8.2} (≈3× forward, +1 forward for recomputation)", 4.0 * e.forward_flops_per_token / 1e9);
    Ok(())
}

/// Write a random-weight GGUF with the engine's synthetic-model builder
/// (`--arch qwen3 --wtype Q4_0 --preset qwen3-0.6b --out FILE`), for VM runs without downloads.
#[cfg(feature = "rie")]
fn cmd_synth(a: &Args) -> Result<(), String> {
    use rust_trainer::rie::rie_core::{quant::GgmlType, testing::SyntheticSpec};
    let wtype = match a.get("wtype").unwrap_or("Q4_0") {
        "F32" => GgmlType::F32,
        "F16" => GgmlType::F16,
        "Q8_0" => GgmlType::Q8_0,
        "Q4_0" => GgmlType::Q4_0,
        "Q4_K" => GgmlType::Q4_K,
        "Q6_K" => GgmlType::Q6_K,
        w => return Err(format!("unsupported --wtype {w}")),
    };
    let mut spec = match a.get("preset") {
        Some(p) => SyntheticSpec::preset(p, wtype).ok_or_else(|| format!("unknown preset {p}; one of {:?}", SyntheticSpec::PRESETS))?,
        None => SyntheticSpec::tiny(a.get("arch").unwrap_or("qwen3"), wtype),
    };
    spec.seed = a.num("seed", 1)?;
    // Shape overrides, e.g. Qwen3-8B: --preset qwen3-4b --layers 36 --hidden 4096 --ff 12288
    spec.n_layer = a.num("layers", spec.n_layer)?;
    spec.n_embd = a.num("hidden", spec.n_embd)?;
    spec.n_head = a.num("heads", spec.n_head)?;
    spec.n_head_kv = a.num("kv-heads", spec.n_head_kv)?;
    spec.head_dim = a.num("head-dim", spec.head_dim)?;
    spec.rope_dim = a.num("head-dim", spec.rope_dim)?;
    spec.n_ff = a.num("ff", spec.n_ff)?;
    spec.n_vocab = a.num("vocab", spec.n_vocab)?;
    let out = a.req("out")?;
    std::fs::write(out, spec.build()).map_err(|e| e.to_string())?;
    eprintln!("wrote {out}: {} {:?}, {} layers, hidden {}", spec.arch, spec.wtype, spec.n_layer, spec.n_embd);
    Ok(())
}

/// Self-contained demo on a random-weight model: shows the loss falling and the
/// eval gate accepting the adapter on held-out examples.
fn cmd_toy(a: &Args) -> Result<(), String> {
    let tok = ByteTokenizer;
    let mut cfg =
        if a.get("arch") == Some("dense") { ModelConfig::tiny_dense(tok.vocab_size()) } else { ModelConfig::tiny_hybrid(tok.vocab_size()) };
    cfg.hidden_size = 64;
    cfg.intermediate_size = 128;
    let mut model = Model::random(cfg, 7, a.flag("q8"));
    let (train, held) = toy_data();
    let fmt = ChatFormat::qwen(&tok, "", 128)?;
    let mk = |v: &[(String, String)]| -> Vec<Example> {
        v.iter().filter_map(|(p, r)| fmt.example(&tok, None, &Text::Str(p.clone()), &Text::Str(r.clone()))).collect()
    };
    let (tr, he) = (mk(&train), mk(&held));
    let tc =
        TrainConfig { rank: 8, alpha: 16.0, lr: 3e-3, epochs: a.num("epochs", 4)?, batch_size: 4, warmup_steps: 5, ..Default::default() };
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
type Pairs = Vec<(String, String)>;

pub fn toy_data() -> (Pairs, Pairs) {
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
