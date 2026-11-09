//! Drop-in trainer for agentd (agenticOS).
//!
//! agentd's overnight scheduler (`crates/agentd/src/trainer.rs`) runs
//!
//! ```text
//! <bin> train -m MODEL.gguf --data learn/sft.jsonl --grammar learn/grammar.gbnf \
//!             --out learn/adapter.new.lora --steps N --max-seconds S
//! ```
//!
//! and reads the **last JSON line on stdout**: `{"done":false,"step":..}` means
//! "stopped with a checkpoint, run again to continue"; `{"done":true,"base":..,
//! "tuned":..,"kept":..}` means finished, and if `kept` the adapter at `--out` is
//! promoted and served with `agentos-llm --lora`. agentd kills the process when
//! the phone is unplugged, warm or needed, so a checkpoint is written after every
//! step (atomically, so a kill never leaves a torn file).
//!
//! This module reproduces that job exactly as `agentos-infer::finetune::run` does:
//! the same chat parsing and de-duplication, every `holdout`-th request held out,
//! answers in the grammar's key order, judging by 1 for the exact action and 0.5
//! for the right tool, the same defaults (60 steps, batch 4, rank 8, α 16,
//! lr 1e-3) and the same output lines. The adapter is written in agentos-llm's
//! `AGLORA1` format (attention q/k/v/o, as agentd serves it), and beside it as a
//! llama.cpp-style LoRA GGUF for the Rust Inference Engine.
//!
//! Prompts are built exactly as `agentos-llm` builds them (`Engine::chat_prompt`):
//! ChatML, plus the empty think block `<think>\n\n</think>\n\n` after
//! `<|im_start|>assistant\n` for Qwen3 / Qwen3.5 (non-thinking mode).
//!
//! Judging differs in one way: agentos-llm decodes greedily under the grammar.
//! Here greedy decoding is evaluated exactly from one teacher-forced pass (see
//! [`crate::eval::greedy_prefix`]) without the grammar, so a score here never
//! exceeds what grammar-constrained decoding would get.

use crate::eval::greedy_prefix;
use crate::linear::Lora;
use crate::model::{answer_targets, Model};
use crate::optim::{param_names, AdamW, AdamWConfig};
use crate::safetensors::{write_f32, SafeTensors};
use crate::tensor::Rng;
use crate::tokenizer::{fit, Example};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The projections agentos-llm's adapters cover.
pub const ATTENTION_TARGETS: &[&str] = &["q_proj", "k_proj", "v_proj", "o_proj"];

#[derive(Clone, Debug)]
pub struct AgentdJob {
    pub data: PathBuf,
    pub out: PathBuf,
    pub grammar: Option<String>,
    pub steps: usize,
    pub batch: usize,
    pub rank: usize,
    pub alpha: f32,
    pub lr: f32,
    /// Every nth request is held out for judging.
    pub holdout: usize,
    /// Stop (with a checkpoint) after this long; run again to continue.
    pub max_secs: Option<u64>,
    pub max_len: usize,
}

impl Default for AgentdJob {
    fn default() -> Self {
        AgentdJob {
            data: PathBuf::new(),
            out: PathBuf::new(),
            grammar: None,
            steps: 60,
            batch: 4,
            rank: 8,
            alpha: 16.0,
            lr: 1e-3,
            holdout: 5,
            max_secs: None,
            max_len: 2048,
        }
    }
}

/// One chat: (system, user, assistant).
pub type Chat = (String, String, String);

/// Chats from agentd's `sft.jsonl` (`{"messages":[system?, user, assistant]}`);
/// the same request confirmed twice counts once.
pub fn read_chats(text: &str) -> Vec<Chat> {
    let mut seen = std::collections::BTreeSet::new();
    text.lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter_map(|v| {
            let m = v["messages"].as_array()?;
            let get = |role: &str| m.iter().find(|x| x["role"] == role).and_then(|x| x["content"].as_str()).map(str::to_string);
            Some((get("system").unwrap_or_default(), get("user")?, get("assistant")?))
        })
        .filter(|c| seen.insert(c.1.trim().to_lowercase()))
        .collect()
}

/// 1 for the exact action, 0.5 for the right tool with other arguments.
pub fn score(got: &str, want: &str) -> f32 {
    let parse = |s: &str| serde_json::from_str::<Value>(s.trim()).ok();
    match (parse(got), parse(want)) {
        (Some(g), Some(w)) if g == w => 1.0,
        (Some(g), Some(w)) if g["tool"] == w["tool"] => 0.5,
        _ if got.trim() == want.trim() => 1.0,
        _ => 0.0,
    }
}

/// The answer as the grammar lets the model write it: `{"tool":..,"args":{..}}` with
/// `tool` first and the arguments in the order the grammar spells them for that tool
/// (the export's JSON has sorted keys). The order is read from the grammar's literal
/// `\"key\":` sequence following `\"tool\":\"<name>\"`; without a grammar, or if the
/// tool is not found, the export's argument order is kept.
pub fn canonical(answer: &str, grammar: Option<&str>) -> String {
    let Ok(v) = serde_json::from_str::<Value>(answer.trim()) else { return answer.to_string() };
    let (Some(tool), Some(args)) = (v["tool"].as_str(), v["args"].as_object()) else { return answer.to_string() };
    let mut keys: Vec<&String> = args.keys().collect();
    if let Some(g) = grammar {
        let anchor = format!("\\\"tool\\\":\\\"{tool}\\\"");
        if let Some(at) = g.find(&anchor) {
            // The tool's rule: from the anchor to the end of that line.
            let rest = &g[at..];
            let rule = rest.split('\n').next().unwrap_or(rest);
            keys.sort_by_key(|k| rule.find(&format!("\\\"{k}\\\":")).unwrap_or(usize::MAX));
        }
    }
    let parts: Vec<String> = keys.iter().map(|k| format!("{}:{}", json!(k), args[k.as_str()])).collect();
    format!("{{\"tool\":{},\"args\":{{{}}}}}", json!(tool), parts.join(","))
}

/// agentos-llm's chat prompt (`Engine::chat_prompt(msgs, thinking = false)`).
pub fn agentos_prompt(system: &str, user: &str, arch: &str) -> String {
    let mut p = String::new();
    if !system.is_empty() {
        p.push_str(&format!("<|im_start|>system\n{system}<|im_end|>\n"));
    }
    p.push_str(&format!("<|im_start|>user\n{user}<|im_end|>\n"));
    p.push_str("<|im_start|>assistant\n");
    if arch.starts_with("qwen3") {
        p.push_str("<think>\n\n</think>\n\n");
    }
    p
}

/// Tokenisation hooks: prompts with special tokens parsed, answers as plain text.
pub struct Encoders<'a> {
    pub prompt: &'a dyn Fn(&str) -> Vec<u32>,
    pub text: &'a dyn Fn(&str) -> Vec<u32>,
    pub im_end: u32,
}

fn example(enc: &Encoders, arch: &str, c: &Chat, max_len: usize) -> Option<Example> {
    let p = (enc.prompt)(&agentos_prompt(&c.0, &c.1, arch));
    let mut a = (enc.text)(&c.2);
    a.push(enc.im_end);
    fit(p, a, max_len)
}

/// (layer, tag) for an attention projection name (`blk.3.attn_q`, `model.layers.3.self_attn.q_proj`).
fn aglora_key(name: &str) -> Option<(u32, u8)> {
    let parts: Vec<&str> = name.split('.').collect();
    let layer = parts.iter().find_map(|p| p.parse::<u32>().ok())?;
    let tag = match crate::model::canonical_module(parts.last()?) {
        "q_proj" => b'q',
        "k_proj" => b'k',
        "v_proj" => b'v',
        "o_proj" => b'o',
        _ => return None,
    };
    Some((layer, tag))
}

/// Write the model's attention adapters in agentos-llm's `AGLORA1` format:
/// `"AGLORA1\0"`, rank u32, alpha f32, count u32, then per pair
/// layer u32, tag u8 (`q`/`k`/`v`/`o`), in u32, out u32, A `[r×in]` f32, B `[out×r]` f32,
/// pairs ordered by (layer, tag) as agentos-llm stores them. Atomic write.
pub fn save_aglora(model: &Model, path: impl AsRef<Path>) -> io::Result<()> {
    let mut pairs: BTreeMap<(u32, u8), (&Lora, usize, usize)> = BTreeMap::new();
    let (mut rank, mut alpha) = (0usize, 0.0f32);
    for l in model.lora_linears() {
        let lo = l.lora.as_ref().unwrap();
        let key = aglora_key(&l.name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("{}: AGLORA1 holds attention q/k/v/o only", l.name)))?;
        rank = lo.rank;
        alpha = lo.alpha;
        pairs.insert(key, (lo, l.in_dim(), l.out_dim()));
    }
    // agentos-llm orders its targets Q < K < V < O.
    let order = |t: u8| match t {
        b'q' => 0,
        b'k' => 1,
        b'v' => 2,
        _ => 3,
    };
    let mut items: Vec<_> = pairs.into_iter().collect();
    items.sort_by_key(|((layer, t), _)| (*layer, order(*t)));
    let mut out = Vec::new();
    out.extend(b"AGLORA1\0");
    out.extend((rank as u32).to_le_bytes());
    out.extend(alpha.to_le_bytes());
    out.extend((items.len() as u32).to_le_bytes());
    for ((layer, t), (lo, inp, outd)) in items {
        out.extend(layer.to_le_bytes());
        out.push(t);
        out.extend((inp as u32).to_le_bytes());
        out.extend((outd as u32).to_le_bytes());
        for v in lo.a.iter().chain(&lo.b) {
            out.extend(v.to_le_bytes());
        }
    }
    let path = path.as_ref();
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, out)?;
    std::fs::rename(tmp, path)
}

/// Load an `AGLORA1` adapter into the model (replacing its adapters). Returns the pair count.
pub fn load_aglora(model: &mut Model, path: impl AsRef<Path>) -> io::Result<usize> {
    let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
    let b = std::fs::read(path)?;
    if b.len() < 20 || &b[..8] != b"AGLORA1\0" {
        return Err(bad("not an agentos LoRA adapter"));
    }
    let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
    let rank = u32_at(8);
    let alpha = f32::from_le_bytes(b[12..16].try_into().unwrap());
    let n = u32_at(16);
    let mut at = 20;
    let mut pairs: BTreeMap<(u32, u8), crate::adapter::LoraModule> = BTreeMap::new();
    for _ in 0..n {
        if at + 13 > b.len() {
            return Err(bad("adapter file is cut short"));
        }
        let layer = u32_at(at) as u32;
        let t = b[at + 4];
        let (inp, outd) = (u32_at(at + 5), u32_at(at + 9));
        at += 13;
        let count = rank * inp + outd * rank;
        if at + count * 4 > b.len() {
            return Err(bad("adapter file is cut short"));
        }
        let vals: Vec<f32> = b[at..at + count * 4].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        at += count * 4;
        let (a, bb) = vals.split_at(rank * inp);
        pairs.insert((layer, t), (a.to_vec(), bb.to_vec(), rank, inp, outd));
    }
    let mut rng = Rng::new(0);
    let mut loaded = 0;
    for blk in &mut model.blocks {
        for l in blk.linears_mut() {
            l.lora = None;
            let Some(key) = aglora_key(&l.name) else { continue };
            let Some((a, bb, _, inp, outd)) = pairs.get(&key) else { continue };
            if (*inp, *outd) != (l.in_dim(), l.out_dim()) {
                return Err(bad(&format!("{}: adapter is {inp}x{outd}, model is {}x{}", l.name, l.in_dim(), l.out_dim())));
            }
            let mut lo = Lora::new(rank, alpha, *inp, *outd, &mut rng);
            lo.a = a.clone();
            lo.b = bb.clone();
            l.lora = Some(lo);
            l.lora_enabled = true;
            loaded += 1;
        }
    }
    if loaded != pairs.len() {
        return Err(bad(&format!("adapter has {} pairs, {loaded} matched the model", pairs.len())));
    }
    Ok(loaded)
}

/// Judge like agentos-llm: mean over held-out chats of 1 (exact action) / 0.5 (right tool).
pub fn judge(model: &Model, held: &[(Example, usize)]) -> f32 {
    if held.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    for (ex, tool_len) in held {
        let (ok, n) = greedy_prefix(model, ex);
        total += if ok == n {
            1.0
        } else if *tool_len > 0 && ok >= *tool_len {
            0.5
        } else {
            0.0
        };
    }
    total / held.len() as f32
}

fn data_hash(chats: &[Chat]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for c in chats {
        for b in c.0.bytes().chain([0]).chain(c.1.bytes()).chain([0]).chain(c.2.bytes()).chain([1]) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    format!("{h:016x}")
}

fn ckpt_paths(out: &Path) -> (PathBuf, PathBuf) {
    (out.with_extension("ckpt"), out.with_extension("ckpt.json"))
}

/// Run (or continue) agentd's training job on `model` (adapters are attached here).
/// Returns the summary agentd reads; `log` gets progress lines.
pub fn run(model: &mut Model, enc: &Encoders, job: &AgentdJob, log: &mut dyn FnMut(&str)) -> io::Result<Value> {
    let t0 = Instant::now();
    let text = std::fs::read_to_string(&job.data).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", job.data.display())))?;
    let mut chats = read_chats(&text);
    let n = job.holdout.max(2);
    for c in chats.iter_mut() {
        c.2 = canonical(&c.2, job.grammar.as_deref());
    }
    let (mut train, mut held) = (Vec::new(), Vec::new());
    for (i, c) in chats.iter().enumerate() {
        if i % n == n - 1 {
            held.push(c.clone())
        } else {
            train.push(c.clone())
        }
    }
    if train.is_empty() || held.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("need more confirmed requests to train and judge ({} to train, {} held out)", train.len(), held.len()),
        ));
    }
    let arch = model.cfg.arch.clone();
    let examples: Vec<Example> = train.iter().filter_map(|c| example(enc, &arch, c, job.max_len)).collect();
    let held_ex: Vec<(Example, usize)> = held
        .iter()
        .filter_map(|c| {
            let ex = example(enc, &arch, c, job.max_len)?;
            // Tokens through the tool name: `{"tool":"name` (stops before the quote,
            // which BPE may merge with what follows).
            let tool_len = serde_json::from_str::<Value>(&c.2)
                .ok()
                .and_then(|v| v["tool"].as_str().map(|t| (enc.text)(&format!("{{\"tool\":\"{t}")).len()))
                .unwrap_or(0);
            Some((ex, tool_len))
        })
        .collect();

    model.attach_lora(job.rank, job.alpha, ATTENTION_TARGETS, 0x5eed);
    let mut opt = AdamW::new(AdamWConfig { weight_decay: 0.0, ..Default::default() }, model);
    let names = param_names(model);
    let (ck, ck_meta) = ckpt_paths(&job.out);
    let hash = data_hash(&chats);
    let cfg_sig = format!("r{}a{}lr{}b{}s{}", job.rank, job.alpha, job.lr, job.batch, job.steps);

    // Continue from a checkpoint (same data and settings), or start by judging the base model.
    let resumed = (|| -> Result<(usize, f32), String> {
        let st = SafeTensors::open(&ck).map_err(|e| format!("no checkpoint ({e})"))?;
        let meta: Value =
            serde_json::from_str(st.metadata.get("agentd_job").ok_or("checkpoint has no job metadata")?).map_err(|e| e.to_string())?;
        if meta["data"] != hash {
            return Err("the exported requests changed".into());
        }
        if meta["cfg"] != cfg_sig.as_str() {
            return Err(format!("settings changed ({} → {cfg_sig})", meta["cfg"]));
        }
        let mut idx = 0;
        let mut loaded = Vec::new();
        for l in model.lora_linears() {
            let lo = l.lora.as_ref().unwrap();
            for len in [lo.a.len(), lo.b.len()] {
                let n = &names[idx];
                let rd = |k: &str| st.read_f32(k).map(|x| x.0).map_err(|e| format!("{k}: {e}"));
                let (w, m, v) = (rd(n)?, rd(&format!("{n}.adam_m"))?, rd(&format!("{n}.adam_v"))?);
                if w.len() != len {
                    return Err(format!("{n}: shape changed"));
                }
                loaded.push((w, m, v));
                idx += 1;
            }
        }
        let mut it = loaded.into_iter();
        let mut i = 0;
        for l in model.lora_linears_mut() {
            let lo = l.lora.as_mut().unwrap();
            let (a, ma, va) = it.next().ok_or("checkpoint is short")?;
            let (b, mb, vb) = it.next().ok_or("checkpoint is short")?;
            lo.a = a;
            lo.b = b;
            opt.m[i] = ma;
            opt.v[i] = va;
            opt.m[i + 1] = mb;
            opt.v[i + 1] = vb;
            i += 2;
        }
        opt.t = meta["opt_t"].as_u64().ok_or("no opt_t")?;
        Ok((meta["step"].as_u64().ok_or("no step")? as usize, meta["base"].as_f64().ok_or("no base")? as f32))
    })();
    let (mut step, base) = match resumed {
        Ok((step, base)) => {
            log(&format!("continuing from step {step} (base score {base:.2})"));
            (step, base)
        }
        Err(why) => {
            if ck.exists() {
                log(&format!("starting over: {why}"));
            }
            model.set_lora_enabled(false);
            let base = judge(model, &held_ex);
            model.set_lora_enabled(true);
            log(&format!("without an adapter: {:.0}% on {} held-out requests", base * 100.0, held_ex.len()));
            (0, base)
        }
    };

    let save = |model: &Model, opt: &AdamW, step: usize| -> io::Result<()> {
        let mut tensors: Vec<(String, Vec<usize>, &[f32])> = Vec::new();
        let mut i = 0;
        for l in model.lora_linears() {
            let lo = l.lora.as_ref().unwrap();
            for p in [&lo.a, &lo.b] {
                let n = &names[i];
                tensors.push((n.clone(), vec![p.len()], p.as_slice()));
                tensors.push((format!("{n}.adam_m"), vec![p.len()], opt.m[i].as_slice()));
                tensors.push((format!("{n}.adam_v"), vec![p.len()], opt.v[i].as_slice()));
                i += 1;
            }
        }
        let meta = json!({"step": step, "base": base, "data": hash, "cfg": cfg_sig, "opt_t": opt.t});
        let mut md = std::collections::BTreeMap::new();
        md.insert("agentd_job".to_string(), meta.to_string());
        write_f32(&ck, &tensors, &md)?;
        // Human-readable companion (agentos-llm's layout); the safetensors file is authoritative.
        crate::adapter::write_json(&ck_meta, &meta)
    };

    let mut last_loss = f32::NAN;
    let b = job.batch.max(1).min(examples.len());
    while step < job.steps {
        if job.max_secs.is_some_and(|s| t0.elapsed().as_secs() >= s) {
            save(model, &opt, step)?;
            return Ok(json!({"done": false, "step": step, "steps": job.steps, "loss": last_loss}));
        }
        model.zero_grad();
        let mut total = 0.0;
        for k in 0..b {
            let ex = &examples[(step * b + k) % examples.len()];
            let targets = answer_targets(&ex.tokens, ex.answer_start);
            let w = vec![1.0 / (targets.len().max(1) * b) as f32; targets.len()];
            let lp = model.forward_backward(&ex.tokens, &targets, &w);
            total += -lp.iter().sum::<f32>() / targets.len().max(1) as f32;
        }
        opt.step(model, job.lr);
        last_loss = total / b as f32;
        step += 1;
        log(&format!("step {step}/{} loss {last_loss:.4}", job.steps));
        // Every step: on a phone a step takes minutes and agentd may kill us at any time.
        save(model, &opt, step)?;
    }

    let tuned = judge(model, &held_ex);
    let kept = tuned > base;
    if kept {
        save_aglora(model, &job.out)?;
        #[cfg(feature = "rie")]
        if model.lora_linears().iter().all(|l| l.name.starts_with("blk.")) {
            crate::rie::write_gguf_adapter(model, job.out.with_extension("gguf"))?;
        }
    }
    let _ = std::fs::remove_file(&ck);
    let _ = std::fs::remove_file(&ck_meta);
    log(&format!(
        "with the adapter: {:.0}% (was {:.0}%): {}",
        tuned * 100.0,
        base * 100.0,
        if kept { "kept" } else { "not better, discarded" }
    ));
    Ok(json!({
        "done": true, "base": base, "tuned": tuned, "kept": kept, "step": step,
        "loss": last_loss, "train": train.len(), "held_out": held.len(),
        "adapter": kept.then(|| job.out.display().to_string()),
        "secs": t0.elapsed().as_secs()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_agentos_conventions() {
        // Same cases as agentos-infer's finetune test.
        let g = r#"root ::= "{\"tool\":\"send_sms\",\"args\":{\"to\":" s ",\"text\":" s "}}"
s ::= "\"" [a-zA-Z ]* "\"""#;
        let a = r#"{"args":{"text":"hi there","to":"Ravi"},"tool":"send_sms"}"#;
        assert_eq!(canonical(a, Some(g)), r#"{"tool":"send_sms","args":{"to":"Ravi","text":"hi there"}}"#);
        assert_eq!(score(&canonical(a, Some(g)), a), 1.0);
        assert_eq!(score(r#"{"tool":"send_sms","args":{}}"#, a), 0.5);
        let chats = read_chats(&format!(
            "{}\n{}\n",
            json!({"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"x"}]}),
            json!({"messages":[{"role":"user","content":"hi "},{"role":"assistant","content":"y"}]})
        ));
        assert_eq!(chats.len(), 1);
        assert_eq!(
            agentos_prompt("sys", "hi", "qwen3"),
            "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert_eq!(agentos_prompt("", "hi", "qwen2"), "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
        assert_eq!(aglora_key("blk.12.attn_output"), Some((12, b'o')));
        assert_eq!(aglora_key("model.layers.3.self_attn.k_proj"), Some((3, b'k')));
        assert_eq!(aglora_key("blk.1.ffn_up"), None);
    }
}
