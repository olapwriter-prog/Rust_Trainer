//! Training sets exported by `agentd ctl learn`.
//!
//! Input is JSON Lines. Each record is either
//! * **supervised**: a confirmed request with its right answer
//!   `{"id", "prompt" | "messages", "response", "confirmed": true, ...}`, or
//! * **preference**: a correction as "this, not that"
//!   `{"id", "prompt", "chosen", "rejected", "confirmed": true, ...}`.
//!
//! Field aliases are accepted (`answer`/`completion`/`corrected` for the
//! response, `input`/`request` for the prompt, `original`/`model_output` for the
//! rejected side), and either side may be given pre-tokenised as `*_ids`.
//!
//! **Only confirmed data is used.** A record is kept only if it carries an
//! explicit confirmation (`"confirmed": true`, or `"status"` of
//! `confirmed`/`approved`/`corrected`). Anything else, in particular the
//! model's own unconfirmed output, is dropped and counted.

use serde_json::Value;
use std::io::{self, BufRead};
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub enum Text {
    Str(String),
    Ids(Vec<u32>),
    /// Multi-turn prompt as `(role, content)` pairs (from `messages`).
    Turns(Vec<(String, String)>),
}

#[derive(Clone, Debug)]
pub struct SftRecord {
    pub id: String,
    pub system: Option<String>,
    pub prompt: Text,
    pub response: Text,
    pub split: Option<String>,
}

#[derive(Clone, Debug)]
pub struct PrefRecord {
    pub id: String,
    pub system: Option<String>,
    pub prompt: Text,
    pub chosen: Text,
    pub rejected: Text,
    pub split: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct Dataset {
    pub sft: Vec<SftRecord>,
    pub pref: Vec<PrefRecord>,
    /// Records dropped because they were not explicitly confirmed.
    pub dropped_unconfirmed: usize,
    /// Records dropped because they could not be parsed.
    pub dropped_malformed: usize,
}

fn is_confirmed(v: &Value) -> bool {
    if v.get("confirmed").and_then(Value::as_bool) == Some(true) {
        return true;
    }
    matches!(
        v.get("status").and_then(Value::as_str),
        Some("confirmed") | Some("approved") | Some("corrected")
    )
}

fn text_field(v: &Value, names: &[&str]) -> Option<Text> {
    for n in names {
        if let Some(ids) = v.get(format!("{n}_ids")).and_then(Value::as_array) {
            let ids: Option<Vec<u32>> = ids.iter().map(|x| x.as_u64().map(|u| u as u32)).collect();
            if let Some(ids) = ids {
                return Some(Text::Ids(ids));
            }
        }
        if let Some(s) = v.get(*n).and_then(Value::as_str) {
            return Some(Text::Str(s.to_string()));
        }
    }
    None
}

/// `messages: [{role, content}]` → (system, prompt rendered as turns, final assistant answer).
fn from_messages(v: &Value) -> Option<(Option<String>, Text, Text)> {
    let msgs = v.get("messages")?.as_array()?;
    let (last, rest) = msgs.split_last()?;
    if last.get("role")?.as_str()? != "assistant" {
        return None;
    }
    let answer = last.get("content")?.as_str()?.to_string();
    let mut system = None;
    let mut turns = Vec::new();
    for m in rest {
        let role = m.get("role")?.as_str()?;
        let content = m.get("content")?.as_str()?;
        if role == "system" && system.is_none() && turns.is_empty() {
            system = Some(content.to_string());
        } else {
            turns.push((role.to_string(), content.to_string()));
        }
    }
    let prompt = if turns.len() == 1 && turns[0].0 == "user" {
        Text::Str(turns[0].1.clone())
    } else {
        Text::Turns(turns)
    };
    Some((system, prompt, Text::Str(answer)))
}

impl Dataset {
    pub fn load_jsonl(path: impl AsRef<Path>) -> io::Result<Self> {
        let f = std::fs::File::open(path)?;
        let mut ds = Dataset::default();
        for (ln, line) in io::BufReader::new(f).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<Value>(&line) {
                Ok(v) => ds.push_value(&v, ln),
                Err(_) => ds.dropped_malformed += 1,
            }
        }
        Ok(ds)
    }

    pub fn push_value(&mut self, v: &Value, ln: usize) {
        if !is_confirmed(v) {
            self.dropped_unconfirmed += 1;
            return;
        }
        let id = v
            .get("id")
            .map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string()))
            .unwrap_or_else(|| format!("line{ln}"));
        let split = v.get("split").and_then(Value::as_str).map(str::to_string);
        let mut system = v.get("system").and_then(Value::as_str).map(str::to_string);
        let prompt = text_field(v, &["prompt", "input", "request"]);

        let chosen = text_field(v, &["chosen", "corrected"]);
        let rejected = text_field(v, &["rejected", "original", "model_output"]);
        if let (Some(p), Some(c), Some(r)) = (prompt.clone(), chosen.clone(), rejected) {
            self.pref.push(PrefRecord { id, system, prompt: p, chosen: c, rejected: r, split });
            return;
        }
        let response = text_field(v, &["response", "answer", "completion", "corrected", "chosen"]);
        match (prompt, response) {
            (Some(p), Some(r)) => self.sft.push(SftRecord { id, system, prompt: p, response: r, split }),
            _ => match from_messages(v) {
                Some((s, p, r)) => {
                    if system.is_none() {
                        system = s;
                    }
                    self.sft.push(SftRecord { id, system, prompt: p, response: r, split })
                }
                None => self.dropped_malformed += 1,
            },
        }
    }

    /// Deterministic train / held-out split. An explicit `"split"` field
    /// (`"heldout"`, `"eval"`, `"test"`) wins; otherwise ids are hashed so the
    /// same example always lands on the same side across nights.
    pub fn split(&self, heldout_frac: f32) -> (Dataset, Dataset) {
        let is_held = |id: &str, split: &Option<String>| -> bool {
            match split.as_deref() {
                Some("heldout") | Some("eval") | Some("test") | Some("validation") => true,
                Some("train") => false,
                _ => (fnv(id) % 10_000) as f32 / 10_000.0 < heldout_frac,
            }
        };
        let mut train = Dataset::default();
        let mut held = Dataset::default();
        for r in &self.sft {
            if is_held(&r.id, &r.split) { held.sft.push(r.clone()) } else { train.sft.push(r.clone()) }
        }
        for r in &self.pref {
            if is_held(&r.id, &r.split) { held.pref.push(r.clone()) } else { train.pref.push(r.clone()) }
        }
        (train, held)
    }

    pub fn len(&self) -> usize {
        self.sft.len() + self.pref.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn only_confirmed_records_are_kept() {
        let mut ds = Dataset::default();
        ds.push_value(&json!({"id":"a","prompt":"p","response":"r","confirmed":true}), 0);
        ds.push_value(&json!({"id":"b","prompt":"p","response":"r"}), 1);
        ds.push_value(&json!({"id":"c","prompt":"p","response":"r","confirmed":false}), 2);
        ds.push_value(&json!({"id":"d","prompt":"p","response":"r","status":"proposed"}), 3);
        ds.push_value(&json!({"id":"e","prompt":"p","chosen":"good","rejected":"bad","status":"corrected"}), 4);
        ds.push_value(
            &json!({"id":"f","confirmed":true,"messages":[{"role":"system","content":"s"},{"role":"user","content":"u"},{"role":"assistant","content":"a"}]}),
            5,
        );
        assert_eq!(ds.sft.len(), 2);
        assert_eq!(ds.pref.len(), 1);
        assert_eq!(ds.dropped_unconfirmed, 3);
        assert_eq!(ds.sft[1].system.as_deref(), Some("s"));
        assert_eq!(ds.sft[1].prompt, Text::Str("u".into()));
    }

    #[test]
    fn split_is_deterministic() {
        let mut ds = Dataset::default();
        for i in 0..200 {
            ds.push_value(&json!({"id": format!("x{i}"),"prompt":"p","response":"r","confirmed":true}), i);
        }
        let (a, b) = ds.split(0.2);
        let (c, d) = ds.split(0.2);
        assert_eq!(a.sft.len(), c.sft.len());
        assert_eq!(b.sft.len(), d.sft.len());
        assert!(b.sft.len() > 20 && b.sft.len() < 60);
    }
}
