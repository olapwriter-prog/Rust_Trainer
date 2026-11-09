//! Adapter registry next to the model: versions, the active pointer, rollback
//! and an append-only ledger.
//!
//! ```text
//! <model_dir>/adapters/
//!   ACTIVE          id of the adapter in use (missing or empty = base model)
//!   history.json    accepted ids, oldest first (rollback walks this backwards)
//!   ledger.jsonl    one JSON object per event
//!   <id>/           adapter directories (see adapter.rs)
//!   work/           the in-progress run: checkpoint + job description
//! ```
//! Only this directory is ever written by the trainer. Approval tiers and the
//! approval key live with the executor and are never read or written here.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub ts: u64,
    pub event: String,
    pub adapter: Option<String>,
    pub detail: Value,
}

#[derive(Default, Serialize, Deserialize)]
struct History {
    accepted: Vec<String>,
}

pub struct Registry {
    pub root: PathBuf,
    /// Number of accepted adapters kept on disk for rollback.
    pub keep: usize,
}

impl Registry {
    /// Registry next to the model: `<dir>/adapters/` for a model directory,
    /// `<name>.adapters/` beside a model file such as `qwen3-4b-q4_0.gguf`.
    pub fn open(model: impl AsRef<Path>) -> io::Result<Self> {
        let p = model.as_ref();
        let root = if p.is_file() { p.with_extension("adapters") } else { p.join("adapters") };
        fs::create_dir_all(&root)?;
        Ok(Registry { root, keep: 5 })
    }

    pub fn work_dir(&self) -> PathBuf {
        self.root.join("work")
    }

    pub fn adapter_dir(&self, id: &str) -> PathBuf {
        self.root.join(id)
    }

    pub fn active(&self) -> Option<String> {
        let s = fs::read_to_string(self.root.join("ACTIVE")).ok()?;
        let s = s.trim();
        (!s.is_empty()).then(|| s.to_string())
    }

    pub fn active_dir(&self) -> Option<PathBuf> {
        self.active().map(|id| self.adapter_dir(&id))
    }

    fn set_active(&self, id: Option<&str>) -> io::Result<()> {
        let tmp = self.root.join("ACTIVE.tmp");
        fs::write(&tmp, id.unwrap_or(""))?;
        fs::rename(tmp, self.root.join("ACTIVE"))
    }

    fn history(&self) -> History {
        fs::read(self.root.join("history.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
    }

    fn save_history(&self, h: &History) -> io::Result<()> {
        crate::adapter::write_json(self.root.join("history.json"), h)
    }

    /// Accepted adapters, oldest first.
    pub fn accepted(&self) -> Vec<String> {
        self.history().accepted
    }

    /// A fresh, unused adapter id.
    pub fn new_id(&self) -> String {
        let base = format!("v{}", now_unix());
        let mut id = base.clone();
        let mut n = 1;
        while self.adapter_dir(&id).exists() {
            id = format!("{base}-{n}");
            n += 1;
        }
        id
    }

    pub fn ledger(&self, event: &str, adapter: Option<&str>, detail: Value) -> io::Result<()> {
        let e = LedgerEntry { ts: now_unix(), event: event.to_string(), adapter: adapter.map(str::to_string), detail };
        let mut f = OpenOptions::new().create(true).append(true).open(self.root.join("ledger.jsonl"))?;
        writeln!(f, "{}", serde_json::to_string(&e)?)?;
        f.sync_all()
    }

    pub fn read_ledger(&self) -> Vec<LedgerEntry> {
        fs::read_to_string(self.root.join("ledger.jsonl"))
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect()
    }

    /// Make `id` the active adapter and prune old versions beyond `keep`.
    pub fn promote(&self, id: &str, detail: Value) -> io::Result<()> {
        if !self.adapter_dir(id).is_dir() {
            return Err(io::Error::new(io::ErrorKind::NotFound, format!("no adapter {id}")));
        }
        let previous = self.active();
        let mut h = self.history();
        h.accepted.retain(|x| x != id);
        h.accepted.push(id.to_string());
        while h.accepted.len() > self.keep {
            let old = h.accepted.remove(0);
            let _ = fs::remove_dir_all(self.adapter_dir(&old));
        }
        self.save_history(&h)?;
        self.set_active(Some(id))?;
        self.ledger("accepted", Some(id), serde_json::json!({"previous": previous, "eval": detail}))
    }

    /// Throw a candidate away.
    pub fn reject(&self, id: &str, detail: Value) -> io::Result<()> {
        let _ = fs::remove_dir_all(self.adapter_dir(id));
        self.ledger("rejected", Some(id), serde_json::json!({"eval": detail}))
    }

    /// Return to the previously accepted adapter (or the base model when none).
    /// The adapter rolled back from is kept on disk but leaves the history.
    pub fn rollback(&self, reason: &str) -> io::Result<Option<String>> {
        let current = self.active();
        let mut h = self.history();
        if let Some(c) = &current {
            h.accepted.retain(|x| x != c);
        }
        let target = h.accepted.last().cloned();
        self.save_history(&h)?;
        self.set_active(target.as_deref())?;
        self.ledger("rollback", target.as_deref(), serde_json::json!({"from": current, "to": target, "reason": reason}))?;
        Ok(target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn promote_reject_rollback() {
        let dir = std::env::temp_dir().join(format!("reg-test-{}-{}", std::process::id(), now_unix()));
        let r = Registry::open(&dir).unwrap();
        for id in ["a", "b", "c"] {
            fs::create_dir_all(r.adapter_dir(id)).unwrap();
        }
        assert_eq!(r.active(), None);
        r.promote("a", Value::Null).unwrap();
        r.promote("b", Value::Null).unwrap();
        r.reject("c", Value::Null).unwrap();
        assert!(!r.adapter_dir("c").exists());
        assert_eq!(r.active().as_deref(), Some("b"));
        assert_eq!(r.rollback("test").unwrap().as_deref(), Some("a"));
        assert_eq!(r.rollback("test").unwrap(), None);
        let events: Vec<String> = r.read_ledger().into_iter().map(|e| e.event).collect();
        assert_eq!(events, vec!["accepted", "accepted", "rejected", "rollback", "rollback"]);
        fs::remove_dir_all(dir).ok();
    }
}
