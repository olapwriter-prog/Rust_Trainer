//! Eval gate: a candidate adapter is used only if it measurably beats the
//! current model on held-out confirmed requests.

use crate::model::{answer_targets, Model};
use crate::tokenizer::Example;
use crate::train::PrefPair;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GateConfig {
    /// Required relative reduction of held-out answer NLL, e.g. 0.02 = 2 %.
    pub min_rel_improvement: f32,
    /// Below this many held-out examples the gate refuses (not enough evidence).
    pub min_examples: usize,
    /// Greedy next-token accuracy on answers may not drop by more than this.
    pub max_accuracy_drop: f32,
}

impl Default for GateConfig {
    fn default() -> Self {
        GateConfig { min_rel_improvement: 0.02, min_examples: 8, max_accuracy_drop: 0.0 }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Metrics {
    /// Mean negative log-likelihood per answer token.
    pub nll: f32,
    /// Fraction of answer tokens that are the greedy argmax.
    pub accuracy: f32,
    /// Fraction of preference pairs where chosen outscores rejected (if any pairs).
    pub pref_accuracy: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EvalReport {
    pub examples: usize,
    pub tokens: usize,
    pub current: Metrics,
    pub candidate: Metrics,
    pub rel_improvement: f32,
    pub accepted: bool,
    pub reason: String,
}

/// Additive evaluation totals, so results over several groups (e.g. one per
/// expert of a mixture) can be combined exactly.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sums {
    pub nll: f64,
    pub hits: usize,
    pub tokens: usize,
    pub pref_wins: usize,
    pub pref_n: usize,
}

impl Sums {
    pub fn add(&mut self, o: &Sums) {
        self.nll += o.nll;
        self.hits += o.hits;
        self.tokens += o.tokens;
        self.pref_wins += o.pref_wins;
        self.pref_n += o.pref_n;
    }
    pub fn metrics(&self) -> Metrics {
        let t = self.tokens.max(1) as f64;
        Metrics {
            nll: (self.nll / t) as f32,
            accuracy: (self.hits as f64 / t) as f32,
            pref_accuracy: (self.pref_n > 0).then(|| self.pref_wins as f32 / self.pref_n as f32),
        }
    }
}

/// Totals for the model as currently configured (adapters on or off as set).
pub fn measure_sums(model: &Model, sft: &[Example], pref: &[PrefPair]) -> Sums {
    let mut s = Sums::default();
    for ex in sft {
        let targets = answer_targets(&ex.tokens, ex.answer_start);
        if targets.is_empty() {
            continue;
        }
        let (lp, acc) = logprobs_and_hits(model, &ex.tokens, &targets);
        s.nll -= lp.iter().map(|v| *v as f64).sum::<f64>();
        s.hits += acc;
        s.tokens += targets.len();
    }
    for p in pref {
        let sc = |e: &Example| model.logprobs(&e.tokens, &answer_targets(&e.tokens, e.answer_start)).iter().sum::<f32>();
        if sc(&p.chosen) > sc(&p.rejected) {
            s.pref_wins += 1;
        }
        s.pref_n += 1;
    }
    s
}

/// Measure the model as currently configured (adapters on or off as set).
pub fn measure(model: &Model, sft: &[Example], pref: &[PrefPair]) -> (Metrics, usize) {
    let s = measure_sums(model, sft, pref);
    (s.metrics(), s.tokens)
}

fn logprobs_and_hits(model: &Model, tokens: &[u32], targets: &[crate::model::Target]) -> (Vec<f32>, usize) {
    model.score(tokens, targets)
}

/// Decide whether `candidate` beats `current` (both measured on the same held-out set).
pub fn gate(cfg: &GateConfig, current: &Metrics, candidate: &Metrics, examples: usize, tokens: usize) -> EvalReport {
    let rel = if current.nll > 0.0 { (current.nll - candidate.nll) / current.nll } else { 0.0 };
    let (accepted, reason) = if examples < cfg.min_examples {
        (false, format!("only {examples} held-out examples (< {})", cfg.min_examples))
    } else if !candidate.nll.is_finite() {
        (false, "candidate NLL is not finite".to_string())
    } else if rel < cfg.min_rel_improvement {
        (false, format!("NLL improvement {:.2}% < required {:.2}%", rel * 100.0, cfg.min_rel_improvement * 100.0))
    } else if candidate.accuracy + cfg.max_accuracy_drop < current.accuracy {
        (false, format!("answer accuracy fell {:.3} → {:.3}", current.accuracy, candidate.accuracy))
    } else if let (Some(a), Some(b)) = (current.pref_accuracy, candidate.pref_accuracy) {
        if b < a {
            (false, format!("preference accuracy fell {a:.3} → {b:.3}"))
        } else {
            (true, format!("NLL −{:.2}%", rel * 100.0))
        }
    } else {
        (true, format!("NLL −{:.2}%", rel * 100.0))
    };
    EvalReport { examples, tokens, current: current.clone(), candidate: candidate.clone(), rel_improvement: rel, accepted, reason }
}

/// Convenience: evaluate one model with its adapters off (= current base) and on (= candidate).
pub fn evaluate_vs_base(model: &mut Model, cfg: &GateConfig, sft: &[Example], pref: &[PrefPair]) -> EvalReport {
    model.set_lora_enabled(false);
    let (cur, tokens) = measure(model, sft, pref);
    model.set_lora_enabled(true);
    let (cand, _) = measure(model, sft, pref);
    gate(cfg, &cur, &cand, sft.len() + pref.len(), tokens)
}
