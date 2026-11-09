//! Mixture of LoRA experts, trained only on your confirmed data.
//!
//! A single adapter has to serve every kind of request at once, so skills
//! compete for the same few weights. Instead, each night:
//!
//! 1. Confirmed examples are grouped by skill: by the exporter's label
//!    (`skill`/`task`/`domain`/`category`/`tool`) when present, otherwise by
//!    cosine k-means over prompt features.
//! 2. A *general* expert trains on everything first (it also serves requests the
//!    router is unsure about). Each group with enough examples then gets its own
//!    LoRA expert, initialised from the general expert and specialised on that
//!    group (branch-train), so experts are never starved by seeing less data.
//! 3. A router picks the expert per request from cheap prompt features: hashed
//!    TF-IDF over the prompt's token ids and adjacent token pairs. Tokens that
//!    occur in every prompt (chat-template tokens) get zero weight, and the words
//!    that make a request distinctive dominate. The router takes the nearest group
//!    centroid by cosine similarity. No forward pass is needed, and the engine
//!    reproduces it with a hash and a table lookup per token.
//! 4. Each expert is kept only where it beats the fallback on its own held-out
//!    requests, and the routed bundle as a whole must pass the normal eval gate
//!    against the current model.
//!
//! This is a mixture of *adapters* on a frozen base. It does not turn the base
//! into a sparse MoE: retraining per-layer routers and experts needs far more
//! data and compute than a phone and a few hundred examples provide.
//!
//! On disk a bundle is an adapter directory with
//! `router.json` and `experts/<name>/` (each a normal adapter directory with
//! `adapter.gguf`), so the registry's promote / reject / rollback work unchanged.

use crate::adapter::load_adapter_into;
use crate::eval::{measure_sums, Sums};
use crate::model::Model;
use crate::tensor::{dot, Rng};
use crate::tokenizer::Example;
use crate::train::PrefPair;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

pub const ROUTER_FILE: &str = "router.json";
pub const EXPERTS_DIR: &str = "experts";
pub const GENERAL: &str = "general";
/// Identifies the routing-feature recipe so the engine can check it implements the same one.
pub const FEATURE_KIND: &str = "tfidf_hash_v1";
/// Hash buckets of the routing features.
pub const FEATURE_DIM: usize = 4096;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MoeConfig {
    /// Upper bound on the number of skill experts (the general expert is extra).
    pub max_experts: usize,
    /// A group needs at least this many training examples to get its own expert.
    pub min_expert_examples: usize,
    /// Also train a general expert on all data, used when routing is unsure.
    pub general_expert: bool,
    /// Below this cosine similarity to every centroid a request goes to the fallback.
    pub min_similarity: f32,
    pub seed: u64,
}

impl Default for MoeConfig {
    fn default() -> Self {
        MoeConfig { max_experts: 4, min_expert_examples: 24, general_expert: true, min_similarity: 0.1, seed: 7 }
    }
}

fn normalize(v: &mut [f32]) {
    let n = dot(v, v).sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// SplitMix64 finaliser: the hash behind the feature buckets.
pub fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Feature buckets of a prompt: one per token (`mix64(t) % D`) and one per adjacent
/// pair (`mix64(1<<63 | a<<32 | b) % D`), each with its count.
pub fn buckets(prompt: &[u32], dim: usize) -> BTreeMap<usize, u32> {
    let mut c: BTreeMap<usize, u32> = BTreeMap::new();
    for &t in prompt {
        *c.entry((mix64(t as u64) % dim as u64) as usize).or_default() += 1;
    }
    for w in prompt.windows(2) {
        let key = (1u64 << 63) | ((w[0] as u64) << 32) | w[1] as u64;
        *c.entry((mix64(key) % dim as u64) as usize).or_default() += 1;
    }
    c
}

/// Inverse document frequency per bucket: `ln((1 + N) / (1 + df))`, which is 0 for
/// buckets present in every prompt.
pub fn idf(prompts: &[&[u32]], dim: usize) -> Vec<f32> {
    let mut df = vec![0usize; dim];
    for p in prompts {
        for b in buckets(p, dim).keys() {
            df[*b] += 1;
        }
    }
    let n = prompts.len() as f32;
    df.iter().map(|&d| ((1.0 + n) / (1.0 + d as f32)).ln()).collect()
}

/// Unit-length TF-IDF vector: `(1 + ln count) · idf` per bucket, L2-normalised.
pub fn tfidf(prompt: &[u32], idf: &[f32]) -> Vec<f32> {
    let mut v = vec![0.0f32; idf.len()];
    for (b, c) in buckets(prompt, idf.len()) {
        v[b] = (1.0 + (c as f32).ln()) * idf[b];
    }
    normalize(&mut v);
    v
}

/// The router stored next to the experts (`router.json`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Router {
    pub feature: String,
    pub dim: usize,
    /// IDF weight per feature bucket (see [`tfidf`]).
    pub idf: Vec<f32>,
    /// Unit-length group centroids.
    pub centroids: Vec<Vec<f32>>,
    /// Group name per centroid (the skill label, or `c0`, `c1`, … for clusters).
    pub names: Vec<String>,
    /// Expert used for each centroid; `None` means the fallback.
    pub experts: Vec<Option<String>>,
    /// Expert for unsure requests and groups without an expert; `None` = base model.
    pub fallback: Option<String>,
    pub min_similarity: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Route {
    /// Nearest group, if similar enough.
    pub group: Option<usize>,
    pub similarity: f32,
    /// Expert to apply; `None` = base model.
    pub expert: Option<String>,
}

impl Router {
    pub fn features(&self, prompt: &[u32]) -> Vec<f32> {
        tfidf(prompt, &self.idf)
    }

    pub fn route_unit(&self, u: &[f32]) -> Route {
        let mut best = (None, f32::NEG_INFINITY);
        for (i, c) in self.centroids.iter().enumerate() {
            let s = dot(u, c);
            if s > best.1 {
                best = (Some(i), s);
            }
        }
        match best.0 {
            Some(i) if best.1 >= self.min_similarity => {
                Route { group: Some(i), similarity: best.1, expert: self.experts[i].clone().or_else(|| self.fallback.clone()) }
            }
            _ => Route { group: None, similarity: best.1, expert: self.fallback.clone() },
        }
    }

    pub fn route(&self, prompt: &[u32]) -> Route {
        self.route_unit(&self.features(prompt))
    }

    /// Experts the router can select (including the fallback).
    pub fn used_experts(&self) -> Vec<String> {
        let mut v: Vec<String> = self.experts.iter().flatten().cloned().collect();
        v.extend(self.fallback.clone());
        v.sort();
        v.dedup();
        v
    }
}

/// Cosine k-means with k-means++ seeding on unit vectors. Deterministic for a seed.
pub fn kmeans_cosine(points: &[Vec<f32>], k: usize, seed: u64, iters: usize) -> (Vec<Vec<f32>>, Vec<usize>) {
    let n = points.len();
    assert!(k >= 1 && n >= k);
    let mut rng = Rng::new(seed);
    let mut cents: Vec<Vec<f32>> = vec![points[rng.below(n)].clone()];
    while cents.len() < k {
        // distance to nearest centroid: 1 - cos
        let d: Vec<f32> = points.iter().map(|p| cents.iter().map(|c| 1.0 - dot(p, c)).fold(f32::INFINITY, f32::min).max(0.0)).collect();
        let total: f32 = d.iter().map(|x| x * x).sum();
        let pick = if total <= 0.0 {
            rng.below(n)
        } else {
            let mut r = rng.uniform() * total;
            let mut idx = n - 1;
            for (i, x) in d.iter().enumerate() {
                r -= x * x;
                if r <= 0.0 {
                    idx = i;
                    break;
                }
            }
            idx
        };
        cents.push(points[pick].clone());
    }
    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        let mut changed = false;
        for (i, p) in points.iter().enumerate() {
            let a = (0..k).max_by(|&x, &y| dot(p, &cents[x]).total_cmp(&dot(p, &cents[y]))).unwrap();
            if a != assign[i] {
                assign[i] = a;
                changed = true;
            }
        }
        let mut sums = vec![vec![0.0f32; points[0].len()]; k];
        let mut counts = vec![0usize; k];
        for (i, p) in points.iter().enumerate() {
            crate::tensor::add_into(&mut sums[assign[i]], p);
            counts[assign[i]] += 1;
        }
        for c in 0..k {
            if counts[c] == 0 {
                // Re-seed an empty cluster at the point worst served by its centroid.
                let far =
                    (0..n).min_by(|&x, &y| dot(&points[x], &cents[assign[x]]).total_cmp(&dot(&points[y], &cents[assign[y]]))).unwrap();
                sums[c] = points[far].clone();
                changed = true;
            }
            normalize(&mut sums[c]);
        }
        cents = sums;
        if !changed {
            break;
        }
    }
    (cents, assign)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Group {
    pub name: String,
    /// Indices into the training items.
    pub members: Vec<usize>,
    pub has_expert: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Plan {
    pub router: Router,
    /// Aligned with `router.centroids`.
    pub groups: Vec<Group>,
    /// "labels" or "clusters".
    pub grouping: String,
}

impl Plan {
    pub fn experts(&self) -> impl Iterator<Item = &Group> {
        self.groups.iter().filter(|g| g.has_expert)
    }
}

/// Decide the groups and build the router from training prompts and optional labels.
pub fn plan(prompts: &[&[u32]], labels: &[Option<String>], cfg: &MoeConfig) -> Plan {
    let n = prompts.len();
    assert_eq!(labels.len(), n);
    let d = FEATURE_DIM;
    let idf = idf(prompts, d);
    let units: Vec<Vec<f32>> = prompts.iter().map(|p| tfidf(p, &idf)).collect();

    let labeled = labels.iter().filter(|l| l.is_some()).count();
    let (names, assign, grouping): (Vec<String>, Vec<usize>, &str) = if n > 0 && labeled * 5 >= n * 4 {
        let mut names: Vec<String> = labels.iter().flatten().cloned().collect();
        names.sort();
        names.dedup();
        // Unlabelled items go to the nearest labelled centroid below.
        let mut assign: Vec<usize> =
            labels.iter().map(|l| l.as_ref().map(|s| names.binary_search(s).unwrap()).unwrap_or(usize::MAX)).collect();
        let mut cents = vec![vec![0.0f32; d]; names.len()];
        for (i, a) in assign.iter().enumerate() {
            if *a != usize::MAX {
                crate::tensor::add_into(&mut cents[*a], &units[i]);
            }
        }
        cents.iter_mut().for_each(|c| normalize(c));
        for (i, a) in assign.iter_mut().enumerate() {
            if *a == usize::MAX {
                *a = (0..names.len()).max_by(|&x, &y| dot(&units[i], &cents[x]).total_cmp(&dot(&units[i], &cents[y]))).unwrap();
            }
        }
        (names, assign, "labels")
    } else {
        let k = cfg.max_experts.min(n / cfg.min_expert_examples.max(1)).max(1).min(n.max(1));
        if k <= 1 {
            (vec!["all".to_string()], vec![0; n], "clusters")
        } else {
            let (_, assign) = kmeans_cosine(&units, k, cfg.seed, 100);
            ((0..k).map(|i| format!("c{i}")).collect(), assign, "clusters")
        }
    };

    let mut groups: Vec<Group> = names.iter().map(|nm| Group { name: nm.clone(), members: Vec::new(), has_expert: false }).collect();
    for (i, a) in assign.iter().enumerate() {
        groups[*a].members.push(i);
    }
    // Experts for the largest groups that have enough data, at most max_experts.
    let mut order: Vec<usize> = (0..groups.len()).collect();
    order.sort_by_key(|&g| std::cmp::Reverse(groups[g].members.len()));
    for &g in order.iter().take(cfg.max_experts) {
        groups[g].has_expert = groups[g].members.len() >= cfg.min_expert_examples;
    }
    let centroids: Vec<Vec<f32>> = groups
        .iter()
        .map(|g| {
            let mut c = vec![0.0f32; d];
            for &i in &g.members {
                crate::tensor::add_into(&mut c, &units[i]);
            }
            normalize(&mut c);
            c
        })
        .collect();
    let router = Router {
        feature: FEATURE_KIND.to_string(),
        dim: d,
        idf,
        centroids,
        names: groups.iter().map(|g| g.name.clone()).collect(),
        experts: groups.iter().map(|g| g.has_expert.then(|| g.name.clone())).collect(),
        fallback: cfg.general_expert.then(|| GENERAL.to_string()),
        min_similarity: cfg.min_similarity,
    };
    Plan { router, groups, grouping: grouping.to_string() }
}

pub fn is_bundle(dir: impl AsRef<Path>) -> bool {
    dir.as_ref().join(ROUTER_FILE).exists()
}

pub fn expert_dir(bundle: impl AsRef<Path>, name: &str) -> PathBuf {
    bundle.as_ref().join(EXPERTS_DIR).join(name)
}

pub fn load_router(bundle: impl AsRef<Path>) -> io::Result<Router> {
    let b = std::fs::read(bundle.as_ref().join(ROUTER_FILE))?;
    let r: Router = serde_json::from_slice(&b).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    if r.feature != FEATURE_KIND {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("unknown router feature {}", r.feature)));
    }
    Ok(r)
}

/// What "the current model" is when measuring a candidate against it.
pub enum Current {
    Base,
    Single(PathBuf),
    Bundle(PathBuf, Router),
}

impl Current {
    pub fn from_dir(dir: Option<PathBuf>) -> io::Result<Self> {
        Ok(match dir {
            None => Current::Base,
            Some(d) if is_bundle(&d) => {
                let r = load_router(&d)?;
                Current::Bundle(d, r)
            }
            Some(d) => Current::Single(d),
        })
    }
}

/// Load `expert` (`None` = base model) into the model.
pub fn select_expert(model: &mut Model, bundle: &Path, expert: Option<&str>) -> io::Result<()> {
    match expert {
        None => {
            model.set_lora_enabled(false);
            Ok(())
        }
        Some(e) => load_adapter_into(model, expert_dir(bundle, e)).map(|_| ()),
    }
}

/// The prompt of a training/eval item: everything before the answer.
pub fn prompt_of(ex: &Example) -> &[u32] {
    &ex.tokens[..ex.answer_start.min(ex.tokens.len())]
}

/// Evaluate a configuration on held-out data; bundles route every example and load
/// each expert once.
pub fn measure_current(model: &mut Model, cur: &Current, sft: &[Example], pref: &[PrefPair]) -> io::Result<Sums> {
    match cur {
        Current::Base => {
            model.set_lora_enabled(false);
            let s = measure_sums(model, sft, pref);
            model.set_lora_enabled(true);
            Ok(s)
        }
        Current::Single(d) => {
            load_adapter_into(model, d)?;
            Ok(measure_sums(model, sft, pref))
        }
        Current::Bundle(d, r) => {
            let mut by: BTreeMap<Option<String>, (Vec<Example>, Vec<PrefPair>)> = BTreeMap::new();
            for ex in sft {
                by.entry(r.route(prompt_of(ex)).expert).or_default().0.push(ex.clone());
            }
            for p in pref {
                by.entry(r.route(prompt_of(&p.chosen)).expert).or_default().1.push(p.clone());
            }
            let mut total = Sums::default();
            for (expert, (s, p)) in by {
                select_expert(model, d, expert.as_deref())?;
                total.add(&measure_sums(model, &s, &p));
            }
            model.set_lora_enabled(true);
            Ok(total)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_separates_clear_clusters() {
        let mut rng = Rng::new(1);
        let mut pts = Vec::new();
        for c in 0..3 {
            for _ in 0..20 {
                let mut v: Vec<f32> = (0..8).map(|i| if i == c { 1.0 } else { 0.0 } + rng.normal() * 0.05).collect();
                normalize(&mut v);
                pts.push(v);
            }
        }
        let (_, assign) = kmeans_cosine(&pts, 3, 3, 50);
        for c in 0..3 {
            let first = assign[c * 20];
            assert!(assign[c * 20..(c + 1) * 20].iter().all(|a| *a == first));
        }
        let mut uniq = assign.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), 3);
    }

    #[test]
    fn tfidf_ignores_shared_tokens() {
        let a = [1u32, 2, 10, 11];
        let b = [1u32, 2, 20];
        let c = [1u32, 2, 30, 31];
        let p: Vec<&[u32]> = vec![&a, &b, &c];
        let w = idf(&p, FEATURE_DIM);
        let b1 = (mix64(1) % FEATURE_DIM as u64) as usize;
        let b10 = (mix64(10) % FEATURE_DIM as u64) as usize;
        assert_eq!(w[b1], 0.0);
        assert!(w[b10] > 0.0);
        let v = tfidf(&a, &w);
        assert!((dot(&v, &v) - 1.0).abs() < 1e-5);
        assert_eq!(v[b1], 0.0);
    }
}
