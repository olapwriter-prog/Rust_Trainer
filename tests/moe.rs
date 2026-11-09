//! Mixture of LoRA experts: grouping, routing, per-expert training with
//! pause/resume, gating against the current model, and fallback to one adapter.

use rust_trainer::data::Text;
use rust_trainer::eval::GateConfig;
use rust_trainer::model::{Model, ModelConfig};
use rust_trainer::moe::{is_bundle, load_router, plan, prompt_of, MoeConfig};
use rust_trainer::registry::Registry;
use rust_trainer::scheduler::{
    run_overnight, run_overnight_moe, DeviceMonitor, DeviceState, FixedMonitor, Heldout, JobConfig, Outcome, PowerPolicy, Stage,
};
use rust_trainer::tokenizer::{ByteTokenizer, ChatFormat, Example};
use rust_trainer::train::{TrainConfig, TrainData};
use std::io;

const SKILLS: [&str; 3] = ["home", "math", "msg"];

/// (example, skill index) for three clearly different kinds of request.
fn dataset(n_per_skill: usize, offset: usize) -> Vec<(Example, usize)> {
    let tok = ByteTokenizer;
    let f = ChatFormat::qwen(&tok, "", 96).unwrap();
    let devs = ["lamp", "fan", "heater", "radio", "tv", "kettle", "door", "blinds"];
    let rooms = ["kitchen", "hall", "office", "garage", "bedroom"];
    let names = ["bob", "ana", "li", "omar", "kim", "sara", "tom", "eva", "raj"];
    let mut out = Vec::new();
    for i in offset..offset + n_per_skill {
        let items = [
            (
                format!("turn on the {} in the {}", devs[i % devs.len()], rooms[(i / devs.len()) % rooms.len()]),
                format!("home.on({},{})", devs[i % devs.len()], rooms[(i / devs.len()) % rooms.len()]),
            ),
            (format!("what is {} plus {}", 10 + i * 7 % 83, 3 + i * 13 % 59), format!("{}", 10 + i * 7 % 83 + 3 + i * 13 % 59)),
            (
                format!("text {} that i am {} minutes late", names[i % names.len()], 5 + i % 40),
                format!("sms({},late {})", names[i % names.len()], 5 + i % 40),
            ),
        ];
        for (s, (p, a)) in items.into_iter().enumerate() {
            out.push((f.example(&tok, None, &Text::Str(p), &Text::Str(a)).unwrap(), s));
        }
    }
    out
}

fn model() -> Model {
    let mut cfg = ModelConfig::tiny_hybrid(259);
    cfg.hidden_size = 48;
    cfg.intermediate_size = 96;
    Model::random(cfg, 3, true)
}

fn moe_cfg() -> MoeConfig {
    MoeConfig { max_experts: 3, min_expert_examples: 20, ..Default::default() }
}

fn job() -> JobConfig {
    JobConfig {
        base_model: "tiny".into(),
        train: TrainConfig {
            rank: 4,
            alpha: 8.0,
            lr: 4e-3,
            epochs: 2,
            batch_size: 8,
            warmup_steps: 2,
            checkpoint_every: 2,
            ..Default::default()
        },
        gate: GateConfig { min_examples: 6, ..Default::default() },
        power: PowerPolicy::default(),
        min_train_examples: 10,
    }
}

fn ok_state() -> DeviceState {
    DeviceState { stage: Stage::Deep, charging: true, battery_pct: 90.0, temp_c: 30.0 }
}

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("rt-moe-{name}-{}-{}", std::process::id(), rust_trainer::registry::now_unix()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Majority-label purity of each group.
fn purity(groups: &[Vec<usize>], truth: &[usize]) -> f32 {
    let mut right = 0;
    let mut total = 0;
    for g in groups {
        let mut c = [0usize; 3];
        for &i in g {
            c[truth[i]] += 1;
        }
        right += c.iter().max().unwrap();
        total += g.len();
    }
    right as f32 / total.max(1) as f32
}

#[test]
fn clustering_recovers_skills_without_labels() {
    let data = dataset(30, 0);
    let prompts: Vec<&[u32]> = data.iter().map(|(e, _)| prompt_of(e)).collect();
    let truth: Vec<usize> = data.iter().map(|(_, s)| *s).collect();
    let p = plan(&prompts, &vec![None; data.len()], &moe_cfg());
    assert_eq!(p.grouping, "clusters");
    assert_eq!(p.experts().count(), 3, "{:?}", p.groups.iter().map(|g| g.members.len()).collect::<Vec<_>>());
    let groups: Vec<Vec<usize>> = p.groups.iter().map(|g| g.members.clone()).collect();
    let pu = purity(&groups, &truth);
    assert!(pu >= 0.9, "cluster purity {pu}");
    // Unseen requests route to the group of their own skill.
    let held = dataset(10, 500);
    let mut right = 0;
    for (e, s) in &held {
        let r = p.router.route(prompt_of(e));
        let g = r.group.expect("routed");
        let members = &p.groups[g].members;
        let maj = (0..3).max_by_key(|k| members.iter().filter(|&&i| truth[i] == *k).count()).unwrap();
        right += (maj == *s) as usize;
    }
    assert!(right as f32 / held.len() as f32 >= 0.9, "routing accuracy {right}/{}", held.len());
}

#[test]
fn labels_define_the_experts() {
    let data = dataset(25, 0);
    let prompts: Vec<&[u32]> = data.iter().map(|(e, _)| prompt_of(e)).collect();
    let labels: Vec<Option<String>> = data.iter().enumerate().map(|(i, (_, s))| (i % 10 != 0).then(|| SKILLS[*s].to_string())).collect();
    let p = plan(&prompts, &labels, &moe_cfg());
    assert_eq!(p.grouping, "labels");
    assert_eq!(p.router.names, vec!["home", "math", "msg"]);
    assert_eq!(p.experts().count(), 3);
    // The 10 % unlabelled examples were attached to the right labelled group.
    for (gi, g) in p.groups.iter().enumerate() {
        for &i in &g.members {
            assert_eq!(data[i].1, gi, "item {i} in group {}", g.name);
        }
    }
}

/// Device that overheats once, after `heat_at` reads.
struct HotOnce {
    calls: usize,
    heat_at: usize,
    hot: bool,
}
impl DeviceMonitor for HotOnce {
    fn state(&mut self) -> io::Result<DeviceState> {
        self.calls += 1;
        if self.calls == self.heat_at {
            self.hot = true;
        }
        let mut s = ok_state();
        if self.hot {
            s.temp_c = 45.0;
        }
        Ok(s)
    }
}

#[test]
fn overnight_mixture_trains_routes_gates_and_resumes() {
    let dir = tmp("night");
    let reg = Registry::open(&dir).unwrap();
    let all = dataset(36, 0);
    let held = dataset(6, 900);
    let train: Vec<Example> = all.iter().map(|(e, _)| e.clone()).collect();
    let held_ex: Vec<Example> = held.iter().map(|(e, _)| e.clone()).collect();
    let labels = vec![None; train.len()];

    // Night 1: overheats during the second expert and the deadline passes → deferred.
    let mut m = model();
    // The general expert (108 examples × 2 epochs ≈ 220 reads) finishes; a skill expert is cut off.
    let mut mon = HotOnce { calls: 0, heat_at: 260, hot: false };
    let mut give_up = || false;
    let out = run_overnight_moe(
        &mut m,
        &reg,
        &job(),
        &moe_cfg(),
        TrainData::Sft(train.clone()),
        &labels,
        Heldout { sft: &held_ex, pref: &[] },
        &mut mon,
        &mut give_up,
        &mut |_| {},
    )
    .unwrap();
    assert!(matches!(out, Outcome::Deferred { .. }), "{out:?}");
    let trained_before = reg.read_ledger().iter().filter(|e| e.event == "expert_trained").count();
    assert!(trained_before >= 1, "at least one expert finished before the heat");

    // Night 2: cool all night → finished experts are reused, the rest train, bundle is gated.
    let mut m = model();
    let mut mon = FixedMonitor(ok_state());
    let mut wait = || true;
    let out = run_overnight_moe(
        &mut m,
        &reg,
        &job(),
        &moe_cfg(),
        TrainData::Sft(train.clone()),
        &labels,
        Heldout { sft: &held_ex, pref: &[] },
        &mut mon,
        &mut wait,
        &mut |_| {},
    )
    .unwrap();
    let id = match out {
        Outcome::Accepted { id, report } => {
            assert!(report.candidate.nll < report.current.nll);
            id
        }
        o => panic!("expected acceptance, got {o:?}"),
    };
    let events: Vec<String> = reg.read_ledger().into_iter().map(|e| e.event).collect();
    let trained = events.iter().filter(|e| *e == "expert_trained").count();
    assert_eq!(trained, 4, "3 skill experts + general, each trained exactly once: {events:?}");
    assert_eq!(events.first().map(String::as_str), Some("started"));
    assert!(events.contains(&"paused".to_string()) && events.contains(&"deferred".to_string()) && events.contains(&"resumed".to_string()));
    assert_eq!(events.last().map(String::as_str), Some("accepted"));

    // The promoted bundle: router + engine-loadable experts, routing each skill to its own expert.
    let bundle = reg.adapter_dir(&id);
    assert!(is_bundle(&bundle));
    let router = load_router(&bundle).unwrap();
    for e in router.used_experts() {
        assert!(bundle.join("experts").join(&e).join("adapter_model.safetensors").exists(), "{e}");
    }
    let mut seen = std::collections::BTreeMap::new();
    for (e, s) in &held {
        let r = router.route(prompt_of(e));
        seen.entry(*s).or_insert_with(Vec::new).push(r.group);
    }
    let mut groups_per_skill = Vec::new();
    for (s, gs) in &seen {
        let first = gs[0];
        let agree = gs.iter().filter(|g| **g == first).count();
        assert!(agree * 10 >= gs.len() * 8, "skill {} routed inconsistently: {gs:?}", SKILLS[*s]);
        groups_per_skill.push(first);
    }
    groups_per_skill.sort();
    groups_per_skill.dedup();
    assert_eq!(groups_per_skill.len(), 3, "each skill has its own group");

    // A later single-adapter night is measured against the routed mixture.
    let mut m = model();
    let out = run_overnight(
        &mut m,
        &reg,
        &job(),
        TrainData::Sft(train.clone()),
        Heldout { sft: &held_ex, pref: &[] },
        &mut mon,
        &mut wait,
        &mut |_| {},
    )
    .unwrap();
    assert!(matches!(out, Outcome::Accepted { .. } | Outcome::Rejected { .. }));
    if let Outcome::Rejected { .. } = out {
        assert_eq!(reg.active().as_deref(), Some(id.as_str()), "the mixture stays active");
    }
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn single_skill_data_falls_back_to_one_adapter() {
    let dir = tmp("single");
    let reg = Registry::open(&dir).unwrap();
    let train: Vec<Example> = dataset(30, 0).into_iter().filter(|(_, s)| *s == 0).map(|(e, _)| e).collect();
    let held: Vec<Example> = dataset(8, 700).into_iter().filter(|(_, s)| *s == 0).map(|(e, _)| e).collect();
    let mut m = model();
    let mut mon = FixedMonitor(ok_state());
    let mut wait = || true;
    let cfg = MoeConfig { max_experts: 3, min_expert_examples: 20, ..Default::default() };
    let out = run_overnight_moe(
        &mut m,
        &reg,
        &job(),
        &cfg,
        TrainData::Sft(train.clone()),
        &vec![None; train.len()],
        Heldout { sft: &held, pref: &[] },
        &mut mon,
        &mut wait,
        &mut |_| {},
    )
    .unwrap();
    if let Outcome::Accepted { id, .. } = &out {
        assert!(!is_bundle(reg.adapter_dir(id)), "one group → a plain adapter");
    }
    assert!(!reg.read_ledger().iter().any(|e| e.event == "expert_trained"));
    std::fs::remove_dir_all(dir).ok();
}
