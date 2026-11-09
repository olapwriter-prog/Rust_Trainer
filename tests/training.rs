//! Training-loop behaviour: loss falls, pause/resume is exact, DPO moves
//! preferences, adapters round-trip, and the overnight job gates, swaps,
//! rolls back and writes the ledger.

use rust_trainer::adapter::{base_fingerprint, load_adapter_into, save_adapter, AdapterMeta, LoraAdapter};
use rust_trainer::data::Text;
use rust_trainer::eval::{evaluate_vs_base, measure, GateConfig};
use rust_trainer::model::{answer_targets, Model, ModelConfig};
use rust_trainer::registry::Registry;
use rust_trainer::scheduler::{run_overnight, DeviceMonitor, DeviceState, Heldout, JobConfig, Outcome, PowerPolicy, Stage};
use rust_trainer::tokenizer::{ByteTokenizer, ChatFormat, Example};
use rust_trainer::train::{Objective, PrefPair, RunToEnd, StopAfter, StopReason, TrainConfig, TrainData, Trainer};
use std::cell::RefCell;
use std::io;
use std::rc::Rc;

fn tmpdir(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("rt-{name}-{}-{}", std::process::id(), rust_trainer::registry::now_unix()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn small_model(seed: u64, q8: bool) -> Model {
    let mut cfg = ModelConfig::tiny_hybrid(259);
    cfg.hidden_size = 32;
    Model::random(cfg, seed, q8)
}

fn examples(pairs: &[(&str, &str)]) -> Vec<Example> {
    let tok = ByteTokenizer;
    let f = ChatFormat::qwen(&tok, "", 96).unwrap();
    pairs
        .iter()
        .map(|(p, r)| f.example(&tok, None, &Text::Str(p.to_string()), &Text::Str(r.to_string())).unwrap())
        .collect()
}

fn toy_sets() -> (Vec<Example>, Vec<Example>) {
    let mut tr = Vec::new();
    let mut he = Vec::new();
    let devs = ["lamp", "fan", "tv", "oven", "radio"];
    let rooms = ["hall", "den", "loft", "yard"];
    for (i, d) in devs.iter().enumerate() {
        for (j, r) in rooms.iter().enumerate() {
            let p = format!("on {d} {r}");
            let a = format!("act=on dev={d} room={r}");
            if (i + j) % 4 == 0 { he.push((p, a)) } else { tr.push((p, a)) }
        }
    }
    let conv = |v: Vec<(String, String)>| {
        let refs: Vec<(&str, &str)> = v.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
        examples(&refs)
    };
    (conv(tr), conv(he))
}

fn cfg(epochs: usize) -> TrainConfig {
    TrainConfig {
        rank: 4,
        alpha: 8.0,
        lr: 5e-3,
        epochs,
        batch_size: 3,
        warmup_steps: 2,
        checkpoint_every: 2,
        ..Default::default()
    }
}

#[test]
fn sft_loss_falls_and_gate_accepts() {
    let (tr, he) = toy_sets();
    let mut model = small_model(1, true);
    let mut t = Trainer::new(&mut model, cfg(12), TrainData::Sft(tr));
    assert_eq!(t.run(&mut RunToEnd, None, |_| {}).unwrap(), StopReason::Finished);
    let l = &t.state.losses;
    let first: f32 = l[..3].iter().sum::<f32>() / 3.0;
    let last: f32 = l[l.len() - 3..].iter().sum::<f32>() / 3.0;
    assert!(last < 0.5 * first, "loss {first} -> {last}");
    let rep = evaluate_vs_base(t.model, &GateConfig { min_examples: 3, ..Default::default() }, &he, &[]);
    assert!(rep.accepted, "{rep:?}");
    assert!(rep.candidate.accuracy > rep.current.accuracy);
}

/// Unplugging mid-run and plugging back in must give *exactly* the same adapter
/// as an uninterrupted run (same data order, optimizer state and schedule).
#[test]
fn pause_resume_is_exact() {
    let (tr, _) = toy_sets();
    let dir = tmpdir("resume");
    let ckpt = dir.join("ck.safetensors");

    let mut m1 = small_model(2, false);
    let mut t1 = Trainer::new(&mut m1, cfg(2), TrainData::Sft(tr.clone()));
    t1.run(&mut RunToEnd, None, |_| {}).unwrap();
    let full: Vec<Vec<f32>> = t1.model.lora_linears().iter().map(|l| l.lora.as_ref().unwrap().b.clone()).collect();

    // Pause twice (once mid-batch), resuming each time from the checkpoint in a fresh process-like state.
    for stop_after in [7usize, 5] {
        let mut m = small_model(2, false);
        let mut t = Trainer::new(&mut m, cfg(2), TrainData::Sft(tr.clone()));
        if ckpt.exists() {
            t.load_checkpoint(&ckpt).unwrap();
        }
        assert_eq!(t.run(&mut StopAfter(stop_after), Some(&ckpt), |_| {}).unwrap(), StopReason::Paused);
    }
    let mut m2 = small_model(2, false);
    let mut t2 = Trainer::new(&mut m2, cfg(2), TrainData::Sft(tr.clone()));
    t2.load_checkpoint(&ckpt).unwrap();
    assert!(t2.state.step > 0);
    assert_eq!(t2.run(&mut RunToEnd, Some(&ckpt), |_| {}).unwrap(), StopReason::Finished);
    let resumed: Vec<Vec<f32>> = t2.model.lora_linears().iter().map(|l| l.lora.as_ref().unwrap().b.clone()).collect();
    assert_eq!(full, resumed, "resumed run diverged from uninterrupted run");

    // A checkpoint is refused if the data changed.
    let mut m3 = small_model(2, false);
    let mut t3 = Trainer::new(&mut m3, cfg(2), TrainData::Sft(tr[1..].to_vec()));
    assert!(t3.load_checkpoint(&ckpt).is_err());
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn dpo_moves_preferences() {
    let pairs: Vec<PrefPair> = ["lamp", "fan", "tv", "oven", "radio", "door"]
        .iter()
        .map(|d| {
            let ch = examples(&[(&format!("open {d}"), &format!("tool=open dev={d}"))]).remove(0);
            let rj = examples(&[(&format!("open {d}"), &format!("rm -rf {d}"))]).remove(0);
            PrefPair { chosen: ch, rejected: rj }
        })
        .collect();
    let margin = |m: &Model| -> f32 {
        pairs
            .iter()
            .map(|p| {
                let s = |e: &Example| m.logprobs(&e.tokens, &answer_targets(&e.tokens, e.answer_start)).iter().sum::<f32>();
                s(&p.chosen) - s(&p.rejected)
            })
            .sum::<f32>()
            / pairs.len() as f32
    };
    let mut model = small_model(3, false);
    let before = margin(&model);
    let mut c = cfg(6);
    c.objective = Objective::Dpo { beta: 0.1 };
    c.lr = 1e-2;
    let mut t = Trainer::new(&mut model, c, TrainData::Pref(pairs.clone()));
    t.run(&mut RunToEnd, None, |_| {}).unwrap();
    let l = &t.state.losses;
    assert!(l.last().unwrap() < &l[0], "DPO loss did not fall: {l:?}");
    let after = margin(t.model);
    assert!(after > before + 5.0, "margin {before} -> {after}");
    let (m, _) = measure(t.model, &[], &pairs);
    assert_eq!(m.pref_accuracy, Some(1.0));
}

#[test]
fn adapter_roundtrip_apply_and_merge() {
    let (tr, he) = toy_sets();
    let dir = tmpdir("adapter");
    let mut model = small_model(4, false);
    let mut t = Trainer::new(&mut model, cfg(1), TrainData::Sft(tr));
    t.run(&mut RunToEnd, None, |_| {}).unwrap();
    let meta = AdapterMeta {
        id: "x".into(),
        base_fingerprint: base_fingerprint(t.model),
        rank: 4,
        alpha: 8.0,
        ..Default::default()
    };
    save_adapter(t.model, &dir, &meta).unwrap();
    let ex = &he[0];
    let targets = answer_targets(&ex.tokens, ex.answer_start);
    let want = t.model.logprobs(&ex.tokens, &targets);

    // Fresh base model + loaded adapter reproduces the trained model exactly.
    let mut fresh = small_model(4, false);
    load_adapter_into(&mut fresh, &dir).unwrap();
    assert_eq!(fresh.logprobs(&ex.tokens, &targets), want);

    // Engine-side application: y_base + adapter.apply == y_lora, and merged W gives the same.
    let ad = LoraAdapter::load(&dir).unwrap();
    let lin = &fresh.blocks[3].mlp.up_proj;
    let x: Vec<f32> = (0..2 * lin.in_dim()).map(|i| (i as f32 * 0.37).sin()).collect();
    let y_lora = lin.forward(&x, 2);
    let mut y = lin.w.forward(&x, 2);
    assert!(ad.apply(&lin.name, &x, 2, &mut y));
    for (a, b) in y.iter().zip(&y_lora) {
        assert!((a - b).abs() < 1e-5);
    }
    let mut w = vec![0.0; lin.out_dim() * lin.in_dim()];
    for o in 0..lin.out_dim() {
        lin.w.row(o, &mut w[o * lin.in_dim()..(o + 1) * lin.in_dim()]);
    }
    assert!(ad.merge_into(&lin.name, &mut w));
    let ym = rust_trainer::tensor::matmul_nt(&x, &w, 2, lin.in_dim(), lin.out_dim());
    for (a, b) in ym.iter().zip(&y_lora) {
        assert!((a - b).abs() < 1e-4);
    }

    // PEFT-compatible tensor names and config.
    let st = rust_trainer::safetensors::SafeTensors::open(dir.join("adapter_model.safetensors")).unwrap();
    assert!(st.contains("base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight"));
    assert!(st.contains("base_model.model.model.layers.3.self_attn.q_proj.lora_B.weight"));
    let cfgj: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("adapter_config.json")).unwrap()).unwrap();
    assert_eq!(cfgj["peft_type"], "LORA");

    // A different architecture refuses the adapter.
    let mut other = Model::random(ModelConfig::tiny_dense(259), 4, false);
    assert!(load_adapter_into(&mut other, &dir).is_err());
    std::fs::remove_dir_all(dir).ok();
}

/// Device whose state the test script changes between calls.
struct Script {
    state: Rc<RefCell<DeviceState>>,
    calls: usize,
    heat_at: usize,
}
impl DeviceMonitor for Script {
    fn state(&mut self) -> io::Result<DeviceState> {
        self.calls += 1;
        if self.calls == self.heat_at {
            self.state.borrow_mut().temp_c = 44.0; // phone heats up mid-run
        }
        Ok(self.state.borrow().clone())
    }
}

#[test]
fn overnight_gate_swap_rollback_and_ledger() {
    let (tr, he) = toy_sets();
    let dir = tmpdir("overnight");
    let registry = Registry::open(&dir).unwrap();
    let job = JobConfig {
        base_model: "tiny".into(),
        train: cfg(4),
        gate: GateConfig { min_examples: 3, ..Default::default() },
        power: PowerPolicy::default(),
        min_train_examples: 5,
    };
    let state = Rc::new(RefCell::new(DeviceState { stage: Stage::Idle, charging: false, battery_pct: 80.0, temp_c: 30.0 }));
    let mut mon = Script { state: state.clone(), calls: 0, heat_at: 25 };
    let st2 = state.clone();
    let mut waits = 0;
    // Each wait: the phone gets plugged in / reaches Deep / cools down.
    let mut wait = move || {
        waits += 1;
        let mut s = st2.borrow_mut();
        s.stage = Stage::Deep;
        s.charging = true;
        s.temp_c = 30.0;
        waits < 10
    };
    let mut model = small_model(5, true);
    let out = run_overnight(
        &mut model,
        &registry,
        &job,
        TrainData::Sft(tr.clone()),
        Heldout { sft: &he, pref: &[] },
        &mut mon,
        &mut wait,
        &mut |_| {},
    )
    .unwrap();
    let id1 = match out {
        Outcome::Accepted { id, .. } => id,
        o => panic!("expected acceptance, got {o:?}"),
    };
    assert_eq!(registry.active().as_deref(), Some(id1.as_str()));
    let events: Vec<String> = registry.read_ledger().into_iter().map(|e| e.event).collect();
    assert_eq!(events, vec!["started", "paused", "accepted"]);
    assert!(!registry.work_dir().exists(), "work dir should be cleaned after a finished run");

    // Second night: the candidate must beat the *current* adapter, which an
    // identical retrain cannot do by 2 %, so it is thrown away.
    let mut model = small_model(5, true);
    let mut mon = Script { state: state.clone(), calls: 0, heat_at: usize::MAX };
    let mut wait = || true;
    let out = run_overnight(
        &mut model,
        &registry,
        &job,
        TrainData::Sft(tr.clone()),
        Heldout { sft: &he, pref: &[] },
        &mut mon,
        &mut wait,
        &mut |_| {},
    )
    .unwrap();
    let id2 = match out {
        Outcome::Rejected { id, .. } => id,
        o => panic!("expected rejection, got {o:?}"),
    };
    assert!(!registry.adapter_dir(&id2).exists());
    assert_eq!(registry.active().as_deref(), Some(id1.as_str()));

    // Not enough data: skipped. Never allowed to run: deferred with checkpoint kept.
    let mut model = small_model(5, true);
    let out = run_overnight(
        &mut model,
        &registry,
        &JobConfig { min_train_examples: 1000, ..job.clone() },
        TrainData::Sft(tr.clone()),
        Heldout { sft: &he, pref: &[] },
        &mut mon,
        &mut wait,
        &mut |_| {},
    )
    .unwrap();
    assert!(matches!(out, Outcome::Skipped { .. }));
    state.borrow_mut().charging = false;
    let mut model = small_model(5, true);
    let mut give_up = || false;
    let out = run_overnight(
        &mut model,
        &registry,
        &job,
        TrainData::Sft(tr),
        Heldout { sft: &he, pref: &[] },
        &mut mon,
        &mut give_up,
        &mut |_| {},
    )
    .unwrap();
    assert!(matches!(out, Outcome::Deferred { .. }), "{out:?}");
    assert!(registry.work_dir().join("checkpoint.safetensors").exists());

    // Rollback returns to the base model.
    assert_eq!(registry.rollback("test").unwrap(), None);
    let events: Vec<String> = registry.read_ledger().into_iter().map(|e| e.event).collect();
    assert_eq!(
        events,
        vec!["started", "paused", "accepted", "started", "rejected", "skipped", "started", "deferred", "rollback"]
    );
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn power_policy_hysteresis() {
    let p = PowerPolicy::default();
    let s = |stage, charging, b, t| DeviceState { stage, charging, battery_pct: b, temp_c: t };
    assert!(p.check(&s(Stage::Deep, true, 80.0, 30.0), false).is_ok());
    assert!(p.check(&s(Stage::Light, true, 80.0, 30.0), true).is_err());
    assert!(p.check(&s(Stage::Deep, false, 80.0, 30.0), true).is_err());
    assert!(p.check(&s(Stage::Deep, true, 40.0, 30.0), true).is_err());
    // 38.5 °C: keeps running, but a paused run does not restart until < 37 °C.
    assert!(p.check(&s(Stage::Deep, true, 80.0, 38.5), true).is_ok());
    assert!(p.check(&s(Stage::Deep, true, 80.0, 38.5), false).is_err());
    assert!(p.check(&s(Stage::Deep, true, 80.0, 40.0), true).is_err());
}
