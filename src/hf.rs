//! Load Qwen2.5 / Qwen3 / Qwen3.5 (and Qwen3-Next) checkpoints from a Hugging
//! Face directory (`config.json` + `*.safetensors`).
//!
//! Frozen matrices are converted to f32 or Q8_0 one tensor at a time. On the
//! phone the engine's own compressed weights are used instead, through
//! [`crate::frozen::FrozenMatmul`]; this loader is for the Mac/VM path and for
//! verifying numerics against the reference implementation.

use crate::attention::{AttnConfig, Attention};
use crate::deltanet::{DeltaNet, DeltaNetConfig};
use crate::frozen::{Frozen, FrozenWeight};
use crate::linear::Linear;
use crate::model::{Block, LayerKind, Mixer, Mlp, Model, ModelConfig};
use crate::safetensors::SafeTensors;
use serde_json::Value;
use std::collections::HashMap;
use std::io;
use std::path::Path;

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn get_usize(v: &Value, k: &str) -> io::Result<usize> {
    v.get(k).and_then(Value::as_u64).map(|x| x as usize).ok_or_else(|| bad(format!("config.json: missing {k}")))
}
fn get_f32_or(v: &Value, k: &str, d: f32) -> f32 {
    v.get(k).and_then(Value::as_f64).map(|x| x as f32).unwrap_or(d)
}

/// Parse `config.json` into a [`ModelConfig`].
pub fn parse_config(root: &Value) -> io::Result<ModelConfig> {
    let tc = root.get("text_config").unwrap_or(root);
    let model_type = tc
        .get("model_type")
        .or_else(|| root.get("model_type"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let hidden = get_usize(tc, "hidden_size")?;
    let n_layers = get_usize(tc, "num_hidden_layers")?;
    let n_heads = get_usize(tc, "num_attention_heads")?;
    let n_kv = tc.get("num_key_value_heads").and_then(Value::as_u64).map(|x| x as usize).unwrap_or(n_heads);
    let head_dim = tc.get("head_dim").and_then(Value::as_u64).map(|x| x as usize).unwrap_or(hidden / n_heads);
    let rope_params = tc.get("rope_parameters").or_else(|| tc.get("rope_scaling"));
    let rope_theta = tc
        .get("rope_theta")
        .and_then(Value::as_f64)
        .or_else(|| rope_params.and_then(|r| r.get("rope_theta")).and_then(Value::as_f64))
        .unwrap_or(1_000_000.0) as f32;
    let tie = root
        .get("tie_word_embeddings")
        .or_else(|| tc.get("tie_word_embeddings"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let vocab = get_usize(tc, "vocab_size")?;
    let eps = get_f32_or(tc, "rms_norm_eps", 1e-6);
    let inter = get_usize(tc, "intermediate_size")?;

    if tc.get("num_experts").and_then(Value::as_u64).unwrap_or(0) > 0 {
        return Err(bad("mixture-of-experts checkpoints are not supported"));
    }

    let hybrid = model_type.starts_with("qwen3_5") || model_type.starts_with("qwen3_next");
    if hybrid {
        let kinds: Vec<LayerKind> = match tc.get("layer_types").and_then(Value::as_array) {
            Some(a) => a
                .iter()
                .map(|x| match x.as_str() {
                    Some("full_attention") => Ok(LayerKind::FullAttention),
                    Some("linear_attention") => Ok(LayerKind::LinearAttention),
                    other => Err(bad(format!("unknown layer type {other:?}"))),
                })
                .collect::<io::Result<_>>()?,
            None => {
                let every = tc.get("full_attention_interval").and_then(Value::as_u64).unwrap_or(4) as usize;
                (0..n_layers)
                    .map(|i| if (i + 1) % every == 0 { LayerKind::FullAttention } else { LayerKind::LinearAttention })
                    .collect()
            }
        };
        let prf = tc
            .get("partial_rotary_factor")
            .and_then(Value::as_f64)
            .or_else(|| rope_params.and_then(|r| r.get("partial_rotary_factor")).and_then(Value::as_f64))
            .unwrap_or(0.25) as f32;
        let rot = ((head_dim as f32 * prf) as usize) & !1;
        Ok(ModelConfig {
            arch: "qwen3_5".into(),
            vocab_size: vocab,
            hidden_size: hidden,
            intermediate_size: inter,
            layer_kinds: kinds,
            attn: AttnConfig {
                n_heads,
                n_kv_heads: n_kv,
                head_dim,
                rot_dim: rot,
                rope_theta,
                qk_norm: true,
                output_gate: tc.get("attn_output_gate").and_then(Value::as_bool).unwrap_or(true),
                rope_neox: true,
                rope_freq_scale: 1.0,
                kv_f16: false,
            },
            delta: Some(DeltaNetConfig {
                n_k_heads: get_usize(tc, "linear_num_key_heads")?,
                n_v_heads: get_usize(tc, "linear_num_value_heads")?,
                k_head_dim: get_usize(tc, "linear_key_head_dim")?,
                v_head_dim: get_usize(tc, "linear_value_head_dim")?,
                conv_kernel: tc.get("linear_conv_kernel_dim").and_then(Value::as_u64).unwrap_or(4) as usize,
                bptt_chunk: 16,
            }),
            rms_eps: eps,
            norm_offset: 1.0,
            tie_embeddings: tie,
            attn_bias: false,
        })
    } else {
        let qwen3 = model_type == "qwen3";
        if !(qwen3 || model_type == "qwen2") {
            return Err(bad(format!("unsupported model_type {model_type:?} (qwen2, qwen3, qwen3_5, qwen3_next)")));
        }
        Ok(ModelConfig {
            arch: model_type.clone(),
            vocab_size: vocab,
            hidden_size: hidden,
            intermediate_size: inter,
            layer_kinds: vec![LayerKind::FullAttention; n_layers],
            attn: AttnConfig {
                n_heads,
                n_kv_heads: n_kv,
                head_dim,
                rot_dim: head_dim,
                rope_theta,
                qk_norm: qwen3,
                output_gate: false,
                rope_neox: true,
                rope_freq_scale: 1.0,
                kv_f16: false,
            },
            delta: None,
            rms_eps: eps,
            norm_offset: 0.0,
            tie_embeddings: tie,
            attn_bias: !qwen3,
        })
    }
}

struct Store {
    files: Vec<SafeTensors>,
    index: HashMap<String, usize>,
}

impl Store {
    fn open(dir: &Path) -> io::Result<Self> {
        let mut files = Vec::new();
        let mut paths: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors"))
            .filter(|p| !p.file_name().and_then(|n| n.to_str()).unwrap_or("").starts_with("adapter"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(bad(format!("no .safetensors files in {}", dir.display())));
        }
        let mut index = HashMap::new();
        for p in paths {
            let st = SafeTensors::open(&p)?;
            for k in st.tensors.keys() {
                index.insert(k.clone(), files.len());
            }
            files.push(st);
        }
        Ok(Store { files, index })
    }
    fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }
    fn get(&self, name: &str) -> io::Result<(Vec<f32>, Vec<usize>)> {
        let i = *self.index.get(name).ok_or_else(|| bad(format!("missing tensor {name}")))?;
        self.files[i].read_f32(name)
    }
    fn vec(&self, name: &str, len: usize) -> io::Result<Vec<f32>> {
        let (v, _) = self.get(name)?;
        if v.len() != len {
            return Err(bad(format!("{name}: expected {len} values, got {}", v.len())));
        }
        Ok(v)
    }
}

fn to_frozen(w: Vec<f32>, out: usize, inp: usize, q8: bool) -> Frozen {
    if q8 {
        FrozenWeight::q8_0(&w, out, inp).into_arc()
    } else {
        FrozenWeight::f32(w, out, inp).into_arc()
    }
}

/// Take rows `[r0, r0+n)` of a `[rows × cols]` matrix.
fn rows(w: &[f32], cols: usize, r0: usize, n: usize) -> &[f32] {
    &w[r0 * cols..(r0 + n) * cols]
}

pub struct LoadOptions {
    /// Store frozen matrices as Q8_0 (≈ 1.1 bytes/param) instead of f32.
    pub quantize_q8: bool,
}

pub fn load_model(dir: impl AsRef<Path>, opts: &LoadOptions) -> io::Result<Model> {
    let dir = dir.as_ref();
    let root: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?).map_err(|e| bad(e.to_string()))?;
    let cfg = parse_config(&root)?;
    let st = Store::open(dir)?;
    let q8 = opts.quantize_q8;
    let d = cfg.hidden_size;
    let prefix = if st.has("model.language_model.embed_tokens.weight") { "model.language_model." } else { "model." };

    let lin = |name: &str, out: usize, inp: usize, bias: bool| -> io::Result<Linear> {
        let (w, shape) = st.get(&format!("{name}.weight"))?;
        if shape != vec![out, inp] {
            return Err(bad(format!("{name}: shape {shape:?}, expected [{out}, {inp}]")));
        }
        let b = if bias && st.has(&format!("{name}.bias")) { Some(st.vec(&format!("{name}.bias"), out)?) } else { None };
        Ok(Linear::new(name, to_frozen(w, out, inp, q8), b))
    };

    let (emb, eshape) = st.get(&format!("{prefix}embed_tokens.weight"))?;
    if eshape != vec![cfg.vocab_size, d] {
        return Err(bad(format!("embed_tokens shape {eshape:?}")));
    }
    let embed = to_frozen(emb, cfg.vocab_size, d, q8);
    let lm_head = if cfg.tie_embeddings || !st.has("lm_head.weight") {
        embed.clone()
    } else {
        let (w, _) = st.get("lm_head.weight")?;
        to_frozen(w, cfg.vocab_size, d, q8)
    };

    let a = cfg.attn.clone();
    let mut blocks = Vec::with_capacity(cfg.n_layers());
    for (i, kind) in cfg.layer_kinds.iter().enumerate() {
        let p = format!("{prefix}layers.{i}");
        let mixer = match kind {
            LayerKind::FullAttention => {
                let qo = a.n_heads * a.head_dim * if a.output_gate { 2 } else { 1 };
                let kvo = a.n_kv_heads * a.head_dim;
                let sa = format!("{p}.self_attn");
                Mixer::Attn(Attention {
                    cfg: a.clone(),
                    q_proj: lin(&format!("{sa}.q_proj"), qo, d, cfg.attn_bias)?,
                    k_proj: lin(&format!("{sa}.k_proj"), kvo, d, cfg.attn_bias)?,
                    v_proj: lin(&format!("{sa}.v_proj"), kvo, d, cfg.attn_bias)?,
                    o_proj: lin(&format!("{sa}.o_proj"), d, a.n_heads * a.head_dim, false)?,
                    q_norm: if a.qk_norm { Some(st.vec(&format!("{sa}.q_norm.weight"), a.head_dim)?) } else { None },
                    k_norm: if a.qk_norm { Some(st.vec(&format!("{sa}.k_norm.weight"), a.head_dim)?) } else { None },
                    eps: cfg.rms_eps,
                    norm_offset: cfg.norm_offset,
                })
            }
            LayerKind::LinearAttention => {
                let dc = cfg.delta.clone().unwrap();
                let la = format!("{p}.linear_attn");
                let (nk, nv, dk, dv) = (dc.n_k_heads, dc.n_v_heads, dc.k_head_dim, dc.v_head_dim);
                let cd = dc.conv_dim();
                let (qkv, z, b, a_) = if st.has(&format!("{la}.in_proj_qkv.weight")) {
                    (
                        lin(&format!("{la}.in_proj_qkv"), cd, d, false)?,
                        lin(&format!("{la}.in_proj_z"), nv * dv, d, false)?,
                        lin(&format!("{la}.in_proj_b"), nv, d, false)?,
                        lin(&format!("{la}.in_proj_a"), nv, d, false)?,
                    )
                } else {
                    // Qwen3-Next fused layout: per key-head group g,
                    // in_proj_qkvz rows = [q dk | k dk | v r·dv | z r·dv], in_proj_ba rows = [b r | a r].
                    let r = nv / nk;
                    let (w, _) = st.get(&format!("{la}.in_proj_qkvz.weight"))?;
                    let grp = 2 * dk + 2 * r * dv;
                    let (mut q, mut k, mut v, mut zz) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                    for g in 0..nk {
                        let base = g * grp;
                        q.extend_from_slice(rows(&w, d, base, dk));
                        k.extend_from_slice(rows(&w, d, base + dk, dk));
                        v.extend_from_slice(rows(&w, d, base + 2 * dk, r * dv));
                        zz.extend_from_slice(rows(&w, d, base + 2 * dk + r * dv, r * dv));
                    }
                    let mut qkv = q;
                    qkv.extend(k);
                    qkv.extend(v);
                    let (wba, _) = st.get(&format!("{la}.in_proj_ba.weight"))?;
                    let (mut bb, mut aa) = (Vec::new(), Vec::new());
                    for g in 0..nk {
                        bb.extend_from_slice(rows(&wba, d, g * 2 * r, r));
                        aa.extend_from_slice(rows(&wba, d, g * 2 * r + r, r));
                    }
                    (
                        Linear::new(format!("{la}.in_proj_qkv"), to_frozen(qkv, cd, d, q8), None),
                        Linear::new(format!("{la}.in_proj_z"), to_frozen(zz, nv * dv, d, q8), None),
                        Linear::new(format!("{la}.in_proj_b"), to_frozen(bb, nv, d, q8), None),
                        Linear::new(format!("{la}.in_proj_a"), to_frozen(aa, nv, d, q8), None),
                    )
                };
                Mixer::Delta(DeltaNet {
                    in_proj_qkv: qkv,
                    in_proj_z: z,
                    in_proj_b: b,
                    in_proj_a: a_,
                    conv_w: st.vec(&format!("{la}.conv1d.weight"), cd * dc.conv_kernel)?,
                    dt_bias: st.vec(&format!("{la}.dt_bias"), nv)?,
                    a_log: st.vec(&format!("{la}.A_log"), nv)?,
                    norm_w: st.vec(&format!("{la}.norm.weight"), dv)?,
                    out_proj: lin(&format!("{la}.out_proj"), d, nv * dv, false)?,
                    eps: cfg.rms_eps,
                    cfg: dc,
                })
            }
        };
        blocks.push(Block {
            input_norm: st.vec(&format!("{p}.input_layernorm.weight"), d)?,
            mixer,
            post_norm: st.vec(&format!("{p}.post_attention_layernorm.weight"), d)?,
            mlp: Mlp {
                gate_proj: lin(&format!("{p}.mlp.gate_proj"), cfg.intermediate_size, d, false)?,
                up_proj: lin(&format!("{p}.mlp.up_proj"), cfg.intermediate_size, d, false)?,
                down_proj: lin(&format!("{p}.mlp.down_proj"), d, cfg.intermediate_size, false)?,
            },
        });
    }
    let final_norm = st.vec(&format!("{prefix}norm.weight"), d)?;
    Ok(Model { cfg, embed, blocks, final_norm, lm_head, checkpointing: true, logits_chunk: 64, rope_factors: None })
}

/// Rough resource estimate for a config (used by `rust_trainer info`).
pub struct Estimate {
    pub base_params: usize,
    pub lora_params: usize,
    pub forward_flops_per_token: f64,
}

pub fn estimate(cfg: &ModelConfig, rank: usize, targets: &[&str]) -> Estimate {
    let d = cfg.hidden_size;
    let mut base = cfg.vocab_size * d * if cfg.tie_embeddings { 1 } else { 2 };
    let mut lora = 0;
    let mut matmul = 0usize;
    let mut add = |name: &str, out: usize, inp: usize, base: &mut usize, matmul: &mut usize| {
        *base += out * inp;
        *matmul += out * inp;
        if targets.contains(&name) {
            lora += rank * (out + inp);
        }
    };
    let a = &cfg.attn;
    for k in &cfg.layer_kinds {
        match k {
            LayerKind::FullAttention => {
                let qo = a.n_heads * a.head_dim * if a.output_gate { 2 } else { 1 };
                add("q_proj", qo, d, &mut base, &mut matmul);
                add("k_proj", a.n_kv_heads * a.head_dim, d, &mut base, &mut matmul);
                add("v_proj", a.n_kv_heads * a.head_dim, d, &mut base, &mut matmul);
                add("o_proj", d, a.n_heads * a.head_dim, &mut base, &mut matmul);
            }
            LayerKind::LinearAttention => {
                let dc = cfg.delta.as_ref().unwrap();
                add("in_proj_qkv", dc.conv_dim(), d, &mut base, &mut matmul);
                add("in_proj_z", dc.value_dim(), d, &mut base, &mut matmul);
                add("in_proj_b", dc.n_v_heads, d, &mut base, &mut matmul);
                add("in_proj_a", dc.n_v_heads, d, &mut base, &mut matmul);
                add("out_proj", d, dc.value_dim(), &mut base, &mut matmul);
            }
        }
        add("gate_proj", cfg.intermediate_size, d, &mut base, &mut matmul);
        add("up_proj", cfg.intermediate_size, d, &mut base, &mut matmul);
        add("down_proj", d, cfg.intermediate_size, &mut base, &mut matmul);
    }
    matmul += cfg.vocab_size * d;
    Estimate { base_params: base, lora_params: lora, forward_flops_per_token: 2.0 * matmul as f64 }
}
