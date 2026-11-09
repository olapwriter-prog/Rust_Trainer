//! LoRA adapter format.
//!
//! An adapter is a directory saved next to the model:
//! ```text
//! adapter_model.safetensors   PEFT tensor names, F32:
//!     base_model.model.<module>.lora_A.weight  [r × in]
//!     base_model.model.<module>.lora_B.weight  [out × r]
//! adapter_config.json         PEFT config (peft_type=LORA, r, lora_alpha, target_modules)
//! trainer_meta.json           provenance: base model fingerprint, data, steps, eval report
//! ```
//! This is the layout Hugging Face PEFT writes and llama.cpp's
//! `convert_lora_to_gguf.py` reads, so an adapter can also be converted for
//! llama.cpp. At inference `agentos-llm` adds `(α/r)·B·(A·x)` to each adapted
//! projection; [`LoraAdapter::apply`] is the reference implementation of that.

use crate::linear::Lora;
use crate::model::Model;
use crate::safetensors::{write_f32, SafeTensors};
use crate::tensor::{matmul_nt, Rng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::io;
use std::path::Path;

pub const WEIGHTS_FILE: &str = "adapter_model.safetensors";
pub const CONFIG_FILE: &str = "adapter_config.json";
pub const META_FILE: &str = "trainer_meta.json";

fn peft_name(module: &str, which: &str) -> String {
    format!("base_model.model.{module}.lora_{which}.weight")
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AdapterMeta {
    pub id: String,
    pub base_model: String,
    /// Fingerprint of the base model config + shapes, checked on load.
    pub base_fingerprint: String,
    pub rank: usize,
    pub alpha: f32,
    pub targets: Vec<String>,
    pub created_unix: u64,
    pub train_steps: u64,
    pub train_examples: usize,
    pub final_train_loss: Option<f32>,
    pub eval: Option<serde_json::Value>,
}

/// Stable fingerprint of the architecture (not the weights) so an adapter is
/// never applied to a different model shape.
pub fn base_fingerprint(model: &Model) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |s: &str| {
        for b in s.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    };
    feed(&serde_json::to_string(&model.cfg).unwrap_or_default());
    for b in &model.blocks {
        for l in b.linears() {
            feed(&format!("{}:{}x{};", l.name, l.out_dim(), l.in_dim()));
        }
    }
    format!("{h:016x}")
}

pub fn save_adapter(model: &Model, dir: impl AsRef<Path>, meta: &AdapterMeta) -> io::Result<()> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir)?;
    let lins = model.lora_linears();
    if lins.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "model has no LoRA adapters"));
    }
    let mut tensors = Vec::new();
    let mut targets: Vec<String> = Vec::new();
    let (mut rank, mut alpha) = (0, 0.0);
    for l in &lins {
        let lo = l.lora.as_ref().unwrap();
        rank = lo.rank;
        alpha = lo.alpha;
        tensors.push((peft_name(&l.name, "A"), vec![lo.rank, l.in_dim()], lo.a.as_slice()));
        tensors.push((peft_name(&l.name, "B"), vec![l.out_dim(), lo.rank], lo.b.as_slice()));
        let short = l.name.rsplit('.').next().unwrap_or("").to_string();
        if !targets.contains(&short) {
            targets.push(short);
        }
    }
    let mut md = BTreeMap::new();
    md.insert("format".to_string(), "pt".to_string());
    md.insert("base_fingerprint".to_string(), meta.base_fingerprint.clone());
    write_f32(dir.join(WEIGHTS_FILE), &tensors, &md)?;
    let cfg = serde_json::json!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": meta.base_model,
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": false,
        "use_rslora": false,
        "target_modules": targets,
    });
    write_json(dir.join(CONFIG_FILE), &cfg)?;
    write_json(dir.join(META_FILE), meta)?;
    // Models loaded from GGUF through the engine use GGUF tensor names (`blk.N.…`):
    // also write the llama.cpp-style LoRA GGUF the engine loads.
    #[cfg(feature = "rie")]
    if lins.iter().all(|l| l.name.starts_with("blk.")) {
        crate::rie::write_gguf_adapter(model, dir.join(crate::rie::GGUF_ADAPTER_FILE))?;
    }
    Ok(())
}

pub fn write_json<T: Serialize + ?Sized>(path: impl AsRef<Path>, v: &T) -> io::Result<()> {
    let path = path.as_ref();
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(v)?)?;
    std::fs::rename(tmp, path)
}

pub fn read_meta(dir: impl AsRef<Path>) -> io::Result<AdapterMeta> {
    let s = std::fs::read(dir.as_ref().join(META_FILE))?;
    serde_json::from_slice(&s).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Load an adapter into the model's linears (replacing any existing adapters).
pub fn load_adapter_into(model: &mut Model, dir: impl AsRef<Path>) -> io::Result<AdapterMeta> {
    let dir = dir.as_ref();
    let meta = read_meta(dir)?;
    let fp = base_fingerprint(model);
    if !meta.base_fingerprint.is_empty() && meta.base_fingerprint != fp {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("adapter was trained for base {} but this model is {}", meta.base_fingerprint, fp),
        ));
    }
    let st = SafeTensors::open(dir.join(WEIGHTS_FILE))?;
    let mut rng = Rng::new(0);
    let mut loaded = 0;
    for b in &mut model.blocks {
        for l in b.linears_mut() {
            l.lora = None;
            let (an, bn) = (peft_name(&l.name, "A"), peft_name(&l.name, "B"));
            if !st.contains(&an) {
                continue;
            }
            let (a, ashape) = st.read_f32(&an)?;
            let (bw, bshape) = st.read_f32(&bn)?;
            let r = ashape[0];
            if ashape != vec![r, l.in_dim()] || bshape != vec![l.out_dim(), r] {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{}: adapter shape mismatch", l.name)));
            }
            let mut lo = Lora::new(r, meta.alpha, l.in_dim(), l.out_dim(), &mut rng);
            lo.a = a;
            lo.b = bw;
            l.lora = Some(lo);
            l.lora_enabled = true;
            loaded += 1;
        }
    }
    if loaded == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "adapter matched no modules"));
    }
    Ok(meta)
}

/// Standalone adapter for an inference engine: `y += (α/r)·B·(A·x)` per module.
pub struct LoraAdapter {
    pub scale: f32,
    /// module name → (A `[r×in]`, B `[out×r]`, r, in, out)
    pub modules: HashMap<String, (Vec<f32>, Vec<f32>, usize, usize, usize)>,
}

impl LoraAdapter {
    pub fn load(dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = dir.as_ref();
        let meta = read_meta(dir)?;
        let st = SafeTensors::open(dir.join(WEIGHTS_FILE))?;
        let mut modules = HashMap::new();
        for name in st.tensors.keys() {
            if let Some(module) = name.strip_prefix("base_model.model.").and_then(|s| s.strip_suffix(".lora_A.weight")) {
                let (a, ash) = st.read_f32(name)?;
                let (b, bsh) = st.read_f32(&peft_name(module, "B"))?;
                modules.insert(module.to_string(), (a, b, ash[0], ash[1], bsh[0]));
            }
        }
        Ok(LoraAdapter { scale: meta.alpha / meta.rank.max(1) as f32, modules })
    }

    /// Add the adapter's contribution for `module` to `y` (`[t × out]`) given input `x` (`[t × in]`).
    /// Returns false if the module is not adapted.
    pub fn apply(&self, module: &str, x: &[f32], t: usize, y: &mut [f32]) -> bool {
        let Some((a, b, r, inp, out)) = self.modules.get(module) else { return false };
        let h = matmul_nt(x, a, t, *inp, *r);
        let d = matmul_nt(&h, b, t, *r, *out);
        for (yv, dv) in y.iter_mut().zip(&d) {
            *yv += self.scale * dv;
        }
        true
    }

    /// Merge into a dense `[out × in]` weight: `W += (α/r)·B·A`.
    pub fn merge_into(&self, module: &str, w: &mut [f32]) -> bool {
        let Some((a, b, r, inp, out)) = self.modules.get(module) else { return false };
        for o in 0..*out {
            for k in 0..*r {
                let c = self.scale * b[o * r + k];
                if c != 0.0 {
                    crate::tensor::axpy(c, &a[k * inp..(k + 1) * inp], &mut w[o * inp..(o + 1) * inp]);
                }
            }
        }
        true
    }
}
