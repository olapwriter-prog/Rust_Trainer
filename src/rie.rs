//! Bridge to the Rust Inference Engine (`rie-core`).
//!
//! Training has to see the same model the phone runs. This module makes that so:
//!
//! * **Same file, same weights.** The GGUF model is opened by `rie-core` and every
//!   frozen matrix stays a zero-copy `rie_core::tensor::Weight` view into the mmap,
//!   in its native ggml block format (Q4_0, Q4_K, Q6_K, Q8_0, F16, …).
//! * **Same forward kernels.** [`RieFrozen::forward`] calls the engine's own
//!   `kernels::matmul::matmul`, including its on-the-fly int8 quantisation of
//!   activations, so logits during training equal the engine's logits (up to the
//!   engine's f16 KV cache). The input gradient dequantises one row at a time
//!   (straight-through for the activation quantisation, as in QLoRA).
//! * **Same RoPE, norms and prompt.** Hyper-parameters come from
//!   `rie_core::ModelConfig`; prompts are rendered with the engine's
//!   `ChatTemplate` and tokenised with its GGUF tokenizer exactly as `Engine::chat`
//!   does, so the trained tokens are the tokens the engine will see.
//! * **Adapter the engine can load.** [`write_gguf_adapter`] writes a llama.cpp-style
//!   LoRA GGUF (`general.type = adapter`, `adapter.type = lora`,
//!   `blk.N.attn_q.weight.lora_a/lora_b`) with the engine's own GGUF writer.

use crate::attention::{AttnConfig, Attention};
use crate::data::Text;
use crate::frozen::{Frozen, FrozenMatmul};
use crate::linear::{Linear, Lora};
use crate::model::{Block, LayerKind, Mixer, Mlp, Model, ModelConfig};
use crate::tensor::{axpy, Rng};
use crate::tokenizer::{fit, Example, Formatter, Tokenizer};
use rayon::prelude::*;
use rie_core::chat::{ChatTemplate, Message, Role};
use rie_core::engine::{make_pool, EngineConfig};
use rie_core::gguf::{writer::GgufWriter, GgufFile, Value};
use rie_core::kernels::matmul::{formats_for, matmul, QuantActs};
use rie_core::model::{Activation, Arch};
use rie_core::quant::GgmlType;
use rie_core::tensor::Weight;
use rie_core::threadpool::ThreadPool;
use std::io;
use std::path::Path;
use std::sync::Arc;

pub use rie_core;

/// File name of the engine-loadable adapter inside an adapter directory.
pub const GGUF_ADAPTER_FILE: &str = "adapter.gguf";

fn io_err(e: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
}

/// A frozen engine weight. Holds the GGUF mapping alive.
pub struct RieFrozen {
    pub w: Weight,
    pool: Arc<ThreadPool>,
    _file: Arc<GgufFile>,
}

impl RieFrozen {
    pub fn new(w: Weight, pool: Arc<ThreadPool>, file: Arc<GgufFile>) -> Self {
        RieFrozen { w, pool, _file: file }
    }
}

/// Rows dequantised together in the input-gradient pass.
const BWD_ROWS: usize = 32;

impl FrozenMatmul for RieFrozen {
    fn out_dim(&self) -> usize {
        self.w.rows
    }
    fn in_dim(&self) -> usize {
        self.w.cols
    }

    fn forward(&self, x: &[f32], t: usize) -> Vec<f32> {
        let (rows, k) = (self.w.rows, self.w.cols);
        assert_eq!(x.len(), t * k);
        let mut acts = QuantActs::new();
        acts.prepare(&self.pool, x, t, k, &formats_for(&[&self.w]));
        let mut y = vec![0.0; t * rows];
        matmul(&self.pool, &acts, &self.w, &mut y);
        y
    }

    fn backward_input(&self, dy: &[f32], t: usize, dx: &mut [f32]) {
        let (rows, k) = (self.w.rows, self.w.cols);
        assert_eq!(dy.len(), t * rows);
        assert_eq!(dx.len(), t * k);
        let n_chunks = rows.div_ceil(BWD_ROWS);
        let acc = (0..n_chunks)
            .into_par_iter()
            .fold(
                || (vec![0.0f32; t * k], vec![0.0f32; BWD_ROWS * k]),
                |(mut acc, mut buf), c| {
                    let r0 = c * BWD_ROWS;
                    let r1 = (r0 + BWD_ROWS).min(rows);
                    for r in r0..r1 {
                        self.w.dequant_row(r, &mut buf[(r - r0) * k..(r - r0 + 1) * k]);
                    }
                    for ti in 0..t {
                        let dxr = &mut acc[ti * k..(ti + 1) * k];
                        for r in r0..r1 {
                            let g = dy[ti * rows + r];
                            if g != 0.0 {
                                axpy(g, &buf[(r - r0) * k..(r - r0 + 1) * k], dxr);
                            }
                        }
                    }
                    (acc, buf)
                },
            )
            .map(|(acc, _)| acc)
            .reduce(
                || vec![0.0f32; t * k],
                |mut a, b| {
                    crate::tensor::add_into(&mut a, &b);
                    a
                },
            );
        crate::tensor::add_into(dx, &acc);
    }

    fn row(&self, r: usize, out: &mut [f32]) {
        self.w.dequant_row(r, out);
    }

    fn nbytes(&self) -> usize {
        self.w.size_bytes()
    }
}

/// A GGUF model loaded through the engine, ready to train.
pub struct GgufModel {
    pub model: Model,
    pub tokenizer: Arc<rie_core::Tokenizer>,
    pub template: ChatTemplate,
    pub arch_name: String,
}

pub struct GgufOptions {
    /// Engine worker threads (default: the engine's choice of performance cores).
    pub n_threads: Option<usize>,
    /// Pin engine workers to performance cores, as the engine does.
    pub pin_threads: bool,
}

impl Default for GgufOptions {
    fn default() -> Self {
        GgufOptions { n_threads: None, pin_threads: true }
    }
}

/// Architectures whose full forward pass the trainer reproduces.
pub fn trainable_arch(a: Arch) -> bool {
    matches!(a, Arch::Llama | Arch::Qwen2 | Arch::Qwen3)
}

/// Load a GGUF file (path) through `rie-core`.
pub fn load_gguf(path: impl AsRef<Path>, opts: &GgufOptions) -> io::Result<GgufModel> {
    let file = Arc::new(GgufFile::open(path).map_err(io_err)?);
    load_gguf_from(file, opts)
}

/// Load an in-memory GGUF image (tests, synthetic models).
pub fn load_gguf_from(file: Arc<GgufFile>, opts: &GgufOptions) -> io::Result<GgufModel> {
    let tok = Arc::new(rie_core::Tokenizer::from_gguf(&file).map_err(io_err)?);
    let rm = rie_core::model::Model::load(file.clone()).map_err(io_err)?;
    let c = &rm.cfg;
    if !trainable_arch(c.arch) {
        return Err(io_err(format!(
            "architecture '{}' is not trainable yet (supported: llama, mistral, qwen2, qwen3)",
            c.arch_name
        )));
    }
    let plain = c.embd_scale == 1.0
        && c.residual_scale == 1.0
        && c.logit_scale == 1.0
        && c.attn_softcap.is_none()
        && c.final_softcap.is_none()
        && c.swa_window == 0
        && c.act == Activation::Silu
        && (c.attn_scale - 1.0 / (c.head_dim as f32).sqrt()).abs() < 1e-6;
    if !plain {
        return Err(io_err(format!("{}: unsupported scaling/softcap/sliding-window settings", c.arch_name)));
    }
    let pool = make_pool(&EngineConfig { n_threads: opts.n_threads, pin_threads: opts.pin_threads, ..Default::default() });
    let fz = |w: &Weight| -> Frozen { Arc::new(RieFrozen::new(w.clone(), pool.clone(), file.clone())) };

    let d = c.n_embd;
    let tied = std::ptr::eq(rm.output.data().as_ptr(), rm.token_embd.data().as_ptr());
    let out_w = if rm.output.rows > c.n_vocab { rm.output.slice_rows(0, c.n_vocab) } else { rm.output.clone() };
    let embed = fz(&rm.token_embd);
    let lm_head = if tied && rm.output.rows == c.n_vocab { embed.clone() } else { fz(&out_w) };
    let attn = AttnConfig {
        n_heads: c.n_head,
        n_kv_heads: c.n_head_kv,
        head_dim: c.head_dim,
        rot_dim: c.rope_dim,
        rope_theta: c.rope_base,
        qk_norm: rm.layers.first().map(|l| l.q_norm.is_some()).unwrap_or(false),
        output_gate: false,
        rope_neox: c.rope_neox,
        rope_freq_scale: c.rope_freq_scale,
        kv_f16: true,
    };
    let mut blocks = Vec::with_capacity(c.n_layer);
    for (i, l) in rm.layers.iter().enumerate() {
        if l.post_attn_norm.is_some() || l.post_ffn_norm.is_some() {
            return Err(io_err("post-norm layers are not supported"));
        }
        let p = |s: &str| format!("blk.{i}.{s}");
        blocks.push(Block {
            input_norm: l.attn_norm.clone(),
            mixer: Mixer::Attn(Attention {
                cfg: attn.clone(),
                q_proj: Linear::new(p("attn_q"), fz(&l.wq), l.bq.clone()),
                k_proj: Linear::new(p("attn_k"), fz(&l.wk), l.bk.clone()),
                v_proj: Linear::new(p("attn_v"), fz(&l.wv), l.bv.clone()),
                o_proj: Linear::new(p("attn_output"), fz(&l.wo), None),
                q_norm: l.q_norm.clone(),
                k_norm: l.k_norm.clone(),
                eps: c.rms_eps,
                norm_offset: 0.0,
            }),
            post_norm: l.ffn_norm.clone(),
            mlp: Mlp {
                gate_proj: Linear::new(p("ffn_gate"), fz(&l.w_gate), None),
                up_proj: Linear::new(p("ffn_up"), fz(&l.w_up), None),
                down_proj: Linear::new(p("ffn_down"), fz(&l.w_down), None),
            },
        });
    }
    let cfg = ModelConfig {
        arch: c.arch_name.clone(),
        vocab_size: c.n_vocab,
        hidden_size: d,
        intermediate_size: c.n_ff,
        layer_kinds: vec![LayerKind::FullAttention; c.n_layer],
        attn,
        delta: None,
        rms_eps: c.rms_eps,
        norm_offset: 0.0,
        tie_embeddings: tied,
        attn_bias: rm.layers.first().map(|l| l.bq.is_some()).unwrap_or(false),
    };
    let model = Model {
        cfg,
        embed,
        blocks,
        final_norm: rm.output_norm.clone(),
        lm_head,
        checkpointing: true,
        logits_chunk: 64,
        rope_factors: rm.rope_factors.clone(),
    };
    let template = ChatTemplate::detect(&tok);
    Ok(GgufModel { model, tokenizer: tok, template, arch_name: c.arch_name.clone() })
}

/// The engine's GGUF tokenizer behind the trainer's [`Tokenizer`] trait.
/// Plain text is encoded without special-token parsing.
pub struct RieTokenizer(pub Arc<rie_core::Tokenizer>);

impl Tokenizer for RieTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.0.encode(text, false, false)
    }
    fn decode(&self, ids: &[u32]) -> String {
        self.0.decode(ids, true)
    }
    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }
    fn special(&self, name: &str) -> Option<u32> {
        self.0.token_id(name)
    }
}

/// Builds examples exactly as `rie_core::Engine::chat` builds its prompt:
/// `template.render(messages, true)` tokenised with BOS and special-token parsing.
/// The answer is the plain-text reply followed by the template's end-of-turn token,
/// which is where the engine stops generating.
pub struct EngineChat {
    pub tok: Arc<rie_core::Tokenizer>,
    pub template: ChatTemplate,
    pub end_token: u32,
    pub max_len: usize,
}

impl EngineChat {
    pub fn new(tok: Arc<rie_core::Tokenizer>, template: ChatTemplate, max_len: usize) -> io::Result<Self> {
        let named = match template {
            ChatTemplate::ChatMl => Some("<|im_end|>"),
            ChatTemplate::Llama3 => Some("<|eot_id|>"),
            ChatTemplate::Gemma => Some("<end_of_turn>"),
            ChatTemplate::Phi3 => Some("<|end|>"),
            ChatTemplate::Granite => Some("<|end_of_text|>"),
            ChatTemplate::Mistral | ChatTemplate::Raw => None,
        };
        let end_token = named
            .and_then(|n| tok.token_id(n))
            .or(tok.eot)
            .or(tok.eos)
            .ok_or_else(|| io_err("tokenizer has no end-of-turn token"))?;
        Ok(EngineChat { tok, template, end_token, max_len })
    }

    pub fn prompt_tokens(&self, system: Option<&str>, prompt: &Text) -> Vec<u32> {
        let mut msgs = Vec::new();
        if let Some(s) = system {
            msgs.push(Message::new(Role::System, s));
        }
        match prompt {
            Text::Ids(ids) => return ids.clone(),
            Text::Str(s) => msgs.push(Message::new(Role::User, s.as_str())),
            Text::Turns(turns) => {
                for (role, content) in turns {
                    msgs.push(Message::new(Role::parse(role).unwrap_or(Role::User), content.as_str()));
                }
            }
        }
        let rendered = self.template.render(&msgs, true);
        self.tok.encode(&rendered, true, true)
    }
}

impl Formatter for EngineChat {
    fn example(&self, system: Option<&str>, prompt: &Text, answer: &Text) -> Option<Example> {
        let p = self.prompt_tokens(system, prompt);
        let a = match answer {
            Text::Ids(ids) => ids.clone(),
            Text::Str(s) => {
                let mut v = self.tok.encode(s, false, false);
                v.push(self.end_token);
                v
            }
            Text::Turns(_) => return None,
        };
        fit(p, a, self.max_len)
    }
}

/// Write the model's LoRA adapters as a llama.cpp-style LoRA GGUF:
/// `general.architecture`, `general.type = "adapter"`, `adapter.type = "lora"`,
/// `adapter.lora.alpha`, and per module `<name>.weight.lora_a` (ggml ne `[in, r]`)
/// and `<name>.weight.lora_b` (ne `[r, out]`), F32.
pub fn write_gguf_adapter(model: &Model, path: impl AsRef<Path>) -> io::Result<()> {
    let lins = model.lora_linears();
    let alpha = lins.first().and_then(|l| l.lora.as_ref()).map(|l| l.alpha).ok_or_else(|| io_err("no LoRA adapters"))?;
    let mut w = GgufWriter::new();
    w.kv("general.architecture", Value::String(model.cfg.arch.clone()));
    w.kv("general.type", Value::String("adapter".into()));
    w.kv("adapter.type", Value::String("lora".into()));
    w.kv("adapter.lora.alpha", Value::F32(alpha));
    for l in &lins {
        let lo = l.lora.as_ref().unwrap();
        let f32b = |v: &[f32]| rie_core::quant::quantize::f32(v);
        w.tensor(&format!("{}.weight.lora_a", l.name), &[l.in_dim() as u64, lo.rank as u64], GgmlType::F32, f32b(&lo.a));
        w.tensor(&format!("{}.weight.lora_b", l.name), &[lo.rank as u64, l.out_dim() as u64], GgmlType::F32, f32b(&lo.b));
    }
    let path = path.as_ref();
    let tmp = path.with_extension("gguf.tmp");
    std::fs::write(&tmp, w.to_bytes())?;
    std::fs::rename(tmp, path)
}

/// Load a LoRA GGUF (as written by [`write_gguf_adapter`] or llama.cpp's
/// `convert_lora_to_gguf.py` with F32/F16 tensors) into the model. Returns the
/// number of adapted modules.
pub fn read_gguf_adapter(model: &mut Model, path: impl AsRef<Path>) -> io::Result<usize> {
    let g = GgufFile::open(path).map_err(io_err)?;
    if g.get_str("adapter.type") != Some("lora") {
        return Err(io_err("not a LoRA adapter GGUF"));
    }
    if let Some(a) = g.get_str("general.architecture") {
        if a != model.cfg.arch {
            return Err(io_err(format!("adapter is for '{a}', model is '{}'", model.cfg.arch)));
        }
    }
    let alpha = g.get_f32("adapter.lora.alpha").unwrap_or(0.0);
    let read = |name: &str| -> io::Result<Option<(Vec<f32>, usize, usize)>> {
        let Some(t) = g.tensor(name) else { return Ok(None) };
        // SAFETY: `g` outlives the temporary view.
        let w = unsafe { Weight::from_raw(t.ty, t.n_rows(), t.row_len(), g.tensor_data(t)) };
        Ok(Some((w.to_f32(), t.n_rows(), t.row_len())))
    };
    let mut rng = Rng::new(0);
    let mut n = 0;
    for b in &mut model.blocks {
        for l in b.linears_mut() {
            l.lora = None;
            let (Some((a, ar, ac)), Some((bw, br, bc))) =
                (read(&format!("{}.weight.lora_a", l.name))?, read(&format!("{}.weight.lora_b", l.name))?)
            else {
                continue;
            };
            if ac != l.in_dim() || br != l.out_dim() || bc != ar {
                return Err(io_err(format!("{}: adapter shape mismatch", l.name)));
            }
            let mut lo = Lora::new(ar, if alpha > 0.0 { alpha } else { ar as f32 }, l.in_dim(), l.out_dim(), &mut rng);
            lo.a = a;
            lo.b = bw;
            l.lora = Some(lo);
            l.lora_enabled = true;
            n += 1;
        }
    }
    if n == 0 {
        return Err(io_err("adapter matched no modules"));
    }
    Ok(n)
}
