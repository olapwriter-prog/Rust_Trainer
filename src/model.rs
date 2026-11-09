//! Qwen2.5 / Qwen3.5 decoder with frozen base weights, LoRA adapters and a
//! full backward pass with per-block activation recomputation (gradient
//! checkpointing).

use crate::attention::{Attention, AttnCache, AttnConfig};
use crate::deltanet::{DeltaCache, DeltaNet, DeltaNetConfig};
use crate::frozen::{Frozen, FrozenWeight};
use crate::linear::Linear;
use crate::ops::{rmsnorm, rmsnorm_backward, softmax_row, Rope};
use crate::tensor::{silu, silu_grad, Rng};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerKind {
    FullAttention,
    LinearAttention,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    /// "qwen2" or "qwen3_5" (informational; the layer list drives the graph).
    pub arch: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_kinds: Vec<LayerKind>,
    pub attn: AttnConfig,
    pub delta: Option<DeltaNetConfig>,
    pub rms_eps: f32,
    /// 0 for Qwen2.5 RMSNorm, 1 for Qwen3.5's zero-centred RMSNorm (`x̂·(1+w)`).
    pub norm_offset: f32,
    pub tie_embeddings: bool,
    pub attn_bias: bool,
}

impl ModelConfig {
    pub fn n_layers(&self) -> usize {
        self.layer_kinds.len()
    }

    /// A small dense Qwen2.5-shaped config for tests.
    pub fn tiny_dense(vocab: usize) -> Self {
        ModelConfig {
            arch: "qwen2".into(),
            vocab_size: vocab,
            hidden_size: 32,
            intermediate_size: 48,
            layer_kinds: vec![LayerKind::FullAttention; 2],
            attn: AttnConfig {
                n_heads: 4,
                n_kv_heads: 2,
                head_dim: 8,
                rot_dim: 8,
                rope_theta: 10000.0,
                qk_norm: false,
                output_gate: false,
                rope_neox: true,
                rope_freq_scale: 1.0,
                kv_f16: false,
            },
            delta: None,
            rms_eps: 1e-6,
            norm_offset: 0.0,
            tie_embeddings: true,
            attn_bias: true,
        }
    }

    /// A small Qwen3.5-shaped hybrid config (3 DeltaNet layers : 1 attention layer).
    pub fn tiny_hybrid(vocab: usize) -> Self {
        ModelConfig {
            arch: "qwen3_5".into(),
            vocab_size: vocab,
            hidden_size: 32,
            intermediate_size: 48,
            layer_kinds: vec![LayerKind::LinearAttention, LayerKind::LinearAttention, LayerKind::LinearAttention, LayerKind::FullAttention],
            attn: AttnConfig {
                n_heads: 4,
                n_kv_heads: 2,
                head_dim: 8,
                rot_dim: 4,
                rope_theta: 10000.0,
                qk_norm: true,
                output_gate: true,
                rope_neox: true,
                rope_freq_scale: 1.0,
                kv_f16: false,
            },
            delta: Some(DeltaNetConfig { n_k_heads: 2, n_v_heads: 4, k_head_dim: 8, v_head_dim: 8, conv_kernel: 4, bptt_chunk: 3 }),
            rms_eps: 1e-6,
            norm_offset: 1.0,
            tie_embeddings: true,
            attn_bias: false,
        }
    }
}

pub struct Mlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

pub struct MlpCache {
    x: Vec<f32>,
    g: Vec<f32>,
    u: Vec<f32>,
    h: Vec<f32>,
}

impl Mlp {
    pub fn forward(&self, x: &[f32], t: usize) -> (Vec<f32>, MlpCache) {
        let g = self.gate_proj.forward(x, t);
        let u = self.up_proj.forward(x, t);
        let h: Vec<f32> = g.iter().zip(&u).map(|(a, b)| silu(*a) * b).collect();
        let y = self.down_proj.forward(&h, t);
        (y, MlpCache { x: x.to_vec(), g, u, h })
    }
    pub fn backward(&mut self, c: MlpCache, dy: &[f32], t: usize) -> Vec<f32> {
        let dh = self.down_proj.backward(&c.h, dy, t);
        let mut dg = vec![0.0; dh.len()];
        let mut du = vec![0.0; dh.len()];
        for i in 0..dh.len() {
            dg[i] = dh[i] * c.u[i] * silu_grad(c.g[i]);
            du[i] = dh[i] * silu(c.g[i]);
        }
        let mut dx = self.gate_proj.backward(&c.x, &dg, t);
        let dxu = self.up_proj.backward(&c.x, &du, t);
        for i in 0..dx.len() {
            dx[i] += dxu[i];
        }
        dx
    }
}

#[allow(clippy::large_enum_variant)] // one per layer; boxing would only add indirection
pub enum Mixer {
    Attn(Attention),
    Delta(DeltaNet),
}

enum MixerCache {
    Attn(AttnCache),
    Delta(DeltaCache),
}

pub struct Block {
    pub input_norm: Vec<f32>,
    pub mixer: Mixer,
    pub post_norm: Vec<f32>,
    pub mlp: Mlp,
}

pub struct BlockCache {
    x: Vec<f32>,
    inv1: Vec<f32>,
    h: Vec<f32>,
    inv2: Vec<f32>,
    mixer: MixerCache,
    mlp: MlpCache,
}

impl Block {
    fn forward(&self, x: &[f32], t: usize, rope: &Rope, eps: f32, off: f32) -> (Vec<f32>, BlockCache) {
        let d = self.input_norm.len();
        let (n1, inv1) = rmsnorm(x, &self.input_norm, d, eps, off);
        let (m, mc) = match &self.mixer {
            Mixer::Attn(a) => {
                let (y, c) = a.forward(&n1, t, rope);
                (y, MixerCache::Attn(c))
            }
            Mixer::Delta(dn) => {
                let (y, c) = dn.forward(&n1, t);
                (y, MixerCache::Delta(c))
            }
        };
        let h: Vec<f32> = x.iter().zip(&m).map(|(a, b)| a + b).collect();
        let (n2, inv2) = rmsnorm(&h, &self.post_norm, d, eps, off);
        let (f, fc) = self.mlp.forward(&n2, t);
        let y: Vec<f32> = h.iter().zip(&f).map(|(a, b)| a + b).collect();
        (y, BlockCache { x: x.to_vec(), inv1, h, inv2, mixer: mc, mlp: fc })
    }

    fn backward(&mut self, c: BlockCache, dy: &[f32], t: usize, rope: &Rope, off: f32) -> Vec<f32> {
        let d = self.input_norm.len();
        let dn2 = self.mlp.backward(c.mlp, dy, t);
        let dh_n = rmsnorm_backward(&c.h, &self.post_norm, &c.inv2, &dn2, d, off);
        let dh: Vec<f32> = dy.iter().zip(&dh_n).map(|(a, b)| a + b).collect();
        let dn1 = match (&mut self.mixer, c.mixer) {
            (Mixer::Attn(a), MixerCache::Attn(mc)) => a.backward(mc, &dh, t, rope),
            (Mixer::Delta(dn), MixerCache::Delta(mc)) => dn.backward(mc, &dh, t),
            _ => unreachable!("mixer/cache mismatch"),
        };
        let dx_n = rmsnorm_backward(&c.x, &self.input_norm, &c.inv1, &dn1, d, off);
        dh.iter().zip(&dx_n).map(|(a, b)| a + b).collect()
    }

    pub fn linears_mut(&mut self) -> Vec<&mut Linear> {
        let mut v = match &mut self.mixer {
            Mixer::Attn(a) => a.linears_mut(),
            Mixer::Delta(d) => d.linears_mut(),
        };
        v.push(&mut self.mlp.gate_proj);
        v.push(&mut self.mlp.up_proj);
        v.push(&mut self.mlp.down_proj);
        v
    }
    pub fn linears(&self) -> Vec<&Linear> {
        let mut v = match &self.mixer {
            Mixer::Attn(a) => a.linears(),
            Mixer::Delta(d) => d.linears(),
        };
        v.push(&self.mlp.gate_proj);
        v.push(&self.mlp.up_proj);
        v.push(&self.mlp.down_proj);
        v
    }
}

/// "Predict `token` from the hidden state at `pos`."
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    pub pos: usize,
    pub token: u32,
}

/// Targets for supervised training: every answer token `tokens[i]`
/// (`i >= answer_start`) is predicted from position `i-1`.
pub fn answer_targets(tokens: &[u32], answer_start: usize) -> Vec<Target> {
    (answer_start.max(1)..tokens.len()).map(|i| Target { pos: i - 1, token: tokens[i] }).collect()
}

pub struct Model {
    pub cfg: ModelConfig,
    pub embed: Frozen,
    pub blocks: Vec<Block>,
    pub final_norm: Vec<f32>,
    pub lm_head: Frozen,
    /// Recompute each block's activations in the backward pass instead of storing them.
    pub checkpointing: bool,
    /// Rows of logits computed at once (bounds the `[rows × vocab]` buffer).
    pub logits_chunk: usize,
    /// Per-frequency RoPE divisors (Llama 3.1+ `rope_freqs`), as in the engine.
    pub rope_factors: Option<Vec<f32>>,
}

/// Map GGUF module names to the Hugging Face names used for LoRA targets.
pub fn canonical_module(name: &str) -> &str {
    match name {
        "attn_q" => "q_proj",
        "attn_k" => "k_proj",
        "attn_v" => "v_proj",
        "attn_output" => "o_proj",
        "ffn_gate" => "gate_proj",
        "ffn_up" => "up_proj",
        "ffn_down" => "down_proj",
        other => other,
    }
}

/// Which projections receive LoRA adapters (matched against the last path component).
pub const DEFAULT_TARGETS: &[&str] =
    &["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "out_proj"];

impl Model {
    pub fn hidden(&self) -> usize {
        self.cfg.hidden_size
    }

    /// Random-weight model (tests / toy runs). `quantize` stores the frozen
    /// weights as Q8_0 when the row length allows it.
    pub fn random(cfg: ModelConfig, seed: u64, quantize: bool) -> Self {
        let mut rng = Rng::new(seed);
        let d = cfg.hidden_size;
        let std = 0.15;
        let mk = |rng: &mut Rng, out: usize, inp: usize| -> Frozen {
            let w = rng.vec_normal(out * inp, std);
            if quantize {
                FrozenWeight::q8_0(&w, out, inp).into_arc()
            } else {
                FrozenWeight::f32(w, out, inp).into_arc()
            }
        };
        let norm = |rng: &mut Rng, n: usize, off: f32| -> Vec<f32> { (0..n).map(|_| 1.0 - off + rng.uniform_sym(0.2)).collect() };
        let off = cfg.norm_offset;
        let embed = mk(&mut rng, cfg.vocab_size, d);
        let a = cfg.attn.clone();
        let mut blocks = Vec::new();
        for (li, kind) in cfg.layer_kinds.iter().enumerate() {
            let p = format!("model.layers.{li}");
            let mixer = match kind {
                LayerKind::FullAttention => {
                    let qo = a.n_heads * a.head_dim * if a.output_gate { 2 } else { 1 };
                    let kvo = a.n_kv_heads * a.head_dim;
                    let bias = |rng: &mut Rng, n| if cfg.attn_bias { Some(rng.vec_normal(n, 0.1)) } else { None };
                    Mixer::Attn(Attention {
                        cfg: a.clone(),
                        q_proj: Linear::new(format!("{p}.self_attn.q_proj"), mk(&mut rng, qo, d), bias(&mut rng, qo)),
                        k_proj: Linear::new(format!("{p}.self_attn.k_proj"), mk(&mut rng, kvo, d), bias(&mut rng, kvo)),
                        v_proj: Linear::new(format!("{p}.self_attn.v_proj"), mk(&mut rng, kvo, d), bias(&mut rng, kvo)),
                        o_proj: Linear::new(format!("{p}.self_attn.o_proj"), mk(&mut rng, d, a.n_heads * a.head_dim), None),
                        q_norm: a.qk_norm.then(|| norm(&mut rng, a.head_dim, off)),
                        k_norm: a.qk_norm.then(|| norm(&mut rng, a.head_dim, off)),
                        eps: cfg.rms_eps,
                        norm_offset: off,
                    })
                }
                LayerKind::LinearAttention => {
                    let dc = cfg.delta.clone().expect("delta config required for linear_attention layers");
                    let cd = dc.conv_dim();
                    let nv = dc.n_v_heads;
                    Mixer::Delta(DeltaNet {
                        in_proj_qkv: Linear::new(format!("{p}.linear_attn.in_proj_qkv"), mk(&mut rng, cd, d), None),
                        in_proj_z: Linear::new(format!("{p}.linear_attn.in_proj_z"), mk(&mut rng, dc.value_dim(), d), None),
                        in_proj_b: Linear::new(format!("{p}.linear_attn.in_proj_b"), mk(&mut rng, nv, d), None),
                        in_proj_a: Linear::new(format!("{p}.linear_attn.in_proj_a"), mk(&mut rng, nv, d), None),
                        conv_w: rng.vec_normal(cd * dc.conv_kernel, 0.4),
                        dt_bias: rng.vec_normal(nv, 0.5),
                        a_log: (0..nv).map(|_| (1.0 + rng.uniform() * 3.0).ln() - 1.0).collect(),
                        norm_w: norm(&mut rng, dc.v_head_dim, 0.0),
                        out_proj: Linear::new(format!("{p}.linear_attn.out_proj"), mk(&mut rng, d, dc.value_dim()), None),
                        eps: cfg.rms_eps,
                        cfg: dc,
                    })
                }
            };
            blocks.push(Block {
                input_norm: norm(&mut rng, d, off),
                mixer,
                post_norm: norm(&mut rng, d, off),
                mlp: Mlp {
                    gate_proj: Linear::new(format!("{p}.mlp.gate_proj"), mk(&mut rng, cfg.intermediate_size, d), None),
                    up_proj: Linear::new(format!("{p}.mlp.up_proj"), mk(&mut rng, cfg.intermediate_size, d), None),
                    down_proj: Linear::new(format!("{p}.mlp.down_proj"), mk(&mut rng, d, cfg.intermediate_size), None),
                },
            });
        }
        let final_norm = norm(&mut rng, d, off);
        let lm_head = if cfg.tie_embeddings { embed.clone() } else { mk(&mut rng, cfg.vocab_size, d) };
        Model { cfg, embed, blocks, final_norm, lm_head, checkpointing: true, logits_chunk: 64, rope_factors: None }
    }

    /// Attach fresh LoRA adapters to every linear whose last name component is in `targets`
    /// (Hugging Face names; GGUF names such as `attn_q`/`ffn_down` are matched through
    /// [`canonical_module`]).
    /// Returns the number of trainable parameters.
    pub fn attach_lora(&mut self, rank: usize, alpha: f32, targets: &[&str], seed: u64) -> usize {
        let mut rng = Rng::new(seed);
        let mut n = 0;
        for b in &mut self.blocks {
            for l in b.linears_mut() {
                let short = canonical_module(l.name.rsplit('.').next().unwrap_or(""));
                if targets.iter().any(|t| canonical_module(t) == short) {
                    l.attach_lora(rank, alpha, &mut rng);
                    n += l.lora.as_ref().unwrap().num_params();
                }
            }
        }
        n
    }

    pub fn set_lora_enabled(&mut self, on: bool) {
        for b in &mut self.blocks {
            for l in b.linears_mut() {
                l.lora_enabled = on;
            }
        }
    }

    pub fn lora_linears_mut(&mut self) -> Vec<&mut Linear> {
        self.blocks.iter_mut().flat_map(|b| b.linears_mut()).filter(|l| l.lora.is_some()).collect()
    }
    pub fn lora_linears(&self) -> Vec<&Linear> {
        self.blocks.iter().flat_map(|b| b.linears()).filter(|l| l.lora.is_some()).collect()
    }

    pub fn zero_grad(&mut self) {
        for l in self.lora_linears_mut() {
            l.lora.as_mut().unwrap().zero_grad();
        }
    }

    pub fn trainable_params(&self) -> usize {
        self.lora_linears().iter().map(|l| l.lora.as_ref().unwrap().num_params()).sum()
    }

    fn embed_tokens(&self, tokens: &[u32]) -> Vec<f32> {
        let d = self.hidden();
        let mut x = vec![0.0; tokens.len() * d];
        for (i, &tok) in tokens.iter().enumerate() {
            assert!((tok as usize) < self.cfg.vocab_size, "token {tok} out of vocab");
            self.embed.row(tok as usize, &mut x[i * d..(i + 1) * d]);
        }
        x
    }

    fn rope(&self, t: usize) -> Rope {
        let a = &self.cfg.attn;
        Rope::with(t, a.rot_dim, a.rope_neox, a.rope_theta, a.rope_freq_scale, self.rope_factors.as_deref())
    }

    /// Final hidden states (after the last block, before the final norm).
    pub fn forward_hidden(&self, tokens: &[u32]) -> Vec<f32> {
        let t = tokens.len();
        let rope = self.rope(t);
        let mut x = self.embed_tokens(tokens);
        for b in &self.blocks {
            x = b.forward(&x, t, &rope, self.cfg.rms_eps, self.cfg.norm_offset).0;
        }
        x
    }

    /// Full logits for the last position (used by greedy decoding in the CLI).
    pub fn last_logits(&self, tokens: &[u32]) -> Vec<f32> {
        let d = self.hidden();
        let x = self.forward_hidden(tokens);
        let last = &x[(tokens.len() - 1) * d..];
        let (n, _) = rmsnorm(last, &self.final_norm, d, self.cfg.rms_eps, self.cfg.norm_offset);
        self.lm_head.forward(&n, 1)
    }

    /// Log-probabilities of each target (no gradient).
    pub fn logprobs(&self, tokens: &[u32], targets: &[Target]) -> Vec<f32> {
        self.score(tokens, targets).0
    }

    /// Log-probabilities of each target and how many targets are the greedy argmax.
    pub fn score(&self, tokens: &[u32], targets: &[Target]) -> (Vec<f32>, usize) {
        let x = self.forward_hidden(tokens);
        let (lp, _, hits) = self.head(&x, targets, None);
        (lp, hits)
    }

    /// Final norm + lm_head at target positions. When `weights` is given, also
    /// returns `dL/dx_final` for `L = -Σ wᵢ·logpᵢ`.
    fn head(&self, x: &[f32], targets: &[Target], weights: Option<&[f32]>) -> (Vec<f32>, Vec<f32>, usize) {
        let d = self.hidden();
        let t = x.len() / d;
        let (eps, off) = (self.cfg.rms_eps, self.cfg.norm_offset);
        let mut logp = Vec::with_capacity(targets.len());
        let mut hits = 0;
        let mut dx = if weights.is_some() { vec![0.0; x.len()] } else { Vec::new() };
        let v = self.cfg.vocab_size;
        for (ci, chunk) in targets.chunks(self.logits_chunk.max(1)).enumerate() {
            let n = chunk.len();
            let mut xs = vec![0.0; n * d];
            for (i, tg) in chunk.iter().enumerate() {
                assert!(tg.pos < t);
                xs[i * d..(i + 1) * d].copy_from_slice(&x[tg.pos * d..(tg.pos + 1) * d]);
            }
            let (ns, inv) = rmsnorm(&xs, &self.final_norm, d, eps, off);
            let logits = self.lm_head.forward(&ns, n);
            let mut dlogits = if weights.is_some() { vec![0.0; n * v] } else { Vec::new() };
            for (i, tg) in chunk.iter().enumerate() {
                let row = &logits[i * v..(i + 1) * v];
                let (lse, p) = softmax_row(row);
                let tl = row[tg.token as usize];
                logp.push(tl - lse);
                if row.iter().all(|&v| v <= tl) {
                    hits += 1;
                }
                if let Some(w) = weights {
                    // d(-w·logp)/dlogits = w·(softmax - onehot)
                    let wi = w[ci * self.logits_chunk.max(1) + i];
                    let dl = &mut dlogits[i * v..(i + 1) * v];
                    for j in 0..v {
                        dl[j] = wi * p[j];
                    }
                    dl[tg.token as usize] -= wi;
                }
            }
            if weights.is_some() {
                let mut dns = vec![0.0; n * d];
                self.lm_head.backward_input(&dlogits, n, &mut dns);
                let dxs = rmsnorm_backward(&xs, &self.final_norm, &inv, &dns, d, off);
                for (i, tg) in chunk.iter().enumerate() {
                    for j in 0..d {
                        dx[tg.pos * d + j] += dxs[i * d + j];
                    }
                }
            }
        }
        (logp, dx, hits)
    }

    /// Forward + backward for `L = -Σ wᵢ·logpᵢ`. Accumulates LoRA gradients
    /// (call [`Model::zero_grad`] first) and returns the per-target log-probs.
    pub fn forward_backward(&mut self, tokens: &[u32], targets: &[Target], weights: &[f32]) -> Vec<f32> {
        assert_eq!(targets.len(), weights.len());
        let t = tokens.len();
        let rope = self.rope(t);
        let (eps, off) = (self.cfg.rms_eps, self.cfg.norm_offset);
        let mut x = self.embed_tokens(tokens);
        let mut inputs: Vec<Vec<f32>> = Vec::new();
        let mut caches: Vec<BlockCache> = Vec::new();
        for b in &self.blocks {
            if self.checkpointing {
                inputs.push(x.clone());
                x = b.forward(&x, t, &rope, eps, off).0;
            } else {
                let (y, c) = b.forward(&x, t, &rope, eps, off);
                caches.push(c);
                x = y;
            }
        }
        let (logp, mut dx, _) = self.head(&x, targets, Some(weights));
        drop(x);
        for li in (0..self.blocks.len()).rev() {
            let cache = if self.checkpointing {
                let input = inputs.pop().unwrap();
                self.blocks[li].forward(&input, t, &rope, eps, off).1
            } else {
                caches.pop().unwrap()
            };
            dx = self.blocks[li].backward(cache, &dx, t, &rope, off);
        }
        logp
    }

    /// Mean negative log-likelihood of `targets` (convenience for eval).
    pub fn nll(&self, tokens: &[u32], targets: &[Target]) -> (f32, usize) {
        if targets.is_empty() {
            return (0.0, 0);
        }
        let lp = self.logprobs(tokens, targets);
        (-lp.iter().sum::<f32>(), lp.len())
    }

    pub fn frozen_bytes(&self) -> usize {
        let mut n = self.embed.nbytes();
        if !self.cfg.tie_embeddings {
            n += self.lm_head.nbytes();
        }
        for b in &self.blocks {
            for l in b.linears() {
                n += l.w.nbytes();
            }
        }
        n
    }
}
