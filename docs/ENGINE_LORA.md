# Applying trained adapters in the Rust Inference Engine

This is the engine-side counterpart of `rust_trainer`. The trainer produces LoRA adapters for a GGUF model. The engine (`rie-core`, repository `Rust_Interference_engine`) has to load the active one and add it to the forward pass. Nothing else in the engine changes.

## 1. Where the adapter is

For a model file `…/qwen3-4b-q4_0.gguf` the trainer keeps its registry next to it:

```
…/qwen3-4b-q4_0.adapters/
  ACTIVE                 id of the adapter in use; missing or empty = base model only
  <id>/adapter.gguf      the adapter to load
  <id>/…                 trainer files (ignore)
  ledger.jsonl           history (ignore)
```

On `Engine::load`, read `ACTIVE`. If it names an id, load `<id>/adapter.gguf`. A rollback or a newly accepted adapter only changes `ACTIVE`, so the engine (or `agentd`) reloads after such an event. Writes to `ACTIVE` are atomic (temp file + rename).

## 2. File format (llama.cpp LoRA GGUF)

Metadata:

| key | value |
|---|---|
| `general.architecture` | same as the base model (`qwen2`, `qwen3`, `llama`) |
| `general.type` | `"adapter"` |
| `adapter.type` | `"lora"` |
| `adapter.lora.alpha` | f32 α |

Tensors are F32, one pair per adapted projection of layer `i`:

| tensor | ggml `ne` | as `rie_core::tensor::Weight` |
|---|---|---|
| `blk.{i}.{m}.weight.lora_a` | `[n_in, r]` | `rows = r`, `cols = n_in` (A) |
| `blk.{i}.{m}.weight.lora_b` | `[r, n_out]` | `rows = n_out`, `cols = r` (B) |

Here `{m}` is one of `attn_q, attn_k, attn_v, attn_output, ffn_gate, ffn_up, ffn_down`. Any module may be absent. The scale is `s = α / r`. This is the same layout llama.cpp's `--lora` reads, so llama.cpp can load these files too.

`GgufFile::open` plus `Weight::from_raw(t.ty, t.n_rows(), t.row_len(), data)` reads them with the engine's existing code. `rust_trainer::rie::read_gguf_adapter` is a working reference.

## 3. Forward pass

For each adapted projection `W` with input `x` (the f32 activations **before** the engine quantises them to int8):

```
y = matmul(W, x)          // existing code
y += bias                 // existing code (qwen2 q/k/v)
h = A · x                 // r values per token, f32
y += s · (B · h)          // n_out values per token
```

To stay **bit-exact with training**, do it in this order and with these primitives:

* `h[k] = dot_f32(A.row(k), x)`
* `d[o] = dot_f32(B.row(o), h)`
* `y[o] += s * d[o]`, applied after the bias.

The trainer computes the same with `rie_core::kernels::dot_f32`, so the engine with the adapter reproduces the trained model's logits exactly, just as the base forward pass already matches (`rust_trainer/tests/engine.rs::logits_match_engine`, 36 configurations, and three AArch64 tiers under QEMU).

In `model/cpu.rs` the inputs are already at hand: `self.xn` for `wq/wk/wv` and `w_gate/w_up`, `self.att` for `wo`, and the SiLU product (in `self.gate`) for `w_down`. Each costs `r·(n_in + n_out)` multiply-adds per token, about 0.4 % of a 4B model's matmul work at r = 8.

Do **not** merge B·A into a quantised W and requantise. That loses most of what was learned at 4 bits. Merging is fine only for F16/F32 weights (`rust_trainer::adapter::LoraAdapter::merge_into`).

GPU backend (`rie-gpu`): either run the two small products on the CPU between kernels, or upload A and B as F32 storage buffers and add a rank-r kernel. Both give the same result up to f32 summation order.

## 4. Suggested API

```rust
// rie-core
pub struct LoraAdapter { /* per layer, per module: Option<(Weight /*A*/, Weight /*B*/)>; scale */ }
impl LoraAdapter { pub fn load(path: impl AsRef<Path>, cfg: &ModelConfig) -> Result<Self>; }
impl Engine { pub fn set_adapter(&mut self, a: Option<Arc<LoraAdapter>>); }   // clears the KV cache
pub fn active_adapter_path(model_path: &Path) -> Option<PathBuf>;            // reads <stem>.adapters/ACTIVE
```

C ABI / Kotlin / Swift: `rie_set_adapter(ctx, path_or_null)`.

Changing the adapter invalidates the KV cache (keys and values depend on it), so `set_adapter` must reset the prompt-prefix cache.

## 5. Acceptance test for the engine side

With both crates in one workspace or as dev-dependencies:

1. Build a synthetic model (`SyntheticSpec::tiny("qwen3", Q4_0)`).
2. Train a few steps with `rust_trainer`, then `write_gguf_adapter`.
3. Load the model in the engine with the adapter, run `forward(all_logits)`, and compare with `rust_trainer::model::Model::logprobs` on the same tokens. The result should be bit-identical, as it already is without an adapter.

## 6. Mixture of experts (bundles)

When the night produced a mixture, `ACTIVE` names a **bundle**: an adapter directory containing `router.json`.

```
<id>/router.json
<id>/experts/<name>/adapter.gguf      one per expert, same format as above
```

Check with `exists(<id>/router.json)`. Without it, the directory is a single adapter (§2).

### Routing recipe (`"feature": "tfidf_hash_v1"`)

The input is the prompt token ids exactly as `Engine::chat` builds them: `template.render(messages, true)` encoded with BOS and special-token parsing. The trainer routes on the same tokens.

```text
mix64(z):  z += 0x9E3779B97F4A7C15
           z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
           z = (z ^ (z >> 27)) * 0x94D049BB133111EB
           return z ^ (z >> 31)                          (all wrapping u64)

count[b] for each token t:          b = mix64(t)                              % dim
         for each adjacent (a, c):  b = mix64(1<<63 | a<<32 | c)              % dim
v[b]   = (1 + ln count[b]) * idf[b]        (0 where count is 0)
v      = v / |v|₂
s_i    = v · centroids[i]
i*     = argmax s_i
expert = (s_{i*} >= min_similarity) ? (experts[i*] ?? fallback) : fallback
```

`fallback` = `null` means the base model with no adapter.

The router fields are `dim`, `idf[dim]`, `centroids[k][dim]`, `names[k]`, `experts[k]` (a name or `null`), `fallback` and `min_similarity`. `rust_trainer::moe::Router::route` is the reference implementation. `rust_trainer route --model M.gguf --prompt "…"` prints the decision for a prompt.

### Serving

* **Route once per conversation.** Do it on the first user turn (system + first message) and keep that expert for the rest of the conversation. Switching experts mid-conversation invalidates the KV cache, because K and V depend on the adapter.
* **Keep experts loaded.** Each is a few MB to a few tens of MB (r = 8 on a 4B model ≈ 60 MB in F32). Keep them all in memory, or load on first use.
* **Applying the chosen expert:** identical to §3.
