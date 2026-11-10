# Rust_Trainer

The Phase 3 trainer for agentOS. It retrains the local model overnight on what you have **approved and corrected**, and swaps the result in only if it measurably does better.

It trains a **LoRA adapter**: the base weights stay frozen and compressed, and small extra matrices learn the change. Backpropagation is written by hand through the same forward pass the phone runs. For GGUF models that forward pass *is* the [Rust Inference Engine](https://github.com/olapwriter-prog/Rust_Interference_engine) (`rie-core`): its file loader, quant formats and SIMD kernels, and it is **bit-exact** with the engine's own `CpuBackend`. llama.cpp's trainer crashes on Qwen, and `candle` has no DeltaNet.

```
agentd ctl learn ──► confirmed.jsonl ──► rust_trainer run ──────────────► model.adapters/<id>/adapter.gguf ──► rie engine
   (export)          (confirmed only)    Deep + charging + battery + heat;  (llama.cpp LoRA GGUF)              y += (α/r)·B·A·x
                                         pause/resume from checkpoints;
                                         eval gate vs current model; ledger
```

## Status

| Part | State |
|---|---|
| **GGUF models through `rie-core`**: llama / mistral / qwen2 (Qwen2.5) / qwen3; weights stay mmapped in their ggml format (F32, F16, BF16, Q4_0…Q8_0, Q2_K…Q6_K) | ✅ |
| **Forward bit-exact with the engine**: 3 architectures × F32/F16/Q8_0/Q4_0/Q4_K/Q6_K × tied/untied on x86 (AVX2 and scalar) and AArch64 i8mm / dotprod / Cortex-A53 (QEMU) | ✅ tested |
| Prompts tokenised exactly like `Engine::chat` (engine chat template + GGUF tokenizer) | ✅ tested |
| Engine-loadable adapter: llama.cpp LoRA GGUF (`adapter.gguf`), plus PEFT safetensors | ✅ written; engine side specified in [`docs/ENGINE_LORA.md`](docs/ENGINE_LORA.md) |
| Backward through Qwen2.5/Qwen3/Llama: RMSNorm, GQA attention, RoPE (NEOX and normal, rope_freqs), q/k/v bias, QK-norm, SwiGLU, frozen matmuls (input gradient only), LoRA | ✅ gradient-checked, also through the engine-backed model |
| Backward through Qwen3.5: zero-centred RMSNorm, partial RoPE, gated attention, **Gated DeltaNet with BPTT** | ✅ gradient-checked (HF safetensors path; the engine has no Qwen3.5 yet) |
| Training loop: AdamW, warm-up + cosine LR, clipping, answer-token loss, activation recomputation, chunked BPTT state checkpoints; DPO on corrections | ✅ |
| Confirmed-only data, exact pause/resume, eval gate, swap, keep-5 rollback, ledger, Deep/charging/battery/heat policy with hysteresis | ✅ tested; end-to-end CLI run in CI |
| **Mixture of LoRA experts** (`--experts K`): skill grouping (labels or clustering), branch-trained experts, TF-IDF router, per-expert gate | ✅ tested; routing recipe for the engine in [`docs/ENGINE_LORA.md`](docs/ENGINE_LORA.md#6-mixture-of-experts-bundles) |
| **Open-source sandbox**: real Qwen3-0.6B GGUF (Docker model catalog), SNIPS requests → actions, phone-night simulator (stage, charging, battery, CPU-driven thermal model, unplug), exact-action eval | ✅ [`docs/SANDBOX.md`](docs/SANDBOX.md); found and fixed two heat-limit overshoots |
| **agentd drop-in** (`rust_trainer train -m … --data learn/sft.jsonl --grammar … --out adapter.new.lora --steps N --max-seconds S`): agentd's real job protocol, AGLORA1 adapters served unchanged by `agentos-llm --lora` | ✅ verified with agentd's real export on Qwen3-0.6B: agentos-llm's held-out score 4.2 % → 75 % with our adapter (30 steps); batches share agentd's system prompt in one pass (3.3× faster, exact); agentd needs one config key to launch it ([`docs/AGENTD.md`](docs/AGENTD.md)) |
| Engine side (load and apply `adapter.gguf`) | ⏳ in the engine's own session; spec and acceptance test in [`docs/ENGINE_LORA.md`](docs/ENGINE_LORA.md) |
| Measured improvement on held-out requests with a real model (plan step 3) | ✅ in the sandbox: real Qwen3-0.6B Q4_0, one simulated night, 140 open-dataset requests → exact action 0 % → 42.9 %, intent 0 % → 97.6 % on 42 unseen requests. ⏳ on your own corrections |
| Parity of the HF/Qwen3.5 path with `transformers` + `peft` | ⏳ harness in `tools/parity.py`, not yet run (no PyTorch here) |

## Build

Same toolchain settings as the engine: Rust ≥ 1.82, edition 2021, `rustfmt` at width 140, `clippy -D warnings`, fat LTO, `panic = "abort"`, and the same `.cargo/config.toml` for Android/iOS/AArch64. `rie-core` comes from the engine's repository, pinned in `Cargo.lock`.

```sh
cargo build --release                          # with rie-core (default)
cargo build --release --no-default-features    # HF safetensors only, no engine
cargo build --release --features hf-tokenizers # raw-text data for HF directories
cargo check --target aarch64-linux-android     # phone build check (also aarch64-apple-ios)
cargo test                                     # 28 tests: engine parity, gradient checks, training, scheduler
```

## Commands

```sh
rust_trainer info     --model qwen3-4b-q4_0.gguf                  # sizes, adapter memory, FLOP/token
rust_trainer train    --model M.gguf --data confirmed.jsonl --out M.adapters/manual
rust_trainer eval     --model M.gguf --adapter DIR --data confirmed.jsonl      # DIR may hold only adapter.gguf
rust_trainer run      --model M.gguf --data confirmed.jsonl --status /run/agentd/device.json --hours 7
rust_trainer run      --model M.gguf … --experts 4                 # mixture of experts instead of one adapter
rust_trainer route    --model M.gguf --prompt "dim the hall lamp"   # which expert a request goes to
rust_trainer train    -m M.gguf --data learn/sft.jsonl --grammar learn/grammar.gbnf \
                      --out learn/adapter.new.lora --steps 60 --max-seconds 1800  # agentd's job (drop-in for agentos-llm train)
rust_trainer status   --model M.gguf
rust_trainer rollback --model M.gguf --reason "user asked"
rust_trainer synth    --out m.gguf --arch qwen3 --wtype Q4_0       # random-weight GGUF (or --preset qwen3-0.6b;
                                                                   #  shape overrides --layers --hidden --heads --kv-heads --head-dim --ff --vocab)
rust_trainer toy                                                   # Qwen3.5-shaped demo, no files needed
```

`--model` takes a `.gguf` file (engine path) or a Hugging Face directory. The options are:

* **Training:** `--rank 8 --alpha 16 --lr 2e-4 --epochs 2 --batch 4 --max-len 512 --heldout 0.1 --dpo BETA`
* **Gate:** `--min-improvement 0.02 --min-examples 8`
* **Scheduling:** `--min-train 50 --min-battery 50 --max-temp 40`
* **Engine:** `--threads N --no-pin --template chatml`

An end-to-end run on a synthetic Q4_0 Qwen3 GGUF (it is also in CI) went like this:

* 2 unconfirmed records were dropped.
* Training took 3 s, held-out NLL fell 81%, and the gate accepted.
* The next "night" was deferred at 43 °C, then resumed from the checkpoint once the phone cooled. Its candidate was measured against the *current* adapter (not the base), beat it by 19%, and was promoted.
* `eval` from `adapter.gguf` alone reproduced the result, and `rollback` restored the previous adapter.

## How it matches the engine

| Engine behaviour | Trainer |
|---|---|
| Weights are `rie_core::tensor::Weight` views of the GGUF mmap | the same views (`rie::RieFrozen`), no copies |
| `matmul` quantises activations to int8 / Q8_K and runs SDOT/SMMLA/AVX2 kernels | calls that same `matmul`. The backward pass `dx = dy·W` dequantises rows (straight-through for the int8 rounding, as in QLoRA) |
| f16 KV cache read with `dot_f16_f32` / `axpy_f16` | K/V rounded to f16 and read with the same functions (identity in the backward pass) |
| `dot_f32`, `x·s·w` RMSNorm, in-place `x·(s·w)` QK-norm, `x/(1+e^{-x})` SiLU, scaled-q attention, RoPE tables | the same functions and operation order |
| `ChatTemplate::render` + `encode(…, add_bos, parse_special)` | identical prompt tokens, and the answer ends with the template's end-of-turn token |

Because int8 rounding of activations amplifies any sub-ulp difference, the test requires exact equality (`tests/engine.rs::logits_match_engine`).

## 1. Learn from your data, confirmed only

A record is used only if it says `"confirmed": true` or has `status` `confirmed`/`approved`/`corrected`. Supervised records (`prompt`/`messages` + `response`) train with cross-entropy on the answer tokens only. Corrections (`chosen`/`rejected`) train with DPO (`--dpo β`). The split is deterministic by id. The full schema is in [`docs/AGENTD.md`](docs/AGENTD.md).

## 2. LoRA, not the whole model

`y = W·x + b + (α/r)·B·(A·x)` with B initialised to 0, so a fresh adapter is a no-op. The default targets are every projection (`attn_q/k/v/output`, `ffn_gate/up/down`, plus DeltaNet `in_proj_qkv/z`, `out_proj` for Qwen3.5).

| Qwen3.5-4B-shaped, all projections | LoRA params | adapter + grads + AdamW |
|---|---|---|
| r = 8 | 15.2 M | 244 MB |
| r = 16 | 30.5 M | 488 MB |

## 3. Run only when harmless

Training runs only in the **Deep** stage, **charging**, with **battery ≥ min** and **temperature < limit**. The device state is re-read before every example. When a condition fails, the run pauses with a checkpoint and waits; restarting needs 3 °C cooler and 5 % more battery. Resuming is bit-identical to an uninterrupted run. A checkpoint whose data or config changed is discarded, and training restarts.

## 4. Eval gate, swap, rollback

The candidate must beat the **current** model (base plus active adapter) on held-out confirmed examples:

* NLL must fall by at least 2 %;
* greedy accuracy and preference accuracy must not fall;
* there must be at least 8 examples.

Otherwise the candidate is deleted. The last 5 accepted adapters are kept, and `rollback` walks back to the base model. Every transition is a ledger line. The registry lives next to the model: `model.adapters/` beside `model.gguf`, or `<dir>/adapters/` for an HF directory.

## 4b. Mixture of experts, on your data alone

With `--experts K`, a night produces specialists instead of one adapter that has to cover everything:

1. **Group:** confirmed examples are grouped by skill. The exporter's `skill`/`task`/`tool` label is used when present; otherwise the trainer clusters your requests with cosine k-means.
2. **General first:** a *general* expert trains on all your data. It also answers anything the router is unsure about.
3. **Branch-train:** every skill group with enough examples gets an expert, initialised from the general one and specialised on that group.
4. **Route:** a router picks the expert per request using hashed TF-IDF over the prompt's tokens and token pairs. Chat-template tokens get zero weight. No forward pass is needed, and the engine can reproduce it exactly.
5. **Gate twice:**
   * each expert is kept only if it beats the general expert on *its own* held-out requests;
   * the routed mixture must beat the current model (base, adapter or previous mixture) through the normal gate.

   Experts can be dropped individually, and the whole bundle rolls back like any adapter.

This is a mixture of *adapters* on the frozen base, not a conversion of the base into a sparse MoE. Training per-layer routers and experts inside the model would need far more data and compute than a phone and a few hundred examples provide.

**Measured** on a synthetic Q4_0 Qwen3 through the engine, with 240 confirmed examples across 3 skills (home control, arithmetic, messages) and 20 % held out:

| | held-out NLL | answer accuracy |
|---|---|---|
| base | 6.48 | 0.1 % |
| single adapter, 3 epochs | 0.99 | 75.1 % |
| single adapter, 6 epochs (same compute as the mixture) | **0.48** | **87.5 %** |
| mixture (general + 3 experts), r = 4 | 0.49 | 85.3 % |
| single adapter, r = 1, same compute | 1.47 | 67.8 % |
| mixture, r = 1 | **1.41** | **68.5 %** |

Routing put unseen requests in the right group ≥ 90 % of the time without any labels.

At equal compute the mixture matched a single adapter when the adapter had room for all skills (r = 4). It was about 4 % better when capacity was tight (r = 1). Expect gains when your skills pull the model in different directions; otherwise the gate keeps whichever is better.

Before branch-training, skill experts trained from scratch saw only a third of the steps, and the per-expert gate dropped 2 of 3. Starting them from the general expert fixed that.

## 5. Never touches the approval rules

The trainer reads the model and the export and writes only the adapter registry. Tiers and the approval key stay with the executor.

## Measured speed

These were measured in this build sandbox: 4 x86 vCPUs, AVX2, engine threads = 4. Weights are random, but the shapes are real (`rie synth` presets). Sequence length 128, loss on 64 answer tokens, r = 8.

| model (Q4_0) | forward | training step (forward + recompute + backward) |
|---|---|---|
| Qwen2.5-0.5B | 93 tok/s | 21 tok/s |
| Qwen3-0.6B | 70 tok/s | 17 tok/s |
| Qwen3-8B (4.4 GB file, r = 8: 21.8 M LoRA params) | 6.9 tok/s | 1.4 tok/s |

For comparison, the engine itself (`rie bench`, same files and threads) does prompt processing / generation at 144 / 32 tok/s for Qwen2.5-0.5B and 8.1 / 2.3 tok/s for Qwen3-8B.

A training step costs about 4.4× a forward pass. Most of the backward time is `dx = dy·W` from dequantised rows. The engine has no transposed int8 kernel, and keeping transposed copies would double weight memory on the phone.

At these rates, 200 examples × 100 tokens × 2 epochs (40 k trained tokens) take about 30 min for 0.5B. Scaled by size, 4B would take about 4 h on this sandbox. Phone performance cores with dotprod/i8mm run the int8 forward faster than these vCPUs, but the f32 backward has not been measured on a device. Reproduce with `cargo run --release --example bench_train -- qwen3-4b Q4_0 128 3`.

## Backward pass, briefly

* **Frozen matmuls:** `dx = dy·W` only.
* **LoRA:** `dA = s·(dy·B)ᵀ·x`, `dB = s·dyᵀ·(A·x)`.
* **Norms, SwiGLU, gates:** closed form.
* **Attention:** softmax backward per head, with RoPE backward as the inverse rotation.
* **Gated DeltaNet:** backpropagation through time over `S' = αS, u = v − S'k, S = S' + βukᵀ, o = Sq`. Only every `bptt_chunk`-th state is stored; the rest are recomputed.
* **Memory:** each block is recomputed in the backward pass (activation checkpointing), and logits are computed only at answer positions.

## Tests

* **`tests/engine.rs`:**
  * bit-exact logits against `CpuBackend` (36 configurations; also run on the three AArch64 tiers and with the scalar kernels);
  * dequantised input gradient against a dense reference for 8 quant formats;
  * numerical LoRA gradients through engine-backed Llama/Qwen2/Qwen3;
  * f16 rounding identical to the engine's KV conversion;
  * prompt tokens identical to `Engine::chat`;
  * training a Q4_0 model plus the `adapter.gguf` round-trip;
  * refusal of unsupported architectures.
* **`tests/gradcheck.rs`:** every op and mixer (including DeltaNet with BPTT chunk sizes 1, 3, 4, 7 and 20) against finite differences, and end-to-end LoRA gradients in f32 and Q8_0.
* **`tests/training.rs`:** loss falls and the gate accepts; pause/resume is exact; DPO; PEFT adapter round-trip with apply/merge; the overnight flow with exact ledger sequence; power hysteresis.
* **Unit tests:** data filter, chat formatting, safetensors, registry, LR schedule, Q8_0.

## Integration still to do (other repositories)

1. **Engine:** load `model.adapters/ACTIVE` → `<id>/adapter.gguf` and add `s·B·(A·x)` after each adapted matmul, as specified in [`docs/ENGINE_LORA.md`](docs/ENGINE_LORA.md) with an acceptance test that should be bit-exact.
2. **Engine:** Qwen3.5 (Gated DeltaNet) support. The trainer's DeltaNet forward and backward are ready. Once the engine can read Qwen3.5 GGUFs, the same `RieFrozen` bridge applies.
3. **agentd:** write the device-state file, launch `rust_trainer run` in the charging Deep stage under the Training mode, and reload the LLM after `accepted`, per [`docs/AGENTD.md`](docs/AGENTD.md).
4. **Plan step 3:** run on a real GGUF (e.g. Qwen2.5-0.5B/Qwen3-0.6B Q4_0) with real confirmed corrections, on the Mac and then in the VM.
