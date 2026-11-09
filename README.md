# Rust_Trainer

The Phase 3 trainer for agentOS. It retrains the local model overnight on what you have **approved and corrected**, and swaps the result in only if it measurably does better.

It trains a **LoRA adapter**: the base weights stay frozen and compressed, and small extra matrices learn the change. Backpropagation is written by hand through our own Qwen forward pass, including Qwen3.5's Gated DeltaNet layers. llama.cpp's trainer crashes on Qwen, and `candle` has no DeltaNet.

```
agentd ctl learn ──► confirmed.jsonl ──► rust_trainer run ──► adapters/<id>/ ──► agentos-llm
   (export)          (confirmed only)    (Deep + charging,      (PEFT format)     W·x + (α/r)·B·A·x
                                          battery, heat;
                                          pause/resume;
                                          eval gate; ledger)
```

## Status

| Part | State |
|---|---|
| Backward through Qwen2.5 (dense): RMSNorm, GQA attention + RoPE + q/k/v bias, SwiGLU, frozen matmuls (input gradient only), LoRA | ✅ gradient-checked |
| Backward through Qwen3.5: zero-centred RMSNorm, QK-norm, partial RoPE, gated attention output, **Gated DeltaNet with BPTT** (conv1d, L2-norm, β/α gates, delta-rule recurrence, gated norm) | ✅ gradient-checked |
| Frozen weights as Q8_0 (or any engine format via the `FrozenMatmul` trait) | ✅ |
| Training loop: AdamW, warm-up + cosine LR, gradient clipping, loss on answer tokens only, activation recomputation per block, chunked BPTT state checkpoints | ✅ |
| DPO on corrections ("this, not that") against the frozen base | ✅ |
| Confirmed-only data loading from `agentd ctl learn` JSONL | ✅ |
| Adapter format (PEFT-compatible safetensors, so it also converts for llama.cpp) and reference `apply`/`merge` for the engine | ✅ |
| Resumable checkpoints (exact: an interrupted run gives bit-identical weights) | ✅ |
| Eval gate, swap, keep-N rollback, ledger | ✅ |
| Power/heat policy (Deep + charging + battery + temperature, with hysteresis) and overnight orchestration | ✅ |
| Loader for HF Qwen2.5 / Qwen3 / Qwen3.5 / Qwen3-Next safetensors | ✅ written; parity with `transformers` checked on random-weight models (see below) |
| Hook into `agentos-llm` and `agentd` | ⏳ both repositories were empty when this was written; the integration points are listed below |
| Measured improvement on real held-out corrections (plan step 3) | ⏳ needs real weights and real data |

## Commands

```sh
cargo build --release                                # add --features hf-tokenizers to train on raw text

rust_trainer toy                                     # self-contained demo on a random-weight Qwen3.5-shaped model
rust_trainer info   --model ~/models/qwen3.5-4b      # parameter counts, adapter memory, FLOP/token
rust_trainer train  --model DIR --data confirmed.jsonl --out DIR/adapters/manual   # one-shot train + eval
rust_trainer eval   --model DIR --adapter DIR/adapters/<id> --data confirmed.jsonl
rust_trainer run    --model DIR --data confirmed.jsonl --status /run/agentd/device.json --hours 8
rust_trainer status --model DIR
rust_trainer rollback --model DIR --reason "user asked"
```

Common options are `--rank 8 --alpha 16 --lr 2e-4 --epochs 2 --batch 4 --max-len 512 --heldout 0.1 --q8 --dpo 0.1 --min-improvement 0.02 --min-examples 8 --min-train 50 --min-battery 50 --max-temp 40 --bptt-chunk 16`.

`toy` output on this machine (4 vCPU): loss falls from 6.37 to 0.53 in 60 steps (6 s). On held-out commands with **unseen** device/room combinations, answer-token accuracy goes from 0% to 94.5% and NLL drops 92%, so the gate accepts.

## 1. Learn from your data, confirmed only

The input is JSON Lines from `agentd ctl learn`:

```jsonl
{"id":"r17","prompt":"dim the hall lamp","response":"{\"act\":\"dim\",\"dev\":\"lamp\",\"room\":\"hall\"}","confirmed":true}
{"id":"c03","prompt":"clean up downloads","chosen":"fs.trash(~/Downloads/*.tmp)","rejected":"rm -rf ~/Downloads","status":"corrected"}
{"id":"m09","confirmed":true,"messages":[{"role":"system","content":"…"},{"role":"user","content":"…"},{"role":"assistant","content":"…"}]}
```

* A record is used **only** if it says `"confirmed": true` or has a `status` of `confirmed`/`approved`/`corrected`. Everything else is dropped and counted, including the model's own unconfirmed output (`data.rs`, tested in `only_confirmed_records_are_kept`).
* Supervised records (`prompt`/`messages` + `response`) train with cross-entropy on the answer tokens only. Preference records (`chosen`/`rejected`) train with DPO (`--dpo β`).
* Field aliases are accepted (`answer`, `completion`, `corrected`, `original`, `model_output`, …). Any text field may instead come pre-tokenised as `<field>_ids`, so `agentd` can use the engine's tokenizer and the trainer needs none.
* The train/held-out split is deterministic by id hash, or explicit via `"split":"heldout"`. An example never moves sides between nights.
* Formatting is Qwen ChatML. For Qwen3.5 the assistant turn is prefixed with the empty think block (`<think>\n\n</think>\n\n`), matching non-thinking inference. That prefix belongs to the prompt and is not trained.

## 2. LoRA, not the whole model

`Linear` = frozen `W` (+ bias) + optional adapter: `y = W·x + b + (α/r)·B·(A·x)`, with A initialised kaiming-uniform and B = 0, so a fresh adapter is an exact no-op. The default targets are every projection: `q,k,v,o` / `in_proj_qkv, in_proj_z, out_proj` / `gate, up, down`.

For a Qwen3.5-4B-shaped config (`rust_trainer info`):

| rank | LoRA params | adapter + grads + AdamW (f32) |
|---|---|---|
| 8 | 15.2 M | 244 MB |
| 16 | 30.5 M | 488 MB |

Frozen weights only need `forward` (x·Wᵀ) and `backward_input` (dy·W). That is the whole `FrozenMatmul` trait, so the engine's compressed kernels plug in directly and no dequantised copy is made. The built-in types are f32 and Q8_0.

## 3. Run only when harmless

`scheduler::PowerPolicy` allows training only in the **Deep** stage, **while charging**, with **battery ≥ min** and **temperature < limit**. A paused run restarts only 3 °C cooler and 5 % fuller than the pause limits (hysteresis). The device state is read before **every example**. When a condition fails, the partial step is discarded, a checkpoint of the last completed step is written, and the run waits. Plugging back in resumes from the checkpoint. The resumed run is bit-identical to an uninterrupted one (`pause_resume_is_exact`). A checkpoint whose data or config changed is refused, and the run restarts.

`agentd` publishes the state as JSON, which the trainer reads with `--status`:

```json
{"stage":"deep","charging":true,"battery_pct":83,"temp_c":31.5}
```

An unreadable state counts as "not allowed". `agentd` launches `rust_trainer run` under its existing chip-priority Training mode. The trainer writes `adapters/work/status.json` (`waiting`/`training`/`evaluating`, with step counts) for the UI.

## 4. Eval gate, swap, rollback

When training finishes, the candidate is measured on held-out confirmed examples against the **current** model: the base plus the active adapter, or the base alone if none is active. It is promoted only if:

* there are at least `min_examples` held-out examples,
* mean answer-token NLL falls by ≥ `min_rel_improvement` (default 2 %),
* greedy answer accuracy does not fall, and
* preference accuracy on held-out corrections does not fall.

Otherwise the candidate is deleted. The registry keeps the last 5 accepted adapters. `rollback` steps back through them to the base model.

```
<model_dir>/adapters/
  ACTIVE  history.json  ledger.jsonl  <id>/  work/
```

`ledger.jsonl` gets one line per event: `started`, `resumed`, `paused` (with reason), `deferred`, `restarted`, `skipped`, `accepted`, `rejected` (with the full eval report), and `rollback`.

## 5. Never touches the approval rules

The trainer reads only the training set and the model, and writes only `<model_dir>/adapters/`. It has no access to the executor's tiers or approval key, and nothing it produces is more than additive weight deltas. Whatever the model learns, approval stays with the executor.

## Adapter format

```
adapters/<id>/adapter_model.safetensors   base_model.model.<module>.lora_A.weight [r×in], lora_B.weight [out×r]  (F32)
adapters/<id>/adapter_config.json         PEFT LoRA config
adapters/<id>/trainer_meta.json           base fingerprint, data size, steps, loss, eval report
```

This is exactly the layout Hugging Face PEFT writes. llama.cpp's `convert_lora_to_gguf.py` reads it, which covers the optional llama.cpp compatibility. In `agentos-llm`, apply it per projection as `y += (α/r)·B·(A·x)` (`adapter::LoraAdapter::apply`), or merge it once into dense weights (`merge_into`). The base fingerprint (config + module shapes) is checked on load, so an adapter is never applied to a different model.

## How the backward pass is built

* **Frozen matmuls:** gradient to the input only, `dx = dy·W`, computed straight from the compressed weights.
* **LoRA:** `dA = s·(dy·B)ᵀ·x`, `dB = s·dyᵀ·(A·x)`, `dx += s·(dy·B)·A`.
* **RMSNorm, L2-norm, SwiGLU, sigmoid gates, softplus:** closed-form input gradients. The norm weights are frozen.
* **Attention:** softmax backward per head (parallel), GQA reduction into the shared KV heads, and RoPE backward as the inverse rotation.
* **Gated DeltaNet:** state `S∈R^{dv×dk}` per value head. Forward: `S'=αS`, `u=v−S'k`, `S=S'+βukᵀ`, `o=Sq`. Backward, through time:
  `dS += do·qᵀ`, `dq = Sᵀdo`, `w = dS·k`, `dv = βw`, `dβ = u·w`, `dk = β(dSᵀu − S'ᵀw)`, `dS' = dS − βwkᵀ`, `dα = ⟨dS', S_{t−1}⟩`, `dS_{t−1} = α·dS'`.
  Only every `bptt_chunk`-th state is stored. The states inside a chunk are recomputed during the backward pass, so memory is `O((T/c + c)·dv·dk)` per head, not `O(T·dv·dk)`.
* **Gradient checkpointing:** only each block's input is kept from the forward pass. Each block is recomputed just before its backward.
* **Head:** logits are computed only at answer positions, in chunks, so no `[T × vocab]` matrix is ever built.

### Tests

`cargo test` runs these:

* `tests/gradcheck.rs` checks every op and every mixer variant (dense attention with bias; gated, QK-normed partial-RoPE attention; DeltaNet with BPTT chunk sizes 1, 3, 4, 7 and 20) against central finite differences. It also checks LoRA gradients end-to-end through dense and hybrid models in f32 and Q8_0, and confirms that activation recomputation gives the same gradients as storing activations.
* `tests/training.rs` checks that loss falls and the gate accepts; that pause/resume is exact; that DPO moves preferences (pairwise accuracy → 100%); that the adapter round-trips and engine `apply`/`merge` agree with training; that the overnight flow works (wait → train → heat pause → resume → accept; a second night rejected against the current adapter; skip; defer; rollback; exact ledger sequence); and the power-policy hysteresis.
* Unit tests cover the data filter, chat formatting and masking, the safetensors round-trip, the registry, the LR schedule and Q8_0.

## Feasibility on a phone (estimates, not measured on device)

| | Qwen3.5-4B (r=8) |
|---|---|
| Frozen weights | ≈ 2.7 GB in the engine's compressed format (memory-mapped) |
| Adapter + grads + AdamW | 244 MB |
| Activations | ≈ 0.5–1 GB at T≈512 with per-block recomputation |
| Compute | ≈ 34 GFLOP per trained token with recomputation (≈ 25 without) |

200 examples × 100 tokens × 2 epochs ≈ 1.3 PFLOP, so roughly 1–4 h on a phone CPU. Qwen3.5-2B/0.8B is 2–5× faster.

## Integration points still to wire up

`Rust_Interference_engine` and `agenticOS` were empty when this was written, so these are interfaces, not calls:

1. **Engine weights.** Implement `frozen::FrozenMatmul` over the engine's compressed tensors (forward = its existing matmul; `backward_input` = the same kernel transposed) and build `Model` from them instead of `hf::load_model`.
2. **Engine adapter.** At load time, read `adapters/ACTIVE` and apply `adapters/<id>` via `LoraAdapter::apply` (or merge).
3. **Tokenizer.** Either `agentd` exports `*_ids` produced by the engine's tokenizer, or implement `tokenizer::Tokenizer` over it.
4. **agentd.** Write the device-state JSON, start `rust_trainer run` in the charging Deep stage under the Training mode, and surface `status`/`ledger.jsonl`. `learn eval` can call `rust_trainer eval`.

## Roadmap (from the plan)

1. ✅ Prove backpropagation: gradient checks, then training loss falling on a toy set (dense and hybrid).
2. ✅ DeltaNet backward pass for Qwen3.5.
3. ⏳ Measured improvement on real held-out corrections on the Mac, through the eval gate.
4. ⏳ Overnight scheduling, swap and rollback in `agentd`, then a run in the VM. The trainer side is done; the agentd side is pending.
