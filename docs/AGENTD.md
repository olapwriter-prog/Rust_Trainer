# Using rust_trainer from agentd (agenticOS)

agentd already schedules overnight training (`crates/agentd/src/trainer.rs`) and runs `agentos-llm train` for it. `rust_trainer train` speaks **the same job protocol**, so agentd can run it instead, unchanged apart from which binary it starts.

## What agentd does (as implemented in agenticOS)

1. **Conditions:** it trains only on the charger, with enough battery, below the heat limit, when the model isn't busy, in the Deep stage, and once enough new confirmed requests exist. A forced `learn train` skips only the schedule, never battery or heat.
2. **Export:** it writes `learn/sft.jsonl` (one chat per confirmed episode, `{"messages":[system, user, assistant]}` with its real system prompt) and `learn/grammar.gbnf` (one rule per tool). Only human-confirmed answers are exported (`Learner::target`), so the model never trains on its own unconfirmed output.
3. **Launch:**
   ```
   <bin> train -m MODEL.gguf --data learn/sft.jsonl --grammar learn/grammar.gbnf \
               --out learn/adapter.new.lora --steps 60 --max-seconds 1800
   ```
4. **Stopping:** it kills the process when the phone is unplugged, warm or needed. The next night it launches the same command again.
5. **Result:** when the process exits it reads the **last JSON line on stdout**:
   * `{"done":false,"step":..}` → paused; continue next time.
   * `{"done":true,"base":..,"tuned":..,"kept":true}` → `adapter.new.lora` is promoted to `adapter.lora` (the old one becomes `adapter.prev.lora` for `learn rollback`) and the model server restarts with `--lora`.
   * `"kept":false` → discarded.

## What rust_trainer does with that command

`rust_trainer train` switches to this mode when it sees `--steps` or `--grammar`. It behaves exactly like `agentos-infer::finetune::run`:

| | agentos-llm train | rust_trainer train |
|---|---|---|
| chats | `messages` → (system, user, assistant); a request confirmed twice counts once | same |
| held out | every 5th chat (`--holdout`) | same |
| answer form | grammar's key order (`"tool"` first) | same (key order read from the tool's grammar rule) |
| prompt | ChatML + `<think>\n\n</think>\n\n` for Qwen3/3.5 (non-thinking) | same |
| loss | answer tokens + `<|im_end|>` | same |
| defaults | 60 steps, batch 4, rank 8, α 16, lr 1e-3 | same |
| adapters on | attention q/k/v/o of attention layers | same |
| checkpoint | every step, `OUT.ckpt` + `OUT.ckpt.json` | every step, `OUT.ckpt` (atomic; LoRA + AdamW moments + job state) + `OUT.ckpt.json`. A checkpoint from different data or settings is not resumed, and the reason is logged |
| `--max-seconds` | stop with checkpoint, `{"done":false,...}` | same |
| judging | grammar-constrained greedy decode: 1 exact / 0.5 right tool | greedy decoding evaluated exactly from one teacher-forced pass, without the grammar (a lower bound on the constrained score), 1 / 0.5 |
| keep | `tuned > base` → write `--out` | same |
| output | `AGLORA1` | `AGLORA1` (loaded unchanged by `agentos-llm --lora`), plus `OUT.gguf` (llama.cpp-style LoRA GGUF for the Rust Inference Engine) |
| summary | `{"done":true,"base","tuned","kept","step","loss","train","held_out","adapter","secs"}` | same keys |

The forward pass runs through the Rust Inference Engine's kernels, bit-exact with that engine. Supported models are GGUF Llama, Qwen2/2.5 and Qwen3. Qwen3.5 (Gated DeltaNet) needs the engine to support it first; the trainer's DeltaNet backward is ready on the safetensors path.

## Verified against agenticOS

The test used agentd's real export (`cargo run -p agentd --example prompt -- crates/infer/examples/phone-requests.tsv DIR`: 60 confirmed phone requests, a 2,867-character system prompt and a 35-tool grammar) on real Qwen3-0.6B Q4_0.

* **Launch 1** (`--max-seconds 240`): judged the base model (0 %), trained 2 steps, printed `{"done":false,"step":2,...}` and left the checkpoint.
* **Launch 2:** printed "continuing from step 2", trained to step 12, judged 29 %, printed `"kept":true`, and wrote `adapter.new.lora` (AGLORA1, 9.2 MB) and `adapter.new.gguf`.
* **agentd's own engine** loaded it: `agentos-llm --lora adapter.new.lora` reported "2293760 parameters". On agentd's 12 held-out requests, with agentd's grammar and scoring (`tools/agentos_judge.py`):

| agentos-llm, Qwen3-0.6B Q4_0 | score on 12 held-out requests |
|---|---|
| no adapter | 4.2 % |
| adapter from `rust_trainer train`, 12 steps (24 example passes) | 25 % |
| adapter from `rust_trainer train`, 30 steps of batch 2 (as agentos-llm's own trainer was measured), in 3 time-boxed launches like agentd's nights | **75 %** |

The 30-step adapter answers 8 of 12 held-out requests exactly and picks the right tool for 11 of 12; the remaining misses are argument details. `rust_trainer`'s own judge (unconstrained greedy) also gave 75 %, the same as agentd's grammar-constrained judge on agentos-llm. For reference, agenticOS's notes record its own trainer taking Qwen2.5-0.5B from 46 % to 67 % with the same step count (a different base model, so not directly comparable).

`tests/agentd.rs` checks the protocol on a synthetic GGUF: a time-boxed launch, then a resumed launch that is **bit-identical** to an uninterrupted job. It also checks that the base score is carried over, the checkpoint is removed at the end, and the AGLORA1 round-trip reproduces the trained model exactly.

## Switching agentd to rust_trainer

agentd starts the trainer with `cfg.llama_server`, the same binary that serves the model (`crates/agentd/src/core.rs`, `self.trainer.start(&bin, …)`). To use this trainer, give it its own config key in agentd, for example:

```toml
# agentd.toml
trainer = "/usr/bin/rust_trainer"   # default: llama_server (agentos-llm)
```

and pass `self.cfg.trainer.clone().unwrap_or(self.cfg.llama_server.clone())` to `trainer.start`. That is the only change agentd needs. `-t` (threads) and `--no-pin` are accepted too; `--memory-mb` is not needed, because layers are always recomputed in the backward pass.

## Approval rules

Unchanged: the trainer reads the export and the model and writes only `--out`, its checkpoint and its `.gguf` sidecar. Tiers and the approval key stay with the executor.
