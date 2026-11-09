# Open-source test sandbox

Every overnight scenario can be tested end to end without a phone and without private data, using only open-source pieces that run on a plain Linux box (no GPU, no KVM).

| Real thing | Open-source stand-in | How |
|---|---|---|
| Phone CPU (ARM: i8mm / dotprod / Cortex-A53) | **QEMU** user-mode AArch64 | `cargo test --target aarch64-unknown-linux-gnu` (see CI) |
| The model on the phone | **Qwen3-0.6B Q4_0 GGUF** from Docker's open model catalog | `tools/fetch_docker_model.py ai/qwen3:0.6B-Q4_0 qwen3.gguf` |
| Your confirmed requests and corrections | **SNIPS NLU benchmark** (sonos/nlu-benchmark): 7 skills, utterances with slot labels → structured actions | `tools/snips_to_jsonl.py` |
| agentd's device monitor during a night | **`tools/phone_sim.py`**: stage, charging, battery, unplug events and a thermal model driven by the trainer's measured CPU use | wraps `rust_trainer run` |
| "Does it do the right thing?" | **`rust_trainer act-eval`**: exact greedy action match and intent accuracy on held-out requests | one teacher-forced pass per example |

Why these:

* **Real models.** Hugging Face and the other model hubs are often blocked in CI sandboxes. Docker Hub's `ai/*` models (Qwen3, Qwen2.5, SmolLM2, …) are ordinary GGUF files stored as OCI artifacts. The fetcher uses only the public registry API with an anonymous token and checks the SHA-256 digest.
* **No virtual phone.** An Android emulator (AOSP emulator, Cuttlefish) needs KVM, which containers usually lack. For the trainer, the relevant phone properties are the CPU instruction set (covered by QEMU and the engine's runtime kernel dispatch) and the night's conditions (covered by the simulator).

## Reproduce

```sh
cargo build --release
mkdir -p sbx && cd sbx
python3 ../tools/fetch_docker_model.py ai/qwen3:0.6B-Q4_0 qwen3.gguf
git clone --depth 1 https://github.com/sonos/nlu-benchmark
python3 ../tools/snips_to_jsonl.py nlu-benchmark snips.jsonl --per-skill 20 --heldout-per-skill 6 --unconfirmed 3 --corrections 4

# the trainer's forward is bit-exact with the engine on the real model
RT_GGUF=$PWD/qwen3.gguf cargo test --release --test engine real_model -- --ignored

# a night: Light→Deep, low battery at first, unplugged for 45 s at t=240 s
python3 ../tools/phone_sim.py --status dev.json --log night.csv --cores 4 --unplug-at 240 --unplug-for 45 -- \
  ../target/release/rust_trainer run --model qwen3.gguf --data snips.jsonl --heldout 0 --status dev.json \
  --threads 4 --no-pin --epochs 2 --batch 4 --lr 5e-4 --rank 8 --max-len 128 --min-train 20 --poll-secs 5

# a hot phone (thermal limit triggers) training a mixture of experts
python3 ../tools/phone_sim.py --status dev.json --heat 0.32 --battery 80 -- \
  ../target/release/rust_trainer run … --experts 7 --min-expert-examples 15

../target/release/rust_trainer act-eval --model qwen3.gguf --data snips.jsonl --heldout 0 \
  --adapter qwen3.adapters/$(cat qwen3.adapters/ACTIVE)
```

## Results (4 x86 vCPUs, real Qwen3-0.6B Q4_0, 140 confirmed training requests, 42 held out)

**Engine parity:** all 15 × 151,936 logits identical between the trainer and the engine (max |Δ| = 0).

**Data safety:** the 21 unconfirmed "proposed" records were dropped every time, and the 28 corrections were parsed as preference pairs.

**Night 1: single adapter, unplug event.**

| t | event |
|---|---|
| 0–15 s | Light stage: waiting |
| 15–40 s | Deep, but battery below 55 % (46 % → 55 % while charging): waiting |
| 40 s | training starts; loss 2.46 → 1.08 by step 10 |
| 240 s | unplugged: paused within one example at step 17/70, CPU 0 %, phone cools 38.7 → 32.5 °C |
| 285 s | plugged in: resumed at step 17 from the checkpoint |
| 970 s | finished, gated, **accepted** |

| held-out (42 requests, never seen) | NLL | token acc. | **exact action** | **intent** |
|---|---|---|---|---|
| base Qwen3-0.6B | 4.67 | 38.8 % | 0 % | 0 % |
| after one night | 0.17 | 95.3 % | **42.9 %** | **97.6 %** |

**Night 2: mixture of experts on a hot phone.**

* The thermal limit triggered **22 pauses**: pause at 40 °C, resume below 37 °C, roughly every 70 s.
* The general expert trained first, then 7 skill experts (by label).
* The per-expert gate kept 3 skill experts (weather, book rating, showtimes) and sent 4 skills to the general expert.
* The mixture was accepted: held-out NLL 0.163 (vs 0.169 single), exact action 38.1 %, intent 95.2 %.
* That is no meaningful difference from the single adapter on this data, and it took 2.4× the wall time.

## Issues the sandbox found, and the fixes

| | before | after |
|---|---|---|
| Peak temperature on the hot phone (limit 40 °C) | **41.4 °C** | **40.02 °C** for the whole night |

1. **The trainer only checked the device between examples.** That is 3–4 s per example here, and 30–60 s for a 4B model on a phone, so heat kept building until the example finished. It now also checks between layers in the forward and backward pass, and abandons an example mid-way if needed. The partial step is discarded, so resumption stays bit-exact (`tests/training.rs::abort_inside_an_example_resumes_exactly`). Training crossings now stop at 40.00–40.03 °C.
2. **Held-out evaluation ignored the power policy.** That was about 30 s at full CPU at the end of the night, and it reached 40.9 °C. Evaluation now checks the device before each example and waits through cool-downs or unplugging. If the deadline passes during evaluation, the night is deferred: the finished checkpoint and trained experts are kept, and the next night goes straight to evaluation without retraining (`tests/training.rs::evaluation_also_respects_the_power_policy`).

The rerun on the hot phone (real Qwen3-0.6B, 1 epoch) paused 6 times for heat during training and twice during evaluation, peaked at **40.02 °C**, and was accepted (held-out NLL −92.6 %).
