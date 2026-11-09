# agentd ↔ rust_trainer contract

`agentd` owns the device stage, the chip-priority Training mode, the confirmed-data export and the approval rules. `rust_trainer` owns training, the eval gate and the adapter registry. These are the four points of contact.

## 1. Training data: `agentd ctl learn` export

JSON Lines, one record per confirmed example:

```jsonl
{"id":"r17","prompt":"dim the hall lamp","response":"{\"act\":\"dim\",\"dev\":\"lamp\",\"room\":\"hall\"}","confirmed":true}
{"id":"c03","prompt":"clean up downloads","chosen":"fs.trash(~/Downloads/*.tmp)","rejected":"rm -rf ~/Downloads","status":"corrected"}
{"id":"m09","confirmed":true,"system":"…","messages":[{"role":"user","content":"…"},{"role":"assistant","content":"…"}]}
```

* **Only confirmed records are used.** That means `"confirmed": true`, or `"status"` of `confirmed`, `approved` or `corrected`. Everything else, including the model's own proposals, is dropped and counted in the log line `dropped N unconfirmed`.
* `id` must be stable across exports. It decides the train/held-out split (hash), so an example never moves between sides. Set `"split": "heldout"` to pin one.
* Corrections should be exported as `chosen`/`rejected` pairs. Training with `--dpo 0.1` uses them as preferences. Without `--dpo`, `chosen` is used as a supervised answer.
* Text fields may instead be given as token ids (`prompt_ids`, `response_ids`, …). For GGUF models this is unnecessary: the trainer tokenises with the engine's own tokenizer and chat template.

## 2. Device state: status file

`agentd` keeps a small JSON file up to date (atomic write, at least every 30 s):

```json
{"stage":"deep","charging":true,"battery_pct":83,"temp_c":31.5}
```

`stage` is one of `active | idle | light | deep`. The trainer reads it before **every training example**. It pauses, with a checkpoint, as soon as any of these fails:

* stage is `deep`
* charging is true
* `battery_pct ≥ --min-battery` (default 50)
* `temp_c < --max-temp` (default 40)

A paused run restarts only at 3 °C below the heat limit and 5 % above the battery limit. A missing or unreadable file counts as "not allowed". So to pause training, `agentd` only has to update the file; it never needs to kill the process.

## 3. Launching

When the device enters the charging Deep stage, start (under the existing Training mode):

```sh
rust_trainer run \
  --model /data/models/qwen3-4b-q4_0.gguf \
  --data  /data/agentd/learn/confirmed.jsonl \
  --status /run/agentd/device.json \
  --hours 7 --poll-secs 60 \
  --threads 6 \
  --min-train 50 --epochs 2 --rank 8
```

* `--hours` is the deadline. If the conditions never allow finishing, the run exits with `"outcome": "deferred"` and keeps its checkpoint. The next night resumes from it, or restarts if the export has grown, which is logged as `restarted`.
* stdout ends with one JSON object: `accepted` (with the new id and eval report), `rejected`, `deferred` or `skipped` (too few confirmed examples).
* After `accepted`, tell the LLM service to reload. The engine picks up `<model>.adapters/ACTIVE` (see `ENGINE_LORA.md`).
* Progress for the UI is in `<model>.adapters/work/status.json`: `phase` (`waiting`, `training` or `evaluating`), `step`, `total_steps`, `detail`.
* The process uses `panic = "abort"`. If it has to be killed, at most `--checkpoint-every` steps (default 10) are lost. Prefer pausing through the status file.

## 4. Ledger, eval and rollback

* `<model>.adapters/ledger.jsonl` is append-only, one JSON object per event: `started`, `resumed`, `paused` (with reason), `deferred`, `restarted`, `skipped`, `accepted`, `rejected` (both with the full eval report), and `rollback`. `agentd` can mirror it into its own ledger.
* `learn eval` → `rust_trainer eval --model M --adapter <dir> --data confirmed.jsonl` prints the eval report as JSON.
* User-initiated undo → `rust_trainer rollback --model M --reason "…"`. This steps back through the last 5 accepted adapters to the base model.
* `rust_trainer status --model M` shows the active adapter, history, current run and the last ledger entries.

## Approval rules

The trainer reads only the model and the export, and writes only `<model>.adapters/`. It never sees the executor's tiers or the approval key. An adapter only changes model weights, so whatever the model learns, approval stays with the executor.
