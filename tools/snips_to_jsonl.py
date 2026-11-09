#!/usr/bin/env python3
"""Turn the open SNIPS NLU benchmark into an `agentd ctl learn`-style export.

Each utterance becomes a confirmed request whose right answer is a structured
action, which is what agentOS's executor consumes:

    {"id": "GetWeather-17", "prompt": "will it rain in paris tomorrow",
     "response": "{\"intent\":\"GetWeather\",\"slots\":{\"city\":\"paris\",\"timeRange\":\"tomorrow\"}}",
     "confirmed": true, "skill": "getweather"}

To exercise the trainer's safety paths it also emits:
  * unconfirmed proposals (`"status": "proposed"`), which must be dropped,
  * corrections as preference pairs (`chosen` = right action, `rejected` = the
    same request mapped to a wrong intent), when --corrections > 0.

usage: snips_to_jsonl.py NLU_BENCHMARK_DIR OUT.jsonl [--per-skill N] [--heldout-per-skill H]
                         [--skills A,B,...] [--no-labels] [--unconfirmed K] [--corrections C] [--seed S]
Source: https://github.com/sonos/nlu-benchmark (2017-06-custom-intent-engines).
"""
import argparse, json, os, random

def load_intent(root, intent):
    p = os.path.join(root, "2017-06-custom-intent-engines", intent, f"train_{intent}_full.json")
    with open(p, encoding="latin-1") as f:
        return json.load(f)[intent]

def action(intent, parts):
    slots = {}
    for part in parts:
        if "entity" in part:
            slots.setdefault(part["entity"], part["text"].strip().lower())
    return json.dumps({"intent": intent, "slots": slots}, separators=(",", ":"), ensure_ascii=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--per-skill", type=int, default=30)
    ap.add_argument("--heldout-per-skill", type=int, default=8)
    ap.add_argument("--skills", default="")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--unconfirmed", type=int, default=5)
    ap.add_argument("--corrections", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    base = os.path.join(a.root, "2017-06-custom-intent-engines")
    intents = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    if a.skills:
        intents = [i for i in intents if i in a.skills.split(",")]
    n = 0
    with open(a.out, "w") as f:
        for intent in intents:
            items = load_intent(a.root, intent)
            rng.shuffle(items)
            take = items[: a.per_skill + a.heldout_per_skill]
            for i, it in enumerate(take):
                text = "".join(p["text"] for p in it["data"]).strip().lower()
                rec = {"id": f"{intent}-{i}", "prompt": text, "response": action(intent, it["data"]), "confirmed": True}
                if i >= a.per_skill:
                    rec["split"] = "heldout"
                if not a.no_labels:
                    rec["skill"] = intent.lower()
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            # Unconfirmed model proposals: must never be trained on.
            for i, it in enumerate(items[-a.unconfirmed:] if a.unconfirmed else []):
                text = "".join(p["text"] for p in it["data"]).strip().lower()
                f.write(json.dumps({"id": f"{intent}-u{i}", "prompt": text, "response": action("WrongIntent", it["data"]), "status": "proposed"}) + "\n")
            # Corrections: the user fixed a wrong action.
            for i, it in enumerate(items[a.per_skill + a.heldout_per_skill:][: a.corrections]):
                text = "".join(p["text"] for p in it["data"]).strip().lower()
                wrong = rng.choice([x for x in intents if x != intent])
                f.write(json.dumps({"id": f"{intent}-c{i}", "prompt": text, "chosen": action(intent, it["data"]),
                                    "rejected": action(wrong, it["data"]), "status": "corrected",
                                    **({} if a.no_labels else {"skill": intent.lower()})}) + "\n")
    print(f"wrote {a.out}: {n} confirmed requests over {len(intents)} skills")

if __name__ == "__main__":
    main()
