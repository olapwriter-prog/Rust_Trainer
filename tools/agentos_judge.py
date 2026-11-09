#!/usr/bin/env python3
"""Judge an adapter the way agentd does, on agentd's own engine.

Asks a running `agentos-llm` server (optionally started with `--lora ADAPTER`)
the held-out requests of agentd's export: every `holdout`-th chat after
de-duplication, exactly as `agentos-infer::finetune` splits them. Each is
answered greedily under agentd's grammar and scored like agentd: 1 for the exact
action, 0.5 for the right tool, 0 otherwise.

usage: agentos_judge.py URL sft.jsonl grammar.gbnf [--holdout 5]
e.g.   agentos-llm -m qwen3.gguf --port 8090 --lora adapter.new.lora &
       agentos_judge.py http://127.0.0.1:8090 sft.jsonl grammar.gbnf
"""
import argparse, json, urllib.request

def read_chats(path):
    seen, chats = set(), []
    for line in open(path):
        try:
            v = json.loads(line)
        except ValueError:
            continue
        m = v.get("messages") or []
        get = lambda r: next((x["content"] for x in m if x.get("role") == r), None)
        user, ans = get("user"), get("assistant")
        if user is None or ans is None:
            continue
        key = user.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        chats.append((get("system") or "", user, ans))
    return chats

def score(got, want):
    try:
        g, w = json.loads(got.strip()), json.loads(want.strip())
    except ValueError:
        return 1.0 if got.strip() == want.strip() else 0.0
    if g == w:
        return 1.0
    if isinstance(g, dict) and isinstance(w, dict) and g.get("tool") == w.get("tool"):
        return 0.5
    return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("data")
    ap.add_argument("grammar")
    ap.add_argument("--holdout", type=int, default=5)
    a = ap.parse_args()
    grammar = open(a.grammar).read()
    n = max(2, a.holdout)
    held = [c for i, c in enumerate(read_chats(a.data)) if i % n == n - 1]
    total = 0.0
    for system, user, want in held:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        body = json.dumps({"messages": msgs, "max_tokens": 128, "temperature": 0, "grammar": grammar}).encode()
        req = urllib.request.Request(a.url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        got = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]
        s = score(got, want)
        total += s
        print(f"{s:.1f}  {user!r}\n     got  {got}\n     want {want}")
    print(json.dumps({"held_out": len(held), "score": total / max(1, len(held))}))

if __name__ == "__main__":
    main()
