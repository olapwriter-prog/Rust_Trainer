#!/usr/bin/env python3
"""Independent float64 check of the trainer's forward pass and hand-written gradients.

Reads the file written by `examples/export_reference.rs`: a real model's
(dequantised) weights, a LoRA adapter on every projection, one example, and the
trainer's loss and analytic LoRA gradients. Re-implements the Qwen/Llama decoder
from the published model definition in NumPy float64, sharing no code with the
Rust trainer, and checks:

  1. the loss matches the trainer's;
  2. the trainer's gradient, projected on random directions over *all* adapter
     weights, matches float64 central differences of the reference;
  3. individual gradient entries in the first, middle and last layers (A and B of
     every projection type) match float64 central differences.

usage: reference_check.py EXPORT.safetensors [--directions 4] [--coords 24] [--seed 0]
Needs only numpy. Exit status 1 if any check fails.
"""
import argparse, json, struct, sys
import numpy as np


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", {})
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
    tensors = {}
    for name, info in header.items():
        assert info["dtype"] == "F32", info["dtype"]
        a, b = info["data_offsets"]
        tensors[name] = np.frombuffer(mm[a:b], dtype="<f4").reshape(info["shape"])
    return tensors, json.loads(meta["reference"])


class Reference:
    def __init__(self, t, cfg):
        self.cfg = cfg
        f64 = lambda x: np.asarray(x, dtype=np.float64)
        self.W = {k: f64(v) for k, v in t.items() if not k.endswith(("grad_a", "grad_b", "lora_a", "lora_b")) and k != "tokens"}
        self.lora = {n: [f64(t[n + ".lora_a"]), f64(t[n + ".lora_b"])] for n in cfg["lora"]}
        self.tokens = t["tokens"].astype(np.int64)
        hd, rot = cfg["head_dim"], cfg["rot_dim"]
        T = len(self.tokens)
        inv = cfg["rope_theta"] ** (-np.arange(0, rot // 2) * 2.0 / rot) * cfg["rope_freq_scale"]
        ang = np.arange(T)[:, None] * inv[None, :]
        self.cos, self.sin = np.cos(ang), np.sin(ang)

    def linear(self, name, x):
        y = x @ self.W[name + ".weight"].T
        if name + ".bias" in self.W:
            y = y + self.W[name + ".bias"]
        if name in self.lora:
            A, B = self.lora[name]
            y = y + self.cfg["lora_scale"] * ((x @ A.T) @ B.T)
        return y

    def rmsnorm(self, x, w):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.cfg["rms_eps"]) * w

    def rope(self, x):  # x: [T, heads, hd]
        rot, half = self.cfg["rot_dim"], self.cfg["rot_dim"] // 2
        c, s = self.cos[:, None, :], self.sin[:, None, :]
        out = x.copy()
        if self.cfg["rope_neox"]:
            x1, x2 = x[..., :half], x[..., half:rot]
            out[..., :half] = x1 * c - x2 * s
            out[..., half:rot] = x2 * c + x1 * s
        else:
            x1, x2 = x[..., 0:rot:2], x[..., 1:rot:2]
            out[..., 0:rot:2] = x1 * c - x2 * s
            out[..., 1:rot:2] = x2 * c + x1 * s
        return out

    def loss(self):
        cfg = self.cfg
        T, nh, nkv, hd = len(self.tokens), cfg["n_head"], cfg["n_kv_head"], cfg["head_dim"]
        x = self.W["token_embd"][self.tokens]
        mask = np.triu(np.full((T, T), -np.inf), 1)
        for i in range(cfg["n_layer"]):
            p = f"blk.{i}"
            h = self.rmsnorm(x, self.W[p + ".attn_norm"])
            q = self.linear(p + ".attn_q", h).reshape(T, nh, hd)
            k = self.linear(p + ".attn_k", h).reshape(T, nkv, hd)
            v = self.linear(p + ".attn_v", h).reshape(T, nkv, hd)
            if p + ".attn_q_norm" in self.W:
                q = self.rmsnorm(q, self.W[p + ".attn_q_norm"])
                k = self.rmsnorm(k, self.W[p + ".attn_k_norm"])
            q, k = self.rope(q), self.rope(k)
            rep = nh // nkv
            k, v = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
            scores = np.einsum("thd,shd->hts", q, k) / np.sqrt(hd) + mask
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            o = np.einsum("hts,shd->thd", probs, v).reshape(T, nh * hd)
            x = x + self.linear(p + ".attn_output", o)
            h = self.rmsnorm(x, self.W[p + ".ffn_norm"])
            g, u = self.linear(p + ".ffn_gate", h), self.linear(p + ".ffn_up", h)
            x = x + self.linear(p + ".ffn_down", g / (1.0 + np.exp(-g)) * u)
        start = cfg["answer_start"]
        pos = np.arange(start - 1, T - 1)
        hfin = self.rmsnorm(x[pos], self.W["output_norm"])
        out = self.W["output"] if "output" in self.W else self.W["token_embd"]
        logits = hfin @ out.T
        m = logits.max(axis=-1, keepdims=True)
        lse = (m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True)))[:, 0]
        return float(np.mean(lse - logits[np.arange(len(pos)), self.tokens[pos + 1]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--directions", type=int, default=4)
    ap.add_argument("--coords", type=int, default=42)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t, cfg = read_safetensors(a.export)
    ref = Reference(t, cfg)
    grads = {n: [np.asarray(t[n + ".grad_a"], np.float64), np.asarray(t[n + ".grad_b"], np.float64)] for n in cfg["lora"]}
    rng = np.random.default_rng(a.seed)
    ok = True

    L = ref.loss()
    rel = abs(L - cfg["loss"]) / abs(L)
    print(f"loss: reference {L:.8f}  trainer {cfg['loss']:.8f}  rel diff {rel:.2e}")
    ok &= rel < 1e-5

    def perturbed(fn, eps):
        fn(+eps)
        lp = ref.loss()
        fn(-2 * eps)
        lm = ref.loss()
        fn(+eps)
        return (lp - lm) / (2 * eps)

    # 2. directional derivatives over every adapter weight at once. The projection g·d
    # is a sum over millions of terms that largely cancel, so its error is judged
    # against the gradient's norm (|g·d| <= |g| for a unit d), not against g·d itself.
    eps = 1e-4
    gnorm = np.sqrt(sum((g * g).sum() for pair in grads.values() for g in pair))
    for d_i in range(a.directions):
        dirs = {n: [rng.standard_normal(A.shape), rng.standard_normal(B.shape)] for n, (A, B) in ref.lora.items()}
        norm = np.sqrt(sum((x * x).sum() for pair in dirs.values() for x in pair))
        analytic = sum((dirs[n][j] * grads[n][j]).sum() for n in dirs for j in (0, 1)) / norm

        def move(s):
            for n, (A, B) in ref.lora.items():
                A += s * dirs[n][0] / norm
                B += s * dirs[n][1] / norm

        numeric = perturbed(move, eps)
        rel = abs(analytic - numeric) / max(abs(numeric), 1e-12)
        err = abs(analytic - numeric) / gnorm
        print(f"direction {d_i}: trainer {analytic:+.6e}  float64 FD {numeric:+.6e}  rel {rel:.2e}  error/|g| {err:.2e}")
        ok &= err < 1e-5

    # 3. individual entries: first, middle and last layer, every projection, A and B
    n_layer = cfg["n_layer"]
    layers = sorted({0, n_layer // 2, n_layer - 1})
    names = [n for n in cfg["lora"] if int(n.split(".")[1]) in layers]
    worst = 0.0
    for c in range(a.coords):
        n = names[(c // 2) % len(names)]
        j = c % 2
        P, G = ref.lora[n][j], grads[n][j]
        # pick an entry with a gradient that is not negligible, so the relative error means something
        flat = np.abs(G).ravel()
        cand = np.flatnonzero(flat >= np.quantile(flat, 0.9))
        idx = np.unravel_index(rng.choice(cand), P.shape)

        def move(s, P=P, idx=idx):
            P[idx] += s

        numeric = perturbed(move, 1e-3)
        analytic = G[idx]
        rel = abs(analytic - numeric) / max(abs(numeric), 1e-12)
        worst = max(worst, rel)
        print(f"{n}.{'AB'[j]}{[int(i) for i in idx]}: trainer {analytic:+.6e}  float64 FD {numeric:+.6e}  rel {rel:.2e}")
    print(f"worst relative error over {a.coords} entries: {worst:.2e}")
    ok &= worst < 1e-2
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
