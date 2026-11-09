#!/usr/bin/env python3
"""Parity check of rust_trainer against Hugging Face transformers + peft.

Builds small random-weight Qwen2 / Qwen3.5 / Qwen3-Next models in transformers,
attaches a random PEFT LoRA adapter, and compares with our Rust implementation:
  * per-token log-probs of the base model and of the adapted model,
  * the mean answer-token loss,
  * the gradient of that loss w.r.t. every LoRA tensor (torch autograd vs our backward).

usage: python tools/parity.py [qwen2] [qwen3_5] [qwen3_next]
Needs: torch, transformers, peft, and `cargo build --release --example parity`.
"""
import json
import os
import subprocess
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "target", "release", "examples", "parity")
V = 300


def build(kind):
    torch.manual_seed(0)
    common = dict(vocab_size=V, hidden_size=64, intermediate_size=128, num_attention_heads=4,
                  num_key_value_heads=2, tie_word_embeddings=True, max_position_embeddings=256,
                  rms_norm_eps=1e-6)
    if kind == "qwen2":
        from transformers import Qwen2Config, Qwen2ForCausalLM
        m = Qwen2ForCausalLM(Qwen2Config(num_hidden_layers=2, rope_theta=10000.0, **common))
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif kind == "qwen3_5":
        import transformers
        cfg_cls = getattr(transformers, "Qwen3_5TextConfig")
        model_cls = getattr(transformers, "Qwen3_5ForCausalLM")
        cfg = cfg_cls(num_hidden_layers=4, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                      linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
                      layer_types=["linear_attention"] * 3 + ["full_attention"], **common)
        m = model_cls(cfg)
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                   "in_proj_qkv", "in_proj_z", "out_proj"]
    elif kind == "qwen3_next":
        from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
        cfg = Qwen3NextConfig(num_hidden_layers=4, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                              linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
                              num_experts=0, mlp_only_layers=[0, 1, 2, 3],
                              layer_types=["linear_attention"] * 3 + ["full_attention"], **common)
        m = Qwen3NextForCausalLM(cfg)
        # Fused in_proj_qkvz is split on load, so only unfused modules are adapted here.
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "out_proj"]
    else:
        raise SystemExit(f"unknown kind {kind}")
    # Larger-than-default random weights so the comparison is not dominated by ~uniform logits.
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() == 2:
                p.normal_(0.0, 0.08)
            elif "norm" in n:
                p.add_(torch.empty_like(p).uniform_(-0.2, 0.2))
    return m.float().eval(), targets


def main(kinds):
    from peft import LoraConfig, get_peft_model

    ok = True
    for kind in kinds:
        tmp = tempfile.mkdtemp(prefix=f"parity-{kind}-")
        mdir, adir = os.path.join(tmp, "model"), os.path.join(tmp, "adapter")
        model, targets = build(kind)
        model.save_pretrained(mdir, safe_serialization=True)

        g = torch.Generator().manual_seed(1)
        tokens = torch.randint(0, V, (37,), generator=g).tolist()
        start = 20
        r, alpha = 4, 8.0
        pm = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, target_modules=targets,
                                              lora_dropout=0.0, init_lora_weights=False))
        ids = torch.tensor([tokens])
        pos = torch.arange(start - 1, len(tokens) - 1)
        tgt = torch.tensor(tokens[start:])
        with torch.no_grad(), pm.disable_adapter():
            base = torch.log_softmax(pm(ids).logits[0].float(), -1)[pos, tgt]
        logits = pm(ids).logits[0].float()
        lp = torch.log_softmax(logits, -1)[pos, tgt]
        loss = -lp.mean()
        loss.backward()
        pm.save_pretrained(adir)
        grads = {n.replace(".default", ""): p.grad.detach().flatten().tolist()
                 for n, p in pm.named_parameters() if "lora_" in n}

        case = os.path.join(tmp, "case.json")
        out = os.path.join(tmp, "rust.json")
        with open(case, "w") as f:
            json.dump({"tokens": tokens, "answer_start": start, "r": r, "lora_alpha": alpha}, f)
        subprocess.run([BIN, mdir, adir, case, out], check=True)
        rs = json.load(open(out))

        def maxdiff(a, b):
            a, b = torch.tensor(a), torch.tensor(b)
            return (a - b).abs().max().item(), b.abs().max().item()

        d_base, _ = maxdiff(rs["base_logprobs"], base.tolist())
        d_lp, _ = maxdiff(rs["logprobs"], lp.tolist())
        d_loss = abs(rs["loss"] - loss.item())
        worst = 0.0
        for n, gt in grads.items():
            if n not in rs["grads"]:
                print(f"  {kind}: missing grad {n}")
                ok = False
                continue
            d, scale = maxdiff(rs["grads"][n], gt)
            worst = max(worst, d / max(scale, 1e-8))
        good = d_base < 2e-4 and d_lp < 2e-4 and d_loss < 1e-4 and worst < 2e-3 and len(grads) == len(rs["grads"])
        ok &= good
        print(f"{kind:>10}: base logp Δ {d_base:.2e} | adapted logp Δ {d_lp:.2e} | loss Δ {d_loss:.2e} "
              f"| LoRA grads {len(grads)} tensors, worst rel Δ {worst:.2e}  {'OK' if good else 'MISMATCH'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["qwen2", "qwen3_5", "qwen3_next"])
