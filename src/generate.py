"""Qualitative generation from a native checkpoint (base or SFT).

    # base completion
    python -m src.generate --ckpt out/run1/final.pt --prompt "The moon is"

    # chat (matches the SFT template)
    python -m src.generate --ckpt out/sft/final.pt --chat \
        --prompt "Explain photosynthesis in one sentence."
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from .model import build_config, build_model


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature, top_k, rep_penalty, eos_id, block_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(idx_cond)
        logits = logits[:, -1, :] / max(1e-5, temperature)
        if rep_penalty != 1.0:
            # suppress tokens already present in the context (standard repetition penalty)
            for b in range(idx.size(0)):
                for tok_id in set(idx[b].tolist()):
                    logits[b, tok_id] /= rep_penalty
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        idx = torch.cat([idx, nxt], dim=1)
        if eos_id is not None and nxt.item() == eos_id:
            break
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    ap.add_argument("--prompt", default="The meaning of life is")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--block_size", type=int, default=1024)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    eos_id = tok.eos_token_id or 2

    cfg = build_config(vocab_size=args.vocab_size, block_size=args.block_size)
    model = build_model(cfg).to(device).eval()
    sd = torch.load(args.ckpt, map_location="cpu")["model"]
    model.load_state_dict(sd)
    model.lm_head.weight = model.transformer.wte.weight

    if args.chat:
        text = f"<|user|>\n{args.prompt}\n<|assistant|>\n"
    else:
        text = args.prompt
    ids = tok.encode(text, add_special_tokens=True)
    idx = torch.tensor([ids], device=device)
    out = generate(model, idx, args.max_new_tokens, args.temperature,
                    args.top_k, args.repetition_penalty, eos_id, args.block_size)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
