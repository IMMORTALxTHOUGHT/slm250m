"""Phase 4 - light SFT so the base model can follow simple instructions.

Loads a pretrained checkpoint (out/run1/final.pt), fine-tunes 1 epoch on a
permissive instruct set (default: HuggingFaceTB/smoltalk, Apache-2.0) with a
simple chat template and loss masked to assistant turns only.

    python -m src.sft --init out/run1/final.pt --out_dir out/sft \
        --max_samples 50000 --epochs 1

Produces out/sft/final.pt (same checkpoint format as pretraining).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from .model import build_config, build_model

IGNORE = -100


def render_and_mask(messages, tok, eos_id, max_len):
    """Return (input_ids, labels) with loss only on assistant tokens."""
    ids, labels = [], []
    if tok.bos_token_id is not None:
        ids.append(tok.bos_token_id)
        labels.append(IGNORE)
    for m in messages:
        role, content = m.get("role", ""), (m.get("content") or "")
        header = f"<|{role}|>\n"
        h_ids = tok.encode(header, add_special_tokens=False)
        c_ids = tok.encode(content, add_special_tokens=False)
        ids.extend(h_ids); labels.extend([IGNORE] * len(h_ids))
        if role == "assistant":
            c_ids = c_ids + [eos_id]
            ids.extend(c_ids); labels.extend(c_ids)          # supervised
        else:
            nl = tok.encode("\n", add_special_tokens=False)
            ids.extend(c_ids + nl); labels.extend([IGNORE] * (len(c_ids) + len(nl)))
    return ids[:max_len], labels[:max_len]


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="pretrained checkpoint")
    ap.add_argument("--out_dir", default="out/sft")
    ap.add_argument("--dataset", default="HuggingFaceTB/smoltalk")
    ap.add_argument("--dataset_config", default="all")
    ap.add_argument("--data_file", default=None,
                    help="LOCAL instruct file (offline). JSONL (one {'messages':[...]} per "
                         "line) or a JSON array. Also accepts {'conversations':[{'from','value'}]}.")
    ap.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    ap.add_argument("--max_samples", type=int, default=50000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--micro_batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=3e-5)
    ap.add_argument("--warmup_frac", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    return ap.parse_args()


def main():
    args = get_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    eos_id = tok.eos_token_id or 2

    # --- data ---
    examples = []
    if args.data_file:
        print(f"[sft] loading local data file: {args.data_file}")
        with open(args.data_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
        try:
            rows = json.loads(text)            # whole-file JSON array / object
            if isinstance(rows, dict):
                rows = [rows]
        except json.JSONDecodeError:
            rows = [json.loads(l) for l in text.splitlines() if l.strip()]  # JSONL
        for row in rows:
            msgs = row.get("messages") or row.get("conversations")
            if not msgs:
                continue
            if isinstance(msgs, list) and msgs and "from" in msgs[0]:
                msgs = [{"role": ("assistant" if m.get("from") in ("assistant", "gpt")
                                  else "user"),
                         "content": m.get("value") or m.get("content") or ""}
                        for m in msgs]
            ii, ll = render_and_mask(msgs, tok, eos_id, args.block_size)
            if any(l != IGNORE for l in ll):
                examples.append((ii, ll))
    else:
        print(f"[sft] loading {args.dataset}:{args.dataset_config}")
        ds = load_dataset(args.dataset, args.dataset_config, split="train")
        if args.max_samples and len(ds) > args.max_samples:
            ds = ds.shuffle(seed=args.seed).select(range(args.max_samples))
        for row in ds:
            msgs = row.get("messages")
            if not msgs:
                continue
            ii, ll = render_and_mask(msgs, tok, eos_id, args.block_size)
            if any(l != IGNORE for l in ll):
                examples.append((ii, ll))
    print(f"[sft] {len(examples)} usable examples")

    def collate(batch):
        maxlen = max(len(x[0]) for x in batch)
        X = torch.full((len(batch), maxlen), eos_id, dtype=torch.long)
        Y = torch.full((len(batch), maxlen), IGNORE, dtype=torch.long)
        for i, (ii, ll) in enumerate(batch):
            X[i, :len(ii)] = torch.tensor(ii)
            Y[i, :len(ll)] = torch.tensor(ll)
        return X, Y

    # --- model ---
    cfg = build_config(vocab_size=args.vocab_size, block_size=args.block_size)
    model = build_model(cfg).to(device)
    sd = torch.load(args.init, map_location="cpu")["model"]
    model.load_state_dict(sd)
    model.lm_head.weight = model.transformer.wte.weight
    raw_model = model
    if args.compile:
        model = torch.compile(model)
    print(f"[sft] loaded base weights from {args.init}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
        **({"fused": True} if device == "cuda" else {}))

    steps_per_epoch = math.ceil(len(examples) / (args.micro_batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup = max(10, int(total_steps * args.warmup_frac))
    print(f"[sft] {total_steps} optimizer steps ({steps_per_epoch}/epoch), warmup {warmup}")

    def lr_at(s):
        if s < warmup:
            return args.lr * (s + 1) / warmup
        r = (s - warmup) / max(1, total_steps - warmup)
        return args.min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (args.lr - args.min_lr)

    model.train()
    gstep = 0
    for epoch in range(args.epochs):
        random.shuffle(examples)
        i = 0
        while i < len(examples):
            for g in opt.param_groups:
                g["lr"] = lr_at(gstep)
            opt.zero_grad(set_to_none=True)
            lacc = 0.0
            for _ in range(args.grad_accum):
                batch = examples[i:i + args.micro_batch_size]
                i += args.micro_batch_size
                if not batch:
                    break
                X, Y = collate(batch)
                X, Y = X.to(device), Y.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(X)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                           Y.view(-1), ignore_index=IGNORE)
                    loss = loss / args.grad_accum
                loss.backward()
                lacc += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if gstep % 20 == 0:
                print(f"epoch {epoch} step {gstep}/{total_steps} loss {lacc:.4f} "
                      f"lr {lr_at(gstep):.2e}")
            gstep += 1

    out = os.path.join(args.out_dir, "final.pt")
    torch.save({"model": raw_model.state_dict(), "config": cfg.__dict__,
                "args": vars(args), "step": gstep}, out)
    print(f"[sft] saved {out}")


if __name__ == "__main__":
    main()
