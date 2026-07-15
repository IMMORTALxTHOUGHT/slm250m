"""Phase 3 - resumable BF16-mixed pretraining loop for the ~250M SLM.

Uses the LitGPT GPT model (RoPE/RMSNorm/SwiGLU/GQA) with a transparent, checkpoint/
resume-friendly training loop (safer for a 4-day SSH run than a black-box CLI).

Smoke test (Phase 2):
    python -m src.train --data_dir data --out_dir out/smoke --max_steps 100 \
        --log_interval 5 --eval_interval 100000

Full run (Phase 3), after sizing max_steps with src/compute_budget.py:
    python -m src.train --data_dir data --out_dir out/run1 --max_steps 9000 \
        --warmup_steps 150

Resume is automatic: re-running the same command loads out/<run>/last.pt.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .model import build_config, build_model, count_params


# --------------------------------------------------------------------------- #
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="out/run1")
    # batch / model
    ap.add_argument("--micro_batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--vocab_size", type=int, default=32000)
    # optim / schedule
    ap.add_argument("--max_steps", type=int, required=True, help="optimizer steps")
    ap.add_argument("--warmup_steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min_lr", type=float, default=6e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # eval / logging / ckpt
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--eval_iters", type=int, default=100)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--ckpt_interval_min", type=float, default=45.0)
    ap.add_argument("--keep_last", type=int, default=3)
    # misc
    ap.add_argument("--compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--init_from", default="", help="optional base ckpt to init weights")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
def make_get_batch(bin_path, block_size, micro_bs, device):
    def get_batch():
        # reopen memmap each call (avoids memory leak with a persistent handle)
        data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        ix = torch.randint(len(data) - block_size - 1, (micro_bs,))
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        return (x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True))
    return get_batch


def lr_at(step, args):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(1, args.warmup_steps)
    if step >= args.max_steps:
        return args.min_lr
    ratio = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)


def configure_optimizer(model, args):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = "fused" if torch.cuda.is_available() else None
    kw = {"fused": True} if fused else {}
    return torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2),
                             eps=1e-8, **kw)


@torch.no_grad()
def estimate_loss(model, get_val, iters, device):
    model.eval()
    losses = torch.zeros(iters)
    for k in range(iters):
        x, y = get_val()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def save_ckpt(path, raw_model, optimizer, step, cfg, args, best_val):
    tmp = path + ".tmp"
    torch.save({
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
        "config": cfg.__dict__ if hasattr(cfg, "__dict__") else {},
        "args": vars(args),
    }, tmp)
    os.replace(tmp, path)


def rotate_ckpts(out_dir, keep_last):
    ckpts = sorted(glob.glob(os.path.join(out_dir, "ckpt_step*.pt")),
                   key=os.path.getmtime)
    for old in ckpts[:-keep_last] if keep_last > 0 else []:
        try:
            os.remove(old)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
def main():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    meta_path = os.path.join(args.data_dir, "meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        args.vocab_size = meta.get("vocab_size", args.vocab_size)
        print(f"[data] {meta['train_tokens']/1e6:.0f}M train / "
              f"{meta['val_tokens']/1e6:.1f}M val tokens, vocab={args.vocab_size}")

    cfg = build_config(vocab_size=args.vocab_size, block_size=args.block_size)
    model = build_model(cfg).to(device)
    total, unique = count_params(model)
    print(f"[model] {unique/1e6:.1f}M unique params")

    optimizer = configure_optimizer(model, args)

    # optional weight init from a base checkpoint (used by continued training)
    if args.init_from and os.path.exists(args.init_from):
        sd = torch.load(args.init_from, map_location="cpu")["model"]
        model.load_state_dict(sd, strict=False)
        model.lm_head.weight = model.transformer.wte.weight
        print(f"[init] loaded weights from {args.init_from}")

    raw_model = model
    if args.compile:
        model = torch.compile(model)

    # resume ---------------------------------------------------------------
    start_step = 0
    best_val = float("inf")
    last_path = os.path.join(args.out_dir, "last.pt")
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        raw_model.load_state_dict(ck["model"])
        raw_model.lm_head.weight = raw_model.transformer.wte.weight
        optimizer.load_state_dict(ck["optimizer"])
        start_step = ck["step"]
        best_val = ck.get("best_val", best_val)
        print(f"[resume] from step {start_step}")

    get_train = make_get_batch(os.path.join(args.data_dir, "train.bin"),
                               args.block_size, args.micro_batch_size, device)
    get_val = make_get_batch(os.path.join(args.data_dir, "val.bin"),
                             args.block_size, args.micro_batch_size, device)

    writer = SummaryWriter(os.path.join(args.out_dir, "tb"))
    tokens_per_step = args.micro_batch_size * args.grad_accum * args.block_size
    print(f"[train] {tokens_per_step/1e3:.0f}k tokens/step, "
          f"max_steps={args.max_steps} -> {tokens_per_step*args.max_steps/1e9:.2f}B tokens")

    model.train()
    last_ckpt_t = time.time()
    t0 = time.time()
    for step in range(start_step, args.max_steps):
        lr = lr_at(step, args)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for micro in range(args.grad_accum):
            x, y = get_train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
            loss = loss / args.grad_accum
            loss.backward()
            loss_acc += loss.item()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # logging ----------------------------------------------------------
        if step % args.log_interval == 0:
            torch.cuda.synchronize() if device == "cuda" else None
            dt = time.time() - t0
            tok_s = tokens_per_step * args.log_interval / dt if step > start_step else 0
            t0 = time.time()
            mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
            print(f"step {step:>6} | loss {loss_acc:.4f} | lr {lr:.2e} | "
                  f"{tok_s/1e3:.1f}k tok/s | {mem:.1f} GB")
            writer.add_scalar("train/loss", loss_acc, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("perf/tok_per_s", tok_s, step)
            writer.add_scalar("perf/vram_gb", mem, step)

        # eval -------------------------------------------------------------
        if step > 0 and step % args.eval_interval == 0:
            vl = estimate_loss(model, get_val, args.eval_iters, device)
            writer.add_scalar("val/loss", vl, step)
            print(f"  [eval] step {step} val_loss {vl:.4f}")
            if vl < best_val:
                best_val = vl
                save_ckpt(os.path.join(args.out_dir, "best.pt"),
                          raw_model, optimizer, step, cfg, args, best_val)

        # time-based checkpoint -------------------------------------------
        if (time.time() - last_ckpt_t) / 60.0 >= args.ckpt_interval_min:
            save_ckpt(last_path, raw_model, optimizer, step, cfg, args, best_val)
            save_ckpt(os.path.join(args.out_dir, f"ckpt_step{step}.pt"),
                      raw_model, optimizer, step, cfg, args, best_val)
            rotate_ckpts(args.out_dir, args.keep_last)
            last_ckpt_t = time.time()
            print(f"  [ckpt] saved at step {step}")

    # final ----------------------------------------------------------------
    save_ckpt(last_path, raw_model, optimizer, args.max_steps, cfg, args, best_val)
    save_ckpt(os.path.join(args.out_dir, "final.pt"),
              raw_model, optimizer, args.max_steps, cfg, args, best_val)
    writer.close()
    print(f"[done] final step {args.max_steps}, best_val {best_val:.4f}")


if __name__ == "__main__":
    main()
