"""Phase 1 - stream FineWeb-Edu-dedup (real) + Cosmopedia v2 (synthetic) 50/50,
tokenize with the Llama 32k tokenizer, and pack into flat uint16 token bins.

Output:
    <out_dir>/train.bin   flat uint16 token stream (documents separated by EOS)
    <out_dir>/val.bin     small held-out stream (also 50/50)
    <out_dir>/meta.json   token counts, tokenizer, vocab, per-source stats

Usage:
    python -m src.prepare_data --out_dir data --total_tokens 2.5e9 --val_tokens 5e6

Notes:
- Uses HF streaming (streaming=True) so nothing is fully downloaded.
- Do this BEFORE the long training run (don't stream live during training).
- uint16 is valid because vocab (<=65535) fits; Llama vocab = 32000.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# (hf_dataset_config_name, text_column)  -- both live in HuggingFaceTB/smollm-corpus
SOURCES = [
    ("fineweb-edu-dedup", "text"),  # real educational web
    ("cosmopedia-v2", "text"),      # synthetic textbooks/stories
]
CORPUS = "HuggingFaceTB/smollm-corpus"


def _write_stream(cfg_name, text_col, tokenizer, eos_id,
                  val_target, train_target, val_fh, train_fh, flush_every=2_000_000):
    """Stream one source: first `val_target` tokens -> val, remainder -> train."""
    ds = load_dataset(CORPUS, name=cfg_name, split="train", streaming=True)
    buf: list[int] = []
    val_written = 0
    train_written = 0
    pbar = tqdm(total=train_target + val_target, unit="tok", unit_scale=True,
                desc=f"{cfg_name}")

    def flush():
        nonlocal buf, val_written, train_written
        if not buf:
            return
        arr = np.array(buf, dtype=np.uint16)
        buf = []
        # fill val first, then train
        if val_written < val_target:
            take = min(len(arr), val_target - val_written)
            arr[:take].tofile(val_fh)
            val_written += take
            pbar.update(take)
            arr = arr[take:]
        if len(arr):
            room = train_target - train_written
            if room <= 0:
                return
            take = min(len(arr), room)
            arr[:take].tofile(train_fh)
            train_written += take
            pbar.update(take)

    for row in ds:
        text = row.get(text_col) or ""
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(eos_id)
        buf.extend(ids)
        if len(buf) >= flush_every:
            flush()
            if train_written >= train_target and val_written >= val_target:
                break
    flush()
    pbar.close()
    return val_written, train_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf",
                    help="ungated Llama 32k tokenizer; swap for any 32k Llama/Mistral tok")
    ap.add_argument("--total_tokens", type=float, default=2.5e9)
    ap.add_argument("--val_tokens", type=float, default=5e6)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else 2
    vocab = tok.vocab_size
    assert vocab <= 65535, f"vocab {vocab} too large for uint16"

    n_src = len(SOURCES)
    val_per = int(args.val_tokens // n_src)
    train_per = int((args.total_tokens - args.val_tokens) // n_src)
    print(f"tokenizer={args.tokenizer} vocab={vocab} eos={eos_id}")
    print(f"per-source targets: train={train_per/1e6:.0f}M val={val_per/1e6:.1f}M "
          f"x {n_src} sources")

    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")
    stats = {}
    t0 = time.time()
    with open(train_path, "wb") as tf, open(val_path, "wb") as vf:
        for cfg_name, col in SOURCES:
            v, t = _write_stream(cfg_name, col, tok, eos_id, val_per, train_per, vf, tf)
            stats[cfg_name] = {"train_tokens": t, "val_tokens": v}
            print(f"  {cfg_name}: train={t/1e6:.1f}M val={v/1e6:.2f}M")

    total_train = sum(s["train_tokens"] for s in stats.values())
    total_val = sum(s["val_tokens"] for s in stats.values())
    meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": vocab,
        "eos_id": eos_id,
        "dtype": "uint16",
        "block_size": 1024,
        "train_tokens": total_train,
        "val_tokens": total_val,
        "sources": stats,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"DONE: train={total_train/1e6:.1f}M val={total_val/1e6:.2f}M "
          f"in {meta['elapsed_sec']}s -> {args.out_dir}/")


if __name__ == "__main__":
    main()
