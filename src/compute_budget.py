"""Phase 2 helper - turn measured throughput into max_steps / schedule length.

After the smoke test prints "XX.Xk tok/s", plug it in here with the hours you
have left for pretraining (reserve time for SFT + eval + export).

    python -m src.compute_budget --tok_s 22000 --hours 60 \
        --micro_batch 16 --grad_accum 32 --block 1024

Prints the recommended --max_steps and --warmup_steps, and the resulting token
budget so you can sanity-check it against your ~2.5B packed tokens.
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok_s", type=float, required=True, help="measured tokens/sec")
    ap.add_argument("--hours", type=float, required=True, help="pretraining hours available")
    ap.add_argument("--micro_batch", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--efficiency", type=float, default=0.85,
                    help="derate for eval/ckpt/restarts overhead")
    ap.add_argument("--unique_tokens", type=float, default=2.5e9)
    ap.add_argument("--warmup_frac", type=float, default=0.015)
    args = ap.parse_args()

    tokens_per_step = args.micro_batch * args.grad_accum * args.block
    usable_sec = args.hours * 3600 * args.efficiency
    total_tokens = args.tok_s * usable_sec
    max_steps = int(total_tokens // tokens_per_step)
    warmup = max(50, int(max_steps * args.warmup_frac))
    epochs = total_tokens / args.unique_tokens

    print(f"tokens/step      : {tokens_per_step/1e3:.0f}k")
    print(f"usable seconds   : {usable_sec/3600:.1f} h (eff {args.efficiency})")
    print(f"token budget     : {total_tokens/1e9:.2f}B exposures "
          f"(~{epochs:.2f} epochs over {args.unique_tokens/1e9:.1f}B unique)")
    print(f"--max_steps      : {max_steps}")
    print(f"--warmup_steps   : {warmup}")
    if epochs > 4:
        print("WARNING: >4 epochs over unique data - stream more unique tokens instead.")
    if epochs < 1:
        print("NOTE: <1 epoch - you won't consume all unique data; that's fine.")


if __name__ == "__main__":
    main()
