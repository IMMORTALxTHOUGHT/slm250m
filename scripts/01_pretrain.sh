#!/usr/bin/env bash
# Phase 3 launcher - run pretraining detached in tmux, logged to file.
# Edit MAX_STEPS/WARMUP from `python -m src.compute_budget ...` output first.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

RUN="${RUN:-run1}"
MAX_STEPS="${MAX_STEPS:?set MAX_STEPS (see src.compute_budget)}"
WARMUP="${WARMUP:-150}"
MICRO="${MICRO:-4}"      # 16 OOM'd on A5000 24GB (N^2 attention); 4 fits (~10GB peak)
ACCUM="${ACCUM:-128}"    # keeps 524k tok/step (4*128*1024) -> identical training dynamics

mkdir -p "out/$RUN"
LOG="out/$RUN/train_$(date +%Y%m%d_%H%M%S).log"

CMD="python -m src.train --data_dir data --out_dir out/$RUN \
  --micro_batch_size $MICRO --grad_accum $ACCUM \
  --max_steps $MAX_STEPS --warmup_steps $WARMUP"

echo "[launch] $CMD"
echo "[launch] logging to $LOG"
tmux new-session -d -s "slm_$RUN" "stdbuf -oL $CMD 2>&1 | tee $LOG"
echo "[launch] attached session: tmux attach -t slm_$RUN"
echo "[launch] tensorboard:       tensorboard --logdir out/$RUN/tb --port 6006"
