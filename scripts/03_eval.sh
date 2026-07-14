#!/usr/bin/env bash
# Phase 4 - evaluate the exported HF model on standard SLM benchmarks.
#   bash scripts/03_eval.sh hf_model
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

MODEL="${1:-hf_model}"
pip install -q "lm-eval>=0.4.2"

lm_eval --model hf \
  --model_args "pretrained=$MODEL,dtype=bfloat16" \
  --tasks hellaswag,arc_easy,arc_challenge,piqa,openbookqa \
  --device cuda:0 \
  --batch_size auto \
  --output_path "out/eval_$(basename "$MODEL")"
