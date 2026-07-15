#!/usr/bin/env bash
# Phase 0 - environment setup on the remote A5000.
# Run once inside a tmux session:  bash scripts/00_setup.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PROJ="$(pwd)"
echo "[setup] project root: $PROJ"

# --- 0. Sanity: GPU + disk ---------------------------------------------------
echo "[setup] GPU:"; nvidia-smi || { echo "!! nvidia-smi failed - no GPU?"; exit 1; }
echo "[setup] Disk (need ~15-20 GB free for bins + checkpoints):"; df -h .

# --- 1. Python venv ----------------------------------------------------------
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel setuptools

# --- 2. PyTorch (CUDA) -------------------------------------------------------
# A5000 = Ampere (sm_86). Runtime CUDA MUST be <= driver CUDA (nvidia-smi, top-right).
# This driver is 12.4. The cu124 wheel index tops out at torch 2.6.0, and torch>=2.7
# (required by litgpt 0.5.x) is only built for cu128+ which this driver CANNOT run.
# So we use the highest runnable torch: 2.6.0+cu124 (exact 12.4 match), paired with
# litgpt 0.4.x (requirements.txt pins litgpt<0.5). CUDA is backward-compatible.
# We pin an explicit torch build so pip never silently grabs a too-new wheel, and we
# PURGE any leftover nvidia-* wheels to avoid libcusparse/nvJitLink symbol clashes.
TORCH_CUDA="${TORCH_CUDA:-cu124}"
TORCH_VER="${TORCH_VER:-2.6.0}"
DESIRED="torch==${TORCH_VER}+${TORCH_CUDA}"
if ! python -c "import torch,sys; assert torch.__version__.startswith('${TORCH_VER}+${TORCH_CUDA}'), torch.__version__; assert torch.cuda.is_available(); sys.exit(0)" 2>/dev/null; then
  echo "[setup] clean (re)install of ${DESIRED} (driver-compatible)..."
  pip uninstall -y torch 2>/dev/null || true
  pip uninstall -y $(pip list 2>/dev/null | grep -iE '^nvidia-' | awk '{print $1}') 2>/dev/null || true
  pip install "${DESIRED}" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
fi

# --- 3. Project deps ---------------------------------------------------------
pip install -r requirements.txt

# --- 4. FlashAttention-2 (optional; PyTorch SDPA already gives flash) --------
# The training loop uses torch SDPA (flash kernel) by default, so flash-attn is
# NOT required. Uncomment to install the standalone package if you want it.
# pip install flash-attn --no-build-isolation || echo "[setup] flash-attn skipped"

# --- 5. HF cache on the roomy volume ----------------------------------------
export HF_HOME="${HF_HOME:-$PROJ/.hf}"
mkdir -p "$HF_HOME"
echo "[setup] HF_HOME=$HF_HOME"

# --- 6. Report ---------------------------------------------------------------
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu:", p.name, f"{p.total_memory/1e9:.1f} GB", "sm", f"{p.major}.{p.minor}")
PY

echo "[setup] done. Activate later with:  source $PROJ/.venv/bin/activate"
