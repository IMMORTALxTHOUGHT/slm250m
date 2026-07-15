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
# A5000 = Ampere (sm_86). The runtime CUDA MUST be <= the driver's CUDA.
# Check the driver version with `nvidia-smi` (top-right "CUDA Version").
#   driver 12.4 -> use cu124 (or cu121); driver 12.1 -> cu121; driver 11.8 -> cu118
# We pin an explicit torch build so pip never silently grabs a too-new wheel.
TORCH_CUDA="${TORCH_CUDA:-cu124}"      # override with:  TORCH_CUDA=cu121 bash scripts/00_setup.sh
TORCH_VER="${TORCH_VER:-2.5.1}"
if ! python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "[setup] (re)installing torch ${TORCH_VER}+${TORCH_CUDA} (driver-compatible)..."
  pip install "torch==${TORCH_VER}+${TORCH_CUDA}" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
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
