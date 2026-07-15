# SLM-250M on a single A5000 (4-day SSH run)

From-scratch pretraining of a ~258M-param general-English SLM, then a light SFT.
Built to run on **one RTX A5000 (24 GB)** over SSH within a **4-day** window.
See `../.local/share/kilo/plans/*slm-250m*.md` for the full plan and rationale.

## What this is
- **Framework:** LitGPT model (RoPE / RMSNorm / SwiGLU / GQA) + a transparent,
  resumable PyTorch training loop (robust to SSH drops).
- **Model (Config A):** d_model 1024, 20 layers, 16 heads, 4 KV (GQA),
  intermediate 2816, ctx 1024, tied embeddings, Llama 32k tokenizer -> **~258M**.
- **Data:** SmolLM-Corpus — FineWeb-Edu-dedup (real) + Cosmopedia v2 (synthetic)
  streamed 50/50 to **~2.5B unique tokens**, packed into uint16 bins.
- **Recipe:** BF16-mixed, micro-batch 16 x grad-accum 32 (~0.5M tok/step), AdamW
  (0.9/0.95, wd 0.1), grad-clip 1.0, LR 6e-4 → cosine 6e-5, warmup ~1.5%.

> Data (not compute) is the limiter: a 250M model is Chinchilla-optimal near ~5B
> tokens. We train ~2.5–4B exposures (~1–1.5 epochs) to fit the 4-day budget.

## Design deviations from the plan (why)
- **uint16 memmap bins instead of `litdata`:** simpler, transparent, and more
  robust for random-window sampling + resume. Same "pack to 1024" outcome.
- **Custom loop using the LitGPT `GPT` model** instead of the `litgpt pretrain`
  CLI: full control over checkpoint/resume/schedule for an interruptible SSH run.

---

## Runbook (all commands run on the A5000, inside tmux)

### Day 1 — setup + data + smoke test
```bash
tmux new -s slm            # survive SSH drops
cd slm250m
bash scripts/00_setup.sh   # venv, torch(cuda), deps, GPU/disk check
source .venv/bin/activate

# sanity: exact parameter count (expect ~258M unique)
python -m src.model

# Phase 1: build ~2.5B-token bins (streams from HF; do NOT stream during training)
python -m src.prepare_data --out_dir data --total_tokens 2.5e9 --val_tokens 5e6

# Phase 2: smoke test (~100 steps) to measure tok/s + peak VRAM
python -m src.train --data_dir data --out_dir out/smoke \
    --max_steps 100 --log_interval 5 --eval_interval 100000
#   -> try raising --micro_batch_size to 24/32 if VRAM < ~22 GB and re-check tok/s.

# size the real run from measured throughput (reserve ~1 day for SFT+eval+export)
python -m src.compute_budget --tok_s <MEASURED> --hours 60 \
    --micro_batch 16 --grad_accum 32 --block 1024
```

### Day 1–3 — pretraining
```bash
MAX_STEPS=<from compute_budget> WARMUP=<from compute_budget> \
    bash scripts/01_pretrain.sh          # launches in its own tmux session
tensorboard --logdir out/run1/tb --port 6006   # monitor loss/lr/tok_s/val
```
Crashed or disconnected? Just re-run the same launch — it resumes from
`out/run1/last.pt`. Checkpoints rotate every ~45 min (keep last 3) + `best.pt`.

### Day 4 — SFT + eval + export
```bash
# light instruction tuning (1 epoch on smoltalk)
python -m src.sft --init out/run1/final.pt --out_dir out/sft --max_samples 50000

# qualitative checks
python -m src.generate --ckpt out/run1/final.pt --prompt "The moon is"
python -m src.generate --ckpt out/sft/final.pt --chat \
    --prompt "Explain photosynthesis in one sentence."

# export to HF format (with logit parity check) + benchmark
python -m src.export_hf --ckpt out/sft/final.pt --out hf_model
bash scripts/03_eval.sh hf_model   # HellaSwag, ARC-e/c, PIQA, OpenBookQA
```

## Files
| Path | Purpose |
|---|---|
| `scripts/00_setup.sh` | env + deps + GPU/disk check |
| `src/prepare_data.py` | stream+tokenize+pack 50/50 → `data/{train,val}.bin` |
| `src/model.py` | Config A model + param count |
| `src/train.py` | resumable BF16 pretraining loop |
| `src/compute_budget.py` | tok/s + hours → `max_steps`/`warmup` |
| `scripts/01_pretrain.sh` | tmux launcher |
| `src/sft.py` | light instruction tuning |
| `src/generate.py` | native completion / chat |
| `src/export_hf.py` | LitGPT → HF Llama export + parity check |
| `scripts/03_eval.sh` | lm-eval-harness benchmarks |

## Notes / gotchas
- **Tokenizer:** default `NousResearch/Llama-2-7b-hf` (ungated, 32k). Keep the
  SAME tokenizer for prepare_data, sft, generate, and export.
- **Disk:** ~5 GB for bins + ~2 GB/checkpoint. Verify `df -h` before the long run.
- **CUDA wheel:** `00_setup.sh` installs `torch 2.6.0+cu124` (the highest build the
  CUDA 12.4 driver can run). `litgpt` is pinned to `0.4.x` because `0.5.x` needs
  `torch>=2.7` (=cu128+, which this driver cannot run). If your driver differs,
  override with `TORCH_CUDA=cu121 TORCH_VER=2.5.1 bash scripts/00_setup.sh`.
- **flash-attn:** not required — the loop uses PyTorch SDPA (flash kernel). Install
  the standalone package only if you want it (see setup script).
- **Undertraining is expected** at 2.5B tokens; if smoke-test throughput is higher
  than planned, stream MORE unique tokens (raise `--total_tokens`) rather than
  adding epochs.
