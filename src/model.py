"""Model definition for the ~250M SLM (LitGPT Config A).

Decoder-only, Llama-style: RoPE, RMSNorm, SwiGLU (LLaMAMLP), GQA, tied embeddings.

Config A (tuned so GQA-reduced attention still lands near ~250M):
    d_model=1024, n_layer=20, n_head=16, n_kv=4, intermediate=2816, ctx=1024
=> ~258M params with a 32k (padded 32256) vocab and tied embeddings.

Run `python -m src.model` to print the exact parameter count.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from litgpt import Config, GPT


def build_config(vocab_size: int = 32000, block_size: int = 1024) -> Config:
    return Config(
        name="slm-250m",
        block_size=block_size,
        vocab_size=vocab_size,
        padding_multiple=512,          # -> padded_vocab_size = 32256 for 32000
        n_layer=20,
        n_head=16,
        n_embd=1024,
        n_query_groups=4,              # GQA: 4 KV heads
        head_size=64,                  # 1024 / 16
        rotary_percentage=1.0,         # full RoPE
        rope_base=10000,
        parallel_residual=False,       # Llama-style sequential residual
        shared_attention_norm=False,
        bias=False,
        lm_head_bias=False,
        norm_class_name="RMSNorm",
        norm_eps=1e-5,
        mlp_class_name="LLaMAMLP",     # SwiGLU
        intermediate_size=2816,
    )


def build_model(config: Config, tie_embeddings: bool = True) -> GPT:
    model = GPT(config)
    model.max_seq_length = config.block_size

    # Stable from-scratch init. litgpt's default fresh init leaves the residual
    # output projections at std=0.02. For Llama-style blocks (parallel_residual
    # = False) litgpt does NOT apply the GPT-NeoX 1/sqrt(2*n_layer) scaling, so
    # the wide SwiGLU down-projection (fan-in 2816) amplifies the residual stream
    # ~1.5x per layer and it explodes (logits -> ~1000, loss -> ~880). Scale only
    # the output projections, matching the GPT-NeoX convention. Embeddings/norms
    # keep litgpt's defaults.
    out_std = 0.02 / math.sqrt(2 * config.n_layer)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(".proj.weight"):            # attn.proj, mlp.proj (output)
                nn.init.normal_(p, mean=0.0, std=out_std)
            elif "wte" in name and name.endswith(".weight"):  # shared embedding
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif name.endswith(".bias"):
                nn.init.zeros_(p)
            elif name.endswith(".weight") and ("norm" in name or "ln" in name):
                pass  # RMSNorm weight: keep litgpt default (=1)
            elif name.endswith(".weight"):               # input projections (qkv, fc_1/2)
                nn.init.normal_(p, mean=0.0, std=0.02)

    if tie_embeddings:
        # Share input embedding and output projection weights.
        model.lm_head.weight = model.transformer.wte.weight
    return model


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total, unique) parameter counts (unique de-duplicates tied weights)."""
    total = sum(p.numel() for p in model.parameters())
    seen, unique = set(), 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        unique += p.numel()
    return total, unique


if __name__ == "__main__":
    cfg = build_config()
    model = build_model(cfg)
    total, unique = count_params(model)
    print(f"config: n_layer={cfg.n_layer} n_embd={cfg.n_embd} n_head={cfg.n_head} "
          f"n_kv={cfg.n_query_groups} intermediate={cfg.intermediate_size} "
          f"block_size={cfg.block_size} padded_vocab={cfg.padded_vocab_size}")
    print(f"parameters: total(incl. tied dup)={total/1e6:.1f}M  unique={unique/1e6:.1f}M")
