"""Export a native checkpoint to HuggingFace LlamaForCausalLM format.

Maps LitGPT GPT weights -> HF Llama weights (incl. GQA qkv un-interleave) and
runs a CPU parity check (max abs logit diff should be < ~1e-3). If parity fails,
prefer evaluating with the native model (src/generate.py) instead.

    python -m src.export_hf --ckpt out/sft/final.pt --out hf_model \
        --tokenizer NousResearch/Llama-2-7b-hf
"""
from __future__ import annotations

import argparse

import torch

from .model import build_config, build_model


def convert(raw_sd, cfg):
    from transformers import LlamaConfig, LlamaForCausalLM

    n_head, n_kv = cfg.n_head, cfg.n_query_groups
    hs = cfg.head_size
    q_per_kv = n_head // n_kv
    V = cfg.padded_vocab_size

    hf_cfg = LlamaConfig(
        vocab_size=V,
        hidden_size=cfg.n_embd,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.n_layer,
        num_attention_heads=n_head,
        num_key_value_heads=n_kv,
        max_position_embeddings=cfg.block_size,
        rms_norm_eps=cfg.norm_eps,
        rope_theta=cfg.rope_base,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
    )
    hf = LlamaForCausalLM(hf_cfg)
    sd = hf.state_dict()

    def set(name, tensor):
        # lm_head may be absent from a tied HF state_dict; skip (tie handles it).
        if name not in sd:
            return
        assert sd[name].shape == tensor.shape, f"{name}: {sd[name].shape} vs {tensor.shape}"
        sd[name] = tensor

    # LitGPT renamed the fused qkv linear (attn.attn -> attn.qkv) around 0.4.
    qkv_key = "attn.qkv.weight" if "transformer.h.0.attn.qkv.weight" in raw_sd else "attn.attn.weight"

    set("model.embed_tokens.weight", raw_sd["transformer.wte.weight"])
    set("model.norm.weight", raw_sd["transformer.ln_f.weight"])
    set("lm_head.weight", raw_sd["transformer.wte.weight"])  # tied

    for i in range(cfg.n_layer):
        p = f"transformer.h.{i}"
        h = f"model.layers.{i}"
        set(f"{h}.input_layernorm.weight", raw_sd[f"{p}.norm_1.weight"])
        set(f"{h}.post_attention_layernorm.weight", raw_sd[f"{p}.norm_2.weight"])

        # qkv un-interleave: litgpt rows grouped [q_per_kv q, 1 k, 1 v] per group
        W = raw_sd[f"{p}.{qkv_key}"]                   # ((n_head+2*n_kv)*hs, n_embd)
        W = W.view(n_kv, q_per_kv + 2, hs, cfg.n_embd)
        q = W[:, :q_per_kv].reshape(n_head * hs, cfg.n_embd)
        k = W[:, q_per_kv:q_per_kv + 1].reshape(n_kv * hs, cfg.n_embd)
        v = W[:, q_per_kv + 1:q_per_kv + 2].reshape(n_kv * hs, cfg.n_embd)
        set(f"{h}.self_attn.q_proj.weight", q)
        set(f"{h}.self_attn.k_proj.weight", k)
        set(f"{h}.self_attn.v_proj.weight", v)
        set(f"{h}.self_attn.o_proj.weight", raw_sd[f"{p}.attn.proj.weight"])

        # SwiGLU: litgpt fc_1->gate, fc_2->up, proj->down
        set(f"{h}.mlp.gate_proj.weight", raw_sd[f"{p}.mlp.fc_1.weight"])
        set(f"{h}.mlp.up_proj.weight", raw_sd[f"{p}.mlp.fc_2.weight"])
        set(f"{h}.mlp.down_proj.weight", raw_sd[f"{p}.mlp.proj.weight"])

    hf.load_state_dict(sd, strict=False)
    hf.tie_weights()
    return hf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="hf_model")
    ap.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--block_size", type=int, default=1024)
    args = ap.parse_args()

    cfg = build_config(vocab_size=args.vocab_size, block_size=args.block_size)
    ck = torch.load(args.ckpt, map_location="cpu")
    raw_sd = ck["model"]

    hf = convert(raw_sd, cfg)

    # --- parity check on CPU ---
    native = build_model(cfg)
    native.load_state_dict(raw_sd)
    native.lm_head.weight = native.transformer.wte.weight
    native.eval(); hf.eval()
    x = torch.randint(0, args.vocab_size, (1, 16))
    with torch.no_grad():
        a = native(x)
        b = hf(x).logits[..., :a.size(-1)]
    diff = (a - b).abs().max().item()
    print(f"[parity] max abs logit diff = {diff:.2e} "
          f"({'OK' if diff < 1e-3 else 'MISMATCH - prefer native eval'})")

    hf.save_pretrained(args.out)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(args.tokenizer).save_pretrained(args.out)
    print(f"[export] saved HF model + tokenizer to {args.out}/")


if __name__ == "__main__":
    main()
