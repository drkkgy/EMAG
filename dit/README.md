# EMAG on DiT (class-conditional, ImageNet)

Class-conditional EMAG sampling with [DiT-XL/2](https://github.com/facebookresearch/DiT).
This directory keeps the upstream DiT flat-import layout (`from models import ...`,
`from custom_code.custom_attention import ...`), so **run the scripts from inside `dit/`**
(or add `dit/` to `PYTHONPATH`).

## What EMAG changes here

- `custom_code/custom_attention.py` — a unified DiT self-attention module that exposes the
  attention map and supports EMAG (`enable_emag`, `attn_ema`, `do_ema`, `update_ema_to_attn`,
  `attn_gradient`) alongside the baselines PAG / SAG / SEG / ERG.
- `models.py` — `DiT.forward_with_emag` (and `_apg` / `_cads` composition variants), plus the
  statistics-based adaptive layer selection.
- `diffusion/gaussian_diffusion.py` — `p_sample_loop_emag`, the EMAG-aware sampling loop.

## Sample (DDP, multi-GPU)

```bash
cd dit
torchrun --nnodes=1 --nproc_per_node=1 sample_emag_ddp.py \
    --model DiT-XL/2 --image-size 256 \
    --num-fid-samples 50000 --per-proc-batch-size 32 \
    --use_emag --emag_mode cond \
    --emag_scale 3 --emag_beta 0.988 \
    --emag_start_step 150 --emag_time_delta 50 \
    --emag_layers 12 13 14 15 --emag_adaptive_mode 2 \
    --global-seed 8 --sample-dir samples
```

The DiT-XL/2 checkpoint auto-downloads (256/512). Output is a folder of PNGs plus an `.npz`
for FID via the [ADM evaluator](https://github.com/openai/guided-diffusion/tree/main/evaluations).

## Key EMAG arguments (DiT defaults)

| Argument             | Default            | Meaning                                                        |
|----------------------|--------------------|----------------------------------------------------------------|
| `--emag_scale`       | `3`                | EMAG guidance weight.                                          |
| `--emag_beta`        | `0.988`            | EMA decay for the attention accumulator.                      |
| `--emag_start_step`  | `150`              | Step (of 1000) after which EMA accumulation begins.           |
| `--emag_time_delta`  | `50`               | Delay before the EMA map is applied.                          |
| `--emag_layers`      | `12 13 14 15`      | Candidate transformer blocks.                                 |
| `--emag_adaptive_mode` | `2` (recommended)  | Attention-gradient layer selection: picks the layer with the largest Δ = MAE(EMA, attn) each step (paper method). `0` = fixed layers. |
| `--emag_mode`        | `cond`             | `cond` or `uncond` generation.                               |

`sample_ddp.py` is the plain CFG baseline for reference.

## EMAG combos (DiT)

EMAG composes with CFG / APG / CADS on DiT via dedicated samplers (the class-conditional analogue of
the SD3 `combo_mode`). These are the exact samplers used for the paper's full (50K-sample) DiT runs.
Each takes the same `--emag_*` arguments as `sample_emag_ddp.py` plus its combo-specific scale:

| Sampler | Combo | `models.py` method |
|---|---|---|
| `sample_emag_ddp.py`       | EMAG only (Eq. 14+15); `--emag_mode cond`/`uncond`; 256 & 512 via `--image-size`; supports `--start-index` resume | `forward_with_emag` |
| `sample_emag_cfg_ddp.py`   | EMAG + CFG            | `forward_with_emag_cfg` |
| `sample_emag_apg_ddp.py`   | EMAG + APG            | `forward_with_emag_apg` |
| `sample_emag_cads_ddp.py`  | EMAG + CADS           | `forward_with_emag_cads` |

```bash
cd dit && torchrun --nproc_per_node=1 sample_emag_apg_ddp.py \
    --model DiT-XL/2 --image-size 256 --num-fid-samples 50000 \
    --use_emag --emag_scale 3 --emag_layers 12 13 14 15 --emag_adaptive_mode 2 --global-seed 8
```

> The plain-baseline samplers (PAG/SAG/SEG/ERG/S²) and the full evaluation harness are a later
> release pass — see the roadmap in the top-level README.
