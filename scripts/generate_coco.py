"""
Batch text-to-image generation over COCO-2014 validation captions, for reproducing Table 2.

Self-contained: uses only the `emag` SD3 pipeline (no EvalGIM). Writes generated PNGs plus an
`index.json` of {prompt, image_path} entries that `eval/hps.py` and `eval/fid.py` consume.

Paper protocol (defaults below): 40K captions, 28 steps, 1024x1024, seed 8, CFG 7.

Captions source (`--captions`): a COCO `captions_val2014.json` (official annotations), OR a JSON
list of strings / objects with a "caption" field, OR a plain-text file (one caption per line).
For COCO annotations we keep one caption per image (first seen) to get ~40.5K unique prompts.

Example:
    python scripts/generate_coco.py \
        --captions /data/coco/annotations/captions_val2014.json \
        --out_dir outputs/emag_q --method emag-q --combo_mode No \
        --num_samples 40000 --batch_size 8
"""
import argparse
import json
import os
from pathlib import Path

import torch

from emag import StableDiffusion3Pipeline


def load_captions(path: str, num_samples: int):
    """Return a list of at most `num_samples` prompt strings from various file formats."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    prompts = []
    if p.suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict) and "annotations" in data:  # COCO captions_val2014.json
            seen_images = set()
            for ann in data["annotations"]:
                img_id = ann.get("image_id")
                if img_id in seen_images:
                    continue  # one caption per image
                seen_images.add(img_id)
                prompts.append(ann["caption"].strip())
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    prompts.append(item.strip())
                elif isinstance(item, dict) and "caption" in item:
                    prompts.append(item["caption"].strip())
                elif isinstance(item, dict) and "prompt" in item:
                    prompts.append(item["prompt"].strip())
        else:
            raise ValueError(f"Unrecognised JSON caption format in {path}")
    else:  # plain text, one caption per line
        prompts = [line.strip() for line in text.splitlines() if line.strip()]

    if not prompts:
        raise ValueError(f"No captions parsed from {path}")
    return prompts[:num_samples]


def parse_args():
    p = argparse.ArgumentParser(description="Batch COCO generation for EMAG Table 2")
    p.add_argument("--captions", type=str, required=True, help="COCO captions_val2014.json / JSON list / .txt")
    p.add_argument("--out_dir", type=str, required=True, help="output directory (images/ + index.json)")
    p.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    p.add_argument("--num_samples", type=int, default=40000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--num_inference_steps", type=int, default=28)
    p.add_argument("--cfg_scale", type=float, default=7.0)
    # ---- EMAG (paper defaults) ----
    p.add_argument("--method", type=str, default="emag-q", choices=["emag-q", "emag"],
                   help="emag-q = query-EMA + fused SDPA (default); emag = attention-map EMA")
    p.add_argument("--combo_mode", type=str, default="No",
                   help="EMAG composition: No (EMAG+CFG), APG, CADS")
    p.add_argument("--emag_scale", type=float, default=1.5)
    p.add_argument("--emag_beta", type=float, default=0.988)
    p.add_argument("--emag_start_step", type=int, default=250)
    p.add_argument("--emag_time_delta", type=int, default=50)
    p.add_argument("--emag_layers", type=int, nargs="+", default=[6, 7, 8])
    p.add_argument("--emag_mode", type=str, default="cond", choices=["cond", "uncond"])
    p.add_argument("--emag_adaptive_mode", type=int, default=2)
    p.add_argument("--emag_renorm_mode", type=str, default="row", choices=["row", "img2txt_only", "none"])
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = load_captions(args.captions, args.num_samples)
    print(f"Loaded {len(prompts)} captions from {args.captions}")

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.float16).to(device)
    pipe.set_progress_bar_config(disable=True)

    use_q_ema = args.method == "emag-q"
    gen_kwargs = dict(
        guidance_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        use_q_ema=use_q_ema,
        emag_scale=args.emag_scale,
        emag_beta=args.emag_beta,
        emag_start_step=args.emag_start_step,
        emag_time_delta=args.emag_time_delta,
        emag_layers=args.emag_layers,
        emag_mode=args.emag_mode,
        emag_adaptive_mode=args.emag_adaptive_mode,
        emag_renorm_mode=args.emag_renorm_mode,
        combo_mode=args.combo_mode,
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    index = []
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start:start + args.batch_size]
        images = pipe(prompt=batch, generator=generator, **gen_kwargs).images
        for j, img in enumerate(images):
            idx = start + j
            img_path = images_dir / f"{idx:06d}.png"
            img.save(img_path)
            index.append({"prompt": batch[j], "image_path": str(img_path.resolve())})
        print(f"  {min(start + args.batch_size, len(prompts))}/{len(prompts)} generated", flush=True)

    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    # record the exact config next to the images for provenance
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Done: {len(index)} images -> {images_dir}\nindex.json + config.json written to {out_dir}")


if __name__ == "__main__":
    main()
