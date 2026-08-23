"""
Minimal EMAG demo on Stable Diffusion 3: generate a single image with EMAG guidance.

Usage:
    python scripts/quickstart.py --prompt "a photograph of an astronaut riding a horse"

Requires a GPU and access to the SD3-medium weights on Hugging Face
(`stabilityai/stable-diffusion-3-medium-diffusers`, gated — run `huggingface-cli login`).

The EMAG defaults below match the values used in the paper's experiments.
"""
import argparse

import torch

from emag import StableDiffusion3Pipeline


def parse_args():
    p = argparse.ArgumentParser(description="EMAG SD3 quickstart")
    p.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    p.add_argument("--prompt", type=str, default="a photograph of an astronaut riding a horse")
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--out", type=str, default="emag_sample.png")
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--num_inference_steps", type=int, default=28)
    p.add_argument("--method", type=str, default="emag-q", choices=["emag-q", "emag"],
                   help="emag-q = query-EMA + fused SDPA (default; ~half the overhead); "
                        "emag = full attention-map EMA")
    p.add_argument("--cfg_scale", type=float, default=7.0, help="classifier-free guidance scale")
    # ---- EMAG hyper-parameters (paper defaults) ----
    p.add_argument("--emag_scale", type=float, default=1.5,
                   help="EMAG guidance weight w_e (paper default for conditional SD3; 0 disables EMAG)")
    p.add_argument("--emag_beta", type=float, default=0.988, help="EMA decay for the attention accumulator")
    p.add_argument("--emag_start_step", type=int, default=250)
    p.add_argument("--emag_time_delta", type=int, default=50)
    p.add_argument("--emag_layers", type=int, nargs="+", default=[6, 7, 8],
                   help="candidate transformer blocks for EMAG (paper SD3 range: layers 6-8)")
    p.add_argument("--emag_mode", type=str, default="cond", choices=["cond", "uncond"])
    p.add_argument("--emag_adaptive_mode", type=int, default=2,
                   help="attention-gradient layer selection (paper default): 2 picks, each step, the "
                        "layer with the largest Delta = MAE(EMA, attention); 0 disables (fixed layers)")
    p.add_argument("--emag_renorm_mode", type=str, default="row", choices=["row", "img2txt_only", "none"])
    p.add_argument("--combo_mode", type=str, default="No",
                   help="How EMAG composes with the base guidance: No, CFG, APG, CADS, ...")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    use_q_ema = args.method == "emag-q"

    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.cfg_scale,
        generator=generator,
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
    ).images[0]

    image.save(args.out)
    print(f"Saved EMAG sample to {args.out}")


if __name__ == "__main__":
    main()
