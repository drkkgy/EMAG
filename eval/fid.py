"""
FID for a generated run against a real-image reference, self-contained.

Thin wrapper over `pytorch-fid`. The generated images live under `<run>/images`; the reference is a
directory of real COCO-2014 validation images (or another folder). For repeated evaluations, cache
the reference by pointing `--ref` at the same folder — pytorch-fid recomputes stats each call, so for
large sweeps you may prefer to precompute an `.npz` with `python -m pytorch_fid --save-stats`.

Requires:
    pip install pytorch-fid

Example:
    python -m eval.fid --run outputs/emag_q --ref /data/coco/val2014
"""
import argparse
from pathlib import Path

import torch
from pytorch_fid.fid_score import calculate_fid_given_paths


def evaluate_run(run_dir: str, ref_path: str, batch_size: int = 50, dims: int = 2048,
                 num_workers: int = 8) -> float:
    images_dir = Path(run_dir) / "images"
    gen_path = str(images_dir if images_dir.is_dir() else run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fid = calculate_fid_given_paths(
        [ref_path, gen_path], batch_size=batch_size, device=device, dims=dims, num_workers=num_workers
    )
    return float(fid)


def parse_args():
    p = argparse.ArgumentParser(description="Compute FID for a generated run")
    p.add_argument("--run", type=str, required=True, help="run dir (uses <run>/images if present)")
    p.add_argument("--ref", type=str, required=True, help="reference: real-image dir or precomputed .npz")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--dims", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    fid = evaluate_run(args.run, args.ref, args.batch_size, args.dims, args.num_workers)
    print(f"[FID] {args.run} vs {args.ref} | FID={fid:.4f}")


if __name__ == "__main__":
    main()
