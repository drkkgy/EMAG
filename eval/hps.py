"""
HPS v2 scoring for a generated run, self-contained (SD3 text-to-image).

Reads an `index.json` (list of {"prompt", "image_path"}) produced by scripts/generate_coco.py and
reports mean / median HPS v2. Adapted from CVPR2026_Results/hps_calc.py (batched + cached).

Requires the `hpsv2` package (checkpoints auto-download from Hugging Face):
    pip install hpsv2

Example:
    python -m eval.hps --run outputs/emag_q            # scores all entries in index.json
    python -m eval.hps --run outputs/emag_q --sample-size 40000 --seed 8
"""
import argparse
import json
import random
import statistics
from pathlib import Path
from typing import List

import torch
from PIL import Image

import huggingface_hub
import hpsv2
from hpsv2.utils import hps_version_map
from hpsv2.img_score import initialize_model as _hps_initialize_model
from hpsv2.img_score import model_dict as _hps_model_dict
from hpsv2.src.open_clip import get_tokenizer as _hps_get_tokenizer


_CACHE = {"device": "cuda" if torch.cuda.is_available() else "cpu", "tokenizer": None, "version": None}


def _prepare(hps_version: str):
    if not _hps_model_dict:
        _hps_initialize_model()
    model = _hps_model_dict["model"]
    preprocess_val = _hps_model_dict["preprocess_val"]
    if _CACHE["tokenizer"] is None:
        _CACHE["tokenizer"] = _hps_get_tokenizer("ViT-H-14")
    if _CACHE["version"] != hps_version:
        cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])
        checkpoint = torch.load(cp, map_location=_CACHE["device"])
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(_CACHE["device"]).eval()
        _CACHE["version"] = hps_version
    return model, preprocess_val, _CACHE["tokenizer"], _CACHE["device"]


def score_pairs(image_paths: List[str], prompts: List[str], hps_version: str = "v2.1",
                batch_size: int = 64) -> List[float]:
    assert len(image_paths) == len(prompts)
    if not image_paths:
        return []
    model, preprocess_val, tokenizer, device = _prepare(hps_version)
    scores: List[float] = []
    for start in range(0, len(image_paths), batch_size):
        end = min(len(image_paths), start + batch_size)
        imgs = torch.cat([preprocess_val(Image.open(p)).unsqueeze(0) for p in image_paths[start:end]], dim=0)
        imgs = imgs.to(device=device, non_blocking=True)
        texts = tokenizer(prompts[start:end]).to(device=device, non_blocking=True)
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
            out = model(imgs, texts)
            logits = out["image_features"] @ out["text_features"].T
            scores.extend(float(x) for x in torch.diagonal(logits).cpu().numpy().tolist())
    return scores


def evaluate_run(run_dir: str, sample_size: int = 0, seed: int = 8, hps_version: str = "v2.1"):
    index_file = Path(run_dir) / "index.json"
    entries = json.loads(index_file.read_text(encoding="utf-8"))
    entries = [e for e in entries if e.get("prompt") and e.get("image_path")]
    if sample_size and sample_size < len(entries):
        entries = [entries[i] for i in random.Random(seed).sample(range(len(entries)), sample_size)]
    prompts = [e["prompt"] for e in entries]
    images = [e["image_path"] for e in entries]
    scores = score_pairs(images, prompts, hps_version=hps_version)
    # HPS is reported x100 in the paper
    mean = 100.0 * sum(scores) / len(scores)
    median = 100.0 * statistics.median(scores)
    return mean, median, len(scores)


def parse_args():
    p = argparse.ArgumentParser(description="Compute HPS v2 for a generated run")
    p.add_argument("--run", type=str, required=True, help="run dir containing index.json")
    p.add_argument("--sample-size", type=int, default=0, help="0 = score all entries")
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--hps-version", type=str, default="v2.1")
    return p.parse_args()


def main():
    args = parse_args()
    mean, median, n = evaluate_run(args.run, args.sample_size, args.seed, args.hps_version)
    print(f"[HPS v2] {args.run} | n={n} | mean={mean:.4f} | median={median:.4f}")


if __name__ == "__main__":
    main()
