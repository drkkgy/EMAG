"""
EMAG: Self-Rectifying Diffusion Sampling with Exponential Moving Average Guidance.

Official implementation (ECCV 2026). Paper: https://arxiv.org/abs/2512.17303

Public API:
    - StableDiffusion3Pipeline: SD3 (MMDiT) pipeline with EMAG guidance built in.
    - EMAGAttnProcessor2_0: attention processor that maintains an EMA of the
      self-attention map and applies it as a self-rectifying perturbation.
    - emag_forward: one transformer forward wrapped with EMAG accumulate/apply
      toggling and statistics-based adaptive layer selection.
"""

from .emag_util import EMAGAttnProcessor2_0, emag_forward, k2idx

__all__ = [
    "StableDiffusion3Pipeline",
    "EMAGAttnProcessor2_0",
    "emag_forward",
    "k2idx",
]

__version__ = "0.1.0"


def __getattr__(name):
    # Lazy import: the SD3 pipeline pulls in the full diffusers SD3 stack (needs a
    # compatible diffusers, e.g. 0.31.x with SD3IPAdapterMixin). Importing it lazily keeps
    # the lightweight core (EMAGAttnProcessor2_0 / emag_forward) and the unit tests usable
    # without that heavy dependency being importable.
    if name == "StableDiffusion3Pipeline":
        from .sd3_pipeline import StableDiffusion3Pipeline
        return StableDiffusion3Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
