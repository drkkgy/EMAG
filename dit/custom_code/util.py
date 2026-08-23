"""
Author : Ankit Yadav
Date: 2025-10-10

This file contains utility functions for the custom attention modules.
"""

import torch
import os
import atexit
import threading
from typing import List, Dict, Optional
import csv
import torch.nn.functional as F

# SEG Helper functions ##########################################################
# Gaussian blur
def gaussian_blur_2d(img, kernel_size, sigma):
    height = img.shape[-1]
    kernel_size = min(kernel_size, height - (height % 2 - 1))
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img
# EMAG Helper functions ##########################################################

def Attention_Map_entropy_calculation(attn_map: torch.Tensor) -> torch.Tensor:
    """
    Compute average attention entropy per head.
    attn_map: [H, Q, K], each row over K sums to 1 (softmaxed).
    Returns: [H] average entropy across queries for each head. and average entropy across all heads.
    """
    eps = 1e-12
    row_entropies = -(attn_map * (attn_map.clamp_min(eps)).log()).sum(dim=-1)  # [H, Q]
    avg_entropy_per_head = row_entropies.mean(dim=-2)   # [H]
    avg_entropy = avg_entropy_per_head.mean(dim=-1)
    return avg_entropy