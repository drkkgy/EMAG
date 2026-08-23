"""
Author : Ankit Yadav
Date: 2025-10-16

This file contains helper functions for the APG sampling strategy. adapted from the paper :- https://arxiv.org/pdf/2410.02416
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import atexit
import threading
from typing import List, Dict, Optional
import csv


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0
    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average



def project(
        v0: torch.Tensor, # [B, C, H, W]
        v1: torch.Tensor, # [B, C, H, W]
        ):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)