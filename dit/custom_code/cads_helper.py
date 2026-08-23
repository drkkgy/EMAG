"""
Author : Ankit Yadav
Date: 2025-10-17

This file contains helper functions for the CADS sampling strategy. adapted from the paper :- https://arxiv.org/pdf/2310.17347
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import atexit
import threading

def Timestetp_convertor(t:torch.Tensor, num_sampling_steps:int) -> torch.Tensor:
    timestep =  torch.round((1-t)*(num_sampling_steps-1))
    timestep = timestep.clamp(0, num_sampling_steps-1).to(t.device)
    return timestep


def linear_schedule(t, tau1, tau2):
    if t <= tau1:
        return torch.tensor(1.0)
    if t >= tau2:
        return torch.tensor(0.0)
    gamma = (tau2 - t)/(tau2 - tau1)
    return torch.tensor(gamma)

def add_noise(y, gamma, noise_scale, psi, rescale=False):
    
    try:
        y_mean, y_std = torch.mean(y), torch.std(y)
    except Exception as e:
        print(f"Pass Embeddings not index value for condtion:: {e}")
        return y
    y = torch.sqrt(gamma) * y + noise_scale * torch.sqrt(1 - gamma) * torch.randn_like(y)
    if rescale:
        y_scaled = (y - torch.mean(y)) / torch.std(y) * y_std + y_mean
        y = psi * y_scaled + (1 - psi) * y
        return y