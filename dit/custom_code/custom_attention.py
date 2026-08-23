"""
Author : Ankit Yadav
Date: 2025-09-28

This file contains custom attention modules for the DiT model.

Adapted from 
1. https://github.com/facebookresearch/DiT
2. https://github.com/openai/guided-diffusion
3. https://github.com/hojonathanho/diffusion
4. https://github.com/facebookresearch/mae

This will be used to replace the original attention module in the DiT model for our experiments. Based on TIMM implementation of attention.

"""
from typing import Final, Optional, Type

import torch
from torch import nn as nn
from torch.nn import functional as F
import os
import math

from ._fx import register_notrace_function
from .config import use_fused_attn
from .util import gaussian_blur_2d,Attention_Map_entropy_calculation

@torch.fx.wrap
@register_notrace_function
def maybe_add_mask(scores: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    return scores if attn_mask is None else scores + attn_mask



class Attention(nn.Module):
    """Standard Multi-head Self Attention module with QKV projection.

    This module implements the standard multi-head attention mechanism used in transformers.
    It supports both the fused attention implementation (scaled_dot_product_attention) for
    efficiency when available, and a manual implementation otherwise. The module includes
    options for QK normalization, attention dropout, and projection dropout.
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in the query, key, value projections
            qk_norm: Whether to apply normalization to query and key vectors
            proj_bias: Whether to use bias in the output projection
            attn_drop: Dropout rate applied to the attention weights
            proj_drop: Dropout rate applied after the output projection
            norm_layer: Normalization layer constructor for QK normalization if enabled
        """
        #print("Initializing Attention module with custom attention modules.......")
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if qk_norm or scale_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        #self.fused_attn = use_fused_attn() # Disabling this to access the attention matrix
        self.fused_attn = False
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # To support PAG 
        self.enable_pag = False

        # To support SAG
        self.capture_attn = False
        self.last_attn = None  # (B, H, L, L)

        # To Support SEG
        self.enable_seg = False
        self.blur_sigma = 1

        # To Support EMAG
        self.enable_emag = False
        self.beta = float(0.988)
        self.ema_strength = float(1)
        self.register_buffer("attn_ema", None,persistent=False) # We will not save the EMA values in the checkpoint

        # Entropy and attention gradient registers
        self.register_buffer("attn_entropy", None,persistent=False)
        self.register_buffer("attn_gradient", None,persistent=False)

        # EMA control flags
        self.do_ema = False
        self.update_ema_to_attn = False

        # To Support ERG
        self.enable_erg = False
        self.erg_alpha = 1.0
        self.erg_gamma = 1.0
        self.erg_tau_i = 0.01
        self.erg_K = 1
        
    # Supporting function for EMAG

    def update_attn_ema(self, attn: torch.Tensor, reduce_over_batch: bool = False):
        """
        attn: [B,H,Q,K] attention probs (already softmaxed).
        If reduce_over_batch=True, we first average over B -> [H,Q,K] before EMA.
        """
        if not (self.enable_emag and self.do_ema):
            return

        with torch.no_grad():  # bookkeeping only
            x = attn.detach()
            if reduce_over_batch and x.dim() == 4:
                x = x.mean(dim=0)  # [H,Q,K]
            # keep accumulator in fp32 for stability under AMP
            x = x.to(torch.float32)

            if self.attn_ema is None:
                self.attn_ema = x.clone()


                #self.ema_t.zero_()
            else:
                # In-place EMA: ema = beta*ema + (1-beta)*x
                self.attn_ema.mul_(self.beta).add_(x, alpha=1.0 - self.beta)
            #self.ema_t.add_(1)
            # Calculate the entropy of the attention map
            self.attn_entropy = Attention_Map_entropy_calculation(x)
            self.attn_gradient = torch.mean(torch.abs(self.attn_ema - x))  # L1 loss between EMA and current attention map

    def maybe_apply_ema(self, attn: torch.Tensor,strength: float = 1):
        """
        mode='soft' -> linear blend with current attn (using 1-beta as weight).
        mode='hard' -> hard replace by EMA.

        We will keep it hard as from previous literature inuclding SEG better performance comes when the preturbation is strong.
        """
        if not (self.enable_emag and self.update_ema_to_attn and (self.attn_ema is not None)):
            return attn
        #ema = self.attn_ema.to(attn.dtype)
        ema = strength * self.attn_ema.to(attn.dtype) + (1 - strength) * attn.to(attn.dtype)
        return ema

    # ERG Helper functions
    def multistep_attention(self, q, k, v, step_size=1, steps=1, gamma=1.0):
        q_new = q.clone()
        for i in range(steps):
            att = F.scaled_dot_product_attention(q_new, k, v)
            q_new = q_new - gamma * (q_new - step_size * att)
        return q_new

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        elif self.enable_erg:
            if self.erg_tau_i > 0:
                q = q * self.erg_tau_i
            # Performing multistep attention
            x = self.multistep_attention(q, k, v, step_size=self.erg_alpha, steps=self.erg_K, gamma=self.erg_gamma)
        else:
            #print("Using custom attention module XX.......")
            q = q * self.scale

            # To support SEG
            # Taken from the original implementation of SEG: https://github.com/SusungHong/SEG-SDXL/blob/master/pipeline_seg.py
            if self.enable_seg:
                attn_res = math.isqrt(q.shape[2])
                assert attn_res * attn_res == q.shape[2], "Token count must be a square."

                q_pret = q.permute(0, 1, 3, 2).view(B, self.num_heads * self.head_dim, attn_res, attn_res)

                if self.blur_sigma > 9999.0: # From the original implementation of SEG: https://github.com/SusungHong/SEG-SDXL/blob/master/pipeline_seg.py
                    q_pret[:] = q_pret.mean(dim=(-2,-1),keepdim=True)
                else:
                    kernel_size = math.ceil(6 * self.blur_sigma) + 1 - math.ceil(6 * self.blur_sigma) % 2
                    q_pret = gaussian_blur_2d(q_pret, kernel_size, self.blur_sigma)

                q = q_pret.view(B, self.num_heads, self.head_dim, attn_res * attn_res).permute(0, 1, 3, 2)
                
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            
            # To support EMAG
            if self.do_ema or self.update_ema_to_attn:
                assert self.enable_emag, "EMAG flags set but EMAG is disabled."
            
            if self.enable_emag:

                # if self.do_ema:
                #     # Calcuating the EMA of the attention matrix
                #     if self.attn_ema is None:
                #         self.attn_ema = attn.detach()
                #     else:
                #         self.attn_ema = self.beta * self.attn_ema + (1 - self.beta) * attn.detach()

                self.update_attn_ema(attn)

                # Replacing the attention matrix with the EMA for the assigned timestep and layers
                attn = self.maybe_apply_ema(attn,strength=self.ema_strength)
            
            
            # To support SAG
            if self.capture_attn:
                self.last_attn = attn.detach()

            attn = self.attn_drop(attn)
            if self.enable_pag:
                # PAG method 
                L = attn.size(-1)
                I = torch.eye(L,device=attn.device,dtype=attn.dtype).unsqueeze(0).unsqueeze(0)
                attn = I.expand(attn.size(0),attn.size(1), L, L)
                
            
            x = attn @ v


        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
