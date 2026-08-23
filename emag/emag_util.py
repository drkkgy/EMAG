"""
This file contains the utility functions for the ERG pipeline. 
"""

## Helper Functions for ERG #########################################################
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Dict, Optional
import random

import math
from diffusers.models.attention_processor import Attention,JointAttnProcessor2_0
  
#####################################################################################


def k2idx(k, N, k_base=250):
    return max(0, min(N - 1, int(round(k * (N - 1) / float(k_base)))))


# EMAG attention processor to preturb the self attetnion map
class EMAGAttnProcessor2_0:
    """
    EMAG attention processor for SD3-like MMDiT self-attention.

    - Maintains EMA for image-to-image attention block only.
    - When enabled to update, replaces only the image-to-image block with its EMA-mixed version.
    - Optionally reweights the image-to-text block after replacement and renormalizes row-wise.
    - Exposes attn_gradient (MAE of live attn vs its EMA) on the image-to-image block for adaptive layer selection.
    - External controller (e.g., emag_forward) should toggle:
        - self.do_ema (bool)
        - self.update_ema_to_attn (bool)
    """

    def __init__(self, ema_beta=0.988, ema_strength=1.0, img2txt_scale=1.0, reduce_over_batch=False, renorm_mode="row",use_q_ema=False):

        # Required for PyTorch 2+ fused attn availability check
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("EMAGAttnProcessor2_0 requires PyTorch 2.0+ (for APIs compatibility).")
        # EMAG controls
        self.enable_emag = True
        self.do_ema = False
        self.update_ema_to_attn = False

        # Additional controls for EMAG-Qembed
        self.use_q_ema = use_q_ema
        self.q_ema = None
        

        # EMA hyper-params
        self.ema_beta = float(ema_beta)           # decay for EMA accumulator
        self.ema_strength = float(ema_strength)   # blend factor when applying EMA to img->img block
        self.img2txt_scale = float(img2txt_scale) # optional reweight for img->txt block (row-wise renorm after)
        self.reduce_over_batch = bool(reduce_over_batch)

        # State (buffers as plain attributes since processor isn't an nn.Module)
        self.attn_ema = None           # [H, Q_img, K_img] if reduce_over_batch else [B, H, Q_img, K_img]
        self.attn_gradient = None      # Tensor/Scalar for selection (e.g., median())
        self.attn_ema_full = False
        self.attn_ema_img2_txt_only = False

        # Attention Reweighting
        if renorm_mode not in ("row", "img2txt_only", "none"):
            raise ValueError(f"Invalid renorm_mode: {renorm_mode}. Choose from 'row', 'img2txt_only', 'none'.")
        self.renorm_mode = renorm_mode

    def _update_ema_and_stats(self, attn_img_img: torch.Tensor):
        """
        attn_img_img: [B, H, Q_img, K_img] softmax probabilities (detached for stats/EMA).
        We will resue this function to update the EMA for the image-query rows (img->img + img->txt) as well so no changes required here.
        so in that case it will be like this
        attn_img_rows: [B, H, Q_img, K_total] softmax probabilities (detached for stats/EMA).
        """
        x = attn_img_img.detach().to(torch.float16)
        if self.reduce_over_batch and x.dim() == 4:
            # [H, Q_img, K_img]
            x_reduced = x.mean(dim=0)
        else:
            x_reduced = x  # keep [B, H, Q_img, K_img]

        with torch.no_grad():
            if self.attn_ema is None:
                self.attn_ema = x_reduced.clone()
                self.attn_gradient = torch.as_tensor(0.0, device=x.device, dtype=torch.float16)
            else:
                # L1 diff mean as a "gradient/change" proxy
                self.attn_gradient = (self.attn_ema - x_reduced).abs().mean()
                # EMA update
                self.attn_ema.mul_(self.ema_beta).add_(x_reduced, alpha=1.0 - self.ema_beta)

    # Method to support the updating the the Q embedding EMA replacement

    def _update_q_ema(self, q_img: torch.Tensor):
        """q_img: [B, H, Q_img, D] — image query embeddings after norm_q."""
        x = q_img.detach().to(torch.float16)
        if self.reduce_over_batch:
            x = x.mean(dim=0)
        
        with torch.no_grad():
            if self.q_ema is None:
                self.q_ema = x.clone()
                self.attn_gradient = torch.tensor(0.0, device=x.device, dtype=torch.float16)
            else:
                self.attn_gradient = (self.q_ema - x).abs().mean()
                self.q_ema.mul_(self.ema_beta).add_(x, alpha=1.0 - self.ema_beta)


    def _apply_q_ema(self, q_img: torch.Tensor) -> torch.Tensor:
        """Replace q_img with EMA-blended version."""
        if self.q_ema is None:
            return q_img
        ema = self.q_ema
        if ema.dim() == 3:
            ema = ema.unsqueeze(0).expand_as(q_img)
        ema = ema.to(q_img.dtype)
        return self.ema_strength * ema + (1.0 - self.ema_strength) * q_img

    def _apply_ema_to_img_img(self, attn_img_img: torch.Tensor) -> torch.Tensor:
        """
        Blend EMA into image->image block only and return updated block.
        attn_img_img: [B, H, Q_img, K_img]
        """
        if self.attn_ema is None:
            return attn_img_img
        ema = self.attn_ema
        if ema.dim() == 3:
            # [H, Q_img, K_img] -> [B, H, Q_img, K_img]
            ema = ema.unsqueeze(0).expand_as(attn_img_img)
        ema = ema.to(attn_img_img.dtype)
        return self.ema_strength * ema + (1.0 - self.ema_strength) * attn_img_img

    def _apply_ema_to_img_rows(self, attn_img_rows: torch.Tensor) -> torch.Tensor:
        """
        Blend EMA into full image-query rows (img->img + img->txt) and return updated rows.
        attn_img_rows: [B, H, Q_img, K_total]
        """
        if self.attn_ema is None:
            return attn_img_rows
        ema = self.attn_ema
        if ema.dim() == 3:
            # [H, Q_img, K_total] -> [B, H, Q_img, K_total]
            ema = ema.unsqueeze(0).expand_as(attn_img_rows)
        ema = ema.to(attn_img_rows.dtype)
        return self.ema_strength * ema + (1.0 - self.ema_strength) * attn_img_rows


    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        B = hidden_states.shape[0]

        # Projections for image tokens
        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)

        inner_dim = k.shape[-1]
        H = attn.heads
        head_dim = inner_dim // H

        q = q.view(B, -1, H, head_dim).transpose(1, 2)  # [B,H,Q_img,D]
        k = k.view(B, -1, H, head_dim).transpose(1, 2)  # [B,H,K_img,D]
        v = v.view(B, -1, H, head_dim).transpose(1, 2)  # [B,H,K_img,D]

        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)

        Q_img = q.shape[2]
        K_img = k.shape[2]

        # Inserting the patch to support the Q EMA version
        if self.use_q_ema and self.enable_emag:
            if self.do_ema:
                self._update_q_ema(q)
            if self.update_ema_to_attn:
                q = self._apply_q_ema(q)
        # Projections for text/context tokens (if any)
        if encoder_hidden_states is not None:
            q_ctx = attn.add_q_proj(encoder_hidden_states)
            k_ctx = attn.add_k_proj(encoder_hidden_states)
            v_ctx = attn.add_v_proj(encoder_hidden_states)
            q_ctx = q_ctx.view(B, -1, H, head_dim).transpose(1, 2)  # [B,H,Q_txt,D]
            k_ctx = k_ctx.view(B, -1, H, head_dim).transpose(1, 2)  # [B,H,K_txt,D]
            v_ctx = v_ctx.view(B, -1, H, head_dim).transpose(1, 2)  # [B,H,K_txt,D]

            if attn.norm_added_q is not None:
                q_ctx = attn.norm_added_q(q_ctx)
            if attn.norm_added_k is not None:
                k_ctx = attn.norm_added_k(k_ctx)

            # Joint-attention: queries = [image_queries, text_queries]
            q_joint = torch.cat([q, q_ctx], dim=2)            # [B,H,Q_img+Q_txt,D]
            k_joint = torch.cat([k, k_ctx], dim=2)            # [B,H,K_img+K_txt,D]
            v_joint = torch.cat([v, v_ctx], dim=2)            # [B,H,K_img+K_txt,D]
            Q_total = q_joint.shape[2]
        else:
            q_joint, k_joint, v_joint = q, k, v
            Q_total = q_joint.shape[2]

        if self.use_q_ema and self.enable_emag:
            hidden = F.scaled_dot_product_attention(q_joint, k_joint, v_joint, attn_mask=attention_mask)
            hidden = hidden.transpose(1, 2).reshape(B, Q_total, H * head_dim)
        else:
            
            # Manual attention to access probabilities (no fused kernel, to keep attn map)
            scale = head_dim**-0.5
            attn_scores = torch.matmul(q_joint, k_joint.transpose(-2, -1)) * scale  # [B,H,Q_total,K_total]
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask

            attn_probs = attn_scores.softmax(dim=-1)  # [B,H,Q_total,K_total]
            
            # EMAG control – operate ONLY on image->image block (rows: image queries; cols: image keys)
            if self.attn_ema_full:
                if self.enable_emag:
                    # Full rows for image queries
                    attn_img_rows = attn_probs[:, :, :Q_img, :]  # [B, H, Q_img, K_total]
                    #attn_img_rows = attn_probs[:, :, :,K_img :]  # [B, H, Q_img, K_img]
                    # Update EMA and stats if requested
                    if self.do_ema:
                        self._update_ema_and_stats(attn_img_rows)

                    # Apply EMA replacement to full rows if requested
                    if self.update_ema_to_attn:
                        attn_img_rows = self._apply_ema_to_img_rows(attn_img_rows)
                        attn_probs[:, :, :Q_img, :] = attn_img_rows
                        #attn_probs[:, :, :, K_img:] = attn_img_rows

                        # Optional: reweight img->txt block and renormalize (only when text keys exist)
                        if encoder_hidden_states is not None and self.img2txt_scale != 1.0:
                            # scale img->txt portion
                            attn_img_txt = attn_probs[:, :, :Q_img, K_img:] * self.img2txt_scale

                            if self.renorm_mode == "row":
                                attn_probs[:, :, :Q_img, K_img:] = attn_img_txt
                                row_sum = attn_probs[:, :, :Q_img, :].sum(dim=-1, keepdim=True).clamp_min(1e-12)
                                attn_probs[:, :, :Q_img, :] = attn_probs[:, :, :Q_img, :] / row_sum

                            elif self.renorm_mode == "img2txt_only":
                                attn_probs[:, :, :Q_img, K_img:] = attn_img_txt
                                sum_img_img = attn_probs[:, :, :Q_img, :K_img].sum(dim=-1, keepdim=True)                # [B,H,Q_img,1]
                                sum_img_txt = attn_img_txt.sum(dim=-1, keepdim=True).clamp_min(1e-12)                   # [B,H,Q_img,1]
                                target_img_txt = (1.0 - sum_img_img).clamp_min(0.0)
                                scale = target_img_txt / sum_img_txt
                                attn_probs[:, :, :Q_img, K_img:] = attn_img_txt * scale

                            else:  # "none"
                                attn_probs[:, :, :Q_img, K_img:] = attn_img_txt

                        elif encoder_hidden_states is None and self.renorm_mode == "row":
                            # no text keys: rows already sum to 1 (EMA mix of softmax rows), renorm not needed
                            pass
            elif self.attn_ema_img2_txt_only:
                if self.enable_emag and encoder_hidden_states is not None:
                    # Slice img->txt only
                    attn_img_txt = attn_probs[:, :, :Q_img, K_img:]  # [B,H,Q_img,K_txt]

                    # Update EMA and stats
                    if self.do_ema:
                        self._update_ema_and_stats(attn_img_txt)  # reuses the generic updater

                    # Apply EMA replacement
                    if self.update_ema_to_attn:
                        # Reuse existing EMA applier; it just mixes shapes, works for K_txt too
                        attn_img_txt = self._apply_ema_to_img_img(attn_img_txt)
                        attn_probs[:, :, :Q_img, K_img:] = attn_img_txt

                        # Renormalization options
                        if self.renorm_mode == "row":
                            row_sum = attn_probs[:, :, :Q_img, :].sum(dim=-1, keepdim=True).clamp_min(1e-12)
                            attn_probs[:, :, :Q_img, :] = attn_probs[:, :, :Q_img, :] / row_sum
                        elif self.renorm_mode == "img2txt_only":
                            # Keep img->img fixed; scale img->txt to fill remainder of 1
                            sum_img_img = attn_probs[:, :, :Q_img, :K_img].sum(dim=-1, keepdim=True)                 # [B,H,Q_img,1]
                            sum_img_txt = attn_img_txt.sum(dim=-1, keepdim=True).clamp_min(1e-12)                    # [B,H,Q_img,1]
                            target_img_txt = (1.0 - sum_img_img).clamp_min(0.0)
                            scale = target_img_txt / sum_img_txt
                            attn_probs[:, :, :Q_img, K_img:] = attn_img_txt * scale
                        else:
                            # "none" – leave as-is
                            pass
        # fall through to rest of method
            else:
                if self.enable_emag:
                    
                    # Slice blocks
                    attn_img_img = attn_probs[:, :, :Q_img, :K_img]                    # [B,H,Q_img,K_img]
                    if encoder_hidden_states is not None:
                        attn_img_txt = attn_probs[:, :, :Q_img, K_img:]                # [B,H,Q_img,K_txt]

                    # Update EMA and stats if requested
                    if self.do_ema:
                        self._update_ema_and_stats(attn_img_img)

                    # Apply EMA replacement to image->image part if requested
                    if self.update_ema_to_attn:
                        attn_img_img = self._apply_ema_to_img_img(attn_img_img)

                        # Optional reweight img->txt block then renormalize row-wise (for image queries only)
                        if encoder_hidden_states is not None and self.img2txt_scale != 1.0:
                            attn_img_txt = attn_img_txt * self.img2txt_scale

                        # Put img->img back first
                        attn_probs[:, :, :Q_img, :K_img] = attn_img_img

                        if encoder_hidden_states is not None:
                            if self.renorm_mode == "row":
                                # full row renorm for image-query rows
                                attn_probs[:, :, :Q_img, K_img:] = attn_img_txt
                                row_sum = attn_probs[:, :, :Q_img, :].sum(dim=-1, keepdim=True).clamp_min(1e-12)
                                attn_probs[:, :, :Q_img, :] = attn_probs[:, :, :Q_img, :] / row_sum

                            elif self.renorm_mode == "img2txt_only":
                                # adjust only img->txt so row sums to 1, keep img->img unchanged
                                attn_probs[:, :, :Q_img, K_img:] = attn_img_txt
                                sum_img_img = attn_img_img.sum(dim=-1, keepdim=True)                                # [B,H,Q_img,1]
                                sum_img_txt = attn_img_txt.sum(dim=-1, keepdim=True).clamp_min(1e-12)               # [B,H,Q_img,1]
                                target_img_txt = (1.0 - sum_img_img).clamp_min(0.0)                                  # ensure non-negative
                                scale = target_img_txt / sum_img_txt
                                attn_img_txt = attn_img_txt * scale
                                attn_probs[:, :, :Q_img, K_img:] = attn_img_txt

                            else:  # "none"
                                # leave as-is (no renorm)
                                attn_probs[:, :, :Q_img, K_img:] = attn_img_txt
                        else:
                            # no text/context branch
                            if self.renorm_mode == "row":
                                row_sum = attn_probs[:, :, :Q_img, :K_img].sum(dim=-1, keepdim=True).clamp_min(1e-12)
                                attn_probs[:, :, :Q_img, :K_img] = attn_probs[:, :, :Q_img, :K_img] / row_sum
                            # "img2txt_only" and "none": nothing to do when no text keys

            # Dropout on attn probs (if training)
            attn_probs = F.dropout(attn_probs, p=0.0 if not hasattr(attn, "dropout") else attn.dropout, training=attn.training)

            # Context aggregation
            hidden = torch.matmul(attn_probs, v_joint)  # [B,H,Q_total,D]
            hidden = hidden.transpose(1, 2).reshape(B, Q_total, H * head_dim)
            hidden = hidden.to(q_joint.dtype)

        # Split outputs back to (image part, ctx part)
        if encoder_hidden_states is not None:
            hidden_states_out = hidden[:, :residual.shape[1], :]
            enc_hidden_states_out = hidden[:, residual.shape[1]:, :]
            if not attn.context_pre_only:
                enc_hidden_states_out = attn.to_add_out(enc_hidden_states_out)
        else:
            hidden_states_out = hidden
            enc_hidden_states_out = None

        # Final projection
        hidden_states_out = attn.to_out[0](hidden_states_out)
        hidden_states_out = attn.to_out[1](hidden_states_out)

        if enc_hidden_states_out is not None:
            return hidden_states_out, enc_hidden_states_out
        else:
            return hidden_states_out


def emag_forward(
    transformer,
    hidden_states,
    timestep,
    encoder_hidden_states,
    pooled_projections,
    joint_attention_kwargs,
    return_dict,
    emag_start_step=250,
    emag_stop_step=50,
    emag_time_delta=50,
    emag_adaptive_mode=2,
    emag_layers=None,              # optional: which layers to consider (defaults to [6,7,8])
    emag_groups=None,              # optional: list of lists for grouped selection like [[6,7],[8,9],...]
    emag_spaced=False,             # optional: randomly pick one from selected groups if True
    timestep_map=None,
    step_index=None,             # optional: list/array to map raw timestep -> index (like DiT)
):

    # 1) Resolve current step index 'curr_t' in the same semantics as DiT
    # - If a mapping is provided (preferred), use it
    # - Else, best-effort heuristics:
    #     - integers > 1 are treated as native discrete indices (e.g., 999..0)
    #     - floats in [0,1] are mapped to [0..1000] by (1 - t) * 1000
    # if isinstance(timestep, torch.Tensor):
    #     old_curr_t = float(timestep[0].item())
    # else:
    #     old_curr_t = float(timestep)

    # curr_t = None
    # if timestep_map is not None:
    #     try:
    #         # typical case: map raw scheduler "t" to index position
    #         t_int = int(round(old_curr_t))
    #         curr_t = int(list(timestep_map).index(t_int))
    #     except Exception:
    #         # fallback to heuristic below
    #         curr_t = None

    # if curr_t is None:
    #     if old_curr_t > 1.0:
    #         curr_t = int(round(old_curr_t))
    #     else:
    #         curr_t = int(round((1.0 - old_curr_t) * 1000.0))

    # 2) Decide windows for EMA accumulation and where to apply the EMA map
    # Convert the emag_start_step and emag_stop_step to the index of the current step
    do_ema = (step_index <= emag_start_step) and (step_index >= emag_stop_step)
    do_update = step_index <= (emag_start_step - emag_time_delta) and step_index >= emag_stop_step

    # 3) Default layers to consider if none provided
    if emag_layers is None:
        # safe default (matches your pipeline default)
        emag_layers = [6, 7, 8]

    # 4) Helper to fetch a processor for a given transformer block
    def get_proc(layer_idx):
        return transformer.transformer_blocks[layer_idx].attn.processor

    # 5) Select update target layers by attention gradient (MAE of live attn vs its EMA).
    def pick_layers_by_attn_grad(candidates):
        scored = []
        for i in candidates:
            proc = get_proc(i)
            val = getattr(proc, "attn_gradient", None)
            if val is None:
                continue
            try:
                scored.append((float(val.median().item()), i))
            except Exception:
                continue
        if not scored:
            print("No attention gradient found for the layers please checl")
            return candidates  # fallback
        scored.sort(key=lambda x: x[0], reverse=True)  # highest gradient first
        return [scored[0][1]]

    update_layers = list(emag_layers)
    if do_update:
        if emag_adaptive_mode == 0:
            # fixed: use the given layers as-is
            update_layers = list(emag_layers)
        elif emag_adaptive_mode == 2:
            # attention-gradient selection (paper method): pick the layer whose live attention
            # deviates most from its EMA (largest MAE).
            if emag_groups:
                selected = []
                for group in emag_groups:
                    pick = pick_layers_by_attn_grad(group)
                    selected.extend(pick)
                update_layers = [random.choice(selected)] if emag_spaced and selected else selected
            else:
                update_layers = pick_layers_by_attn_grad(emag_layers)
        else:
            raise ValueError(
                f"Invalid EMAG adaptive mode: {emag_adaptive_mode} (supported: 0=fixed, 2=attention-gradient)"
            )

    # 6) Apply toggles to processors
    toggled_do_ema = []
    toggled_update = []

    if do_ema:
        for i in emag_layers:
            proc = get_proc(i)
            try:
                setattr(proc, "do_ema", True)
                toggled_do_ema.append(i)
            except Exception:
                pass  # processor might not support it yet

    if do_update and update_layers:
        for i in update_layers:
            proc = get_proc(i)
            try:
                setattr(proc, "update_ema_to_attn", True)
                toggled_update.append(i)
            except Exception:
                pass  # processor might not support it yet
    
    # 7) Forward
    
    out = transformer(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
        joint_attention_kwargs=joint_attention_kwargs,
        return_dict=return_dict,
    )

    # 8) Reset toggles right after the call
    for i in toggled_do_ema:
        proc = get_proc(i)
        try:
            setattr(proc, "do_ema", False)
        except Exception:
            pass
    for i in toggled_update:
        proc = get_proc(i)
        try:
            setattr(proc, "update_ema_to_attn", False)
        except Exception:
            pass

    return out


