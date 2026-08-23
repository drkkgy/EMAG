# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

from warnings import warn
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp #Attention
from custom_code.custom_attention import Attention 
import torchvision.transforms as T
import torch.nn.functional as F
# SAG Helper functions

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
# APG imports
from custom_code.apg_helper import MomentumBuffer, project

# CADS imports
from custom_code.cads_helper import linear_schedule, add_noise,Timestetp_convertor

# S2 support imports
import random
#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        # Setting the layer_idx for each block  #### Custom Change ######
        for i,blk in enumerate(self.blocks):
            setattr(blk.attn, "layer_idx", i)
        ##########################################################
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x


    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    
    def enable_pag(self, mask=None, layers=[12,13,14,15]):
        for blk in self.blocks:
            if blk.attn.layer_idx in layers:
                blk.attn.enable_pag = True

    def disable_pag(self):
        for blk in self.blocks:
            blk.attn.enable_pag = False
    
    
    def forward_with_pag(self, x, t, y, pag_scale,layers=[12,13,14,15],negative_sample=False,y_null=None):
        """
        Forward pass of DiT, but also batches the pag forward pass for PAG.
        Dont send duplicated x for PAG.
        """
        if x.shape[0] % 2 == 0:
            # Test if x is duplicated 
            n = x.shape[0] // 2
            # exact equality (fast, strict). Use allclose if needed.
            is_dup = torch.equal(x[:n], x[n:])
            # or: is_dup = torch.allclose(x[:n], x[n:], rtol=0, atol=0)
            assert not is_dup, "x appears duplicated (CFG-style)."
        

        out = self.forward(x, t, y)
        eps, rest = out[:, :3], out[:, 3:]
        self.enable_pag(layers=layers)
        out_pag = self.forward(x, t, y)
        self.disable_pag()
        eps_pag, rest_pag = out_pag[:, :3], out_pag[:, 3:]
        if y_null is not None:
            cfg_scale = 1.5
            unoncd_out = self.forward(x, t, y_null)
            eps_uncond, rest_uncond = unoncd_out[:, :3], unoncd_out[:, 3:]
            eps_final = eps + cfg_scale * (eps - eps_uncond) + pag_scale * (eps - eps_pag)
        else:
            eps_final = eps + pag_scale * (eps - eps_pag)

        if negative_sample:
            print("Warning: Negative sampling is being generated with PAG")
            return torch.cat([eps_pag, rest], dim=1)
        return torch.cat([eps_final, rest], dim=1)

######## SAG Helper functions ##########################################################
    def enable_attn_capture(self, layers):
        for blk in self.blocks:
            if blk.attn.layer_idx in layers:
                blk.attn.capture_attn = True
                blk.attn.last_attn = None

    def disable_attn_capture(self):
        for blk in self.blocks:
            blk.attn.capture_attn = False

    def get_captured_attn(self):
        out = {}
        for blk in self.blocks:
            if getattr(blk.attn, "capture_attn", False) and blk.attn.last_attn is not None:
                out[blk.attn.layer_idx] = blk.attn.last_attn
                blk.attn.last_attn = None
        return out
    def get_sag_mask(self,attn_maps,threshold=1.0,size=(4,32,32)):

        C,H_img,W_img = size
        
        # Checking is multi layers or single layers for multiple layers we will take the mean of the attention maps
        if len(attn_maps.keys()) > 1:
            layers = [attn_maps[i] for i in attn_maps.keys()]
            A = torch.stack(layers, dim=0).mean(dim=0)               # B H Q K (batch, Heads, query, key) [Averaged across layers]
        else:
            A = list(attn_maps.values())[0]        # B H Q K (batch, Heads, query, key) [Single layers]
        # B = A.shape[0]
        # # Perform Global Average Pooling on the attention maps Link:- https://github.com/cvlab-kaist/Self-Attention-Guidance/blob/main/guided_diffusion/gaussian_diffusion.py
        A = A.mean(dim=1)                                     # B Q K (batch, query, key) [Averaged across Heads]
        A = A.sum(dim=1)                                     # B K (batch, Key) [summed across Queries]

        # Set the values greter than threshold to 1 and less than threshold to 0
        A = torch.where(A > threshold, 1, 0)
        
        # Rehsape to (B,1,hattn,wattn where hattn and wattn are the height and width of the attention map and should be equal to each other squared attention map)
        attn_res =  int(math.sqrt(A.shape[1]))
        

        
        A = A.reshape(A.shape[0],attn_res,attn_res).unsqueeze(1).repeat(1,C,1,1).int().float()
        
        M = F.interpolate(A,size=(H_img,W_img),mode='nearest')
        return M

    def apply_gausian_blur(self,x,sigma=1):
        # Setting kernel_size to 31 as per the paper 
        trasnform = T.GaussianBlur(kernel_size=31, sigma=sigma)
        return trasnform(x)
    # This function is taken from the SAG orignal implementation "https://github.com/cvlab-kaist/Self-Attention-Guidance/blob/main/guided_diffusion/gaussian_diffusion.py"
    
    # def _extract_into_tensor(self,arr, timesteps, broadcast_shape):
    #     """
    #     Extract values from a 1-D numpy array for a batch of indices.

    #     :param arr: the 1-D numpy array.
    #     :param timesteps: a tensor of indices into the array to extract.
    #     :param broadcast_shape: a larger shape of K dimensions with the batch
    #                             dimension equal to the length of timesteps.
    #     :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    #     """
    #     res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    #     while len(res.shape) < len(broadcast_shape):
    #         res = res[..., None]
    #     return res.expand(broadcast_shape)
    
    # def q_sample(self, x_start, t, noise=None,**kwargs):
    #     """
    #     Diffuse the data for a given number of diffusion steps.

    #     In other words, sample from q(x_t | x_0).

    #     :param x_start: the initial data batch.
    #     :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
    #     :param noise: if specified, the split-out normal noise.
    #     :return: A noisy version of x_start.
    #     """
    #     sqrt_alphas_cumprod = kwargs.get("sqrt_alphas_cumprod", None)
    #     sqrt_one_minus_alphas_cumprod = kwargs.get("sqrt_one_minus_alphas_cumprod", None)
    #     if noise is None:
    #         noise = torch.randn_like(x_start)
    #     assert noise.shape == x_start.shape
    #     return (
    #         self._extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape) * x_start
    #         + self._extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    #         * noise
    #     )

    def forward_with_sag(self, x, t, y, sag_scale,layers=[12,13,14,15],threshold=1.0,schedule=None,negative_sample=False,cfg_scale=None,y_null=None):
        """
        Forward pass of DiT, but also batches the forward pass for SAG.
        Dont send duplicated x for SAG.
        We keep the default threshold as 1.0 as per the paper.
        """

        if x.shape[0] % 2 == 0:
            # Test if x is duplicated 
            n = x.shape[0] // 2
            # exact equality (fast, strict). Use allclose if needed.
            is_dup = torch.equal(x[:n], x[n:])
            # or: is_dup = torch.allclose(x[:n], x[n:], rtol=0, atol=0)
            assert not is_dup, "x appears duplicated (CFG-style)."
        B,C,H,W = x.shape
        # Extracting the attention maps for the target layers
        self.enable_attn_capture(layers=layers)
        
        if cfg_scale is not None:
            out = self.forward(x, t, y_null)
        else:
            out = self.forward(x, t, y)
        #eps = out # Apply SAG to the entire output
        eps, rest = out[:, :C], out[:, C:]
        attn_maps = self.get_captured_attn()
        self.disable_attn_capture()

        

        mask = self.get_sag_mask(attn_maps,threshold=threshold,size=(C,H,W)) # We pass C to scale the mask for all channels
        
        x_0 = schedule.base_diffusion._predict_xstart_from_eps(x, t,eps)

        x_gaussian = self.apply_gausian_blur(x_0,sigma=1)
        I = torch.ones_like(mask)
        x_gaussian_til = schedule.base_diffusion.q_sample(x_gaussian, t,noise=eps)
        x_sag = (I-mask)*x + mask*x_gaussian_til

        # Add the noise to the x_sag for the current timestep
        #x_sag_bar = schedule.base_diffusion.q_sample(x_sag, t,noise=eps)

        if cfg_scale is not None:
            out_sag = self.forward(x_sag, t, y_null)
        else:
            out_sag = self.forward(x_sag, t, y)
        
        eps_sag, rest_sag = out_sag[:, :C], out_sag[:, C:]
        #eps_sag = out_sag

        if negative_sample:
            print("Warning: Negative sampling is being generated with SAG")
            return torch.cat([eps_sag, rest], dim=1)
        if cfg_scale is not None:
            eps_uncond = eps # For the CFG version we force the eps of the uncond as well as eps_sag 
            out_cond = self.forward(x, t, y)
            eps_cond, rest_cond = out_cond[:, :C], out_cond[:, C:]
            eps_final = eps_cond + cfg_scale * (eps_cond - eps_uncond) + sag_scale * (eps_uncond - eps_sag)
        else:
            eps_final = eps + sag_scale * (eps - eps_sag)
        #return eps_final
        return torch.cat([eps_final, rest], dim=1)

######## SAG Helper functions ##########################################################



######## SEG Helper functions ##########################################################

    def enable_seg(self, mask=None, layers=[12,13,14,15],blur_sigma=1):
        for blk in self.blocks:
            if blk.attn.layer_idx in layers:
                blk.attn.enable_seg = True
                blk.attn.blur_sigma = blur_sigma
    def disable_seg(self):
        for blk in self.blocks:
            blk.attn.enable_seg = False

    def forward_with_seg(self, x, t, y, seg_scale,layers=[12,13,14,15],blur_sigma=1,schedule=None,negative_sample=False,cfg_scale=None,y_null=None):
        """
        Forward pass of DiT, but also batches the forward pass for SEG.
        Dont send duplicated x for SEG.
        """
        if x.shape[0] % 2 == 0:
            # Test if x is duplicated 
            n = x.shape[0] // 2
            # exact equality (fast, strict). Use allclose if needed.
            is_dup = torch.equal(x[:n], x[n:])
            # or: is_dup = torch.allclose(x[:n], x[n:], rtol=0, atol=0)
            assert not is_dup, "x appears duplicated (CFG-style)."
        B,C,H,W = x.shape
        out = self.forward(x, t, y)
        eps, rest = out[:, :C], out[:, C:]
        
        
        if cfg_scale is not None:
            self.enable_seg(layers=layers,blur_sigma=blur_sigma)
            out_seg = self.forward(x, t, y)
            self.disable_seg()
            eps_seg, rest_seg = out_seg[:, :C], out_seg[:, C:]

            out = self.forward(x, t, y_null)
            eps_uncond, rest_cfg = out[:, :C], out[:, C:]

            #eps_final = eps_uncond + cfg_scale * (eps - eps_uncond) + seg_scale * (eps_uncond - eps_seg)
            eps_final = eps + (cfg_scale - 1.0)* (eps - eps_uncond) + seg_scale * (eps - eps_seg)
        else:
            self.enable_seg(layers=layers,blur_sigma=blur_sigma)
            out_seg = self.forward(x, t, y)
            self.disable_seg()
            eps_seg, rest_seg = out_seg[:, :C], out_seg[:, C:]
            eps_final = eps_seg + seg_scale * (eps - eps_seg)
        
        if negative_sample:
            print("Warning: Negative sampling is being generated with SEG")
            return torch.cat([eps_final, rest], dim=1)
        return torch.cat([eps_final, rest], dim=1)

######## SEG Helper functions ##########################################################
    
######## EMAG Helper functions ##########################################################

    # To control the EMA calculation and updating the EMA to the attention matrix
    def enable_do_ema(self,layers=[12,13,14,15]):
        for blk in self.blocks:
            if blk.attn.layer_idx in layers:
                blk.attn.do_ema = True
    
    def enable_update_ema_to_attn(self,layers=[12,13,14,15]):
        for blk in self.blocks:
            if blk.attn.layer_idx in layers:
                blk.attn.update_ema_to_attn = True
    
    def disable_do_ema(self,layers=[12,13,14,15]):
        for blk in self.blocks:
            if blk.attn.layer_idx in layers:
                blk.attn.do_ema = False
    
    def disable_update_ema_to_attn(self,layers=[12,13,14,15]):
        for blk in self.blocks:
            if blk.attn.layer_idx in layers:
                blk.attn.update_ema_to_attn = False


    def forward_with_emag(self, x, t, y, emag_scale,layers=[12,13,14,15],emag_start_step=150,emag_time_delta=50,emag_adaptive_mode=0,schedule=None,emag_spaced=False,negative_sample=False,y_null=None,mode="cond"):
        """
        Forward pass of DiT, but also batches the forward pass for EMAG.
        Dont send duplicated x for EMAG.
        """
        emag_stop_step = 50

        Lowest_entropy_layer = None
        Lowest_attention_gradient_layer = None
        grouped_list = None
        
        # Checking if the layers is a list of lists or a single list
        if isinstance(layers[0], list):
            grouped_list = layers
            layers = [item for sublist in layers for item in sublist]


        

        if x.shape[0] % 2 == 0:
            # Test if x is duplicated 
            n = x.shape[0] // 2
            # exact equality (fast, strict). Use allclose if needed.
            is_dup = torch.equal(x[:n], x[n:])
            # or: is_dup = torch.allclose(x[:n], x[n:], rtol=0, atol=0)
            assert not is_dup, "x appears duplicated (CFG-style)."
        
        do_emag = False
        update_ema_to_attn = False
        old_curr_t = t[0].item()
        # Mapping the current timestep to the diffusion schedule since we use DDIM sampler
        curr_t = schedule.timestep_map.index(old_curr_t)
        # Enabling EMA calcuation if the t is less than equal to the target step
        if curr_t <= emag_start_step and curr_t >= emag_stop_step:
            self.enable_do_ema(layers=layers)
            do_emag = True
        
        out = self.forward(x, t, y)
        eps, rest = out[:, :3], out[:, 3:]
        out_uncond = self.forward(x, t, y_null)
        eps_uncond, rest_uncond = out_uncond[:, :3], out_uncond[:, 3:]
        
        # Disabling the EMA if it was enabled
        if do_emag:
            do_emag = False
            self.disable_do_ema(layers=layers)

        # Updating the EMA to the attention matrix if the t is less than equal to the target step - time delta

        if curr_t <= emag_start_step - emag_time_delta and curr_t >= emag_stop_step:    
            if emag_adaptive_mode == 0:
                self.enable_update_ema_to_attn(layers=layers)
                update_ema_to_attn = True
            elif emag_adaptive_mode == 1:
                """
                Adapt using Entropy only for a batched input we take median of the entropy values
                """
                if grouped_list is not None:
                    # Grouped wise entropy calcuation and layer selection 
                    target_layers = []
                    for group in grouped_list:
                        entropy_values = [(blk.attn.attn_entropy.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in group]
                        entropy_values.sort(key=lambda x: x[0])
                        target_layers.append(entropy_values[0][1])
                    # target_layers = [item for sublist in target_layers for item in sublist]
                    if emag_spaced:
                        choice = random.choice(target_layers)
                        self.enable_update_ema_to_attn(layers=[choice])
                    else:
                        Lowest_entropy_layer = target_layers
                        self.enable_update_ema_to_attn(layers=Lowest_entropy_layer)
                    update_ema_to_attn = True
                else:
                    # Extract the entropy values for the target layers
                    entropy_values = [(blk.attn.attn_entropy.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in layers]
                    # Sort the entropy values in ascending order
                    entropy_values.sort(key=lambda x: x[0])
                    Lowest_entropy_layer = entropy_values[0][1]
                
                    self.enable_update_ema_to_attn(layers=[Lowest_entropy_layer])
                    update_ema_to_attn = True

            elif emag_adaptive_mode == 2:
                """
                Adapt using Attention Gradient only for a batched input we take median of the attention gradient values
                """
                if grouped_list is not None:
                    # Grouped wise attention gradient calcuation and layer selection 
                    target_layers = []
                    for group in grouped_list:
                        attention_gradient_values = [(blk.attn.attn_gradient.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in group]
                        attention_gradient_values.sort(key=lambda x: x[0],reverse=True)
                        target_layers.append(attention_gradient_values[0][1])
                    
                    if emag_spaced:
                        choice = random.choice(target_layers)
                        self.enable_update_ema_to_attn(layers=[choice])
                    else:
                        highest_attention_gradient_layer = target_layers
                        self.enable_update_ema_to_attn(layers=highest_attention_gradient_layer)
                    update_ema_to_attn = True
                    
                else:
                    # Extract the attention gradient values for the target layers
                    attention_gradient_values = [(blk.attn.attn_gradient.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in layers]
                    # Sort the attention gradient values in ascending order
                    attention_gradient_values.sort(key=lambda x: x[0],reverse=True)
                    highest_attention_gradient_layer = attention_gradient_values[0][1]

                    self.enable_update_ema_to_attn(layers=[highest_attention_gradient_layer])
                    update_ema_to_attn = True
            elif emag_adaptive_mode == 3:
                """
                Adapt using both Entropy and Attention Gradient for a batched input we take median of the entropy and attention gradient values
                """
                if grouped_list is not None:
                    raise ValueError("Grouped adaptive mode is not supported for EMAG adaptive mode 3")
                # Extract the entropy values for the target layers
                entropy_values = [(blk.attn.attn_entropy.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in layers]
                # Sort the entropy values in ascending order
                entropy_values.sort(key=lambda x: x[0])
                lowest_five_entropy_layers = entropy_values[:5]
                lowest_layer_idx = [x[1] for x in lowest_five_entropy_layers]
                # Extract the attention gradient values for the target layers
                attention_gradient_values = [(blk.attn.attn_gradient.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in lowest_layer_idx]
                # Sort the attention gradient values in ascending order
                attention_gradient_values.sort(key=lambda x: x[0],reverse=True)
                highest_attention_gradient_layer = attention_gradient_values[0][1]

                self.enable_update_ema_to_attn(layers=[highest_attention_gradient_layer])
                update_ema_to_attn = True
            else:
                raise ValueError(f"Invalid EMAG adaptive mode: {emag_adaptive_mode}")
        
        # out_emag = self.forward(x, t, y_null)
        # eps_emag_uncond, rest_emag = out_emag[:, :3], out_emag[:, 3:]
        out_emag_cond = self.forward(x, t, y)
        eps_emag_cond, rest_emag_cond = out_emag_cond[:, :3], out_emag_cond[:, 3:]

        # Disabling the EMA update if it was enabled
        if update_ema_to_attn:
            update_ema_to_attn = False
            if Lowest_entropy_layer is not None:
                self.disable_update_ema_to_attn(layers=[Lowest_entropy_layer])
            elif Lowest_attention_gradient_layer is not None:
                self.disable_update_ema_to_attn(layers=[Lowest_attention_gradient_layer])
            else:
                self.disable_update_ema_to_attn(layers=layers)

        
        
        #eps_final = eps + emag_scale * (eps - eps_emag)
        #emag_diff = emag_scale * (eps - eps_emag)
        #diff_parallel, diff_orthogonal = project(emag_diff,- eps)
        #eps_final = eps + emag_diff
        #eps_final = eps_emag_uncond + emag_scale * (eps - eps_emag_uncond) # Changed from eps + emag_scale * (eps - eps_emag) Ver 4_imp
        # eps = eps_emag_cond + emag_scale * (eps - eps_emag_cond) # Ver 5_imp 2
        # eps_final = eps_emag_uncond + emag_scale * (eps - eps_emag_uncond) # Ver 5_imp 2
        #diff = eps - eps_emag_cond
        #diff_parallel, diff_orthogonal = project(diff,eps) #Ver_7_imp
        # Ver 9 we keep everythong same as Ver 5 but nopw we set the emag scale to 1.75 for EMAG and the main scle for now is CFG fix the pipe for this.

        #eps = eps_emag_cond  + 3 * diff_orthogonal + emag_scale * diff_parallel#Ver_7_imp   Ver_8 -> replaced eps with eps_emag_cond
        if mode == "cond":
            eps = eps_emag_cond + emag_scale * (eps - eps_emag_cond) # Ver 5_imp 3 only different from Ver 5_imp 2 is the scale factor of 2 on emag_scale
            
            eps_final = eps_uncond + 1.5 * (eps - eps_uncond) # Ver 5_imp 3
        elif mode == "uncond":
            
            eps_final = eps_emag_cond + emag_scale * (eps_uncond - eps_emag_cond) # Ver 5_imp 3 # out_emag_cond will be uncond for the uncond mode where the y lable are also Null. 7 is the perfect scle for uncond mode.
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if negative_sample:
            print("Warning: Negative sampling is being generated with EMAG")
            return torch.cat([eps_emag_cond, rest], dim=1)
        
        return torch.cat([eps_final, rest], dim=1)


    ######## EMAG with CFG Function ##########################################################
    def forward_with_emag_cfg(self, x, t, y, cfg_scale, emag_scale,layers=[12,13,14,15],emag_start_step=150,emag_time_delta=50,emag_adaptive_mode=0,schedule=None,emag_spaced=False,negative_sample=False):
        """
        Forward pass of DiT, but also batches the forward pass for EMAG.
        Dont send duplicated x for EMAG.
        """

        Lowest_entropy_layer = None
        Lowest_attention_gradient_layer = None
        grouped_list = None
        emag_stop_step = 50
        
        # Checking if the layers is a list of lists or a single list
        if isinstance(layers[0], list):
            grouped_list = layers
            layers = [item for sublist in layers for item in sublist]


        

        # if x.shape[0] % 2 == 0:
        #     # Test if x is duplicated 
        #     n = x.shape[0] // 2
        #     # exact equality (fast, strict). Use allclose if needed.
        #     is_dup = torch.equal(x[:n], x[n:])
        #     # or: is_dup = torch.allclose(x[:n], x[n:], rtol=0, atol=0)
        #     assert not is_dup, "x appears duplicated (CFG-style)."
        
        do_emag = False
        update_ema_to_attn = False
        old_curr_t = t[0].item()
        # Mapping the current timestep to the diffusion schedule since we use DDIM sampler
        curr_t = schedule.timestep_map.index(old_curr_t)
        # Enabling EMA calcuation if the t is less than equal to the target step
        if curr_t <= emag_start_step and curr_t >= emag_stop_step:
            self.enable_do_ema(layers=layers)
            do_emag = True
        
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        out = self.forward(combined, t, y)
        eps, rest = out[:, :3], out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        # Disabling the EMA if it was enabled
        if do_emag:
            do_emag = False
            self.disable_do_ema(layers=layers)

        # Updating the EMA to the attention matrix if the t is less than equal to the target step - time delta

        if curr_t <= emag_start_step - emag_time_delta:
            if emag_adaptive_mode == 0:
                self.enable_update_ema_to_attn(layers=layers)
                update_ema_to_attn = True
            elif emag_adaptive_mode == 1:
                """
                Adapt using Entropy only for a batched input we take median of the entropy values
                """
                if grouped_list is not None:
                    # Grouped wise entropy calcuation and layer selection 
                    target_layers = []
                    for group in grouped_list:
                        entropy_values = [(blk.attn.attn_entropy.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in group]
                        entropy_values.sort(key=lambda x: x[0])
                        target_layers.append(entropy_values[0][1])
                    # target_layers = [item for sublist in target_layers for item in sublist]
                    if emag_spaced:
                        choice = random.choice(target_layers)
                        self.enable_update_ema_to_attn(layers=[choice])
                    else:
                        Lowest_entropy_layer = target_layers
                        self.enable_update_ema_to_attn(layers=Lowest_entropy_layer)
                    update_ema_to_attn = True
                else:
                    # Extract the entropy values for the target layers
                    entropy_values = [(blk.attn.attn_entropy.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in layers]
                    # Sort the entropy values in ascending order
                    entropy_values.sort(key=lambda x: x[0])
                    Lowest_entropy_layer = entropy_values[0][1]
                
                    self.enable_update_ema_to_attn(layers=[Lowest_entropy_layer])
                    update_ema_to_attn = True

            elif emag_adaptive_mode == 2:
                """
                Adapt using Attention Gradient only for a batched input we take median of the attention gradient values
                """
                if grouped_list is not None:
                    # Grouped wise attention gradient calcuation and layer selection 
                    target_layers = []
                    for group in grouped_list:
                        attention_gradient_values = [(blk.attn.attn_gradient.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in group]
                        attention_gradient_values.sort(key=lambda x: x[0],reverse=True)
                        target_layers.append(attention_gradient_values[0][1])
                    
                    if emag_spaced:
                        choice = random.choice(target_layers)
                        self.enable_update_ema_to_attn(layers=[choice])
                    else:
                        highest_attention_gradient_layer = target_layers
                        self.enable_update_ema_to_attn(layers=highest_attention_gradient_layer)
                    update_ema_to_attn = True
                    
                else:
                    # Extract the attention gradient values for the target layers
                    attention_gradient_values = [(blk.attn.attn_gradient.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in layers]
                    # Sort the attention gradient values in ascending order
                    attention_gradient_values.sort(key=lambda x: x[0],reverse=True)
                    highest_attention_gradient_layer = attention_gradient_values[0][1]

                    self.enable_update_ema_to_attn(layers=[highest_attention_gradient_layer])
                    update_ema_to_attn = True
            elif emag_adaptive_mode == 3:
                """
                Adapt using both Entropy and Attention Gradient for a batched input we take median of the entropy and attention gradient values
                """
                if grouped_list is not None:
                    raise ValueError("Grouped adaptive mode is not supported for EMAG adaptive mode 3")
                # Extract the entropy values for the target layers
                entropy_values = [(blk.attn.attn_entropy.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in layers]
                # Sort the entropy values in ascending order
                entropy_values.sort(key=lambda x: x[0])
                lowest_five_entropy_layers = entropy_values[:5]
                lowest_layer_idx = [x[1] for x in lowest_five_entropy_layers]
                # Extract the attention gradient values for the target layers
                attention_gradient_values = [(blk.attn.attn_gradient.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in lowest_layer_idx]
                # Sort the attention gradient values in ascending order
                attention_gradient_values.sort(key=lambda x: x[0],reverse=True)
                highest_attention_gradient_layer = attention_gradient_values[0][1]

                self.enable_update_ema_to_attn(layers=[highest_attention_gradient_layer])
                update_ema_to_attn = True
            else:
                raise ValueError(f"Invalid EMAG adaptive mode: {emag_adaptive_mode}")

        
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        out_emag = self.forward(combined, t, y)
        eps_emag, rest_emag = out_emag[:, :3], out_emag[:, 3:]
        cond_eps_emag, uncond_eps_emag = torch.split(eps_emag, len(eps_emag) // 2, dim=0)

        # Disabling the EMA update if it was enabled
        if update_ema_to_attn:
            update_ema_to_attn = False
            if Lowest_entropy_layer is not None:
                self.disable_update_ema_to_attn(layers=[Lowest_entropy_layer])
            elif Lowest_attention_gradient_layer is not None:
                self.disable_update_ema_to_attn(layers=[Lowest_attention_gradient_layer])
            else:
                self.disable_update_ema_to_attn(layers=layers)

        
        # Replacing the cond with the improved cond eps with EMAG
        #cond_eps = cond_eps + emag_scale * (cond_eps - cond_eps_emag)
        #eps_final = cfg_scale*cond_eps + (1 - cfg_scale)*uncond_eps + emag_scale * (cond_eps - cond_eps_emag)
        # cond_eps = cond_eps + emag_scale * (cond_eps - cond_eps_emag)
        # uncond_eps = uncond_eps + emag_scale * (uncond_eps - uncond_eps_emag)
        #eps_final = cfg_scale*cond_eps + (1 - cfg_scale) * uncond_eps - emag_scale * (cond_eps_emag)      # S2 style of guidance Note: Emag scale should be small here like 0.75
        
        eps_final = cfg_scale * cond_eps + (1 - cfg_scale) * uncond_eps + emag_scale * (cond_eps - uncond_eps_emag) # Orignal Style  This is for Ver5 in CFG for EMAG adaptive -> /hpcfs/users/a1791904/Research_3/Self_Refinement/outputs_5000/class_conditional/DiT/EMAG_adaptive/EMAG_CFG/Ver5
        
        # Version 8 Suggestion from Prof Liu 
        # EQ = U + W (C-U) + S(U/2 + C/2 - C_pret)
        #eps_final = uncond_eps + cfg_scale * (cond_eps - uncond_eps) + emag_scale * (uncond_eps/2 + cond_eps/2 - cond_eps_emag)
        
        # Version 9 Suggestion from Prof Liu 
        # EQ X = ((1-w)U + wC )        (1-S)X + SC_pret​
        
        # cond_eps = cond_eps + emag_scale * (cond_eps - cond_eps_emag)
        # eps_final = (1-cfg_scale) * uncond_eps + cfg_scale * cond_eps


        if negative_sample:
            print("Warning: Negative sampling is being generated with EMAG")
            return torch.cat([eps_emag, rest], dim=1)
            
        eps_final = torch.cat([eps_final, eps_final], dim=0)
        return torch.cat([eps_final, rest], dim=1)

######## EMAG with APG Function ##########################################################
    
    def forward_with_emag_apg(self, x, t, y, apg_scale, emag_scale,layers=[12,13,14,15],emag_start_step=150,
                             emag_time_delta=50,emag_adaptive_mode=0,schedule=None,emag_spaced=False,negative_sample=False,
                             momentum_buffer=None):
        """
        Forward pass of DiT, but also batches the forward pass for EMAG.
        Dont send duplicated x for EMAG.
        """
        # Setting up the params for the APG gudiance
        norm_threshold: float = 5 # From the paper recommended value for DIT XL/2 model
        eta: float = 0.0 # From the paper recommended value for DIT XL/2 model

        Lowest_entropy_layer = None
        Lowest_attention_gradient_layer = None
        grouped_list = None
        emag_stop_step = 50
        
        # Checking if the layers is a list of lists or a single list
        if isinstance(layers[0], list):
            grouped_list = layers
            layers = [item for sublist in layers for item in sublist]


        

        # if x.shape[0] % 2 == 0:
        #     # Test if x is duplicated 
        #     n = x.shape[0] // 2
        #     # exact equality (fast, strict). Use allclose if needed.
        #     is_dup = torch.equal(x[:n], x[n:])
        #     # or: is_dup = torch.allclose(x[:n], x[n:], rtol=0, atol=0)
        #     assert not is_dup, "x appears duplicated (CFG-style)."
        
        do_emag = False
        update_ema_to_attn = False
        old_curr_t = t[0].item()
        # Mapping the current timestep to the diffusion schedule since we use DDIM sampler
        curr_t = schedule.timestep_map.index(old_curr_t)
        # Enabling EMA calcuation if the t is less than equal to the target step
        if curr_t <= emag_start_step and curr_t >= emag_stop_step:
            self.enable_do_ema(layers=layers)
            do_emag = True
        
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        out = self.forward(combined, t, y)
        eps, rest = out[:, :3], out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)


        
        # Disabling the EMA if it was enabled
        if do_emag:
            do_emag = False
            self.disable_do_ema(layers=layers)

        # Updating the EMA to the attention matrix if the t is less than equal to the target step - time delta

        if curr_t <= emag_start_step - emag_time_delta:
            if emag_adaptive_mode == 0:
                self.enable_update_ema_to_attn(layers=layers)
                update_ema_to_attn = True
            elif emag_adaptive_mode == 1:
                """
                Adapt using Entropy only for a batched input we take median of the entropy values
                """
                if grouped_list is not None:
                    # Grouped wise entropy calcuation and layer selection 
                    target_layers = []
                    for group in grouped_list:
                        entropy_values = [(blk.attn.attn_entropy.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in group]
                        entropy_values.sort(key=lambda x: x[0])
                        target_layers.append(entropy_values[0][1])
                    # target_layers = [item for sublist in target_layers for item in sublist]
                    if emag_spaced:
                        choice = random.choice(target_layers)
                        self.enable_update_ema_to_attn(layers=[choice])
                    else:
                        Lowest_entropy_layer = target_layers
                        self.enable_update_ema_to_attn(layers=Lowest_entropy_layer)
                    update_ema_to_attn = True
                else:
                    # Extract the entropy values for the target layers
                    entropy_values = [(blk.attn.attn_entropy.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in layers]
                    # Sort the entropy values in ascending order
                    entropy_values.sort(key=lambda x: x[0])
                    Lowest_entropy_layer = entropy_values[0][1]
                
                    self.enable_update_ema_to_attn(layers=[Lowest_entropy_layer])
                    update_ema_to_attn = True

            elif emag_adaptive_mode == 2:
                """
                Adapt using Attention Gradient only for a batched input we take median of the attention gradient values
                """
                if grouped_list is not None:
                    # Grouped wise attention gradient calcuation and layer selection 
                    target_layers = []
                    for group in grouped_list:
                        attention_gradient_values = [(blk.attn.attn_gradient.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in group]
                        attention_gradient_values.sort(key=lambda x: x[0],reverse=True)
                        target_layers.append(attention_gradient_values[0][1])
                    
                    if emag_spaced:
                        choice = random.choice(target_layers)
                        self.enable_update_ema_to_attn(layers=[choice])
                    else:
                        highest_attention_gradient_layer = target_layers
                        self.enable_update_ema_to_attn(layers=highest_attention_gradient_layer)
                    update_ema_to_attn = True
                    
                else:
                    # Extract the attention gradient values for the target layers
                    attention_gradient_values = [(blk.attn.attn_gradient.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in layers]
                    # Sort the attention gradient values in ascending order
                    attention_gradient_values.sort(key=lambda x: x[0],reverse=True)
                    highest_attention_gradient_layer = attention_gradient_values[0][1]

                    self.enable_update_ema_to_attn(layers=[highest_attention_gradient_layer])
                    update_ema_to_attn = True
            elif emag_adaptive_mode == 3:
                """
                Adapt using both Entropy and Attention Gradient for a batched input we take median of the entropy and attention gradient values
                """
                if grouped_list is not None:
                    raise ValueError("Grouped adaptive mode is not supported for EMAG adaptive mode 3")
                # Extract the entropy values for the target layers
                entropy_values = [(blk.attn.attn_entropy.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in layers]
                # Sort the entropy values in ascending order
                entropy_values.sort(key=lambda x: x[0])
                lowest_five_entropy_layers = entropy_values[:5]
                lowest_layer_idx = [x[1] for x in lowest_five_entropy_layers]
                # Extract the attention gradient values for the target layers
                attention_gradient_values = [(blk.attn.attn_gradient.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in lowest_layer_idx]
                # Sort the attention gradient values in ascending order
                attention_gradient_values.sort(key=lambda x: x[0],reverse=True)
                highest_attention_gradient_layer = attention_gradient_values[0][1]

                self.enable_update_ema_to_attn(layers=[highest_attention_gradient_layer])
                update_ema_to_attn = True
            else:
                raise ValueError(f"Invalid EMAG adaptive mode: {emag_adaptive_mode}")

        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        out_emag = self.forward(combined, t, y)
        eps_emag, rest_emag = out_emag[:, :3], out_emag[:, 3:]
        cond_eps_emag, uncond_eps_emag = torch.split(eps_emag, len(eps_emag) // 2, dim=0)

        # Disabling the EMA update if it was enabled
        if update_ema_to_attn:
            update_ema_to_attn = False
            if Lowest_entropy_layer is not None:
                self.disable_update_ema_to_attn(layers=[Lowest_entropy_layer])
            elif Lowest_attention_gradient_layer is not None:
                self.disable_update_ema_to_attn(layers=[Lowest_attention_gradient_layer])
            else:
                self.disable_update_ema_to_attn(layers=layers)

        # # Applying the EMAG to the cond and uncond eps
        # cond_eps = cond_eps + emag_scale * (cond_eps - cond_eps_emag)
        # uncond_eps = uncond_eps + emag_scale * (uncond_eps - uncond_eps_emag)

        #eps_final = eps + emag_scale * (eps - eps_emag)
        # eps_final = cond_eps + (apg_scale - 1) * normalized_update
        
        ############################################################################################
        cond_eps = cond_eps_emag + emag_scale * (cond_eps - cond_eps_emag)
        # Doing the calcaution for the components for the APG gudiance computation

        # Use the half-batch inputs and times for x0 conversion
        t_half = t[: len(half)]

        # Mapping to DDIM timestep
        map_tensor = torch.tensor(schedule.timestep_map, device=t_half.device,dtype=t_half.dtype)
        inv_map = torch.full((schedule.original_num_steps,), -1, device=t_half.device,dtype=t_half.dtype)
        inv_map[map_tensor] = torch.arange(len(map_tensor), device=t_half.device,dtype=t_half.dtype)
        t_half = inv_map[t_half]

        x_t_half = half[:, :cond_eps.shape[1], :, :]  # match channel count (3)

        # eps -> x0
        cond_eps   = schedule._predict_xstart_from_eps(x_t_half, t_half, cond_eps)
        uncond_eps = schedule._predict_xstart_from_eps(x_t_half, t_half, uncond_eps)
        # cond_eps_emag = schedule._predict_xstart_from_eps(x_t_half, t_half, cond_eps_emag)
        
        # emag_delta = emag_scale * (cond_eps - cond_eps_emag)

        diff = cond_eps - uncond_eps
        if momentum_buffer is not None:
            momentum_buffer.update(diff)
            diff = momentum_buffer.running_average

        
        if norm_threshold > 0:
            ones = torch.ones_like(diff)
            diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
            scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
            diff = diff * scale_factor
        diff_parallel, diff_orthogonal = project(diff, cond_eps)
        normalized_update = diff_orthogonal + eta * diff_parallel
        ############################################################################################

        
        #eps_final = cond_eps + (apg_scale - 1) * normalized_update + emag_scale * (cond_eps - cond_eps_emag)
        eps_final = cond_eps + (apg_scale - 1) * normalized_update      #- emag_scale * (cond_eps_emag) # S2 style of guidance

        # x0 -> eps
        eps_final = schedule._predict_eps_from_xstart(x_t_half, t_half, eps_final)
        # Applying EAMG updates now
        #eps_final = eps_final + emag_delta
        ######################3

        if negative_sample:
            print("Warning: Negative sampling is being generated with EMAG")
            return torch.cat([eps_emag, rest], dim=1)

        eps_final = torch.cat([eps_final, eps_final], dim=0)
        return torch.cat([eps_final, rest], dim=1)


###
######## EMAG with CADS Function ##########################################################
    def forward_with_emag_cads(self, x, t, y, cfg_scale, emag_scale,layers=[12,13,14,15],emag_start_step=150,
                             emag_time_delta=50,emag_adaptive_mode=0,schedule=None,emag_spaced=False,negative_sample=False,
                             cads_t1=0.475, cads_t2=0.9, cads_s=0.006, cads_psi=1):
        """
        Forward pass of DiT, but also batches the forward pass for EMAG.
        Dont send duplicated x for EMAG.
        """
        emag_stop_step = 50
        # Setting up the params for the CADS gudiance
        cads_t1 = Timestetp_convertor(torch.tensor(cads_t1), 1000)
        cads_t2 = Timestetp_convertor(torch.tensor(cads_t2), 1000)
        gamma = linear_schedule(t[0].item(), cads_t1, cads_t2)


        Lowest_entropy_layer = None
        Lowest_attention_gradient_layer = None
        grouped_list = None
        
        # Checking if the layers is a list of lists or a single list
        if isinstance(layers[0], list):
            grouped_list = layers
            layers = [item for sublist in layers for item in sublist]


        

        # if x.shape[0] % 2 == 0:
        #     # Test if x is duplicated 
        #     n = x.shape[0] // 2
        #     # exact equality (fast, strict). Use allclose if needed.
        #     is_dup = torch.equal(x[:n], x[n:])
        #     # or: is_dup = torch.allclose(x[:n], x[n:], rtol=0, atol=0)
        #     assert not is_dup, "x appears duplicated (CFG-style)."
        
        do_emag = False
        update_ema_to_attn = False
        old_curr_t = t[0].item()
        # Mapping the current timestep to the diffusion schedule since we use DDIM sampler
        curr_t = schedule.timestep_map.index(old_curr_t)
        # Enabling EMA calcuation if the t is less than equal to the target step
        if curr_t <= emag_start_step and curr_t >= emag_stop_step:
            self.enable_do_ema(layers=layers)
            do_emag = True
        
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        out = self.forward_cads_with_noisy_condtion(combined, t, y, gamma, cads_s, cads_psi)
        eps, rest = out[:, :3], out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)


        
        # Disabling the EMA if it was enabled
        if do_emag:
            do_emag = False
            self.disable_do_ema(layers=layers)

        # Updating the EMA to the attention matrix if the t is less than equal to the target step - time delta

        if curr_t <= emag_start_step - emag_time_delta:
            if emag_adaptive_mode == 0:
                self.enable_update_ema_to_attn(layers=layers)
                update_ema_to_attn = True
            elif emag_adaptive_mode == 1:
                """
                Adapt using Entropy only for a batched input we take median of the entropy values
                """
                if grouped_list is not None:
                    # Grouped wise entropy calcuation and layer selection 
                    target_layers = []
                    for group in grouped_list:
                        entropy_values = [(blk.attn.attn_entropy.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in group]
                        entropy_values.sort(key=lambda x: x[0])
                        target_layers.append(entropy_values[0][1])
                    # target_layers = [item for sublist in target_layers for item in sublist]
                    if emag_spaced:
                        choice = random.choice(target_layers)
                        self.enable_update_ema_to_attn(layers=[choice])
                    else:
                        Lowest_entropy_layer = target_layers
                        self.enable_update_ema_to_attn(layers=Lowest_entropy_layer)
                    update_ema_to_attn = True
                else:
                    # Extract the entropy values for the target layers
                    entropy_values = [(blk.attn.attn_entropy.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in layers]
                    # Sort the entropy values in ascending order
                    entropy_values.sort(key=lambda x: x[0])
                    Lowest_entropy_layer = entropy_values[0][1]
                
                    self.enable_update_ema_to_attn(layers=[Lowest_entropy_layer])
                    update_ema_to_attn = True

            elif emag_adaptive_mode == 2:
                """
                Adapt using Attention Gradient only for a batched input we take median of the attention gradient values
                """
                if grouped_list is not None:
                    # Grouped wise attention gradient calcuation and layer selection 
                    target_layers = []
                    for group in grouped_list:
                        attention_gradient_values = [(blk.attn.attn_gradient.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in group]
                        attention_gradient_values.sort(key=lambda x: x[0],reverse=True)
                        target_layers.append(attention_gradient_values[0][1])
                    
                    if emag_spaced:
                        choice = random.choice(target_layers)
                        self.enable_update_ema_to_attn(layers=[choice])
                    else:
                        highest_attention_gradient_layer = target_layers
                        self.enable_update_ema_to_attn(layers=highest_attention_gradient_layer)
                    update_ema_to_attn = True
                    
                else:
                    # Extract the attention gradient values for the target layers
                    attention_gradient_values = [(blk.attn.attn_gradient.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in layers]
                    # Sort the attention gradient values in ascending order
                    attention_gradient_values.sort(key=lambda x: x[0],reverse=True)
                    highest_attention_gradient_layer = attention_gradient_values[0][1]

                    self.enable_update_ema_to_attn(layers=[highest_attention_gradient_layer])
                    update_ema_to_attn = True
            elif emag_adaptive_mode == 3:
                """
                Adapt using both Entropy and Attention Gradient for a batched input we take median of the entropy and attention gradient values
                """
                if grouped_list is not None:
                    raise ValueError("Grouped adaptive mode is not supported for EMAG adaptive mode 3")
                # Extract the entropy values for the target layers
                entropy_values = [(blk.attn.attn_entropy.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in layers]
                # Sort the entropy values in ascending order
                entropy_values.sort(key=lambda x: x[0])
                lowest_five_entropy_layers = entropy_values[:5]
                lowest_layer_idx = [x[1] for x in lowest_five_entropy_layers]
                # Extract the attention gradient values for the target layers
                attention_gradient_values = [(blk.attn.attn_gradient.median(),blk.attn.layer_idx) for blk in self.blocks if blk.attn.layer_idx in lowest_layer_idx]
                # Sort the attention gradient values in ascending order
                attention_gradient_values.sort(key=lambda x: x[0],reverse=True)
                highest_attention_gradient_layer = attention_gradient_values[0][1]

                self.enable_update_ema_to_attn(layers=[highest_attention_gradient_layer])
                update_ema_to_attn = True
            else:
                raise ValueError(f"Invalid EMAG adaptive mode: {emag_adaptive_mode}")

        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        out_emag = self.forward_cads_with_noisy_condtion(combined, t, y, gamma, cads_s, cads_psi)
        eps_emag, rest_emag = out_emag[:, :3], out_emag[:, 3:]
        cond_eps_emag, uncond_eps_emag = torch.split(eps_emag, len(eps_emag) // 2, dim=0)

        # Disabling the EMA update if it was enabled
        if update_ema_to_attn:
            update_ema_to_attn = False
            if Lowest_entropy_layer is not None:
                self.disable_update_ema_to_attn(layers=[Lowest_entropy_layer])
            elif Lowest_attention_gradient_layer is not None:
                self.disable_update_ema_to_attn(layers=[Lowest_attention_gradient_layer])
            else:
                self.disable_update_ema_to_attn(layers=layers)

        
        # Applying the EMAG to the cond and uncond eps
        # cond_eps = cond_eps + emag_scale * (cond_eps - cond_eps_emag)
        # uncond_eps = uncond_eps + emag_scale * (uncond_eps - uncond_eps_emag)

        
        #eps_final = uncond_eps + cfg_scale * (cond_eps - uncond_eps) + emag_scale * (cond_eps - cond_eps_emag)
        eps_cond = cond_eps_emag + emag_scale * (cond_eps - cond_eps_emag)
        eps_final = uncond_eps + cfg_scale * (eps_cond - uncond_eps) 
        #eps_final = uncond_eps + cfg_scale * cond_eps - emag_scale * (cond_eps_emag) # S2 style of guidance

        if negative_sample:
            print("Warning: Negative sampling is being generated with EMAG")
            return torch.cat([eps_emag, rest], dim=1)

        eps_final = torch.cat([eps_final, eps_final], dim=0)
        return torch.cat([eps_final, rest], dim=1)

######## EMAG Helper functions ##########################################################



############ APG Helper functions ##########################################################

    # Implemntation based on the orignal paper provided code Link:- https://arxiv.org/pdf/2410.02416


    def forward_with_apg(self, x, t, y, apg_scale,momentum_buffer=None,diffusion=None):
        """
            Forward pass of DiT, but also batches the APG forward pass.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]

        # Since its CFG we have duplicated the input and the APG paper mentions to move the model out to image sapce 
        # Upcasting to image space
        # ###########################################################
        # image_part,epsilon = model_out[:, :4], model_out[:, 4:]
        # image_part = vae.decode(image_part / 0.18215).sample
        # model_out = image_part
        # ###########################################################
         
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        
        # Bases on authors inttruction the calcation will bein pixel space so we convert the output to pixel space
        assert diffusion is not None, "diffusion helper is required for APG"

         # Use the half-batch inputs and times for x0 conversion
        t_half = t[: len(half)].long()
        # Mapping to DDIM timestep
        map_tensor = torch.tensor(diffusion.timestep_map, device=t_half.device,dtype=t_half.dtype)
        inv_map = torch.full((diffusion.original_num_steps,), -1, device=t_half.device,dtype=t_half.dtype)
        inv_map[map_tensor] = torch.arange(len(map_tensor), device=t_half.device,dtype=t_half.dtype)
        t_half = inv_map[t_half]


        x_t_half = half[:, :cond_eps.shape[1], :, :]  # match channel count (3)
        
        cond_eps   = diffusion._predict_xstart_from_eps(x_t_half, t_half, cond_eps)
        uncond_eps = diffusion._predict_xstart_from_eps(x_t_half, t_half, uncond_eps)





        diff = cond_eps - uncond_eps
        # APG implementation

        norm_threshold: float = 5 # From the paper recommended value for DIT XL/2 model
        eta: float = 0.0 # From the paper recommended value for DIT XL/2 model
        
        if momentum_buffer is not None:

            momentum_buffer.update(diff)
            diff = momentum_buffer.running_average

        if norm_threshold > 0:

            ones = torch.ones_like(diff)
            diff_norm = diff.norm(p=2,dim=[-1,-2,-3],keepdim=True)
            scale_factor = torch.minimum(ones,norm_threshold / diff_norm)
            diff = diff * scale_factor
        
        diff_parallel, diff_orthogonal = project(diff, cond_eps)
        normalized_update = diff_orthogonal + eta * diff_parallel
        eps_final = cond_eps + (apg_scale - 1) * normalized_update

        # Converting back to latent space
        eps_final = diffusion._predict_eps_from_xstart(x_t_half, t_half, eps_final)
        ######################3

        eps = torch.cat([eps_final, eps_final], dim=0)
        # Convereting back to latent space
        eps = torch.cat([eps, rest], dim=1)
        
        # eps = vae.encode(eps).latent_dist.sample()*0.18215 # Scaling back to latent space
        # eps = torch.cat([eps, epsilon], dim=1)
        return eps




    ########### APG Helper functions ##########################################################

    ########### CADS Helper functions ##########################################################

    def forward_cads_with_noisy_condtion(self, x, t, y, gamma, cads_s, cads_psi):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """


        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        # CADS Noisy condtion
        y = add_noise(y, gamma, cads_s, cads_psi,rescale=True) # (N, D) Noisy condtion
        c = t + y                                # (N, D) Base condtion
        ###########################################################################
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x


    def forward_with_cads(self, x, t, y, cads_t1, cads_t2, cads_s, cads_psi, cfg_scale,sample_steps=1000):

        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance with CADS.
        """

        # Adding noise to the conditon based on CADS Schedule.

        cads_t1 = Timestetp_convertor(torch.tensor(cads_t1), sample_steps)
        cads_t2 = Timestetp_convertor(torch.tensor(cads_t2), sample_steps)
        gamma = linear_schedule(t[0].item(), cads_t1, cads_t2)

        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward_cads_with_noisy_condtion(combined, t, y, gamma, cads_s, cads_psi)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    ########### CADS Helper functions ##########################################################


    ########### S2 Helper functions ##########################################################


    def forward_with_block_dropout(self, x, t, y, blocks_to_dropout):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """


        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D) Base condtion
        
        for i, block in enumerate(self.blocks):
            if i in blocks_to_dropout:
                continue
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x


    def forward_with_s2(self, x, t, y, s2_scale, s2_no_block_dropout, cfg_scale,sample_steps=1000,negative_sample=False):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance with S2.
        """

        # Adding noise to the conditon based on S2 Schedule.

        # Randomly select s2_no_blcoks to dropout
        blocks_to_dropout = random.sample(range(len(self.blocks)), s2_no_block_dropout)
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)

        # S2 Guidance 
        ########################################################################################################
        pretubed_model_out = self.forward_with_block_dropout(combined, t, y, blocks_to_dropout)
        pretubed_eps, pretubed_rest = pretubed_model_out[:, :3], pretubed_model_out[:, 3:]
        pretubed_cond_eps, pretubed_uncond_eps = torch.split(pretubed_eps, len(pretubed_eps) // 2, dim=0)
        ########################################################################################################

        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps) - s2_scale * (pretubed_cond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)

        if negative_sample:
            print("Warning: Negative sampling is being generated with S2")
            return torch.cat([pretubed_eps, rest], dim=1)
        return torch.cat([eps, rest], dim=1)
    ########### S2 Helper functions ##########################################################


    ########### ERG Helper functions ##########################################################


    def enable_erg(self, layers=[12,13,14,15],erg_alpha=1.0, erg_gamma=1.0, erg_tau_i=0.01, erg_K=1):
        for blk in self.blocks:
            if blk.attn.layer_idx in layers:
                blk.attn.enable_erg = True
                blk.attn.erg_alpha = erg_alpha
                blk.attn.erg_gamma = erg_gamma
                blk.attn.erg_tau_i = erg_tau_i
                blk.attn.erg_K = erg_K

    def disable_erg(self):
        for blk in self.blocks:
            blk.attn.enable_erg = False
            blk.attn.erg_alpha = None
            blk.attn.erg_gamma = None
            blk.attn.erg_tau_i = None
            blk.attn.erg_K = None





    def forward_with_erg(self, x, t, y, cfg_scale = 1.5,erg_alpha=1.0, erg_gamma=1.0, erg_tau_i=0.01, erg_kappa=0.2, erg_layers=[12,13,14,15], erg_K=1,negative_sample=False,uncond=False):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance with ERG.
        """
        if uncond:

            erg_enabled = False
            erg_kickoff_threshold  = Timestetp_convertor(torch.tensor(erg_kappa), 1000)
            
            half = x[: len(x) // 2]
            combined = torch.cat([half, half], dim=0)

            if erg_kickoff_threshold > t[0].item():
                self.enable_erg(layers=erg_layers,erg_alpha=erg_alpha, erg_gamma=erg_gamma, erg_tau_i=erg_tau_i, erg_K=erg_K)
                erg_enabled = True
                

                model_out = self.forward(combined, t, y)

            else:
                model_out = self.forward(combined, t, y)
            
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
           
            
            if erg_enabled:
                self.disable_erg()
            uncond_eps = torch.cat([uncond_eps, uncond_eps], dim=0)
            return torch.cat([uncond_eps, rest], dim=1)
        else:
            
            half = x[: len(x) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.forward(combined, t, y)
            erg_enabled = False
            erg_kickoff_threshold  = Timestetp_convertor(torch.tensor(erg_kappa), 1000)

            
            # For exact reproducibility reasons, we apply classifier-free guidance on only
            # three channels by default. The standard approach to cfg applies it to all channels.
            # This can be done by uncommenting the following line and commenting-out the line following that.
            # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            if erg_kickoff_threshold > t[0].item():
                # Enabling ERG after the kickoff threshold
                self.enable_erg(layers=erg_layers,erg_alpha=erg_alpha, erg_gamma=erg_gamma, erg_tau_i=erg_tau_i, erg_K=erg_K)
                erg_enabled = True
                model_out = self.forward(combined, t, y)
                eps_erg, _ = model_out[:, :3], model_out[:, 3:]
                erg_cond_eps, erg_uncond_eps = torch.split(eps_erg, len(eps_erg) // 2, dim=0)
                uncond_eps = erg_uncond_eps


                if negative_sample:
                    print("Warning: Negative sampling is being generated with ERG")
                    return torch.cat([eps_erg, rest], dim=1)

           
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)

            if erg_enabled:
                self.disable_erg()

            return torch.cat([eps, rest], dim=1)



    ########### ERG Helper functions ##########################################################


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
