""" Layer/Module Helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


# MODIFIED BY JONATHAN CUI 01/29/2023

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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
            drop_ids = torch.rand(labels.shape[0]) < self.dropout_prob
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
        approx_gelu = lambda: nn.GELU()
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
    

class FMLPBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, num_channels, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        approx_gelu = lambda: nn.GELU()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.channel_mixing_mlp = Mlp(in_features=num_channels, hidden_features=num_heads * num_channels, act_layer=approx_gelu)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.fourier_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.fourier_cmm = Mlp(in_features=num_channels * 2, hidden_features=num_heads * num_channels * 2, act_layer=approx_gelu)
        self.fourier_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.fourier_tmm = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu)
        self.fourier_final = nn.Sequential(
            nn.Linear(hidden_size, 1), 
            nn.Flatten(),
            nn.Linear(num_channels * 2, hidden_size)
        )

    def forward(self, x):
        c = torch.fft.fft(x)
        c = torch.cat([c.real, c.imag], dim=1)
        c = c + self.fourier_cmm(self.fourier_norm1(c).permute(0, 2, 1)).permute(0, 2, 1)
        c = c + self.fourier_tmm(self.fourier_norm2(c))
        c = self.fourier_final(c)
        shift_cmm, scale_cmm, gate_cmm, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        x = x + gate_cmm.unsqueeze(1) * \
            self.channel_mixing_mlp(
                modulate(self.norm1(x), shift_cmm, scale_cmm).permute(0, 2, 1)
            ).permute(0, 2, 1)
        x = x + gate_mlp.unsqueeze(1) * \
            self.mlp(
                modulate(self.norm2(x), shift_mlp, scale_mlp)
            )
        return x


# class FinalLayer(nn.Module):
#     """
#     The final layer of DiT.
#     """
#     def __init__(self, hidden_size, patch_size, out_channels):
#         super().__init__()
#         self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 2 * hidden_size, bias=True)
#         )

#     def forward(self, x, c):
#         shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
#         x = modulate(self.norm_final(x), shift, scale)
#         x = self.linear(x)
#         return x


class FinalLayer(nn.Module):
    """
    The final layer of FMLP.
    """
    def __init__(self, hidden_size, num_channels, num_classes):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_classes, bias=True)
        self.channel_elim = nn.Linear(num_channels, 1)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        x = self.channel_elim(x.permute(0, 2, 1)).squeeze(-1)
        return x


class FMLP(nn.Module):
    """
    FMLP with DiT-like backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=3,
        hidden_size=1024,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=10
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = num_classes
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.num_channels = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_channels, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            FMLPBlock(hidden_size, num_heads, self.num_channels, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.num_channels, num_classes)
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

        # Zero-out adaLN modulation layers in FMLP blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x):
        """
        Forward pass of FMLP.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        for block in self.blocks:
            x = block(x)                         # (N, T, D)
        x = self.final_layer(x)                  # (N, T, patch_size ** 2 * out_channels)
        return x


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

def FMLPXL_2(**kwargs):
    return FMLP(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def FMLPXL_4(**kwargs):
    return FMLP(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def FMLPXL_8(**kwargs):
    return FMLP(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def FMLPL_2(**kwargs):
    return FMLP(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def FMLPL_4(**kwargs):
    return FMLP(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def FMLPL_8(**kwargs):
    return FMLP(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def FMLPB_2(**kwargs):
    return FMLP(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def FMLPB_4(**kwargs):
    return FMLP(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def FMLPB_8(**kwargs):
    return FMLP(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def FMLPS_2(**kwargs):
    return FMLP(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def FMLPS_4(**kwargs):
    return FMLP(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def FMLPS_8(**kwargs):
    return FMLP(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

if __name__ == "__main__":
    model = FMLPS_4()
    x = torch.randn(4, 3, 32, 32)
    import time
    t0 = time.time()
    y = model(x)
    print(f"Forward pass: {time.time() - t0:.2f} sec(s).")
    print(f"#params: {sum([w.numel() for w in model.parameters()]):,}")
