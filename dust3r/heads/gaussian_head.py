# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# linear head implementation for DUST3R
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from dust3r.heads.postprocess import postprocess
from dust3r.utils.gaussian_render import strip_symmetric, inverse_sigmoid, build_scaling_rotation, get_radius, get_rect


class LinearGaussian (nn.Module):
    """ 
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, net, downsample_ratio=2):
        super().__init__()
        
        self.patch_size = net.patch_embed.patch_size[0]
        self.downsample_ratio = downsample_ratio
        self.resized_patch_size = self.patch_size // self.downsample_ratio

        self.xyz = nn.Linear(net.dec_embed_dim, 3 * self.resized_patch_size**2) # pos, quat, opacity
        # self.scaling = nn.Linear(net.dec_embed_dim, 3 * self.resized_patch_size**2)
        # self.rotation = nn.Linear(net.dec_embed_dim, 4 * self.resized_patch_size**2)
        self.opacity = nn.Linear(net.dec_embed_dim, 1 * self.resized_patch_size**2)

    def setup(self, croconet):
        pass

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        # predict Gaussian features
        xyz = self.xyz(tokens)
        device = tokens.device
        # scaling = self.scaling(tokens)
        # rotation = self.rotation(tokens)
        opacity = self.opacity(tokens)

        # normalize xyz to [-1, 1]
        # xyz = xyz.view(B, S, 3, -1).transpose(1, 2)  # B,3,S,16x16
        xyz = F.tanh(xyz).transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        xyz = F.pixel_shuffle(xyz, self.resized_patch_size) # upscale to (H/4, W/4)

        scaling = torch.ones_like(xyz) * 0.01
        # scaling = scaling.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        # scaling = F.pixel_shuffle(scaling, self.resized_patch_size)

        # normalize the rotation (quaternion) to unit quaternion
        # rotation = F.normalize(rotation, p=2, dim=-1).transpose(-1, -2).view(B, -1,H // self.patch_size, W // self.patch_size)
        # rotation = F.pixel_shuffle(rotation, self.resized_patch_size)
        rotation = torch.zeros((xyz.shape[0], 4, xyz.shape[2], xyz.shape[3]), device=device)
        rotation[:, 0] = 1

        # opacity = torch.ones((xyz.shape[0], 1, xyz.shape[2], xyz.shape[3]), device=device)

        opacity = opacity.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        opacity = F.pixel_shuffle(opacity, self.resized_patch_size)

        results = {'xyz':xyz, 'scaling':scaling, 'rotation':rotation, 'opacity':opacity}
        return results