# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# linear head implementation for DUST3R
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from dust3r.heads.postprocess import postprocess, reg_dense_conf

# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class LinearFourierPts3d (nn.Module):
    """ 
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, net, out_chans=256, has_conf=False, has_neck=False, has_sph=False):
        super().__init__()
        self.patch_size = net.patch_embed.patch_size[0]
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.has_conf = has_conf
        self.out_chans = out_chans
        self.has_neck = has_neck

        if has_neck:
        # TODO: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/image_encoder.py#L27C9-L27C18
        # out_chans: 256
            self.neck = nn.Sequential(
                nn.Conv2d(
                    net.dec_embed_dim,
                    self.out_chans,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(self.out_chans),
                nn.Conv2d(
                    self.out_chans,
                    self.out_chans,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(self.out_chans),
            )
            self.proj = nn.Linear(self.out_chans, (3 + 1)*self.patch_size**2)
            self.proj_fvalue = nn.Linear(self.out_chans, 1 * self.patch_size**2)

        else:
            self.proj = nn.Linear(net.dec_embed_dim, (3 + 1)*self.patch_size**2)
            self.proj_fvalue = nn.Linear(net.dec_embed_dim, 1 * self.patch_size**2)


    def setup(self, croconet):
        pass

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape
        # TODO
        if self.has_neck:
            neck_tokens = tokens.reshape(B, H//self.patch_size, W//self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
            neck_tokens = self.neck(neck_tokens).permute(0, 2, 3, 1)
            neck_tokens = neck_tokens.reshape(B, -1, self.out_chans)
        else:
            neck_tokens = tokens
        
        # extract 3D points
        feat = self.proj(neck_tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W
        fmap = feat.permute(0, 2, 3, 1) # B, H, W, 3

        pts3d = fmap[:, :, :, 0:3]
        # pts3d = pts3d.clip(min=0, max=1.415) / 2

        # extract 1D fourier value
        value = self.proj_fvalue(neck_tokens)
        value = value.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        value = F.pixel_shuffle(value, self.patch_size)  # B,1,H,W
        # fourier_value = fourier_value.permute(0, 2, 3, 1) # B, H, W, 1
        
        res = dict(pts3d=pts3d, fval=value)
        
        if self.has_conf:
            res['conf'] = reg_dense_conf(fmap[..., 3], self.conf_mode)

        return res

        # permute + norm depth
        # return postprocess(feat, self.depth_mode, self.conf_mode)
