# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# head factory
# --------------------------------------------------------
from .linear_head import LinearPts3d
from .dpt_head import create_dpt_head, create_dpt_head_fourier
from .gaussian_head import LinearGaussian
from .fourier_head import LinearFourierPts3d

def head_factory(head_type, output_mode, net, has_conf=False, has_neck=False, has_sph=False):
    """" build a prediction head for the decoder 
    """
    if head_type == 'linear' and output_mode == 'pts3d':
        return LinearPts3d(net, has_conf)
    elif head_type == 'dpt' and output_mode == 'pts3d':
        return create_dpt_head(net, has_conf=has_conf)
    elif head_type == 'gaussian':
        return LinearGaussian(net)
    elif head_type == 'fourier':
        return LinearFourierPts3d(net, has_conf=has_conf, has_neck=has_neck)
    elif head_type == 'fourier_dpt':
        return create_dpt_head_fourier(net)
    else:
        raise NotImplementedError(f"unexpected {head_type=} and {output_mode=}")
