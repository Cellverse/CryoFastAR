import pdb
import torch
import torch.nn as nn
import math
from einops import reduce
import torch.nn.functional as F

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def homogeneous(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def build_rotation(r):
    
    B, N = r.shape[0], r.shape[1]

    q = F.normalize(r, p=2, dim=-1)

    R = torch.zeros((B, N, 3, 3), device='cuda')

    r = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    R[..., 0, 0] = 1 - 2 * (y*y + z*z)
    R[..., 0, 1] = 2 * (x*y - r*z)
    R[..., 0, 2] = 2 * (x*z + r*y)
    R[..., 1, 0] = 2 * (x*y + r*z)
    R[..., 1, 1] = 1 - 2 * (x*x + z*z)
    R[..., 1, 2] = 2 * (y*z - r*x)
    R[..., 2, 0] = 2 * (x*z - r*y)
    R[..., 2, 1] = 2 * (y*z + r*x)
    R[..., 2, 2] = 1 - 2 * (x*x + y*y)
    return R



def build_scaling_rotation(s, r):

    assert len(s.shape) == 3, "s should be [B, N, 3]"
    B, N, _ = s.shape

    L = torch.zeros((B, N, 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[...,0,0] = s[...,0]
    L[...,1,1] = s[...,1]
    L[...,2,2] = s[...,2]

    L = R @ L

    return L


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)



def build_covariance_3d(s, r):
    L = build_scaling_rotation(s, r)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance
    # symm = strip_symmetric(actual_covariance)
    # return symm



def build_covariance_2d(
    mean3d, cov3d, viewmatrix, fov_x, fov_y, focal_x, focal_y
):
    # The following models the steps outlined by equations 29
	# and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	# Additionally considers aspect / scaling of viewport.
	# Transposes used to account for row-/column-major conventions.
    tan_fovx = math.tan(fov_x * 0.5)
    tan_fovy = math.tan(fov_y * 0.5)
    t = (mean3d @ viewmatrix[:3,:3]) + viewmatrix[-1:,:3]

    # truncate the influences of gaussians far outside the frustum.
    tx = (t[..., 0] / t[..., 2]).clip(min=-tan_fovx*1.3, max=tan_fovx*1.3) * t[..., 2]
    ty = (t[..., 1] / t[..., 2]).clip(min=-tan_fovy*1.3, max=tan_fovy*1.3) * t[..., 2]
    tz = t[..., 2]

    # Eq.29 locally affine transform 
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the sceen space
    J = torch.zeros(mean3d.shape[0], 3, 3).to(mean3d)
    J[..., 0, 0] = 1 / tz * focal_x
    J[..., 0, 2] = -tx / (tz * tz) * focal_x
    J[..., 1, 1] = 1 / tz * focal_y
    J[..., 1, 2] = -ty / (tz * tz) * focal_y
    # J[..., 2, 0] = tx / t.norm(dim=-1) # discard
    # J[..., 2, 1] = ty / t.norm(dim=-1) # discard
    # J[..., 2, 2] = tz / t.norm(dim=-1) # discard
    W = viewmatrix[:3,:3].T # transpose to correct viewmatrix
    cov2d = J @ W @ cov3d @ W.T @ J.permute(0,2,1)
    
    # add low pass filter here according to E.q. 32
    filter = torch.eye(2,2).to(cov2d) * 0.3
    return cov2d[:, :2, :2] + filter[None]


def projection_ndc(points, viewmatrix, projmatrix):
    points_o = homogeneous(points) # object space
    points_h = points_o @ viewmatrix @ projmatrix # screen space # RHS
    p_w = 1.0 / (points_h[..., -1:] + 0.000001)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix
    in_mask = p_view[..., 2] >= 0.2
    return p_proj, p_view, in_mask


@torch.no_grad()
def get_radius(cov2d):
    det = cov2d[..., 0, 0] * cov2d[..., 1, 1] - cov2d[..., 0, 1] * cov2d[..., 1, 0]
    mid = 0.5 * (cov2d[..., 0, 0] + cov2d[..., 1, 1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[...,None])
    rect_max = (pix_coord + radii[...,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max



def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result



def _cov2d(scaling, rotation, rot, H, W, translation=None):
    # 1. calculate cov3d based on rotation and scaling
    L = build_scaling_rotation(scaling, rotation)
    cov3d = L @ L.transpose(-1, -2) # (3, 3)
    # 2. calculate cov2d based on cov3d and inputs
    S = torch.zeros(scaling.shape[0], scaling.shape[1], 3, 3).to(rot.device)
    S[..., 0, 0] = W
    S[..., 1, 1] = H

    rot = rot.unsqueeze(1)

    cov2d = S @ rot @ cov3d @ rot.permute(0, 1, 3, 2) @ S.permute(0, 1, 3, 2)

    # add low pass filter here according to E.q. 32
    # filter = torch.eye(2,2).to(cov2d) * 0.3
    return cov2d[..., :2, :2]

def project_rot(rot, xyz, scaling, rotation, opacity, res=128):
    # rot: [B, 3, 3]
    # xyz: [B, 3, H, W]
    # scaling: [B, 3, H, W]
    # rotation: [B, 4, H, W]
    # opactiy: [B, 1, H, W]
    # res: int

    assert len(rot.shape) == 3, "rot should be [B, 3, 3]"
    B = rot.shape[0]
    device = rot.device
    
    # flatten input parameters
    xyz = xyz.permute(0, 2, 3, 1).reshape(B, -1, 3) # [B, H*W, 3]
    scaling = scaling.permute(0, 2, 3, 1).reshape(B, -1, 3) # [B, H*W, 1]
    rotation = rotation.permute(0, 2, 3, 1).reshape(B, -1, 4) # [B, H*W, 4]
    opacity = opacity.permute(0, 2, 3, 1).reshape(B, -1, 1) # [B, H*W, 1]

    xy_mean = ((xyz @ rot.transpose(2,1))[..., :2] + 1) * res / 2 # scaling up to [0, res], (N, 2)
    
    x = torch.linspace(0, res - 1, res, device=device)
    y = torch.linspace(0, res - 1, res, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    xy = torch.stack([xx, yy], dim=-1) # (res, res, 2)

    # 2d covariance of the gaussian
    cov2d = _cov2d(scaling, rotation, rot, res, res)
    radii = get_radius(cov2d)
    rect_min, rect_max = get_rect(xy_mean, radii, width=res, height=res)

    render_alpha = torch.zeros(B, 1, res, res).to(device)

    TILE_SIZE = 16
    for b in range(B):
        for h in range(0, res, TILE_SIZE):
            for w in range(0, res, TILE_SIZE):
                # check if the rectangle penetrate the tile
                over_tl = rect_min[b, ..., 0].clip(min=w), rect_min[b, ..., 1].clip(min=h)
                over_br = rect_max[b, ..., 0].clip(max=w+TILE_SIZE-1), rect_max[b, ..., 1].clip(max=h+TILE_SIZE-1)

                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 
                
                if in_mask.sum() == 0:
                    continue

                P = in_mask.sum()
                tile_coord = xy[h:h+TILE_SIZE, w:w+TILE_SIZE].flatten(0,-2)

                selected_means2D = xy_mean[b][in_mask]
                selected_cov2d = cov2d[b][in_mask] # P 2 2
                selected_conic = selected_cov2d.inverse() # inverse of variance

                selected_opacity = opacity[b][in_mask]
                dx = (tile_coord[:,None,:] - selected_means2D[None,:]) # B P 2

                gauss_weight = torch.exp(-0.5 * (
                    dx[:, :, 0]**2 * selected_conic[:, 0, 0] 
                    + dx[:, :, 1]**2 * selected_conic[:, 1, 1]
                    + dx[:,:,0]*dx[:,:,1] * selected_conic[:, 0, 1]
                    + dx[:,:,0]*dx[:,:,1] * selected_conic[:, 1, 0]))

                alpha = gauss_weight[..., None] * selected_opacity[None] # B P 1
                acc_alpha = alpha.sum(dim=1)

                render_alpha[b, 0, h:h+TILE_SIZE, w:w+TILE_SIZE] = acc_alpha.reshape(TILE_SIZE, TILE_SIZE)
    
    return render_alpha

import torch.autograd.profiler as profiler
USE_PROFILE = False
import contextlib

class GaussRenderer(nn.Module):
    """
    A gaussian splatting renderer

    >>> gaussModel = GaussModel.create_from_pcd(pts)
    >>> gaussRender = GaussRenderer()
    >>> out = gaussRender(pc=gaussModel, camera=camera)
    """

    def __init__(self, active_sh_degree=3, white_bkgd=True, **kwargs):
        super(GaussRenderer, self).__init__()
        self.active_sh_degree = active_sh_degree
        self.debug = False
        self.white_bkgd = white_bkgd
        self.pix_coord = torch.stack(torch.meshgrid(torch.arange(256), torch.arange(256), indexing='xy'), dim=-1).to('cuda')
        
    
    def build_color(self, means3D, shs, camera):
        rays_o = camera.camera_center
        rays_d = means3D - rays_o
        color = eval_sh(self.active_sh_degree, shs.permute(0,2,1), rays_d)
        color = (color + 0.5).clip(min=0.0)
        return color
    
    def render(self, camera, means2D, cov2d, color, opacity, depths):
        radii = get_radius(cov2d)
        rect = get_rect(means2D, radii, width=camera.image_width, height=camera.image_height)
        
        self.render_color = torch.ones(*self.pix_coord.shape[:2], 3).to('cuda')
        self.render_depth = torch.zeros(*self.pix_coord.shape[:2], 1).to('cuda')
        self.render_alpha = torch.zeros(*self.pix_coord.shape[:2], 1).to('cuda')

        TILE_SIZE = 64
        for h in range(0, camera.image_height, TILE_SIZE):
            for w in range(0, camera.image_width, TILE_SIZE):
                # check if the rectangle penetrate the tile
                over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
                over_br = rect[1][..., 0].clip(max=w+TILE_SIZE-1), rect[1][..., 1].clip(max=h+TILE_SIZE-1)
                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 
                
                if not in_mask.sum() > 0:
                    continue

                P = in_mask.sum()
                tile_coord = self.pix_coord[h:h+TILE_SIZE, w:w+TILE_SIZE].flatten(0,-2)
                sorted_depths, index = torch.sort(depths[in_mask])
                sorted_means2D = means2D[in_mask][index]
                sorted_cov2d = cov2d[in_mask][index] # P 2 2
                sorted_conic = sorted_cov2d.inverse() # inverse of variance
                sorted_opacity = opacity[in_mask][index]
                sorted_color = color[in_mask][index]
                dx = (tile_coord[:,None,:] - sorted_means2D[None,:]) # B P 2
                
                gauss_weight = torch.exp(-0.5 * (
                    dx[:, :, 0]**2 * sorted_conic[:, 0, 0] 
                    + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]
                    + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1]
                    + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0]))
                
                alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99) # B P 1
                T = torch.cat([torch.ones_like(alpha[:,:1]), 1-alpha[:,:-1]], dim=1).cumprod(dim=1)
                acc_alpha = (alpha * T).sum(dim=1)
                tile_color = (T * alpha * sorted_color[None]).sum(dim=1) + (1-acc_alpha) * (1 if self.white_bkgd else 0)
                tile_depth = ((T * alpha) * sorted_depths[None,:,None]).sum(dim=1)
                self.render_color[h:h+TILE_SIZE, w:w+TILE_SIZE] = tile_color.reshape(TILE_SIZE, TILE_SIZE, -1)
                self.render_depth[h:h+TILE_SIZE, w:w+TILE_SIZE] = tile_depth.reshape(TILE_SIZE, TILE_SIZE, -1)
                self.render_alpha[h:h+TILE_SIZE, w:w+TILE_SIZE] = acc_alpha.reshape(TILE_SIZE, TILE_SIZE, -1)

        return {
            "render": self.render_color,
            "depth": self.render_depth,
            "alpha": self.render_alpha,
            "visiility_filter": radii > 0,
            "radii": radii
        }


    def forward(self, camera, gs, **kwargs):
        means3D = gs.xyz
        opacity = gs.opacity
        scales = gs.scaling
        rotations = gs.rotation
        # shs = pc.features
        
        if USE_PROFILE:
            prof = profiler.record_function
        else:
            prof = contextlib.nullcontext
            
        with prof("projection"):
            mean_ndc, mean_view, in_mask = projection_ndc(means3D, 
                    viewmatrix=camera.world_view_transform, 
                    projmatrix=camera.projection_matrix)
            mean_ndc = mean_ndc[in_mask]
            mean_view = mean_view[in_mask]
            depths = mean_view[:,2]
        
        with prof("build color"):
            color = self.build_color(means3D=means3D, shs=shs, camera=camera)
        
        with prof("build cov3d"):
            cov3d = build_covariance_3d(scales, rotations)
            
        with prof("build cov2d"):
            cov2d = build_covariance_2d(
                mean3d=means3D, 
                cov3d=cov3d, 
                viewmatrix=camera.world_view_transform,
                fov_x=camera.FoVx, 
                fov_y=camera.FoVy, 
                focal_x=camera.focal_x, 
                focal_y=camera.focal_y)

            mean_coord_x = ((mean_ndc[..., 0] + 1) * camera.image_width - 1.0) * 0.5
            mean_coord_y = ((mean_ndc[..., 1] + 1) * camera.image_height - 1.0) * 0.5
            means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)
        
        with prof("render"):
            rets = self.render(
                camera = camera, 
                means2D=means2D,
                cov2d=cov2d,
                color=color,
                opacity=opacity, 
                depths=depths,
            )

        return rets