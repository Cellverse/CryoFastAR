# https://github.com/hbb1/torch-splatting/blob/main/gaussian_splatting/gauss_model.py
# @ Author: Binbin Huang
import torch
import torch.nn  as nn
import numpy as np
import math
# from simple_knn._C import distCUDA2
from .gaussian_render import strip_symmetric, inverse_sigmoid, build_scaling_rotation, get_radius, get_rect
# from gaussian_splatting.utils.sh_utils import RGB2SH

C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0


class GaussianModel(nn.Module):
    """
    A Gaussian Model

    * Attributes
    _xyz: locations of gaussians
    _feature_dc: DC term of features
    _feature_rest: rest features
    _rotatoin: rotation of gaussians
    _scaling: scaling of gaussians
    _opacity: opacity of gaussians

    >>> gaussModel = GaussModel.create_from_pcd(pts)
    >>> gaussRender = GaussRenderer()
    >>> out = gaussRender(pc=gaussModel, camera=camera)
    """
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
    
    def __init__(self, sh_degree : int=3, debug=False):
        super(GaussianModel, self).__init__()
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.setup_functions()
        self.debug = debug

    def create_from_volume(self, volume, threshold=0.1):
        """
            create the guassian model from a volumetric tensor (N, H, W)
        """
        device = volume.device
        volume = volume.squeeze() # (N, N, N)
        N, H, W = volume.shape
        indices = torch.nonzero(volume > threshold)
        points = indices.float()
        points = points.to(device)

        # normalize points into [-1, 1]^3
        points = 2 * points / torch.tensor([N, H, W], device=device).float() - 1
        print("totally {} points".format(points.shape[0]))
        colors = torch.ones_like(points)
        # colors = volume[indices[:, 0], indices[:, 1], indices[:, 2]].unsqueeze(1)
        # colors = torch.cat((colors, torch.zeros_like(colors), torch.zeros_like(colors)), dim=1)
        colors = RGB2SH(colors)
        features = torch.zeros((colors.shape[0], 3, (self.max_sh_degree + 1) ** 2), device=device).float()
        features[:, :3, 0 ] = colors
        features[:, 3:, 1:] = 0.0

        # dist2 = torch.clamp_min(distCUDA2(points), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((points.shape[0], 4), device=device)
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((points.shape[0], 1), dtype=torch.float))

        if self.debug:
            # easy for visualization
            colors = np.zeros_like(colors)
            opacities = inverse_sigmoid(0.9 * torch.ones((points.shape[0], 1), dtype=torch.float))

        self._xyz = points.requires_grad_(True)
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True)) # Quaternion
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        return self
    
    
    def cov2d(self, rot, H, W, translation=None):
        # 1. calculate cov3d based on rotation and scaling
        L = build_scaling_rotation(self._scaling, self._rotation)
        cov3d = L @ L.transpose(1, 2) # (3, 3)
        # 2. calculate cov2d based on cov3d and inputs
        S = torch.zeros(self._scaling.shape[0], 3, 3).to(rot.device)
        S[:, 0, 0] = W
        S[:, 1, 1] = H
        cov2d = S @ rot @ cov3d @ rot.T @ S.permute(0, 2, 1)

        # add low pass filter here according to E.q. 32
        # filter = torch.eye(2,2).to(cov2d) * 0.3
        return cov2d[:, :2, :2]
    

    def project_rot(self, rot, res):
        xy_mean = self._xyz[..., :2] * res / 2 # scaling up to [-res/2, res/2], (N, 2)
        device = xy_mean.device
        x = torch.linspace(-res / 2, res / 2 - 1, res, device=device) + 1/2
        y = torch.linspace(-res / 2, res / 2 - 1, res, device=device) + 1/2
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        xy = torch.stack([xx, yy], dim=-1).reshape(-1, 2) # (res^2, 2)

        cov2d = self.cov2d(rot, res, res)
        radii = get_radius(cov2d)
        rect = get_rect(xy_mean, radii, width=res, height=res)

        render_alpha = torch.zeros(res, res, 1).to('cuda')

        TILE_SIZE = 16
        for h in range(0, res, TILE_SIZE):
            for w in range(0, res, TILE_SIZE):
                # check if the rectangle penetrate the tile
                over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
                over_br = rect[1][..., 0].clip(max=w+TILE_SIZE-1), rect[1][..., 1].clip(max=h+TILE_SIZE-1)
                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 
                
                if in_mask.sum() == 0:
                    continue

                P = in_mask.sum()
                tile_coord = xy[h:h+TILE_SIZE, w:w+TILE_SIZE].flatten(0,-2)

                selected_means2D = xy_mean[in_mask]
                selected_cov2d = cov2d[in_mask] # P 2 2
                selected_conic = selected_cov2d.inverse() # inverse of variance

                selected_opacity = self._opacity[in_mask]
                dx = (tile_coord[:,None,:] - selected_means2D[None,:]) # B P 2

                gauss_weight = torch.exp(-0.5 * (
                    dx[:, :, 0]**2 * selected_conic[:, 0, 0] 
                    + dx[:, :, 1]**2 * selected_conic[:, 1, 1]
                    + dx[:,:,0]*dx[:,:,1] * selected_conic[:, 0, 1]
                    + dx[:,:,0]*dx[:,:,1] * selected_conic[:, 1, 0]))
                
                alpha = gauss_weight[..., None] * selected_opacity[None] # B P 1
                acc_alpha = alpha.sum(dim=1)

                render_alpha[h:h+TILE_SIZE, w:w+TILE_SIZE] = acc_alpha.reshape(TILE_SIZE, TILE_SIZE, -1)
        
        return render_alpha


    def project(self, res, a = 1, sigma = 10.0):
        "project along z-axis"
        xy_mean = self._xyz[..., :2] * res / 2 # scaling up to [-res/2, res/2], (N, 2)
        device = xy_mean.device
        x = torch.linspace(-res / 2, res / 2 - 1, res, device=device) + 1/2
        y = torch.linspace(-res / 2, res / 2 - 1, res, device=device) + 1/2
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2) # (res^2, 2)
        l2_distance = torch.cdist(grid, xy_mean) # (res^2, N)
        image = torch.sum(a / (2*torch.pi) * (sigma ** -0.5) * torch.exp(-0.5 * l2_distance ** 2 / (sigma ** 2)), dim=-1)
        image = image.reshape(res, res)

        return image

    def create_from_properties(self, xyz, scaling, rotation, opacity):
        """
            create the guassian model from properties
        """
        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._scaling = nn.Parameter(scaling.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))
        self._opacity = nn.Parameter(opacity.requires_grad_(True))
        #  self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        
        # return self

    # def create_from_pcd(self, pcd:PointCloud):
    #     """
    #         create the guassian model from a color point cloud
    #     """
    #     points = pcd.coords
    #     colors = pcd.select_channels(['R', 'G', 'B']) / 255.

    #     fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
    #     fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float().cuda())

    #     print("Number of points at initialisation : ", fused_point_cloud.shape[0])

    #     features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
    #     features[:, :3, 0 ] = fused_color
    #     features[:, 3:, 1:] = 0.0

    #     dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(points)).float().cuda()), 0.0000001)
    #     scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    #     rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    #     rots[:, 0] = 1
    #     opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

    #     if self.debug:
    #         # easy for visualization
    #         colors = np.zeros_like(colors)
    #         opacities = inverse_sigmoid(0.9 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

    #     self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    #     self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
    #     self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
    #     self._scaling = nn.Parameter(scales.requires_grad_(True))
    #     self._rotation = nn.Parameter(rots.requires_grad_(True))
    #     self._opacity = nn.Parameter(opacities.requires_grad_(True))
    #     self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
    #     return self

    @property
    def scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def xyz(self):
        return self._xyz
    
    # @property
    # def features(self):
    #     features_dc = self._features_dc
    #     features_rest = self._features_rest
    #     return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.scaling, scaling_modifier, self._rotation)

    def save_ply(self, path):
        from plyfile import PlyData, PlyElement
        # import os
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l