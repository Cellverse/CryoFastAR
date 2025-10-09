# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------
import tqdm
import torch
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_nans
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf
def load_model(model_path, device):
    print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    print(f"instantiating : {args}")
    net = eval(args)
    print(net.load_state_dict(ckpt['model'], strict=False))
    return net.to(device)

def _interleave_imgs(imgs):

    res = {}
    keys = imgs[0].keys()

    for key in keys:
        values = [img[key] for img in imgs]
        if isinstance(values[0], torch.Tensor):
            stacked = torch.stack(values, dim=1)
            value = stacked.flatten(0, 1)
        else:
            value = [x for pair in zip(*values) for x in pair]
        res[key] = value

    return res


def make_batch_symmetric(batch):
    view_num = len(batch)
    symmetric_views = [[] for _ in range(view_num)]
    for i in range(len(batch)):
        for j in range(len(batch)):
            symmetric_views[(i + j) % view_num].append(batch[j])

    symmetric_views = [_interleave_imgs(symmetric_views[i]) for i in range(view_num)]

    return tuple(symmetric_views)
    # view1, view2 = batch
    # view1, view2 = (_interleave_imgs([view1, view2]), _interleave_imgs([view2, view1]))
    # return view1, view2


def generate_xyz_grid(B, res):
    # Create a mesh grid with x and y values ranging from -1 to 1
    x = torch.linspace(-1, 1, res)
    y = torch.linspace(-1, 1, res)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # Stack the grids and add the z values which are all zero
    zz = torch.zeros_like(xx)
    grid = torch.stack((xx, yy, zz), dim=-1)

    # Repeat the grid B times to match the shape (B, res, res, 3)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

    return grid

def generate_theta_phi_grid(B, res):
    theta = torch.linspace(-torch.pi, torch.pi, res)
    phi = torch.linspace(-torch.pi/2, torch.pi/2, res)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')

    # grid = torch.stack([theta_grid, phi_grid], dim=-1) # (B, res, res, 2)
    # grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

    return theta_grid, phi_grid


# Function to convert spherical coordinates to Cartesian coordinates
def spherical_to_cartesian(theta, phi):
    x = torch.cos(phi) * torch.cos(theta)
    y = torch.cos(phi) * torch.sin(theta)
    z = torch.sin(phi)
    return torch.stack((x, y, z), dim=-1)

# Function to convert Cartesian coordinates back to spherical coordinates
def cartesian_to_spherical(coords):
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
    theta = torch.atan2(y, x)
    phi = torch.asin(z / torch.sqrt(x**2 + y**2 + z**2))
    return theta, phi

# Function to calculate the great-circle distance using the Haversine formula
def haversine_distance(theta1, phi1, theta2, phi2):
    dtheta = theta2 - theta1
    dphi = phi2 - phi1
    a = torch.sin(dphi / 2) ** 2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(dtheta / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return c

# calculate the intersection line
def calc_intersection_line(rot1, rot2):
    norm1 = rot1[:, 2, :]
    norm2 = rot2[:, 2, :]
    line = torch.linalg.cross(norm1, norm2)
    line = line / torch.norm(line, dim=1, keepdim=True)
    return line


def compute_mmd_loss(x, y, kernel='rbf'):
    """
    计算 MMD Loss (Patch-wise)
    :param x: Tensor (B, N, D) - ViT Patch Embeddings for Source Domain
    :param y: Tensor (B, N, D) - ViT Patch Embeddings for Target Domain
    :param kernel: 核函数类型 (默认 RBF)
    :return: MMD Loss 值
    """
    x = x.reshape(-1, x.shape[-1])  # (B*N, D)
    y = y.reshape(-1, y.shape[-1])  # (B*N, D)

    with torch.no_grad():
        pairwise_dists = torch.cdist(torch.cat([x, y]), torch.cat([x, y]), p=2).pow(2)
        sigma = torch.median(pairwise_dists)  # 计算 pairwise 距离的中位数
        beta = 1 / (2 * sigma)
    def rbf_kernel(x, y, beta=1.0):

        dist_xx = torch.cdist(x, x, p=2).pow(2)
        dist_yy = torch.cdist(y, y, p=2).pow(2)
        dist_xy = torch.cdist(x, y, p=2).pow(2)
        K_xx = torch.exp(-beta * dist_xx)
        K_yy = torch.exp(-beta * dist_yy)
        K_xy = torch.exp(-beta * dist_xy)
        return K_xx, K_yy, K_xy

    K_xx, K_yy, K_xy = rbf_kernel(x, y)

    def remove_diag(K):
        return K * (1 - torch.eye(K.shape[0], device=K.device))  # 置零对角线元素

    mmd = remove_diag(K_xx).mean() + remove_diag(K_yy).mean() - 2 * K_xy.mean()
    
    return mmd

def loss_of_one_batch(batch, model, criterion, device, symmetrize_batch=False, use_amp=False, ret=None, grid_loss=True, data_recycle=False):

    # experimental settings
    if hasattr(model, 'module'):
        has_conf = bool(model.module.conf_mode)
    else:
        has_conf = bool(model.conf_mode)
    corr_loss = False
    normal_loss = False

    for view in batch:
        for name in 'img pose pts3d trans clean_img'.split():  # pseudo_focal
            if name not in view:
                continue
            if name == 'img': # (view_num, repeat, W, H)
                repeat = view[name].shape[1]
                if repeat > 1:
                    view['repeat_img'] = view[name][:, 1:repeat, :, :].to(device, non_blocking=True) # (view_num, repeat - 1, W, H)
                    view[name] = view[name][:, 0, :, :]
                view[name] = view[name].reshape(-1, 1, view[name].shape[-2], view[name].shape[-1]) # (view_num, 1, W, H)
            elif name == 'pose':
                view[name] = view[name].reshape(-1, 3, 3)
            view[name] = view[name].to(device, non_blocking=True)

    if symmetrize_batch:
        batch = make_batch_symmetric(batch)

    view1 = batch[0]
    # view2 = batch[1]
    B = batch[0]['pose'].shape[0]
    res = view1['img'].shape[-1]
    assert view1['img'].shape[-1] == view1['img'].shape[-2], f"image should be a square, now {view1['img'].shape[-1], view1['img'].shape[-2]}."

    with torch.amp.autocast('cuda', enabled=bool(use_amp)):
        # forward pass
        preds, _ = model(batch, is_symmetrized=symmetrize_batch)

        fvals = [pred.get('fval') for pred in preds]

        loss_2d = 0
        for i, fval in enumerate(fvals):
            # if exists batch[i]['repeat_img'], then use it
            if 'repeat_img' in batch[i]:
                # loss_2d += criterion(fval, batch[i]["repeat_img"][:, :1, :, :]) # (view_num, repeat-1, W, H) -> (view_num, 1, W, H)
                loss_2d += criterion(fval, batch[i]['clean_img'])
            else:
                loss_2d += criterion(fval, batch[i]['img']) # (view_num, 1, W, H)

        # loss_2d = loss_2d.mean() / len(batch)
        # loss_2d = 0.1 * loss_2d # FIXME: special case for snr 0.1, should be fixed after the training
        
        # TODO: data recycling
        if data_recycle:
            for i, fval in enumerate(fvals):
                batch[i]['img'] = fval.detach()
            preds = model(batch, is_symmetrized=symmetrize_batch)

            fvals = [pred.get('fval') for pred in preds]
            for i, fval in enumerate(fvals):
                # if exists batch[i]['repeat_img'], then use it
                if 'repeat_img' in batch[i]:
                    # loss_2d += criterion(fval, batch[i]["repeat_img"][:, :1, :, :]) # (view_num, repeat-1, W, H) -> (view_num, 1, W, H)
                    loss_2d += criterion(fval, batch[i]['clean_img'])
                else:
                    loss_2d += criterion(fval, batch[i]['img']) # (view_num, 1, W, H)

        loss_2d = loss_2d.mean() / len(batch)

        poses = [view.get('pose') for view in batch] # [# (B, 3, 3)]
        translations = [view.get('trans') for view in batch] # [# (B, 1, 3)]
        pts_list = [pred.get('pts3d').reshape(B, res, res, 3).permute(0, 2, 1, 3).reshape(B, -1, 3) for pred in preds] # [(B, W, H, 3)]
        if has_conf:
            conf_list = [pred.get('conf').reshape(B, -1, 1) for pred in preds]
        # 3D position loss
        # pts1 = pred1.get('pts3d').reshape(B, -1, 3) # (B, H, W, 3)
        # pts2 = pred2.get('pts3d').reshape(B, -1, 3) # (B, H, W, 3)

        pose0 = poses[0]

        raw_loss_3d = 0 # only for visualization, no backward
        loss_3d =  0
        loss_grid = 0
        loss_trans = 0
        # print(grid.shape, poses[0].shape, poses[1].shape)
        for i, pose in enumerate(poses):
            pose_rel = pose @ pose0.transpose(2,1) # (B, 3, 3)
            trans_i = translations[i] # (B, 1, 3)

            grid = generate_xyz_grid(B, res).to(device) # (B, W, H, 3)
            grid = grid.reshape(B, -1, 3) # (B, W * H, 3)
            grid = grid @ pose_rel  # (B, W * H, 3)
            
            if trans_i is not None:
                # We want to directly learn the 2D translation for every image, we don't have to consider it for multi-view
                # translation = trans_i @ pose_rel - trans0 # (B, 1, 3)
                grid += trans_i.repeat(1, res*res, 1) # (B, W * H, 3)

            # normalize the grid to [0, 1]
            grid = grid / 2 + 0.5 #+ tans

            pts = pts_list[i] # (B, W * H, 3)
            batch[i]['pts'] = grid

            raw_loss_3d += (pts - grid).pow(2)
            if has_conf: # we don't add this for the first view
                conf = conf_list[i]
                loss_3d += (pts - grid).pow(2) * conf # torch.sqrt((pts - grid).pow(2).sum(-1)) * conf

            else:
                loss_3d += (pts - grid).pow(2) # torch.sqrt((pts - grid).pow(2).sum(-1))

            center_gt = (grid).mean(dim=1)
            center_pred = (pts).mean(dim=1)
            loss_trans += torch.abs(center_gt - center_pred).sum(-1) / 2 # (B,)
                

            if grid_loss:
                pts = pts.reshape(B, res, res, 3) # (B, W, H, 3)
                true_delta_x = torch.tensor([1 / res, 0, 0], device=device).reshape(1,1,3).repeat(B, (res-1) * res, 1)
                true_delta_x = true_delta_x @ pose_rel
                true_delta_x = true_delta_x.reshape(B, res-1, res, 3)

                true_delta_y = torch.tensor([0, 1 / res, 0], device=device).reshape(1,1,3).repeat(B, (res-1) * res, 1)
                true_delta_y = true_delta_y @ pose_rel
                true_delta_y = true_delta_y.reshape(B, res, res-1, 3)
                # calculate the 3D distance between the points in row and column
                delta_x = pts[:, 1:, :, :] - pts[:, :-1, :, :] # (B, W-1, H, 3)
                delta_x = torch.abs((delta_x - true_delta_x)).sum(dim=-1)

                delta_y = pts[:, :, 1:, :] - pts[:, :, :-1, :] # (B, W, H-1, 3)
                delta_y = torch.abs((delta_y - true_delta_y)).sum(dim=-1)
                # print(f"view {i}: delta_x: {delta_x.mean()}, delta_y: {delta_y.mean()}")

                loss_grid += delta_x.mean() + delta_y.mean()


        loss_3d = loss_3d.mean() / len(batch)
        raw_loss_3d = raw_loss_3d.mean() / len(batch)

        loss_trans = loss_trans.mean() / len(batch)

        if grid_loss:
            loss_grid = loss_grid.mean() / len(batch)

        if corr_loss:
            loss_intersect = loss_intersect.mean() / len(batch)
        if normal_loss:
            loss_normal = loss_normal.mean() / len(batch)

        conf_reg = 0
        conf_avg = 0
        if has_conf:
            for conf in conf_list:
                conf_reg -= 0.1 * torch.log(conf)
                conf_avg += conf.mean()

            conf_reg = conf_reg.mean() / len(batch)
            conf_avg = conf_avg.mean() / len(batch)

        loss = loss_3d + loss_2d + conf_reg + 10 * loss_grid

        # Prepare the result dictionary with all relevant data and metrics
        result = {
                    'view1': batch[0],
                    'view2': batch[1],
                    'pts3d_1': pts_list[0],
                    'pts3d_2': pts_list[1],
                    'fval1': fvals[0],
                    'fval2': fvals[1],
                    'gt_fval1': batch[0]['img'],
                    'gt_fval2': batch[1]['img'],
                    'loss': loss,
                    'loss_3d': raw_loss_3d,
                    'loss_2d': loss_2d,
                    'loss_trans': loss_trans,
                    'loss_grid': 10 * loss_grid,
                    'conf_reg': conf_reg,
                    'conf_avg': conf_avg
                }
    return result[ret] if ret else result


@torch.no_grad()
def inference(pairs, model, device, batch_size=8):
    # print(f'>> Inference with model on {len(pairs)} image pairs')
    result = []

    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    for i in tqdm.trange(0, len(pairs), batch_size):
        res = loss_of_one_batch(collate_with_cat(pairs[i:i+batch_size]), model, None, device)
        result.append(to_cpu(res))

    result = collate_with_cat(result, lists=multiple_shapes)

    torch.cuda.empty_cache()
    return result


def check_if_same_size(pairs):
    shapes1 = [img1['img'].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2['img'].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(shapes2[0] == s for s in shapes2)


def get_pred_pts3d(gt, pred, use_pose=False):
    if 'depth' in pred and 'pseudo_focal' in pred:
        try:
            pp = gt['camera_intrinsics'][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif 'pts3d' in pred:
        # pts3d from my camera
        pts3d = pred['pts3d']

    elif 'pts3d_in_other_view' in pred:
        # pts3d from the other camera, already transformed
        assert use_pose is True
        return pred['pts3d_in_other_view']  # return!

    if use_pose:
        camera_pose = pred.get('camera_pose')
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d


def find_opt_scaling(gt_pts1, gt_pts2, pr_pts1, pr_pts2=None, fit_mode='weiszfeld_stop_grad', valid1=None, valid2=None):
    assert gt_pts1.ndim == pr_pts1.ndim == 4
    assert gt_pts1.shape == pr_pts1.shape
    if gt_pts2 is not None:
        assert gt_pts2.ndim == pr_pts2.ndim == 4
        assert gt_pts2.shape == pr_pts2.shape

    # concat the pointcloud
    nan_gt_pts1 = invalid_to_nans(gt_pts1, valid1).flatten(1, 2)
    nan_gt_pts2 = invalid_to_nans(gt_pts2, valid2).flatten(1, 2) if gt_pts2 is not None else None

    pr_pts1 = invalid_to_nans(pr_pts1, valid1).flatten(1, 2)
    pr_pts2 = invalid_to_nans(pr_pts2, valid2).flatten(1, 2) if pr_pts2 is not None else None

    all_gt = torch.cat((nan_gt_pts1, nan_gt_pts2), dim=1) if gt_pts2 is not None else nan_gt_pts1
    all_pr = torch.cat((pr_pts1, pr_pts2), dim=1) if pr_pts2 is not None else pr_pts1

    dot_gt_pr = (all_pr * all_gt).sum(dim=-1)
    dot_gt_gt = all_gt.square().sum(dim=-1)

    if fit_mode.startswith('avg'):
        # scaling = (all_pr / all_gt).view(B, -1).mean(dim=1)
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
    elif fit_mode.startswith('median'):
        scaling = (dot_gt_pr / dot_gt_gt).nanmedian(dim=1).values
    elif fit_mode.startswith('weiszfeld'):
        # init scaling with l2 closed form
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (all_pr - scaling.view(-1, 1, 1) * all_gt).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip_(min=1e-8).reciprocal()
            # update the scaling with the new weights
            scaling = (w * dot_gt_pr).nanmean(dim=1) / (w * dot_gt_gt).nanmean(dim=1)
    else:
        raise ValueError(f'bad {fit_mode=}')

    if fit_mode.endswith('stop_grad'):
        scaling = scaling.detach()

    scaling = scaling.clip(min=1e-3)
    # assert scaling.isfinite().all(), bb()
    return scaling
