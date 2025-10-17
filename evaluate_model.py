#!/usr/bin/env python3
"""
evaluate_model.py

This script is used to batch evaluate models under the CryoDust3r framework on the pose estimation task.  
It supports varying the SNR for data augmentation and the number of input views (num_views),  
automatically computes various metrics (including 3D point reconstruction loss, rotation error, and translation error),  
and performs volume reconstruction.  

The reconstructed volumes are saved in MRC format, and all evaluation results are summarized into a CSV file.  

Usage example:
  python batch_evaluate.py --ckpt_path /path/to/checkpoint.pth --root_path /path/to/dataset \
       --device cuda --use_amp --augment \
       --batch --snr_list 0.05,1.0 --view_numbers 8,16,32,64 \
       --batch_size 32 --resolution 128 --reference_views 1 --num_views 64 \
       --num_workers 4 --out_dir results

Notes:
  1. Please ensure that dependencies such as torch, torchvision, tqdm, scikit-learn, and matplotlib are installed.  
  2. Depending on your dataset, you may need to adjust the `get_views()` method in `MultiviewDataset`.  

"""
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import os
import sys
import csv
import time
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision.transforms.functional as F

sys.path.append('../CryoDust3r')
from dust3r.model import AsymmetricCroCo3DStereo, inf
from dust3r.datasets.cryomulti import CryoMultiViewDataset
from dust3r.datasets.transform import DatasetTransform
from dust3r.utils.fft import ht2_center, ihtn_center
from dust3r.utils.mrc import MRCFile
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluation of CryoFastAR: pose estimation and volume reconstruction under different SNRs and numbers of views")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument('--root_path', type=str, required=True, help="Root directory of the dataset")
    parser.add_argument('--device', type=str, default='cuda', help="Device to run on, e.g. 'cuda' or 'cpu'")
    parser.add_argument('--use_amp', action='store_true', help="Whether to use Automatic Mixed Precision (AMP)")
    parser.add_argument('--augment', action='store_true', help="Whether to use data augmentation")
    parser.add_argument('--snr_list', type=str, default="0.05", help="Comma-separated list of SNR values, e.g. 0.05,1.0")
    parser.add_argument('--view_numbers', type=str, default="64", help="Comma-separated list of view counts, e.g. 8,16,32,64")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size during evaluation")
    parser.add_argument('--resolution', type=int, default=128, help="Resolution D for volume reconstruction")
    parser.add_argument('--reference_views', type=int, default=1, help="Number of reference views")
    parser.add_argument('--num_views', type=int, default=64, help="Total number of views used for evaluation")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of worker processes for DataLoader")
    parser.add_argument('--gt_image', action='store_true', help="Use ground truth images")
    parser.add_argument('--gt_pose', action='store_true', help="Use ground truth poses")
    parser.add_argument('--reg_pose', action='store_true', help="Whether to regularize poses (using the Kabsch algorithm)")
    parser.add_argument('--out_dir', type=str, default="results", help="Output directory for results")
    parser.add_argument('--apix', type=float, default=None, help="Pixel size in Ångström for volume reconstruction")
    parser.add_argument('--real', action='store_true', help="Whether to use a real dataset (Cryo-EM data)")
    parser.add_argument('--reference_offset', type=int, default=0, help="Offset of the reference view")
    parser.add_argument('--no-per-scene', action='store_true', help="If the dataset is per-scene")
    parser.add_argument('--ref-mode', type=str, choices=['manual', 'auto', 'random'], default='manual', help="Mode of reference view selection")
    parser.add_argument('--repeat', type=int, default=1, help="Number of repetitions per evaluation")

    return parser.parse_args()

# -------------------------------
# Tool Functions
# -------------------------------
def translate_image(image, translation):
    """
    Apply sub-pixel precision translation to a batch of images.

    Args:
        image (torch.Tensor): Input images of shape (B, H, W).
        translation (torch.Tensor): Translation vectors of shape (B, 2),
            where translation[:, 0] is the y-direction shift
            and translation[:, 1] is the x-direction shift.

    Returns:
        torch.Tensor: Translated images of shape (B, H, W).
    """
    B, H, W = image.shape
    translated_images = []

    for i in range(B):
        # Extract translation offsets for the current image
        ty, tx = translation[i]

        # Apply affine transformation with translation
        translated_image = F.affine(
            image[i].unsqueeze(0),  # Add channel dimension (1, H, W)
            angle=0,                # No rotation
            translate=[tx, ty],     # Translation offsets
            scale=1.0,              # No scaling
            shear=0.0,              # No shearing
            interpolation=F.InterpolationMode.BILINEAR,  # Bilinear interpolation
            fill=0                  # Fill value for empty regions
        )
        translated_images.append(translated_image.squeeze(0))  # Remove channel dimension

    # Stack results back into a batch
    translated_image = torch.stack(translated_images, dim=0)
    return translated_image


def fft2_center(image):
    """Perform centered 2D FFT."""
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(image)))

def ifft2_center(frequency):
    """Perform centered 2D IFFT."""
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(frequency)))

def ctf_correction(raw_image, chi, device="cpu"):
    """
    Perform CTF correction on the raw image.

    Parameters:
        raw_image (torch.Tensor): The raw input image, shape (H, W).
        chi (torch.Tensor): The chi phase shift map, shape (H, W).
        device (str): The device for computation (e.g., 'cpu' or 'cuda').

    Returns:
        corrected_image (torch.Tensor): The corrected image, shape (H, W).
    """
    # Ensure the inputs are on the correct device
    assert raw_image.shape == chi.shape, ValueError(f'ctf correction: The shape of input image {raw_image.shape} is not equal to the shape of input chi {chi.shape}')
    raw_image = raw_image.to(device)
    chi = chi.to(device)  # Remove batch dimension if necessary
    
    # Clamp chi values
    chi[chi <= -1.5] = -1.5

    # Initialize CTF filter
    ctf_filter = torch.zeros_like(raw_image, device=device)

    # Compute CTF
    ctf = torch.cos(chi)

    # Apply conditions to compute CTF filter
    ctf_filter[chi < 0] = 1. / ctf[chi < 0]
    ctf_filter[chi >= 0] = torch.sign(ctf[chi >= 0])

    # Apply CTF filter in Fourier domain
    corrected_image_ft = ctf_filter * fft2_center(raw_image)
    corrected_image = ifft2_center(corrected_image_ft).real

    # Convert back to torch.Tensor and move to device
    # corrected_image = torch.from_numpy(corrected_image)

    return corrected_image


def compute_ctf(chi, device="cpu"):
    """
    Perform CTF correction on the raw image.

    Parameters:
        raw_image (torch.Tensor): The raw input image, shape (H, W).
        chi (torch.Tensor): The chi phase shift map, shape (H, W).
        device (str): The device for computation (e.g., 'cpu' or 'cuda').

    Returns:
        corrected_image (torch.Tensor): The corrected image, shape (H, W).
    """
    # Ensure the inputs are on the correct device
    chi = chi.to(device)  # Remove batch dimension if necessary
    
    # Clamp chi values
    chi[chi <= -1.5] = -1.5

    # Initialize CTF filter
    ctf_filter = torch.zeros_like(chi, device=device)

    # Compute CTF
    ctf = torch.cos(chi)

    # Apply conditions to compute CTF filter
    ctf_filter[chi < 0] = 1. / ctf[chi < 0]
    ctf_filter[chi >= 0] = torch.sign(ctf[chi >= 0])

    return ctf_filter

def translate_ht(img, t, mask=None):
    """
    Translate an image by phase shifting its Hartley transform

    Inputs:
        img: HT of image (B x img_dims)
        t: shift in pixels (B x T x 2)
        mask: Mask for lattice coords (img_dims x 1)

    Returns:
        Shifted images (B x T x img_dims)

    img must be 1D unraveled image, symmetric around DC component
    """
    # H'(k) = cos(2*pi*k*t0)H(k) + sin(2*pi*k*t0)H(-k)
    B, H, W = img.shape
    img = img.reshape(B, -1)
    coords = generate_xyz_grid(1, H)[...,:2].squeeze().to(img.device) # (H, W, 2)
    coords = coords.reshape(-1, 2) / 2 # (H*W, 2), ranged from -0.5 to 0.5
    N = coords.shape[0]
    if mask is not None:
        coords = coords[mask]
    img = img.unsqueeze(1)  # Bx1xN
    t = t.reshape(B, 1, 2, 1)  # BxTx2x1 to be able to do bmm
    tfilt = coords @ t * 2 * np.pi  # BxTxNx1
    tfilt = tfilt.squeeze(-1)  # BxTxN
    c = torch.cos(tfilt)  # BxTxN
    s = torch.sin(tfilt)  # BxTxN

    shifted_img = c * img + s * img[:, :, torch.arange(N - 1, -1, -1)]
    shifted_img = shifted_img.reshape(B, H, W)
    return shifted_img

def generate_xyz_grid(B, res):
    """
    Generate a 2D grid in the XY-plane with Z set to zero, 
    repeated for a batch of size B.

    Args:
        B (int): Batch size, i.e., the number of grids to generate.
        res (int): Resolution of the grid (res x res points).

    Returns:
        torch.Tensor: A tensor of shape (B, res, res, 3), where each grid point 
        contains (x, y, z) coordinates in the range [-1, 1] for x and y, and 0 for z.
    """
    # Create a meshgrid in the range [-1, 1]
    x = torch.linspace(-1, 1, res)
    y = torch.linspace(-1, 1, res)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # Stack x, y grids and add zero z values
    zz = torch.zeros_like(xx)
    grid = torch.stack((xx, yy, zz), dim=-1)

    # Expand the grid for batch size B -> (B, res, res, 3)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

    return grid

def kabsch_pose_estimation(points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
    """
    Estimate the homogeneous transformation matrix from points1 to points2
    using the Kabsch algorithm.

    Args:
        points1 (torch.Tensor): Source point sets of shape (B, N, 3).
        points2 (torch.Tensor): Target point sets of shape (B, N, 3).

    Returns:
        torch.Tensor: Transformation matrices of shape (B, 4, 4), where
            T[:, :3, :3] is the rotation matrix and
            T[:, :3, 3] is the translation vector.
    """
    B, N, _ = points1.shape
    mean1 = points1.mean(dim=1, keepdim=True)
    mean2 = points2.mean(dim=1, keepdim=True)
    centered1 = points1 - mean1
    centered2 = points2 - mean2

    # Cross-covariance matrix
    H = torch.matmul(centered1.transpose(1, 2), centered2)

    # Singular Value Decomposition
    U, S, Vh = torch.linalg.svd(H, full_matrices=False)

    # Rotation
    R = torch.matmul(Vh.transpose(-2, -1), U.transpose(-2, -1))

    # Ensure a proper rotation (det(R) = +1)
    det_R = torch.det(R)
    mask = det_R < 0
    if mask.any():
        Vh[mask, -1, :] *= -1
        R = torch.matmul(Vh.transpose(-2, -1), U.transpose(-2, -1))
    
    # Translation
    t = mean2.squeeze(1) - torch.matmul(R, mean1.squeeze(1).unsqueeze(-1)).squeeze(-1)

    # Homogeneous transformation matrix
    T = torch.eye(4, device=points1.device, dtype=points1.dtype).unsqueeze(0).repeat(B, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t

    return T

def compute_translation(pts, D):
    """
    Compute the 2D translation offset of a point cloud's center.

    Args:
        pts (torch.Tensor): Input point cloud of shape (N, 3).
        D (int): Resolution (used to scale the translation to pixel coordinates).

    Returns:
        torch.Tensor: The 2D translation offset in pixel scale.
    """
    pts = pts.reshape(-1, 3)
    center = pts.mean(dim=0)
    center_2d = center[..., :2] * D
    return center_2d

def add_slice(volume, counts, ff_coord, ff, D, ctf_mul):
    d2 = int(D / 2)
    ff_coord = ff_coord.transpose(0, 1)
    xf, yf, zf = ff_coord.floor().long()
    xc, yc, zc = ff_coord.ceil().long()

    def add_for_corner(xi, yi, zi):
        dist = torch.stack([xi, yi, zi]).float() - ff_coord
        w = 1 - dist.pow(2).sum(0).pow(0.5)
        w[w < 0] = 0
        volume[(zi + d2, yi + d2, xi + d2)] += w * ff * ctf_mul
        counts[(zi + d2, yi + d2, xi + d2)] += w * ctf_mul**2 # Important to keep result sharp

    add_for_corner(xf, yf, zf)
    add_for_corner(xc, yf, zf)
    add_for_corner(xf, yc, zf)
    add_for_corner(xf, yf, zc)
    add_for_corner(xc, yc, zf)
    add_for_corner(xf, yc, zc)
    add_for_corner(xc, yf, zc)
    add_for_corner(xc, yc, zc)

def regularize_volume(volume, counts, reg_weight):
    regularized_counts = counts + reg_weight * counts.mean()
    regularized_counts *= counts.mean() / regularized_counts.mean()
    reg_volume = volume / regularized_counts
    return ihtn_center(reg_volume[0:-1, 0:-1, 0:-1].cpu())

def rotation_error_norm(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """
    Compute the error between two rotation matrices using the Frobenius norm
    of their difference.

    Args:
        R1 (torch.Tensor): Rotation matrix of shape (3, 3) or (B, 3, 3).
        R2 (torch.Tensor): Rotation matrix of shape (3, 3) or (B, 3, 3).

    Returns:
        torch.Tensor: If batched input is given, returns a tensor of shape (B,).
                      Otherwise, returns a scalar.
    """
    diff = R1 - R2
    # Frobenius norm: sqrt of the sum of squared matrix elements
    error = torch.norm(diff, p='fro', dim=(-2, -1))
    return error


def rotation_error_angle(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """
    Compute the angular error (in radians) between two rotation matrices.

    Formula:
        θ = arccos((trace(R1^T * R2) - 1) / 2)

    Args:
        R1 (torch.Tensor): Rotation matrix of shape (3, 3) or (B, 3, 3).
        R2 (torch.Tensor): Rotation matrix of shape (3, 3) or (B, 3, 3).

    Returns:
        torch.Tensor: If batched input is given, returns a tensor of shape (B,).
                      Otherwise, returns a scalar.
    """
    # Compute relative rotation matrix R_rel = R1^T * R2
    R_rel = torch.matmul(R1.transpose(-2, -1), R2)
    # Compute the trace of the relative rotation
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    # Compute rotation angle (in radians)
    angle_error = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
    return angle_error


def translation_error(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """
    Compute the L2 norm error between two translation vectors.

    Args:
        t1 (torch.Tensor): Translation vector of shape (2,) or (B, 2).
        t2 (torch.Tensor): Translation vector of shape (2,) or (B, 2).

    Returns:
        torch.Tensor: Translation error. If batched input is given, returns a 
                      tensor of shape (B,). Otherwise, returns a scalar.
    """
    error = torch.norm(t1 - t2, dim=-1)
    return error

class MultiviewDataset(Dataset):
    """
    A wrapper around a single-view dataset to construct multi-view inputs.

    Args:
        dataset_test: The original test dataset (e.g., CryoMultiViewDataset).
        num_views (int): Total number of views per sample.
        reference_views (int): Number of reference views.
        reference_offset (int, optional): Offset for the reference view indices. Default: 0.
        per_scene (bool, optional): If True, iterate by scene with batched inference views.
                                    If False, iterate over dataset samples directly.
    """
    def __init__(self, dataset_test, num_views, reference_views, reference_offset=0, per_scene=True):
        self.dataset_test = dataset_test
        self.num_views = num_views
        self.reference_views = reference_views
        self.inference_views = num_views - reference_views
        self.reference_views = [i + reference_offset for i in range(self.reference_views)]
        self.per_scene = per_scene
        
    def __len__(self):
        if self.per_scene:
            return len(self.dataset_test) // self.inference_views + 1
        else:
            return len(self.dataset_test)

    def __getitem__(self, index):
        if self.per_scene:
            indices = self.reference_views + [
                min(self.inference_views * index + i, len(self.dataset_test) - 1)
                for i in range(self.inference_views)
            ]
            return self.dataset_test.get_views(0, indices)
        else:
            indices = self.reference_views + [
                min(i, len(self.dataset_test) - 1)
                for i in range(self.inference_views)
            ]
            return self.dataset_test.get_views(index, indices)
    


# -------------------------------
# 可视化函数：绘制旋转误差直方图
# -------------------------------
def visualize_pose_errors(gt_poses, pred_poses, out_fig):
    """
    Compute the per-sample rotation error (in degrees), plot a histogram,
    and save it to the path specified by `out_fig`.

    Args:
        gt_poses (torch.Tensor): Ground-truth rotation matrices of shape (N, 3, 3).
        pred_poses (torch.Tensor): Predicted rotation matrices of shape (N, 3, 3).
        out_fig (str): Output file path for the saved figure.
    """
    def rotation_error(R1, R2):
        R = np.dot(R1, R2.T)
        trace_R = np.trace(R)
        cos_angle = (trace_R - 1) / 2
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)
    
    gt_np = gt_poses.cpu().numpy()
    pred_np = pred_poses.cpu().numpy()
    rotation_errors = []
    for i in range(gt_np.shape[0]):
        err = rotation_error(gt_np[i], pred_np[i])
        rotation_errors.append(err)
    rotation_errors = np.array(rotation_errors)
    rotation_errors_deg = rotation_errors * 180 / np.pi
    # If rotation error > 90°, fold it to 180° - error
    rotation_errors_deg[rotation_errors_deg > 90] = 180 - rotation_errors_deg[rotation_errors_deg > 90]
    
    print('Number of poses:', len(rotation_errors_deg))
    print('Average rotation error (degrees):', rotation_errors_deg.mean())
    
    plt.figure(figsize=(8,6))
    plt.hist(rotation_errors_deg, bins=50, color='blue', edgecolor='black')
    plt.xlabel('Rotation Error (degrees)')
    plt.ylabel('Frequency')
    plt.title(f'Rotation Error Distribution (Average: {rotation_errors_deg.mean():.2f}°)')
    plt.savefig(out_fig)
    plt.close()


def save_as_cryodrgn_format(path, pose, trans, ids=None):
    """
    Save pose and translation in CryoDRGN-compatible pkl format.

    Args:
        path (str): Output file path.
        pose (torch.Tensor): Rotation matrices of shape (N, 3, 3).
        trans (torch.Tensor): Translation vectors of shape (N, 2).
        ids (list or ndarray, optional): Reordering indices for poses/translations.
            For example, if ids = [0, 2, 1] and pose = [A, B, C],
            the reordered pose will be [A, C, B].

    Returns:
        None
    """
    assert pose.shape[0] == trans.shape[0], (
        f"the shape of pose and translation do not match: "
        f"pose: {pose.shape}, translation: {trans.shape}"
    )

    pose = pose.reshape(-1, 3, 3).cpu().numpy()
    trans = trans.reshape(-1, 2).cpu().numpy()

    if ids is not None:
        # Reorder pose and trans according to ids
        order = np.argsort(ids)
        pose = pose[order]
        trans = trans[order]

    # Adjust coordinate system convention
    pose[:, 0, 2] *= -1
    pose[:, 1, 2] *= -1
    pose[:, 2, 0] *= -1
    pose[:, 2, 1] *= -1

    # Convert translation to CryoDRGN alignment convention
    shift_x = trans[:, 0]
    shift_y = trans[:, 1]
    pred_trans_aligned = np.stack([-shift_y, -shift_x], axis=1)

    with open(path, 'wb') as f:
        pickle.dump((pose, pred_trans_aligned), f)

    return

# -------------------------------
# 合并评估与体积重建函数
# -------------------------------
def run_evaluation_and_recon(model, dataset_test, args, num_views, ref_offset, device, progress_callback=None):
    """
    Perform a single forward pass on the given dataset for both evaluation 
    (Kabsch-based) and volume reconstruction, in order to avoid redundant inference.  

    The computations include:
      - Evaluation metrics: 3D reconstruction loss (loss3d), rotation angle error 
        (rot_angle), rotation matrix error (rot_norm), and translation error (trans_error), 
        along with their mean, median, and variance;  
      - Volume reconstruction: each slice is accumulated into the volume by calling add_slice.

    Args:
        model: The model to be evaluated.
        dataset_test: The original test dataset (e.g., CryoMultiViewDataset).
        args: Command-line arguments, should include batch_size, resolution, 
              reference_views, use_amp, gt_image, gt_pose, reg_pose, etc.
        num_views: Total number of views used for the current evaluation.
        device: Computation device.
        progress_callback: Optional callable invoked after each slice accumulation.
            It receives three arguments: the current `volume_full` tensor, the
            `counts_full` tensor, and a metadata dictionary describing the
            reconstruction progress (e.g., batch index, view index, elapsed time,
            pose estimates).

    Returns:
        metrics: A dictionary containing raw values and statistics (mean, median, variance) 
                 for all evaluation metrics.
        gt_poses: Ground-truth relative poses of all samples, Tensor of shape (-1, 3, 3).
        pred_poses: Predicted relative poses of all samples, Tensor of shape (-1, 3, 3).
        volume_full: The reconstructed volume, Tensor (normalized and on CPU).
    """

    B = args.batch_size
    D = args.resolution

    if args.per_scene:
        num_images = len(dataset_test)
    else:
        num_images = len(dataset_test) * num_views

    ref_views = args.reference_views

    # Construct the multi-view dataset and DataLoader with the same parameters
    mvdataset = MultiviewDataset(
        dataset_test,
        num_views=num_views,
        reference_views=ref_views,
        reference_offset=ref_offset,
        per_scene=args.per_scene
    )
    dataloader = DataLoader(
        mvdataset,
        batch_size=B,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        persistent_workers=False,
    )
    
    # Variables for evaluation metrics
    gt_poses_list = []
    pred_poses_list = []
    loss3d_list = []       # 3D reconstruction loss, one value per sample (1D tensor)
    rot_angle_errors = []  # Rotation angle errors (in radians), stored as list
    rot_norm_errors = []   # Rotation matrix errors (Frobenius norm), stored as list
    trans_errors = []      # Translation errors (L2 norm), stored as list
    pred_translations = [] # Predicted translation vectors
    pred_ids = []
    
    # Accumulators for volume reconstruction
    volume_full = torch.zeros((D + 1, D + 1, D + 1), device=device)
    counts_full = torch.zeros((D + 1, D + 1, D + 1), device=device)
    
    # Inference time metric
    inference_time = pose_reg_time = recon_time = 0.0

    total_time_start = time.time()
    
    model.eval()

    grid_cache = {}

    def get_grid(batch_size):
        if batch_size not in grid_cache:
            grid_cache[batch_size] = generate_xyz_grid(batch_size, D).to(device)
        return grid_cache[batch_size]

    slice_counter = 0
    view_counter = 0
    total_unique = len(dataset_test)
    processed_mask = torch.zeros(total_unique, dtype=torch.bool)

    if progress_callback is not None:
        metadata = {
            "batch_index": -1,
            "view_index": -1,
            "global_view_index": 0,
            "batch_size": 0,
            "slices_processed": 0,
            "time_elapsed": 0.0,
            "sample_ids": None,
            "pred_poses": None,
            "gt_poses": None,
            "pred_translations": None,
            "reference_views": ref_views,
            "num_views_per_sample": num_views,
            "loss3d": None,
            "rot_angle_deg": None,
            "rot_fro_norm": None,
            "trans_error": None,
            "translation_positions": None,
            "images_processed": 0,
            "images_total": total_unique,
        }
        progress_callback(volume_full, counts_full, metadata)

    for batch_index, batch in enumerate(tqdm(dataloader, desc="Evaluation & Volume Reconstruction")):
        for view in batch:
            for name in 'img pose pts3d clean_img chi translation trans'.split():  # pseudo_focal
                if name not in view:
                    continue
                view[name] = view[name].to(device, non_blocking=True)
                if name == 'img':
                    view[name] = view[name].reshape(-1, 1, view[name].shape[-2], view[name].shape[-1])
        
        # One time inference
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=bool(args.use_amp)):
            start_time = time.time()
            preds, _ = model(batch)
            inference_time += time.time() - start_time
        
        poses = [view.get('pose') for view in batch]
        translates = [view.get('trans') for view in batch]
        gt_translations = [view.get('translation') for view in batch]
        gt_images = [view.get('img') for view in batch]
        gt_image_tensor = torch.cat(gt_images, dim=0)

        B = poses[0].shape[0]  # batch size
        # Iterate over each view except the reference views
        for j, pose in enumerate(poses):

            ctf_mul = 1

            pose0 = poses[0].reshape(-1, 3, 3)
            reference_dirs_cpu = pose0[:, :, 2].detach().cpu()

            if j < ref_views:
                continue

            # Compute relative pose: pose_rel = poses[j] @ poses[0]^T
            pose_rel = pose.reshape(-1, 3, 3) @ pose0.transpose(2, 1)
            
            # ---------------------------
            # Evaluation: compute 3D reconstruction loss and errors
            # ---------------------------
            # Compute ground truth 3D coordinates (for evaluation). 
            # Note: this does not depend on predictions
            ff_coord_gt = get_grid(B).reshape(B, -1, 3)
            ff_coord_gt = ff_coord_gt @ pose_rel
            
            # Compute predicted 3D coordinates for evaluation: taken from preds[j]['pts3d']
            ff_coord_eval = preds[j]['pts3d'].reshape(B, D, D, 3).permute(0, 2, 1, 3).reshape(B, -1, 3)
            ff_coord_eval = (ff_coord_eval - 0.5) * 2
            
            start_time = time.time()
            # Use the generated reference point cloud (same as generate_xyz_grid) 
            # and predicted 3D coordinates to compute Kabsch-estimated rotation
            ref_pc = get_grid(B).reshape(B, -1, 3)
            T = kabsch_pose_estimation(ref_pc, ff_coord_eval)
            pred_poses_eval = T[:, :3, :3].transpose(2, 1)
            pose_reg_time += time.time() - start_time
            
            # Compute 3D reconstruction loss (mean per sample)
            loss3d = torch.abs(ff_coord_eval - ff_coord_gt) * D
            loss3d = loss3d.reshape(B, -1).mean(dim=-1)
            loss3d_list.append(loss3d)
            loss3d_cpu = loss3d.detach().cpu()
            
            gt_poses_list.append(pose_rel)
            pred_poses_list.append(pred_poses_eval)
            
            # Compute rotation errors: call rotation_error_angle and rotation_error_norm (batched)
            rot_angle = rotation_error_angle(pose_rel, pred_poses_eval)  # shape (B,)
            rot_angle_errors.extend(rot_angle.tolist())
            rot_angle_deg = torch.rad2deg(rot_angle).detach().cpu()
            rot_norm = rotation_error_norm(pose_rel, pred_poses_eval)    # shape (B,)
            rot_norm_errors.extend(rot_norm.tolist())
            rot_norm_cpu = rot_norm.detach().cpu()
            
            # Compute translation errors: for each sample, obtain 2D translation vector 
            # via compute_translation, then compute L2 error
            trans_error_vals = []
            trans_position_vals = []
            if args.augment:
                for b in range(B):
                    t_gt = translates[j][b][...,:2] * D
                    t_pred = compute_translation(ff_coord_eval[b], D)
                    trans_err = translation_error(t_gt, t_pred)
                    trans_errors.append(trans_err.item())
                    trans_error_vals.append(trans_err.detach().cpu())
                    trans_position_vals.append(t_gt.detach().cpu())
            else:
                for b in range(B):
                    t_gt = gt_translations[j][b][...,:2] * D
                    t_pred = compute_translation(ff_coord_eval[b], D)
                    trans_err = translation_error(t_gt, t_pred)
                    trans_errors.append(trans_err.item())
                    trans_error_vals.append(trans_err.detach().cpu())
                    trans_position_vals.append(t_gt.detach().cpu())
                    pred_translations.append(t_pred)
            if trans_error_vals:
                trans_error_cpu = torch.stack(trans_error_vals).to(torch.float32)
                trans_position_cpu = torch.stack(trans_position_vals).to(torch.float32)
            else:
                trans_error_cpu = None
                trans_position_cpu = None
            
            pred_ids.append(batch[j]['id'])
            
            # ---------------------------
            # Volume reconstruction: choose how to compute ff_coord depending on branch
            # ---------------------------
            # Note: we use the same basic computation as in the evaluation part.
            # If args.gt_pose is True, use ground truth pose.
            # Otherwise, handle according to args.gt_image and args.reg_pose 
            # (same logic as in run_volume_reconstruction).
            start_time = time.time()
            pred_trans_current = None

            if args.gt_pose:
                ff_coord_vol = ff_coord_gt
            elif not args.gt_image:
                ff_coord_vol = (preds[j]['pts3d']
                                .reshape(B, D, D, 3)
                                .permute(0, 2, 1, 3)
                                .reshape(B, -1, 3) - 0.5)
                ff_coord_vol = ff_coord_vol @ pose0.transpose(2, 1)
            else:
                ff_coord_vol = (preds[j]['pts3d']
                                .reshape(B, D, D, 3)
                                .permute(0, 2, 1, 3)
                                .reshape(B, -1, 3) - 0.5)


                pred_trans = []
                for b in range(B):
                    # multiply by 2 (the coordinate should be [-1, 1], but current are [-0.5, 0.5])
                    t_val = compute_translation(ff_coord_vol[b], D) * 2 
                    pred_trans.append(t_val)
                pred_trans = torch.stack(pred_trans, dim=0).to(device)
                pred_trans_current = pred_trans
                
                if args.reg_pose:
                    pred_poses_reg = pred_poses_eval

                    grid = get_grid(B)
                    ff_coord_vol = grid.reshape(B, -1, 3)
                    ff_coord_vol = ff_coord_vol * (D // 2)
                    ff_coord_vol = ff_coord_vol @ pred_poses_reg @ pose0.transpose(2, 1)
                else:
                    ff_coord_vol = ff_coord_vol @ pose0.transpose(2, 1)
            
            # computing slice for reconstruction
            if args.gt_image:
                img = batch[j]['img'].squeeze(1)  # (B, H, W)

                # ctf correction
                if 'chi' in batch[j]:
                    chi = batch[j]['chi'].squeeze()
                    # img = ctf_correction(img, chi, device=device)
                    ctf_mul = compute_ctf(chi, device=device)

                img = ht2_center(img)
            else:
                img = ht2_center(preds[j]['fval'].squeeze())

            if not args.gt_pose:
                img = translate_ht(img, -pred_trans, mask=None)
            elif 'translation' in batch[j]:
                translation = batch[j]['translation']
                translation = translation.squeeze()[...,:2]
                img = translate_ht(img, translation * D, mask=None)

            img = img.transpose(2, 1)
            img = img.reshape(B, -1)
            
            circular_mask = ff_coord_vol.pow(2).sum(-1) < (D // 2) ** 2
            img = img[circular_mask]
            ff_coord_vol = ff_coord_vol[circular_mask]
            ctf_mul = ctf_mul.reshape(B, -1)[circular_mask]
            add_slice(volume_full, counts_full, ff_coord_vol, img, D, ctf_mul)
            recon_time += time.time() - start_time

            view_counter += 1
            if progress_callback is not None:
                sample_ids = batch[j].get('id')
                if isinstance(sample_ids, torch.Tensor):
                    sample_ids_meta = sample_ids.detach().cpu()
                else:
                    sample_ids_meta = sample_ids
                if sample_ids_meta is not None:
                    ids_tensor = torch.as_tensor(sample_ids_meta, dtype=torch.long)
                    ids_tensor = ids_tensor[(ids_tensor >= 0) & (ids_tensor < total_unique)]
                    processed_mask[ids_tensor] = True
                images_processed = int(processed_mask.sum().item())
                slice_counter = images_processed
                progress_time = inference_time + recon_time
                thumbnails = None
                if 'img' in batch[j]:
                    thumb_tensor = gt_image_tensor.detach().cpu().squeeze(1)
                    # thumb_tensor = batch[j]['img'].detach().cpu().squeeze(1)
                    if thumb_tensor.ndim == 3 and thumb_tensor.shape[0] > 0:
                        max_thumbs = min(10, thumb_tensor.shape[0])
                        thumb_list = []
                        for idx_thumb in range(max_thumbs):
                            frame = thumb_tensor[-idx_thumb-1]
                            frame = frame - frame.min()
                            denom = frame.max()
                            if denom > 0:
                                frame = frame / denom
                            thumb_list.append(frame.to(torch.float32))
                        if thumb_list:
                            thumbnails = torch.stack(thumb_list, dim=0)
                metadata = {
                    "batch_index": batch_index,
                    "view_index": j,
                    "global_view_index": view_counter,
                    "batch_size": B,
                    "slices_processed": slice_counter,
                    "time_elapsed": progress_time,
                    "sample_ids": sample_ids_meta,
                    "pred_poses": pred_poses_eval.detach().cpu(),
                    "gt_poses": pose_rel.detach().cpu(),
                    "pred_translations": pred_trans_current.detach().cpu() if pred_trans_current is not None else None,
                    "reference_dirs": reference_dirs_cpu,
                    "reference_views": ref_views,
                    "num_views_per_sample": len(poses),
                    "loss3d": loss3d_cpu,
                    "rot_angle_deg": rot_angle_deg,
                    "rot_fro_norm": rot_norm_cpu,
                    "trans_error": trans_error_cpu,
                    "translation_positions": trans_position_cpu,
                    "images_processed": images_processed,
                    "images_total": total_unique,
                    "thumbnails": thumbnails,
                }
                progress_callback(volume_full, counts_full, metadata)
        
    
    pred_ids = torch.cat(pred_ids, dim=0).reshape(-1)
    # After finishing all batches, aggregate evaluation data
    def reorder(input):
        order = np.argsort(pred_ids)
        return input[order]

    gt_poses_tensor = reorder(torch.cat(gt_poses_list, dim=0).reshape(-1, 3, 3))[:num_images]
    pred_poses_tensor = reorder(torch.cat(pred_poses_list, dim=0).reshape(-1, 3, 3))[:num_images]
    losses_3d = reorder(torch.cat(loss3d_list, dim=0)).reshape(-1)[:num_images]  # 1D tensor
    if len(pred_translations) > 0:
        pred_translations = reorder(torch.cat(pred_translations, dim=0).reshape(-1, 2))[:num_images]

    rot_angle_tensor = reorder(torch.tensor(rot_angle_errors))[:num_images]
    rot_norm_tensor = reorder(torch.tensor(rot_norm_errors))[:num_images]
    trans_errors_tensor = reorder(torch.tensor(trans_errors))[:num_images]

    loss3d_mean = losses_3d.mean().item()
    loss3d_median = losses_3d.median().item()
    loss3d_var = losses_3d.var(unbiased=False).item()

    rot_angle_mean = rot_angle_tensor.mean().item()
    rot_angle_median = rot_angle_tensor.median().item()
    rot_angle_var = rot_angle_tensor.var(unbiased=False).item()

    rot_norm_mean = rot_norm_tensor.mean().item()
    rot_norm_median = rot_norm_tensor.median().item()
    rot_norm_var = rot_norm_tensor.var(unbiased=False).item()

    trans_mean = trans_errors_tensor.mean().item()
    trans_median = trans_errors_tensor.median().item()
    trans_var = trans_errors_tensor.var(unbiased=False).item()

    metrics = {
        "3d_loss_raw": losses_3d,  # 1D tensor
        "3d_loss_mean": loss3d_mean,
        "3d_loss_median": loss3d_median,
        "3d_loss_variance": loss3d_var,
        "rot_angle_error_raw": rot_angle_tensor,  # 单位：弧度
        "rot_angle_error_mean": rot_angle_mean,
        "rot_angle_error_median": rot_angle_median,
        "rot_angle_error_variance": rot_angle_var,
        "rot_norm_error_raw": rot_norm_tensor,    # Frobenius 范数
        "rot_norm_error_mean": rot_norm_mean,
        "rot_norm_error_median": rot_norm_median,
        "rot_norm_error_variance": rot_norm_var,
        "trans_error_raw": trans_errors_tensor,
        "trans_error_mean": trans_mean,
        "trans_error_median": trans_median,
        "trans_error_variance": trans_var
    }
    
    start_time = time.time()
    # normalizing the volume
    counts_full[counts_full == 0] = 1
    volume_full = regularize_volume(volume_full, counts_full, 10.0)
    recon_time += time.time() - start_time

    wall_clock_total = time.time() - total_time_start
    total_time = inference_time + recon_time
    
    metrics["inference_time"] = inference_time
    metrics["pose_reg_time"] = pose_reg_time
    metrics["recon_time"] = recon_time
    metrics["total_time"] = total_time
    metrics["wall_clock_time"] = wall_clock_total
    
    return metrics, gt_poses_tensor, pred_poses_tensor, pred_translations, pred_ids, volume_full

def select_ref_view_by_conf(model, dataset_test, args, num_views, device):
    """
    Select reference view: choose the reference view based on model confidence.

    Args:
        model: Model to be evaluated.
        dataset_test: Original test dataset (e.g., CryoMultiViewDataset).
        args: Command-line arguments, should include batch_size, resolution,
              reference_views, use_amp, gt_image, gt_pose, reg_pose, etc.
        num_views: Total number of views used for evaluation.
        device: Computation device.

    Returns:
        ref_offset: The offset of the selected reference view.
    """
    # Randomly select num_views views
    indices = np.random.choice(len(dataset_test), num_views, replace=False)
    print('random choose index:', indices)
    best_conf = -100
    best_index = 0
    # Rotate candidate reference view and query dataset to get data
    for i in range(num_views):
        new_indices = indices.copy()
        # Randomly replace the first view
        new_indices[0] = np.random.choice(len(dataset_test), 1)[0]
        views = dataset_test.get_views(0, new_indices)

        for view in views:
            for name in 'img pose pts3d clean_img chi translation trans'.split():  # pseudo_focal
                if name not in view:
                    continue
                # Check if the data type is numpy array
                # If yes, convert to PyTorch tensor
                # Note: view[name] is a numpy array
                if isinstance(view[name], np.ndarray):
                    # Convert numpy array to PyTorch tensor and move to device
                    # Note: view[name] may be multi-dimensional
                    # Reshape according to actual shape
                    view[name] = torch.from_numpy(view[name]).to(device, non_blocking=True)
                    if name == 'img':
                        view[name] = torch.from_numpy(view[name]).reshape(
                            -1, 1, view[name].shape[-2], view[name].shape[-1]
                        )
                else:
                    view[name] = view[name].to(device, non_blocking=True)
                    if name == 'img':
                        view[name] = view[name].reshape(-1, 1, view[name].shape[-2], view[name].shape[-1])
                
        # Run inference on model with selected views
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=bool(args.use_amp)):
            preds, _ = model(views)
        # Compute average confidence across all views
        conf = 0
        for j in range(num_views):
            conf += preds[j]['conf'].reshape(-1).mean().item()
        conf = conf / num_views
        # Select the reference view with the highest confidence
        if conf > best_conf:
            print('find a better reference view:', new_indices[0], 'with confidence:', conf)
            best_conf = conf
            best_index = new_indices[0]
    # Return reference view offset
    return best_index, best_conf

# -------------------------------
# Batch evaluation function: loop over different SNR and number of views
# -------------------------------
def do_evaluation(args, device):

    if args.no_per_scene:
        args.per_scene = False
    else:
        args.per_scene = True
    
    if args.real:
        args.mode = 'std'
    else:
        args.mode = 'real'

    snr_list = [float(x) for x in args.snr_list.split(',')]
    view_numbers = [int(x) for x in args.view_numbers.split(',')]
    summary = []
    repeat = args.repeat
    for i in range(repeat):
        for snr in snr_list:
            for num_views in view_numbers:
                # randomly generate a load reference number from 0 to 50000
                # args.reference_offset = random.randint(0, 50000)
                print(f"\n==================== Processing: snr={snr} (ignore this if you are using real data), num_views={num_views} ====================")
                # Set data augmentation transform according to augment argument (update SNR parameter here)
                if args.augment:
                    data_transform = DatasetTransform(snr=snr, ctf=True, ctf_param_path='ctf_params_filtered.csv', shift=10.0, repeat=1, random_apix=False)
                else:
                    data_transform = None

                # Build test dataset, update num_views parameter
                dataset_test = CryoMultiViewDataset(
                    root_path=args.root_path,
                    transform=data_transform,
                    mode=args.mode,
                    num_views=num_views,
                    per_scene=args.per_scene,
                    apix_path='apix.csv',
                    apix=args.apix,
                    load_draco=False,
                    # first_N=50000,
                )
                # Build model, update num_views parameter
                model = AsymmetricCroCo3DStereo(
                    pos_embed='RoPE100',
                    img_size=(128, 128),
                    head_type='fourier',
                    output_mode='pts3d',
                    depth_mode=('exp', -inf, inf),
                    conf_mode=('exp', 1, inf),
                    enc_embed_dim=1024,
                    enc_depth=24,
                    enc_num_heads=16,
                    dec_type='croco',
                    dec_embed_dim=768,
                    dec_depth=12,
                    dec_num_heads=12,
                    num_views=num_views,
                    has_neck=True
                )
                try:
                    checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
                    model.load_state_dict(checkpoint['model'], strict=True)
                except Exception as e:
                    print(f"Error loading checkpoint: {e}")
                    continue
                model = model.to(device)
                model.eval()

                if args.ref_mode == 'manual':
                    ref_offset = args.reference_offset
                elif args.ref_mode == 'auto':
                    print('automatically select reference view')
                    start_time = time.time()
                    ref_offset, ref_conf = select_ref_view_by_conf(model, dataset_test, args, num_views, device)
                    end_time = time.time()
                    print(f'Selected final reference view: {ref_offset} with confidence {ref_conf}, {end_time - start_time:.2f} seconds' )
                elif args.ref_mode == 'random':
                    ref_offset = np.random.randint(0, len(dataset_test) - args.reference_views)
                else:
                    raise ValueError(f"Unsupported reference mode: {args.ref_mode}")
                print("reference view offset", ref_offset)
                # Run Kabsch evaluation to compute 3D reconstruction loss, rotation error and translation error (new function returns a metrics dict), and perform volume reconstruction
                metrics, gt_poses, pred_poses, pred_trans, pred_ids, volume = run_evaluation_and_recon(model, dataset_test, args, num_views, ref_offset, device)
                
                if not os.path.exists(args.out_dir):
                    os.makedirs(args.out_dir)
                # Save rotation error histogram
                rot_fig = os.path.join(args.out_dir, f"rotation_error_snr{snr}_views{num_views}.png")
                visualize_pose_errors(gt_poses, pred_poses, rot_fig)
                print(f"Saved rotation error histogram to {rot_fig}")

                # Automatically generate summary dictionary: store snr, num_views and all metrics keys ending with "mean", "median", "variance"
                summary_dict = {"snr": snr, "num_views": num_views, 'ref_offset': ref_offset}
                for key, value in metrics.items():
                    # Save only statistics, skip raw tensors (keys containing "raw")
                    if "raw" not in key:
                        summary_dict[key] = value

                summary.append(summary_dict)
                print(f"Combination snr={snr}, num_views={num_views}:")
                for k, v in summary_dict.items():
                    if k not in ["snr", "num_views"]:
                        print(f"  {k} = {v:.4f}")
                
                
                # Save reconstructed volume file, filename contains current snr and number of views
                if args.per_scene:
                    out_filename = os.path.join(args.out_dir, f"volume_snr{snr}_views{num_views}.mrc")
                    apix = dataset_test.cached_apix
                    arr = volume.detach().to(torch.float32).cpu().numpy()
                    MRCFile.write(out_filename, arr, Apix=apix)
                    print(f"Saved volume to {out_filename}")

                # Save estimated poses and translations in cryodrgn format
                if len(pred_trans) > 0:
                    pose_path = os.path.join(args.out_dir, 'poses.pkl')
                    save_as_cryodrgn_format(pose_path, pred_poses, pred_trans / volume.shape[-1])
                    print(f"Saved poses to {pose_path}")

    # Save summary results to CSV file (automatically extract keys from summary dictionaries)
    summary_file = os.path.join(args.out_dir, "summary.csv")
    if summary:
        # Automatically obtain all keys (assuming each summary dict has the same keys)
        fieldnames = list(summary[0].keys())
        with open(summary_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary:
                writer.writerow(row)
        print(f"\nEvaluation complete. Summary saved to {summary_file}")
    else:
        print("No summary data available.")


def main():
    args = parse_args()
    device = args.device

    do_evaluation(args, device)
    
if __name__ == '__main__':
    main()
