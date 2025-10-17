#!/usr/bin/env python3
"""
Utility script to monitor CryoFastAR's reconstruction process.

It reuses the evaluation pipeline and registers a callback that captures
intermediate diagnostics (renders, pose statistics, timings) without
dumping intermediate MRC volumes. Each saved snapshot can visualize the
current volume slices alongside the accumulated pose distribution.
"""

import argparse
import csv
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402
import subprocess  # noqa: E402
from scipy.ndimage import rotate  # noqa: E402

BASE_FONT_SIZE = 12
TITLE_FONT_SIZE = BASE_FONT_SIZE + 2
LABEL_FONT_SIZE = BASE_FONT_SIZE
TICK_FONT_SIZE = BASE_FONT_SIZE - 1
ANNOTATION_FONT_SIZE = BASE_FONT_SIZE
REFERENCE_MARKER_COLOR = "tab:red"
REFERENCE_MARKER_SIZE = 70
REFERENCE_MARKER_EDGE = "black"
INFERENCE_MARKER_SIZE = 16
FIG_BG_COLOR = "#f5f7fb"
CARD_FACE_COLOR = "#ffffff"
CARD_EDGE_COLOR = "#dce3ed"
ACCENT_COLOR = "#4c6ef5"
ACCENT_SECONDARY = "#62b5f0"
NEUTRAL_TEXT_COLOR = "#4a5568"
PROGRESS_BG_COLOR = "#e4e9f2"
PROGRESS_FILL_COLOR = ACCENT_COLOR
plt.rcParams.update({
    "font.size": BASE_FONT_SIZE,
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.labelsize": LABEL_FONT_SIZE,
    "xtick.labelsize": TICK_FONT_SIZE,
    "ytick.labelsize": TICK_FONT_SIZE,
})

from evaluate_model import (  # noqa: E402
    AsymmetricCroCo3DStereo,
    CryoMultiViewDataset,
    DatasetTransform,
    MRCFile,
    inf,
    regularize_volume,
    run_evaluation_and_recon,
    select_ref_view_by_conf,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture intermediate CryoFastAR reconstruction volumes and pose distributions."
    )
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the model checkpoint (.pth).")
    parser.add_argument("--root-path", type=str, required=True, help="Root directory of the dataset.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on, e.g. 'cuda' or 'cpu'.")
    parser.add_argument("--use-amp", action="store_true", help="Enable automatic mixed precision.")
    parser.add_argument("--augment", action="store_true", help="Enable evaluation-time augmentation.")
    parser.add_argument("--snr", type=float, default=0.05, help="SNR value used when --augment is set.")
    parser.add_argument("--num-views", type=int, default=64, help="Number of input views per scene.")
    parser.add_argument("--reference-views", type=int, default=1, help="Number of reference views.")
    parser.add_argument(
        "--reference-offset",
        type=int,
        default=0,
        help="Manual reference view offset (used when --ref-mode manual).",
    )
    parser.add_argument(
        "--ref-mode",
        type=str,
        choices=["manual", "auto", "random"],
        default="manual",
        help="Strategy to select the reference view.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for the DataLoader.")
    parser.add_argument("--resolution", type=int, default=128, help="Volume resolution (D).")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--gt-image", action="store_true", help="Use ground-truth projection images.")
    parser.add_argument("--gt-pose", action="store_true", help="Use ground-truth poses.")
    parser.add_argument("--reg-pose", action="store_true", help="Regularize poses using Kabsch alignment.")
    parser.add_argument("--apix", type=float, default=None, help="Ångström per pixel for volume export.")
    parser.add_argument("--real", action="store_true", help="Flag indicating evaluation on a real dataset.")
    parser.add_argument("--no-per-scene", action="store_true", help="Treat dataset as global rather than per-scene.")
    parser.add_argument("--out-dir", type=str, default="results/progress", help="Directory to store outputs.")
    parser.add_argument("--snapshot-interval", type=int, default=5, help="Record every N reconstructed views.")
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=0,
        help="Maximum number of intermediate volumes to keep (0 means unlimited).",
    )
    parser.add_argument(
        "--render-mode",
        choices=["none", "slices"],
        default="slices",
        help="Visualization mode for rendered PNGs.",
    )
    parser.add_argument(
        "--pose-source",
        choices=["pred", "gt"],
        default="pred",
        help="Use predicted or ground-truth poses when accumulating pose distributions.",
    )
    parser.add_argument(
        "--pose-heatmap-metric",
        choices=["loss3d", "rot", "rot_fro", "shift", "density"],
        default="loss3d",
        help="Color encodes this metric on the pose heatmap ('loss3d' or 'rot' averages per bin, 'density' shows counts).",
    )
    parser.add_argument(
        "--heatmap-vmin",
        type=float,
        default=None,
        help="Optional lower bound for the heatmap colorbar.",
    )
    parser.add_argument(
        "--heatmap-vmax",
        type=float,
        default=None,
        help="Optional upper bound for the heatmap colorbar.",
    )
    parser.add_argument(
        "--no-snapshot-regularize",
        action="store_false",
        dest="snapshot_regularize",
        help="Disable Fourier regularization for intermediate renders (default keeps it on for sharper views).",
    )
    parser.set_defaults(snapshot_regularize=True)
    parser.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="Frames per second for the progress video.",
    )
    parser.add_argument(
        "--no-video",
        action="store_false",
        dest="make_video",
        help="Skip generating the final progress video.",
    )
    parser.set_defaults(make_video=True)
    parser.add_argument(
        "--reg-weight",
        type=float,
        default=10.0,
        help="Regularization weight for intermediate volume normalization.",
    )
    return parser.parse_args()


def rotation_to_dirs(rot: torch.Tensor) -> np.ndarray:
    """
    Project rotation matrices to orientation vectors (camera forward axis).
    """
    if rot is None:
        return np.zeros((0, 3), dtype=np.float32)
    if isinstance(rot, torch.Tensor):
        rot_np = rot.detach().cpu().numpy()
    else:
        rot_np = np.asarray(rot)
    forward_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    dirs = np.einsum("bij,j->bi", rot_np, forward_axis)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return dirs / norms


def dirs_to_heatmap(
    dirs: np.ndarray,
    values: Optional[np.ndarray] = None,
    bins_lat: int = 90,
    bins_lon: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert orientation vectors to an equirectangular histogram. When values are
    provided, compute the mean value per bin; otherwise return counts.
    """
    if dirs.size == 0:
        shape = (bins_lat, bins_lon)
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    z = np.clip(dirs[:, 2], -1.0, 1.0)
    theta = np.arccos(z)  # polar angle [0, pi]
    phi = np.arctan2(dirs[:, 1], dirs[:, 0])  # azimuth [-pi, pi]

    lat_idx = np.clip((theta / math.pi * bins_lat).astype(int), 0, bins_lat - 1)
    lon_idx = np.clip(((phi + math.pi) / (2 * math.pi) * bins_lon).astype(int), 0, bins_lon - 1)

    heat = np.zeros((bins_lat, bins_lon), dtype=np.float64)
    counts = np.zeros((bins_lat, bins_lon), dtype=np.float64)

    if values is None:
        np.add.at(heat, (lat_idx, lon_idx), 1.0)
        counts = heat.copy()
    else:
        values = values.astype(np.float64)
        if values.shape[0] != dirs.shape[0]:
            raise ValueError(f"Pose metric length {values.shape[0]} does not match pose count {dirs.shape[0]}")
        np.add.at(heat, (lat_idx, lon_idx), values)
        np.add.at(counts, (lat_idx, lon_idx), 1.0)
        mask = counts > 0
        heat[mask] /= counts[mask]

    return heat.astype(np.float32), counts.astype(np.float32)


def translation_to_heatmap(
    positions: np.ndarray,
    values: Optional[np.ndarray] = None,
    bins: int = 64,
) -> tuple[np.ndarray, np.ndarray, float]:
    if positions.size == 0:
        return np.zeros((bins, bins), dtype=np.float32), np.zeros((bins, bins), dtype=np.float32), 1.0

    limits = float(np.max(np.abs(positions)))
    if limits <= 0:
        limits = 1.0
    range_spec = [[-limits, limits], [-limits, limits]]

    if values is None:
        hist, xedges, yedges = np.histogram2d(
            positions[:, 0],
            positions[:, 1],
            bins=bins,
            range=range_spec,
        )
        counts = hist.copy()
    else:
        hist, xedges, yedges = np.histogram2d(
            positions[:, 0],
            positions[:, 1],
            bins=bins,
            range=range_spec,
            weights=values,
        )
        counts = np.histogram2d(
            positions[:, 0],
            positions[:, 1],
            bins=bins,
            range=range_spec,
        )[0]
        mask = counts > 0
        hist[mask] /= counts[mask]

    return hist.astype(np.float32), counts.astype(np.float32), limits


class VolumeProgressRecorder:
    """
    Callback for run_evaluation_and_recon that stores intermediate volumes
    and optional visualizations.
    """

    def __init__(
        self,
        output_dir: Path,
        interval: int,
        max_snapshots: Optional[int],
        render_mode: str,
        pose_source: str,
        pose_metric: str,
        heatmap_vmin: Optional[float],
        heatmap_vmax: Optional[float],
        snapshot_regularize: bool,
        reg_weight: float,
        apix: Optional[float],
        header_text: str,
        video_fps: int,
        make_video: bool,
        thumbnail_limit: int = 10,
        scatter_window: Optional[int] = None,
    ) -> None:
        self.output_dir = output_dir
        self.interval = max(1, interval)
        self.max_snapshots = max_snapshots if max_snapshots and max_snapshots > 0 else None
        self.render_mode = render_mode
        self.pose_source = pose_source
        self.pose_metric = pose_metric
        self.heatmap_vmin = heatmap_vmin
        self.heatmap_vmax = heatmap_vmax
        self.snapshot_regularize = snapshot_regularize
        self.reg_weight = reg_weight
        self.apix = apix
        self.header_text = header_text
        self.video_fps = video_fps
        self.make_video = make_video
        self.thumbnail_limit = thumbnail_limit
        self.scatter_window = scatter_window

        self.render_dir = self.output_dir / "renders"
        self.meta_path = self.output_dir / "progress.csv"
        self.video_path = self.output_dir / "progress.mp4"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.render_mode != "none":
            self.render_dir.mkdir(parents=True, exist_ok=True)

        self.reference_dirs: List[np.ndarray] = []
        self.inference_dirs: List[np.ndarray] = []
        self.inference_metrics: List[np.ndarray] = []
        self.recent_reference_dirs: List[np.ndarray] = []
        self.recent_inference_dirs: List[np.ndarray] = []
        self.recent_inference_metrics: List[np.ndarray] = []
        self.translation_samples: List[dict] = []
        self.last_thumbnails: Optional[np.ndarray] = None
        self.timeline: List[dict] = []

    def __call__(self, volume: torch.Tensor, counts: torch.Tensor, metadata: dict) -> None:
        self._accumulate_pose(metadata)
        self._accumulate_translation(metadata)
        self._accumulate_thumbnails(metadata)

        if metadata["global_view_index"] % self.interval != 0:
            return
        if self.max_snapshots is not None and len(self.timeline) >= self.max_snapshots:
            return

        snapshot_index = len(self.timeline)

        volume_clone = volume.detach().clone()
        counts_clone = counts.detach().clone()
        if counts_clone.sum().item() <= 0:
            edge = volume_clone.shape[0] - 1
            volume_cpu = torch.zeros((edge, edge, edge), dtype=torch.float32)
        elif self.snapshot_regularize:
            volume_cpu = regularize_volume(
                volume_clone,
                counts_clone,
                self.reg_weight,
            ).to(torch.float32)
        else:
            denom = counts_clone.clone()
            denom[denom == 0] = 1
            volume_cpu = (volume_clone / denom)[0:-1, 0:-1, 0:-1].detach().cpu().to(torch.float32)
        vol_np = volume_cpu.numpy()

        render_path = None
        if self.render_mode == "slices":
            render_path = self.render_dir / f"render_{metadata['global_view_index']:06d}.png"
            render_metadata = dict(metadata)
            render_metadata["snapshot_index"] = snapshot_index
            self._render_snapshot(vol_np, render_metadata, render_path)

        pose_count = int(sum(arr.shape[0] for arr in self.inference_dirs))
        translation_count = int(
            sum(sample["pos"].shape[0] for sample in self.translation_samples)
        )
        entry = {
            "snapshot_index": snapshot_index,
            "global_view_index": metadata["global_view_index"],
            "batch_index": metadata["batch_index"],
            "view_index": metadata["view_index"],
            "time_elapsed": metadata["time_elapsed"],
            "slices_processed": metadata["slices_processed"],
            "images_processed": metadata.get("images_processed"),
            "images_total": metadata.get("images_total"),
            "batch_size": metadata["batch_size"],
            "num_pose_samples": pose_count,
            "num_translation_samples": translation_count,
            "volume_path": "",
            "render_path": str(render_path.relative_to(self.output_dir)) if render_path else "",
            "sample_ids": self._format_sample_ids(metadata.get("sample_ids")),
        }
        self.timeline.append(entry)

    def _accumulate_pose(self, metadata: dict) -> None:
        ref_dirs = metadata.get("reference_dirs")
        if ref_dirs is not None:
            if isinstance(ref_dirs, torch.Tensor):
                ref_np = ref_dirs.detach().cpu().numpy()
            else:
                ref_np = np.asarray(ref_dirs)
            ref_np = ref_np.reshape(-1, 3)
            if ref_np.size > 0:
                ref_np = ref_np.astype(np.float32)
                self.reference_dirs.append(ref_np)
                self._append_with_limit(self.recent_reference_dirs, ref_np, self.scatter_window)

        pose_tensor = (
            metadata.get("pred_poses")
            if self.pose_source == "pred"
            else metadata.get("gt_poses")
        )
        if pose_tensor is None:
            return
        dirs = rotation_to_dirs(pose_tensor)
        if dirs.size == 0:
            return
        metric_values = self._extract_pose_metric(metadata, dirs.shape[0])
        if metric_values is not None and metric_values.shape[0] < dirs.shape[0]:
            dirs = dirs[: metric_values.shape[0]]
            metric_values = metric_values[: dirs.shape[0]]
        dirs = dirs.astype(np.float32)
        self.inference_dirs.append(dirs)
        self._append_with_limit(self.recent_inference_dirs, dirs, self.scatter_window)
        if metric_values is not None and metric_values.size > 0:
            metric_values = metric_values.astype(np.float32)
            self.inference_metrics.append(metric_values)
            self._append_with_limit(self.recent_inference_metrics, metric_values, self.scatter_window)

    def _accumulate_translation(self, metadata: dict) -> None:
        positions = metadata.get("translation_positions")
        errors = metadata.get("trans_error")
        if positions is None or errors is None:
            return
        if isinstance(positions, torch.Tensor):
            pos_np = positions.detach().cpu().numpy()
        else:
            pos_np = np.asarray(positions)
        if isinstance(errors, torch.Tensor):
            err_np = errors.detach().cpu().numpy()
        else:
            err_np = np.asarray(errors)
        pos_np = pos_np.reshape(-1, 2)
        err_np = err_np.reshape(-1)
        if pos_np.size == 0 or err_np.size == 0:
            return
        min_len = min(pos_np.shape[0], err_np.shape[0])
        if min_len == 0:
            return
        sample = {
            "pos": pos_np[:min_len].astype(np.float32),
            "err": err_np[:min_len].astype(np.float32),
        }
        self.translation_samples.append(sample)

    def _accumulate_thumbnails(self, metadata: dict) -> None:
        thumbs = metadata.get("thumbnails")
        if thumbs is None:
            return
        if isinstance(thumbs, torch.Tensor):
            thumbs_np = thumbs.detach().cpu().numpy()
        else:
            thumbs_np = np.asarray(thumbs)
        if thumbs_np.ndim != 3 or thumbs_np.shape[0] == 0:
            return
        limit = min(self.thumbnail_limit, thumbs_np.shape[0])
        self.last_thumbnails = thumbs_np[:limit].astype(np.float32)

    def _make_thumbnail_strip(self) -> np.ndarray:
        if self.last_thumbnails is None or self.last_thumbnails.size == 0:
            return np.zeros((1, 1), dtype=np.float32)
        images = self.last_thumbnails
        height = max(img.shape[0] for img in images)
        pad = 2
        pieces = []
        for idx, img in enumerate(images):
            if img.shape[0] != height:
                pad_top = (height - img.shape[0]) // 2
                pad_bottom = height - img.shape[0] - pad_top
                img = np.pad(img, ((pad_top, pad_bottom), (0, 0)), mode="constant", constant_values=0.0)
            pieces.append(img)
            if idx < len(images) - 1:
                pieces.append(np.ones((height, pad), dtype=np.float32))
        strip = np.concatenate(pieces, axis=1)
        return strip

    def _append_with_limit(
        self,
        buffer: List[np.ndarray],
        new_item: Optional[np.ndarray],
        limit: Optional[int],
    ) -> None:
        if new_item is None:
            return
        buffer.append(new_item)
        if limit is None or limit <= 0:
            return
        total = sum(arr.shape[0] for arr in buffer)
        while buffer and total > limit:
            removed = buffer.pop(0)
            total -= removed.shape[0]

    def _extract_pose_metric(self, metadata: dict, expected_len: int) -> Optional[np.ndarray]:
        if self.pose_metric == "density":
            return None
        if self.pose_metric == "loss3d":
            key = "loss3d"
        elif self.pose_metric == "rot":
            key = "rot_angle_deg"
        elif self.pose_metric == "rot_fro":
            key = "rot_fro_norm"
        else:
            key = "trans_error"
        values = metadata.get(key)
        if values is None:
            return None
        if isinstance(values, torch.Tensor):
            values_np = values.detach().cpu().numpy()
        else:
            values_np = np.asarray(values)
        values_np = values_np.reshape(-1)
        if values_np.size == 0:
            return None
        if values_np.size != expected_len:
            min_len = min(values_np.size, expected_len)
            values_np = values_np[:min_len]
        return values_np.astype(np.float32)

    def _heatmap_label(self) -> str:
        if self.pose_metric == "loss3d":
            return "Mean 3D loss"
        if self.pose_metric == "rot":
            return "Mean rotation error (deg)"
        if self.pose_metric == "rot_fro":
            return "Mean rotation F-norm"
        if self.pose_metric == "shift":
            return "Mean 2D shift error"
        return "Pose density"

    def _render_snapshot(self, volume: np.ndarray, metadata: dict, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(16, 9))
        fig.patch.set_facecolor(FIG_BG_COLOR)
        gs = fig.add_gridspec(3, 4, height_ratios=[0.35, 1.0, 1.0], hspace=0.32, wspace=0.3)

        mid = tuple(dim // 2 for dim in volume.shape)

        thumb_ax = fig.add_subplot(gs[0, :])
        thumb_ax.set_facecolor(CARD_FACE_COLOR)
        for spine in thumb_ax.spines.values():
            spine.set_visible(False)
        thumb_ax.set_title("Sampled input views", fontsize=TITLE_FONT_SIZE, loc="left", pad=6, color=NEUTRAL_TEXT_COLOR)
        if self.last_thumbnails is not None and self.last_thumbnails.size > 0:
            strip = self._make_thumbnail_strip()
            thumb_ax.imshow(strip, cmap="gray", aspect="equal", interpolation="nearest")
            thumb_ax.set_axis_off()
        else:
            thumb_ax.set_axis_off()
            thumb_ax.text(
                0.5,
                0.5,
                "No thumbnails",
                transform=thumb_ax.transAxes,
                ha="center",
                va="center",
                fontsize=ANNOTATION_FONT_SIZE,
                color="gray",
            )

        xy_ax = fig.add_subplot(gs[1, 0])
        xz_ax = fig.add_subplot(gs[1, 1])
        yz_ax = fig.add_subplot(gs[1, 2])
        mip_ax = fig.add_subplot(gs[1, 3])
        for ax in (xy_ax, xz_ax, yz_ax, mip_ax):
            ax.set_facecolor(CARD_FACE_COLOR)
            for spine in ax.spines.values():
                spine.set_edgecolor(CARD_EDGE_COLOR)

        slices = (
            ("XY slice", volume[mid[0], :, :], xy_ax),
            ("XZ slice", volume[:, mid[1], :], xz_ax),
            ("YZ slice", volume[:, :, mid[2]], yz_ax),
        )
        for title, slc, ax in slices:
            ax.imshow(slc, cmap="viridis")
            ax.set_title(title, fontsize=TITLE_FONT_SIZE, color=NEUTRAL_TEXT_COLOR, pad=6)
            ax.axis("off")

        # abs_volume = np.abs(volume)
        snapshot_index = metadata.get("snapshot_index", metadata.get("global_view_index", 0))
        rotation_angle = snapshot_index % 360
        rotated_volume = rotate(volume, rotation_angle, axes=(1, 2), reshape=False, order=1, mode="nearest")
        mip = rotated_volume.sum(axis=2)
        if mip.max() > 0:
            mip_norm = (mip - mip.min()) / (mip.max() - mip.min() + 1e-6)
        else:
            mip_norm = mip
        mip_ax.imshow(mip_norm, cmap="inferno")
        mip_ax.set_title("Rotating projection", fontsize=TITLE_FONT_SIZE, color=NEUTRAL_TEXT_COLOR, pad=6)
        mip_ax.axis("off")

        scatter_ax = fig.add_subplot(gs[2, 0], projection="3d")
        scatter_ax.set_facecolor(CARD_FACE_COLOR)
        scatter_ax.set_title("Pose distribution", fontsize=TITLE_FONT_SIZE, color=NEUTRAL_TEXT_COLOR, pad=6)
        scatter_ax.set_xlim(-1, 1)
        scatter_ax.set_ylim(-1, 1)
        scatter_ax.set_zlim(-1, 1)
        scatter_ax.set_box_aspect([1, 1, 1])
        scatter_ax.set_xticks([])
        scatter_ax.set_yticks([])
        scatter_ax.set_zticks([])

        reference_dirs_all = (
            np.concatenate(self.reference_dirs, axis=0) if self.reference_dirs else np.zeros((0, 3))
        )
        inference_dirs_all = (
            np.concatenate(self.inference_dirs, axis=0) if self.inference_dirs else np.zeros((0, 3))
        )
        inference_metrics_all = (
            np.concatenate(self.inference_metrics, axis=0)
            if self.inference_metrics
            else np.zeros((0,), dtype=np.float32)
        )

        reference_dirs_recent = (
            np.concatenate(self.recent_reference_dirs, axis=0)
            if self.recent_reference_dirs
            else reference_dirs_all
        )
        inference_dirs_recent = (
            np.concatenate(self.recent_inference_dirs, axis=0)
            if self.recent_inference_dirs
            else inference_dirs_all
        )
        inference_metrics_recent = (
            np.concatenate(self.recent_inference_metrics, axis=0)
            if self.recent_inference_metrics
            else inference_metrics_all
        )

        scatter_limit = self.scatter_window or inference_dirs_recent.shape[0]
        if scatter_limit and reference_dirs_recent.shape[0] > scatter_limit:
            idx = np.linspace(0, reference_dirs_recent.shape[0] - 1, scatter_limit).astype(int)
            reference_dirs_recent = reference_dirs_recent[idx]
        if scatter_limit and inference_dirs_recent.shape[0] > scatter_limit:
            idx = np.linspace(0, inference_dirs_recent.shape[0] - 1, scatter_limit).astype(int)
            inference_dirs_recent = inference_dirs_recent[idx]
            if inference_metrics_recent.size > 0:
                inference_metrics_recent = inference_metrics_recent[idx]

        if reference_dirs_recent.size > 0:
            scatter_ax.scatter(
                reference_dirs_recent[:, 0],
                reference_dirs_recent[:, 1],
                reference_dirs_recent[:, 2],
                c=REFERENCE_MARKER_COLOR,
                marker="*",
                s=REFERENCE_MARKER_SIZE,
                alpha=0.9,
                edgecolors=REFERENCE_MARKER_EDGE,
                linewidths=0.5,
                label="Reference",
            )
        cmap_metric = plt.get_cmap("turbo")
        use_metric_colors = self.pose_metric != "density" and inference_metrics_recent.size > 0
        vmin_metric = self.heatmap_vmin
        vmax_metric = self.heatmap_vmax
        if inference_dirs_recent.size > 0:
            if use_metric_colors:
                data_min = float(inference_metrics_recent.min())
                data_max = float(inference_metrics_recent.max())
                if vmin_metric is None:
                    vmin_metric = data_min
                if vmax_metric is None:
                    vmax_metric = data_max
                if vmax_metric <= vmin_metric:
                    vmax_metric = vmin_metric + 1e-6
                scatter_ax.scatter(
                    inference_dirs_recent[:, 0],
                    inference_dirs_recent[:, 1],
                    inference_dirs_recent[:, 2],
                    c=inference_metrics_recent,
                    cmap=cmap_metric,
                    vmin=vmin_metric,
                    vmax=vmax_metric,
                    s=INFERENCE_MARKER_SIZE,
                    alpha=0.8,
                    label="Inference",
                )
            else:
                colors = np.linspace(0, 1, inference_dirs_recent.shape[0])
                scatter_ax.scatter(
                    inference_dirs_recent[:, 0],
                    inference_dirs_recent[:, 1],
                    inference_dirs_recent[:, 2],
                    c=colors,
                    cmap="plasma",
                    s=INFERENCE_MARKER_SIZE,
                    alpha=0.75,
                    label="Inference",
                )
        if scatter_ax.collections:
            scatter_ax.legend(loc="upper right", fontsize=BASE_FONT_SIZE)
        u = np.linspace(0, 2 * math.pi, 30)
        v = np.linspace(0, math.pi, 15)
        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones_like(u), np.cos(v))
        scatter_ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="#d0d5e5", linewidth=0.3, alpha=0.5)

        bins_lat, bins_lon = 36, 72
        orientation_ax = fig.add_subplot(gs[2, 1:3])
        orientation_ax.set_facecolor(CARD_FACE_COLOR)
        if use_metric_colors:
            heatmap, counts = dirs_to_heatmap(
                inference_dirs_all, inference_metrics_all, bins_lat=bins_lat, bins_lon=bins_lon
            )
            cmap_orient = cmap_metric
        else:
            heatmap, counts = dirs_to_heatmap(
                inference_dirs_all, None, bins_lat=bins_lat, bins_lon=bins_lon
            )
            cmap_orient = "viridis"
        display = heatmap.copy()
        display[counts == 0] = np.nan
        vmin_plot = self.heatmap_vmin if self.heatmap_vmin is not None else 0.0
        vmax_plot = self.heatmap_vmax if self.heatmap_vmax is not None else 5.0
        phi_extent = [-math.pi, math.pi]
        theta_extent = [0, math.pi]
        im_orient = orientation_ax.imshow(
            display,
            cmap=cmap_orient,
            origin="lower",
            aspect="auto",
            extent=[phi_extent[0], phi_extent[1], theta_extent[0], theta_extent[1]],
            interpolation="nearest",
            vmin=vmin_plot,
            vmax=vmax_plot,
        )
        orientation_ax.set_title("Orientation error", fontsize=TITLE_FONT_SIZE)
        orientation_ax.set_xlabel("Azimuth (φ)")
        orientation_ax.set_ylabel("Polar (θ)")
        orientation_ax.set_xlim(phi_extent)
        orientation_ax.set_ylim(theta_extent)
        xticks = np.linspace(-math.pi, math.pi, 7)
        xtick_labels = ["-π", "-2π/3", "-π/3", "0", "π/3", "2π/3", "π"]
        orientation_ax.set_xticks(xticks)
        orientation_ax.set_xticklabels(xtick_labels)
        yticks = np.linspace(0, math.pi, 5)
        ytick_labels = ["0", "π/4", "π/2", "3π/4", "π"]
        orientation_ax.set_yticks(yticks)
        orientation_ax.set_yticklabels(ytick_labels)
        orientation_ax.grid(color="white", alpha=0.3, linewidth=0.3)
        for tick in orientation_ax.get_xticklabels() + orientation_ax.get_yticklabels():
            tick.set_color(NEUTRAL_TEXT_COLOR)
        if reference_dirs_all.size > 0:
            ref_theta = np.arccos(np.clip(reference_dirs_all[:, 2], -1.0, 1.0))
            ref_phi = np.arctan2(reference_dirs_all[:, 1], reference_dirs_all[:, 0])
            orientation_ax.scatter(
                ref_phi,
                ref_theta,
                c=REFERENCE_MARKER_COLOR,
                marker="*",
                edgecolors=REFERENCE_MARKER_EDGE,
                linewidths=0.5,
                s=REFERENCE_MARKER_SIZE,
                alpha=0.9,
            )
        cbar_orient = fig.colorbar(im_orient, ax=orientation_ax, fraction=0.046, pad=0.02)
        cbar_orient.set_label(self._heatmap_label(), fontsize=LABEL_FONT_SIZE)
        cbar_orient.ax.yaxis.label.set_color(NEUTRAL_TEXT_COLOR)
        for tick in cbar_orient.ax.get_yticklabels():
            tick.set_color(NEUTRAL_TEXT_COLOR)
        if np.isfinite(display).any():
            orientation_mean = float(np.nanmean(display))
            if not math.isfinite(orientation_mean):
                orientation_mean = 0.0
        else:
            orientation_mean = 0.0

        trans_positions = (
            np.concatenate([sample["pos"] for sample in self.translation_samples], axis=0)
            if self.translation_samples
            else np.zeros((0, 2), dtype=np.float32)
        )
        trans_errors = (
            np.concatenate([sample["err"] for sample in self.translation_samples], axis=0)
            if self.translation_samples
            else np.zeros((0,), dtype=np.float32)
        )

        shift_ax = fig.add_subplot(gs[2, 3])
        shift_ax.set_facecolor(CARD_FACE_COLOR)
        shift_heatmap, shift_counts, shift_limit = translation_to_heatmap(trans_positions, trans_errors, bins=48)
        shift_display = shift_heatmap.copy()
        if shift_counts.any():
            shift_display[shift_counts == 0] = np.nan
        shift_vmin = 0.0
        shift_vmax = 3.0
        extent = [-shift_limit, shift_limit, -shift_limit, shift_limit]
        im_shift = shift_ax.imshow(
            shift_display,
            cmap="magma",
            origin="lower",
            aspect="equal",
            extent=extent,
            vmin=shift_vmin,
            vmax=shift_vmax,
        )
        shift_ax.set_title("2D shift error", fontsize=TITLE_FONT_SIZE)
        shift_ax.set_xlabel("Δx")
        shift_ax.set_ylabel("Δy")
        shift_ax.set_xticks([extent[0], 0, extent[1]])
        shift_ax.set_yticks([extent[2], 0, extent[3]])
        shift_ax.grid(color="white", alpha=0.3, linewidth=0.3)
        for tick in shift_ax.get_xticklabels() + shift_ax.get_yticklabels():
            tick.set_color(NEUTRAL_TEXT_COLOR)
        cbar_shift = fig.colorbar(im_shift, ax=shift_ax, fraction=0.046, pad=0.02)
        cbar_shift.ax.yaxis.label.set_color(NEUTRAL_TEXT_COLOR)
        for tick in cbar_shift.ax.get_yticklabels():
            tick.set_color(NEUTRAL_TEXT_COLOR)
        if np.isfinite(shift_display).any():
            shift_mean = float(np.nanmean(shift_display))
            if not math.isfinite(shift_mean):
                shift_mean = 0.0
        else:
            shift_mean = 0.0

        header_ax = fig.add_axes([0.05, 0.88, 0.9, 0.08])
        header_ax.set_facecolor(FIG_BG_COLOR)
        header_ax.axis("off")
        header_ax.text(0.0, 0.75, self.header_text, fontweight="bold", fontsize=TITLE_FONT_SIZE + 2, color=ACCENT_COLOR)
        time_text = f"Elapsed {metadata['time_elapsed']:.1f}s"
        images_total = metadata.get("images_total") or 0
        images_processed = metadata.get("images_processed") or 0
        if images_total > 0:
            progress_ratio = min(max(images_processed / images_total, 0.0), 1.0)
            images_text = f"Images {images_processed}/{images_total} ({progress_ratio*100:.1f}%)"
        else:
            progress_ratio = 0.0
            images_text = "Images N/A"
        header_ax.text(0.0, 0.35, time_text, fontsize=LABEL_FONT_SIZE, color=NEUTRAL_TEXT_COLOR)
        header_ax.text(0.32, 0.35, images_text, fontsize=LABEL_FONT_SIZE, color=NEUTRAL_TEXT_COLOR)
        header_ax.text(0.64, 0.35, f"mean orientation error {orientation_mean:.2f}", fontsize=LABEL_FONT_SIZE, color=ACCENT_COLOR)
        header_ax.text(0.86, 0.35, f"mean shift error {shift_mean:.2f}", fontsize=LABEL_FONT_SIZE, color=ACCENT_COLOR)
        bar_x, bar_y, bar_w, bar_h = 0.0, 0.05, 0.9, 0.18
        header_ax.add_patch(
            patches.FancyBboxPatch(
                (bar_x, bar_y),
                bar_w,
                bar_h,
                boxstyle="round,pad=0.01,rounding_size=0.01",
                linewidth=0,
                facecolor=PROGRESS_BG_COLOR,
            )
        )
        header_ax.add_patch(
            patches.FancyBboxPatch(
                (bar_x, bar_y),
                bar_w * progress_ratio,
                bar_h,
                boxstyle="round,pad=0.01,rounding_size=0.01",
                linewidth=0,
                facecolor=PROGRESS_FILL_COLOR,
            )
        )

        fig.subplots_adjust(left=0.05, right=0.98, top=0.85, bottom=0.06, wspace=0.25, hspace=0.28)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    @staticmethod
    def _format_sample_ids(sample_ids) -> str:
        if sample_ids is None:
            return ""
        if isinstance(sample_ids, torch.Tensor):
            sample_ids = sample_ids.detach().cpu().numpy()
        sample_ids = np.asarray(sample_ids).reshape(-1)
        return "|".join(str(int(x)) for x in sample_ids)

    def finalize(self) -> None:
        if not self.timeline:
            return
        with self.meta_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(self.timeline[0].keys()))
            writer.writeheader()
            for row in self.timeline:
                writer.writerow(row)

        if self.reference_dirs:
            np.save(
                self.output_dir / "pose_reference_dirs.npy",
                np.concatenate(self.reference_dirs, axis=0).astype(np.float32),
            )
        if self.inference_dirs:
            np.save(
                self.output_dir / "pose_inference_dirs.npy",
                np.concatenate(self.inference_dirs, axis=0).astype(np.float32),
            )
        if self.inference_metrics:
            np.save(
                self.output_dir / f"pose_{self.pose_metric}_values.npy",
                np.concatenate(self.inference_metrics, axis=0).astype(np.float32),
            )

        if self.translation_samples:
            trans_positions = np.concatenate([sample["pos"] for sample in self.translation_samples], axis=0)
            trans_errors = np.concatenate([sample["err"] for sample in self.translation_samples], axis=0)
            np.save(self.output_dir / "translation_positions.npy", trans_positions.astype(np.float32))
            np.save(self.output_dir / "translation_errors.npy", trans_errors.astype(np.float32))

        if self.make_video and self.render_mode != "none" and any(self.render_dir.glob("render_*.png")):
            self._export_video()

    def _export_video(self) -> None:

        video_path = Path(self.video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)

        pattern = "render_*.png"
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(self.video_fps),
            "-pattern_type",
            "glob",
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "progress.mp4",
        ]
        try:
            subprocess.run(cmd, check=True, cwd=self.render_dir)
            print(f"Saved progress video to {self.video_path}")
        except Exception as exc:
            print(f"Warning: failed to export video ({exc})")


def build_dataset(args: argparse.Namespace) -> CryoMultiViewDataset:
    transform = (
        DatasetTransform(
            snr=args.snr,
            ctf=True,
            ctf_param_path="ctf_params_filtered.csv",
            shift=10.0,
            repeat=1,
            random_apix=False,
        )
        if args.augment
        else None
    )

    dataset = CryoMultiViewDataset(
        root_path=args.root_path,
        transform=transform,
        mode=args.mode,
        num_views=args.num_views,
        per_scene=args.per_scene,
        apix_path="apix.csv",
        apix=args.apix,
        load_draco=False,
    )
    return dataset


def build_model(args: argparse.Namespace, device: torch.device) -> AsymmetricCroCo3DStereo:
    model = AsymmetricCroCo3DStereo(
        pos_embed="RoPE100",
        img_size=(128, 128),
        head_type="fourier",
        output_mode="pts3d",
        depth_mode=("exp", -inf, inf),
        conf_mode=("exp", 1, inf),
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_type="croco",
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12,
        num_views=args.num_views,
        has_neck=True,
    )
    checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device)
    model.eval()
    return model


def determine_reference_offset(
    model: AsymmetricCroCo3DStereo,
    dataset: CryoMultiViewDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> int:
    if args.ref_mode == "manual":
        return args.reference_offset
    if args.ref_mode == "auto":
        print("Automatically selecting reference view …")
        ref_offset, ref_conf = select_ref_view_by_conf(model, dataset, args, args.num_views, device)
        print(f"Selected reference index {ref_offset} (mean confidence {ref_conf:.4f})")
        return ref_offset
    if args.ref_mode == "random":
        ref_offset = np.random.randint(0, len(dataset) - args.reference_views)
        print(f"Random reference index {ref_offset}")
        return ref_offset
    raise ValueError(f"Unsupported reference mode: {args.ref_mode}")


def main() -> None:
    args = parse_args()

    args.per_scene = not args.no_per_scene
    args.mode = "std" if args.real else "real"

    device = torch.device(args.device)

    dataset = build_dataset(args)
    model = build_model(args, device)
    ref_offset = determine_reference_offset(model, dataset, args, device)

    out_dir = Path(args.out_dir)
    data_tag = "Real data" if args.real else "Synthetic data"
    header_components = [
        data_tag,
        f"num_views={args.num_views}",
        f"reference_views={args.reference_views}",
        "thumbnails: 10 sampled views",
    ]
    if args.augment:
        header_components.append(f"snr={args.snr}")
    header_text = " | ".join(header_components)
    recorder = VolumeProgressRecorder(
        output_dir=out_dir,
        interval=args.snapshot_interval,
        max_snapshots=args.max_snapshots,
        render_mode=args.render_mode,
        pose_source=args.pose_source,
        pose_metric=args.pose_heatmap_metric,
        heatmap_vmin=args.heatmap_vmin,
        heatmap_vmax=args.heatmap_vmax,
        snapshot_regularize=args.snapshot_regularize,
        reg_weight=args.reg_weight,
        apix=dataset.cached_apix if args.per_scene else args.apix,
        header_text=header_text,
        video_fps=args.video_fps,
        make_video=args.make_video,
        scatter_window=args.num_views,
    )

    print("Starting evaluation with progress capture …")
    metrics, gt_poses, pred_poses, pred_trans, pred_ids, final_volume = run_evaluation_and_recon(
        model,
        dataset,
        args,
        args.num_views,
        ref_offset,
        device,
        progress_callback=recorder,
    )

    recorder.finalize()

    final_volume_path = out_dir / "final_volume.mrc"
    final_volume_np = final_volume.detach().to(torch.float32).cpu().numpy()
    MRCFile.write(str(final_volume_path), final_volume_np, Apix=dataset.cached_apix if args.per_scene else args.apix)

    pose_source = pred_poses if args.pose_source == "pred" else gt_poses
    pose_output_path = out_dir / f"{args.pose_source}_poses.npy"
    np.save(pose_output_path, pose_source.detach().cpu().numpy())

    metrics_path = out_dir / "metrics_summary.txt"
    with metrics_path.open("w") as f:
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.ndimension() == 0:
                    f.write(f"{key}: {value.item():.6f}\n")
            elif isinstance(value, (float, int)):
                f.write(f"{key}: {value:.6f}\n")

    print(f"Finished. Final volume saved to {final_volume_path}")
    print(f"Progress timeline written to {recorder.meta_path}")
    print(f"Pose directions stored at {out_dir / 'pose_directions.npy'}")


if __name__ == "__main__":
    main()
