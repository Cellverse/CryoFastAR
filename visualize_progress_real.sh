#!/usr/bin/env bash

set -euo pipefail

DATE=$(date +%Y%m%d_%H%M%S)
ROOT_PATH=${1:-data/real/10180}
OUT_DIR=${2:-results/${DATE}_$(basename "${ROOT_PATH}")_progress}

python visualize_reconstruction_progress.py \
  --ckpt-path checkpoints/real.pth \
  --root-path "${ROOT_PATH}" \
  --device cuda \
  --use-amp \
  --real \
  --num-views 128 \
  --reference-views 1 \
  --ref-mode auto \
  --batch-size 4 \
  --resolution 128 \
  --num-workers 4 \
  --gt-image \
  --reg-pose \
  --out-dir "${OUT_DIR}" \
  --snapshot-interval 128 \
  --pose-heatmap-metric rot_fro \
  --render-mode slices
