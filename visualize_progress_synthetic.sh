#!/usr/bin/env bash

set -euo pipefail

DATE=$(date +%Y%m%d_%H%M%S)

python visualize_reconstruction_progress.py \
  --ckpt-path checkpoints/synthetic.pth \
  --root-path data/synthetic/spliceosome \
  --device cuda \
  --use-amp \
  --augment \
  --snr 0.05 \
  --num-views 128 \
  --reference-views 1 \
  --batch-size 4 \
  --resolution 128 \
  --num-workers 4 \
  --gt-image \
  --reg-pose \
  --out-dir "results/${DATE}_spliceosome_progress" \
  --snapshot-interval 128 \
  --pose-heatmap-metric rot_fro \
  --render-mode slices
