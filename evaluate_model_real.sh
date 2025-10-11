#!/usr/bin/env bash
set -euo pipefail

# Common args
BASE_ARGS=(
  --ckpt_path checkpoints/real.pth
  --device cuda
  --use_amp
  --snr_list 0.1
  --batch_size 8
  --resolution 128
  --reference_views 1
  --num_workers 16
  --view_numbers 128
  --gt_image
  --reg_pose
  --real
  --ref-mode auto
)

# Datasets list
DATASETS=(
  "data/real/10028/"
  # "data/real/10049/"
  # "data/real/10076/"
  # "data/real/10180/"
)

# Current date
DATE=$(date +%Y%m%d)

for root in "${DATASETS[@]}"; do
  name=$(basename "$root")   # e.g., 10028
  out_dir="results/${DATE}_${name}"
  echo ">>> Running dataset $name, saving to $out_dir"
  python evaluate_model.py "${BASE_ARGS[@]}" --root_path "$root" --out_dir "$out_dir"
done
