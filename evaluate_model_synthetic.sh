# Current date
DATE=$(date +%Y%m%d)

python evaluate_model.py --ckpt_path checkpoints/synthetic.pth \
                         --root_path data/synthetic/spliceosome \
                         --device cuda \
                         --use_amp \
                         --augment \
                         --snr_list 0.05,0.1,1.0,10.0 \
                         --view_numbers 128 \
                         --batch_size 4 \
                         --resolution 128 \
                         --reference_views 1 \
                         --num_workers 1 \
                         --gt_image \
                         --reg_pose \
                         --out_dir "results/${DATE}_spliceosome_synthetic" \