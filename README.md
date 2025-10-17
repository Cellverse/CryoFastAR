# CryoFastAR: Fast Cryo-EM Ab Initio Reconstruction Made Easy

Accepted by ICCV 2025

[Jiakai Zhang](https://jiakai-zhang.github.io), [Shouchen Zhou](https://scholar.google.com/citations?user=6D6uyxAAAAAJ&hl=en), [Haizhao Dai](https://scholar.google.com/citations?hl=en&user=gQy4BcYAAAAJ), [Xinhang Liu](https://xinhangliu.com/), [Peihao Wang](https://peihaowang.github.io/), [Zhiwen Fan](https://zhiwenfan.github.io/), [Yuan Pei](https://orcid.org/0000-0003-4065-2540), [Jingyi Yu](https://www.yu-jingyi.com/)

1 ShanghaiTech University, 2 Cellverse Co., Ltd., 3 HKUST, 4 UT Austin

<p align="center">
  <img src="images/teaser.png", width=800>
</p>

### This repo is under construction.

### Installation

1. Clone repo

   ```bash
   cd CryoFastAR
   ```
2. create conda envionment and activate

```bash
conda create -n cryofastar python=3.11
conda activate cryofastar
pip install -r requirement.txt
```

### Inference (Experimental Data)

Due to the limited experimental data in public, our model only supports four types of experimental data from cryoDRGN's datasets. We split them into the training data (30,000 for each) and unseen remained test data. To play with them, please first download the [cryofastar-ckpts.zip](https://drive.google.com/file/d/1uiYss_brxczMM5FiyTZV7a5hAYZyuFHV/view?usp=sharing), and unzip it to `checkpoints/`, then we provide one example for real data and one example for synthetic data to reproduce our results in our paper, please download [data.zip](https://drive.google.com/file/d/1ryYszBurL6_vyQ9VDcArP6oGk1PNFElZ/view?usp=sharing), and unzip it to `data/`

Simply running below commands can generate results under `results/` including the reconstructed volume as well as the angluar histograms.

```
bash evaluate_model_real.sh
bash evaluate_model_synthetic.sh
```

### Reconstruction Progress Visualization

To inspect how the 3D volume grows while the model consumes new views, use the helper below:

```bash
python visualize_reconstruction_progress.py \
    --ckpt-path checkpoints/cryofastar.pth \
    --root-path data/your_dataset \
    --num-views 64 \
    --out-dir results/progress_example
```

This script runs a single evaluation pass but saves intermediate diagnostics in `results/progress_example/`:
- `renders/render_XXXX.png`: top-row input thumbnails, volume slices, a max-intensity projection, the pose scatter, an orientation heat map, and the 2D shift error map (disable via `--render-mode none`).
- `progress.csv`: timeline metadata (time, processed images vs. total, sampled ids, etc.).
- `pose_reference_dirs.npy` / `pose_inference_dirs.npy`: accumulated pose orientations for reference and inference views; optional `pose_<metric>_values.npy` matches the chosen pose heat-map metric.
- `translation_positions.npy` / `translation_errors.npy`: shift statistics gathered during the run.
- `progress.mp4`: automatically generated 16:9 video (10 fps by default) that strings the renders together; skip via `--no-video` or adjust speed with `--video-fps`.

Use `--snapshot-interval` to control how often a snapshot is written, pass `--max-snapshots` if you want to cap how many are kept (the default keeps all), and switch to `--pose-source gt` to visualise the ground-truth pose distribution instead of predictions.
`--pose-heatmap-metric` now supports `loss3d`, rotation angle (`rot`), F-norm rotation error (`rot_fro`), 2D shift error (`shift`), or plain density; combine with `--heatmap-vmin/--heatmap-vmax` to clamp the color bar. Turn on `--snapshot-regularize` if you prefer the slower but sharper Fourier-domain filtering for each saved volume.

### Citation

```
@inproceedings{zhang2025cryofastar,
  author    = {Zhang,Jiakai and Zhou, Shouchen and Dai, Haizhao and Liu, Xinhang and Wang, Peihao and Fan, Zhiwen and Pei, Yuan and Yu, Jingyi},
  title     = {CryoFastAR: Fast Cryo-EM Ab Initio Reconstruction Made Easy},
  booktitle = {Proc. ICCV},
  year      = {2025},
}
```
