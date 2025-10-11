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

### Citation

```
@inproceedings{zhang2025cryofastar,
  author    = {Zhang,Jiakai and Zhou, Shouchen and Dai, Haizhao and Liu, Xinhang and Wang, Peihao and Fan, Zhiwen and Pei, Yuan and Yu, Jingyi},
  title     = {CryoFastAR: Fast Cryo-EM Ab Initio Reconstruction Made Easy},
  booktitle = {Proc. ICCV},
  year      = {2025},
}
```
