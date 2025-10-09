import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import torch
import torchvision.transforms.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dust3r.datasets.ctf import generate_random_ctf_params, compute_ctf, compute_ctf_chi_2D, defocus_polar_to_cartesian
from utils import fft2_center, ifft2_center
import pandas as pd
import random
from scipy.stats import poisson


class DatasetTransform:
    def __init__(self, snr=None, ctf=False, shift=None, repeat=2, ctf_param_path='ctf_params.csv', invert=False, random_apix=True):
        self.snr = snr
        self.ctf = ctf
        self.shift = shift
        self.repeat = repeat
        self.invert = invert
        self.random_apix = random_apix
        if os.path.exists(ctf_param_path) and self.ctf:
            self.ctf_params = pd.read_csv(ctf_param_path)
            print(f'load {ctf_param_path}')
        else:
            self.ctf_params = None


    def add_noise(self, image):
        # image : (1, w, h) or (w, h) are both ok
        sidelen, apix = 128, 1.0
        snr = self.snr
        noise_std = np.sqrt(np.var(image, axis=(-2, -1), keepdims=True) / snr)
        expand_noise_std = np.tile(noise_std, (1, sidelen, sidelen))
        image = np.random.normal(image, expand_noise_std)

        return image # (1, w, h)
    

    
    def sample_poisson_noise(self, image):
        """
        在包含负值的CTF调制图像上添加具有给定SNR的泊松噪声。

        :param image: 浮点数CTF调制图像 (numpy array)
        :param snr_db: 目标信噪比 (dB)
        :return: 添加噪声后的图像
        """
        # 处理负值：方法 1 (Shift，使最小值为 0)
        min_val = np.min(image)
        shifted_image = image - min_val  # 使所有值 >= 0

        # 计算原始信号功率
        P_signal = np.mean(shifted_image**2)

        # 计算目标噪声功率
        P_noise = P_signal # / self.snr # snr = 1

        # 归一化到 [0, 255] 以进行泊松采样
        scale_factor = np.max(shifted_image)  
        scaled_image = shifted_image / scale_factor * 255  
        poisson_noise = poisson.rvs(scaled_image) / 255.0 * scale_factor  

        # 计算实际噪声功率并调整
        P_generated_noise = np.mean((poisson_noise - shifted_image)**2)
        noise_adjusted = (poisson_noise - shifted_image) * np.sqrt(P_noise / P_generated_noise)

        return noise_adjusted


    def apply_ctf(self, image, apix=1.0):
        # image : (w, h)
        sidelen = 128

        freq_pix_1d = torch.arange(-0.5, 0.5, 1 / sidelen)
        freq_pix_1d_safe = freq_pix_1d[:sidelen]
        x, y = torch.meshgrid(freq_pix_1d_safe, freq_pix_1d_safe, indexing='ij')
        freqs = torch.stack([x, y],dim=-1) / apix
        if self.ctf_params is None:
            ctf_param = generate_random_ctf_params(batch_size=1)
            # ctf = compute_ctf(freqs_mag, angles_rad, *ctf_params).reshape(batch, sidelen, sidelen)
            # ctf_corrupted_fourier_images = ctf * fft2_center(image)
            # image = ifft2_center(ctf_corrupted_fourier_images).real
        else:
            ctf_param = self.ctf_params.sample(n=1)
            dfu, dfv, dfang_deg, volt, cs, w, phase_shift_deg = torch.from_numpy(ctf_param.values[0])

        # def compute_ctf_chi_2D(freqs, akv, csmm, wgh, DF,  dfxx, dfxy, phase_shift):
        DFs, dfxxs, dfxys = defocus_polar_to_cartesian(
            df1_A=dfu,
            df2_A=dfv,
            df_angle_rad=dfang_deg * torch.pi / 180
        )
        phase_shift = phase_shift_deg * torch.pi / 180
        # w1 = torch.sqrt(1 - w**2)
        chi = compute_ctf_chi_2D(freqs, volt, cs, w, DFs, dfxxs, dfxys, phase_shift).reshape(1, sidelen, sidelen)
        # ctf = compute_ctf(freqs, dfu, dfv, dfang_deg, volt, cs, w, phase_shift_deg).reshape(1, sidelen, sidelen)
        ctf = torch.cos(chi)
        ctf_corrupted_fourier_images = ctf * fft2_center(image)
        image = ifft2_center(ctf_corrupted_fourier_images).real

        return image, chi, ctf_param # (1, w, h)



    def apply_shift(self, image, trans):
        # image : (1, w, h)
        image = F.affine(torch.from_numpy(image), 0, translate=(trans[1], trans[0]), scale=1, shear=(0, 0))
        return image.numpy() # (1, w, h)


    def __call__(self, views, apix=None):
        sidelen = 128
        if apix is None and self.random_apix:
            apix = random.uniform(1.0, 6.0)
        elif apix is None and not self.random_apix:
            apix = 1.0
            print('Warning: apix is not provided, set to 1.0')
        elif apix is not None and self.random_apix:
            apix = apix + random.uniform(-1.0, 1.0)
        else:
            apix = apix

        for view in views:
            images = []

            same_image = view['img'] # (1, w, h)
            dtype = same_image.dtype
            device = same_image.device

            same_image = same_image.cpu().numpy() # tensor to numpy.array

            # apply ctf
            if self.ctf:
                # randomly shift image in x and y direction within 10 pixels in positive and negative direction
                same_image, chi, ctf_param = self.apply_ctf(same_image, apix=apix)
                view['chi'] = chi # (1, w, h)
                view['ctf_param'] = np.concatenate(([sidelen, apix], ctf_param.values[0])) # sidelen, apix, dfu, dfv, dfang_deg, volt, cs, w, phase_shift_deg

            if self.invert:
                same_image = -same_image

            if self.shift is not None:
                trans = (np.random.rand(2) - 0.5) * self.shift * 2 # (1, 2) ranged from [-shift, shift]
                same_image = self.apply_shift(same_image, trans)
                trans = np.concatenate([trans, np.array([0])]).astype(np.float32).reshape(1, -1) # (1, 3)
                trans = trans / same_image.shape[-1] # normalize
                view['trans'] = trans # (1, 3)

            if self.repeat > 0:
                view['clean_img'] = torch.from_numpy(same_image)
                for _ in range(self.repeat):
                    image = same_image.copy()
                    if self.snr is not None: # add random noise each time
                        # TODO: add poisson noise as another kind of data augmentation
                        # noise_p = self.sample_poisson_noise(image) # (1, w, h)
                        # image = image + noise_p
                        image = self.add_noise(image) # (1, w, h)
                    images.append(image)
                images = np.stack(images, axis=0).squeeze(1) # (repeat, w, h)
                # Update the output
                view['img'] = torch.tensor(images, dtype=dtype, device=device)

        return views


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform dataset')
    parser.add_argument('--add_noise', action='store_true', help='Add noise to dataset')

    args = parser.parse_args()

    image_path = '/media/cellverse/share013/zhoushch/cryoDuster/data/N1_proj50000_snrNone_single/1xvi/particle_stack.npy'
    images = np.load(image_path)
    image = images[0]

    # image: (w, h) -> (1, w, h)
    # image = image[None, ...]

    # image = transform(image, True, 1.0)
    # show image
    plt.gray()
    plt.imshow(image)
    plt.show()