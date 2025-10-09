import numpy as np
import torch
from typing import Optional
from torch.fft import fft2, ifft2


def compute_chi_from_drgn_ctfs(target_D, ctfs):
    D = ctfs[0, 0]
    Apix = ctfs[0, 1]
    true_Apix = D / target_D * Apix

    freq_pix_1d = torch.arange(-0.5, 0.5, 1 / target_D)
    freq_pix_1d_safe = freq_pix_1d[:target_D]
    x, y = torch.meshgrid(freq_pix_1d_safe, freq_pix_1d_safe, indexing='ij')
    freqs = torch.stack([x, y],dim=-1) / true_Apix # all true_apix are the same

    dfu, dfv, dfang_deg, volt, cs, w, phase_shift_deg = torch.split(torch.from_numpy(ctfs[...,2:]),1,dim=1)

    DFs, dfxxs, dfxys = defocus_polar_to_cartesian(
        df1_A=dfu, 
        df2_A=dfv, 
        df_angle_rad=dfang_deg * torch.pi / 180
    )
    chi = compute_ctf_chi_2D_batch(freqs, volt, cs, w, DFs, dfxxs, dfxys, phase_shift_deg).reshape(-1, target_D, target_D)
    return chi


def compute_safe_freqs(n_pixels, psize):
    freq_pix_1d = np.arange(-0.5, 0.5, 1 / n_pixels)
    freq_pix_1d_safe = freq_pix_1d[:n_pixels]
    x, y = np.meshgrid(freq_pix_1d_safe, freq_pix_1d_safe)
    rho = np.sqrt(x**2 + y**2)
    angles_rad = np.arctan2(y, x)
    freq_mag_2d = rho / psize
    return freq_mag_2d, angles_rad


def torch_compute_safe_freqs(n_pixels, psize):
    freq_pix_1d = torch.arange(-0.5, 0.5, 1 / n_pixels)
    freq_pix_1d_safe = freq_pix_1d[:n_pixels]
    x, y = torch.meshgrid(freq_pix_1d_safe, freq_pix_1d_safe, indexing='ij')
    rho = torch.sqrt(x**2 + y**2)
    angles_rad = torch.atan2(y, x)
    freq_mag_2d = rho / psize
    return freq_mag_2d, angles_rad


def generate_random_ctf_params(batch_size,
                                df_min=15000,
                                df_max=20000,
                                df_diff_min=100,
                                df_diff_max=500,
                                df_ang_min=0,
                                df_ang_max=360,
                                volt=300,
                                cs=2.7,
                                w=0.1,
                                phase_shift=0,):
    dfs = np.random.uniform(low=df_min, high=df_max, size=(batch_size, 1, 1))
    df_diff = np.random.uniform(low=df_diff_min, high=df_diff_max, size=(batch_size, 1, 1))
    df_ang_deg = np.random.uniform(low=df_ang_min, high=df_ang_max, size=(batch_size, 1, 1))
    dfu = dfs - df_diff / 2
    dfv = dfs + df_diff / 2
    return dfu, dfv, df_ang_deg, np.ones((batch_size, 1, 1)) * volt, np.ones((batch_size, 1, 1)) * cs, np.ones((batch_size, 1, 1)) * w, np.ones((batch_size, 1, 1)) * phase_shift

def defocus_polar_to_cartesian(df1_A, df2_A, df_angle_rad):
    DF = (df1_A + df2_A) / 2
    df = (df1_A - df2_A) / 2
    dfxx = torch.cos(2*df_angle_rad)*df
    dfxy = torch.sin(2*df_angle_rad)*df
    return DF.clone().detach(), dfxx.clone().detach(), dfxy.clone().detach()

def get_chi_consts(akv, csmm, wgh, phase_shift):
    av = akv * 1e3
    e = 12.2643247 / torch.sqrt(av + av**2 * 0.978466e-6)
    CTF2 = torch.pi * e
    CTF4 = - torch.pi * e**3 * (csmm*1e7) / 2.0
    chi_offset = phase_shift - torch.arccos(wgh)
    return CTF2, CTF4, chi_offset

def compute_ctf_chi_2D(freqs, akv, csmm, wgh, DF,  dfxx, dfxy, phase_shift):
    CTF2, CTF4, chi_offset = get_chi_consts(akv, csmm, wgh, phase_shift)
    fx = freqs[...,0]
    fy = freqs[...,1]
    f2 = (fx*fx + fy*fy)
    chi = CTF2*((DF+dfxx)*fx*fx + 2*(dfxy)*fx*fy + (DF-dfxx)*fy*fy) + CTF4*f2*f2 + chi_offset
    return chi

def compute_ctf_chi_2D_batch(freqs, akv, csmm, wgh, DF,  dfxx, dfxy, phase_shift):
    CTF2, CTF4, chi_offset = get_chi_consts(akv, csmm, wgh, phase_shift)
    fx = freqs[...,0]
    fy = freqs[...,1]
    f2 = (fx*fx + fy*fy)

    CTF2 = CTF2.unsqueeze(-1) # (N, 1, 1)
    CTF4 = CTF4.unsqueeze(-1) # (N, 1, 1)
    chi_offset = chi_offset.unsqueeze(-1) # (N, 1, 1)

    DF = DF.unsqueeze(-1) # (N, 1, 1)
    dfxy = dfxy.unsqueeze(-1) # (N, 1, 1)
    dfxx = dfxx.unsqueeze(-1) # (N, 1, 1)

    fx = fx.unsqueeze(0) # (1, 128, 128)
    fy = fy.unsqueeze(0) # (1, 128, 128)
    f2 = f2.unsqueeze(0) # (1, 128, 128)

    chi = CTF2*((DF+dfxx)*fx*fx + 2*(dfxy)*fx*fy + (DF-dfxx)*fy*fy) + CTF4*f2*f2 + chi_offset
    return chi


def compute_ctf(
    freqs: torch.Tensor,
    dfu: torch.Tensor,
    dfv: torch.Tensor,
    dfang: torch.Tensor,
    volt: torch.Tensor,
    cs: torch.Tensor,
    w: torch.Tensor,
    phase_shift: Optional[torch.Tensor] = None,
    scalefactor: Optional[torch.Tensor] = None,
    bfactor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the 2D CTF

    Input:
        freqs: Nx2 array of 2D spatial frequencies
        dfu: DefocusU (Angstrom)
        dfv: DefocusV (Angstrom)
        dfang: DefocusAngle (degrees)
        volt: accelerating voltage (kV)
        cs: spherical aberration (mm)
        w: amplitude contrast ratio
        phase_shift: degrees
        scalefactor : scale factor
        bfactor: envelope fcn B-factor (Angstrom^2)
    """
    # convert units
    volt = volt * 1000
    cs = cs * 10**7
    dfang = dfang * np.pi / 180
    if phase_shift is None:
        phase_shift = torch.tensor(0)
    phase_shift = phase_shift * np.pi / 180

    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
    lam = 12.2639 / torch.sqrt(volt + 0.97845e-6 * volt**2)
    x = freqs[..., 0]
    y = freqs[..., 1]
    ang = torch.arctan2(y, x)
    s2 = x**2 + y**2
    df = 0.5 * (dfu + dfv + (dfu - dfv) * torch.cos(2 * (ang - dfang)))
    gamma = (
        2 * torch.pi * (-0.5 * df * lam * s2 + 0.25 * cs * lam**3 * s2**2)
        - phase_shift
    )
    ctf = torch.sqrt(1 - w**2) * torch.sin(gamma) - w * torch.cos(gamma)
    if scalefactor is not None:
        ctf *= scalefactor
    if bfactor is not None:
        ctf *= torch.exp(-bfactor / 4 * s2)
    return ctf


def torch_compute_ctf(s, a, dfu, dfv, dfang_deg, kv, cs, w, phase=0, bf=0):
    s = s[None, ...] # add batch dimension
    a = a[None, ...] # add batch dimension
    kv = kv * 1e3
    cs = cs * 1e7
    lamb = 12.2643247 / torch.sqrt(kv * (1.0 + kv * 0.978466e-6))

    dfang_deg = torch.deg2rad(dfang_deg)
    def_avg = -(dfu + dfv) * 0.5
    def_dev = -(dfu - dfv) * 0.5
    k1 = np.pi / 2.0 * 2 * lamb
    k2 = np.pi / 2.0 * cs * lamb**3
    k3 = torch.sqrt(1 - w**2)
    k4 = bf / 4.0  # B-factor, follows RELION convention.
    k5 = torch.deg2rad(phase)  # Phase shift.
    s_2 = s**2
    s_4 = s_2**2
    dZ = def_avg + def_dev * (torch.cos(2 * (a - dfang_deg)))
    gamma = (k1 * dZ * s_2) + (k2 * s_4) - k5
    ctf = -(k3 * torch.sin(gamma) - w * torch.cos(gamma))
    if bf != 0:  # Enforce envelope.
        ctf *= torch.exp(-k4 * s_2)
    return ctf

def compute_ctf_drgn(
    freqs: torch.Tensor,
    dfu: torch.Tensor,
    dfv: torch.Tensor,
    dfang: torch.Tensor,
    volt: torch.Tensor,
    cs: torch.Tensor,
    w: torch.Tensor,
    phase_shift: torch.Tensor = torch.Tensor([0]),
    bfactor = None,
) -> torch.Tensor:
    """
    Compute the 2D CTF

    Input:
        freqs (np.ndarray) Nx2 or BxNx2 tensor of 2D spatial frequencies
        dfu (float or Bx1 tensor): DefocusU (Angstrom)
        dfv (float or Bx1 tensor): DefocusV (Angstrom)
        dfang (float or Bx1 tensor): DefocusAngle (degrees)
        volt (float or Bx1 tensor): accelerating voltage (kV)
        cs (float or Bx1 tensor): spherical aberration (mm)
        w (float or Bx1 tensor): amplitude contrast ratio
        phase_shift (float or Bx1 tensor): degrees
        bfactor (float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
    """
    # print(freqs)
    assert freqs.shape[-1] == 2
    # convert units
    volt = volt * 1000
    cs = cs * 10**7
    dfang = dfang * np.pi / 180
    phase_shift = phase_shift * np.pi / 180

    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
    lam = 12.2639 / (volt + 0.97845e-6 * volt**2) ** 0.5
    x = freqs[..., 0]
    y = freqs[..., 1]
    ang = torch.atan2(y, x)
    s2 = x**2 + y**2
    df = 0.5 * (dfu + dfv + (dfu - dfv) * torch.cos(2 * (ang - dfang)))
    gamma = (
        2 * np.pi * (-0.5 * df * lam * s2 + 0.25 * cs * lam**3 * s2**2)
        - phase_shift
    )
    ctf = (1 - w**2) ** 0.5 * torch.sin(gamma) - w * torch.cos(gamma)
    if bfactor is not None:
        ctf *= torch.exp(-bfactor / 4 * s2)
    return ctf

def fft2_center(image):
    """Perform centered 2D FFT."""
    return torch.fft.fftshift(fft2(torch.fft.ifftshift(image)))

def ifft2_center(frequency):
    """Perform centered 2D IFFT."""
    return torch.fft.fftshift(ifft2(torch.fft.ifftshift(frequency)))

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