"""
Unified Normalizing Flow training code
Supports configuration via JSON config file or command-line arguments

Usage examples:
  # Using config file (recommended)
  python train.py --config config_slope.json
  python train.py --config config_conti.json
  python train.py --config config_single.json

  # Command-line arguments (config file takes higher priority)
  python train.py --dataset-type concat --epochs 2000
"""

import json
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.interpolate import interp1d
import os
import argparse


# ==========================================
# Utility: write to both stdout and file (for package training log)
# ==========================================
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()


# ==========================================
# 0-A. LIGO physical noise model
# ==========================================

class LIGONoiseModel:
    """Physical noise model for the LIGO frequency band

    Supports two data sources:
      - 'cc_spectrum': O3 cross-correlation spectrum (C_O1_O2_O3.dat)
        Format: freq[Hz]  C(f)  sigma(f), where sigma is already in ΩGW units
      - 'aplus_asd':  A+ design sensitivity (AplusDesign.txt)
        Format: freq[Hz]  ASD[1/sqrt(Hz)], needs conversion to ΩGW equivalent

    Noise injection modes:
      - 'physical': Add N(0, σ_Ω(f)) in linear ΩGW space, then convert back to log10
                    Most physically accurate; when signal is far below the noise floor,
                    observed value ≈ |noise|
      - 'logscale': Add frequency-weighted noise in log10(ΩGW) space
                    Preserves signal structure while introducing realistic frequency dependence

    Args:
        noise_file:   Path to the noise data file
        noise_type:   'cc_spectrum' or 'aplus_asd'
        target_log10_freq: LIGO target frequency grid, log10(f/Hz), shape (L,)
        noise_scale:  Overall noise scaling factor (1.0 = original amplitude)
        injection_mode: 'physical' or 'logscale'
        logscale_base: Base noise standard deviation in log10 space for logscale mode
    """

    def __init__(self, noise_file, noise_type='cc_spectrum',
                 target_log10_freq=None, noise_scale=1.0,
                 injection_mode='physical', logscale_base=1.0):
        self.noise_scale = noise_scale
        self.injection_mode = injection_mode
        self.logscale_base = logscale_base

        if target_log10_freq is None:
            raise ValueError("target_log10_freq is required")

        self.target_log10_freq = np.asarray(target_log10_freq, dtype=np.float64)

        # Load and interpolate to the target grid
        if noise_type == 'cc_spectrum':
            self.sigma_omega = self._load_cc_spectrum(noise_file)
        elif noise_type == 'aplus_asd':
            self.sigma_omega = self._load_aplus_asd(noise_file)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        # Apply scaling
        self.sigma_omega = self.sigma_omega * self.noise_scale

        # logscale mode: log-compressed frequency weights
        # Use normalized log10(σ) to avoid linear ratios spanning several orders of magnitude
        log_sigma = np.log10(np.maximum(self.sigma_omega, 1e-30))
        log_min = log_sigma.min()
        log_max = log_sigma.max()
        if log_max > log_min:
            # Map to [1, max_ratio], uniformly in log space
            max_ratio = 10.0  # Weight of the noisiest frequency is 10x that of the quietest
            self.freq_weight = 1.0 + (max_ratio - 1.0) * (log_sigma - log_min) / (log_max - log_min)
        else:
            self.freq_weight = np.ones_like(self.sigma_omega)
        self.freq_weight = self.freq_weight.astype(np.float32)

        print(f"[LIGO Noise] type={noise_type}, scale={noise_scale}, "
              f"mode={injection_mode}")
        print(f"  σ_Ω range: [{self.sigma_omega.min():.3e}, "
              f"{self.sigma_omega.max():.3e}]")
        if injection_mode == 'logscale':
            print(f"  logscale_base={logscale_base}, "
                  f"freq_weight range: [{self.freq_weight.min():.3f}, "
                  f"{self.freq_weight.max():.3f}]")

    # ------------------------------------------------------------------
    def _load_cc_spectrum(self, filepath):
        """Load σ_Ω(f) from CC spectrum, bin-averaged to the target grid.

        Processing:
        1. inf / negative values → fill via interpolation in log space
        2. Bin-average: take RMS mean of original points within each target grid bin,
           and divide by √N (averaging over N independent bins reduces σ)
        """
        data = np.loadtxt(filepath, skiprows=1)
        freq_hz = data[:, 0]
        sigma_raw = data[:, 2]

        # Clean inf / nan / non-positive values
        valid = np.isfinite(sigma_raw) & (sigma_raw > 0)
        log_sigma_interp = np.interp(
            np.log10(freq_hz),
            np.log10(freq_hz[valid]),
            np.log10(sigma_raw[valid])
        )
        sigma_clean = 10.0 ** log_sigma_interp

        # Bin-average to the target grid
        target_hz = 10.0 ** self.target_log10_freq
        n = len(target_hz)
        sigma_binned = np.zeros(n, dtype=np.float64)

        for i in range(n):
            # Geometric midpoints as bin edges (equally spaced in log space)
            if i == 0:
                f_lo = freq_hz.min()
            else:
                f_lo = np.sqrt(target_hz[i] * target_hz[i - 1])
            if i == n - 1:
                f_hi = freq_hz.max()
            else:
                f_hi = np.sqrt(target_hz[i] * target_hz[i + 1])

            mask = (freq_hz >= f_lo) & (freq_hz < f_hi)
            n_bins = mask.sum()

            if n_bins > 0:
                # RMS mean / √N (bin-averaging noise reduction)
                sigma_binned[i] = np.sqrt(np.mean(sigma_clean[mask] ** 2)) / np.sqrt(n_bins)
            else:
                sigma_binned[i] = 10.0 ** np.interp(
                    self.target_log10_freq[i],
                    np.log10(freq_hz[valid]),
                    np.log10(sigma_raw[valid])
                )

        return sigma_binned.astype(np.float32)

    # ------------------------------------------------------------------
    def _load_aplus_asd(self, filepath):
        """Compute equivalent σ_Ω(f) from A+ ASD and interpolate to the target grid.

        Uses the single-detector approximation: σ_Ω(f) ∝ f³ Sₙ(f)
        where Sₙ(f) = ASD²(f), coefficient = 2π²/(3H₀²)

        Note: This is a sensitivity proxy; the true CC search σ also depends on γ(f), T, Δf, etc.
        """
        data = np.loadtxt(filepath)
        freq_hz = data[:, 0]
        asd = data[:, 1]

        H0_si = 67.4e3 / 3.0857e22  # H₀ in units: Hz
        sigma_proxy = (2.0 * np.pi ** 2 / (3.0 * H0_si ** 2)) * freq_hz ** 3 * asd ** 2

        # Interpolate in log space to the target grid
        target_hz = 10.0 ** self.target_log10_freq
        interp_fn = interp1d(np.log10(freq_hz), np.log10(sigma_proxy),
                             kind='linear', fill_value='extrapolate')
        sigma_on_grid = 10.0 ** interp_fn(self.target_log10_freq)

        return sigma_on_grid.astype(np.float32)

    # ------------------------------------------------------------------
    def inject(self, log10_omega, log10_freq=None):
        """Inject noise into a single LIGO spectrum.

        Args:
            log10_omega: Clean log10(ΩGW), shape (L,)
            log10_freq:  log10(f/Hz), shape (L,), optional in physical mode
        Returns:
            Noisy log10(ΩGW), shape (L,)
        """
        if self.injection_mode == 'physical':
            return self._inject_physical(log10_omega)
        else:
            return self._inject_logscale(log10_omega)

    def _inject_physical(self, log10_omega):
        """Add Gaussian noise in linear ΩGW space → convert to log10.

        Ω_obs(f) = Ω_true(f) + N(0, σ_Ω(f))
        If Ω_obs < 0, take absolute value (cross-correlation estimator can be negative)
        """
        omega = np.float64(10.0) ** log10_omega.astype(np.float64)
        noise = np.random.normal(0.0, self.sigma_omega.astype(np.float64))
        omega_obs = omega + noise

        omega_obs = np.abs(omega_obs)
        omega_obs = np.maximum(omega_obs, 1e-300)
        return np.log10(omega_obs).astype(np.float32)

    def _inject_logscale(self, log10_omega):
        """Add frequency-weighted noise in log10 space.

        δ(log10 Ω)(f) = N(0, 1) × freq_weight(f) × logscale_base
        freq_weight is normalized so median=1; logscale_base controls the amplitude.
        """
        noise = np.random.normal(0.0, 1.0, log10_omega.shape).astype(np.float32)
        return log10_omega + noise * self.freq_weight * self.logscale_base


# ==========================================
# 0-B. PTA physical noise model
# ==========================================

class PTANoiseModel:
    """Physical noise model for the PTA frequency band (NANOGrav 15yr free spectrum)

    Data source: Zenodo 10344086  NANOGrav15yr/
      - freqs.npy            30 frequency bins (Hz), Δf = 1/T_obs ≈ 1.977e-9 Hz
      - log10rhogrid.npy     log10(ρ) grid (10000 equally spaced points, [-15.5, -1.0])
      - density_mock.npy     shape (1, 30, 10000), stores log(p(log10ρ_i | data))
                             from NGmock free spectrum (Li & Shapiro 2025, Sec.3)
                             — mock version, corresponding to design sensitivity
      - density.npy          shape (1, 30, 10000), same as above but from the real
                             NANOGrav 15yr observation — observed version

    Physical pathway (mode='density_mock' or 'density_obs', recommended):
      1. For each bin i ∈ [0, 14), sample from posterior p_i(log10ρ) → ρ_i (seconds)
      2. Convert to Ω_GW via Li & Shapiro 2025 Eq.(3):
             Ω_i(f_i) = (8π⁴/H₀²) · T_obs · f_i^5 · ρ_i²
      3. ★ Subtract the per-bin Ω mean (across the posterior) to obtain pure noise deviation:
             Ω_noise(f_i) = Ω_i_sampled - <Ω_i>_posterior
         Intuition: data_NG - μ_NG = Noise_NG. The posterior itself represents the
         total "signal + noise"; what we need is the noise component, so we subtract the mean.
      4. Inject in linear Ω space: Ω_obs = Ω_signal + Ω_noise, then convert back to log10

    Args:
        noise_dir:        Directory containing freqs.npy / log10rhogrid.npy / density{_mock}.npy
        mode:             'density_mock' → density_mock.npy (recommended, mock version)
                          'density_obs'  → density.npy (NANOGrav 15yr real observation)
                          'white' / 'posterior' (legacy)
        noise_scale:      Overall scaling factor applied to the Ω_noise deviation
                          (curriculum learning: 0=no noise, 1=full noise)
        n_bins_use:       Number of frequency bins to use (default 14, aligned with PTA dataset)
        per_sample_noise: True  → independently sample ρ for each training sample (default)
                          False → refresh once per batch, sharing the same set of ρ
        H0_fiducial:      Hubble constant for ρ→Ω conversion (km/s/Mpc), default 67.4 (Planck18)
        T_obs:            Observation duration (seconds). Default None → inferred from freqs.npy as 1/Δf
    """

    # mode → density filename mapping
    _DENSITY_FILE_MAP = {
        'density_mock': 'density_mock.npy',
        'density_obs':  'density.npy',
    }

    # H0 SI unit (s^-1) used in ρ→Ω conversion
    _KM_PER_MPC = 3.0856775814913673e19  # 1 Mpc in km
    # numpy 1.x: trapz; numpy 2.x: trapezoid
    _trapz = staticmethod(getattr(np, 'trapezoid', getattr(np, 'trapz', None)))

    def __init__(self, noise_dir,
                 mode='density_mock',
                 noise_scale=1.0,
                 n_bins_use=14,
                 per_sample_noise=True,
                 H0_fiducial=67.4,
                 T_obs=None):

        self.mode = mode
        self.noise_scale = float(noise_scale)
        self.per_sample_noise = bool(per_sample_noise)
        self.H0_fiducial = float(H0_fiducial)
        # Flag: density_mock and density_obs use the same physical pathway
        self.is_density_mode = mode in self._DENSITY_FILE_MAP

        freqs_all = np.load(os.path.join(noise_dir, 'freqs.npy'))         # (30,)
        log10rho  = np.load(os.path.join(noise_dir, 'log10rhogrid.npy'))  # (10000,)

        # Select density file based on mode
        if self.is_density_mode:
            density_filename = self._DENSITY_FILE_MAP[mode]
        else:
            # Legacy mode: default to density_mock.npy for computing sigma_per_bin
            density_filename = 'density_mock.npy'
        density_path = os.path.join(noise_dir, density_filename)
        if not os.path.isfile(density_path):
            raise FileNotFoundError(
                f"PTA density file not found: {density_path}\n"
                f"  mode='{mode}' expects file '{density_filename}' under {noise_dir}")
        density = np.load(density_path)                                    # (1,30,10000)

        # Compatible with multiple storage layouts
        if density.ndim == 3:
            density2d = density[0]                # (30, 10000)
        elif density.ndim == 2:
            density2d = density                   # (30, 10000) or (14, 10000)
        else:
            raise ValueError(f"density_mock.npy has unexpected dimensions: {density.shape}")

        self.n_bins = int(n_bins_use)
        if self.n_bins > density2d.shape[0]:
            raise ValueError(f"n_bins_use={self.n_bins} > density rows={density2d.shape[0]}")

        self.freqs    = freqs_all[:self.n_bins].astype(np.float64)        # (14,) Hz
        self.log10rho = log10rho.astype(np.float64)                       # (10000,)
        self.log_pdf  = density2d[:self.n_bins].astype(np.float64)        # (14, 10000)

        # T_obs: default inferred from equally-spaced freqs.npy (= 1/Δf)
        if T_obs is None:
            df = float(freqs_all[1] - freqs_all[0])
            self.T_obs = 1.0 / df
        else:
            self.T_obs = float(T_obs)

        # ρ→Ω conversion coefficient: K_i = (8π⁴/H₀²) T_obs f_i^5
        # Convert H0 to SI: H0_SI [s^-1] = H0 [km/s/Mpc] / (Mpc in km)
        H0_SI = self.H0_fiducial / self._KM_PER_MPC
        self.K_omega = (8.0 * np.pi**4 / H0_SI**2) * self.T_obs * self.freqs**5  # (14,)

        # ---- Precompute inverse-CDF table per bin (10000 points → direct np.interp sampling) ----
        # Subtract max in log space then exp for numerical stability
        pdf = np.exp(self.log_pdf - self.log_pdf.max(axis=1, keepdims=True))  # (14, G)
        # Normalize using trapezoidal rule
        norm = self._trapz(pdf, self.log10rho, axis=1)                          # (14,)
        pdf  = pdf / norm[:, None]
        # Discrete CDF
        cdf = np.cumsum(pdf, axis=1)
        cdf -= cdf[:, :1]
        cdf /= cdf[:, -1:].clip(min=1e-300)
        self._cdf = cdf                                                       # (14, G)

        # ---- ★ Precompute per-bin Ω posterior mean (to subtract mean and obtain pure noise deviation) ----
        # Physics: Noise_NG = data_NG - μ_NG. The posterior represents the total "signal + noise";
        # subtracting the cross-sample mean yields the additive pure noise component.
        # Computed via Monte Carlo at noise_scale=1; subsequent sampling multiplies by noise_scale².
        N_mc = 50000
        u_mc = np.random.uniform(0.0, 1.0, size=(N_mc, self.n_bins))
        log10_rho_mc = np.empty_like(u_mc)
        for i in range(self.n_bins):
            log10_rho_mc[:, i] = np.interp(u_mc[:, i], self._cdf[i], self.log10rho)
        omega_mc = self.K_omega[None, :] * (10.0 ** log10_rho_mc) ** 2
        self.omega_mean = omega_mc.mean(axis=0).astype(np.float64)            # (14,)

        # Also store a σ_log10rho for sanity checks (posterior width), only used in legacy mode
        mean = self._trapz(self.log10rho[None, :] * pdf, self.log10rho, axis=1)  # (14,)
        var  = self._trapz((self.log10rho[None, :] - mean[:, None])**2 * pdf,
                        self.log10rho, axis=1)
        self.sigma_per_bin = np.sqrt(np.maximum(var, 1e-30)).astype(np.float32) * self.noise_scale
        self.sigma_white = float(np.median(self.sigma_per_bin))

        # Report
        rho_lo = self.log10rho[(cdf >= 0.025).argmax(axis=1)]
        rho_hi = self.log10rho[(cdf >= 0.975).argmax(axis=1)]
        print(f"[PTA Noise] mode={mode}, per_sample={self.per_sample_noise}, "
              f"n_bins={self.n_bins}, T_obs={self.T_obs:.3e} s ({self.T_obs/3.15576e7:.2f} yr)")
        print(f"[PTA Noise]   freqs: [{self.freqs[0]:.3e}, {self.freqs[-1]:.3e}] Hz")
        print(f"[PTA Noise]   log10ρ 95% CR: bin0=[{rho_lo[0]:.2f},{rho_hi[0]:.2f}]  "
              f"bin{self.n_bins-1}=[{rho_lo[-1]:.2f},{rho_hi[-1]:.2f}]")
        if self.is_density_mode:
            # Report an equivalent Ω_noise order of magnitude (using per-bin ρ median)
            rho_med = self.log10rho[(cdf >= 0.5).argmax(axis=1)]
            omega_med = self.K_omega * (10.0**rho_med)**2
            print(f"[PTA Noise]   density file: {density_filename}")
            print(f"[PTA Noise]   Ω (posterior median) range: "
                  f"[{omega_med.min():.3e}, {omega_med.max():.3e}]")
            print(f"[PTA Noise]   Ω (posterior mean) range:   "
                  f"[{self.omega_mean.min():.3e}, {self.omega_mean.max():.3e}]")
            print(f"[PTA Noise]   ★ noise = sampled - mean (zero-mean deviation)")
        elif mode in ('white', 'posterior'):
            print(f"[PTA Noise]   [legacy] σ_log10Ω: white={self.sigma_white:.4f}, "
                  f"per-bin range=[{self.sigma_per_bin.min():.4f}, "
                  f"{self.sigma_per_bin.max():.4f}]")

    # ------------------------------------------------------------------
    # Core: sample log10ρ from posterior → Ω_noise (seconds²)
    # ------------------------------------------------------------------
    def sample_omega_noise(self, n_samples):
        """Sample N sets of pure noise deviation Ω_noise(f_i), shape (N, n_bins).

        ★ Key: returns (Ω_sampled - <Ω>_posterior), i.e., the zero-mean pure noise component,
        not the posterior values themselves. This allows correct superposition onto any
        theoretical signal:
            Ω_obs = Ω_signal + Ω_noise   (Ω_noise can be positive or negative)

        noise_scale acts on the deviation (curriculum-learning friendly):
            noise_scale=0 → zero noise; noise_scale=1 → full noise.

        If per_sample_noise=False, only one set is generated and broadcast to N.
        """
        n_draw = n_samples if self.per_sample_noise else 1
        u = np.random.uniform(0.0, 1.0, size=(n_draw, self.n_bins))
        log10_rho = np.empty_like(u)
        for i in range(self.n_bins):
            log10_rho[:, i] = np.interp(u[:, i], self._cdf[i], self.log10rho)
        # Ω_sampled at noise_scale=1
        omega_sampled = self.K_omega[None, :] * (10.0 ** log10_rho) ** 2  # (n_draw,14)
        # ★ Subtract mean to get pure noise deviation, then scale by noise_scale²
        omega_dev = (omega_sampled - self.omega_mean[None, :]) * (self.noise_scale ** 2)
        if not self.per_sample_noise:
            omega_dev = np.broadcast_to(omega_dev, (n_samples, self.n_bins)).copy()
        return omega_dev.astype(np.float64)

    def inject(self, log10_omega):
        """Single-sample injection (fallback, compatible with __getitem__ slow path).

        Args:
            log10_omega: shape (14,)  log10(Ω_signal)
        Returns:
            shape (14,) log10(Ω_obs) = log10(Ω_signal + Ω_noise)
        """
        if self.is_density_mode:
            omega_lin = 10.0 ** log10_omega.astype(np.float64)
            omega_noise = self.sample_omega_noise(1)[0]                      # (14,)
            omega_obs = np.maximum(omega_lin + omega_noise, 1e-300)
            return np.log10(omega_obs).astype(np.float32)
        # ---- Legacy modes (log-space additive Gaussian) ----
        if self.mode == 'white':
            noise = np.random.normal(0.0, self.sigma_white, log10_omega.shape)
        else:  # 'posterior'
            noise = np.random.normal(0.0, self.sigma_per_bin[:len(log10_omega)])
        return (log10_omega + noise).astype(np.float32)

    def validate_freqs(self, freq_log10_dataset, atol=0.02):
        """Check that the first 14 frequencies of the dataset align with freqs.npy (in log10 space)."""
        f_data = 10.0 ** np.asarray(freq_log10_dataset[:self.n_bins], dtype=np.float64)
        if not np.allclose(np.log10(f_data), np.log10(self.freqs), atol=atol):
            print(f"[PTA Noise] WARNING: dataset freqs vs noise freqs mismatch!")
            print(f"  data:  {f_data}")
            print(f"  noise: {self.freqs}")
        else:
            print(f"[PTA Noise]   freq alignment OK (max log10 diff "
                  f"{np.abs(np.log10(f_data)-np.log10(self.freqs)).max():.4f})")


# ==========================================
# 0-C. LISA physical noise model
# ==========================================

class LISANoiseModel:
    """Physical noise model for the LISA frequency band

    Supports two data sources:
      - 'analytical': Analytical PSD based on Robson+ 2019 (arXiv:1803.01944)
                      Parameters from LISA SciRD (OMS=15 pm/√Hz, acc=3 fm/s²/√Hz)
      - 'datafile':   External sensitivity curve file (freq[Hz], ASD[1/√Hz])

    Noise injection modes are the same as LIGONoiseModel (physical / logscale).

    Additional parameters:
        T_obs:       Observation duration [yr], used for Galactic foreground confusion noise (default 4 yr)
        include_confusion: Whether to include Galactic foreground confusion noise (default True)

    Args:
        target_log10_freq: LISA target frequency grid log10(f/Hz), shape (L,)
        noise_source:      'analytical' or file path
        noise_scale:       Overall scaling factor
        injection_mode:    'physical' or 'logscale'
        logscale_base:     Base σ in logscale mode
        T_obs:             Observation duration in years (affects confusion noise)
        include_confusion: Whether to include Galactic confusion noise
    """

    # Robson+2019 Table 1: Galactic confusion noise fitting parameters
    _CONFUSION_PARAMS = {
        0.5: {'alpha': 0.133, 'beta': 243., 'kappa': 482., 'gamma': 917., 'f_k': 2.58e-3},
        1.0: {'alpha': 0.171, 'beta': 292., 'kappa': 1020., 'gamma': 1680., 'f_k': 2.15e-3},
        2.0: {'alpha': 0.165, 'beta': 299., 'kappa': 611., 'gamma': 1340., 'f_k': 1.73e-3},
        4.0: {'alpha': 0.138, 'beta': -221., 'kappa': 521., 'gamma': 1680., 'f_k': 1.13e-3},
    }

    # Noise presets: directly correspond to the curves in LISA Definition Study Report Figure 7.1
    # Adjusted via underlying OMS / acc noise parameters, not post-hoc scaling factors.
    #
    # 'scird':  LISA-LCST-SGS-TN-001 (Babak+2021) official SciRD formula parameters
    #           OMS = 15 pm/√Hz, acc = 3 fm/s²/√Hz
    #           ≈ Allocation (green band) in Red Book Figure 7.1
    #           = mission requirement curve
    #
    # 'cbe':    Current Best Estimate
    #           OMS ≈ 7.9 pm/√Hz, acc ≈ 2.4 fm/s²/√Hz
    #           ≈ CBE (blue line) in Red Book Figure 7.1
    #           = actual expected performance, significantly better than requirement
    #           Parameters from LDC AnalyticNoise (Sangria, LDC2a)
    #
    # 'robson2019': alias, identical to 'scird' (for backward compatibility)
    _NOISE_PRESETS = {
        'scird':      {'oms': 15.0e-12, 'acc': 3.0e-15},
        'cbe':        {'oms':  7.9e-12, 'acc': 2.4e-15},
        'robson2019': {'oms': 15.0e-12, 'acc': 3.0e-15},
    }

    # Legacy calibration name → new noise_preset name (backward compatibility)
    _LEGACY_CAL_MAP = {
        'robson2019':             'scird',
        'redbook2023_allocation': 'scird',
        'redbook2023_cbe':        'cbe',
    }

    def __init__(self, target_log10_freq, noise_source='analytical',
                 noise_scale=1.0, injection_mode='physical',
                 logscale_base=1.0, T_obs=4.0, include_confusion=True,
                 noise_preset='scird', calibration=None):
        self.noise_scale = noise_scale
        self.injection_mode = injection_mode
        self.logscale_base = logscale_base
        self.target_log10_freq = np.asarray(target_log10_freq, dtype=np.float64)

        # Backward compatibility: if user passed calibration, auto-map to noise_preset
        if calibration is not None:
            mapped = self._LEGACY_CAL_MAP.get(calibration)
            if mapped is None:
                raise ValueError(
                    f"Unknown legacy calibration '{calibration}'. "
                    f"Valid: {list(self._LEGACY_CAL_MAP.keys())}, "
                    f"or use new 'noise_preset': {list(self._NOISE_PRESETS.keys())}")
            print(f"[LISA Noise] [DEPRECATED] calibration='{calibration}' "
                  f"→ noise_preset='{mapped}'")
            noise_preset = mapped

        if noise_preset not in self._NOISE_PRESETS:
            raise ValueError(
                f"Unknown LISA noise_preset '{noise_preset}'. "
                f"Valid: {list(self._NOISE_PRESETS.keys())}")
        self.noise_preset = noise_preset
        preset = self._NOISE_PRESETS[noise_preset]
        self._oms_noise = preset['oms']
        self._acc_noise = preset['acc']

        target_hz = 10.0 ** self.target_log10_freq

        if noise_source == 'analytical':
            sn = self._analytical_psd(target_hz, T_obs, include_confusion,
                                       oms_noise=self._oms_noise,
                                       acc_noise=self._acc_noise)
        else:
            sn = self._load_datafile(noise_source, target_hz)

        # Convert to σ_Ω(f)
        H0_si = 67.4e3 / 3.0857e22
        self.sigma_omega = ((2 * np.pi ** 2 / (3 * H0_si ** 2))
                            * target_hz ** 3 * sn).astype(np.float32)
        self.sigma_omega *= self.noise_scale

        # logscale weights (log-compressed)
        log_sigma = np.log10(np.maximum(self.sigma_omega, 1e-30))
        log_min, log_max = log_sigma.min(), log_sigma.max()
        if log_max > log_min:
            max_ratio = 10.0
            self.freq_weight = (1.0 + (max_ratio - 1.0)
                                * (log_sigma - log_min) / (log_max - log_min))
        else:
            self.freq_weight = np.ones_like(self.sigma_omega)
        self.freq_weight = self.freq_weight.astype(np.float32)

        print(f"[LISA Noise] source={noise_source}, preset={noise_preset} "
              f"(OMS={self._oms_noise*1e12:.2f} pm/√Hz, "
              f"acc={self._acc_noise*1e15:.2f} fm/s²/√Hz), "
              f"scale={noise_scale}, mode={injection_mode}, "
              f"T_obs={T_obs}yr, confusion={include_confusion}")
        print(f"  σ_Ω range: [{self.sigma_omega.min():.3e}, "
              f"{self.sigma_omega.max():.3e}]")

    @staticmethod
    def _analytical_psd(f, T_obs=4.0, include_confusion=True,
                        oms_noise=15.0e-12, acc_noise=3.0e-15):
        """Robson+ 2019 Eq.(10)+(12)+(14) — LISA effective strain PSD [1/Hz]

        Noise parameters (oms_noise, acc_noise) determine SciRD vs CBE:
            SciRD/Allocation: oms=15 pm/√Hz,  acc=3 fm/s²/√Hz
            CBE:              oms=7.9 pm/√Hz, acc=2.4 fm/s²/√Hz
        """
        L = 2.5e9           # Arm length [m]
        f_star = 19.09e-3   # c/(2πL) [Hz]

        P_oms = oms_noise ** 2 * (1 + (2e-3 / f) ** 4)
        P_acc = acc_noise ** 2 * (1 + (0.4e-3 / f) ** 2) * (1 + (f / 8e-3) ** 4)

        Sn = (10.0 / (3 * L ** 2)) * (
            P_oms + 2 * (1 + np.cos(f / f_star) ** 2) * P_acc / (2 * np.pi * f) ** 4
        ) * (1 + 0.6 * (f / f_star) ** 2)

        if include_confusion and T_obs > 0:
            Sc = LISANoiseModel._confusion_noise(f, T_obs)
            Sn = Sn + Sc

        return Sn

    @staticmethod
    def _confusion_noise(f, T_obs):
        """Robson+ 2019 Eq.(14) — Galactic foreground confusion noise"""
        # Select the nearest parameter set
        years = sorted(LISANoiseModel._CONFUSION_PARAMS.keys())
        yr = min(years, key=lambda y: abs(y - T_obs))
        p = LISANoiseModel._CONFUSION_PARAMS[yr]

        A = 9e-45
        Sc = (A * f ** (-7.0 / 3.0)
              * np.exp(-f ** p['alpha'] + p['beta'] * f * np.sin(p['kappa'] * f))
              * (1 + np.tanh(p['gamma'] * (p['f_k'] - f))))
        return Sc

    @staticmethod
    def _load_datafile(filepath, target_hz):
        """Load ASD from external file, convert to PSD, and interpolate to target grid"""
        data = np.loadtxt(filepath)
        freq_hz, asd = data[:, 0], data[:, 1]
        psd = asd ** 2
        fn = interp1d(np.log10(freq_hz), np.log10(psd),
                      kind='linear', fill_value='extrapolate')
        return 10.0 ** fn(np.log10(target_hz))

    def inject(self, log10_omega, log10_freq=None):
        """Inject noise, interface consistent with LIGONoiseModel."""
        if self.injection_mode == 'physical':
            omega = np.float64(10.0) ** log10_omega.astype(np.float64)
            noise = np.random.normal(0.0, self.sigma_omega.astype(np.float64))
            omega_obs = np.abs(omega + noise)
            omega_obs = np.maximum(omega_obs, 1e-300)
            return np.log10(omega_obs).astype(np.float32)
        else:
            noise = np.random.normal(0.0, 1.0, log10_omega.shape).astype(np.float32)
            return log10_omega + noise * self.freq_weight * self.logscale_base

# ==========================================
# 0. Frequency band configuration
# ==========================================

# Concat dataset configuration (non-uniform concatenation)
BAND_CONFIG_CONCAT = {
    'PTA':  {'start': 0,   'len': 14,  'freq_range': (-8.70, -7.56)},
    'LISA': {'start': 14,  'len': 500, 'freq_range': (-3.00, -1.00)},
    'LIGO': {'start': 270, 'len': 546, 'freq_range': (1.30, 3.24)},
}

# Conti dataset configuration (continuous interpolation)
BAND_FREQ_RANGES = {
    'PTA':  (-8.70, -7.56),
    'LISA': (-3.00, -1.00),
    'LIGO': (1.30, 3.24),
}
FREQ_INTERVAL = 0.01465073

def calc_band_points(freq_range, interval):
    return int(round((freq_range[1] - freq_range[0]) / interval)) + 1

BAND_CONFIG_CONTI = {
    'PTA':  {'len': calc_band_points(BAND_FREQ_RANGES['PTA'], FREQ_INTERVAL),  'freq_range': BAND_FREQ_RANGES['PTA']},
    'LISA': {'len': calc_band_points(BAND_FREQ_RANGES['LISA'], FREQ_INTERVAL), 'freq_range': BAND_FREQ_RANGES['LISA']},
    'LIGO': {'len': calc_band_points(BAND_FREQ_RANGES['LIGO'], FREQ_INTERVAL), 'freq_range': BAND_FREQ_RANGES['LIGO']},
}
BAND_CONFIG_CONTI['PTA']['start'] = 0
BAND_CONFIG_CONTI['LISA']['start'] = int(round((BAND_FREQ_RANGES['LISA'][0] - BAND_FREQ_RANGES['PTA'][0]) / FREQ_INTERVAL))
BAND_CONFIG_CONTI['LIGO']['start'] = int(round((BAND_FREQ_RANGES['LIGO'][0] - BAND_FREQ_RANGES['PTA'][0]) / FREQ_INTERVAL))

# Single-band dataset paths
SINGLE_BAND_PATHS = {
    'PTA':  'PTA_fixed_14pts.json',
    'LISA': 'LISA_fixed_500pts.json',
    'LIGO': 'LIGO_fixed_546pts.json',
}

# 9-parameter configuration
PARAM_CONFIG = {
    'names': ['r', 'n_t', 'kappa10', 'T_re', 'DN_re', 'Omega_bh2', 'Omega_ch2', 'H0', 'A_s'],
    'log10_idx': [0, 2, 3],
    'log_idx': [8],
    'raw_idx': [1, 4, 5, 6, 7],
}


# ==========================================
# 1. Dataset classes
# ==========================================

class BaseDataset(Dataset):
    """Base dataset class with shared methods"""

    def __init__(self, noise_config=None, use_slope=True, use_curvature=True):
        self.noise_config = noise_config if noise_config else {}
        self.use_complex = self.noise_config.get('USE_COMPLEX_NOISE', False)
        self.noise_level = self.noise_config.get('noise_level', 0.0)
        self.glitch_prob = self.noise_config.get('glitch_prob', 0.0)
        self.use_slope = use_slope
        self.use_curvature = use_curvature

    @staticmethod
    def _build_ligo_noise_model(noise_config, target_log10_freq):
        """Build LIGONoiseModel if LIGO physical noise is configured."""
        ligo_cfg = noise_config.get('ligo', {}) if noise_config else {}
        noise_file = ligo_cfg.get('noise_file', None)
        if noise_file is None or not os.path.isfile(noise_file):
            return None
        return LIGONoiseModel(
            noise_file=noise_file,
            noise_type=ligo_cfg.get('noise_type', 'cc_spectrum'),
            target_log10_freq=target_log10_freq,
            noise_scale=ligo_cfg.get('noise_scale', 1.0),
            injection_mode=ligo_cfg.get('injection_mode', 'physical'),
            logscale_base=ligo_cfg.get('logscale_base', 1.0),
        )

    @staticmethod
    def _build_pta_noise_model(noise_config):
        """Build PTANoiseModel if PTA physical noise is configured."""
        pta_cfg = noise_config.get('pta', {}) if noise_config else {}
        noise_dir = pta_cfg.get('noise_dir', None)
        if noise_dir is None or not os.path.isdir(noise_dir):
            return None
        return PTANoiseModel(
            noise_dir=noise_dir,
            mode=pta_cfg.get('mode', 'density_mock'),
            noise_scale=pta_cfg.get('noise_scale', 1.0),
            n_bins_use=pta_cfg.get('n_bins_use', 14),
            per_sample_noise=pta_cfg.get('per_sample_noise', True),
            H0_fiducial=pta_cfg.get('H0_fiducial', 67.4),
            T_obs=pta_cfg.get('T_obs', None),
        )

    @staticmethod
    def _build_lisa_noise_model(noise_config, target_log10_freq):
        """Build LISANoiseModel if LISA physical noise is configured."""
        lisa_cfg = noise_config.get('lisa', {}) if noise_config else {}
        if not lisa_cfg.get('enabled', False):
            return None
        # Prefer the new noise_preset parameter, fall back to legacy calibration
        noise_preset = lisa_cfg.get('noise_preset', None)
        legacy_cal = lisa_cfg.get('calibration', None)
        if noise_preset is None and legacy_cal is None:
            noise_preset = 'scird'  # default
        return LISANoiseModel(
            target_log10_freq=target_log10_freq,
            noise_source=lisa_cfg.get('noise_source', 'analytical'),
            noise_scale=lisa_cfg.get('noise_scale', 1.0),
            injection_mode=lisa_cfg.get('injection_mode', 'physical'),
            logscale_base=lisa_cfg.get('logscale_base', 1.0),
            T_obs=lisa_cfg.get('T_obs', 4.0),
            include_confusion=lisa_cfg.get('include_confusion', True),
            noise_preset=noise_preset if noise_preset is not None else 'scird',
            calibration=legacy_cal,
        )

    @staticmethod
    def _standardize_single(features, scalers):
        """Standardize a single sample's multi-channel features using pre-fitted scalers.

        Args:
            features: shape (C, L) - raw features
            scalers:  list of StandardScaler, one per channel
        Returns:
            standardized features, shape (C, L)
        """
        result = np.empty_like(features)
        for c in range(features.shape[0]):
            # scaler was fit on reshape(-1, 1), so mean_ and scale_ are scalar arrays
            result[c] = (features[c] - scalers[c].mean_[0]) / scalers[c].scale_[0]
        return result

    def compute_features(self, omega_vals, freq_vals):
        """Compute feature channels"""
        features = [omega_vals, freq_vals]
        if self.use_slope:
            slope = np.gradient(omega_vals, freq_vals)
            features.append(slope.astype(np.float32))
        if self.use_curvature:
            if self.use_slope:
                curv = np.gradient(features[-1], freq_vals)
            else:
                curv = np.gradient(np.gradient(omega_vals, freq_vals), freq_vals)
            features.append(curv.astype(np.float32))
        return np.stack(features, axis=0)

    def extract_params(self, item):
        """Extract and transform the 9 parameters"""
        return [
            np.log10(item['r']),
            item['n_t'],
            np.log10(item['kappa10']),
            np.log10(item['T_re']),
            item['DN_re'],
            item['Omega_bh2'],
            item['Omega_ch2'],
            item['H0'],
            np.log(1e10 * item['A_s'])
        ]

    def add_spectral_glitch(self, log_curve_normed):
        """Add glitch interference in the frequency domain"""
        seq_len = log_curve_normed.shape[0]
        linear_energy = np.exp(log_curve_normed)
        num_glitches = np.random.randint(1, 3)
        glitch_energy = np.zeros_like(linear_energy)

        for _ in range(num_glitches):
            center_idx = np.random.randint(0, seq_len)
            sigma = np.random.uniform(2.0, 15.0)
            amplitude = np.random.uniform(0.5, 3.0) * np.max(linear_energy)
            x = np.arange(seq_len)
            bump = amplitude * np.exp(-0.5 * ((x - center_idx) / sigma)**2)
            glitch_energy += bump

        total_energy = linear_energy + glitch_energy
        return np.log(total_energy + 1e-10)

    def apply_noise(self, curves):
        """Apply noise to the omega channel of curves"""
        if self.use_complex:
            if self.noise_level > 0:
                for curve in curves:
                    curve[0, :] += np.random.normal(0, self.noise_level, curve[0].shape)
            if self.glitch_prob > 0 and np.random.rand() < self.glitch_prob:
                for curve in curves:
                    curve[0, :] = self.add_spectral_glitch(curve[0, :])
        else:
            if self.noise_level > 0:
                for curve in curves:
                    curve[0, :] += np.random.normal(0, self.noise_level, curve[0].shape)
        return curves


class MultiBandDataset(BaseDataset):
    """Three-band joint dataset (concat or conti)"""

    def __init__(self, data, band_config, noise_config=None, use_slope=True, use_curvature=True):
        super().__init__(noise_config, use_slope, use_curvature)
        self.band_config = band_config

        pta_curves, lisa_curves, ligo_curves = [], [], []

        for item in data:
            omega = np.array(item['log10OmegaGW'], dtype=np.float32)
            freq = np.array(item['f'], dtype=np.float32)

            pta_omega = omega[band_config['PTA']['start']:band_config['PTA']['start']+band_config['PTA']['len']]
            pta_freq = freq[band_config['PTA']['start']:band_config['PTA']['start']+band_config['PTA']['len']]

            lisa_omega = omega[band_config['LISA']['start']:band_config['LISA']['start']+band_config['LISA']['len']]
            lisa_freq = freq[band_config['LISA']['start']:band_config['LISA']['start']+band_config['LISA']['len']]

            ligo_omega = omega[band_config['LIGO']['start']:band_config['LIGO']['start']+band_config['LIGO']['len']]
            ligo_freq = freq[band_config['LIGO']['start']:band_config['LIGO']['start']+band_config['LIGO']['len']]

            pta_curves.append(self.compute_features(pta_omega, pta_freq))
            lisa_curves.append(self.compute_features(lisa_omega, lisa_freq))
            ligo_curves.append(self.compute_features(ligo_omega, ligo_freq))

        self.pta_curves_raw = np.stack(pta_curves, axis=0)
        self.lisa_curves_raw = np.stack(lisa_curves, axis=0)
        self.ligo_curves_raw = np.stack(ligo_curves, axis=0)

        self.num_channels = self.pta_curves_raw.shape[1]
        print(f"Using {self.num_channels} channels: omega, freq" +
              (", slope" if use_slope else "") +
              (", curvature" if use_curvature else ""))

        # Parameter extraction
        params = [self.extract_params(item) for item in data]
        self.params_raw = np.array(params, dtype=np.float32)

        # ---- Three-band physical noise models ----
        self.pta_noise_model = self._build_pta_noise_model(noise_config)
        if self.pta_noise_model is not None and hasattr(self.pta_noise_model, 'validate_freqs'):
            self.pta_noise_model.validate_freqs(self.pta_curves_raw[0, 1, :])

        self.lisa_freq_grid = self.lisa_curves_raw[0, 1, :]
        self.lisa_noise_model = self._build_lisa_noise_model(
            noise_config, self.lisa_freq_grid)

        self.ligo_freq_grid = self.ligo_curves_raw[0, 1, :]
        self.ligo_noise_model = self._build_ligo_noise_model(
            noise_config, self.ligo_freq_grid)

        # ---- Standardization ----
        self.pta_scalers = [StandardScaler() for _ in range(self.num_channels)]
        self.lisa_scalers = [StandardScaler() for _ in range(self.num_channels)]
        self.ligo_scalers = [StandardScaler() for _ in range(self.num_channels)]
        self.param_scaler = StandardScaler()

        # PTA scaler
        if self.pta_noise_model is not None:
            print("[PTA Noise] Fitting scalers on noise-injected reference data...")
            pta_ref = self._generate_noisy_reference('pta')
            self.pta_curves = self._scale_curves(pta_ref, self.pta_scalers, band_config['PTA']['len'])
        else:
            self.pta_curves = self._scale_curves(self.pta_curves_raw, self.pta_scalers, band_config['PTA']['len'])

        # LISA scaler
        if self.lisa_noise_model is not None and \
                self.lisa_noise_model.injection_mode == 'physical':
            print("[LISA Noise] Fitting scalers on noise-injected reference data...")
            lisa_ref = self._generate_noisy_reference('lisa')
            self.lisa_curves = self._scale_curves(lisa_ref, self.lisa_scalers, band_config['LISA']['len'])
        else:
            self.lisa_curves = self._scale_curves(self.lisa_curves_raw, self.lisa_scalers, band_config['LISA']['len'])

        # LIGO scaler
        if self.ligo_noise_model is not None and \
                self.ligo_noise_model.injection_mode == 'physical':
            print("[LIGO Noise] Fitting scalers on noise-injected reference data...")
            ligo_ref = self._generate_noisy_reference('ligo')
            self.ligo_curves = self._scale_curves(ligo_ref, self.ligo_scalers, band_config['LIGO']['len'])
        else:
            self.ligo_curves = self._scale_curves(self.ligo_curves_raw, self.ligo_scalers, band_config['LIGO']['len'])

        self.params = self.param_scaler.fit_transform(self.params_raw)

        print(f"Dataset loaded: {len(self.params)} samples")
        print(f"  PTA shape: {self.pta_curves.shape}")
        print(f"  LISA shape: {self.lisa_curves.shape}")
        print(f"  LIGO shape: {self.ligo_curves.shape}")

    def _generate_noisy_reference(self, band):
        """Generate noise-injected features for physical mode, used to fit scalers."""
        if band == 'pta':
            raw, model = self.pta_curves_raw, self.pta_noise_model
        elif band == 'lisa':
            raw, model = self.lisa_curves_raw, self.lisa_noise_model
        else:
            raw, model = self.ligo_curves_raw, self.ligo_noise_model

        noisy_curves = []
        for i in range(len(raw)):
            omega_raw = raw[i, 0, :].copy()
            freq_raw = raw[i, 1, :]
            omega_noisy = model.inject(omega_raw, freq_raw)
            features = self.compute_features(omega_noisy, freq_raw)
            noisy_curves.append(features)
        return np.stack(noisy_curves, axis=0)

    # ------------------------------------------------------------------
    # Vectorized batch noise cache (Optimization B)
    # ------------------------------------------------------------------
    def _compute_features_batch(self, omega_batch, freq):
        """Vectorized computation of feature channels for an entire batch."""
        freq_batch = np.broadcast_to(freq[None, :], omega_batch.shape).copy()
        features = [omega_batch, freq_batch]
        if self.use_slope:
            slope = np.gradient(omega_batch, freq, axis=1).astype(np.float32)
            features.append(slope)
        if self.use_curvature:
            if self.use_slope:
                curv = np.gradient(features[-1], freq, axis=1).astype(np.float32)
            else:
                curv = np.gradient(
                    np.gradient(omega_batch, freq, axis=1),
                    freq, axis=1
                ).astype(np.float32)
            features.append(curv)
        return np.stack(features, axis=1)

    def _standardize_batch(self, features_batch, scalers):
        """Vectorized standardization"""
        result = np.empty_like(features_batch)
        for c in range(features_batch.shape[1]):
            result[:, c, :] = ((features_batch[:, c, :]
                                - scalers[c].mean_[0]) / scalers[c].scale_[0])
        return result

    def _inject_noise_batch(self, omega_batch, noise_model, band_name, noise_scale_override=1.0):
        """Vectorized noise injection"""
        if noise_model is None:
            return None

        if band_name == 'PTA':
            if noise_model.is_density_mode:
                # Physical pathway: additive in linear Ω space (Li & Shapiro 2025 Eq.3, mean-subtracted deviation)
                omega_lin = (np.float64(10.0) ** omega_batch.astype(np.float64))
                omega_noise = noise_model.sample_omega_noise(omega_batch.shape[0])  # (N,14)
                omega_obs = np.maximum(omega_lin + omega_noise * noise_scale_override,
                                       1e-300)
                return np.log10(omega_obs).astype(np.float32)
            # ---- Legacy log-space additive ----
            if noise_model.mode == 'white':
                noise = np.random.normal(0.0, noise_model.sigma_white, omega_batch.shape)
            else:
                noise = np.random.normal(0.0,
                    noise_model.sigma_per_bin[:omega_batch.shape[1]], omega_batch.shape)
            return (omega_batch + noise * noise_scale_override).astype(np.float32)

        if getattr(noise_model, 'injection_mode', 'physical') == 'physical':
            omega_lin = np.float64(10.0) ** omega_batch.astype(np.float64)
            sigma = noise_model.sigma_omega.astype(np.float64)
            noise = np.random.normal(0.0, 1.0, omega_batch.shape) * sigma[None, :] * noise_scale_override
            omega_obs = np.maximum(np.abs(omega_lin + noise), 1e-300)
            return np.log10(omega_obs).astype(np.float32)
        else:
            noise = np.random.normal(0.0, 1.0, omega_batch.shape).astype(np.float32)
            return omega_batch + noise * noise_model.freq_weight[None, :] * noise_model.logscale_base * noise_scale_override

    def refresh_noisy_cache(self, noise_scale_override=1.0):
        """Called at the start of each epoch: vectorized regeneration of noisy data for all three bands."""
        self._cache = {}

        for band_name, raw, model, scalers, freq_grid in [
            ('PTA',  self.pta_curves_raw,  self.pta_noise_model,  self.pta_scalers,  None),
            ('LISA', self.lisa_curves_raw,  self.lisa_noise_model, self.lisa_scalers,  self.lisa_freq_grid),
            ('LIGO', self.ligo_curves_raw,  self.ligo_noise_model, self.ligo_scalers,  self.ligo_freq_grid),
        ]:
            if model is None:
                self._cache[band_name] = None
                continue
            omega_raw = raw[:, 0, :].copy()
            freq = raw[0, 1, :]
            omega_noisy = self._inject_noise_batch(omega_raw, model, band_name, noise_scale_override)
            features = self._compute_features_batch(omega_noisy, freq)
            self._cache[band_name] = self._standardize_batch(features, scalers)

    def _scale_curves(self, curves_raw, scalers, band_len):
        scaled = []
        for c in range(self.num_channels):
            channel_data = curves_raw[:, c, :].reshape(-1, 1)
            scalers[c].fit(channel_data)
            scaled.append(scalers[c].transform(channel_data).reshape(-1, band_len))
        return np.stack(scaled, axis=1)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        params = self.params[idx].copy()

        # ---- PTA ----
        if self.pta_noise_model is not None:
            if hasattr(self, '_cache') and self._cache.get('PTA') is not None:
                pta = self._cache['PTA'][idx]
            else:
                omega_raw = self.pta_curves_raw[idx, 0, :].copy()
                freq_raw = self.pta_curves_raw[idx, 1, :]
                omega_noisy = self.pta_noise_model.inject(omega_raw)
                features = self.compute_features(omega_noisy, freq_raw)
                pta = self._standardize_single(features, self.pta_scalers)
        else:
            pta = self.pta_curves[idx].copy()

        # ---- LISA ----
        if self.lisa_noise_model is not None:
            if hasattr(self, '_cache') and self._cache.get('LISA') is not None:
                lisa = self._cache['LISA'][idx]
            else:
                omega_raw = self.lisa_curves_raw[idx, 0, :].copy()
                freq_raw = self.lisa_curves_raw[idx, 1, :]
                omega_noisy = self.lisa_noise_model.inject(omega_raw, freq_raw)
                features = self.compute_features(omega_noisy, freq_raw)
                lisa = self._standardize_single(features, self.lisa_scalers)
        else:
            lisa = self.lisa_curves[idx].copy()

        # ---- LIGO ----
        if self.ligo_noise_model is not None:
            if hasattr(self, '_cache') and self._cache.get('LIGO') is not None:
                ligo = self._cache['LIGO'][idx]
            else:
                omega_raw = self.ligo_curves_raw[idx, 0, :].copy()
                freq_raw = self.ligo_curves_raw[idx, 1, :]
                omega_noisy = self.ligo_noise_model.inject(omega_raw, freq_raw)
                features = self.compute_features(omega_noisy, freq_raw)
                ligo = self._standardize_single(features, self.ligo_scalers)
        else:
            ligo = self.ligo_curves[idx].copy()

        # For bands without a physical noise model, fall back to legacy Gaussian noise
        if self.noise_level > 0:
            if self.pta_noise_model is None:
                pta[0, :] += np.random.normal(0, self.noise_level, pta[0].shape)
            if self.lisa_noise_model is None:
                lisa[0, :] += np.random.normal(0, self.noise_level, lisa[0].shape)
            if self.ligo_noise_model is None:
                ligo[0, :] += np.random.normal(0, self.noise_level, ligo[0].shape)

        return (
            torch.tensor(pta, dtype=torch.float32),
            torch.tensor(lisa, dtype=torch.float32),
            torch.tensor(ligo, dtype=torch.float32),
            torch.tensor(params, dtype=torch.float32)
        )


class SingleBandDataset(BaseDataset):
    """Single-band dataset"""

    def __init__(self, data, band_name, band_config, noise_config=None, use_slope=True, use_curvature=True):
        super().__init__(noise_config, use_slope, use_curvature)
        self.band_name = band_name
        self.band_len = band_config['len']

        curves = []
        for item in data:
            omega = np.array(item['log10OmegaGW'], dtype=np.float32)
            freq = np.array(item['f'], dtype=np.float32)
            curves.append(self.compute_features(omega, freq))

        self.curves_raw = np.stack(curves, axis=0)
        self.num_channels = self.curves_raw.shape[1]

        print(f"[{band_name}] Using {self.num_channels} channels: omega, freq" +
              (", slope" if use_slope else "") +
              (", curvature" if use_curvature else ""))

        params = [self.extract_params(item) for item in data]
        self.params_raw = np.array(params, dtype=np.float32)

        # ---- Build the physical noise model for this band ----
        self.noise_model = None
        freq_grid = self.curves_raw[0, 1, :]  # log10(f/Hz)
        if band_name == 'PTA':
            self.noise_model = self._build_pta_noise_model(noise_config)
            if self.noise_model is not None and hasattr(self.noise_model, 'validate_freqs'):
                self.noise_model.validate_freqs(freq_grid)
        elif band_name == 'LISA':
            self.noise_model = self._build_lisa_noise_model(noise_config, freq_grid)
        elif band_name == 'LIGO':
            self.noise_model = self._build_ligo_noise_model(noise_config, freq_grid)

        # ---- Standardization ----
        self.scalers = [StandardScaler() for _ in range(self.num_channels)]
        self.param_scaler = StandardScaler()

        needs_noisy_scaler = (
            self.noise_model is not None and
            (band_name == 'PTA' or getattr(self.noise_model, 'injection_mode', '') == 'physical')
        )
        if needs_noisy_scaler:
            print(f"[{band_name} Noise] Fitting scalers on noise-injected reference...")
            noisy_ref = self._generate_noisy_reference()
            self.curves = self._scale_curves_single(noisy_ref)
        else:
            self.curves = self._scale_curves_single(self.curves_raw)

        self.params = self.param_scaler.fit_transform(self.params_raw)

        print(f"[{band_name}] Dataset loaded: {len(self.params)} samples, "
              f"shape: {self.curves.shape}")

    def _scale_curves_single(self, curves_data):
        scaled = []
        for c in range(self.num_channels):
            channel_data = curves_data[:, c, :].reshape(-1, 1)
            self.scalers[c].fit(channel_data)
            scaled.append(self.scalers[c].transform(channel_data).reshape(-1, self.band_len))
        return np.stack(scaled, axis=1)

    def _generate_noisy_reference(self):
        noisy = []
        for i in range(len(self.curves_raw)):
            omega_raw = self.curves_raw[i, 0, :].copy()
            freq_raw = self.curves_raw[i, 1, :]
            if self.band_name == 'PTA':
                omega_noisy = self.noise_model.inject(omega_raw)
            else:
                omega_noisy = self.noise_model.inject(omega_raw, freq_raw)
            features = self.compute_features(omega_noisy, freq_raw)
            noisy.append(features)
        return np.stack(noisy, axis=0)

    # ------------------------------------------------------------------
    # Vectorized batch noise cache (Optimization B)
    # ------------------------------------------------------------------
    def _compute_features_batch(self, omega_batch, freq):
        """Vectorized computation of feature channels for an entire batch.

        Args:
            omega_batch: shape (N, L)
            freq:        shape (L,)  shared across all samples
        Returns:
            shape (N, C, L)
        """
        N = omega_batch.shape[0]
        freq_batch = np.broadcast_to(freq[None, :], omega_batch.shape).copy()
        features = [omega_batch, freq_batch]
        if self.use_slope:
            slope = np.gradient(omega_batch, freq, axis=1).astype(np.float32)
            features.append(slope)
        if self.use_curvature:
            if self.use_slope:
                curv = np.gradient(features[-1], freq, axis=1).astype(np.float32)
            else:
                curv = np.gradient(
                    np.gradient(omega_batch, freq, axis=1),
                    freq, axis=1
                ).astype(np.float32)
            features.append(curv)
        return np.stack(features, axis=1)

    def _standardize_batch(self, features_batch, scalers):
        """Vectorized standardization, features_batch shape (N, C, L)"""
        result = np.empty_like(features_batch)
        for c in range(features_batch.shape[1]):
            result[:, c, :] = ((features_batch[:, c, :]
                                - scalers[c].mean_[0]) / scalers[c].scale_[0])
        return result

    def _inject_noise_batch(self, omega_batch, noise_scale_override=1.0):
        """Vectorized noise injection, omega_batch shape (N, L) is log10(Ω)

        Args:
            noise_scale_override: Noise scaling factor for curriculum learning.
                0.0 = no noise, 1.0 = full noise.
        """
        model = self.noise_model
        if model is None:
            return omega_batch.copy()

        if self.band_name == 'PTA':
            if model.is_density_mode:
                omega_lin = (np.float64(10.0) ** omega_batch.astype(np.float64))
                omega_noise = model.sample_omega_noise(omega_batch.shape[0])  # (N,14)
                omega_obs = np.maximum(omega_lin + omega_noise * noise_scale_override,
                                       1e-300)
                return np.log10(omega_obs).astype(np.float32)
            # ---- Legacy log-space additive ----
            if model.mode == 'white':
                noise = np.random.normal(0.0, model.sigma_white, omega_batch.shape)
            else:
                noise = np.random.normal(0.0, model.sigma_per_bin[:omega_batch.shape[1]],
                                         omega_batch.shape)
            return (omega_batch + noise * noise_scale_override).astype(np.float32)

        if getattr(model, 'injection_mode', 'physical') == 'physical':
            # Physical: add noise in linear space → log
            omega_lin = (np.float64(10.0)
                         ** omega_batch.astype(np.float64))
            sigma = model.sigma_omega.astype(np.float64)
            noise = np.random.normal(0.0, 1.0, omega_batch.shape) * sigma[None, :] * noise_scale_override
            omega_obs = np.maximum(np.abs(omega_lin + noise), 1e-300)
            return np.log10(omega_obs).astype(np.float32)
        else:
            # Logscale
            noise = np.random.normal(0.0, 1.0, omega_batch.shape).astype(np.float32)
            return omega_batch + noise * model.freq_weight[None, :] * model.logscale_base * noise_scale_override

    def refresh_noisy_cache(self, noise_scale_override=1.0):
        """Called at the start of each epoch: vectorized regeneration of the entire batch of noisy data.

        Args:
            noise_scale_override: Curriculum noise scaling, gradually increasing from 0→1.
        """
        if self.noise_model is None:
            return  # No physical noise model, no cache needed

        omega_raw = self.curves_raw[:, 0, :].copy()       # (N, L)
        freq = self.curves_raw[0, 1, :]                    # (L,)

        # 1) Vectorized noise injection
        omega_noisy = self._inject_noise_batch(omega_raw, noise_scale_override)  # (N, L)

        # 2) Vectorized feature computation
        features = self._compute_features_batch(omega_noisy, freq)  # (N, C, L)

        # 3) Vectorized standardization
        self._noisy_cache = self._standardize_batch(features, self.scalers)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        params = self.params[idx].copy()

        if self.noise_model is not None:
            if hasattr(self, '_noisy_cache') and self._noisy_cache is not None:
                # Read from cache (fast path)
                curve = self._noisy_cache[idx]
            else:
                # Fallback: per-sample computation (first epoch or refresh not called)
                omega_raw = self.curves_raw[idx, 0, :].copy()
                freq_raw = self.curves_raw[idx, 1, :]
                if self.band_name == 'PTA':
                    omega_noisy = self.noise_model.inject(omega_raw)
                else:
                    omega_noisy = self.noise_model.inject(omega_raw, freq_raw)
                features = self.compute_features(omega_noisy, freq_raw)
                curve = self._standardize_single(features, self.scalers)
        else:
            curve = self.curves[idx].copy()
            curve = self.apply_noise([curve])[0]

        return (
            torch.tensor(curve, dtype=torch.float32),
            torch.tensor(params, dtype=torch.float32)
        )


# ==========================================
# 2. Model components
# ==========================================

class ResBlock1D(nn.Module):
    """1D residual block"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class BandEncoder(nn.Module):
    """Encoder for a single frequency band"""
    def __init__(self, in_channels=4, hidden_dim=64, output_dim=64, band_type='lisa', seq_len=256):
        super().__init__()
        self.band_type = band_type

        if band_type == 'pta':
            if seq_len <= 20:
                # Very few points, use MLP
                self.encoder = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(in_channels * seq_len, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                    nn.Tanh()
                )
            else:
                # Moderate number of points, use lightweight CNN
                self.encoder = nn.Sequential(
                    nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
                    nn.BatchNorm1d(32),
                    nn.ReLU(inplace=True),
                    ResBlock1D(32),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(32, output_dim),
                    nn.Tanh()
                )
        elif band_type == 'lisa':
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                ResBlock1D(32),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(32, output_dim),
                nn.Tanh()
            )
        else:  # ligo
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                ResBlock1D(64),
                ResBlock1D(64),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, output_dim),
                nn.Tanh()
            )

    def forward(self, x):
        return self.encoder(x)


class MultiBandContextEncoder(nn.Module):
    """Three-band fusion encoder"""
    def __init__(self, band_config, context_dim=64, in_channels=4):
        super().__init__()
        per_band_dim = context_dim // 3
        remainder = context_dim - per_band_dim * 3

        self.pta_encoder = BandEncoder(in_channels, hidden_dim=32, output_dim=per_band_dim,
                                       band_type='pta', seq_len=band_config['PTA']['len'])
        self.lisa_encoder = BandEncoder(in_channels, hidden_dim=64, output_dim=per_band_dim,
                                        band_type='lisa', seq_len=band_config['LISA']['len'])
        self.ligo_encoder = BandEncoder(in_channels, hidden_dim=64, output_dim=per_band_dim + remainder,
                                        band_type='ligo', seq_len=band_config['LIGO']['len'])

    def forward(self, pta, lisa, ligo):
        h_pta = self.pta_encoder(pta)
        h_lisa = self.lisa_encoder(lisa)
        h_ligo = self.ligo_encoder(ligo)
        return torch.cat([h_pta, h_lisa, h_ligo], dim=1)


class SingleBandEncoder(nn.Module):
    """Single-band encoder"""
    def __init__(self, band_name, in_channels=4, hidden_dim=64, output_dim=64, seq_len=256):
        super().__init__()
        band_type = 'pta' if band_name == 'PTA' else ('lisa' if band_name == 'LISA' else 'ligo')
        self.encoder = BandEncoder(in_channels, hidden_dim, output_dim, band_type, seq_len)

    def forward(self, x):
        return self.encoder(x)


class ConditionalCouplingLayer(nn.Module):
    """Conditional coupling layer"""
    def __init__(self, num_inputs, num_hidden, mask, context_dim):
        super().__init__()
        self.mask = nn.Parameter(mask, requires_grad=False)
        input_dim = num_inputs + context_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, num_inputs * 2)
        )

    def forward(self, inputs, context, mode='direct'):
        mask = self.mask
        masked_inputs = inputs * mask
        net_input = torch.cat([masked_inputs, context], dim=1)
        out = self.net(net_input)
        s, t = out.chunk(2, dim=1)
        s = torch.tanh(s) * (1 - mask)
        t = t * (1 - mask)
        if mode == 'direct':
            y = inputs * torch.exp(s) + t
            log_det = torch.sum(s, dim=1)
            return y, log_det
        else:
            x = (inputs - t) * torch.exp(-s)
            return x, None


class MultiBandFlow(nn.Module):
    """Three-band joint conditional Normalizing Flow"""
    def __init__(self, band_config, num_inputs=9, num_hidden=128, num_layers=10,
                 context_dim=64, in_channels=4):
        super().__init__()
        self.num_inputs = num_inputs
        self.context_encoder = MultiBandContextEncoder(band_config, context_dim, in_channels)
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            mask = torch.zeros(num_inputs)
            if i % 2 == 0:
                mask[::2] = 1
            else:
                mask[1::2] = 1
            self.layers.append(ConditionalCouplingLayer(num_inputs, num_hidden, mask, context_dim))

        # Auxiliary parameter prediction head: direct regression from context,
        # reinforcing the encoder's ability to encode key parameters
        self.param_predictor = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, num_inputs)
        )

    def _get_context(self, pta, lisa, ligo):
        return self.context_encoder(pta, lisa, ligo)

    def log_prob(self, inputs, pta, lisa, ligo):
        context = self._get_context(pta, lisa, ligo)
        log_det_sum = 0
        z = inputs
        for layer in self.layers:
            z, log_det = layer(z, context, mode='direct')
            log_det_sum += log_det
        log_p_z = -0.5 * (np.log(2 * np.pi) + z.pow(2)).sum(dim=1)
        return log_p_z + log_det_sum

    def predict_params(self, pta, lisa, ligo):
        """Auxiliary parameter prediction (used for T_re optimization, etc.)"""
        context = self._get_context(pta, lisa, ligo)
        return self.param_predictor(context)

    def sample(self, pta, lisa, ligo, num_samples=1):
        context = self._get_context(pta, lisa, ligo)
        B = context.shape[0]

        context_rep = context.unsqueeze(1).repeat(1, num_samples, 1).reshape(B * num_samples, -1)
        z = torch.randn(B * num_samples, 9, device=pta.device)

        for layer in reversed(self.layers):
            z, _ = layer(z, context_rep, mode='inverse')

        return z.reshape(B, num_samples, -1)


class SingleBandFlow(nn.Module):
    """Single-band conditional Normalizing Flow"""
    def __init__(self, band_name, seq_len, num_inputs=9, num_hidden=128, num_layers=10,
                 context_dim=64, in_channels=4):
        super().__init__()
        self.band_name = band_name
        self.num_inputs = num_inputs
        self.context_encoder = SingleBandEncoder(band_name, in_channels, 64, context_dim, seq_len)
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            mask = torch.zeros(num_inputs)
            if i % 2 == 0:
                mask[::2] = 1
            else:
                mask[1::2] = 1
            self.layers.append(ConditionalCouplingLayer(num_inputs, num_hidden, mask, context_dim))

        # Auxiliary parameter prediction head
        self.param_predictor = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, num_inputs)
        )

    def _get_context(self, curve):
        return self.context_encoder(curve)

    def log_prob(self, inputs, curve):
        context = self._get_context(curve)
        log_det_sum = 0
        z = inputs
        for layer in self.layers:
            z, log_det = layer(z, context, mode='direct')
            log_det_sum += log_det
        log_p_z = -0.5 * (np.log(2 * np.pi) + z.pow(2)).sum(dim=1)
        return log_p_z + log_det_sum

    def predict_params(self, curve):
        """Auxiliary parameter prediction (used for T_re optimization, etc.)"""
        context = self._get_context(curve)
        return self.param_predictor(context)

    def sample(self, curve, num_samples=1):
        context = self._get_context(curve)
        B = context.shape[0]

        context_rep = context.unsqueeze(1).repeat(1, num_samples, 1).reshape(B * num_samples, -1)
        z = torch.randn(B * num_samples, 9, device=curve.device)

        for layer in reversed(self.layers):
            z, _ = layer(z, context_rep, mode='inverse')

        return z.reshape(B, num_samples, -1)


# ==========================================
# 3. Training functions
# ==========================================

def _build_param_weights(config, device):
    """Build parameter weight vector and auxiliary loss coefficient from config.

    Example config (in JSON "training" or top-level):
        "param_emphasis": {
            "enabled": true,
            "aux_weight": 0.1,
            "weights": {
                "T_re": 3.0,
                "DN_re": 0.3
            }
        }

    Returns:
        aux_weight (float):  Global coefficient λ for auxiliary loss
        param_weights (Tensor): shape (9,), MSE weight per parameter
        Returns (0.0, None) if disabled
    """
    pe = config.get('param_emphasis', {})
    if not pe.get('enabled', False):
        return 0.0, None

    aux_weight = pe.get('aux_weight', 0.1)
    weight_map = pe.get('weights', {})

    names = PARAM_CONFIG['names']
    w = torch.ones(len(names), device=device)
    for name, val in weight_map.items():
        if name in names:
            w[names.index(name)] = val
        else:
            print(f"[WARNING] param_emphasis: unknown parameter '{name}', ignored")

    print(f"[Param Emphasis] aux_weight={aux_weight}, "
          f"weights={dict(zip(names, w.tolist()))}")
    return aux_weight, w


def _get_curriculum_scale(epoch, total_epochs, curriculum_cfg):
    """Compute the noise scaling factor for the current epoch (curriculum noise schedule).

    Example config:
        "noise_curriculum": {
            "enabled": true,
            "warmup_fraction": 0.2,
            "start_scale": 0.05,
            "end_scale": 1.0
        }

    During warmup_fraction * total_epochs, noise increases linearly from start_scale to end_scale,
    then stays at end_scale afterwards.

    Returns:
        float: Noise scaling factor for the current epoch
    """
    if not curriculum_cfg.get('enabled', False):
        return 1.0

    warmup_frac = curriculum_cfg.get('warmup_fraction', 0.2)
    start_scale = curriculum_cfg.get('start_scale', 0.05)
    end_scale = curriculum_cfg.get('end_scale', 1.0)

    warmup_epochs = int(total_epochs * warmup_frac)
    if warmup_epochs <= 0:
        return end_scale

    if epoch >= warmup_epochs:
        return end_scale

    # Linear interpolation
    t = epoch / warmup_epochs
    return start_scale + (end_scale - start_scale) * t

def train_multiband(config):
    """Train the three-band joint model"""
    # Parse config
    dataset_type = config['dataset']['type']
    json_path = config['dataset']['path']
    use_slope = config['features']['use_slope']
    use_curvature = config['features']['use_curvature']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    grad_clip = config['training']['grad_clip']
    scheduler_T_max = config['training']['scheduler_T_max']

    noise_config = {
        'USE_COMPLEX_NOISE': config['noise']['use_complex'],
        'noise_level': config['noise']['level'],
        'glitch_prob': config['noise']['glitch_prob'],
    }

    # ---- Physical noise configuration ----
    for band_key in ('pta', 'lisa', 'ligo'):
        band_cfg = config['noise'].get(band_key, {})
        if band_cfg:
            noise_config[band_key] = band_cfg
            print(f"{band_key.upper()} physical noise enabled: {band_cfg}")

    save_dir = config['output']['save_dir']
    save_name = config['output']['save_name']
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)

    # Select band configuration
    band_config = BAND_CONFIG_CONCAT if dataset_type == 'concat' else BAND_CONFIG_CONTI

    # Load data
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print(f"Experiment: {config['name']}")
    print(f"{'='*60}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"Loaded {len(data)} samples from {json_path}")
    print(f"Band config: PTA={band_config['PTA']['len']}, "
          f"LISA={band_config['LISA']['len']}, LIGO={band_config['LIGO']['len']}")

    train_idx, val_idx = train_test_split(np.arange(len(data)), test_size=0.2, random_state=42)
    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]

    train_dataset = MultiBandDataset(train_data, band_config, noise_config, use_slope, use_curvature)

    # Validation set: keep LIGO physical noise (evaluate noisy performance), but disable legacy Gaussian noise
    val_noise_config = {'USE_COMPLEX_NOISE': False, 'noise_level': 0.0}
    for band_key in ('pta', 'lisa', 'ligo'):
        if band_key in noise_config:
            val_noise_config[band_key] = noise_config[band_key]
    val_dataset = MultiBandDataset(val_data, band_config, val_noise_config,
                                   use_slope, use_curvature)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=4, pin_memory=True, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    in_channels = train_dataset.num_channels
    model = MultiBandFlow(band_config,
                          num_inputs=config['model']['num_inputs'],
                          num_hidden=config['model']['num_hidden'],
                          num_layers=config['model']['num_layers'],
                          context_dim=config['model']['context_dim'],
                          in_channels=in_channels).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_T_max)

    print(f"Noise Mode: {'COMPLEX (Glitch)' if noise_config['USE_COMPLEX_NOISE'] else 'SIMPLE (Gaussian)'}")
    print(f"Features: slope={use_slope}, curvature={use_curvature}")

    # ---- Parameter emphasis (T_re optimization, etc.) ----
    aux_weight, param_weights = _build_param_weights(config, device)

    # ---- Curriculum noise schedule ----
    curriculum_cfg = config.get('noise_curriculum', {})
    if curriculum_cfg.get('enabled', False):
        print(f"[Curriculum] warmup={curriculum_cfg.get('warmup_fraction', 0.2):.0%} of {epochs} epochs, "
              f"scale: {curriculum_cfg.get('start_scale', 0.05)} → {curriculum_cfg.get('end_scale', 1.0)}")

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # ---- Refresh noise cache each epoch (vectorized, millisecond-scale) ----
        noise_scale = _get_curriculum_scale(epoch, epochs, curriculum_cfg)
        train_dataset.refresh_noisy_cache(noise_scale)
        val_dataset.refresh_noisy_cache()  # Validation set always uses full noise

        model.train()
        train_loss = 0.0

        for pta, lisa, ligo, params in train_loader:
            pta, lisa, ligo, params = pta.to(device), lisa.to(device), ligo.to(device), params.to(device)

            loss_flow = -torch.mean(model.log_prob(params, pta, lisa, ligo))

            # Auxiliary parameter prediction loss
            if aux_weight > 0 and param_weights is not None:
                pred = model.predict_params(pta, lisa, ligo)
                loss_aux = torch.mean(param_weights * (pred - params) ** 2)
                loss = loss_flow + aux_weight * loss_aux
            else:
                loss = loss_flow

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pta, lisa, ligo, params in val_loader:
                pta, lisa, ligo, params = pta.to(device), lisa.to(device), ligo.to(device), params.to(device)
                val_loss += -torch.mean(model.log_prob(params, pta, lisa, ligo)).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'pta_scalers': train_dataset.pta_scalers,
                'lisa_scalers': train_dataset.lisa_scalers,
                'ligo_scalers': train_dataset.ligo_scalers,
                'param_scaler': train_dataset.param_scaler,
                'band_config': band_config,
                'param_config': PARAM_CONFIG,
                'num_channels': in_channels,
                'use_slope': use_slope,
                'use_curvature': use_curvature,
                'dataset_type': dataset_type,
                'config': config,
            }, save_path)

        if epoch % 10 == 0 or epoch < 20:
            ns_str = f" | NS: {noise_scale:.3f}" if curriculum_cfg.get('enabled', False) else ""
            print(f"Epoch {epoch:4d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} (Best: {best_val_loss:.4f}){ns_str}")

    print(f"\nTraining complete! Best model saved to: {save_path}")
    return best_val_loss


def train_single_band(config):
    """Train single-band model"""
    bands = config['dataset']['bands']
    data_dir = config['dataset']['data_dir']
    band_files = config['dataset']['band_files']

    use_slope = config['features']['use_slope']
    use_curvature = config['features']['use_curvature']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    grad_clip = config['training']['grad_clip']
    scheduler_T_max = config['training']['scheduler_T_max']

    noise_config = {
        'USE_COMPLEX_NOISE': config['noise']['use_complex'],
        'noise_level': config['noise']['level'],
        'glitch_prob': config['noise']['glitch_prob'],
    }

    # ---- Physical noise configuration ----
    for band_key in ('pta', 'lisa', 'ligo'):
        band_cfg = config['noise'].get(band_key, {})
        if band_cfg:
            noise_config[band_key] = band_cfg

    save_dir = config['output']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # ---- Parameter emphasis (T_re optimization, etc.) — build once outside the band loop ----
    # device is not yet determined, build on CPU first, then .to(device) inside the band loop
    _aux_weight, _param_weights_cpu = _build_param_weights(config, torch.device('cpu'))

    results = {}

    for band_name in bands:
        json_path = os.path.join(data_dir, band_files[band_name])
        band_config = {'len': BAND_CONFIG_CONCAT[band_name]['len'],
                       'freq_range': BAND_CONFIG_CONCAT[band_name]['freq_range']}
        # Single band: respect save_name in config; multiple bands: auto-name
        if len(bands) == 1 and config['output'].get('save_name'):
            save_path = os.path.join(save_dir, config['output']['save_name'])
        else:
            save_path = os.path.join(save_dir, f"{band_name.lower()}_flow.pt")

        # Load data
        with open(json_path, 'r') as f:
            data = json.load(f)

        print(f"\n{'='*60}")
        print(f"Experiment: {config['name']} - {band_name}")
        print(f"{'='*60}")
        print(f"Loaded {len(data)} samples from {json_path}")

        train_idx, val_idx = train_test_split(np.arange(len(data)), test_size=0.2, random_state=42)
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]

        # Training set with noise; validation set keeps physical noise but disables legacy Gaussian noise
        train_dataset = SingleBandDataset(train_data, band_name, band_config, noise_config, use_slope, use_curvature)

        val_noise_config = {'USE_COMPLEX_NOISE': False, 'noise_level': 0.0}
        band_key = band_name.lower()
        if band_key in noise_config:
            val_noise_config[band_key] = noise_config[band_key]
        val_dataset = SingleBandDataset(val_data, band_name, band_config, val_noise_config,
                                        use_slope, use_curvature)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                  num_workers=4, pin_memory=True, persistent_workers=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=2, pin_memory=True, persistent_workers=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on: {device}")

        in_channels = train_dataset.num_channels
        model = SingleBandFlow(band_name, band_config['len'],
                               num_inputs=config['model']['num_inputs'],
                               num_hidden=config['model']['num_hidden'],
                               num_layers=config['model']['num_layers'],
                               context_dim=config['model']['context_dim'],
                               in_channels=in_channels).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_T_max)

        print(f"Noise Mode: {'COMPLEX (Glitch)' if noise_config['USE_COMPLEX_NOISE'] else 'SIMPLE (Gaussian)'}")
        print(f"Features: slope={use_slope}, curvature={use_curvature}")

        # Move parameter emphasis weights to current device
        aux_weight = _aux_weight
        param_weights = _param_weights_cpu.to(device) if _param_weights_cpu is not None else None

        # ---- Curriculum noise schedule ----
        curriculum_cfg = config.get('noise_curriculum', {})
        if curriculum_cfg.get('enabled', False):
            print(f"[Curriculum] warmup={curriculum_cfg.get('warmup_fraction', 0.2):.0%} of {epochs} epochs, "
                  f"scale: {curriculum_cfg.get('start_scale', 0.05)} → {curriculum_cfg.get('end_scale', 1.0)}")

        best_val_loss = float('inf')
        patience_counter = 0
        patience = config['training'].get('early_stopping', {}).get('patience', 100)
        min_epochs = config['training'].get('early_stopping', {}).get('min_epochs', 500)

        for epoch in range(epochs):
            # ---- Refresh noise cache each epoch (vectorized, millisecond-scale) ----
            noise_scale = _get_curriculum_scale(epoch, epochs, curriculum_cfg)
            train_dataset.refresh_noisy_cache(noise_scale)
            val_dataset.refresh_noisy_cache()  # Validation set always uses full noise

            model.train()
            train_loss = 0.0

            for curve, params in train_loader:
                curve, params = curve.to(device), params.to(device)

                loss_flow = -torch.mean(model.log_prob(params, curve))

                # Auxiliary parameter prediction loss
                if aux_weight > 0 and param_weights is not None:
                    pred = model.predict_params(curve)
                    loss_aux = torch.mean(param_weights * (pred - params) ** 2)
                    loss = loss_flow + aux_weight * loss_aux
                else:
                    loss = loss_flow

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for curve, params in val_loader:
                    curve, params = curve.to(device), params.to(device)
                    val_loss += -torch.mean(model.log_prob(params, curve)).item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'scalers': train_dataset.scalers,
                    'param_scaler': train_dataset.param_scaler,
                    'band_config': band_config,
                    'band_name': band_name,
                    'num_channels': in_channels,
                    'use_slope': use_slope,
                    'use_curvature': use_curvature,
                    'config': config,
                }, save_path)
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch < 20:
                ns_str = f" | NS: {noise_scale:.3f}" if curriculum_cfg.get('enabled', False) else ""
                print(f"Epoch {epoch:4d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} (Best: {best_val_loss:.4f}){ns_str}")

            if patience_counter >= patience and epoch > min_epochs:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"\n{band_name} training complete! Best model saved to: {save_path}")
        results[band_name] = best_val_loss

    print("\n" + "="*60)
    print("All single-band training complete!")
    print("="*60)
    for band_name, loss in results.items():
        print(f"  {band_name}: Best Val Loss = {loss:.4f}")

    return results


# ==========================================
# 3.5 Experiment package: 1 joint + 3 single, one-click execution
# ==========================================

def _build_single_subconfig(pkg_cfg, package_dir, band_name):
    """Derive a single-band sub-task config from the package config."""
    sub = copy.deepcopy(pkg_cfg)
    sub.pop('tasks', None)
    sub.pop('output', None)
    sub['name'] = f"{pkg_cfg.get('name', 'package')}_{band_name}"
    sub['dataset'] = {
        'type': 'single',
        'data_dir': pkg_cfg['dataset']['single_dir'],
        'bands': [band_name],
        'band_files': pkg_cfg['dataset']['single_files'],
    }
    sub['output'] = {
        'save_dir': os.path.join(package_dir, 'single'),
        'save_name': f'{band_name}_flow.pt',
    }
    return sub


def _build_joint_subconfig(pkg_cfg, package_dir):
    """Derive a joint sub-task config from the package config."""
    sub = copy.deepcopy(pkg_cfg)
    sub.pop('tasks', None)
    sub.pop('output', None)
    sub['name'] = f"{pkg_cfg.get('name', 'package')}_joint"
    sub['dataset'] = {
        'type': pkg_cfg['dataset'].get('joint_type', 'concat'),
        'path': pkg_cfg['dataset']['joint_path'],
    }
    sub['output'] = {
        'save_dir': os.path.join(package_dir, 'joint'),
        'save_name': 'flow_joint.pt',
    }
    return sub


def train_experiment_package(config):
    """Run a complete experiment package: 1 joint + 3 single, sharing all training hyperparameters.

    Required config fields:
        dataset.joint_path:    Joint dataset JSON path
        dataset.single_dir:    Directory containing single-band datasets
        dataset.single_files:  {'PTA': '...', 'LISA': '...', 'LIGO': '...'}
        output.package_dir:    Experiment package root directory
        Other fields (features, training, model, noise, ...) are the same as normal config.

    Optional:
        tasks:  list, default ['LIGO', 'LISA', 'PTA', 'joint']
                Order: LIGO runs first (least data, exposes config errors early); joint last.
        dataset.joint_type: 'concat' (default) or 'conti'
    """
    package_dir = config['output']['package_dir']
    os.makedirs(package_dir, exist_ok=True)
    os.makedirs(os.path.join(package_dir, 'single'), exist_ok=True)
    os.makedirs(os.path.join(package_dir, 'joint'), exist_ok=True)

    # Save a config snapshot so settings can be traced back later
    with open(os.path.join(package_dir, 'package_config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Tee stdout → write to both terminal and training_log.txt
    log_path = os.path.join(package_dir, 'training_log.txt')
    log_file = open(log_path, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, log_file)

    try:
        tasks = config.get('tasks', ['LIGO', 'LISA', 'PTA', 'joint'])
        print("\n" + "#" * 70)
        print(f"# EXPERIMENT PACKAGE: {config.get('name', 'unnamed')}")
        print(f"# Description: {config.get('description', 'N/A')}")
        print(f"# Output dir:  {package_dir}")
        print(f"# Tasks:       {tasks}")
        print("#" * 70)

        for task in tasks:
            print("\n" + "#" * 70)
            print(f"# [PACKAGE TASK]  {task}")
            print("#" * 70)

            if task == 'joint':
                sub_cfg = _build_joint_subconfig(config, package_dir)
                train_multiband(sub_cfg)
            elif task in ('PTA', 'LISA', 'LIGO'):
                sub_cfg = _build_single_subconfig(config, package_dir, task)
                train_single_band(sub_cfg)
            else:
                print(f"[WARNING] Unknown task '{task}', skipping")

        print("\n" + "#" * 70)
        print(f"# PACKAGE COMPLETE: {package_dir}")
        print("#" * 70)
        print("Output files:")
        for task in tasks:
            if task == 'joint':
                print(f"  joint:  {os.path.join(package_dir, 'joint', 'flow_joint.pt')}")
            elif task in ('PTA', 'LISA', 'LIGO'):
                print(f"  {task:<6}: {os.path.join(package_dir, 'single', f'{task}_flow.pt')}")
        print(f"  log:    {log_path}")
    finally:
        sys.stdout = original_stdout
        log_file.close()


# ==========================================
# 4. Main function
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Unified Normalizing Flow Training')

    # Config file (highest priority)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file (overrides other arguments)')

    # Command-line arguments (as fallback)
    parser.add_argument('--dataset-type', type=str, default='concat',
                        choices=['concat', 'conti', 'single'],
                        help='Dataset type')
    parser.add_argument('--band', type=str, default='all',
                        choices=['PTA', 'LISA', 'LIGO', 'all'],
                        help='Which band to train (single mode)')
    parser.add_argument('--data-dir', type=str,
                        default='./dataset',
                        help='Data directory')
    parser.add_argument('--json-path', type=str, default=None,
                        help='Direct path to JSON file')
    parser.add_argument('--save-dir', type=str,
                        default='./models',
                        help='Directory to save models')
    parser.add_argument('--save-name', type=str, default='flow.pt',
                        help='Save filename')
    parser.add_argument('--no-slope', action='store_true',
                        help='Disable slope feature')
    parser.add_argument('--no-curvature', action='store_true',
                        help='Disable curvature feature')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='Number of training epochs')
    parser.add_argument('--ligo-noise', type=str, default=None,
                        help='Path to LIGO noise file (C_O1_O2_O3.dat or AplusDesign.txt)')
    parser.add_argument('--ligo-noise-type', type=str, default='cc_spectrum',
                        choices=['cc_spectrum', 'aplus_asd'],
                        help='LIGO noise data type')
    parser.add_argument('--ligo-noise-mode', type=str, default='physical',
                        choices=['physical', 'logscale'],
                        help='LIGO noise injection mode')
    parser.add_argument('--ligo-noise-scale', type=float, default=1.0,
                        help='LIGO noise scale factor')

    args = parser.parse_args()

    # If a config file is provided, use it
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from: {args.config}")

        # Priority: package mode > single > joint
        if 'package_dir' in config.get('output', {}):
            train_experiment_package(config)
        elif config['dataset']['type'] == 'single':
            train_single_band(config)
        else:
            train_multiband(config)
        return

    # Otherwise build config from command-line arguments
    config = {
        'name': 'cli_config',
        'description': 'Configuration from command line arguments',
        'dataset': {
            'type': args.dataset_type,
            'path': args.json_path or os.path.join(args.data_dir, 'joint/Dataset_PTA_LISA_LIGO.json'),
        },
        'features': {
            'use_slope': not args.no_slope,
            'use_curvature': not args.no_curvature,
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': 256,
            'learning_rate': 0.0002,
            'weight_decay': 1e-5,
            'grad_clip': 5.0,
            'scheduler_T_max': 300,
        },
        'model': {
            'num_inputs': 9,
            'num_hidden': 128,
            'num_layers': 10,
            'context_dim': 64,
        },
        'noise': {
            'use_complex': False,
            'level': 1.0,
            'glitch_prob': 0.5,
        },
        'output': {
            'save_dir': args.save_dir,
            'save_name': args.save_name,
        },
    }

    # ---- Physical noise configuration (CLI mode) ----
    # LIGO: requires external data file
    if args.ligo_noise:
        config['noise']['ligo'] = {
            'noise_file': args.ligo_noise,
            'noise_type': args.ligo_noise_type,
            'noise_scale': args.ligo_noise_scale,
            'injection_mode': args.ligo_noise_mode,
            'logscale_base': 1.0,
        }
    # LISA: built-in analytical formula, enabled by default (auto-enabled in CLI mode)
    # To disable, use JSON config with "enabled": false
    config['noise']['lisa'] = {
        'enabled': True,
        'noise_source': 'analytical',
        'noise_scale': 1.0,
        'injection_mode': 'physical',
        'T_obs': 4.0,
        'include_confusion': True,
    }
    # PTA: requires NANOGrav .npy files, not enabled in CLI mode by default
    # Use JSON config: "pta": {"noise_dir": "/path/to/NANOGrav15yr", ...}

    if args.dataset_type == 'single':
        config['dataset']['bands'] = ['PTA', 'LISA', 'LIGO'] if args.band == 'all' else [args.band]
        config['dataset']['data_dir'] = os.path.join(args.data_dir, 'single')
        config['dataset']['band_files'] = SINGLE_BAND_PATHS
        config['output']['save_dir'] = os.path.join(args.save_dir, 'single')
        train_single_band(config)
    else:
        if args.dataset_type == 'conti':
            config['dataset']['path'] = args.json_path or os.path.join(args.data_dir, 'joint/Dataset_PTA_LISA_LIGO_conti.json')
        train_multiband(config)


if __name__ == "__main__":
    main()
