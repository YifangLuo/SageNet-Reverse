#!/usr/bin/env python3
"""Verify two noise presets of LISANoiseModel against Red Book Figure 7.1.

Plot three sets of curves (each ASD = sqrt(S_n)):
  1. SciRD / Allocation preset   (OMS=15 pm/sqrt(Hz), acc=3 fm/s^2/sqrt(Hz))
     ~ Red Book Fig 7.1 green band (Allocation, requirement)
  2. CBE preset                  (OMS=7.9 pm/sqrt(Hz), acc=2.4 fm/s^2/sqrt(Hz))
     ~ Red Book Fig 7.1 blue line (Current Best Estimate)
  3. (Optional) Digitized reference curve from Fig 7.1
     If ./red_book_fig7_1_cbe.txt exists (two columns: f[Hz]  ASD[1/sqrt(Hz)])
     it will be overlaid as ground truth.

References:
  - Babak, Hewitson, Petiteau 2021, "LISA Sensitivity and SNR Calculations",
    LISA-LCST-SGS-TN-001 (arXiv:2108.01167)
  - LISA Definition Study Report, ESA-SCI-DIR-RP-002, Figure 7.1
  - Robson, Cornish, Liu 2019 (arXiv:1803.01944)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from train import LISANoiseModel

# -- Frequency grid --
f = np.logspace(-5, 0, 2000)  # 0.01 mHz - 1 Hz

# -- Compute PSD for both presets --
presets_to_plot = ['scird', 'cbe']
results = {}
for preset_name in presets_to_plot:
    p = LISANoiseModel._NOISE_PRESETS[preset_name]
    Sn_total = LISANoiseModel._analytical_psd(
        f, T_obs=4.0, include_confusion=True,
        oms_noise=p['oms'], acc_noise=p['acc'])
    Sn_inst = LISANoiseModel._analytical_psd(
        f, T_obs=4.0, include_confusion=False,
        oms_noise=p['oms'], acc_noise=p['acc'])
    results[preset_name] = {
        'Sn_total': Sn_total,
        'Sn_inst':  Sn_inst,
        'asd_total': np.sqrt(Sn_total),
        'asd_inst':  np.sqrt(Sn_inst),
        'oms': p['oms'],
        'acc': p['acc'],
    }

# -- Optional: load digitized Fig 7.1 CBE curve --
ref_file = os.path.join(os.path.dirname(__file__), 'red_book_fig7_1_cbe.txt')
ref_data = None
if os.path.isfile(ref_file):
    try:
        ref_data = np.loadtxt(ref_file)
        print(f'[loaded reference] {ref_file}  ({len(ref_data)} points)')
    except Exception as e:
        print(f'[WARN] failed to load {ref_file}: {e}')

# -- Plotting --
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# ---- Left panel: ASD full-band comparison ----
ax = axes[0]
preset_styles = {
    'scird': {'color': 'green', 'lw': 2.0, 'ls': '-',
              'label': 'SciRD / Allocation (15 pm/sqrt(Hz), 3 fm/s^2/sqrt(Hz))'},
    'cbe':   {'color': 'blue',  'lw': 2.0, 'ls': '-',
              'label': 'CBE (7.9 pm/sqrt(Hz), 2.4 fm/s^2/sqrt(Hz))'},
}
for preset_name in presets_to_plot:
    s = preset_styles[preset_name]
    ax.loglog(f, results[preset_name]['asd_total'],
              color=s['color'], lw=s['lw'], ls=s['ls'], label=s['label'])
    # Instrument noise (without confusion) shown as dashed line
    ax.loglog(f, results[preset_name]['asd_inst'],
              color=s['color'], lw=1.0, ls='--', alpha=0.6)

if ref_data is not None:
    ax.loglog(ref_data[:, 0], ref_data[:, 1],
              'r*', ms=8, label='Red Book Fig 7.1 CBE (digitized)')

ax.set_xlabel('Frequency [Hz]', fontsize=13)
ax.set_ylabel(r'Strain ASD $\sqrt{S_n(f)}$  [Hz$^{-1/2}$]', fontsize=13)
ax.set_title('LISA Strain ASD: SciRD vs CBE\n'
             '(solid = total inc. confusion, dashed = instrument only)',
             fontsize=12)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(1e-5, 1.0)
ax.set_ylim(1e-21, 1e-15)

# ---- Right panel: zoomed view within Red Book Fig 7.1 frequency range (10^-4 ~ 10^0) ----
ax = axes[1]
for preset_name in presets_to_plot:
    s = preset_styles[preset_name]
    ax.loglog(f, results[preset_name]['asd_total'],
              color=s['color'], lw=s['lw'], ls=s['ls'], label=s['label'])

if ref_data is not None:
    ax.loglog(ref_data[:, 0], ref_data[:, 1],
              'r*', ms=8, label='Red Book Fig 7.1 CBE (digitized)')

ax.set_xlabel('Frequency [Hz]', fontsize=13)
ax.set_ylabel(r'Strain ASD $\sqrt{S_n(f)}$  [Hz$^{-1/2}$]', fontsize=13)
ax.set_title('Zoomed to Red Book Fig 7.1 range', fontsize=12)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(1e-4, 1.0)
ax.set_ylim(1e-21, 1e-16)

plt.tight_layout()
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'lisa_noise_verify.png')
fig.savefig(outpath, dpi=150)
print(f'[saved] {outpath}')

# -- Numerical check: at 1 mHz, compare against Red Book Fig 7.1 --
print('\n-- Numerical check (1 mHz) --')
print(f'{"preset":<10} {"OMS [pm/sqrt(Hz)]":<14} {"acc [fm/s^2/sqrt(Hz)]":<18} '
      f'{"S_n [1/Hz]":<14} {"ASD [1/sqrt(Hz)]":<14}')
print('-' * 75)
idx = np.argmin(np.abs(f - 1e-3))
for preset_name in presets_to_plot:
    r = results[preset_name]
    print(f'{preset_name:<10} '
          f'{r["oms"]*1e12:<14.2f} '
          f'{r["acc"]*1e15:<18.2f} '
          f'{r["Sn_total"][idx]:<14.3e} '
          f'{r["asd_total"][idx]:<14.3e}')

print('\nReference from Red Book Fig 7.1 (approximate readings at 1 mHz):')
print('  Allocation (green band):  ASD ~ 1.0e-19   1/sqrt(Hz)')
print('  CBE        (blue line):   ASD ~ 3-4e-20   1/sqrt(Hz)')

# -- Numerical check: 10 mHz --
print('\n-- Numerical check (10 mHz) --')
print(f'{"preset":<10} {"S_n [1/Hz]":<14} {"ASD [1/sqrt(Hz)]":<14}')
print('-' * 45)
idx = np.argmin(np.abs(f - 1e-2))
for preset_name in presets_to_plot:
    r = results[preset_name]
    print(f'{preset_name:<10} {r["Sn_total"][idx]:<14.3e} '
          f'{r["asd_total"][idx]:<14.3e}')

print('\nReference from Red Book Fig 7.1 (at 10 mHz):')
print('  Allocation (green band):  ASD ~ 1.2e-20   1/sqrt(Hz)')
print('  CBE        (blue line):   ASD ~ 6e-21     1/sqrt(Hz)')

if ref_data is None:
    print('\n[TIP] Want a precise comparison? Use WebPlotDigitizer (https://apps.automeris.io/wpd/)')
    print('     Digitize 20-30 points from the Red Book Fig 7.1 blue line and save as two-column text:')
    print(f'        f[Hz]   ASD[1/sqrt(Hz)]')
    print(f'     Filename: red_book_fig7_1_cbe.txt')
    print(f'     Place it in the same directory as this script and rerun to overlay the plot.')
