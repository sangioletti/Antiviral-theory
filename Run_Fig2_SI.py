#!/usr/bin/env python3
"""
SI Figure: F_z* vs DG0 (same parameters as Figure 2 but vertical force).

Same panels and parameters as Run_Fig2.py:
  sigma*_R = 0.06, 0.60, 6.00;  N_L = 2, 4, 6, 8
"""

import sys, os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Theory_v9_mp import *

# ── Publication-quality style ────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.grid': False,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.top': True,
    'ytick.right': True,
    'font.size': 12,
})

# ── Physical parameters ───────────────────────────────────────────────
Nmono = 20
amono = 0.38        # nm
akuhn = 0.76        # nm
Nkuhn = Nmono * amono / akuhn   # = 10
kbT   = 1.0

R_ee = np.sqrt(Nkuhn) * akuhn
k_ee = 3.0 * kbT / R_ee**2

print(f"Nkuhn = {Nkuhn},  amono = {amono} nm,  akuhn = {akuhn} nm")
print(f"R_ee  = {R_ee:.4f} nm")
print(f"k_ee  = {k_ee:.6f} kbT/nm^2")

# ── Figure parameters ────────────────────────────────────────────────
NL_values = [2, 4, 6, 8]
sigma_star_values = [0.06, 0.60, 6.00]
sigma_values = [sstar / R_ee**2 for sstar in sigma_star_values]

DG0_values = np.linspace(-20, 4, 75)

# ── Base data dictionary ─────────────────────────────────────────────
data_template = {
    'kEff':     k_ee,
    'kEffRep':  k_ee,
    'x0':       0,
    'NL':       5,
    'Nrep':     0,
    'kbT':      kbT,
    'DG0':      -10.0,
    'nIntSamples': 150,
    'verbose':  False,
    'NP_type':  'star_polymer',
    'Nkuhn':    Nkuhn,
    'amono':    amono,
    'akuhn':    akuhn,
    'sigma':    0.01,
    'cV0':      1e-9,
    'cNP0':     1e-4,
    'rV':       100,
    'rNP':      10,
    'epsilon_self': 1e-7,
}

# ── Main sweep ───────────────────────────────────────────────────────
results = {}

for sstar, sigma in zip(sigma_star_values, sigma_values):
    for NL in NL_values:
        print(f"\n{'='*60}")
        print(f"sigma* = {sstar},  sigma = {sigma:.6e},  NL = {NL}")
        print(f"{'='*60}")

        Fstar_list = []
        for DG0 in DG0_values:
            data = dict(data_template)
            data['sigma'] = sigma
            data['NL'] = NL
            data['DG0'] = DG0

            rLimit(data)

            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    _, Fz = averageForce(direction='z', data=data)
                    for warning in w:
                        print(f"  [WARN DG0={DG0}] {warning.message}")

                Fstar = Fz * R_ee / kbT
                print(f"  DG0={DG0:6.1f}  F_z={Fz:+.6e}  F*={Fstar:+.6e}")
            except Exception as e:
                print(f"  DG0={DG0:6.1f}  ERROR: {e}")
                Fstar = np.nan

            Fstar_list.append(Fstar)

        results[(sstar, NL)] = (DG0_values.tolist(), Fstar_list)

# ── Plot: 3 subplots (one per sigma*) ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Global y-range across all panels
all_y = np.concatenate([
    np.abs(np.asarray(results[(sstar, NL)][1], dtype=float))
    for sstar in sigma_star_values for NL in NL_values
])
all_y = all_y[np.isfinite(all_y)]
y_min, y_max = all_y.min(), all_y.max()
pad = 0.05 * (y_max - y_min)
y_lim = (y_min - pad, y_max + pad)

for panel_idx, (sstar, ax) in enumerate(zip(sigma_star_values, axes)):
    for idx, NL in enumerate(NL_values):
        DG0s, Fstars = results[(sstar, NL)]
        x = np.array(DG0s, dtype=float)
        y = np.abs(Fstars)
        mask = np.isfinite(y)
        if mask.sum() >= 4:
            spl = make_interp_spline(x[mask], y[mask], k=2)
            x_smooth = np.linspace(x[mask].min(), x[mask].max(), 300)
            y_smooth = spl(x_smooth)
        else:
            x_smooth, y_smooth = x[mask], y[mask]
        label = rf"$N_L = {NL}$" if panel_idx == 0 else None
        ax.plot(x_smooth, y_smooth, '-', color=colors[idx],
                linewidth=1.5, label=label)

    ax.set_xlabel(r"$\Delta G^0\;(k_BT)$", fontsize=13)
    if panel_idx == 0:
        ax.set_ylabel(r"$|F_z^*|$", fontsize=13)
        ax.legend(fontsize=9, framealpha=0.9)
    ax.set_title(rf"$\sigma_R^* = {sstar}$", fontsize=12)
    ax.set_ylim(y_lim)

fig.tight_layout()

outdir = os.path.dirname(os.path.abspath(__file__))
for ext in ('pdf', 'png'):
    outpath = os.path.join(outdir, f"Fig2_SI.{ext}")
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {outpath}")
