#!/usr/bin/env python3
"""
F* vs d_{R-R}^* = sqrt(1/sigma^*)  for NL=8, DG0 = -5, -10, -15.

Parameters:
    akuhn = 0.76 nm, amono = 0.38 nm, Nkuhn = 20
    R_ee = sqrt(Nkuhn) * amono * akuhn
    k_ee = 3 kbT / R_ee^2
    sigma* = sigma * R_ee^2
    d_{R-R}^* = sqrt(1 / sigma^*)
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
Nkuhn = Nmono * amono / akuhn 
kbT   = 1.0
NL    = 20 

R_ee = np.sqrt(Nkuhn) * akuhn
k_ee = 3.0 * kbT / R_ee**2
rMax_expected = Nkuhn * amono

print(f"Nkuhn = {Nkuhn},  amono = {amono} nm,  akuhn = {akuhn} nm")
print(f"R_ee  = {R_ee:.4f} nm")
print(f"k_ee  = {k_ee:.6f} kbT/nm^2")
print(f"rMax  = {rMax_expected:.2f} nm")
print(f"NL    = {NL}")

DG0_values = [-6, -10, -14]

# ── Sweep sigma* values ───────────────────────────────────────────────
# d_{R-R}^* = sqrt(1/sigma^*), so sigma^* = 1/d^2
# Choose a range of d_{R-R}^* values that gives interesting physics
# d ~ 0.3 to 5 → sigma* ~ 0.04 to 11
d_RR_star_range = np.linspace(0.3, 12.0, 240)
sigma_star_range = 1.0 / d_RR_star_range**2
sigma_range = sigma_star_range / R_ee**2

print(f"\nd_RR* range: [{d_RR_star_range[0]:.2f}, {d_RR_star_range[-1]:.2f}]")
print(f"sigma* range: [{sigma_star_range[-1]:.4f}, {sigma_star_range[0]:.4f}]")

# ── Base data dictionary ──────────────────────────────────────────────
data_template = {
    'kEff':     k_ee,
    'kEffRep':  k_ee,
    'x0':       0,
    'NL':       NL,
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

# ── Main sweep ────────────────────────────────────────────────────────
results = {}  # results[DG0] = (d_RR_list, Fstar_list)

for DG0 in DG0_values:
    print(f"\n{'='*60}")
    print(f"DG0 = {DG0} kbT")
    print(f"{'='*60}")

    d_list = []
    Fstar_list = []

    for d_star, sstar, sigma in zip(d_RR_star_range, sigma_star_range, sigma_range):
        data = dict(data_template)
        data['sigma'] = sigma
        data['DG0'] = DG0

        rLimit(data)

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _,Fr = averageForce(direction='r', data=data)

                for warning in w:
                    print(f"  [WARN d*={d_star:.2f}] {warning.message}")

            Fstar = Fr * R_ee / kbT
            print(f"  d_RR*={d_star:5.2f}  sigma*={sstar:8.4f}  F*={Fstar:+.6e}")
        except Exception as e:
            print(f"  d_RR*={d_star:5.2f}  sigma*={sstar:8.4f}  ERROR: {e}")
            Fstar = np.nan

        d_list.append(d_star)
        Fstar_list.append(Fstar)

    results[DG0] = (d_list, Fstar_list)

# ── Plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for idx, DG0 in enumerate(DG0_values):
    d_list, Fstar_list = results[DG0]
    x = np.array(d_list, dtype=float)
    y = np.abs(Fstar_list)
    mask = np.isfinite(y)
    if mask.sum() >= 4:
        spl = make_interp_spline(x[mask], y[mask], k=2)
        x_smooth = np.linspace(x[mask].min(), x[mask].max(), 300)
        y_smooth = spl(x_smooth)
    else:
        x_smooth, y_smooth = x[mask], y[mask]
    label = rf"$\Delta G_0 = {DG0}\, k_BT$"
    ax.plot(x_smooth, y_smooth, '-', color=colors[idx],
            linewidth=1.5, label=label)

ax.set_xlabel(r"$d_{R\text{-}R}^*$", fontsize=13)
ax.set_ylabel(r"$F^*$", fontsize=13)
ax.legend(fontsize=10, framealpha=0.9)

outdir = os.path.dirname(os.path.abspath(__file__))
for ext in ('pdf', 'png'):
    outpath = os.path.join(outdir, f"Fig4.{ext}")
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {outpath}")
