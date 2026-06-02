#!/usr/bin/env python3
"""
Consolidated Figure 4 script.
Generates: Fig4, Fig4_SI
  - Fig4:    |Fr*| vs d_{R-R}^*, NL=20, DG0=[-6,-10,-14]
  - Fig4_SI: |Fz*| vs d_{R-R}^*, same parameters
"""

import sys, os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Theory_v9_mp import *

outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_plots')
os.makedirs(outdir, exist_ok=True)

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

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# ── Physical parameters ───────────────────────────────────────────────
Nmono = 20
amono = 0.38
akuhn = 0.76
Nkuhn = Nmono * amono / akuhn
kbT   = 1.0
NL    = 20

R_ee = np.sqrt(Nkuhn) * akuhn
k_ee = 3.0 * kbT / R_ee**2

print(f"Nkuhn = {Nkuhn},  R_ee = {R_ee:.4f} nm,  k_ee = {k_ee:.6f} kbT/nm^2")
print(f"NL = {NL}")

DG0_values = [-6, -10, -14]
d_RR_star_range = np.linspace(0.3, 12.0, 240)
sigma_star_range = 1.0 / d_RR_star_range**2
sigma_range = sigma_star_range / R_ee**2


def data_template(sigma_val):
    return {
        'kEff': k_ee, 'kEffRep': k_ee, 'x0': 0,
        'NL': NL, 'Nrep': 0, 'kbT': kbT, 'DG0': -10.0,
        'nIntSamples': 150, 'verbose': False,
        'NP_type': 'star_polymer',
        'Nkuhn': Nkuhn, 'amono': amono, 'akuhn': akuhn,
        'sigma': sigma_val, 'cV0': 1e-9, 'cNP0': 1e-4,
        'rV': 100, 'rNP': 10, 'epsilon_self': 1e-7,
    }


def save_fig(fig, name):
    for ext in ('pdf', 'png'):
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p, dpi=150, bbox_inches='tight')
        print(f"  Saved {p}")
    plt.close(fig)


# ── Compute both Fr and Fz in a single sweep ────────────────────────
print("\nComputing F* vs d_RR* ...")
results = {}
for DG0 in DG0_values:
    print(f"  DG0 = {DG0}")
    d_list, Fr_list, Fz_list = [], [], []
    for d_star, sigma in zip(d_RR_star_range, sigma_range):
        data = data_template(sigma)
        data['DG0'] = DG0
        rLimit(data)
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                _, Fr = averageForce(direction='r', data=data)
                _, Fz = averageForce(direction='z', data=data)
            Fr_list.append(Fr * R_ee / kbT)
            Fz_list.append(Fz * R_ee / kbT)
        except Exception as e:
            Fr_list.append(np.nan)
            Fz_list.append(np.nan)
        d_list.append(d_star)
    results[DG0] = (d_list, Fr_list, Fz_list)


def plot_fig4(force_idx, ylabel, output_name):
    """force_idx: 1=Fr, 2=Fz in the results tuple."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, DG0 in enumerate(DG0_values):
        d_list = results[DG0][0]
        Fstar_list = results[DG0][force_idx]
        x = np.array(d_list, dtype=float)
        y = np.abs(Fstar_list)
        mask = np.isfinite(y)
        if mask.sum() >= 4:
            spl = make_interp_spline(x[mask], y[mask], k=2)
            x_smooth = np.linspace(x[mask].min(), x[mask].max(), 300)
            y_smooth = spl(x_smooth)
        else:
            x_smooth, y_smooth = x[mask], y[mask]
        ax.plot(x_smooth, y_smooth, '-', color=colors[idx], linewidth=1.5,
                label=rf"$\Delta G_0 = {DG0}\, k_BT$")

    ax.set_xlabel(r"$d_{R\text{-}R}^*$", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, output_name)


# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nFig4\n" + "="*60)
plot_fig4(1, r"$F^*$", "Fig4")

print("\n" + "="*60 + "\nFig4_SI\n" + "="*60)
plot_fig4(2, r"$F^*$", "Fig4_SI")

print("\nAll Fig4 variants done.")
