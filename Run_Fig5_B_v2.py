#!/usr/bin/env python3
"""
Variant of Run_Fig5_B.py adding a 4th panel: the force angle
theta* = atan2(|Fz*|, |Fr*|) in degrees, showing how steric polymers
rotate the net binding force from lateral toward normal.

Outputs Fig5_B_2nd_version.pdf/.png — Run_Fig5_B.py is unchanged.
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
amono = 0.38
akuhn = 0.76
Nkuhn = Nmono * amono / akuhn
kbT   = 1.0

R_ee = np.sqrt(Nkuhn) * akuhn
k_ee = 3.0 * kbT / R_ee**2

print(f"Nkuhn = {Nkuhn},  R_ee = {R_ee:.4f} nm,  k_ee = {k_ee:.6f} kbT/nm^2")

# ── Sweep parameters ─────────────────────────────────────────────────
DG0 = -14.0
sigma_star = 0.06
sigma = sigma_star / R_ee**2
Nrep_ratios = [0, 5, 10, 20]
NL_values   = np.arange(1, 31)

data_template = {
    'kEff':        k_ee,
    'kEffRep':     k_ee,
    'x0':          0,
    'NL':          5,
    'Nrep':        0,
    'kbT':         kbT,
    'DG0':         DG0,
    'nIntSamples': 150,
    'verbose':     False,
    'NP_type':     'star_polymer',
    'Nkuhn':       Nkuhn,
    'amono':       amono,
    'akuhn':       akuhn,
    'sigma':       sigma,
    'cV0':         1e-9,
    'cNP0':        1e-4,
    'rV':          100,
    'rNP':         10,
    'epsilon_self': 1e-7,
}

# ── Main sweep ───────────────────────────────────────────────────────
results = {}

for ratio in Nrep_ratios:
    print(f"\n{'='*60}")
    print(f"N_steric/N_L = {ratio},  DG0 = {DG0}")
    print(f"{'='*60}")

    Fr_list = []; Fz_list = []; Ftot_list = []; theta_list = []

    for NL in NL_values:
        Nrep = ratio * int(NL)
        data = dict(data_template)
        data['NL']   = int(NL)
        data['Nrep'] = int(Nrep)
        rLimit(data)

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _, Fr = averageForce(direction='r', data=data)
                _, Fz = averageForce(direction='z', data=data)

            Fr_star   = Fr * R_ee / kbT
            Fz_star   = Fz * R_ee / kbT
            Ftot_star = np.sqrt(Fr_star**2 + Fz_star**2)
            theta_deg = np.degrees(np.arctan2(abs(Fz_star), abs(Fr_star)))

            print(f"  NL={NL:3d}  Fr*={Fr_star:+.4f}  Fz*={Fz_star:+.4f}"
                  f"  |F*|={Ftot_star:.4f}  theta={theta_deg:.2f}°"
                  f"  check={np.sqrt(Fr_star**2+Fz_star**2) - Ftot_star:.2e}")
        except Exception as e:
            print(f"  NL={NL:3d}  ERROR: {e}")
            Fr_star = Fz_star = Ftot_star = theta_deg = np.nan

        Fr_list.append(Fr_star)
        Fz_list.append(Fz_star)
        Ftot_list.append(Ftot_star)
        theta_list.append(theta_deg)

    results[ratio] = (NL_values.tolist(), Fr_list, Fz_list, Ftot_list, theta_list)

# ── Verify |F*| = sqrt(Fr²+Fz²) ─────────────────────────────────────
print("\n── Verification: max |Ftot - sqrt(Fr²+Fz²)| ──")
for ratio in Nrep_ratios:
    NLs, Fr_l, Fz_l, Ft_l, _ = results[ratio]
    Fr_a = np.array(Fr_l); Fz_a = np.array(Fz_l); Ft_a = np.array(Ft_l)
    residual = np.nanmax(np.abs(Ft_a - np.sqrt(Fr_a**2 + Fz_a**2)))
    print(f"  N_steric/N_L = {ratio:2d}  max residual = {residual:.2e}")

# ── Plot: 4 panels ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

colors       = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
panel_labels = [r"$|F_r^*|$", r"$|F_z^*|$", r"$|\mathbf{F}^*|$",
                r"$\theta^* = \arctan(|F_z^*|/|F_r^*|)$ [deg]"]

# Shared y-range for the first 3 panels only
all_force = []
for ratio in Nrep_ratios:
    _, Fr_l, Fz_l, Ft_l, _ = results[ratio]
    all_force.extend([np.abs(Fr_l), np.abs(Fz_l), Ft_l])
all_force = np.concatenate(all_force)
all_force = all_force[np.isfinite(all_force)]
y_min, y_max = all_force.min(), all_force.max()
pad = 0.05 * (y_max - y_min)
y_lim = (y_min - pad, y_max + pad)

for panel_idx, ax in enumerate(axes):
    for idx, ratio in enumerate(Nrep_ratios):
        NLs, Fr_l, Fz_l, Ft_l, theta_l = results[ratio]

        if panel_idx == 0:
            ydata = np.abs(Fr_l)
        elif panel_idx == 1:
            ydata = np.abs(Fz_l)
        elif panel_idx == 2:
            ydata = np.array(Ft_l)
        else:
            ydata = np.array(theta_l)

        x = np.array(NLs, dtype=float)
        y = np.array(ydata, dtype=float)
        mask = np.isfinite(y)
        if mask.sum() >= 3:
            spl = make_interp_spline(x[mask], y[mask], k=2)
            x_smooth = np.linspace(x[mask].min(), x[mask].max(), 300)
            y_smooth = spl(x_smooth)
        else:
            x_smooth, y_smooth = x[mask], y[mask]

        label = rf"$N_{{\rm steric}}/N_L = {ratio}$" if panel_idx == 0 else None
        ax.plot(x_smooth, y_smooth, '-', color=colors[idx],
                linewidth=1.5, label=label)

    ax.set_xlabel(r"$N_L$", fontsize=13)
    ax.set_ylabel(panel_labels[panel_idx], fontsize=12)
    if panel_idx == 0:
        ax.legend(fontsize=9, framealpha=0.9)
    if panel_idx < 3:
        ax.set_ylim(y_lim)

fig.tight_layout()

outdir = os.path.dirname(os.path.abspath(__file__))
for ext in ('pdf', 'png'):
    outpath = os.path.join(outdir, f"Fig5_B_2nd_version.{ext}")
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {outpath}")
