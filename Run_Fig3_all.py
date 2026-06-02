#!/usr/bin/env python3
"""
Consolidated Figure 3 script.
Generates: Fig3, Fig3_SI
  - Fig3:    F* vs NL (log-log), 3 DG0 panels, for different sigma*
  - Fig3_SI: |Fz*| vs d_{R-R}^*, NL=20, DG0=[-6,-10,-14,-18]
"""

import sys, os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
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

R_ee = np.sqrt(Nkuhn) * akuhn
k_ee = 3.0 * kbT / R_ee**2
rMax_expected = Nkuhn * akuhn

print(f"Nkuhn = {Nkuhn},  R_ee = {R_ee:.4f} nm,  k_ee = {k_ee:.6f} kbT/nm^2")


def data_template(sigma_val):
    return {
        'kEff': k_ee, 'kEffRep': k_ee, 'x0': 0,
        'NL': 5, 'Nrep': 0, 'kbT': kbT, 'DG0': -10.0,
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


# ════════════════════════════════════════════════════════════════════
# Fig3: F* vs N_L (log-log)
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nFig3\n" + "="*60)

x_values = [3, 6, 9, 12]
sigma_from_x = [x / (np.pi * rMax_expected**2) for x in x_values]
sigma_star = [s * R_ee**2 for s in sigma_from_x]

NL_values = list(range(2, 21))
DG0_values_fig3 = [-10, -14, -18]

results_fig3 = {}
for DG0 in DG0_values_fig3:
    for x_val, sigma, sstar in zip(x_values, sigma_from_x, sigma_star):
        print(f"  DG0={DG0}, x={x_val}, sigma*={sstar:.4f}")
        Fstar_list = []
        for NL in NL_values:
            data = data_template(sigma)
            data['NL'] = NL
            data['DG0'] = DG0
            rLimit(data)
            try:
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    _, Fr = averageForce(direction='r', data=data)
                Fstar_list.append(Fr * R_ee / kbT)
            except Exception as e:
                print(f"    NL={NL} ERROR: {e}")
                Fstar_list.append(np.nan)
        results_fig3[(DG0, x_val)] = (NL_values, Fstar_list)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

for panel_idx, (DG0, ax) in enumerate(zip(DG0_values_fig3, axes)):
    is_last = (panel_idx == len(DG0_values_fig3) - 1)

    for idx, x_val in enumerate(x_values):
        NLs, Fstars = results_fig3[(DG0, x_val)]
        sstar_val = sigma_star[idx]
        x = np.array(NLs, dtype=float)
        y = np.abs(Fstars)
        mask = np.isfinite(y) & (y > 0)
        if mask.sum() >= 3:
            spl = make_interp_spline(np.log(x[mask]), np.log(y[mask]), k=2)
            x_smooth = np.linspace(x[mask].min(), x[mask].max(), 300)
            y_smooth = np.exp(spl(np.log(x_smooth)))
        else:
            x_smooth, y_smooth = x[mask], y[mask]
        label = rf"$\sigma^* = {sstar_val:.4f}$" if is_last else None
        ax.plot(x_smooth, y_smooth, '-', color=colors[idx], linewidth=1.5, label=label)

    for idx, x_val in enumerate(x_values):
        label_vline = rf"$N_{{R,max}}(\sigma) = {x_val}$" if is_last else None
        ax.axvline(x=x_val, color=colors[idx], linestyle=':', alpha=0.5,
                   linewidth=1.2, label=label_vline)

    if is_last:
        NL_ref = np.linspace(10, 20, 50)
        Fstar_at_20 = [np.abs(results_fig3[(DG0, x)][1][-1]) for x in x_values]
        anchor = min(Fstar_at_20) * 0.3
        line_quad = anchor * (NL_ref / 20)**2
        ax.plot(NL_ref, line_quad, 'k-.', linewidth=1.0, alpha=0.6,
                label=r"$\sim N_L^2$")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"$N_L$", fontsize=13)
    if panel_idx == 0:
        ax.set_ylabel(r"$|F^*|$", fontsize=13)
    ax.text(0.95, 0.05, rf"$\Delta G^0 = {DG0}\, k_BT$",
            fontsize=15, fontweight='bold',
            transform=ax.transAxes, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='none', alpha=0.8))
    ax.set_xticks([2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if is_last:
        ax.legend(fontsize=8, framealpha=0.9, ncol=1, loc='best')

fig.tight_layout()
save_fig(fig, "Fig3")


# ════════════════════════════════════════════════════════════════════
# Fig3_SI: |Fz*| vs d_{R-R}^* (NL=20, 4 DG0 values)
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nFig3_SI\n" + "="*60)

NL_SI = 20
DG0_values_SI = [-6, -10, -14, -18]
d_RR_star_range = np.linspace(0.3, 12.0, 240)
sigma_star_range = 1.0 / d_RR_star_range**2
sigma_range = sigma_star_range / R_ee**2

results_SI = {}
for DG0 in DG0_values_SI:
    print(f"  DG0 = {DG0}")
    d_list, Fstar_list = [], []
    for d_star, sigma in zip(d_RR_star_range, sigma_range):
        data = data_template(sigma)
        data['NL'] = NL_SI
        data['DG0'] = DG0
        rLimit(data)
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                _, Fz = averageForce(direction='z', data=data)
            Fstar_list.append(Fz * R_ee / kbT)
        except Exception as e:
            Fstar_list.append(np.nan)
        d_list.append(d_star)
    results_SI[DG0] = (d_list, Fstar_list)

fig, ax = plt.subplots(figsize=(8, 5))
for idx, DG0 in enumerate(DG0_values_SI):
    d_list, Fstar_list = results_SI[DG0]
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
ax.set_ylabel(r"$|F_z^*|$", fontsize=13)
ax.legend(fontsize=10, framealpha=0.9)
fig.tight_layout()
save_fig(fig, "Fig3_SI")

print("\nAll Fig3 variants done.")
