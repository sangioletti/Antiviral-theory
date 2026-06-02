#!/usr/bin/env python3
"""
Consolidated Figure 5 2D heatmap script.
Generates: Fig5_2D, NEW_Fig5_2D
  - Fig5_2D:     4x3 heatmaps (ratios x force components) vs DG0 and NL
  - NEW_Fig5_2D: same with KD secondary axis on top row
"""

import sys, os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Theory_v9_mp import *

basedir = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(basedir, 'final_plots')
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
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.top': True,
    'ytick.right': True,
    'font.size': 11,
})

# ── Physical parameters ───────────────────────────────────────────────
Nmono = 20
amono = 0.38
akuhn = 0.76
Nkuhn = Nmono * amono / akuhn
kbT   = 1.0

R_ee = np.sqrt(Nkuhn) * akuhn
k_ee = 3.0 * kbT / R_ee**2
sigma_star = 0.06
sigma = sigma_star / R_ee**2

print(f"Nkuhn = {Nkuhn},  R_ee = {R_ee:.4f} nm,  k_ee = {k_ee:.6f} kbT/nm^2")

Nrep_ratios = [0, 5, 10, 20]
DG0_values = np.linspace(-20, 4, 39)
NL_values = np.arange(1, 31)


def save_fig(fig, name):
    for ext in ('pdf', 'png'):
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p, dpi=150, bbox_inches='tight')
        print(f"  Saved {p}")
    plt.close(fig)


def add_KD_axis(ax):
    ax2 = ax.twiny()
    dg_min, dg_max = ax.get_xlim()
    ax2.set_xlim(np.exp(dg_min), np.exp(dg_max))
    ax2.set_xscale('log')
    ax2.set_xlabel(r"$K_D\;(\mathrm{M})$", fontsize=11)
    ax2.tick_params(direction='in', which='both')
    return ax2


# ── Load or compute ──────────────────────────────────────────────────
cache_path = os.path.join(basedir, 'Fig5_2D_cache.npz')

if os.path.exists(cache_path):
    print(f"Loading {cache_path}")
    cache = np.load(cache_path)
    results = {}
    for ratio in Nrep_ratios:
        results[ratio] = {
            'Fr':   cache[f'Fr_{ratio}'],
            'Fz':   cache[f'Fz_{ratio}'],
            'Ftot': cache[f'Ftot_{ratio}'],
        }
else:
    data_template = {
        'kEff': k_ee, 'kEffRep': k_ee, 'x0': 0,
        'NL': 5, 'Nrep': 0, 'kbT': kbT, 'DG0': -10.0,
        'nIntSamples': 150, 'verbose': False,
        'NP_type': 'star_polymer',
        'Nkuhn': Nkuhn, 'amono': amono, 'akuhn': akuhn,
        'sigma': sigma, 'cV0': 1e-9, 'cNP0': 1e-4,
        'rV': 100, 'rNP': 10, 'epsilon_self': 1e-7,
    }

    results = {}
    n_DG0 = len(DG0_values)
    n_NL = len(NL_values)

    for ratio in Nrep_ratios:
        Fr_grid = np.full((n_NL, n_DG0), np.nan)
        Fz_grid = np.full((n_NL, n_DG0), np.nan)
        Ftot_grid = np.full((n_NL, n_DG0), np.nan)

        print(f"  N_steric/N_L = {ratio}")
        for j, NL in enumerate(NL_values):
            Nrep = ratio * int(NL)
            for i, DG0 in enumerate(DG0_values):
                data = dict(data_template)
                data['NL'] = int(NL)
                data['Nrep'] = int(Nrep)
                data['DG0'] = DG0
                rLimit(data)
                try:
                    with warnings.catch_warnings(record=True):
                        warnings.simplefilter("always")
                        _, Fr = averageForce(direction='r', data=data)
                        _, Fz = averageForce(direction='z', data=data)
                    Fr_star = Fr * R_ee / kbT
                    Fz_star = Fz * R_ee / kbT
                    Fr_grid[j, i] = Fr_star
                    Fz_grid[j, i] = Fz_star
                    Ftot_grid[j, i] = np.sqrt(Fr_star**2 + Fz_star**2)
                except Exception as e:
                    pass
            print(f"    NL={NL:3d} done")

        results[ratio] = {
            'Fr': np.abs(Fr_grid),
            'Fz': np.abs(Fz_grid),
            'Ftot': Ftot_grid,
        }

    sd = {}
    for ratio in Nrep_ratios:
        for k in ('Fr', 'Fz', 'Ftot'):
            sd[f'{k}_{ratio}'] = results[ratio][k]
    np.savez(cache_path, **sd)
    print(f"  Cached -> {cache_path}")


# ── Plotting helper ──────────────────────────────────────────────────
panel_keys = ['Fr', 'Fz', 'Ftot']
panel_labels = [r"$|F_r^*|$", r"$|F_z^*|$", r"$|\mathbf{F}^*|$"]
VMIN, VMAX = 0.0, 8.0
CONTOUR_LEVELS = [1, 2, 3, 4, 5, 6, 7]


def plot_2d(output_name, with_kd=False):
    fig, axes = plt.subplots(4, 3, figsize=(13, 14), constrained_layout=True)

    for row, ratio in enumerate(Nrep_ratios):
        for col, (key, plabel) in enumerate(zip(panel_keys, panel_labels)):
            ax = axes[row, col]
            img = results[ratio][key]

            pc = ax.pcolormesh(DG0_values, NL_values, img,
                               cmap='viridis', vmin=VMIN, vmax=VMAX,
                               shading='nearest')
            cs = ax.contour(DG0_values, NL_values, img,
                            levels=CONTOUR_LEVELS,
                            colors='white', linewidths=0.7, alpha=0.7)
            ax.clabel(cs, fmt='%d', fontsize=7, inline=True)

            if row == 0:
                ax.set_title(plabel, fontsize=13)
                if with_kd:
                    add_KD_axis(ax)

            ax.set_ylabel(r"$N_L$", fontsize=11)
            if col == 0:
                ax.text(-0.42, 0.5, rf"$N_{{\rm steric}}/N_L={ratio}$",
                        transform=ax.transAxes,
                        va='center', ha='center', rotation=90, fontsize=11)

            if row == 3:
                ax.set_xlabel(r"$\Delta G^0\;(k_BT)$", fontsize=11)

    cbar = fig.colorbar(pc, ax=axes, orientation='vertical',
                        fraction=0.02, pad=0.02, label=r"$F^*$")
    cbar.ax.tick_params(labelsize=9)
    save_fig(fig, output_name)


# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nFig5_2D\n" + "="*60)
plot_2d("Fig5_2D", with_kd=False)

print("\n" + "="*60 + "\nNEW_Fig5_2D\n" + "="*60)
plot_2d("NEW_Fig5_2D", with_kd=True)

print("\nAll Fig5_2D variants done.")
