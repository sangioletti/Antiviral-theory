#!/usr/bin/env python3
"""
2D heatmap summary of Fig5 and Fig5_B:
  x-axis: Delta G0 (swept as in Run_Fig5.py)
  y-axis: N_L     (swept as in Run_Fig5_B.py)
  Layout: 4 rows x 3 columns
    rows    -> N_steric/N_L = [0, 5, 10, 20]
    columns -> |F_r*|, |F_z*|, |F*|

Computed grids are cached in Fig5_2D_cache.npz so the sweep only runs once.

Parameters (consistent with Run_Fig5.py):
    Nmono = 20, amono = 0.38 nm, akuhn = 0.76 nm
    Nkuhn = Nmono * amono / akuhn = 10
    R_ee  = sqrt(Nkuhn) * akuhn
    k_ee  = 3 kbT / R_ee^2
    F*    = F * R_ee / kbT
    sigma*= sigma * R_ee^2
"""

import sys, os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.top': True,
    'ytick.right': True,
    'font.size': 11,
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

# ── Sweep axes ───────────────────────────────────────────────────────
sigma_star  = 0.06
sigma       = sigma_star / R_ee**2
Nrep_ratios = [0, 5, 10, 20]

DG0_values = np.linspace(-20, 4, 39)
NL_values  = np.arange(1, 31)          # 1 to 30 inclusive

outdir     = os.path.dirname(os.path.abspath(__file__))
cache_path = os.path.join(outdir, 'Fig5_2D_cache.npz')

# ── Load or compute ───────────────────────────────────────────────────
if os.path.exists(cache_path):
    print(f"Loading cached results from {cache_path}")
    cache = np.load(cache_path)
    results = {}
    for ratio in Nrep_ratios:
        results[ratio] = {
            'Fr':   cache[f'Fr_{ratio}'],
            'Fz':   cache[f'Fz_{ratio}'],
            'Ftot': cache[f'Ftot_{ratio}'],
        }
else:
    # ── Base data dictionary ──────────────────────────────────────────
    data_template = {
        'kEff':        k_ee,
        'kEffRep':     k_ee,
        'x0':          0,
        'NL':          5,
        'Nrep':        0,
        'kbT':         kbT,
        'DG0':         -10.0,
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

    results = {}
    n_DG0 = len(DG0_values)
    n_NL  = len(NL_values)

    for ratio in Nrep_ratios:
        Fr_grid   = np.full((n_NL, n_DG0), np.nan)
        Fz_grid   = np.full((n_NL, n_DG0), np.nan)
        Ftot_grid = np.full((n_NL, n_DG0), np.nan)

        print(f"\n{'='*70}")
        print(f"N_steric/N_L = {ratio}")
        print(f"{'='*70}")

        for j, NL in enumerate(NL_values):
            Nrep = ratio * int(NL)
            for i, DG0 in enumerate(DG0_values):
                data = dict(data_template)
                data['NL']   = int(NL)
                data['Nrep'] = int(Nrep)
                data['DG0']  = DG0

                rLimit(data)

                try:
                    with warnings.catch_warnings(record=True):
                        warnings.simplefilter("always")
                        _, Fr = averageForce(direction='r', data=data)
                        _, Fz = averageForce(direction='z', data=data)

                    Fr_star   = Fr * R_ee / kbT
                    Fz_star   = Fz * R_ee / kbT
                    Ftot_star = np.sqrt(Fr_star**2 + Fz_star**2)

                    Fr_grid[j, i]   = Fr_star
                    Fz_grid[j, i]   = Fz_star
                    Ftot_grid[j, i] = Ftot_star
                except Exception as e:
                    print(f"  NL={NL:3d}  DG0={DG0:6.1f}  ERROR: {e}")

            print(f"  NL={NL:3d}  Nrep={Nrep:4d}  done")

        results[ratio] = {
            'Fr':   np.abs(Fr_grid),
            'Fz':   np.abs(Fz_grid),
            'Ftot': Ftot_grid,
        }

    # Save cache
    save_dict = {}
    for ratio in Nrep_ratios:
        save_dict[f'Fr_{ratio}']   = results[ratio]['Fr']
        save_dict[f'Fz_{ratio}']   = results[ratio]['Fz']
        save_dict[f'Ftot_{ratio}'] = results[ratio]['Ftot']
    np.savez(cache_path, **save_dict)
    print(f"\nResults cached to {cache_path}")

# ── Plot: 4 rows x 3 columns of heatmaps ────────────────────────────
panel_keys   = ['Fr',   'Fz',   'Ftot']
panel_labels = [r"$|F_r^*|$", r"$|F_z^*|$", r"$|\mathbf{F}^*|$"]

VMIN, VMAX = 0.0, 8.0   # shared colour scale for all panels
CONTOUR_LEVELS = [1, 2, 3, 4, 5, 6, 7]

# extent: [x_left, x_right, y_bottom, y_top]
extent = [DG0_values[0], DG0_values[-1], NL_values[0], NL_values[-1]]

fig, axes = plt.subplots(
    4, 3,
    figsize=(13, 14),
    constrained_layout=True,
)

for row, ratio in enumerate(Nrep_ratios):
    for col, (key, plabel) in enumerate(zip(panel_keys, panel_labels)):
        ax  = axes[row, col]
        img = results[ratio][key]

        im = ax.imshow(
            img,
            origin='lower',
            aspect='auto',
            extent=extent,
            cmap='viridis',
            vmin=VMIN,
            vmax=VMAX,
        )

        # Contour lines at integer force values
        cs = ax.contour(
            DG0_values, NL_values, img,
            levels=CONTOUR_LEVELS,
            colors='white',
            linewidths=0.7,
            alpha=0.7,
        )
        ax.clabel(cs, fmt='%d', fontsize=7, inline=True)

        # Column header (top row only)
        if row == 0:
            ax.set_title(plabel, fontsize=13)

        # y-axis label
        ax.set_ylabel(r"$N_L$", fontsize=11)

        # Row label (leftmost column only): placed further left than the ylabel
        if col == 0:
            ax.text(
                -0.42, 0.5,
                rf"$N_{{\rm steric}}/N_L={ratio}$",
                transform=ax.transAxes,
                va='center', ha='center',
                rotation=90, fontsize=11,
            )

        # x-label (bottom row only)
        if row == 3:
            ax.set_xlabel(r"$\Delta G^0\;(k_BT)$", fontsize=11)

# Single shared colorbar on the right
cbar = fig.colorbar(
    im,
    ax=axes,
    orientation='vertical',
    fraction=0.02,
    pad=0.02,
    label=r"$F^*$",
)
cbar.ax.tick_params(labelsize=9)

for ext in ('pdf', 'png'):
    outpath = os.path.join(outdir, f"Fig5_2D.{ext}")
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {outpath}")
