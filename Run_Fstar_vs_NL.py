#!/usr/bin/env python3
"""
F* vs N_L plot for DG0 = -10, -20, -100  (3 subplots in one figure).

Parameters:
    N = 20, a = 2 nm (Kuhn length), R_ee = sqrt(N)*a, k_ee = 3 kbT/R_ee^2
    x(sigma) = pi * rMax^2 * sigma = 3, 6, 9, 12
    N_L from 2 to 20 (inclusive)

Plot:
    F* = F_r * R_ee / kbT  vs  N_L  (log x-axis)
    Legend with sigma* only on the last (DG0=-100) panel
    N_L^2 reference line only on the last panel
"""

import sys, os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Theory_v9_mp import *

# ── Physical parameters ───────────────────────────────────────────────
Nmono = 20 
amono = 0.38        # nm  (= Kuhn length)
akuhn = 0.76       # dimensionless multiplier so Ree = sqrt(N)*amono*akuhn
Nkuhn = Nmono * amono / akuhn 
kbT   = 1.0

#R_ee = np.sqrt(Nkuhn) * amono * akuhn
R_ee = np.sqrt(Nkuhn) * akuhn
k_ee = 3.0 * kbT / R_ee**2
#rMax_expected = Nkuhn * amono           # contour length = 120 nm
rMax_expected = Nkuhn * akuhn           # contour length = 120 nm

print(f"Nkuhn = {Nkuhn},  amono = {amono} nm,  akuhn = {akuhn}")
print(f"R_ee  = {R_ee:.4f} nm")
print(f"k_ee  = {k_ee:.6f} kbT/nm^2")
print(f"rMax  = {rMax_expected:.1f} nm  (contour length)")

# ── Sigma values from x(sigma) = pi * rMax^2 * sigma ─────────────────
x_values = [3, 6, 9, 12]
sigma_from_x = [x / (np.pi * rMax_expected**2) for x in x_values]
sigma_star   = [s * R_ee**2 for s in sigma_from_x]

print(f"\nx values:     {x_values}")
print(f"sigma values: {[f'{s:.6e}' for s in sigma_from_x]}")
print(f"sigma* values:{[f'{s:.4f}' for s in sigma_star]}")

NL_values = list(range(2, 21))
DG0_values = [-14, -18, -22]

# ── Base data dictionary ──────────────────────────────────────────────
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

# ── Main sweep ────────────────────────────────────────────────────────
# results[(DG0, x)] = (NL_list, Fstar_list)
results = {}

for DG0 in DG0_values:
    for x_val, sigma, sstar in zip(x_values, sigma_from_x, sigma_star):
        print(f"\n{'='*60}")
        print(f"DG0 = {DG0},  x = {x_val},  sigma* = {sstar:.4f},  sigma = {sigma:.6e}")
        print(f"{'='*60}")

        Fstar_list = []
        for NL in NL_values:
            data = dict(data_template)
            data['sigma'] = sigma
            data['NL'] = NL
            data['DG0'] = DG0

            rLimit(data)

            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    _, Fr = averageForce(direction='r', data=data)

                    for warning in w:
                        print(f"  [WARNING NL={NL}] {warning.message}")

                Fstar = Fr * R_ee / kbT
                print(f"  NL={NL:3d}  F_r = {Fr:+.6e}  F* = {Fstar:+.6e}")
            except Exception as e:
                print(f"  NL={NL:3d}  ERROR: {e}")
                Fstar = np.nan

            Fstar_list.append(Fstar)

        results[(DG0, x_val)] = (NL_values, Fstar_list)

# ── Plot: 3 subplots (one per DG0) ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']

for panel_idx, (DG0, ax) in enumerate(zip(DG0_values, axes)):
    is_last = (panel_idx == len(DG0_values) - 1)

    for idx, x_val in enumerate(x_values):
        NLs, Fstars = results[(DG0, x_val)]
        sstar = sigma_star[idx]
        label = rf"$\sigma^* = {sstar:.4f}$" if is_last else None
        ax.plot(NLs, np.abs(Fstars), markers[idx] + '-', color=colors[idx],
                linewidth=1.5, markersize=5, label=label)

    # Vertical lines at x(sigma) values
    for idx, x_val in enumerate(x_values):
        label_vline = rf"$N_{{R,max}}(\sigma) = {x_val}$" if is_last else None
        ax.axvline(x=x_val, color=colors[idx], linestyle=':', alpha=0.5,
                   linewidth=1.2, label=label_vline)

    # N_L^2 reference line only on last panel
    if is_last:
        NL_ref = np.linspace(10, 20, 50)
        Fstar_at_20 = [np.abs(results[(DG0, x)][1][-1]) for x in x_values]
        anchor = min(Fstar_at_20) * 0.3
        line_quad = anchor * (NL_ref / 20)**2
        ax.plot(NL_ref, line_quad, 'k-.', linewidth=1.0, alpha=0.6,
                label=r"$\sim N_L^2$")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"$N_L$", fontsize=13)
    if panel_idx == 0:
        ax.set_ylabel(r"$|F^*|$", fontsize=13)
    #if DG0 < -80 and DG0 == DG0_values[-1]:
    #    ax.set_title(rf"$-\infty$", fontsize=12)
    #else:
    ax.set_title(rf"$\Delta G_0^* = {DG0}$", fontsize=12)
    ax.grid(True, alpha=0.3, which='both')

    ax.set_xticks([2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    if is_last:
        ax.legend(fontsize=8, framealpha=0.9, ncol=1, loc='best')

fig.tight_layout()

outdir = os.path.dirname(os.path.abspath(__file__))
for ext in ('pdf', 'png'):
    outpath = os.path.join(outdir, f"Fstar_vs_NL.{ext}")
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {outpath}")
