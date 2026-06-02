#!/usr/bin/env python3
"""
SI Figure: Effect of a generic 1/z^2 repulsive interaction on the force.

Three panels: F_r* (left), F_z* (middle), |F*| (right) vs DG0,
for different dimensionless repulsion amplitudes A_0*.

A_steric(z) = A_0 / (z^2 + z_c^2)
A_0* = A_0 / (kbT * R_ee^2)
z_c* = z_c / R_ee ≈ 0.13

Parameters: sigma*_R = 0.06, NL = 5, Nrep = 0.
"""

import sys, os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Theory_v9_mp as theory

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
NL = 5
sigma_star = 0.06
sigma = sigma_star / R_ee**2
zc_star = 0.13                        # z_c / R_ee
zc = zc_star * R_ee                   # physical z_c in nm
A0_star_values = [0, 5, 20, 50]       # A_0 / (kbT * R_ee^2)

DG0_values = np.linspace(-20, 4, 39)

print(f"sigma* = {sigma_star},  sigma = {sigma:.6e}")
print(f"z_c*   = {zc_star},  z_c = {zc:.4f} nm")
print(f"A_0* values: {A0_star_values}")

# ── Monkey-patch ARep with custom 1/z^2 repulsion ────────────────────
_original_ARep = theory.ARep

def _custom_ARep(z, data):
    A0 = data.get('A0_custom', 0.0)
    zc_val = data.get('zc_custom', 1.0)
    if A0 == 0.0:
        return 0.0
    return A0 / (z**2 + zc_val**2)

theory.ARep = _custom_ARep

# ── Base data dictionary ─────────────────────────────────────────────
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
    'sigma':    sigma,
    'cV0':      1e-9,
    'cNP0':     1e-4,
    'rV':       100,
    'rNP':      10,
    'epsilon_self': 1e-7,
    'zc_custom': zc,
}

# ── Main sweep ───────────────────────────────────────────────────────
# results[A0_star] = (DG0_list, Fr_star_list, Fz_star_list, Ftot_star_list)
results = {}

for A0_star in A0_star_values:
    A0 = A0_star * kbT * R_ee**2      # physical A_0
    print(f"\n{'='*60}")
    print(f"A_0* = {A0_star},  A_0 = {A0:.4f} kbT nm^2")
    print(f"{'='*60}")

    Fr_list = []
    Fz_list = []
    Ftot_list = []

    for DG0 in DG0_values:
        data = dict(data_template)
        data['DG0'] = DG0
        data['A0_custom'] = A0

        theory.rLimit(data)

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _, Fr = theory.averageForce(direction='r', data=data)
                _, Fz = theory.averageForce(direction='z', data=data)
                for warning in w:
                    print(f"  [WARN DG0={DG0}] {warning.message}")

            Fr_star = Fr * R_ee / kbT
            Fz_star = Fz * R_ee / kbT
            Ftot_star = np.sqrt(Fr_star**2 + Fz_star**2)
            print(f"  DG0={DG0:6.1f}  Fr*={Fr_star:+.6e}  Fz*={Fz_star:+.6e}  |F*|={Ftot_star:.6e}")
        except Exception as e:
            print(f"  DG0={DG0:6.1f}  ERROR: {e}")
            Fr_star = np.nan
            Fz_star = np.nan
            Ftot_star = np.nan

        Fr_list.append(Fr_star)
        Fz_list.append(Fz_star)
        Ftot_list.append(Ftot_star)

    results[A0_star] = (DG0_values.tolist(), Fr_list, Fz_list, Ftot_list)

# Restore original ARep
theory.ARep = _original_ARep

# ── Plot: 3 subplots ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
panel_labels = [r"$|F_r^*|$", r"$|F_z^*|$", r"$|\mathbf{F}^*|$"]

for panel_idx, ax in enumerate(axes):
    for idx, A0_star in enumerate(A0_star_values):
        DG0s, Fr_list, Fz_list, Ftot_list = results[A0_star]

        if panel_idx == 0:
            ydata = np.abs(Fr_list)
        elif panel_idx == 1:
            ydata = np.abs(Fz_list)
        else:
            ydata = np.array(Ftot_list)

        x = np.array(DG0s, dtype=float)
        y = np.array(ydata, dtype=float)
        mask = np.isfinite(y)
        if mask.sum() >= 4:
            spl = make_interp_spline(x[mask], y[mask], k=2)
            x_smooth = np.linspace(x[mask].min(), x[mask].max(), 300)
            y_smooth = spl(x_smooth)
        else:
            x_smooth, y_smooth = x[mask], y[mask]

        label = rf"$A_0^* = {A0_star}$" if panel_idx == 0 else None
        ax.plot(x_smooth, y_smooth, '-', color=colors[idx],
                linewidth=1.5, label=label)

    ax.set_xlabel(r"$\Delta G^0\;(k_BT)$", fontsize=13)
    ax.set_ylabel(panel_labels[panel_idx], fontsize=13)
    if panel_idx == 0:
        ax.legend(fontsize=9, framealpha=0.9)

fig.tight_layout()

outdir = os.path.dirname(os.path.abspath(__file__))
for ext in ('pdf', 'png'):
    outpath = os.path.join(outdir, f"Fig4-SI.{ext}")
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {outpath}")
