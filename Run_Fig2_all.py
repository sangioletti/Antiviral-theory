#!/usr/bin/env python3
"""
Consolidated Figure 2 script.
Generates: Fig2, Fig2_SI, Fig2_diff_rigidity
  - Fig2:      |Fr*| vs DG0, 3 sigma* panels, KD secondary axis
  - Fig2_SI:   |Fz*| vs DG0, 3 sigma* panels, KD secondary axis
  - Fig2_diff_rigidity: |Fr*| vs DG0 with Nmono=5 (stiffer polymer)
"""

import sys, os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

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
amono = 0.38        # nm
akuhn = 0.76        # nm
Nkuhn = Nmono * amono / akuhn   # = 10
kbT   = 1.0

R_ee = np.sqrt(Nkuhn) * akuhn
k_ee = 3.0 * kbT / R_ee**2

print(f"Nkuhn = {Nkuhn},  amono = {amono} nm,  akuhn = {akuhn} nm")
print(f"R_ee  = {R_ee:.4f} nm,  k_ee = {k_ee:.6f} kbT/nm^2")

# ── Shared figure parameters ────────────────────────────────────────
NL_values = [2, 4, 6, 8]
sigma_star_values = [0.06, 0.60, 6.00]
DG0_39 = np.linspace(-20, 4, 39)
DG0_75 = np.linspace(-20, 4, 75)


# ── Helpers ──────────────────────────────────────────────────────────
def make_data_template(k_ee_val, Nkuhn_val, amono_val, akuhn_val, sigma_val):
    return {
        'kEff': k_ee_val, 'kEffRep': k_ee_val, 'x0': 0,
        'NL': 5, 'Nrep': 0, 'kbT': kbT, 'DG0': -10.0,
        'nIntSamples': 150, 'verbose': False,
        'NP_type': 'star_polymer',
        'Nkuhn': Nkuhn_val, 'amono': amono_val, 'akuhn': akuhn_val,
        'sigma': sigma_val, 'cV0': 1e-9, 'cNP0': 1e-4,
        'rV': 100, 'rNP': 10, 'epsilon_self': 1e-7,
    }


def sweep_fig2(DG0_values, cache_path, k_ee_v=None, Nkuhn_v=None,
               amono_v=None, akuhn_v=None, R_ee_v=None):
    if k_ee_v is None: k_ee_v = k_ee
    if Nkuhn_v is None: Nkuhn_v = Nkuhn
    if amono_v is None: amono_v = amono
    if akuhn_v is None: akuhn_v = akuhn
    if R_ee_v is None: R_ee_v = R_ee

    if os.path.exists(cache_path):
        print(f"  Loading {cache_path}")
        c = np.load(cache_path, allow_pickle=True)
        results = {}
        for sstar in sigma_star_values:
            for NL in NL_values:
                results[(sstar, NL)] = {
                    'Fr': c.get(f'Fr_{sstar}_{NL}'),
                    'Fz': c.get(f'Fz_{sstar}_{NL}'),
                }
        return results

    results = {}
    for sstar in sigma_star_values:
        sigma = sstar / R_ee_v**2
        for NL in NL_values:
            print(f"  sigma*={sstar}, NL={NL}")
            Fr_list, Fz_list = [], []
            for DG0 in DG0_values:
                data = make_data_template(k_ee_v, Nkuhn_v, amono_v, akuhn_v, sigma)
                data['NL'] = NL
                data['DG0'] = DG0
                rLimit(data)
                try:
                    with warnings.catch_warnings(record=True):
                        warnings.simplefilter("always")
                        _, Fr = averageForce(direction='r', data=data)
                        _, Fz = averageForce(direction='z', data=data)
                    Fr_list.append(Fr * R_ee_v / kbT)
                    Fz_list.append(Fz * R_ee_v / kbT)
                except Exception as e:
                    print(f"    DG0={DG0:.1f} ERROR: {e}")
                    Fr_list.append(np.nan)
                    Fz_list.append(np.nan)
            results[(sstar, NL)] = {'Fr': np.array(Fr_list), 'Fz': np.array(Fz_list)}

    sd = {}
    for sstar in sigma_star_values:
        for NL in NL_values:
            sd[f'Fr_{sstar}_{NL}'] = results[(sstar, NL)]['Fr']
            sd[f'Fz_{sstar}_{NL}'] = results[(sstar, NL)]['Fz']
    np.savez(cache_path, **sd)
    print(f"  Cached -> {cache_path}")
    return results


def smooth(x, y, k=2, n=300):
    mask = np.isfinite(y)
    if mask.sum() < max(k + 1, 3):
        return x[mask], y[mask]
    spl = make_interp_spline(x[mask], y[mask], k=k)
    xs = np.linspace(x[mask].min(), x[mask].max(), n)
    return xs, spl(xs)


def add_KD_axis(ax):
    ax2 = ax.twiny()
    dg_min, dg_max = ax.get_xlim()
    ax2.set_xlim(np.exp(dg_min), np.exp(dg_max))
    ax2.set_xscale('log')
    ax2.set_xlabel(r"$K_D\;(\mathrm{M})$", fontsize=13)
    ax2.tick_params(direction='in', which='both')
    return ax2


def save_fig(fig, name):
    for ext in ('pdf', 'png'):
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p, dpi=150, bbox_inches='tight')
        print(f"  Saved {p}")
    plt.close(fig)


def plot_fig2(results, DG0_values, force_key, ylabel, output_name,
              add_kd=False, sigma_inside=False):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5 if add_kd else 5), sharey=False)

    all_y = np.concatenate([
        np.abs(results[(s, NL)][force_key])
        for s in sigma_star_values for NL in NL_values
        if results[(s, NL)][force_key] is not None
    ])
    all_y = all_y[np.isfinite(all_y)]
    pad = 0.05 * (all_y.max() - all_y.min())
    y_lim = (all_y.min() - pad, all_y.max() + pad)

    for pi, (sstar, ax) in enumerate(zip(sigma_star_values, axes)):
        for idx, NL in enumerate(NL_values):
            y = np.abs(results[(sstar, NL)][force_key])
            xs, ys = smooth(DG0_values, y)
            label = rf"$N_L = {NL}$" if pi == 0 else None
            ax.plot(xs, ys, '-', color=colors[idx], linewidth=1.5, label=label)

        ax.set_xlabel(r"$\Delta G^0\;(k_BT)$", fontsize=13)
        if pi == 0:
            ax.set_ylabel(ylabel, fontsize=13)
            ax.legend(fontsize=9, framealpha=0.9)
        ax.set_ylim(y_lim)

        if sigma_inside:
            ax.text(0.95, 0.95, rf"$\sigma_R^* = {sstar}$",
                    fontsize=15, fontweight='bold',
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='none', alpha=0.8))
        else:
            ax.set_title(rf"$\sigma_R^* = {sstar}$", fontsize=12)

        if add_kd:
            add_KD_axis(ax)

    fig.tight_layout()
    save_fig(fig, output_name)


# ════════════════════════════════════════════════════════════════════
# Fig2 (|Fr*| vs DG0, 39 points, KD axis)
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nFig2\n" + "="*60)
res2 = sweep_fig2(DG0_39, os.path.join(basedir, 'Fig2_cache.npz'))
plot_fig2(res2, DG0_39, 'Fr', r"$|F_r^*|$", "Fig2",
          add_kd=True, sigma_inside=True)


# ════════════════════════════════════════════════════════════════════
# Fig2_SI (|Fz*| vs DG0, 75 points, KD axis)
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nFig2_SI\n" + "="*60)
res2_si = sweep_fig2(DG0_75, os.path.join(basedir, 'Fig2_SI_cache.npz'))
plot_fig2(res2_si, DG0_75, 'Fz', r"$|F_z^*|$", "Fig2_SI",
          add_kd=True, sigma_inside=True)


# ════════════════════════════════════════════════════════════════════
# Fig2_diff_rigidity (Nmono=5, no KD axis)
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nFig2_diff_rigidity\n" + "="*60)

Nmono_stiff = 5
Nkuhn_stiff = Nmono_stiff * amono / akuhn
R_ee_stiff = np.sqrt(Nkuhn_stiff) * akuhn
k_ee_stiff = 3.0 * kbT / R_ee_stiff**2
print(f"  Stiff: Nmono={Nmono_stiff}, Nkuhn={Nkuhn_stiff}, R_ee={R_ee_stiff:.4f}")

cache_rigid = os.path.join(basedir, 'Fig2_diff_rigidity_cache.npz')
res_rigid = sweep_fig2(DG0_39, cache_rigid,
                       k_ee_v=k_ee_stiff, Nkuhn_v=Nkuhn_stiff,
                       R_ee_v=R_ee_stiff)
plot_fig2(res_rigid, DG0_39, 'Fr', r"$|F_r^*|$", "Fig2_diff_rigidity")

print("\nAll Fig2 variants done.")
