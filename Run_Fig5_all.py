#!/usr/bin/env python3
"""
Consolidated Figure 5 script.
Generates: NEW_Fig5, NEW_Fig5_SI, Fig5_B_2nd_version, NEW_Fig5_C_2nd_version
  - NEW_Fig5:    |Fr*|, |Fz*|, |F*| vs DG0 with KD axis, N_steric/NL ratios
  - NEW_Fig5_SI: same with custom 1/z^2 repulsion, A0* values, KD axis
  - Fig5_B_2nd_version: 4 panels (Fr, Fz, |F|, theta) vs NL
  - NEW_Fig5_C_2nd_version: 4 panels (Fr, Fz, |F|, theta) vs DG0 with KD axis
"""

import sys, os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Theory_v9_mp as theory
from Theory_v9_mp import rLimit, averageForce

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


def make_data(NL_val, sigma_val):
    return {
        'kEff': k_ee, 'kEffRep': k_ee, 'x0': 0,
        'NL': NL_val, 'Nrep': 0, 'kbT': kbT, 'DG0': -10.0,
        'nIntSamples': 150, 'verbose': False,
        'NP_type': 'star_polymer',
        'Nkuhn': Nkuhn, 'amono': amono, 'akuhn': akuhn,
        'sigma': sigma_val, 'cV0': 1e-9, 'cNP0': 1e-4,
        'rV': 100, 'rNP': 10, 'epsilon_self': 1e-7,
    }


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


# ════════════════════════════════════════════════════════════════════
# Load or compute Fig5 data (DG0 sweep, NL=5, 4 ratios)
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nFig5 data\n" + "="*60)

cache_fig5 = os.path.join(basedir, 'Fig5_cache.npz')

if os.path.exists(cache_fig5):
    print(f"  Loading {cache_fig5}")
    c = np.load(cache_fig5, allow_pickle=True)
    res5 = {}
    for ratio in Nrep_ratios:
        Fr = c[f'Fr_{ratio}']
        Fz = c[f'Fz_{ratio}']
        Ftot = c[f'Ftot_{ratio}']
        theta = c[f'theta_{ratio}'] if f'theta_{ratio}' in c else \
                np.degrees(np.arctan2(np.abs(Fz), np.abs(Fr)))
        res5[ratio] = {'Fr': Fr, 'Fz': Fz, 'Ftot': Ftot, 'theta': theta}
else:
    NL = 5
    res5 = {}
    for ratio in Nrep_ratios:
        Nrep = ratio * NL
        print(f"  ratio={ratio}, Nrep={Nrep}")
        Fr_l, Fz_l, Ft_l, th_l = [], [], [], []
        for DG0 in DG0_values:
            data = make_data(NL, sigma)
            data['DG0'] = DG0
            data['Nrep'] = Nrep
            rLimit(data)
            try:
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    _, Fr = averageForce(direction='r', data=data)
                    _, Fz = averageForce(direction='z', data=data)
                Fr_s = Fr * R_ee / kbT
                Fz_s = Fz * R_ee / kbT
                Ft_s = np.sqrt(Fr_s**2 + Fz_s**2)
                th_s = np.degrees(np.arctan2(abs(Fz_s), abs(Fr_s)))
            except Exception as e:
                print(f"    DG0={DG0:.1f} ERROR: {e}")
                Fr_s = Fz_s = Ft_s = th_s = np.nan
            Fr_l.append(Fr_s); Fz_l.append(Fz_s)
            Ft_l.append(Ft_s); th_l.append(th_s)
        res5[ratio] = {
            'Fr': np.abs(np.array(Fr_l)), 'Fz': np.abs(np.array(Fz_l)),
            'Ftot': np.array(Ft_l), 'theta': np.array(th_l),
        }
    sd = {}
    for r in Nrep_ratios:
        for k in ('Fr', 'Fz', 'Ftot', 'theta'):
            sd[f'{k}_{r}'] = res5[r][k]
    np.savez(cache_fig5, **sd)
    print(f"  Cached -> {cache_fig5}")


# ════════════════════════════════════════════════════════════════════
# NEW_Fig5 (3 panels with KD axis)
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nNEW_Fig5\n" + "="*60)

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
keys5 = ['Fr', 'Fz', 'Ftot']
labels5 = [r"$|F_r^*|$", r"$|F_z^*|$", r"$|\mathbf{F}^*|$"]

all_y5 = np.concatenate([res5[r][k] for r in Nrep_ratios for k in keys5])
all_y5 = all_y5[np.isfinite(all_y5)]
pad = 0.05 * (all_y5.max() - all_y5.min())
ylim5 = (all_y5.min() - pad, all_y5.max() + pad)

for pi, (ax, key, plabel) in enumerate(zip(axes, keys5, labels5)):
    for idx, ratio in enumerate(Nrep_ratios):
        xs, ys = smooth(DG0_values, res5[ratio][key])
        label = rf"$N_{{\rm steric}}/N_L = {ratio}$" if pi == 0 else None
        ax.plot(xs, ys, '-', color=colors[idx], linewidth=1.5, label=label)
    ax.set_xlabel(r"$\Delta G^0\;(k_BT)$", fontsize=13)
    ax.set_ylabel(plabel, fontsize=13)
    ax.set_ylim(ylim5)
    add_KD_axis(ax)
    if pi == 0:
        ax.legend(fontsize=9, framealpha=0.9)

fig.tight_layout()
save_fig(fig, "NEW_Fig5")


# ════════════════════════════════════════════════════════════════════
# NEW_Fig5_C_2nd_version (4 panels: Fr, Fz, |F|, theta with KD axis)
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nNEW_Fig5_C_2nd_version\n" + "="*60)

fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
keysC = ['Fr', 'Fz', 'Ftot', 'theta']
labelsC = [r"$|F_r^*|$", r"$|F_z^*|$", r"$|\mathbf{F}^*|$",
           r"$\theta^* = \arctan(|F_z^*|/|F_r^*|)$ [deg]"]

all_fC = np.concatenate([res5[r][k] for r in Nrep_ratios for k in ('Fr', 'Fz', 'Ftot')])
all_fC = all_fC[np.isfinite(all_fC)]
ylimC = (all_fC.min() - 0.05*(all_fC.max()-all_fC.min()),
         all_fC.max() + 0.05*(all_fC.max()-all_fC.min()))

for pi, (ax, key, plabel) in enumerate(zip(axes, keysC, labelsC)):
    for idx, ratio in enumerate(Nrep_ratios):
        xs, ys = smooth(DG0_values, res5[ratio][key])
        label = rf"$N_{{\rm steric}}/N_L = {ratio}$" if pi == 0 else None
        ax.plot(xs, ys, '-', color=colors[idx], linewidth=1.5, label=label)
    ax.set_xlabel(r"$\Delta G^0\;(k_BT)$", fontsize=13)
    ax.set_ylabel(plabel, fontsize=12)
    if pi < 3:
        ax.set_ylim(ylimC)
    add_KD_axis(ax)
    if pi == 0:
        ax.legend(fontsize=9, framealpha=0.9)

fig.tight_layout()
save_fig(fig, "NEW_Fig5_C_2nd_version")


# ════════════════════════════════════════════════════════════════════
# NEW_Fig5_SI (1/z^2 repulsion, 3 panels with KD axis)
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nNEW_Fig5_SI\n" + "="*60)

A0_star_values = [0, 5, 20, 50]
zc_SI = 0.13 * R_ee

cache_SI = os.path.join(basedir, 'Fig5_SI_cache.npz')

if os.path.exists(cache_SI):
    print(f"  Loading {cache_SI}")
    csi = np.load(cache_SI, allow_pickle=True)
    res_si = {}
    for a in A0_star_values:
        res_si[a] = {k: csi[f'{k}_{a}'] for k in ('Fr', 'Fz', 'Ftot')}
else:
    _orig_ARep = theory.ARep
    def _custom_ARep(z, data):
        A0 = data.get('A0_custom', 0.0)
        zc = data.get('zc_custom', 1.0)
        return 0.0 if A0 == 0.0 else A0 / (z**2 + zc**2)
    theory.ARep = _custom_ARep

    res_si = {}
    for A0_star in A0_star_values:
        A0_phys = A0_star * kbT * R_ee**2
        print(f"  A0*={A0_star}")
        Fr_l, Fz_l, Ft_l = [], [], []
        for DG0 in DG0_values:
            data = make_data(5, sigma)
            data['DG0'] = DG0
            data['A0_custom'] = A0_phys
            data['zc_custom'] = zc_SI
            theory.rLimit(data)
            try:
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    _, Fr = theory.averageForce(direction='r', data=data)
                    _, Fz = theory.averageForce(direction='z', data=data)
                Fr_s = Fr * R_ee / kbT
                Fz_s = Fz * R_ee / kbT
                Fr_l.append(Fr_s); Fz_l.append(Fz_s)
                Ft_l.append(np.sqrt(Fr_s**2 + Fz_s**2))
            except Exception as e:
                Fr_l.append(np.nan); Fz_l.append(np.nan); Ft_l.append(np.nan)
        res_si[A0_star] = {
            'Fr': np.abs(np.array(Fr_l)),
            'Fz': np.abs(np.array(Fz_l)),
            'Ftot': np.array(Ft_l),
        }
    theory.ARep = _orig_ARep

    sd = {f'{k}_{a}': res_si[a][k] for a in A0_star_values for k in ('Fr', 'Fz', 'Ftot')}
    np.savez(cache_SI, **sd)
    print(f"  Cached -> {cache_SI}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
keys_si = ['Fr', 'Fz', 'Ftot']
labels_si = [r"$|F_r^*|$", r"$|F_z^*|$", r"$|\mathbf{F}^*|$"]

for pi, (ax, key, plabel) in enumerate(zip(axes, keys_si, labels_si)):
    for idx, A0_star in enumerate(A0_star_values):
        xs, ys = smooth(DG0_values, res_si[A0_star][key])
        label = rf"$A_0^* = {A0_star}$" if pi == 0 else None
        ax.plot(xs, ys, '-', color=colors[idx], linewidth=1.5, label=label)
    ax.set_xlabel(r"$\Delta G^0\;(k_BT)$", fontsize=13)
    ax.set_ylabel(plabel, fontsize=13)
    add_KD_axis(ax)
    if pi == 0:
        ax.legend(fontsize=9, framealpha=0.9)

fig.tight_layout()
save_fig(fig, "NEW_Fig5_SI")


# ════════════════════════════════════════════════════════════════════
# Fig5_B_2nd_version (4 panels vs NL, fixed DG0=-14)
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nFig5_B_2nd_version\n" + "="*60)

DG0_B = -14.0
NL_values_B = np.arange(1, 31)

results_B = {}
for ratio in Nrep_ratios:
    print(f"  ratio={ratio}, DG0={DG0_B}")
    Fr_l, Fz_l, Ft_l, th_l = [], [], [], []
    for NL in NL_values_B:
        Nrep = ratio * int(NL)
        data = make_data(int(NL), sigma)
        data['DG0'] = DG0_B
        data['Nrep'] = int(Nrep)
        rLimit(data)
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                _, Fr = averageForce(direction='r', data=data)
                _, Fz = averageForce(direction='z', data=data)
            Fr_s = Fr * R_ee / kbT
            Fz_s = Fz * R_ee / kbT
            Ft_s = np.sqrt(Fr_s**2 + Fz_s**2)
            th_s = np.degrees(np.arctan2(abs(Fz_s), abs(Fr_s)))
        except Exception as e:
            print(f"    NL={NL} ERROR: {e}")
            Fr_s = Fz_s = Ft_s = th_s = np.nan
        Fr_l.append(Fr_s); Fz_l.append(Fz_s)
        Ft_l.append(Ft_s); th_l.append(th_s)
    results_B[ratio] = (NL_values_B.tolist(), Fr_l, Fz_l, Ft_l, th_l)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
panel_labels_B = [r"$|F_r^*|$", r"$|F_z^*|$", r"$|\mathbf{F}^*|$",
                  r"$\theta^* = \arctan(|F_z^*|/|F_r^*|)$ [deg]"]

all_fB = []
for ratio in Nrep_ratios:
    _, Fr_l, Fz_l, Ft_l, _ = results_B[ratio]
    all_fB.extend([np.abs(Fr_l), np.abs(Fz_l), Ft_l])
all_fB = np.concatenate(all_fB)
all_fB = all_fB[np.isfinite(all_fB)]
padB = 0.05 * (all_fB.max() - all_fB.min())
ylimB = (all_fB.min() - padB, all_fB.max() + padB)

for pi, ax in enumerate(axes):
    for idx, ratio in enumerate(Nrep_ratios):
        NLs, Fr_l, Fz_l, Ft_l, th_l = results_B[ratio]
        if pi == 0: ydata = np.abs(Fr_l)
        elif pi == 1: ydata = np.abs(Fz_l)
        elif pi == 2: ydata = np.array(Ft_l)
        else: ydata = np.array(th_l)

        x = np.array(NLs, dtype=float)
        y = np.array(ydata, dtype=float)
        xs, ys = smooth(x, y)

        label = rf"$N_{{\rm steric}}/N_L = {ratio}$" if pi == 0 else None
        ax.plot(xs, ys, '-', color=colors[idx], linewidth=1.5, label=label)

    ax.set_xlabel(r"$N_L$", fontsize=13)
    ax.set_ylabel(panel_labels_B[pi], fontsize=12)
    if pi == 0:
        ax.legend(fontsize=9, framealpha=0.9)
    if pi < 3:
        ax.set_ylim(ylimB)

fig.tight_layout()
save_fig(fig, "Fig5_B_2nd_version")

print("\nAll Fig5 variants done.")
