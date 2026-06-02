#!/usr/bin/env python3
"""
Remake all figures that have DeltaG0 on an x-axis.
  - Bottom x-axis: Delta G0 (linear, unchanged)
  - Top x-axis:    K_D = exp(+DG0/kbT) × 1M  (log scale)

Outputs: NEW_Fig2, NEW_Fig2_SI, NEW_Fig5, NEW_Fig5_C_2nd_version,
         NEW_Fig5_2D, NEW_Fig5_SI  (.pdf and .png)
Caches:  Fig2_cache.npz, Fig2_SI_cache.npz, Fig5_cache.npz,
         Fig5_SI_cache.npz  (reused on subsequent runs)
         Fig5_2D_cache.npz  (already exists from Run_Fig5_2D.py)
"""

import sys, os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import matplotlib.ticker as mticker

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
    'xtick.top': False,   # top ticks handled by secondary axis
    'ytick.right': True,
    'font.size': 12,
})

outdir = os.path.dirname(os.path.abspath(__file__))

# ── Physical parameters ───────────────────────────────────────────────
Nmono = 20
amono = 0.38
akuhn = 0.76
Nkuhn = Nmono * amono / akuhn
kbT   = 1.0
R_ee  = np.sqrt(Nkuhn) * akuhn
k_ee  = 3.0 * kbT / R_ee**2
sigma_star = 0.06

print(f"Nkuhn={Nkuhn},  R_ee={R_ee:.4f} nm,  k_ee={k_ee:.6f} kbT/nm²")

# ── Shared sweep axes ────────────────────────────────────────────────
Nrep_ratios = [0, 5, 10, 20]
DG0_values  = np.linspace(-20, 4, 39)
NL_values   = np.arange(1, 31)
colors      = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# ── K_D conversion ────────────────────────────────────────────────────
# K_D [M] = exp(+DG0 / kbT) × 1M   (kbT=1 in code)
# Small K_D ↔ negative DG0 ↔ tight binding
def _dg0_to_kd(dg0):
    return np.exp(np.asarray(dg0, dtype=float))

def _kd_to_dg0(kd):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(np.clip(np.asarray(kd, dtype=float), 1e-300, None))


def add_kd_axis(ax, dg0_range=(-20, 4), xlabel=True):
    """Add a secondary log K_D axis on top of ax."""
    secax = ax.secondary_xaxis('top', functions=(_dg0_to_kd, _kd_to_dg0))

    # Place ticks at every-other power of 10 within the K_D range
    kd_min = np.exp(dg0_range[0])
    kd_max = np.exp(dg0_range[1])
    n_min  = int(np.ceil(np.log10(kd_min)))
    n_max  = int(np.floor(np.log10(kd_max)))
    # Use every other decade to avoid crowding
    step   = 2 if (n_max - n_min) > 6 else 1
    exps   = list(range(n_min, n_max + 1, step))
    ticks  = [10.0**n for n in exps]
    labels = [rf"$10^{{{n}}}$" for n in exps]

    secax.set_ticks(ticks)
    secax.set_xticklabels(labels, fontsize=9)
    if xlabel:
        secax.set_xlabel(r"$K_D\;(\mathrm{M})$", fontsize=11)
    return secax


def _smooth(x, y, k=2, n=300):
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return x[mask], y[mask]
    spl = make_interp_spline(x[mask], y[mask], k=k)
    xs  = np.linspace(x[mask].min(), x[mask].max(), n)
    return xs, spl(xs)


DG0_LABEL = r"$\Delta G^0\;(k_BT)$"


# ════════════════════════════════════════════════════════════════════
# Sweep helpers (with cache)
# ════════════════════════════════════════════════════════════════════

def _data_template(NL_val, sig):
    return {
        'kEff': k_ee, 'kEffRep': k_ee, 'x0': 0,
        'NL': NL_val, 'Nrep': 0, 'kbT': kbT, 'DG0': -10.0,
        'nIntSamples': 150, 'verbose': False,
        'NP_type': 'star_polymer',
        'Nkuhn': Nkuhn, 'amono': amono, 'akuhn': akuhn,
        'sigma': sig, 'cV0': 1e-9, 'cNP0': 1e-4,
        'rV': 100, 'rNP': 10, 'epsilon_self': 1e-7,
    }


def _sweep_Fig5(cache_path):
    """DG0 sweep for Fig5 / Fig5_C: NL=5, 4 Nrep_ratios."""
    if os.path.exists(cache_path):
        print(f"  Loading {cache_path}")
        c = np.load(cache_path)
        return {r: {k: c[f'{k}_{r}'] for k in ('Fr','Fz','Ftot','theta')}
                for r in Nrep_ratios}

    results = {}
    NL = 5
    for ratio in Nrep_ratios:
        Nrep = ratio * NL
        print(f"\n  ratio={ratio}  Nrep={Nrep}  NL={NL}")
        Fr_l, Fz_l, Ft_l, th_l = [], [], [], []
        for DG0 in DG0_values:
            data = _data_template(NL, sigma_star / R_ee**2)
            data['DG0'] = DG0; data['Nrep'] = Nrep
            rLimit(data)
            try:
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    _, Fr = averageForce(direction='r', data=data)
                    _, Fz = averageForce(direction='z', data=data)
                Fr_s = Fr*R_ee/kbT; Fz_s = Fz*R_ee/kbT
                Ft_s = np.sqrt(Fr_s**2 + Fz_s**2)
                th_s = np.degrees(np.arctan2(abs(Fz_s), abs(Fr_s)))
                print(f"    DG0={DG0:5.1f}  Fr*={Fr_s:+.3f}  Fz*={Fz_s:+.3f}")
            except Exception as e:
                print(f"    DG0={DG0:5.1f}  ERROR: {e}")
                Fr_s=Fz_s=Ft_s=th_s=np.nan
            Fr_l.append(Fr_s); Fz_l.append(Fz_s)
            Ft_l.append(Ft_s); th_l.append(th_s)
        results[ratio] = {'Fr': np.abs(np.array(Fr_l)),
                          'Fz': np.abs(np.array(Fz_l)),
                          'Ftot': np.array(Ft_l),
                          'theta': np.array(th_l)}

    sd = {}
    for r in Nrep_ratios:
        for k in ('Fr','Fz','Ftot','theta'):
            sd[f'{k}_{r}'] = results[r][k]
    np.savez(cache_path, **sd)
    print(f"  Cached → {cache_path}")
    return results


def _sweep_Fig2(dg0_arr, cache_path):
    """DG0 sweep for Fig2/Fig2_SI: (sigma*, NL) combinations."""
    NL_vals = [2, 4, 6, 8]
    ss_vals = [0.06, 0.60, 6.00]
    if os.path.exists(cache_path):
        print(f"  Loading {cache_path}")
        c = np.load(cache_path)
        return {(s, NL): {'Fr': c[f'Fr_{s}_{NL}'], 'Fz': c[f'Fz_{s}_{NL}']}
                for s in ss_vals for NL in NL_vals}

    results = {}
    for sstar in ss_vals:
        sig = sstar / R_ee**2
        for NL in NL_vals:
            print(f"\n  sigma*={sstar}  NL={NL}")
            Fr_l, Fz_l = [], []
            for DG0 in dg0_arr:
                data = _data_template(NL, sig)
                data['DG0'] = DG0
                rLimit(data)
                try:
                    with warnings.catch_warnings(record=True):
                        warnings.simplefilter("always")
                        _, Fr = averageForce(direction='r', data=data)
                        _, Fz = averageForce(direction='z', data=data)
                    Fr_l.append(Fr*R_ee/kbT); Fz_l.append(Fz*R_ee/kbT)
                    print(f"    DG0={DG0:5.1f}  Fr*={Fr_l[-1]:+.3f}  Fz*={Fz_l[-1]:+.3f}")
                except Exception as e:
                    print(f"    DG0={DG0:5.1f}  ERROR: {e}")
                    Fr_l.append(np.nan); Fz_l.append(np.nan)
            results[(sstar, NL)] = {'Fr': np.abs(np.array(Fr_l)),
                                    'Fz': np.abs(np.array(Fz_l))}

    sd = {}
    for s in ss_vals:
        for NL in NL_vals:
            sd[f'Fr_{s}_{NL}'] = results[(s, NL)]['Fr']
            sd[f'Fz_{s}_{NL}'] = results[(s, NL)]['Fz']
    np.savez(cache_path, **sd)
    print(f"  Cached → {cache_path}")
    return results


# ════════════════════════════════════════════════════════════════════
# NEW_Fig5  (3 panels, DG0 bottom + K_D top)
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*70 + "\nNEW_Fig5\n" + "="*70)
res5 = _sweep_Fig5(os.path.join(outdir, 'Fig5_cache.npz'))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
keys5   = ['Fr', 'Fz', 'Ftot']
labels5 = [r"$|F_r^*|$", r"$|F_z^*|$", r"$|\mathbf{F}^*|$"]

all_y5 = np.concatenate([res5[r][k] for r in Nrep_ratios for k in keys5])
all_y5 = all_y5[np.isfinite(all_y5)]
ylim5  = (all_y5.min()-0.05*(all_y5.max()-all_y5.min()),
          all_y5.max()+0.05*(all_y5.max()-all_y5.min()))

for pi, (ax, key, plabel) in enumerate(zip(axes, keys5, labels5)):
    for idx, ratio in enumerate(Nrep_ratios):
        xs, ys = _smooth(DG0_values, res5[ratio][key])
        label  = rf"$N_{{\rm steric}}/N_L = {ratio}$" if pi == 0 else None
        ax.plot(xs, ys, '-', color=colors[idx], linewidth=1.5, label=label)
    ax.set_xlabel(DG0_LABEL, fontsize=13)
    ax.set_ylabel(plabel, fontsize=13)
    ax.set_ylim(ylim5)
    add_kd_axis(ax, xlabel=(pi == 1))
    if pi == 0:
        ax.legend(fontsize=9, framealpha=0.9)

fig.tight_layout()
for ext in ('pdf','png'):
    p = os.path.join(outdir, f"NEW_Fig5.{ext}")
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"Saved {p}")
plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# NEW_Fig5_C_2nd_version  (4 panels incl. theta)
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*70 + "\nNEW_Fig5_C_2nd_version\n" + "="*70)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
keysC   = ['Fr', 'Fz', 'Ftot', 'theta']
labelsC = [r"$|F_r^*|$", r"$|F_z^*|$", r"$|\mathbf{F}^*|$",
           r"$\theta^* = \arctan(|F_z^*|/|F_r^*|)$ [deg]"]

all_fC = np.concatenate([res5[r][k] for r in Nrep_ratios for k in ('Fr','Fz','Ftot')])
all_fC = all_fC[np.isfinite(all_fC)]
ylimC  = (all_fC.min()-0.05*(all_fC.max()-all_fC.min()),
          all_fC.max()+0.05*(all_fC.max()-all_fC.min()))

for pi, (ax, key, plabel) in enumerate(zip(axes, keysC, labelsC)):
    for idx, ratio in enumerate(Nrep_ratios):
        xs, ys = _smooth(DG0_values, res5[ratio][key])
        label  = rf"$N_{{\rm steric}}/N_L = {ratio}$" if pi == 0 else None
        ax.plot(xs, ys, '-', color=colors[idx], linewidth=1.5, label=label)
    ax.set_xlabel(DG0_LABEL, fontsize=13)
    ax.set_ylabel(plabel, fontsize=12)
    if pi < 3:
        ax.set_ylim(ylimC)
    add_kd_axis(ax, xlabel=(pi == 1))
    if pi == 0:
        ax.legend(fontsize=9, framealpha=0.9)

fig.tight_layout()
for ext in ('pdf','png'):
    p = os.path.join(outdir, f"NEW_Fig5_C_2nd_version.{ext}")
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"Saved {p}")
plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# NEW_Fig5_2D  (4×3 heatmaps, DG0 bottom + K_D top)
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*70 + "\nNEW_Fig5_2D\n" + "="*70)

cache_2d = os.path.join(outdir, 'Fig5_2D_cache.npz')
if not os.path.exists(cache_2d):
    raise FileNotFoundError(f"Run Run_Fig5_2D.py first to generate {cache_2d}")

c2d = np.load(cache_2d)
res2d = {r: {'Fr': c2d[f'Fr_{r}'], 'Fz': c2d[f'Fz_{r}'], 'Ftot': c2d[f'Ftot_{r}']}
         for r in Nrep_ratios}

keys2d   = ['Fr',   'Fz',   'Ftot']
labels2d = [r"$|F_r^*|$", r"$|F_z^*|$", r"$|\mathbf{F}^*|$"]
VMIN, VMAX     = 0.0, 8.0
CONTOUR_LEVELS = [1, 2, 3, 4, 5, 6, 7]

fig, axes = plt.subplots(4, 3, figsize=(13, 14), constrained_layout=True)

for row, ratio in enumerate(Nrep_ratios):
    for col, (key, plabel) in enumerate(zip(keys2d, labels2d)):
        ax  = axes[row, col]
        img = res2d[ratio][key]

        pc = ax.pcolormesh(DG0_values, NL_values, img,
                           cmap='viridis', vmin=VMIN, vmax=VMAX,
                           shading='nearest')
        cs = ax.contour(DG0_values, NL_values, img,
                        levels=CONTOUR_LEVELS,
                        colors='white', linewidths=0.7, alpha=0.7)
        ax.clabel(cs, fmt='%d', fontsize=7, inline=True)

        if row == 0:
            ax.set_title(plabel, fontsize=13)
            add_kd_axis(ax, xlabel=(col == 1))

        ax.set_ylabel(r"$N_L$", fontsize=11)
        if col == 0:
            ax.text(-0.42, 0.5, rf"$N_{{\rm steric}}/N_L={ratio}$",
                    transform=ax.transAxes,
                    va='center', ha='center', rotation=90, fontsize=11)

        if row == 3:
            ax.set_xlabel(DG0_LABEL, fontsize=11)

cbar = fig.colorbar(pc, ax=axes, orientation='vertical',
                    fraction=0.02, pad=0.02, label=r"$F^*$")
cbar.ax.tick_params(labelsize=9)

for ext in ('pdf','png'):
    p = os.path.join(outdir, f"NEW_Fig5_2D.{ext}")
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"Saved {p}")
plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# NEW_Fig2  (|Fr*| vs DG0, 3 sigma* panels)
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*70 + "\nNEW_Fig2\n" + "="*70)

NL_fig2 = [2, 4, 6, 8]
ss_fig2 = [0.06, 0.60, 6.00]
DG0_fig2 = np.linspace(-20, 4, 39)

res2 = _sweep_Fig2(DG0_fig2, os.path.join(outdir, 'Fig2_cache.npz'))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
all_y2 = np.concatenate([res2[(s,NL)]['Fr'] for s in ss_fig2 for NL in NL_fig2])
all_y2 = all_y2[np.isfinite(all_y2)]
ylim2  = (all_y2.min()-0.05*(all_y2.max()-all_y2.min()),
          all_y2.max()+0.05*(all_y2.max()-all_y2.min()))

for pi, (sstar, ax) in enumerate(zip(ss_fig2, axes)):
    for idx, NL in enumerate(NL_fig2):
        xs, ys = _smooth(DG0_fig2, res2[(sstar,NL)]['Fr'])
        label  = rf"$N_L = {NL}$" if pi == 0 else None
        ax.plot(xs, ys, '-', color=colors[idx], linewidth=1.5, label=label)
    ax.set_xlabel(DG0_LABEL, fontsize=13)
    if pi == 0:
        ax.set_ylabel(r"$|F_r^*|$", fontsize=13)
        ax.legend(fontsize=9, framealpha=0.9)
    ax.set_title(rf"$\sigma_R^* = {sstar}$", fontsize=12)
    ax.set_ylim(ylim2)
    add_kd_axis(ax, xlabel=(pi == 1))

fig.tight_layout()
for ext in ('pdf','png'):
    p = os.path.join(outdir, f"NEW_Fig2.{ext}")
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"Saved {p}")
plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# NEW_Fig2_SI  (|Fz*| vs DG0, denser grid)
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*70 + "\nNEW_Fig2_SI\n" + "="*70)

DG0_fig2_SI = np.linspace(-20, 4, 75)
res2_SI = _sweep_Fig2(DG0_fig2_SI,
                      os.path.join(outdir, 'Fig2_SI_cache.npz'))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
all_y2si = np.concatenate([res2_SI[(s,NL)]['Fz'] for s in ss_fig2 for NL in NL_fig2])
all_y2si = all_y2si[np.isfinite(all_y2si)]
ylim2si  = (all_y2si.min()-0.05*(all_y2si.max()-all_y2si.min()),
            all_y2si.max()+0.05*(all_y2si.max()-all_y2si.min()))

for pi, (sstar, ax) in enumerate(zip(ss_fig2, axes)):
    for idx, NL in enumerate(NL_fig2):
        xs, ys = _smooth(DG0_fig2_SI, res2_SI[(sstar,NL)]['Fz'])
        label  = rf"$N_L = {NL}$" if pi == 0 else None
        ax.plot(xs, ys, '-', color=colors[idx], linewidth=1.5, label=label)
    ax.set_xlabel(DG0_LABEL, fontsize=13)
    if pi == 0:
        ax.set_ylabel(r"$|F_z^*|$", fontsize=13)
        ax.legend(fontsize=9, framealpha=0.9)
    ax.set_title(rf"$\sigma_R^* = {sstar}$", fontsize=12)
    ax.set_ylim(ylim2si)
    add_kd_axis(ax, xlabel=(pi == 1))

fig.tight_layout()
for ext in ('pdf','png'):
    p = os.path.join(outdir, f"NEW_Fig2_SI.{ext}")
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"Saved {p}")
plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# NEW_Fig5_SI  (Fr, Fz, |F*| vs DG0, custom 1/z² repulsion)
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*70 + "\nNEW_Fig5_SI\n" + "="*70)

import Theory_v9_mp as _theory_mod

zc_SI        = 0.13 * R_ee
A0_star_vals = [0, 5, 20, 50]
DG0_SI       = np.linspace(-20, 4, 39)

cache_si = os.path.join(outdir, 'Fig5_SI_cache.npz')
if os.path.exists(cache_si):
    print(f"  Loading {cache_si}")
    csi = np.load(cache_si)
    res_si = {a: {k: csi[f'{k}_{a}'] for k in ('Fr','Fz','Ftot')}
              for a in A0_star_vals}
else:
    _orig_ARep = _theory_mod.ARep
    def _custom_ARep(z, data):
        A0 = data.get('A0_custom', 0.0)
        zc = data.get('zc_custom', 1.0)
        return 0.0 if A0 == 0.0 else A0 / (z**2 + zc**2)
    _theory_mod.ARep = _custom_ARep

    res_si = {}
    for A0_star in A0_star_vals:
        A0_phys = A0_star * kbT * R_ee**2
        print(f"\n  A0*={A0_star}")
        Fr_l, Fz_l, Ft_l = [], [], []
        for DG0 in DG0_SI:
            data = _data_template(5, sigma_star / R_ee**2)
            data['DG0'] = DG0; data['A0_custom'] = A0_phys
            data['zc_custom'] = zc_SI
            _theory_mod.rLimit(data)
            try:
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    _, Fr = _theory_mod.averageForce(direction='r', data=data)
                    _, Fz = _theory_mod.averageForce(direction='z', data=data)
                Fr_s = Fr*R_ee/kbT; Fz_s = Fz*R_ee/kbT
                Fr_l.append(Fr_s); Fz_l.append(Fz_s)
                Ft_l.append(np.sqrt(Fr_s**2 + Fz_s**2))
                print(f"    DG0={DG0:5.1f}  Fr*={Fr_s:+.3f}")
            except Exception as e:
                print(f"    DG0={DG0:5.1f}  ERROR: {e}")
                Fr_l.append(np.nan); Fz_l.append(np.nan); Ft_l.append(np.nan)
        res_si[A0_star] = {'Fr':   np.abs(np.array(Fr_l)),
                           'Fz':   np.abs(np.array(Fz_l)),
                           'Ftot': np.array(Ft_l)}
    _theory_mod.ARep = _orig_ARep

    sd = {f'{k}_{a}': res_si[a][k]
          for a in A0_star_vals for k in ('Fr','Fz','Ftot')}
    np.savez(cache_si, **sd)
    print(f"  Cached → {cache_si}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
keys_si   = ['Fr', 'Fz', 'Ftot']
labels_si = [r"$|F_r^*|$", r"$|F_z^*|$", r"$|\mathbf{F}^*|$"]

all_ysi = np.concatenate([res_si[a][k] for a in A0_star_vals for k in keys_si])
all_ysi = all_ysi[np.isfinite(all_ysi)]
ylim_si = (all_ysi.min()-0.05*(all_ysi.max()-all_ysi.min()),
           all_ysi.max()+0.05*(all_ysi.max()-all_ysi.min()))

for pi, (ax, key, plabel) in enumerate(zip(axes, keys_si, labels_si)):
    for idx, A0_star in enumerate(A0_star_vals):
        xs, ys = _smooth(DG0_SI, res_si[A0_star][key])
        label  = rf"$A_0^* = {A0_star}$" if pi == 0 else None
        ax.plot(xs, ys, '-', color=colors[idx], linewidth=1.5, label=label)
    ax.set_xlabel(DG0_LABEL, fontsize=13)
    ax.set_ylabel(plabel, fontsize=13)
    ax.set_ylim(ylim_si)
    add_kd_axis(ax, xlabel=(pi == 1))
    if pi == 0:
        ax.legend(fontsize=9, framealpha=0.9)

fig.tight_layout()
for ext in ('pdf','png'):
    p = os.path.join(outdir, f"NEW_Fig5_SI.{ext}")
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"Saved {p}")
plt.close(fig)

print("\nAll NEW figures done.")
