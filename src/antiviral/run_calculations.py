"""Driver script for antiviral NP-virus binding theory parameter sweeps."""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer

from .parameters import Parameters
from .theory_reduced import rLimit, averageForce_both
from .units import to_reduced, forces_to_real, compute_Ree, CONV_KBT_NM_TO_PN


# ---------------------------------------------------------------------------
# Worker functions for process pool (must be top-level for pickling)
# ---------------------------------------------------------------------------
def _compute_sweep_point(base_dict, sigma, kEff, kEffRep, Nrep, DG, input_units):
    """Compute one (sigma, kEff, kEffRep, Nrep, DG) point. Returns (DG, fr, fz, fr0, fz0, Ree, kbT)."""
    params = Parameters.from_dict(base_dict)
    params.sigma = sigma
    params.DG0 = DG
    params.Nrep = int(Nrep)
    params.kEff = kEff
    params.kEffRep = kEffRep

    if input_units == "real":
        p_red, Ree, kbT = to_reduced(params)
    else:
        p_red = params
        Ree, kbT = 1.0, 1.0

    rLimit(p_red)
    fr, fz, fr0, fz0 = averageForce_both(p_red)
    return DG, fr, fz, fr0, fz0, Ree, kbT


def _compute_nl_point(base_dict, sigma_red, NL):
    """Compute one (sigma, NL) point in reduced units. Returns (NL, fr, fz, fr0, fz0)."""
    params = Parameters.from_dict(base_dict)
    params.sigma = sigma_red
    params.NL = int(NL)
    rLimit(params)
    fr, fz, fr0, fz0 = averageForce_both(params)
    return int(NL), fr, fz, fr0, fz0


# ---------------------------------------------------------------------------
# Sweep range parser
# ---------------------------------------------------------------------------
def parse_sweep(value: str) -> np.ndarray:
    """Parse a sweep range specification string into an array of values.

    Supported formats:
        logspace:start:stop:n   -> np.logspace(start, stop, n)
        start:stop:step         -> np.arange(start, stop, step)
        v1,v2,v3                -> explicit list of floats
    """
    if value.startswith("logspace:"):
        parts = value.split(":")[1:]
        start, stop, n = float(parts[0]), float(parts[1]), int(parts[2])
        return np.logspace(start, stop, n)

    if ":" in value:
        parts = value.split(":")
        start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
        return np.arange(start, stop, step)

    return np.array([float(v) for v in value.split(",")])


# ---------------------------------------------------------------------------
# Core sweep logic (no CLI dependency)
# ---------------------------------------------------------------------------
def run_sweep(
    params: Parameters,
    sweep_sigma: np.ndarray,
    sweep_DG: np.ndarray,
    sweep_Nrep: np.ndarray,
    sweep_kEff: np.ndarray,
    sweep_kEffRep: np.ndarray,
    output_dir: str = ".",
    plot: bool = True,
    input_units: str = "real",
    output_units: str = "reduced",
    workers: int = None,
) -> None:
    """Run the nested parameter sweep and write results.

    input_units:  'real' (nm, kBT/nm^2, ...) or 'reduced' (Ree, kBT/Ree^2, ...)
    output_units: 'reduced' (kBT/Ree) or 'real' (kBT/nm, pN)
    workers:      number of parallel processes (None = all CPUs)
    """
    os.makedirs(output_dir, exist_ok=True)
    base_dict = params.to_dict()

    for sigma in sweep_sigma:
        for kEff in sweep_kEff:
            for kEffRep in sweep_kEffRep:
                for Nrep in sweep_Nrep:
                    print(f"Calculation for sigma = {sigma}, kEff = {kEff}, kEffRep = {kEffRep}, NRep = {Nrep}")

                    force = []
                    if workers == 1:
                        # Serial execution
                        for DG in sweep_DG:
                            result = _compute_sweep_point(base_dict, sigma, kEff, kEffRep, Nrep, DG, input_units)
                            DG_r, fr, fz, fr0, fz0, Ree, kbT = result
                            if output_units == "real":
                                fr, fz, fr0, fz0, fr_pN, fz_pN = forces_to_real(fr, fz, fr0, fz0, Ree, kbT)
                            else:
                                fr_pN, fz_pN = float('nan'), float('nan')
                            force.append((DG_r, fr, fz, fr_pN, fz_pN, fr0, fz0))
                    else:
                        # Parallel execution
                        with ProcessPoolExecutor(max_workers=workers) as pool:
                            futures = {
                                pool.submit(_compute_sweep_point, base_dict, sigma, kEff, kEffRep, Nrep, DG, input_units): DG
                                for DG in sweep_DG
                            }
                            for future in as_completed(futures):
                                DG_r, fr, fz, fr0, fz0, Ree, kbT = future.result()
                                if output_units == "real":
                                    fr, fz, fr0, fz0, fr_pN, fz_pN = forces_to_real(fr, fz, fr0, fz0, Ree, kbT)
                                else:
                                    fr_pN, fz_pN = float('nan'), float('nan')
                                force.append((DG_r, fr, fz, fr_pN, fz_pN, fr0, fz0))
                        force.sort(key=lambda x: x[0])

                    force = np.array(force)
                    nrep_int = int(Nrep)
                    baseName = f"s{sigma:.3g}_k{kEff}_kr{kEffRep}_Nsteric{nrep_int}"
                    if output_units == "reduced":
                        baseName += "_reduced"
                    dataPath = os.path.join(output_dir, baseName + ".dat")
                    plotBase = os.path.join(output_dir, baseName)

                    f_unit = "kbT/Ree" if output_units == "reduced" else "kbT/nm"
                    with open(dataPath, "w") as myF:
                        myF.write("# <F>: average force weighted by bound fraction; <F|B>: force conditioned on bound state\n")
                        if output_units == "reduced":
                            myF.write("# Output in reduced units (kBT/Ree)\n")
                        myF.write(f"# DG(kbT)  <Fr>({f_unit})  <Fz>({f_unit})  <Fr>(pN)  <Fz>(pN)  <Fr|B>({f_unit})  <Fz|B>({f_unit})\n")
                        for row in force:
                            dg, fr, fz, fr_pN, fz_pN, fr0, fz0 = row
                            myF.write(f"{dg} {fr} {fz} {fr_pN} {fz_pN} {fr0} {fz0} \n")

                    if plot:
                        force_label = rf"$k_BT / R_{{ee}}$" if output_units == "reduced" else r"$k_BT$ / nm"
                        title = (
                            rf"$\sigma$={sigma:.3g}, $k_{{\mathrm{{eff}}}}$={kEff}, "
                            rf"$k_{{\mathrm{{eff,rep}}}}$={kEffRep}, $N_{{\mathrm{{steric}}}}$={nrep_int}"
                        )

                        # Plot 1: <F>
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.plot(force[:, 0], force[:, 1], "o-", color="#d62728", markersize=5, linewidth=1.5, label=r"$\langle F_r \rangle$")
                        ax.plot(force[:, 0], force[:, 2], "s-", color="#1f77b4", markersize=5, linewidth=1.5, label=r"$\langle F_z \rangle$")
                        ax.set_xlabel(r"$\Delta G_0$ ($k_BT$)", fontsize=12)
                        ax.set_ylabel(rf"$\langle F \rangle$ ({force_label})", fontsize=12)
                        ax.set_title(title, fontsize=11)
                        ax.tick_params(labelsize=10)
                        ax.grid(True, alpha=0.3)
                        ax.legend(framealpha=0.9, edgecolor="gray", fontsize=10)
                        fig.tight_layout()
                        fig.savefig(plotBase + "_Favg.pdf", dpi=150)
                        plt.close(fig)

                        # Plot 2: <F|B>
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.plot(force[:, 0], force[:, 5], "o-", color="#d62728", markersize=5, linewidth=1.5, label=r"$\langle F_r | B \rangle$")
                        ax.plot(force[:, 0], force[:, 6], "s-", color="#1f77b4", markersize=5, linewidth=1.5, label=r"$\langle F_z | B \rangle$")
                        ax.set_xlabel(r"$\Delta G_0$ ($k_BT$)", fontsize=12)
                        ax.set_ylabel(rf"$\langle F | B \rangle$ ({force_label})", fontsize=12)
                        ax.set_title(title, fontsize=11)
                        ax.tick_params(labelsize=10)
                        ax.grid(True, alpha=0.3)
                        ax.legend(framealpha=0.9, edgecolor="gray", fontsize=10)
                        fig.tight_layout()
                        fig.savefig(plotBase + "_FgivenB.pdf", dpi=150)
                        plt.close(fig)

                        # Plot 3: combined
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.plot(force[:, 0], force[:, 1], "o-",  color="#d62728", markersize=5, linewidth=1.5, label=r"$\langle F_r \rangle$")
                        ax.plot(force[:, 0], force[:, 2], "s-",  color="#1f77b4", markersize=5, linewidth=1.5, label=r"$\langle F_z \rangle$")
                        ax.plot(force[:, 0], force[:, 5], "o--", color="#ff9896", markersize=5, linewidth=1.5, label=r"$\langle F_r | B \rangle$")
                        ax.plot(force[:, 0], force[:, 6], "s--", color="#aec7e8", markersize=5, linewidth=1.5, label=r"$\langle F_z | B \rangle$")
                        ax.set_xlabel(r"$\Delta G_0$ ($k_BT$)", fontsize=12)
                        ax.set_ylabel(rf"Force ({force_label})", fontsize=12)
                        ax.set_title(title, fontsize=11)
                        ax.tick_params(labelsize=10)
                        ax.grid(True, alpha=0.3)
                        ax.legend(framealpha=0.9, edgecolor="gray", fontsize=10)
                        fig.tight_layout()
                        fig.savefig(plotBase + "_combined.pdf", dpi=150)
                        plt.close(fig)


# ---------------------------------------------------------------------------
# NL sweep at fixed DG (force vs number of ligands)
# ---------------------------------------------------------------------------
def run_nl_sweep(
    params: Parameters,
    sweep_NL: np.ndarray,
    sweep_sigma: np.ndarray,
    output_dir: str = ".",
    input_units: str = "real",
    output_units: str = "reduced",
    plot_mode: str = "favg",
    workers: int = None,
) -> None:
    """Compute forces as a function of NL at fixed DG0, for multiple sigma values.

    Internally all calculations run in reduced units via theory_reduced.
    Conversion happens once at the boundary.

    input_units:  'real' or 'reduced'
    output_units: 'reduced' or 'real'
    plot_mode:    'favg', 'fgivenb', 'combined', 'disjointed', or 'all'
    workers:      number of parallel processes (None = all CPUs)
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Convert to reduced units once at the boundary ---
    if input_units == "real":
        p_red, Ree, kbT = to_reduced(params)
        sweep_sigma_red = sweep_sigma * Ree**2
    else:
        p_red = params
        Ree, kbT = 1.0, 1.0
        sweep_sigma_red = sweep_sigma

    base_red_dict = p_red.to_dict()

    f_unit = "kbT/Ree" if output_units == "reduced" else "kbT/nm"
    force_label = rf"$k_BT / R_{{ee}}$" if output_units == "reduced" else r"$k_BT$ / nm"

    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, len(sweep_sigma) - 1)) for i in range(len(sweep_sigma))]

    # --- Step 2: Loop entirely in reduced units, NL points in parallel ---
    results = {}
    nl_stars = {}
    for idx, (sigma_user, sigma_red) in enumerate(zip(sweep_sigma, sweep_sigma_red)):
        #The a priori average distance from equilibrium in the harmonic potential will depend on keff, DG0
        #and x0. Here we derive an heuristic d_max setting the harmonic energy = -DG0 (i.e. overall energy 0).
        d_heuristic = np.sqrt(-2 * params.DG0 / params.kEff) + params.x0

        # nl_star = sigma_red * np.pi
        nl_star = sigma_red * np.pi * d_heuristic**2
        nl_stars[sigma_user] = nl_star
        print(f"NL sweep for sigma = {sigma_user}, DG0 = {params.DG0}, NL* = {nl_star:.2f}")

        force = []
        if workers == 1:
            for NL in sweep_NL:
                nl_val, fr, fz, fr0, fz0 = _compute_nl_point(base_red_dict, sigma_red, NL)
                if output_units == "real":
                    fr, fz, fr0, fz0, _, _ = forces_to_real(fr, fz, fr0, fz0, Ree, kbT)
                force.append((nl_val, fr, fz, fr0, fz0))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_compute_nl_point, base_red_dict, sigma_red, NL): NL
                    for NL in sweep_NL
                }
                for future in as_completed(futures):
                    nl_val, fr, fz, fr0, fz0 = future.result()
                    if output_units == "real":
                        fr, fz, fr0, fz0, _, _ = forces_to_real(fr, fz, fr0, fz0, Ree, kbT)
                    force.append((nl_val, fr, fz, fr0, fz0))
            force.sort(key=lambda x: x[0])
        results[sigma_user] = np.array(force)

        # Write per-sigma data file
        baseName = f"NLsweep_DG{params.DG0}_s{sigma_user:.3g}"
        if output_units == "reduced":
            baseName += "_reduced"
        dataPath = os.path.join(output_dir, baseName + ".dat")
        with open(dataPath, "w") as f:
            f.write(f"# Force vs NL at DG0 = {params.DG0}, sigma = {sigma_user}\n")
            if output_units == "reduced":
                f.write(f"# Output in reduced units (kBT/Ree), NL* = {nl_star:.4g}\n")
            else:
                f.write(f"# Ree = {Ree:.6g} nm, NL* = {nl_star:.4g}\n")
            f.write(f"# NL  <Fr>({f_unit})  <Fz>({f_unit})  <Fr|B>({f_unit})  <Fz|B>({f_unit})\n")
            for row in results[sigma_user]:
                f.write(f"{int(row[0])} {row[1]} {row[2]} {row[3]} {row[4]}\n")

    # --- Plotting ---
    title = (
        rf"$\Delta G_0$={params.DG0}, $k_{{\mathrm{{eff}}}}$={params.kEff}, "
        rf"$k_{{\mathrm{{eff,rep}}}}$={params.kEffRep}, $N_{{\mathrm{{steric}}}}$={params.Nrep}"
    )
    plotBase = os.path.join(output_dir, f"NLsweep_DG{params.DG0}")
    if output_units == "reduced":
        plotBase += "_reduced"

    do_favg = plot_mode in ("favg", "disjointed", "all")
    do_fgivenb = plot_mode in ("fgivenb", "disjointed", "all")
    do_combined = plot_mode in ("combined", "all")

    def _make_plot(suffix, ylabel, series_spec):
        """series_spec: list of (col_index, label_template, marker, ls, alpha)."""
        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, sigma in enumerate(sweep_sigma):
            data = results[sigma]
            c = colors[idx]
            for col, lbl_tpl, mkr, ls, alpha in series_spec:
                ax.plot(data[:, 0], data[:, col], mkr + ls, color=c, markersize=4,
                        linewidth=1.5, alpha=alpha,
                        label=lbl_tpl.format(s=sigma))
            ax.axvline(nl_stars[sigma], color=c, linestyle=":", alpha=0.7,
                       label=rf"$N_L^*$, $\sigma$={sigma:.3g}")
        ax.set_xlim(sweep_NL[0] - 0.5, sweep_NL[-1] + 0.5)
        ax.set_xlabel(r"$N_L$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(framealpha=0.9, edgecolor="gray", fontsize=8)
        fig.tight_layout()
        fig.savefig(plotBase + suffix, dpi=150)
        plt.close(fig)

    # col indices: 0=NL, 1=fr, 2=fz, 3=fr0, 4=fz0
    if do_favg:
        _make_plot("_Favg_r.pdf", rf"$\langle F_r \rangle$ ({force_label})",
                   [(1, r"$\sigma$={s:.3g}", "o", "-", 1.0)])
        _make_plot("_Favg_z.pdf", rf"$\langle F_z \rangle$ ({force_label})",
                   [(2, r"$\sigma$={s:.3g}", "s", "-", 1.0)])

    if do_fgivenb:
        _make_plot("_FgivenB_r.pdf", rf"$\langle F_r | B \rangle$ ({force_label})",
                   [(3, r"$\sigma$={s:.3g}", "o", "-", 1.0)])
        _make_plot("_FgivenB_z.pdf", rf"$\langle F_z | B \rangle$ ({force_label})",
                   [(4, r"$\sigma$={s:.3g}", "s", "-", 1.0)])

    if do_combined:
        _make_plot("_combined_r.pdf", rf"$F_r$ ({force_label})", [
            (1, r"$\langle F_r \rangle$, $\sigma$={s:.3g}", "o", "-", 1.0),
            (3, r"$\langle F_r | B \rangle$, $\sigma$={s:.3g}", "o", "--", 0.5),
        ])
        _make_plot("_combined_z.pdf", rf"$F_z$ ({force_label})", [
            (2, r"$\langle F_z \rangle$, $\sigma$={s:.3g}", "s", "-", 1.0),
            (4, r"$\langle F_z | B \rangle$, $\sigma$={s:.3g}", "s", "--", 0.5),
        ])



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
APP_HELP = """\
Antiviral NP-virus binding theory: compute average forces between
polymer-grafted nanoparticles and virus surfaces.

All calculations run internally in reduced units (Ree, kBT).
Inputs can be provided in real (nm) or reduced units via --input-units.
Outputs default to reduced units; use --output-units real for kBT/nm and pN.

Sweep range syntax (for --sweep-* options):
  logspace:start:stop:n   Logarithmic spacing, e.g. logspace:-2:0:5
  start:stop:step         Arithmetic range, e.g. -16:5:2
  v1,v2,v3                Explicit comma-separated values, e.g. 0.025,0.05,0.1
"""
app = typer.Typer(
    help=APP_HELP,
    invoke_without_command=True,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command("sweep", no_args_is_help=True)
def main(
    # --- Base physical parameters ---
    k_eff: Annotated[float, typer.Option("--kEff", help="Ligand spring constant", rich_help_panel="Physical parameters")] = 1.0,
    k_eff_rep: Annotated[float, typer.Option("--kEffRep", help="Repulsive (steric) chain spring constant", rich_help_panel="Physical parameters")] = 3.0,
    x0: Annotated[float, typer.Option(help="Equilibrium bond length (must be 0 for Gaussian chains)", rich_help_panel="Physical parameters")] = 0.0,
    sigma: Annotated[float, typer.Option(help="Receptor surface density on the virus", rich_help_panel="Physical parameters")] = 0.01,
    nl: Annotated[int, typer.Option("--NL", help="Number of ligands (binding chains) per NP", rich_help_panel="Physical parameters")] = 5,
    nrep: Annotated[int, typer.Option("--Nrep", help="Number of repulsive (steric, non-binding) polymers per NP", rich_help_panel="Physical parameters")] = 10,
    kbt: Annotated[float, typer.Option("--kbT", help="Thermal energy (set 1.0 for reduced units)", rich_help_panel="Physical parameters")] = 1.0,
    dg0: Annotated[float, typer.Option("--DG0", help="Intrinsic ligand-receptor binding free energy (in kBT)", rich_help_panel="Physical parameters")] = -5.0,
    l0: Annotated[float, typer.Option("--L0", help="Reference length scale", rich_help_panel="Physical parameters")] = 1.0,
    max_dg: Annotated[float, typer.Option("--maxDG", help="Maximum free-energy cutoff", rich_help_panel="Physical parameters")] = 5.0,
    epsilon_self: Annotated[float, typer.Option(help="Legacy convergence parameter (unused in bisection path)", rich_help_panel="Physical parameters")] = 1e-7,
    n_int_samples: Annotated[int, typer.Option("--nIntSamples", help="Number of grid points for numerical integration", rich_help_panel="Physical parameters")] = 100,
    verbose: Annotated[bool, typer.Option("--verbose", help="Print diagnostic output during computation", rich_help_panel="Physical parameters")] = False,
    cv0: Annotated[float, typer.Option("--cV0", help="Number density of viruses", rich_help_panel="Physical parameters")] = 1e-9,
    cnp0: Annotated[float, typer.Option("--cNP0", help="Number density of nanoparticles", rich_help_panel="Physical parameters")] = 1e-6,
    rv: Annotated[float, typer.Option("--rV", help="Virus radius", rich_help_panel="Physical parameters")] = 100.0,
    np_type: Annotated[str, typer.Option("--NP-type", help="NP geometry model: full, star_polymer, or fixed", rich_help_panel="Physical parameters")] = "star_polymer",
    rnp: Annotated[float, typer.Option("--rNP", help="Nanoparticle radius (overwritten by rLimit for some NP types)", rich_help_panel="Physical parameters")] = 10.0,
    # --- Sweep ranges ---
    sweep_sigma: Annotated[str, typer.Option(help="Sweep range for receptor density sigma", rich_help_panel="Sweep ranges")] = "logspace:-2:0:5",
    sweep_dg: Annotated[str, typer.Option("--sweep-DG", help="Sweep range for binding free energy DG0", rich_help_panel="Sweep ranges")] = "-16:5:2",
    sweep_nrep: Annotated[str, typer.Option("--sweep-Nrep", help="Sweep range for number of steric polymers", rich_help_panel="Sweep ranges")] = "0,6,12",
    sweep_k_eff: Annotated[str, typer.Option("--sweep-kEff", help="Sweep range for ligand spring constant", rich_help_panel="Sweep ranges")] = "0.025,0.05,0.1",
    sweep_k_eff_rep: Annotated[str, typer.Option("--sweep-kEffRep", help="Sweep range for steric chain spring constant", rich_help_panel="Sweep ranges")] = "0.001,0.01,0.1",
    # --- Output control ---
    output_dir: Annotated[str, typer.Option(help="Directory for output data files and PDF plots", rich_help_panel="Output")] = ".",
    no_plot: Annotated[bool, typer.Option("--no-plot", help="Skip PDF plot generation (write data files only)", rich_help_panel="Output")] = False,
    input_units: Annotated[str, typer.Option("--input-units", help="Unit system for inputs: 'real' (nm, kBT/nm^2) or 'reduced' (Ree, kBT/Ree^2)", rich_help_panel="Output")] = "real",
    output_units: Annotated[str, typer.Option("--output-units", help="Unit system for outputs: 'reduced' (kBT/Ree) or 'real' (kBT/nm, pN)", rich_help_panel="Output")] = "reduced",
    workers: Annotated[int, typer.Option("--workers", help="Number of parallel worker processes; 0 = all CPUs, 1 = serial", rich_help_panel="Output")] = 0,
) -> None:
    """Sweep forces as a function of DG0 over combinations of (sigma, kEff, kEffRep, Nrep).

    For each parameter combination, computes <F_r>, <F_z> (average force weighted
    by bound fraction) and <F_r|B>, <F_z|B> (force conditioned on the bound state)
    at every DG0 value in the sweep range.

    Outputs per combination:
      - .dat file with force components in the chosen output units
      - PDF plots: _Favg, _FgivenB, _combined (unless --no-plot)

    Examples:
      # Default full sweep in reduced units:
      python run_calculations.py sweep

      # Single sigma, narrow DG range, real-unit output:
      python run_calculations.py sweep --sweep-sigma 0.1 --sweep-DG "-10:0:1" --output-units real

      # Fast test with 4 workers and no plots:
      python run_calculations.py sweep --workers 4 --no-plot --output-dir results/
    """
    params = Parameters(
        kEff=k_eff,
        kEffRep=k_eff_rep,
        x0=x0,
        sigma=sigma,
        NL=nl,
        Nrep=nrep,
        kbT=kbt,
        DG0=dg0,
        L0=l0,
        maxDG=max_dg,
        epsilon_self=epsilon_self,
        nIntSamples=n_int_samples,
        verbose=verbose,
        cV0=cv0,
        cNP0=cnp0,
        rV=rv,
        NP_type=np_type,
        rNP=rnp,
    )

    run_sweep(
        params=params,
        sweep_sigma=parse_sweep(sweep_sigma),
        sweep_DG=parse_sweep(sweep_dg),
        sweep_Nrep=parse_sweep(sweep_nrep),
        sweep_kEff=parse_sweep(sweep_k_eff),
        sweep_kEffRep=parse_sweep(sweep_k_eff_rep),
        output_dir=output_dir,
        plot=not no_plot,
        input_units=input_units,
        output_units=output_units,
        workers=workers or None,
    )


@app.command("nl-sweep", no_args_is_help=True)
def nl_sweep_cmd(
    # --- Base physical parameters ---
    k_eff: Annotated[float, typer.Option("--kEff", help="Ligand spring constant", rich_help_panel="Physical parameters")] = 1.0,
    k_eff_rep: Annotated[float, typer.Option("--kEffRep", help="Repulsive (steric) chain spring constant", rich_help_panel="Physical parameters")] = 3.0,
    x0: Annotated[float, typer.Option(help="Equilibrium bond length (must be 0 for Gaussian chains)", rich_help_panel="Physical parameters")] = 0.0,
    sigma: Annotated[float, typer.Option(help="Base receptor surface density on the virus", rich_help_panel="Physical parameters")] = 0.01,
    nrep: Annotated[int, typer.Option("--Nrep", help="Number of repulsive (steric, non-binding) polymers per NP", rich_help_panel="Physical parameters")] = 10,
    kbt: Annotated[float, typer.Option("--kbT", help="Thermal energy (set 1.0 for reduced units)", rich_help_panel="Physical parameters")] = 1.0,
    dg0: Annotated[float, typer.Option("--DG0", help="Fixed binding free energy for this sweep (in kBT)", rich_help_panel="Physical parameters")] = -8.0,
    l0: Annotated[float, typer.Option("--L0", help="Reference length scale", rich_help_panel="Physical parameters")] = 1.0,
    max_dg: Annotated[float, typer.Option("--maxDG", help="Maximum free-energy cutoff", rich_help_panel="Physical parameters")] = 5.0,
    epsilon_self: Annotated[float, typer.Option(help="Legacy convergence parameter (unused in bisection path)", rich_help_panel="Physical parameters")] = 1e-7,
    n_int_samples: Annotated[int, typer.Option("--nIntSamples", help="Number of grid points for numerical integration", rich_help_panel="Physical parameters")] = 100,
    verbose: Annotated[bool, typer.Option("--verbose", help="Print diagnostic output during computation", rich_help_panel="Physical parameters")] = False,
    cv0: Annotated[float, typer.Option("--cV0", help="Number density of viruses", rich_help_panel="Physical parameters")] = 1e-9,
    cnp0: Annotated[float, typer.Option("--cNP0", help="Number density of nanoparticles", rich_help_panel="Physical parameters")] = 1e-6,
    rv: Annotated[float, typer.Option("--rV", help="Virus radius", rich_help_panel="Physical parameters")] = 100.0,
    np_type: Annotated[str, typer.Option("--NP-type", help="NP geometry model: full, star_polymer, or fixed", rich_help_panel="Physical parameters")] = "full",
    rnp: Annotated[float, typer.Option("--rNP", help="Nanoparticle radius (overwritten by rLimit for some NP types)", rich_help_panel="Physical parameters")] = 10.0,
    # --- NL sweep specific ---
    sweep_nl: Annotated[str, typer.Option("--sweep-NL", help="Range of NL (ligand count) values to sweep", rich_help_panel="Sweep ranges")] = "1:20:1",
    sweep_sigma: Annotated[str, typer.Option("--sweep-sigma", help="Sigma values to plot (one curve per sigma)", rich_help_panel="Sweep ranges")] = "0.01,0.1,1.0",
    # --- Output control ---
    output_dir: Annotated[str, typer.Option(help="Directory for output data files and PDF plots", rich_help_panel="Output")] = ".",
    input_units: Annotated[str, typer.Option("--input-units", help="Unit system for inputs: 'real' (nm, kBT/nm^2) or 'reduced' (Ree, kBT/Ree^2)", rich_help_panel="Output")] = "reduced",
    output_units: Annotated[str, typer.Option("--output-units", help="Unit system for outputs: 'reduced' (kBT/Ree) or 'real' (kBT/nm, pN)", rich_help_panel="Output")] = "reduced",
    plot_mode: Annotated[str, typer.Option("--plot-mode", help="Which plots: favg, fgivenb, combined, disjointed (favg+fgivenb), or all", rich_help_panel="Output")] = "favg",
    workers: Annotated[int, typer.Option("--workers", help="Number of parallel worker processes; 0 = all CPUs, 1 = serial", rich_help_panel="Output")] = 0,
) -> None:
    """Sweep forces as a function of NL (number of ligands) at fixed DG0.

    Produces one curve per sigma value. Each plot includes a vertical dashed
    line at NL* = sigma * pi * Ree^2 (or sigma * pi in reduced units),
    marking the crossover where ligand count matches receptors within chain reach.

    Plot modes:
      favg        Only <F_r> and <F_z> (default)
      fgivenb     Only <F_r|B> and <F_z|B>
      combined    Both <F> and <F|B> overlaid on same plot
      disjointed  Separate PDFs for <F> and <F|B> (= favg + fgivenb)
      all         All of the above

    Outputs per sigma:
      - .dat file with NL, force components
      - PDF plots split by r and z components

    Examples:
      # Default NL sweep with 3 sigma curves:
      python run_calculations.py nl-sweep --DG0 -8

      # Reduced-unit inputs, wide NL range, all plots:
      python run_calculations.py nl-sweep --input-units reduced --sweep-NL "1:30:1" --plot-mode all

      # Real-unit output for a single sigma:
      python run_calculations.py nl-sweep --sweep-sigma 0.1 --output-units real --output-dir results/
    """
    params = Parameters(
        kEff=k_eff,
        kEffRep=k_eff_rep,
        x0=x0,
        sigma=sigma,
        NL=1,  # will be overwritten in sweep
        Nrep=nrep,
        kbT=kbt,
        DG0=dg0,
        L0=l0,
        maxDG=max_dg,
        epsilon_self=epsilon_self,
        nIntSamples=n_int_samples,
        verbose=verbose,
        cV0=cv0,
        cNP0=cnp0,
        rV=rv,
        NP_type=np_type,
        rNP=rnp,
    )

    run_nl_sweep(
        params=params,
        sweep_NL=parse_sweep(sweep_nl),
        sweep_sigma=parse_sweep(sweep_sigma),
        output_dir=output_dir,
        input_units=input_units,
        output_units=output_units,
        plot_mode=plot_mode,
        workers=workers or None,
    )


if __name__ == "__main__":
    app()
