"""Driver script for antiviral NP-virus binding theory parameter sweeps."""

import os
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer

from parameters import Parameters
from theory import rLimit, averageForce_both, CONV_KBT_TO_PN


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
    reduced_units: bool = False,
) -> None:
    """Run the nested parameter sweep and write results."""
    os.makedirs(output_dir, exist_ok=True)

    for sigma in sweep_sigma:
        for kEff in sweep_kEff:
            for kEffRep in sweep_kEffRep:
                for Nrep in sweep_Nrep:
                    force = []
                    print(f"Calculation for sigma = {sigma}, kEff = {kEff}, kEffRep = {kEffRep}, NRep = {Nrep}")

                    for DG in sweep_DG:
                        params.sigma = sigma
                        params.DG0 = DG
                        params.Nrep = int(Nrep)
                        params.kEff = kEff
                        params.kEffRep = kEffRep
                        rLimit(params)
                        print(f"DG = {DG}")

                        fr, fz, fr0, fz0 = averageForce_both(params)

                        if reduced_units:
                            # Inputs already in reduced units — no conversion
                            fr_pN = float('nan')
                            fz_pN = float('nan')
                        else:
                            fr_pN = fr * CONV_KBT_TO_PN
                            fz_pN = fz * CONV_KBT_TO_PN

                        force.append((DG, fr, fz, fr_pN, fz_pN, fr0, fz0))
                        params.clear_caches()

                    force = np.array(force)
                    nrep_int = int(Nrep)
                    baseName = f"s{sigma:.3g}_k{kEff}_kr{kEffRep}_Nsteric{nrep_int}"
                    if reduced_units:
                        baseName += "_reduced"
                    dataPath = os.path.join(output_dir, baseName + ".dat")
                    plotBase = os.path.join(output_dir, baseName)

                    if reduced_units:
                        f_unit = "kbT/Ree"
                        header_note = "# Reduced units (inputs in Ree, kBT)\n"
                    else:
                        f_unit = "kbT/nm"
                        header_note = ""

                    with open(dataPath, "w") as myF:
                        myF.write("# <F>: average force weighted by bound fraction; <F|B>: force conditioned on bound state\n")
                        if header_note:
                            myF.write(header_note)
                        myF.write(f"# DG(kbT)  <Fr>({f_unit})  <Fz>({f_unit})  <Fr>(pN)  <Fz>(pN)  <Fr|B>({f_unit})  <Fz|B>({f_unit})\n")
                        for row in force:
                            dg, fr, fz, fr_pN, fz_pN, fr0, fz0 = row
                            myF.write(f"{dg} {fr} {fz} {fr_pN} {fz_pN} {fr0} {fz0} \n")

                    if plot:
                        force_label = rf"$k_BT / R_{{ee}}$" if reduced_units else r"$k_BT$ / nm"
                        title = (
                            rf"$\sigma$={sigma:.3g}, $k_{{\mathrm{{eff}}}}$={kEff}, "
                            rf"$k_{{\mathrm{{eff,rep}}}}$={kEffRep}, $N_{{\mathrm{{steric}}}}$={nrep_int}"
                        )

                        # Plot 1: <F> (average force)
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

                        # Plot 2: <F|B> (force conditioned on bound)
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
    reduced_units: bool = True,
    plot_mode: str = "favg",
) -> None:
    """Compute forces as a function of NL at fixed DG0, for multiple sigma values.

    plot_mode: 'favg', 'fgivenb', 'combined', or 'all'.
    """
    os.makedirs(output_dir, exist_ok=True)

    kEff = params.kEff

    Ree = np.sqrt(3.0 * params.kbT / kEff)
    if reduced_units:
        # Inputs already in reduced units (Ree=1, kBT=1) — no conversion
        f_unit = "kbT/Ree"
        force_label = rf"$k_BT / R_{{ee}}$"
    else:
        f_unit = "kbT/nm"
        force_label = r"$k_BT$ / nm"

    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, len(sweep_sigma) - 1)) for i in range(len(sweep_sigma))]

    # Collect results: dict sigma -> array of (NL, fr, fz, fr0, fz0)
    results = {}
    nl_stars = {}
    for idx, sigma in enumerate(sweep_sigma):
        force = []
        if reduced_units:
            nl_star = sigma * np.pi
        else:
            nl_star = sigma * np.pi * Ree**2
        nl_stars[sigma] = nl_star
        print(f"NL sweep for sigma = {sigma}, DG0 = {params.DG0}, NL* = {nl_star:.2f}")
        for NL in sweep_NL:
            if reduced_units:
                params.sigma = sigma * Ree**2
            else:   
                params.sigma = sigma
            params.NL = int(NL)
            rLimit(params)
            fr, fz, fr0, fz0 = averageForce_both(params)
            force.append((int(NL), fr, fz, fr0, fz0))
            params.clear_caches()
        results[sigma] = np.array(force)

        # Write per-sigma data file
        baseName = f"NLsweep_DG{params.DG0}_s{sigma:.3g}"
        if reduced_units:
            baseName += "_reduced"
        dataPath = os.path.join(output_dir, baseName + ".dat")
        with open(dataPath, "w") as f:
            f.write(f"# Force vs NL at DG0 = {params.DG0}, sigma = {sigma}\n")
            if reduced_units:
                f.write(f"# Reduced units (inputs in Ree, kBT), NL* = sigma * pi = {nl_star:.4g}\n")
            else:
                f.write(f"# Ree = sqrt(3 kbT / kEff) = {Ree:.6g} nm, NL* = sigma * pi * Ree^2 = {nl_star:.4g}\n")
            f.write(f"# NL  <Fr>({f_unit})  <Fz>({f_unit})  <Fr|B>({f_unit})  <Fz|B>({f_unit})\n")
            for row in results[sigma]:
                f.write(f"{int(row[0])} {row[1]} {row[2]} {row[3]} {row[4]}\n")

    # --- Plotting helper ---
    title = (
        rf"$\Delta G_0$={params.DG0}, $k_{{\mathrm{{eff}}}}$={kEff}, "
        rf"$k_{{\mathrm{{eff,rep}}}}$={params.kEffRep}, $N_{{\mathrm{{steric}}}}$={params.Nrep}"
    )
    plotBase = os.path.join(output_dir, f"NLsweep_DG{params.DG0}")
    if reduced_units:
        plotBase += "_reduced"

    do_favg = plot_mode in ("favg", "all")
    do_fgivenb = plot_mode in ("fgivenb", "all")
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
app = typer.Typer(help="Antiviral NP-virus binding theory parameter sweeps.")


@app.command("sweep")
def main(
    # --- Base parameters (defaults match the original hardcoded base_data) ---
    k_eff: Annotated[float, typer.Option("--kEff", help="Ligand spring constant (kBT/nm^2)")] = 1.0,
    k_eff_rep: Annotated[float, typer.Option("--kEffRep", help="Repulsive chain spring constant (kBT/nm^2)")] = 3.0,
    x0: Annotated[float, typer.Option(help="Equilibrium bond length (must be 0)")] = 0.0,
    sigma: Annotated[float, typer.Option(help="Receptor surface density (nm^-2)")] = 0.01,
    nl: Annotated[int, typer.Option("--NL", help="Number of ligands per NP")] = 5,
    nrep: Annotated[int, typer.Option("--Nrep", help="Number of repulsive polymers per NP")] = 10,
    kbt: Annotated[float, typer.Option("--kbT", help="Thermal energy")] = 1.0,
    dg0: Annotated[float, typer.Option("--DG0", help="Intrinsic binding free energy (kBT)")] = -5.0,
    l0: Annotated[float, typer.Option("--L0", help="Reference length scale")] = 1.0,
    max_dg: Annotated[float, typer.Option("--maxDG", help="Maximum free energy cutoff")] = 5.0,
    epsilon_self: Annotated[float, typer.Option(help="Convergence parameter")] = 1e-7,
    n_int_samples: Annotated[int, typer.Option("--nIntSamples", help="Number of integration grid points")] = 100,
    verbose: Annotated[bool, typer.Option("--verbose", help="Enable diagnostic output")] = False,
    cv0: Annotated[float, typer.Option("--cV0", help="Molar concentration of viruses")] = 1e-9,
    cnp0: Annotated[float, typer.Option("--cNP0", help="Molar concentration of nanoparticles")] = 1e-6,
    rv: Annotated[float, typer.Option("--rV", help="Virus radius (nm)")] = 100.0,
    np_type: Annotated[str, typer.Option("--NP-type", help="NP geometry: full, star_polymer, fixed")] = "star_polymer",
    rnp: Annotated[float, typer.Option("--rNP", help="Nanoparticle radius (nm)")] = 10.0,
    # --- Sweep ranges ---
    sweep_sigma: Annotated[str, typer.Option(help="Sweep range for sigma")] = "logspace:-2:0:5",
    sweep_dg: Annotated[str, typer.Option("--sweep-DG", help="Sweep range for DG0")] = "-16:5:2",
    sweep_nrep: Annotated[str, typer.Option("--sweep-Nrep", help="Sweep range for Nrep")] = "0,6,12",
    sweep_k_eff: Annotated[str, typer.Option("--sweep-kEff", help="Sweep range for kEff")] = "0.025,0.05,0.1",
    sweep_k_eff_rep: Annotated[str, typer.Option("--sweep-kEffRep", help="Sweep range for kEffRep")] = "0.001,0.01,0.1",
    # --- Output control ---
    output_dir: Annotated[str, typer.Option(help="Output directory for results and plots")] = ".",
    no_plot: Annotated[bool, typer.Option("--no-plot", help="Skip PDF plot generation")] = False,
    reduced_units: Annotated[bool, typer.Option("--reduced-units", help="Output forces in reduced units (kBT/Ree where Ree=sqrt(3 kBT/kEff))")] = False,
) -> None:
    """Run a parameter sweep over the NP-virus binding theory."""
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
        reduced_units=reduced_units,
    )


@app.command("nl-sweep")
def nl_sweep_cmd(
    # --- Base parameters ---
    k_eff: Annotated[float, typer.Option("--kEff", help="Ligand spring constant (kBT/nm^2)")] = 1.0,
    k_eff_rep: Annotated[float, typer.Option("--kEffRep", help="Repulsive chain spring constant (kBT/nm^2)")] = 3.0,
    x0: Annotated[float, typer.Option(help="Equilibrium bond length (must be 0)")] = 0.0,
    sigma: Annotated[float, typer.Option(help="Base receptor surface density (nm^-2)")] = 0.01,
    nrep: Annotated[int, typer.Option("--Nrep", help="Number of repulsive polymers per NP")] = 10,
    kbt: Annotated[float, typer.Option("--kbT", help="Thermal energy")] = 1.0,
    dg0: Annotated[float, typer.Option("--DG0", help="Fixed binding free energy (kBT)")] = -8.0,
    l0: Annotated[float, typer.Option("--L0", help="Reference length scale")] = 1.0,
    max_dg: Annotated[float, typer.Option("--maxDG", help="Maximum free energy cutoff")] = 5.0,
    epsilon_self: Annotated[float, typer.Option(help="Convergence parameter")] = 1e-7,
    n_int_samples: Annotated[int, typer.Option("--nIntSamples", help="Number of integration grid points")] = 100,
    verbose: Annotated[bool, typer.Option("--verbose", help="Enable diagnostic output")] = False,
    cv0: Annotated[float, typer.Option("--cV0", help="Molar concentration of viruses")] = 1e-9,
    cnp0: Annotated[float, typer.Option("--cNP0", help="Molar concentration of nanoparticles")] = 1e-6,
    rv: Annotated[float, typer.Option("--rV", help="Virus radius (nm)")] = 100.0,
    np_type: Annotated[str, typer.Option("--NP-type", help="NP geometry: full, star_polymer, fixed")] = "star_polymer",
    rnp: Annotated[float, typer.Option("--rNP", help="Nanoparticle radius (nm)")] = 10.0,
    # --- NL sweep specific ---
    sweep_nl: Annotated[str, typer.Option("--sweep-NL", help="Range of NL values to sweep")] = "1:20:1",
    sweep_sigma: Annotated[str, typer.Option("--sweep-sigma", help="Sigma values (one curve per sigma)")] = "0.01,0.1,1.0",
    # --- Output control ---
    output_dir: Annotated[str, typer.Option(help="Output directory for results and plots")] = ".",
    reduced_units: Annotated[bool, typer.Option("--reduced-units/--no-reduced-units", help="Inputs already in reduced units (Ree=1, kBT=1)")] = True,
    plot_mode: Annotated[str, typer.Option("--plot-mode", help="Which plots to generate: favg, fgivenb, combined, all")] = "favg",
) -> None:
    """Sweep NL at fixed DG0 for multiple sigma values."""
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
        reduced_units=reduced_units,
        plot_mode=plot_mode,
    )


if __name__ == "__main__":
    app()
