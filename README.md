# Antiviral NP-Virus Binding Theory

Statistical-mechanical theory for multivalent nanoparticle-virus binding via receptor-ligand interactions with steric repulsion from grafted polymers. Companion code for the JCP paper.

## Physical model

The code models the equilibrium binding between polymer-grafted nanoparticles (NPs) and a virus surface covered with receptors. The geometry is cylindrical: the NP sits at some distance *z* above a locally flat patch of the virus surface, with radial coordinate *r* parameterising receptor positions within the contact area.

Each NP carries two populations of grafted Gaussian chains:
- **Ligands** (`NL`): can form bonds with surface receptors, modelled as Gaussian springs with effective stiffness `kEff` and intrinsic binding free energy `DG0`.
- **Repulsive polymers** (`Nrep`): non-binding chains that generate steric repulsion upon confinement, with stiffness `kEffRep`.

The theory self-consistently solves for:
1. **Ligand occupancy** *p_L* (Eq. 4) -- fraction of free ligands, solved via bisection on `pL + integral(pL) - 1 = 0`.
2. **Receptor occupancy** *p_R* (Eq. 5) -- probability that a receptor at (*r*, *z*) is unbound.
3. **Single-bond force** (Eq. 6) -- force from all bonds at a given (*r*, *z*), projected onto radial or axial components.
4. **Bound-state average force** (Eq. 8) -- force at radial position *r*, averaged over *z* weighted by the conditional bound probability *P(z | bound)* (Eq. 9).
5. **Total average force** (Eq. 7) -- force integrated over all receptors and weighted by the fraction of occupied adsorption sites.

Free energy at distance *z* is decomposed as:
- **A_bonds**: binding contribution from ligand-receptor bonds.
- **A_rep**: steric repulsion from confining both ligand and repulsive chains against the surface, using the partition-function ratio of a Gaussian chain with a hard wall at *-z* (error-function form).
- **A_tot = A_bonds + A_rep**, which enters the bound partition function Omega (Eq. 10).

The fraction of occupied adsorption sites on the virus is determined from a chemical-equilibrium calculation between NP and site concentrations.

## Repository structure

| File | Description |
|---|---|
| `parameters.py` | `Parameters` dataclass holding all physical inputs, computed grid quantities, and internal caches |
| `theory.py` | Core physics functions implementing Eqs. 4-10 of the paper |
| `run_calculations.py` | Parameter-sweep driver: loops over `sigma`, `kEff`, `kEffRep`, `Nrep`, `DG0`; writes result files and PDF plots |
| `test_regression.py` | Regression tests comparing refactored output against golden reference from the original `Theory_v8.py` |

## Parameters

All parameters are fields of the `Parameters` dataclass. Energies are in units of *k_BT*, lengths in *nm*.

### User-specified (constructor arguments)

| Parameter | Type | Description |
|---|---|---|
| `kEff` | float | Effective spring constant of ligand chains (k_BT/nm^2) |
| `kEffRep` | float | Effective spring constant of repulsive (non-binding) chains (k_BT/nm^2) |
| `x0` | float | Equilibrium bond length; must be 0 (Gaussian chain assumption) |
| `sigma` | float | Surface density of receptors on the virus (nm^-2) |
| `NL` | int | Number of ligands per NP |
| `Nrep` | int | Number of repulsive (non-binding) polymers per NP |
| `kbT` | float | Thermal energy (set to 1.0 for reduced units) |
| `DG0` | float | Intrinsic ligand-receptor binding free energy (k_BT) |
| `L0` | float | Reference length scale |
| `maxDG` | float | Maximum free energy cutoff |
| `epsilon_self` | float | Convergence parameter (legacy, unused in bisection path) |
| `nIntSamples` | int | Number of grid points for numerical integration |
| `verbose` | bool | Enable diagnostic output |
| `cV0` | float | Molar concentration of viruses |
| `cNP0` | float | Molar concentration of nanoparticles |
| `rV` | float | Virus radius (nm) |
| `NP_type` | str | NP geometry: `'full'`, `'star_polymer'`, or `'fixed'` |
| `rNP` | float | Nanoparticle radius (nm); overwritten by `rLimit()` for some NP types |

### Computed by `rLimit()`

| Field | Description |
|---|---|
| `rMax` | Upper integration limit in *r* and *z* |
| `dr` | Grid spacing |
| `rSamples` / `zSamples` | 1-D sampling arrays (identical objects) |
| `areaAds` | Area of a single adsorption site on the virus surface |

### NP types

- **`full`**: solid NP with radius `rNP`; effective size augmented by ligand fluctuation amplitude sqrt(3 k_BT / kEff).
- **`star_polymer`**: no hard core; effective size set by the softer of the two chain populations.
- **`fixed`**: integration domain set to 3x the gyration radius; `rNP` is overwritten to `rMax`.

## Usage

### Environment setup

```bash
mamba activate antiviral
```

### Run a parameter sweep

```bash
python run_calculations.py
```

This sweeps over receptor density (`sigma`), ligand stiffness (`kEff`), repulsive chain stiffness (`kEffRep`), number of repulsive polymers (`Nrep`), and binding free energy (`DG0`). For each combination it writes:
- A text file `RESULTS_sigma_<s>_kEff_<k>_kEffRep_<kr>_NRep_<n>` with columns: `DG0`, `Fr`, `Fz`, `Fr(pN)`, `Fz(pN)`, `Fr0`, `Fz0`.
- A corresponding PDF plot.

`Fr` and `Fz` are radial and axial force components weighted by the fraction of bound sites. `Fr0` and `Fz0` are the same forces normalised by total receptors only (without the bound-fraction prefactor). The conversion factor from k_BT/nm to pN at T = 300 K is 2.6.

### Regression tests

```bash
# Generate golden reference (requires Theory_v8.py from main branch):
python test_regression.py generate

# Test refactored code against golden reference:
python test_regression.py test
```

Tests use exact floating-point equality (not tolerances) against 8 parameter combinations covering the full sweep ranges.

## Dependencies

- Python >= 3.10
- NumPy
- SciPy
- Matplotlib

## License

MIT -- see [LICENSE](LICENSE).
