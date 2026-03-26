# Antiviral NP-Virus Binding Theory

Statistical-mechanical theory for multivalent nanoparticle-virus binding via receptor-ligand interactions with steric repulsion from grafted polymers.

## Physical model

The model computes equilibrium binding and force between a polymer-grafted nanoparticle (NP) and a receptor-covered virus surface.

Each NP has two polymer populations:
- Ligands (`NL`): binding chains with spring constant `kEff` and intrinsic binding free energy `DG0`.
- Repulsive chains (`Nrep`): non-binding steric chains with spring constant `kEffRep`.

Main calculated quantities:
1. Ligand occupancy `pL` (Eq. 4; solved by bisection).
2. Receptor occupancy `pR` (Eq. 5).
3. Bond-level forces (Eq. 6), projected into radial/axial components.
4. Bound-state averaged force (Eq. 8, with `P(z|bound)` from Eq. 9).
5. Total average force (Eq. 7), weighted by the bound-site fraction.

Free energy at distance `z` is decomposed into:
- `ABonds`: receptor-ligand contribution.
- `ARep`: steric confinement contribution.
- `ATot = ABonds + ARep`, which sets the bound partition function `Omega` (Eq. 10).

## Repository layout

| Path | Description |
|---|---|
| `src/antiviral/parameters.py` | `Parameters` dataclass (inputs, computed grids, caches) |
| `src/antiviral/theory.py` | Real-unit implementation |
| `src/antiviral/theory_reduced.py` | Reduced-unit implementation (`kBT=1`, length in `Ree`) |
| `src/antiviral/units.py` | Real/reduced unit conversion utilities |
| `src/antiviral/run_calculations.py` | Typer CLI (`sweep`, `nl-sweep`) and sweep drivers |
| `tests/` | Golden, unit-conversion, reduced-vs-real, NL sweep, and CLI smoke tests |

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

After installation, the CLI entrypoint is `antiviral`.

If you do not install the package, run with:

```bash
PYTHONPATH=src python3 -m antiviral.run_calculations --help
```

## CLI usage

Show help:

```bash
antiviral --help
```

### 1. DG sweep (`sweep`)

Sweeps `DG0` across combinations of `sigma`, `kEff`, `kEffRep`, and `Nrep`.

```bash
# Default parameter sweep
antiviral sweep

# Custom parameter ranges
antiviral sweep \
  --sweep-sigma "0.01,0.1" \
  --sweep-DG "-10:0:2" \
  --sweep-Nrep "0,6" \
  --sweep-kEff "0.025,0.05" \
  --sweep-kEffRep "0.001,0.01"

# Real-unit output and no plots
antiviral sweep --output-units real --no-plot --output-dir results
```

### 2. Ligand-count sweep (`nl-sweep`)

Sweeps force vs `NL` at fixed `DG0`, with one curve per `sigma`.

```bash
# Default NL sweep
antiviral nl-sweep

# Custom NL sweep in reduced units
antiviral nl-sweep \
  --input-units reduced \
  --sweep-NL "1:30:1" \
  --sweep-sigma "0.01,0.1,1.0" \
  --plot-mode all \
  --output-dir results
```

### Sweep-range syntax (`--sweep-*`)

- `logspace:start:stop:n` (for example, `logspace:-2:0:5`)
- `start:stop:step` (for example, `-16:5:2`)
- `v1,v2,v3` (for example, `0,6,12`)

Note: `start:stop:step` uses `numpy.arange`, so the stop value is not guaranteed to be included.

## Output files

### `sweep`

For each `(sigma, kEff, kEffRep, Nrep)` combination:
- Data file: `s{s}_k{k}_kr{kr}_Nsteric{n}[_reduced].dat`
- Plots (unless `--no-plot`):
  - `_Favg.pdf`
  - `_FgivenB.pdf`
  - `_combined.pdf`

Columns in `.dat`:
`DG`, `<Fr>`, `<Fz>`, `<Fr>(pN)`, `<Fz>(pN)`, `<Fr|B>`, `<Fz|B>`

When `--output-units reduced` is used, the `pN` columns are `nan` by design.

### `nl-sweep`

For each `sigma`:
- Data file: `NLsweep_DG{DG}_s{sigma}[_reduced].dat`
- Plots depend on `--plot-mode` (`favg`, `fgivenb`, `combined`, `disjointed`, `all`)

## Parameters

All CLI physical parameters map directly to `antiviral.parameters.Parameters`, including:
`kEff`, `kEffRep`, `x0`, `sigma`, `NL`, `Nrep`, `kbT`, `DG0`, `L0`, `maxDG`,
`epsilon_self`, `nIntSamples`, `verbose`, `cV0`, `cNP0`, `rV`, `NP_type`, `rNP`.

`NP_type` options:
- `full`
- `star_polymer`
- `fixed`

## Testing

Run all tests:

```bash
python3 tests/run_all.py
```

Or run individual suites:

```bash
python3 tests/test_units.py
python3 tests/test_reduced_vs_real.py
python3 tests/test_cli.py
python3 tests/test_nl_sweep.py
python3 tests/test_golden.py
```

Notes:
- Golden tests in `tests/test_golden.py` require legacy `Theory_v8.py` at `../main/Theory_v8.py` (relative to this worktree).
- Tests are script-style and intended to be run with `python3`, not necessarily via `pytest`.

## Dependencies

- Python >= 3.10
- NumPy
- SciPy
- Matplotlib
- Typer

## License

MIT; see [LICENSE](LICENSE).
