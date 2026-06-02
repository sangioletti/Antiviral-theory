# Multivalent Forces on Surfaces -- Theory

Theoretical framework for computing multivalent binding forces on surfaces,
based on the model in `Theory_v9_mp.py`.

## Directory structure

```
.
├── Theory_v9_mp.py          # Core theory module (force calculations)
├── Run_Fig2_all.py           # Consolidated script for Figure 2 variants
├── Run_Fig3_all.py           # Consolidated script for Figure 3 variants
├── Run_Fig4_all.py           # Consolidated script for Figure 4 variants
├── Run_Fig5_all.py           # Consolidated script for Figure 5 variants
├── Run_Fig5_2D_all.py        # Consolidated script for Figure 5 heatmaps
├── run_all_figures.sh        # Master bash script to regenerate all figures
├── final_plots/              # Output directory for generated figures
├── old_plots/                # Archive of previously generated plots
├── old_scripts/              # Archive of individual plotting scripts
├── *_cache.npz               # Cached computation results (auto-generated)
├── Manuscript/               # Paper source files
└── Final_Multivalent_.../    # Final submission bundle
```

## Plotting scripts

All scripts output PDF and PNG files into `final_plots/`. Expensive
computations are cached in `.npz` files at the project root; if a cache
exists it is loaded automatically, otherwise the sweep is computed from
scratch and cached for future runs.

### Run_Fig2_all.py
Generates figures for the radial/vertical binding force vs binding strength.

| Output file             | Description |
|-------------------------|-------------|
| `NEW_Fig2`              | \|Fr\*\| vs DG0, 3 panels (sigma\*_R = 0.06, 0.60, 6.00), NL = 2, 4, 6, 8. Secondary log K_D axis on top. Sigma labels bold inside panels. |
| `NEW_Fig2_SI`           | Same as above but vertical force \|Fz\*\|. Uses a denser 75-point DG0 grid. |
| `Fig2_diff_rigidity`    | Same as Fig2 but with stiffer polymer (Nmono = 5 instead of 20). No K_D axis. |

Caches: `Fig2_cache.npz`, `Fig2_SI_cache.npz`, `Fig2_diff_rigidity_cache.npz`

### Run_Fig3_all.py
Generates figures for force scaling with number of ligands and receptor spacing.

| Output file   | Description |
|---------------|-------------|
| `Fig3`        | \|F\*\| vs N_L (log-log), 3 panels for DG0 = -10, -14, -18. Curves for different sigma\* values. Includes N_L^2 reference line. |
| `Fig3_SI`     | \|Fz\*\| vs d_{R-R}\* (receptor spacing), NL = 20, DG0 = -6, -10, -14, -18. |

No cache (computes from scratch).

### Run_Fig4_all.py
Generates figures for force vs receptor spacing.

| Output file   | Description |
|---------------|-------------|
| `Fig4`        | \|Fr\*\| vs d_{R-R}\*, NL = 20, DG0 = -6, -10, -14. |
| `Fig4_SI`     | \|Fz\*\| vs d_{R-R}\*, same parameters. |

No cache (computes from scratch).

### Run_Fig5_all.py
Generates figures for the effect of steric (inert) polymer chains on binding forces.

| Output file               | Description |
|---------------------------|-------------|
| `NEW_Fig5`                | 3 panels (\|Fr\*\|, \|Fz\*\|, \|F\*\|) vs DG0, for N_steric/N_L = 0, 5, 10, 20. K_D axis on top. |
| `NEW_Fig5_SI`             | Same layout but with custom 1/z^2 repulsive interaction for A_0\* = 0, 5, 20, 50. K_D axis on top. |
| `Fig5_B_2nd_version`      | 4 panels (\|Fr\*\|, \|Fz\*\|, \|F\*\|, theta) vs N_L at fixed DG0 = -14. Shows force direction angle. |
| `NEW_Fig5_C_2nd_version`  | 4 panels (\|Fr\*\|, \|Fz\*\|, \|F\*\|, theta) vs DG0. K_D axis on top. |

Caches: `Fig5_cache.npz`, `Fig5_SI_cache.npz`

### Run_Fig5_2D_all.py
Generates 2D heatmap figures of force components.

| Output file     | Description |
|-----------------|-------------|
| `Fig5_2D`       | 4x3 grid of heatmaps. Rows: N_steric/N_L = 0, 5, 10, 20. Columns: \|Fr\*\|, \|Fz\*\|, \|F\*\|. Axes: DG0 vs N_L. Contour lines at integer force values. |
| `NEW_Fig5_2D`   | Same as above with K_D secondary axis on the top row. |

Cache: `Fig5_2D_cache.npz`

## Physical parameters (shared across all figures)

- Nmono = 20, amono = 0.38 nm, akuhn = 0.76 nm
- Nkuhn = Nmono * amono / akuhn = 10
- R_ee = sqrt(Nkuhn) * akuhn ~ 2.40 nm
- k_ee = 3 kbT / R_ee^2
- Dimensionless force: F* = F * R_ee / kbT
- Dimensionless density: sigma* = sigma * R_ee^2
- K_D = exp(+DG0 / kbT) [M]

## Running

```bash
# Generate all figures at once
bash run_all_figures.sh

# Or run individual scripts
python Run_Fig2_all.py
python Run_Fig3_all.py
# etc.
```

Figures 3 and 4 compute from scratch and may take significant time.
Figures 2, 5, and 5_2D use cached results and are fast if caches exist.
