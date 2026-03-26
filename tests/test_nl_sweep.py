"""Tests for the NL sweep function — verifies reduced-unit calculation pipeline."""

import sys
import numpy as np

from conftest import make_params
from antiviral.units import to_reduced, forces_to_real
from antiviral import theory as real_theory
from antiviral import theory_reduced as red_theory


# A single test case: sigma=0.1, kEff=0.05, kEffRep=0.01, Nrep=6, DG0=-8, NL=5
NL_CASE = {'sigma': 0.1, 'kEff': 0.05, 'kEffRep': 0.01, 'Nrep': 6, 'DG0': -8}
NL_VAL = 5


def _compute_reduced_point(overrides, NL):
    """Compute one (sigma, NL) point via theory_reduced with proper conversion."""
    p = make_params(overrides)
    p.NL = NL
    p_red, Ree, kbT = to_reduced(p)
    red_theory.rLimit(p_red)
    fr, fz, fr0, fz0 = red_theory.averageForce_both(p_red)
    p_red.clear_caches()
    return fr, fz, fr0, fz0, Ree, kbT


def _compute_real_point(overrides, NL):
    """Compute one point via theory.py in real units."""
    p = make_params(overrides)
    p.NL = NL
    real_theory.rLimit(p)
    fr, fz, fr0, fz0 = real_theory.averageForce_both(p)
    return fr, fz, fr0, fz0


def test_nl_real_input_reduced_output():
    """run_nl_sweep with real inputs should produce the same reduced forces
    as manually converting and calling theory_reduced."""
    from antiviral.run_calculations import run_nl_sweep
    import tempfile, os

    p = make_params(NL_CASE)
    with tempfile.TemporaryDirectory() as d:
        run_nl_sweep(p, np.array([NL_VAL]), np.array([NL_CASE['sigma']]),
                     output_dir=d, input_units="real", output_units="reduced",
                     plot_mode="favg", workers=1)
        dat = [f for f in os.listdir(d) if f.endswith('.dat')][0]
        with open(os.path.join(d, dat)) as f:
            lines = [l for l in f if not l.startswith('#')]
        parts = lines[0].split()
        fr_file, fz_file = float(parts[1]), float(parts[2])

    # Manual reference
    fr_ref, fz_ref, _, _, _, _ = _compute_reduced_point(NL_CASE, NL_VAL)

    ok = np.isclose(fr_file, fr_ref, rtol=1e-12) and np.isclose(fz_file, fz_ref, rtol=1e-12)
    print(f'test_nl_real_input_reduced_output: {"PASSED" if ok else "FAILED"}')
    if not ok:
        print(f'  file: fr={fr_file}, fz={fz_file}')
        print(f'  ref:  fr={fr_ref}, fz={fz_ref}')
    return ok


def test_nl_reduced_input_reduced_output():
    """run_nl_sweep with reduced inputs should give the same result."""
    from antiviral.run_calculations import run_nl_sweep
    import tempfile, os

    # First get the reduced params manually
    p_real = make_params(NL_CASE)
    p_red, Ree, kbT = to_reduced(p_real)
    sigma_red = NL_CASE['sigma'] * Ree**2

    with tempfile.TemporaryDirectory() as d:
        run_nl_sweep(p_red, np.array([NL_VAL]), np.array([sigma_red]),
                     output_dir=d, input_units="reduced", output_units="reduced",
                     plot_mode="favg", workers=1)
        dat = [f for f in os.listdir(d) if f.endswith('.dat')][0]
        with open(os.path.join(d, dat)) as f:
            lines = [l for l in f if not l.startswith('#')]
        parts = lines[0].split()
        fr_file, fz_file = float(parts[1]), float(parts[2])

    # Manual reference (same reduced calculation)
    fr_ref, fz_ref, _, _, _, _ = _compute_reduced_point(NL_CASE, NL_VAL)

    ok = np.isclose(fr_file, fr_ref, rtol=1e-10) and np.isclose(fz_file, fz_ref, rtol=1e-10)
    print(f'test_nl_reduced_input_reduced_output: {"PASSED" if ok else "FAILED"}')
    if not ok:
        print(f'  file: fr={fr_file}, fz={fz_file}')
        print(f'  ref:  fr={fr_ref}, fz={fz_ref}')
    return ok


def test_nl_real_input_real_output():
    """run_nl_sweep with real input + real output should match theory.py (real units)."""
    from antiviral.run_calculations import run_nl_sweep
    import tempfile, os

    p = make_params(NL_CASE)
    with tempfile.TemporaryDirectory() as d:
        run_nl_sweep(p, np.array([NL_VAL]), np.array([NL_CASE['sigma']]),
                     output_dir=d, input_units="real", output_units="real",
                     plot_mode="favg", workers=1)
        dat = [f for f in os.listdir(d) if f.endswith('.dat')][0]
        with open(os.path.join(d, dat)) as f:
            lines = [l for l in f if not l.startswith('#')]
        parts = lines[0].split()
        fr_file, fz_file = float(parts[1]), float(parts[2])

    # Direct real-unit reference
    fr_real, fz_real, _, _ = _compute_real_point(NL_CASE, NL_VAL)

    ok = np.isclose(fr_file, fr_real, rtol=1e-8) and np.isclose(fz_file, fz_real, rtol=1e-8)
    print(f'test_nl_real_input_real_output: {"PASSED" if ok else "FAILED"}')
    if not ok:
        print(f'  file: fr={fr_file}, fz={fz_file}')
        print(f'  real: fr={fr_real}, fz={fz_real}')
    return ok


if __name__ == '__main__':
    ok = all([
        test_nl_real_input_reduced_output(),
        test_nl_reduced_input_reduced_output(),
        test_nl_real_input_real_output(),
    ])
    sys.exit(0 if ok else 1)
