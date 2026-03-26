"""Function-level comparison: theory_reduced.py vs theory.py."""

import sys
import numpy as np

from conftest import TEST_CASES, make_params
from antiviral.units import to_reduced
from antiviral import theory as real_theory
from antiviral import theory_reduced as red_theory


def _setup(case):
    """Return (params_real, params_red, Ree, kbT) with rLimit called."""
    p_real = make_params(case)
    real_theory.rLimit(p_real)
    p_red, Ree, kbT = to_reduced(make_params(case))
    red_theory.rLimit(p_red)
    return p_real, p_red, Ree, kbT


def test_chi():
    failures = 0
    for case in TEST_CASES[:3]:
        p_r, p_d, Ree, _ = _setup(case)
        for r in [0.1, 1.0, 5.0]:
            if not np.isclose(real_theory.chi(r, r, p_r),
                              red_theory.chi(r/Ree, r/Ree, p_d), rtol=1e-12):
                failures += 1
    print(f'test_chi: {"PASSED" if failures == 0 else f"{failures} FAILURES"}')
    return failures == 0


def test_pureForce():
    failures = 0
    for case in TEST_CASES[:3]:
        p_r, p_d, Ree, kbT = _setup(case)
        for r in [0.1, 1.0, 5.0]:
            f_real = real_theory.pureForce(r, r, p_r)
            f_back = red_theory.pureForce(r/Ree, r/Ree, p_d) * kbT / Ree
            if not np.isclose(f_real, f_back, rtol=1e-12):
                failures += 1
    print(f'test_pureForce: {"PASSED" if failures == 0 else f"{failures} FAILURES"}')
    return failures == 0


def test_ABonds():
    failures = 0
    for case in TEST_CASES[:3]:
        p_r, p_d, Ree, kbT = _setup(case)
        for z in [0.5, 2.0]:
            a_real = real_theory.ABonds(z, p_r)
            a_back = red_theory.ABonds(z/Ree, p_d) * kbT
            if np.isinf(a_real) and np.isinf(a_back):
                continue
            if not np.isclose(a_real, a_back, rtol=1e-10):
                failures += 1
    print(f'test_ABonds: {"PASSED" if failures == 0 else f"{failures} FAILURES"}')
    return failures == 0


def test_ARep():
    failures = 0
    for case in TEST_CASES[:3]:
        p_r, p_d, Ree, kbT = _setup(case)
        for z in [0.5, 2.0]:
            a_real = real_theory.ARep(z, p_r)
            a_back = red_theory.ARep(z/Ree, p_d) * kbT
            if not np.isclose(a_real, a_back, rtol=1e-10):
                failures += 1
    print(f'test_ARep: {"PASSED" if failures == 0 else f"{failures} FAILURES"}')
    return failures == 0


def test_Omega():
    failures = 0
    for case in TEST_CASES[:4]:
        p_r, p_d, Ree, _ = _setup(case)
        omega_real = real_theory.Omega(p_r)
        omega_back = red_theory.Omega(p_d) * Ree
        if not np.isclose(omega_real, omega_back, rtol=1e-8):
            failures += 1
    print(f'test_Omega: {"PASSED" if failures == 0 else f"{failures} FAILURES"}')
    return failures == 0


if __name__ == '__main__':
    ok = all([test_chi(), test_pureForce(), test_ABonds(), test_ARep(), test_Omega()])
    sys.exit(0 if ok else 1)
