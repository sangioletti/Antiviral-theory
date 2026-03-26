"""Tests for the unit conversion layer (units.py)."""

import sys
import numpy as np

from conftest import TEST_CASES, make_params
from antiviral.units import to_reduced, forces_to_real, compute_Ree


def test_keff_always_3():
    """kEff must always reduce to 3.0 regardless of input kEff."""
    failures = 0
    for case in TEST_CASES:
        p_red, _, _ = to_reduced(make_params(case))
        if p_red.kEff != 3.0:
            print(f'  FAIL kEff={case["kEff"]}: reduced kEff={p_red.kEff}')
            failures += 1
    print(f'test_keff_always_3: {"PASSED" if failures == 0 else f"{failures} FAILURES"}')
    return failures == 0


def test_kbt_always_1():
    """kbT must always reduce to 1.0."""
    failures = 0
    for case in TEST_CASES:
        p_red, _, _ = to_reduced(make_params(case))
        if p_red.kbT != 1.0:
            failures += 1
    print(f'test_kbt_always_1: {"PASSED" if failures == 0 else f"{failures} FAILURES"}')
    return failures == 0


def test_param_round_trip():
    """Converting to reduced and back should recover original values."""
    failures = 0
    for case in TEST_CASES[:3]:
        p = make_params(case)
        p_red, Ree, kbT = to_reduced(p)
        # Check a few key params
        sigma_back = p_red.sigma / Ree**2
        rV_back = p_red.rV * Ree
        cV0_back = p_red.cV0 / Ree**3
        kEffRep_back = p_red.kEffRep * kbT / Ree**2
        for name, orig, back in [
            ('sigma', p.sigma, sigma_back),
            ('rV', p.rV, rV_back),
            ('cV0', p.cV0, cV0_back),
            ('kEffRep', p.kEffRep, kEffRep_back),
        ]:
            if not np.isclose(orig, back, rtol=1e-14):
                print(f'  FAIL {name}: orig={orig}, back={back}')
                failures += 1
    print(f'test_param_round_trip: {"PASSED" if failures == 0 else f"{failures} FAILURES"}')
    return failures == 0


def test_forces_to_real_scale():
    """forces_to_real should scale by kbT/Ree."""
    Ree, kbT = 7.74597, 1.0
    fr_red = -0.2
    fr_r, _, _, _, fr_pN, _ = forces_to_real(fr_red, 0, 0, 0, Ree, kbT)
    expected = fr_red * kbT / Ree
    if not np.isclose(fr_r, expected, rtol=1e-14):
        print(f'  FAIL: fr_r={fr_r}, expected={expected}')
        print('test_forces_to_real_scale: FAILED')
        return False
    print('test_forces_to_real_scale: PASSED')
    return True


if __name__ == '__main__':
    ok = all([test_keff_always_3(), test_kbt_always_1(),
              test_param_round_trip(), test_forces_to_real_scale()])
    sys.exit(0 if ok else 1)
