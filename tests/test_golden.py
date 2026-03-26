"""Golden reference tests: compare theory.py and theory_reduced.py against Theory_v8.py."""

import copy
import sys
import numpy as np

from conftest import BASE_DATA, TEST_CASES, V8_DIR, make_params, make_data

# Add V8 directory to path for importing Theory_v8
sys.path.insert(0, V8_DIR)


def run_v8(params_dict):
    """Run original Theory_v8.py, return (fr, fz, fr0, fz0)."""
    from Theory_v8 import rLimit, averageForce
    data = copy.deepcopy(params_dict)
    rLimit(data)
    fr, fr0 = averageForce('r', data)
    data.pop('boundPartition')
    data.pop('PBoundZ')
    fz, fz0 = averageForce('z', data)
    return fr, fz, fr0, fz0


def run_real(params_dict):
    """Run theory.py (real units), return (fr, fz, fr0, fz0)."""
    from antiviral.theory import rLimit, averageForce_both
    params = make_params(params_dict)
    rLimit(params)
    return averageForce_both(params)


def run_reduced(params_dict):
    """Run theory_reduced.py via to_reduced, return forces converted back to real."""
    from antiviral.units import to_reduced, forces_to_real
    from antiviral.theory_reduced import rLimit, averageForce_both
    params = make_params(params_dict)
    p_red, Ree, kbT = to_reduced(params)
    rLimit(p_red)
    fr, fz, fr0, fz0 = averageForce_both(p_red)
    fr_r, fz_r, fr0_r, fz0_r, _, _ = forces_to_real(fr, fz, fr0, fz0, Ree, kbT)
    return fr_r, fz_r, fr0_r, fz0_r


# ---------------------------------------------------------------------------
def test_real_vs_v8():
    """theory.py must match Theory_v8.py exactly (same code, same math)."""
    labels = ['fr', 'fz', 'fr0', 'fz0']
    failures = 0
    for i, case in enumerate(TEST_CASES):
        data = make_data(case)
        v8 = run_v8(data)
        real = run_real(case)
        for j, (a, b) in enumerate(zip(v8, real)):
            if a != b:
                print(f'  FAIL case {i} {labels[j]}: v8={a!r}, real={b!r}')
                failures += 1
    total = len(TEST_CASES) * 4
    if failures == 0:
        print(f'test_real_vs_v8: {total}/{total} PASSED (exact)')
    else:
        print(f'test_real_vs_v8: {failures} FAILURES out of {total}')
    return failures == 0


def test_reduced_vs_v8():
    """theory_reduced.py → convert back must match Theory_v8.py within rtol=1e-8."""
    labels = ['fr', 'fz', 'fr0', 'fz0']
    failures = 0
    for i, case in enumerate(TEST_CASES):
        data = make_data(case)
        v8 = run_v8(data)
        red = run_reduced(case)
        for j, (a, b) in enumerate(zip(v8, red)):
            if a == 0.0 and b == 0.0:
                continue
            if not np.isclose(a, b, rtol=1e-8):
                print(f'  FAIL case {i} {labels[j]}: v8={a:.15e}, reduced={b:.15e}')
                failures += 1
    total = len(TEST_CASES) * 4
    if failures == 0:
        print(f'test_reduced_vs_v8: {total}/{total} PASSED (rtol=1e-8)')
    else:
        print(f'test_reduced_vs_v8: {failures} FAILURES out of {total}')
    return failures == 0


def test_favg_vs_fgivenb():
    """At weak binding (DG >= 0), <F> and <F|B> must differ."""
    weak = [c for c in TEST_CASES if c['DG0'] >= 0]
    failures = 0
    for case in weak:
        fr, fz, fr0, fz0 = run_real(case)
        if fr == fr0:
            print(f'  FAIL DG={case["DG0"]}: <Fr> == <Fr|B>')
            failures += 1
        if fz == fz0:
            print(f'  FAIL DG={case["DG0"]}: <Fz> == <Fz|B>')
            failures += 1
    total = len(weak) * 2
    if failures == 0:
        print(f'test_favg_vs_fgivenb: {total}/{total} PASSED')
    else:
        print(f'test_favg_vs_fgivenb: {failures} FAILURES out of {total}')
    return failures == 0


if __name__ == '__main__':
    ok = all([test_real_vs_v8(), test_reduced_vs_v8(), test_favg_vs_fgivenb()])
    sys.exit(0 if ok else 1)
