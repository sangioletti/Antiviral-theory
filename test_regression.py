"""Regression test for antiviral theory refactoring.

Generates golden reference from Theory_v8.py, then compares
refactored theory.py output against it. All comparisons use
exact equality (==), not np.allclose.
"""
import copy
import numpy as np

# ---------------------------------------------------------------------------
# Test parameter sets: cover different sigma, kEff, kEffRep, Nrep, DG
# ---------------------------------------------------------------------------
BASE_DATA = {
    'kEff': 1.0, 'kEffRep': 3.0, 'x0': 0, 'sigma': 0.01,
    'NL': 5, 'Nrep': 10, 'kbT': 1.0, 'DG0': -5.0, 'L0': 1.0,
    'maxDG': 5, 'epsilon_self': 1e-7, 'nIntSamples': 100,
    'countBeta': 1000, 'rhoNP': 0.001, 'verbose': False,
    'cV0': 1e-9, 'cNP0': 1e-6, 'rV': 100,
    'NP_type': 'star_polymer', 'rNP': 10,
}

TEST_CASES = [
    {'sigma': 0.01,  'kEff': 0.025, 'kEffRep': 0.001, 'Nrep': 0,  'DG0': -14},
    {'sigma': 0.01,  'kEff': 0.05,  'kEffRep': 0.01,  'Nrep': 6,  'DG0': -10},
    {'sigma': 0.1,   'kEff': 0.05,  'kEffRep': 0.01,  'Nrep': 6,  'DG0': -6},
    {'sigma': 0.1,   'kEff': 0.1,   'kEffRep': 0.1,   'Nrep': 12, 'DG0': -2},
    {'sigma': 1.0,   'kEff': 0.1,   'kEffRep': 0.1,   'Nrep': 12, 'DG0': 0},
    {'sigma': 1.0,   'kEff': 0.025, 'kEffRep': 0.001, 'Nrep': 0,  'DG0': 2},
    {'sigma': 0.01,  'kEff': 0.05,  'kEffRep': 0.01,  'Nrep': 0,  'DG0': -4},
    {'sigma': 0.1,   'kEff': 0.025, 'kEffRep': 0.1,   'Nrep': 6,  'DG0': -8},
]

GOLDEN_FILE = 'golden_reference.npz'


def run_old(params_dict):
    """Run original Theory_v8 code, return (fr, fz, fr0, fz0)."""
    from Theory_v8 import rLimit, averageForce
    data = copy.deepcopy(params_dict)
    rLimit(data)
    fr, fr0 = averageForce('r', data)
    data.pop('boundPartition')
    data.pop('PBoundZ')
    fz, fz0 = averageForce('z', data)
    return fr, fz, fr0, fz0


def run_new(params_dict):
    """Run refactored code, return (fr, fz, fr0, fz0)."""
    from parameters import Parameters
    from theory import rLimit, averageForce_both
    params = Parameters.from_dict(params_dict)
    rLimit(params)
    fr, fz, fr0, fz0 = averageForce_both(params)
    return fr, fz, fr0, fz0


def generate_golden():
    """Generate golden reference from Theory_v8.py."""
    results = []
    for i, overrides in enumerate(TEST_CASES):
        data = copy.deepcopy(BASE_DATA)
        data.update(overrides)
        fr, fz, fr0, fz0 = run_old(data)
        results.append([fr, fz, fr0, fz0])
        print(f'  Case {i}: DG={overrides["DG0"]}, sigma={overrides["sigma"]}, '
              f'kEff={overrides["kEff"]} -> fr={fr:.15e}, fz={fz:.15e}')
    results = np.array(results)
    np.savez(GOLDEN_FILE, results=results)
    print(f'Saved golden reference to {GOLDEN_FILE} ({len(TEST_CASES)} cases)')
    return results


def test_against_golden():
    """Compare refactored code against golden reference."""
    golden = np.load(GOLDEN_FILE)['results']
    labels = ['fr', 'fz', 'fr0', 'fz0']
    failures = 0
    for i, overrides in enumerate(TEST_CASES):
        data = copy.deepcopy(BASE_DATA)
        data.update(overrides)
        new = run_new(data)
        for j, (old_val, new_val) in enumerate(zip(golden[i], new)):
            if old_val != new_val:
                print(f'  FAIL case {i} {labels[j]}: '
                      f'old={old_val!r}, new={new_val!r}, diff={abs(old_val-new_val):.2e}')
                failures += 1
    if failures == 0:
        print(f'All {len(TEST_CASES) * 4} comparisons PASSED (exact equality)')
    else:
        print(f'{failures} FAILURES out of {len(TEST_CASES) * 4} comparisons')
    return failures == 0


def test_favg_vs_fgivenb():
    """Sanity check: <F> and <F|B> must differ when bound fraction < 1.

    At weak binding (DG >= 0), the bound fraction is well below 1,
    so fr != fr0 and fz != fz0. If they are equal, something is wrong
    with the force weighting logic.
    """
    weak_cases = [c for c in TEST_CASES if c['DG0'] >= 0]
    if not weak_cases:
        print('  No weak-binding test cases (DG >= 0), skipping.')
        return True

    failures = 0
    for i, overrides in enumerate(weak_cases):
        data = copy.deepcopy(BASE_DATA)
        data.update(overrides)
        fr, fz, fr0, fz0 = run_new(data)
        if fr == fr0:
            print(f'  FAIL weak case DG={overrides["DG0"]}: <Fr> == <Fr|B> = {fr!r}')
            failures += 1
        if fz == fz0:
            print(f'  FAIL weak case DG={overrides["DG0"]}: <Fz> == <Fz|B> = {fz!r}')
            failures += 1
    total = len(weak_cases) * 2
    if failures == 0:
        print(f'All {total} <F> vs <F|B> distinctness checks PASSED')
    else:
        print(f'{failures} FAILURES out of {total} distinctness checks')
    return failures == 0


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'generate':
        print('Generating golden reference...')
        generate_golden()
    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        print('Testing against golden reference...')
        ok1 = test_against_golden()
        print()
        print('Testing <F> vs <F|B> distinctness...')
        ok2 = test_favg_vs_fgivenb()
        sys.exit(0 if (ok1 and ok2) else 1)
    else:
        print('Usage: python test_regression.py [generate|test]')
        sys.exit(1)
