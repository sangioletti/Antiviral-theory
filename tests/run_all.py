#!/usr/bin/env python
"""Run all tests. Exit non-zero on any failure."""

import sys
import os

# Ensure tests/ is on path for conftest, and chdir to project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from test_golden import test_real_vs_v8, test_reduced_vs_v8, test_favg_vs_fgivenb
from test_reduced_vs_real import test_chi, test_pureForce, test_ABonds, test_ARep, test_Omega
from test_units import test_keff_always_3, test_kbt_always_1, test_param_round_trip, test_forces_to_real_scale
from test_cli import test_sweep_produces_files, test_nl_sweep_produces_files, test_output_units_real
from test_nl_sweep import test_nl_real_input_reduced_output, test_nl_reduced_input_reduced_output, test_nl_real_input_real_output

results = []


def run(name, fn):
    print(f'\n--- {name} ---')
    ok = fn()
    results.append((name, ok))
    return ok


print('=' * 60)
print('GOLDEN REFERENCE TESTS (vs Theory_v8.py)')
print('=' * 60)
run('real_vs_v8', test_real_vs_v8)
run('reduced_vs_v8', test_reduced_vs_v8)
run('favg_vs_fgivenb', test_favg_vs_fgivenb)

print('\n' + '=' * 60)
print('FUNCTION-LEVEL: REDUCED vs REAL')
print('=' * 60)
run('chi', test_chi)
run('pureForce', test_pureForce)
run('ABonds', test_ABonds)
run('ARep', test_ARep)
run('Omega', test_Omega)

print('\n' + '=' * 60)
print('UNIT CONVERSION')
print('=' * 60)
run('keff_always_3', test_keff_always_3)
run('kbt_always_1', test_kbt_always_1)
run('param_round_trip', test_param_round_trip)
run('forces_to_real_scale', test_forces_to_real_scale)

print('\n' + '=' * 60)
print('CLI SMOKE TESTS')
print('=' * 60)
run('sweep_produces_files', test_sweep_produces_files)
run('nl_sweep_produces_files', test_nl_sweep_produces_files)
run('output_units_real', test_output_units_real)

print('\n' + '=' * 60)
print('NL SWEEP UNIT PIPELINE')
print('=' * 60)
run('nl_real_in_reduced_out', test_nl_real_input_reduced_output)
run('nl_reduced_in_reduced_out', test_nl_reduced_input_reduced_output)
run('nl_real_in_real_out', test_nl_real_input_real_output)

# Summary
print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
passed = sum(1 for _, ok in results if ok)
total = len(results)
for name, ok in results:
    print(f'  {"PASS" if ok else "FAIL"}  {name}')
print(f'\n{passed}/{total} tests passed.')
sys.exit(0 if passed == total else 1)
