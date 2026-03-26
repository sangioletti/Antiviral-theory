"""CLI smoke tests: verify commands produce expected output files."""

import os
import sys
import subprocess
import tempfile
import math


def _run_cli(args):
    """Run the antiviral CLI with given args, return (returncode, stdout)."""
    cmd = [sys.executable, '-m', 'antiviral.run_calculations'] + args
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=os.path.join(os.path.dirname(__file__), '..'))
    return result.returncode, result.stdout + result.stderr


def test_sweep_produces_files():
    with tempfile.TemporaryDirectory() as d:
        rc, out = _run_cli([
            'sweep', '--sweep-sigma', '0.1', '--sweep-kEff', '0.05',
            '--sweep-kEffRep', '0.01', '--sweep-Nrep', '6',
            '--sweep-DG', '-8', '--output-dir', d,
        ])
        if rc != 0:
            print(f'  FAIL: sweep exited {rc}\n{out}')
            return False
        files = os.listdir(d)
        has_dat = any(f.endswith('.dat') for f in files)
        has_pdf = any(f.endswith('.pdf') for f in files)
        if not (has_dat and has_pdf):
            print(f'  FAIL: expected .dat and .pdf, got {files}')
            return False
    print('test_sweep_produces_files: PASSED')
    return True


def test_nl_sweep_produces_files():
    with tempfile.TemporaryDirectory() as d:
        rc, out = _run_cli([
            'nl-sweep', '--DG0', '-8', '--kEff', '0.05',
            '--sweep-sigma', '0.1', '--sweep-NL', '1:5:1',
            '--input-units', 'reduced', '--output-dir', d,
        ])
        if rc != 0:
            print(f'  FAIL: nl-sweep exited {rc}\n{out}')
            return False
        files = os.listdir(d)
        has_dat = any(f.endswith('.dat') for f in files)
        has_pdf = any('Favg' in f and f.endswith('.pdf') for f in files)
        if not (has_dat and has_pdf):
            print(f'  FAIL: expected .dat and Favg .pdf, got {files}')
            return False
    print('test_nl_sweep_produces_files: PASSED')
    return True


def test_output_units_real():
    """When --output-units real, pN columns must not be nan."""
    with tempfile.TemporaryDirectory() as d:
        rc, _ = _run_cli([
            'sweep', '--sweep-sigma', '0.1', '--sweep-kEff', '0.05',
            '--sweep-kEffRep', '0.01', '--sweep-Nrep', '6',
            '--sweep-DG', '-8', '--no-plot',
            '--output-units', 'real', '--output-dir', d,
        ])
        if rc != 0:
            print('  FAIL: sweep exited non-zero')
            return False
        dat_files = [f for f in os.listdir(d) if f.endswith('.dat')]
        for fn in dat_files:
            with open(os.path.join(d, fn)) as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    if 'nan' in line:
                        print(f'  FAIL: found nan in {fn} with --output-units real')
                        return False
    print('test_output_units_real: PASSED')
    return True


if __name__ == '__main__':
    ok = all([test_sweep_produces_files(), test_nl_sweep_produces_files(),
              test_output_units_real()])
    sys.exit(0 if ok else 1)
