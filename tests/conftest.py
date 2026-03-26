"""Shared test data and helpers."""

import copy
import os

from antiviral.parameters import Parameters

# ---------------------------------------------------------------------------
# Canonical test data — single source of truth
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

# Path to Theory_v8.py on the main worktree
V8_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'main')


def make_params(overrides: dict = None) -> Parameters:
    """Create a Parameters object from BASE_DATA with optional overrides."""
    data = copy.deepcopy(BASE_DATA)
    if overrides:
        data.update(overrides)
    return Parameters.from_dict(data)


def make_data(overrides: dict = None) -> dict:
    """Create a raw data dict from BASE_DATA with optional overrides."""
    data = copy.deepcopy(BASE_DATA)
    if overrides:
        data.update(overrides)
    return data
