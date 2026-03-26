"""Unit conversion utilities for the antiviral NP-virus binding theory.

Reduced unit system:
    Length:          Ree = sqrt(3 kBT / kEff)
    Energy:          kBT
    Force:           kBT / Ree
    Spring constant: kBT / Ree^2  (kEff -> 3 always)
    Surface density: 1 / Ree^2
    Number density:  1 / Ree^3
"""

import copy
import numpy as np

from .parameters import Parameters

CONV_KBT_NM_TO_PN = 2.6  # kBT/nm -> pN at T = 300 K


def compute_Ree(kEff: float, kbT: float) -> float:
    """End-to-end distance of the ligand Gaussian chain."""
    return np.sqrt(3.0 * kbT / kEff)


def to_reduced(params: Parameters) -> tuple[Parameters, float, float]:
    """Convert real-unit Parameters to reduced-unit Parameters.

    Returns (reduced_params, Ree, kbT_real) so the caller can convert
    outputs back to real units.
    """
    kbT = params.kbT
    kEff = params.kEff
    Ree = compute_Ree(kEff, kbT)

    rp = Parameters(
        kEff=3.0,                                # always 3 in reduced units
        kEffRep=params.kEffRep * Ree**2 / kbT,
        x0=params.x0 / Ree,                      # should be 0
        sigma=params.sigma * Ree**2,
        NL=params.NL,
        Nrep=params.Nrep,
        kbT=1.0,
        DG0=params.DG0 / kbT,                    # already in kBT → no change when kbT=1
        L0=params.L0 / Ree,
        maxDG=params.maxDG / kbT,
        epsilon_self=params.epsilon_self,
        nIntSamples=params.nIntSamples,
        verbose=params.verbose,
        cV0=params.cV0 * Ree**3,
        cNP0=params.cNP0 * Ree**3,
        rV=params.rV / Ree,
        NP_type=params.NP_type,
        rNP=params.rNP / Ree,
    )
    return rp, Ree, kbT


def forces_to_real(fr, fz, fr0, fz0, Ree, kbT):
    """Convert reduced-unit forces (kBT/Ree) to real units.

    Returns (fr_real, fz_real, fr0_real, fz0_real, fr_pN, fz_pN)
    where real forces are in kBT/nm and pN.
    """
    scale = kbT / Ree  # kBT/Ree -> kBT/nm
    fr_r = fr * scale
    fz_r = fz * scale
    fr0_r = fr0 * scale
    fz0_r = fz0 * scale
    fr_pN = fr_r * CONV_KBT_NM_TO_PN
    fz_pN = fz_r * CONV_KBT_NM_TO_PN
    return fr_r, fz_r, fr0_r, fz0_r, fr_pN, fz_pN
