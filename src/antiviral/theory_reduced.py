"""Antiviral NP-virus binding theory — reduced-unit implementation.

All calculations assume kBT = 1 and lengths in units of Ree.
Parameters must be converted to reduced units before calling these
functions (see units.to_reduced).

Equations referenced correspond to the JCP paper.
"""

import numpy as np
import scipy
from scipy.integrate import simpson
from scipy.optimize import bisect

from .parameters import Parameters

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EPSILON_MIN = 1e-9   # Minimum offset to prevent r=0 singularity


# ---------------------------------------------------------------------------
# Spatial sampling
# ---------------------------------------------------------------------------
def rLimit(params: Parameters) -> None:
    """Calculate integration limits and sampling grid (reduced units)."""
    kEff = params.kEff
    kEffRep = params.kEffRep
    if kEffRep == 0.0 and params.NP_type == "star_polymer":
        print("Somewhat unexpected, check all is fine. Resetting kEffRep to kEff for simple ligand")
        kEffRep = kEff

    if params.NP_type == 'full':
        rNP = params.rNP + np.sqrt(3.0 / kEff)
        params.rNP = rNP
        rMax = np.sqrt(np.pi * rNP**2)
    elif params.NP_type == 'star_polymer':
        rNP = np.sqrt(3.0 / min(kEff, kEffRep))
        params.rNP = rNP
        rMax = np.sqrt(np.pi * rNP**2)
    elif params.NP_type == 'fixed':
        rMax = 3 * np.sqrt(3.0 / kEff)
        params.rNP = rMax
    else:
        raise ValueError('NP_type not recognized')

    params.rMax = rMax
    DGeff = -np.log(chi(rMax, z=0, params=params))
    if params.verbose:
        print(f'rLimit is {rMax}')
        print(f'DG at rLimit is {DGeff}')

    tot = params.nIntSamples
    dr = rMax / tot
    params.dr = dr
    params.rSamples = np.array(range(tot) * dr + EPSILON_MIN)
    params.zSamples = params.rSamples

    params.calculated = True


# ---------------------------------------------------------------------------
# Bond strength and force primitives
# ---------------------------------------------------------------------------
def chi(r, z, params: Parameters):
    """Bond strength: exp(-(DG0 + 0.5*kEff*d^2)).  [kBT = 1]"""
    DGcnf = 0.5 * params.kEff * (np.sqrt(r**2 + z**2) - params.x0)**2
    return np.exp(-(params.DG0 + DGcnf))


def pureForce(r, z, params: Parameters):
    """Force magnitude from a single Gaussian spring bond at (r, z)."""
    return -params.kEff * np.abs(np.sqrt(r**2 + z**2) - params.x0)


# ---------------------------------------------------------------------------
# Ligand occupancy (Eq. 4)
# ---------------------------------------------------------------------------
def integrand(r, z, sigma, pL, params: Parameters):
    """Integrand for pLEq calculation (Eq. 4)."""
    x = chi(r, z, params) * pL
    return 2 * np.pi * r * sigma * x / (1.0 + params.NL * x)


def integral(pL, z, params: Parameters):
    """Numerical integral for pLEq (Eq. 4) via Simpson's rule."""
    sigma = params.sigma
    try:
        rSamples = params.rSamples
    except KeyError:
        print("Need to run rLimit once first")
        raise ValueError

    myIntegral = simpson(integrand(rSamples, z, sigma, pL, params), rSamples)
    return myIntegral


def pLEq(z, params: Parameters):
    """Solve for equilibrium ligand occupancy pL (Eq. 4) via bisection."""
    a = 1.0
    b = 0.0
    pLOut = bisect(function_pL, a, b, args=(z, params))
    if pLOut == 0.0:
        print("pL too small?")
    return pLOut


def function_pL(pL, z, params: Parameters):
    """Objective function for bisection: pL + integral(pL, z) - 1."""
    return pL + integral(pL, z, params) - 1.0


# ---------------------------------------------------------------------------
# Receptor occupancy (Eq. 5)
# ---------------------------------------------------------------------------
def pREq(r, z, pLEq, params: Parameters):
    """Receptor occupancy probability (Eq. 5)."""
    myChi = chi(r, z, params)
    return 1.0 / (1.0 + params.NL * pLEq * myChi)


# ---------------------------------------------------------------------------
# Force decomposition
# ---------------------------------------------------------------------------
def versor(r, z, direction: str):
    """Directional projection factor (radial or axial component)."""
    angle = np.arctan(r / z)
    if direction == 'r':
        return np.sin(angle)
    elif direction == 'z':
        return np.cos(angle)
    else:
        raise ValueError("Direction is wrong / undeclared")


def forceSingle(r, z, direction: str, params: Parameters):
    """Force from bonds at position (r, z) projected along direction (Eq. 6)."""
    myChi = chi(r, z, params)
    myVersor = versor(r, z, direction=direction)
    myForce = pureForce(r, z, params) * myVersor
    pL = pLEq(z, params)
    pR = pREq(r, z, pL, params)
    return params.NL * pR * pL * myChi * myForce


# ---------------------------------------------------------------------------
# Force averaged over bound state (Eq. 8)
# ---------------------------------------------------------------------------
def forceBound(r, direction: str, params: Parameters):
    """Force at radial position r, averaged over z in the bound state (Eq. 8)."""
    zSamples = params.zSamples
    myForce = forceSingle(r, zSamples, direction, params)

    if params._PBoundZ is not None:
        myPBound = params._PBoundZ
    else:
        myPBound = np.zeros(len(zSamples))
        for i in range(len(zSamples)):
            myPBound[i] = pCondBound(zSamples[i], params)
        params._PBoundZ = myPBound

    return simpson(myForce * myPBound, zSamples)


# ---------------------------------------------------------------------------
# Total average force (Eq. 7)
# ---------------------------------------------------------------------------
def averageForce(direction: str, params: Parameters):
    """Average force over all receptors including bound fraction (Eq. 7)."""
    frac = fractionBound(params)
    totRec = totRecAds(params)

    rSamples = params.rSamples
    sigma = params.sigma
    myForce = np.zeros(len(rSamples))
    for i in range(len(myForce)):
        myForce[i] = forceBound(rSamples[i], direction, params)

    finalIntegral = simpson(2 * np.pi * rSamples * sigma * myForce, rSamples)

    result = (frac / totRec) * finalIntegral
    result2 = (1.0 / totRec) * finalIntegral
    return result, result2


def averageForce_both(params: Parameters):
    """Compute both r and z force components in a single pass (Eq. 7).

    Returns (fr, fz, fr0, fz0) in reduced units.
    """
    frac = fractionBound(params)
    totRec = totRecAds(params)

    rSamples = params.rSamples
    sigma = params.sigma

    force_r = np.zeros(len(rSamples))
    force_z = np.zeros(len(rSamples))
    for i in range(len(rSamples)):
        force_r[i] = forceBound(rSamples[i], 'r', params)
        force_z[i] = forceBound(rSamples[i], 'z', params)

    integrand_r = 2 * np.pi * rSamples * sigma * force_r
    integrand_z = 2 * np.pi * rSamples * sigma * force_z

    int_r = simpson(integrand_r, rSamples)
    int_z = simpson(integrand_z, rSamples)

    fr = (frac / totRec) * int_r
    fz = (frac / totRec) * int_z
    fr0 = (1.0 / totRec) * int_r
    fz0 = (1.0 / totRec) * int_z

    return fr, fz, fr0, fz0


# ---------------------------------------------------------------------------
# Binding equilibrium
# ---------------------------------------------------------------------------
def fractionBound(params: Parameters):
    """Fraction of bound adsorption sites via chemical equilibrium."""
    cV0 = params.cV0
    cNP0 = params.cNP0
    rV = params.rV
    rNP = params.rNP

    nSites = max(2, int(4.0 * (rV / rNP)**2))
    cS0 = cV0 * nSites
    As = 4 * np.pi * rV**2 / nSites

    Kbind = As * Omega(params)

    num1 = (cNP0 + cS0) * Kbind + 1.0
    num2 = 1.0 - np.sqrt(1.0 - (4 * Kbind**2 * cNP0 * cS0 / num1**2))
    den = 2.0 * Kbind * cS0
    return num1 * num2 / den


def totRecAds(params: Parameters):
    """Total number of receptors on a single adsorption site."""
    rV = params.rV
    rNP = params.rNP

    if not params.calculated:
        raise ValueError('You need to call rLimit first')

    nSites = max(1, int(4.0 * (rV / rNP)**2))
    params.areaAds = 4 * np.pi * rV**2 / nSites
    return params.sigma * params.areaAds


# ---------------------------------------------------------------------------
# Conditional bound probability (Eq. 9)
# ---------------------------------------------------------------------------
def pCondBound(z, params: Parameters):
    """P(z | bound state) — Eq. 9.  [kBT = 1]"""
    return np.exp(-ATot(z, params)) / Omega(params)


# ---------------------------------------------------------------------------
# Free energy components  [all in units of kBT = 1]
# ---------------------------------------------------------------------------
def integralPR(z, pLEq, params: Parameters):
    """Receptor contribution to bond free energy."""
    rSamples = params.rSamples
    return simpson(integrandPR(rSamples, z, pLEq, params), rSamples)


def ABonds(z, params: Parameters):
    """Binding free energy contribution at distance z.  [kBT = 1]"""
    pLequi = pLEq(z, params)
    NL = params.NL
    part1 = NL * (np.log(pLequi) + 0.5 * (1 - pLequi))
    part2 = integralPR(z, pLequi, params)

    qBound = np.exp(-(part1 + part2)) - 1.0
    if qBound > 0:
        return -np.log(qBound)
    else:
        return +np.inf


def RepIntegral(z, K):
    """Partition function ratio for a Gaussian chain with confining surface at -z."""
    return 0.5 * (1.0 + scipy.special.erf(np.sqrt(K) * z))


def ARep(z, params: Parameters):
    """Steric repulsion free energy at distance z.  [kBT = 1]"""
    assert params.x0 == 0, "Theory only works for Gaussian chains of zero mean"
    steric = -params.NL * np.log(RepIntegral(z, params.kEff))
    steric += -params.Nrep * np.log(RepIntegral(z, params.kEffRep))
    return steric


def integrandPR(r, z, pLEq, params: Parameters):
    """Integrand for receptor contribution to bond energy."""
    PR = pREq(r, z, pLEq, params)
    return 2 * np.pi * params.sigma * r * (np.log(PR) + 0.5 * (1 - PR))


def ATot(z, params: Parameters):
    """Total free energy at distance z: binding + steric.  [kBT = 1]"""
    return ABonds(z, params) + ARep(z, params)


# ---------------------------------------------------------------------------
# Bound partition function (Eq. 10)
# ---------------------------------------------------------------------------
def Omega(params: Parameters):
    """Bound partition function (Eq. 10).  [kBT = 1]"""
    if params._boundPartition is not None:
        return params._boundPartition

    zSamples = params.zSamples
    myA = np.zeros(len(zSamples))
    for i in range(len(zSamples)):
        myA[i] = np.exp(-ATot(zSamples[i], params))
    result = simpson(myA, zSamples)
    params._boundPartition = result

    return result
