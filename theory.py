"""Antiviral NP-virus binding theory.

Implements the theoretical model for nanoparticle-virus interactions
via receptor-ligand binding with steric repulsion from grafted polymers.
Equations referenced correspond to the JCP paper.
"""

import numpy as np
import scipy
from scipy.integrate import quad, simpson
from scipy.optimize import bisect

from parameters import Parameters

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EPSILON_MIN = 1e-9   # Minimum offset to prevent r=0 singularity in samples
CONV_KBT_TO_PN = 2.6  # kbT/nm -> pN at T=300K


# ---------------------------------------------------------------------------
# Spatial sampling
# ---------------------------------------------------------------------------
def rLimit(params: Parameters) -> None:
    """Calculate integration limits and sampling grid.

    Sets params.rMax, params.dr, params.rSamples, params.zSamples.
    Must be called before any force/energy calculation.
    """
    kEff = params.kEff
    kEffRep = params.kEffRep
    if kEffRep == 0.0:
        print("Somewhat unexpected, check all is fine. Resetting kEffRep to kEff for simple ligand")
        kEffRep = kEff
    kbT = params.kbT

    if params.NP_type == 'full':
        rNP = params.rNP + np.sqrt(3.0 * kbT / kEff)
        params.rNP = rNP
        rMax = np.sqrt(np.pi * rNP**2)
    elif params.NP_type == 'star_polymer':
        rNP = np.sqrt(3.0 * kbT / min(kEff, kEffRep))
        params.rNP = rNP
        rMax = np.sqrt(np.pi * rNP**2)
    elif params.NP_type == 'fixed':
        rMax = 3 * np.sqrt(3.0 * kbT / kEff)
        params.rNP = rMax
    else:
        raise ValueError('NP_type not recognized')

    params.rMax = rMax
    DGeff = -np.log(chi(rMax, z=0, params=params))
    if params.verbose:
        print(f'rLimit is {rMax}')
        print(f'RESETTING rLimit')
        print(f'WARNING NEW rLimit to calculation based on size of site. New rLimit = {rMax} ')
        print(f'DG at rLimit is {DGeff}')

    tot = params.nIntSamples
    dr = rMax / tot
    params.dr = dr
    params.rSamples = np.array(range(tot) * dr + EPSILON_MIN)
    params.zSamples = params.rSamples  # NOTE: zSamples IS rSamples (same object)

    params.calculated = True

    if params.verbose:
        aa = params.rSamples[::10]
        print(f'(PRINTED EVERY TEN ONLY - rSamples {aa}')


# ---------------------------------------------------------------------------
# Bond strength and force primitives
# ---------------------------------------------------------------------------
def chi(r, z, params: Parameters):
    """Bond strength: exp(-(DG0 + DGcnf) / kbT)."""
    DG0 = params.DG0
    kEff = params.kEff
    x0 = params.x0
    kbT = params.kbT
    DGcnf = 0.5 * kEff * (np.sqrt(r**2 + z**2) - x0)**2
    return np.exp(-(DG0 + DGcnf) / kbT)


def pureForce(r, z, params: Parameters):
    """Force magnitude from a single Gaussian spring bond at (r, z)."""
    kEff = params.kEff
    x0 = params.x0
    return -kEff * np.abs(np.sqrt(r**2 + z**2) - x0)


# ---------------------------------------------------------------------------
# Ligand occupancy (Eq. 4)
# ---------------------------------------------------------------------------
def integrand(r, z, sigma, pL, params: Parameters):
    """Integrand for pLEq calculation (Eq. 4)."""
    NL = params.NL
    x = chi(r, z, params) * pL
    return 2 * np.pi * r * sigma * x / (1.0 + NL * x)


def integral(pL, z, params: Parameters):
    """Numerical integral for pLEq (Eq. 4) via Simpson's rule."""
    sigma = params.sigma
    try:
        rSamples = params.rSamples
    except KeyError:
        print("Need to run rLimit once first")
        raise ValueError

    myIntegral = simpson(integrand(rSamples, z, sigma, pL, params), rSamples)
    if params.verbose:
        print(f'Calculated via Simpson: {myIntegral}')

    return myIntegral


def pLEq(z, params: Parameters):
    """Solve for equilibrium ligand occupancy pL (Eq. 4) via bisection.

    Finds pL such that pL + integral(pL, z) - 1 = 0.
    """
    # NOTE: When z is an array (as called from forceSingle via forceBound),
    # the integral computes along the diagonal (r_i, z_i) since
    # zSamples IS rSamples. This yields a single scalar pLEq.
    # This is the intended behavior — do not change.
    a = 1.0   # function_pL is positive here
    b = 0.0   # function_pL is negative here

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
    NL = params.NL
    myChi = chi(r, z, params)
    return 1.0 / (1.0 + NL * pLEq * myChi)


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
        print("Direction is wrong / undeclared")
        raise ValueError


def forceSingle(r, z, direction: str, params: Parameters):
    """Force from bonds at position (r, z) projected along direction (Eq. 6).

    NOTE: When called from forceBound, z is the full zSamples array.
    pLEq(z) then solves a single scalar equation using diagonal integration
    (see pLEq docstring). This is the intended behavior.
    """
    myChi = chi(r, z, params)
    myVersor = versor(r, z, direction=direction)
    myForce = pureForce(r, z, params) * myVersor
    pL = pLEq(z, params)
    pR = pREq(r, z, pL, params)

    NL = params.NL
    result = NL * pR * pL * myChi * myForce

    return result


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

    myIntegrand = myForce * myPBound
    result = simpson(myIntegrand, zSamples)

    return result


# ---------------------------------------------------------------------------
# Total average force (Eq. 7)
# ---------------------------------------------------------------------------
def averageForce(direction: str, params: Parameters):
    """Average force over all receptors including bound fraction (Eq. 7).

    Returns (result, result2) where:
      result  = (fractionBound / totRecAds) * integral
      result2 = (1.0 / totRecAds) * integral
    """
    frac = fractionBound(params)
    totRec = totRecAds(params)

    rSamples = params.rSamples
    sigma = params.sigma
    myForce = np.zeros(len(rSamples))

    for i in range(len(myForce)):
        myForce[i] = forceBound(rSamples[i], direction, params)

    myIntegrand = 2 * np.pi * rSamples * sigma * myForce
    finalIntegral = simpson(myIntegrand, rSamples)

    result = (frac / totRec) * finalIntegral
    result2 = (1.0 / totRec) * finalIntegral

    return result, result2


def averageForce_both(params: Parameters):
    """Compute both r and z force components in a single pass (Eq. 7).

    Equivalent to calling averageForce('r') then averageForce('z'),
    but avoids recomputing fractionBound, totRecAds, and sharing the
    PBoundZ cache more efficiently.

    Returns (fr, fz, fr0, fz0).
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

    # NOTE: nSites uses max(2,...) here vs max(1,...) in totRecAds — intentional.
    nSites = max(2, int(4.0 * (rV / rNP)**2))
    cS0 = cV0 * nSites
    As = 4 * np.pi * rV**2 / nSites

    Kbind = As * Omega(params)

    num1 = (cNP0 + cS0) * Kbind + 1.0
    num2 = 1.0 - np.sqrt(1.0 - (4 * Kbind**2 * cNP0 * cS0 / num1**2))
    den = 2.0 * Kbind * cS0
    result = num1 * num2 / den

    return result


def totRecAds(params: Parameters):
    """Total number of receptors on a single adsorption site."""
    rV = params.rV
    rNP = params.rNP

    if not params.calculated:
        raise ValueError('You need to call rLimit first')

    # NOTE: nSites uses max(1,...) here vs max(2,...) in fractionBound — intentional.
    nSites = max(1, int(4.0 * (rV / rNP)**2))
    params.areaAds = 4 * np.pi * rV**2 / nSites
    sigma = params.sigma
    area = params.areaAds
    result = sigma * area
    return result


# ---------------------------------------------------------------------------
# Conditional bound probability (Eq. 9)
# ---------------------------------------------------------------------------
def pCondBound(z, params: Parameters):
    """P(z | bound state) — Eq. 9."""
    den = Omega(params)
    kbT = params.kbT
    unnormalised = np.exp(-ATot(z, params) / kbT)
    result = unnormalised / den
    return result


# ---------------------------------------------------------------------------
# Free energy components
# ---------------------------------------------------------------------------
def integralPR(z, pLEq, params: Parameters):
    """Receptor contribution to bond free energy."""
    rSamples = params.rSamples
    myIntegral = simpson(integrandPR(rSamples, z, pLEq, params), rSamples)
    if params.verbose:
        sol = quad(integrandPR, 0, np.inf, args=(z, pLEq, params))[0]
        diff = abs(sol - myIntegral)
        print(f'Compare For integralOmega. Numerical {myIntegral}  semi-analitical {sol}, difference {diff}')
    return myIntegral


def ABonds(z, params: Parameters):
    """Binding free energy contribution at distance z."""
    kbT = params.kbT
    pLequi = pLEq(z, params)
    NL = params.NL
    part1 = kbT * NL * (np.log(pLequi) + 0.5 * (1 - pLequi))
    part2 = integralPR(z, pLequi, params)

    qBound = np.exp(-(part1 + part2) / kbT) - 1.0
    if qBound > 0:
        bound = -kbT * np.log(qBound)
    else:
        bound = +np.inf

    return bound


def RepIntegral(z, K):
    """Partition function ratio for a Gaussian chain with confining surface at -z."""
    return 0.5 * (1.0 + scipy.special.erf(np.sqrt(K) * z))


def ARep(z, params: Parameters):
    """Steric repulsion free energy at distance z."""
    kbT = params.kbT
    x0 = params.x0
    assert x0 == 0, "Theory only works for Gaussian chains of zero mean"

    # Contribution from ligands
    steric = -kbT * params.NL * np.log(RepIntegral(z, params.kEff))

    # Contribution from purely steric polymers
    steric += -kbT * params.Nrep * np.log(RepIntegral(z, params.kEffRep))

    return steric


def integrandPR(r, z, pLEq, params: Parameters):
    """Integrand for receptor contribution to bond energy."""
    sigma = params.sigma
    PR = pREq(r, z, pLEq, params)
    result = 2 * np.pi * sigma * r * (np.log(PR) + 0.5 * (1 - PR))
    return result


def ATot(z, params: Parameters):
    """Total free energy at distance z: binding + steric."""
    return ABonds(z, params) + ARep(z, params)


# ---------------------------------------------------------------------------
# Bound partition function (Eq. 10)
# ---------------------------------------------------------------------------
def Omega(params: Parameters):
    """Bound partition function (Eq. 10)."""
    if params._boundPartition is not None:
        return params._boundPartition

    zSamples = params.zSamples
    myA = np.zeros(len(zSamples))
    for i in range(len(zSamples)):
        myA[i] = np.exp(-ATot(zSamples[i], params))
    result = simpson(myA, zSamples)
    params._boundPartition = result

    return result
