# Theory_v9_mp.py – Hybrid mpmath/numpy rewrite of Theory_v8.py
#
# Strategy: use mpmath ONLY where float64 overflows (the pL self-consistency
# equation involves exp(-DG0/kbT) which is exp(40) ≈ 2e17 – fine in float64).
# The real danger is when sigma is large AND DG0 is very negative, making
# the integral in the pL equation very large, so pL becomes extremely small.
# We solve for log(pL) instead, which is numerically stable.
#
# Key improvements over v8:
#   1. Solve for log(pL) via bisection – avoids underflow when pL → 0.
#   2. The pL integral is evaluated in log-space where needed.
#   3. pL(z) is cached across z-samples.
#   4. All outer integrals use numpy Simpson (fast).
#   5. mpmath used only as fallback for extreme parameter regimes.

import numpy as np
import scipy
import scipy.special
from scipy.integrate import simpson as sp_simpson
from scipy.optimize import brentq
import mpmath
import warnings

mpmath.mp.dps = 50

def _mp(x):
    if isinstance(x, (mpmath.mpf, mpmath.mpc)):
        return mpmath.re(x) if isinstance(x, mpmath.mpc) else x
    return mpmath.mpf(str(float(x)))

def _float(x):
    if isinstance(x, mpmath.mpc):
        return float(mpmath.re(x))
    return float(x)

# ======================================================================
#  Core physics
# ======================================================================

def chi_np(r, z, data):
    """Bond Boltzmann factor in float64 with overflow protection."""
    DG0 = data['DG0']; kEff = data['kEff']
    x0 = data['x0']; kbT = data['kbT']
    dist = np.sqrt(r**2 + z**2)
    DGcnf = 0.5 * kEff * (dist - x0)**2
    arg = -(DG0 + DGcnf) / kbT
    if np.any(np.abs(arg) > 700):
        raise ValueError(f"Unsolvable Overflow in chi_np: arg = {arg}")
    return np.exp(arg)


def log_chi_np(r, z, data):
    """log of chi – never overflows."""
    DG0 = data['DG0']; kEff = data['kEff']
    x0 = data['x0']; kbT = data['kbT']
    dist = np.sqrt(r**2 + z**2)
    DGcnf = 0.5 * kEff * (dist - x0)**2
    return -(DG0 + DGcnf) / kbT


def pureForce_np(r, z, data):
    kEff = data['kEff']; x0 = data['x0']
    return -kEff * np.abs(np.sqrt(r**2 + z**2) - x0)


# ======================================================================
#  rLimit
# ======================================================================

def rLimit(data):
    '''Calculates the limit of integration for r(z).'''
    verbose = data.get('verbose', False)
    epsMin = 1e-9
    kEff = data['kEff']; kEffRep = data['kEffRep']; kbT = data['kbT']

    if kEffRep == 0.0:
        print("Resetting kEffRep to kEff")
        kEffRep = kEff; data['kEffRep'] = kEffRep

    npt = data['NP_type']
    #factor = data.get('rLimitFactor', 0)
    #if factor == 0:
    #    raise ValueError('rLimitFactor must be set to a non-zero value')

    if npt == 'full':
        rNP = data['rNP_hard'] + np.sqrt(3.0 * kbT / kEff)
        data['rNP'] = rNP
        Nkuhn = data['Nkuhn']
        akuhn = data['akuhn']
        rMax = rNP + Nkuhn * akuhn
        data['rMax'] = rMax
    elif npt == 'star_polymer':
        Nkuhn = data['Nkuhn']
        akuhn = data['akuhn']
        data['Ree'] = np.sqrt(Nkuhn) * akuhn
        data['rNP'] = data['Ree']
        rMax = Nkuhn * akuhn
        data['rMax'] = rMax
    else:
        raise ValueError('NP_type not recognized')

    data['rMax'] = rMax
    tot = data['nIntSamples']
    dr = rMax / tot
    data['dr'] = dr
    data['rSamples'] = np.arange(tot) * dr + epsMin
    data['zSamples'] = data['rSamples'].copy()
    data['calculated'] = True
    for key in ('boundPartition', 'PBoundZ', '_pL_cache'):
        data.pop(key, None)

    if verbose:
        print(f"rLimit = {rMax:.6g}, dr = {dr:.6g}, nSamples = {tot}")


# ======================================================================
#  pL equation – solved in log-space for numerical stability
# ======================================================================

def _integral_pL_logspace(log_pL, z, data):
    """Evaluate ∫ 2π r σ χ pL / (1 + NL χ pL) dr using log-sum-exp tricks.

    Let log_x_i = log(χ_i) + log_pL  for each r sample point.
    Then χ·pL = exp(log_x_i) and the integrand is:
        2π r σ exp(log_x_i) / (1 + NL exp(log_x_i))

    For stability when exp(log_x_i) is huge:
        exp(log_x) / (1 + NL exp(log_x))  =  1/NL * 1/(1 + NL^{-1} exp(-log_x))
                                            =  1/NL * sigmoid(log_x - log(NL))
    We use this form when log_x > 30.
    """
    rSamples = data['rSamples']
    sigma = data['sigma']
    NL = data['NL']

    log_chi = log_chi_np(rSamples, z, data)  # array
    log_x = log_chi + log_pL  # array

    # Compute integrand element-wise
    integrand = np.zeros_like(rSamples)
    for i in range(len(rSamples)):
        lx = log_x[i]
        if lx > 500:
            # χ·pL >> 1, so  σ·χ·pL / (1 + NL·χ·pL) ≈ σ/NL
            val = sigma / NL
        elif lx < -500:
            # χ·pL ≈ 0
            val = 0.0
        elif lx > 30:
            # Use stable form: 1/NL * sigmoid
            val = sigma / NL / (1.0 + np.exp(-lx) / NL)
        else:
            ex = np.exp(lx)
            val = sigma * ex / (1.0 + NL * ex)
        integrand[i] = 2 * np.pi * rSamples[i] * val

    return sp_simpson(integrand, x=rSamples)


def _function_logpL(log_pL, z, data):
    """exp(log_pL) + integral(log_pL) - 1 = 0"""
    pL = np.exp(log_pL) if log_pL > -700 else 0.0
    integral_val = _integral_pL_logspace(log_pL, z, data)
    return pL + integral_val - 1.0


def pLEq(z, data):
    """Solve for pL at given z. Works in log(pL) space for stability."""
    cache = data.get('_pL_cache', {})
    z_key = round(float(z), 12)
    if z_key in cache:
        return cache[z_key]

    z_val = float(z)

    # Bounds for log(pL): log(pL) in [-1000, 0]
    # At log_pL = 0 (pL=1): f = 1 + integral - 1 = integral >= 0  → f >= 0
    # At log_pL = -1000 (pL≈0): f = 0 + ~0 - 1 = -1  → f < 0
    # So root exists in [-1000, 0].

    f = lambda lp: _function_logpL(lp, z_val, data)

    # Check bounds
    fa = f(0.0)
    fb = f(-1000.0)

    if fa * fb > 0:
        # Both same sign – check if pL ≈ 1 (integral ≈ 0)
        if abs(fa) < 1e-10:
            sol_float = 1.0
        else:
            # Try mpmath fallback
            sol_float = _pLEq_mpmath(z_val, data)
            cache[z_key] = sol_float
            data['_pL_cache'] = cache
            return sol_float
    else:
        try:
            log_pL_sol = brentq(f, -1000.0, 0.0, xtol=1e-14, rtol=1e-14)
            sol_float = np.exp(log_pL_sol)
        except ValueError:
            sol_float = _pLEq_mpmath(z_val, data)

    cache[z_key] = sol_float
    data['_pL_cache'] = cache
    return sol_float


def _pLEq_mpmath(z, data):
    """Fallback: solve pL equation with full mpmath precision."""
    NL = _mp(data['NL'])
    sigma = _mp(data['sigma'])
    rMax = _mp(data['rMax'])
    z_mp = _mp(z)
    DG0 = _mp(data['DG0'])
    kEff = _mp(data['kEff'])
    x0 = _mp(data['x0'])
    kbT = _mp(data['kbT'])

    def chi_m(r):
        dist = mpmath.sqrt(r**2 + z_mp**2)
        DGcnf = kEff / 2 * (dist - x0)**2
        return mpmath.exp(-(DG0 + DGcnf) / kbT)

    def f(pL):
        def integrand(r):
            c = chi_m(r)
            x = c * pL
            return 2 * mpmath.pi * r * sigma * x / (1 + NL * x)
        integral = mpmath.quad(integrand, [mpmath.mpf('1e-12'), rMax])
        return pL + mpmath.re(integral) - 1

    # Bisection in mpmath
    a = mpmath.mpf('1e-100')
    b = mpmath.mpf('1')
    for _ in range(300):
        mid = (a + b) / 2
        fm = f(mid)
        if mpmath.fabs(fm) < mpmath.mpf('1e-40') or (b - a) / 2 < mpmath.mpf('1e-45'):
            return _float(mid)
        if f(a) * fm < 0:
            b = mid
        else:
            a = mid
    return _float((a + b) / 2)


# ======================================================================
#  pR
# ======================================================================

def pREq(r, z, pL, data):
    NL = data['NL']
    c = chi_np(np.asarray(r, dtype=float), np.asarray(z, dtype=float), data)
    return 1.0 / (1.0 + NL * pL * c)


# ======================================================================
#  Free energy
# ======================================================================

def integralPR(z, pL, data):
    """∫ 2π σ r [ln(pR) + (1-pR)/2] dr   using numpy Simpson."""
    rSamples = data['rSamples']
    sigma = data['sigma']
    NL = data['NL']

    # pR = 1/(1 + NL pL chi)
    c = chi_np(rSamples, z, data)
    NL_pL_c = NL * pL * c
    PR = 1.0 / (1.0 + NL_pL_c)

    # For very small PR, log(PR) = -log(1 + NL pL chi)
    # Use log1p for better accuracy when NL_pL_c is large
    log_PR = -np.log1p(NL_pL_c)

    integrand = 2 * np.pi * sigma * rSamples * (log_PR + (1 - PR) / 2)
    return sp_simpson(integrand, x=rSamples)


def ABonds(z, data):
    kbT = data['kbT']
    pL = pLEq(z, data)
    NL = data['NL']

    if pL == 0:
        return -np.inf

    part1 = kbT * NL * (np.log(pL) + 0.5 * (1 - pL))
    part2 = integralPR(z, pL, data)

    exponent = -(part1 + part2) / kbT

    if exponent < 0 :
        raise ValueError(f"Numerical error, mathematically this cannot happen Abond >=0 but here equal to {exponent} < 0")
    if exponent > 700:
        # qBound is huge → ABonds ≈ -(part1+part2) - kbT*ln(1) (approx)
        return -np.inf
    else:
        qBound = np.exp(exponent) - 1.0

    if qBound > 0:
        return -kbT * np.log(qBound)
    else:
        return +np.inf


def RepIntegral(z, K):
    return 0.5 * (1.0 + scipy.special.erf(np.sqrt(K) * z))


def ARep(z, data):
    kbT = data['kbT']
    assert data['x0'] == 0, "Theory requires x0 = 0"
    steric  = -kbT * data['NL']   * np.log(max(RepIntegral(z, data['kEff']),    1e-300))
    steric += -kbT * data['Nrep'] * np.log(max(RepIntegral(z, data['kEffRep']), 1e-300))
    return steric


def ATot(z, data):
    return ABonds(z, data) + ARep(z, data)


# ======================================================================
#  Omega
# ======================================================================

def Omega(data):
    if 'boundPartition' in data:
        return data['boundPartition']
    zSamples = data['zSamples']
    kbT = data['kbT']
    myA = np.zeros(len(zSamples))
    for i, zi in enumerate(zSamples):
        a_val = ATot(zi, data)
        if np.isposinf(a_val):
            myA[i] = 0.0
        else:
            myA[i] = np.exp(np.clip(-a_val / kbT, -700, 700))
    result = sp_simpson(myA, x=zSamples)
    data['boundPartition'] = result
    return result


def pCondBound(z, data):
    kbT = data['kbT']
    den = Omega(data)
    if den <= 0:
        return 0.0
    a_val = ATot(z, data)
    if np.isposinf(a_val):
        return 0.0
    return np.exp(np.clip(-a_val / kbT, -700, 700)) / den


# ======================================================================
#  Force
# ======================================================================

def versor(r, z, direction):
    angle = np.arctan2(r, z)
    if direction == 'r':
        return np.sin(angle)
    elif direction == 'z':
        return np.cos(angle)
    raise ValueError(f"Unknown direction: {direction}")


def forceSingle(r, z, direction, data):
    """Force at (r,z). r scalar, z scalar or array."""
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    r_val = float(r)
    NL = data['NL']

    pL_arr = np.array([pLEq(float(zi), data) for zi in z_arr])
    c = chi_np(r_val, z_arr, data)
    myV = versor(r_val, z_arr, direction)
    fMag = pureForce_np(r_val, z_arr, data) * myV
    pR = 1.0 / (1.0 + NL * pL_arr * c)

    result = NL * pR * pL_arr * c * fMag
    return result if len(result) > 1 else result[0]


def forceBound(r, direction, data):
    zSamples = data['zSamples']
    myForce = forceSingle(r, zSamples, direction, data)

    if 'PBoundZ' not in data:
        data['PBoundZ'] = np.array([pCondBound(zi, data) for zi in zSamples])

    return sp_simpson(myForce * data['PBoundZ'], x=zSamples)


def averageForce(direction, data):
    #frac = fractionBound(data)
    totRec = totRecAds(data)

    rSamples = data['rSamples']
    sigma = data['sigma']
    myForce = np.array([forceBound(ri, direction, data) for ri in rSamples])
    myIntegrand = 2 * np.pi * rSamples * sigma * myForce
    finalIntegral = sp_simpson(myIntegrand, x=rSamples)

    # Previous version that had problems
    frac = fractionBound( data )
    result  = (1.0 / totRec) * finalIntegral if totRec > 0 else 0.0
    result2 = (frac / totRec) * finalIntegral if totRec > 0 else 0.0
    print(f"frac = {frac}, result = {result}, result2 = {result2}")
    return result, result2

def totRecAds(data):
    if not data.get('calculated', False):
        raise ValueError('Call rLimit first')
    # integrate 2 pi r sigma dr from 0 to rMax
    sigma = data['sigma']
    result = np.pi * data['rMax']**2 * sigma
    return result

def fractionBound( data ):
  '''Use chemical equilibrium between sites and NPs to calculate the fraction
  of bound sites'''
  cV0 = data[ 'cV0'] #This is the molar concentration of viruses
  cNP0 = data[ 'cNP0'] #This is the molar concentration of nanoparticles 
  rV = data[ 'rV' ] #This is the radius of a single virus 
  rNP = data[ 'rNP' ] #This is the radius of a Nanoparticle
  
  #Nr of adsorption sites on a single virus
  nSites = max( 2, int( 4.0 * ( rV / rNP )**2 ) )
  #Concentration of adsorption sites in solution
  cS0 = cV0 * nSites
  #Area of an adsorption site
  As = 4 * np.pi * rV**2 / nSites

  #Calculate the bound partition function

  try:
    Kbind = data[ 'Kbind' ]
    #print( f"K bind is: {Kbind} ")
  except KeyError:
    Kbind = As * Omega( data )
    #print( f"Calculate K bind is: {Kbind} ")
  
  #Calculate the overall result
  num1 = ( cNP0 + cS0 ) * Kbind + 1.0
  num2 = 1.0 - np.sqrt( 1.0 - ( 4 * Kbind**2 * cNP0 * cS0 / num1**2 ) ) 
  den = 2.0 * Kbind * cS0
  result = num1 * num2 / den

  return result