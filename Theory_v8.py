#Python program implementing the theory for the future JCP paper

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import scipy
from scipy.integrate import quad
from scipy.integrate import simpson
from scipy.optimize import bisect 

def rLimit( data ):
  '''Calculates the limit of integration for r(z). In
  practice, integrates only until the value of the bond energy is 
  +maxDG. Note that due to the symmetric role played we use the same 
  for zSamples'''
  try:
    verbose = data[ 'verbose' ]
  except KeyError:
    verbose = False
  epsMin = 10**(-9)
  x0 = data[ 'x0' ]
  maxDG = data[ 'maxDG' ]
  DG0 = data[ 'DG0' ]
  kEff = data[ 'kEff' ]
  kEffRep = data[ 'kEffRep' ]
  if kEffRep == 0.0:
    print( "Somewhat unexpected, check all is fine. Resetting kEffRep to kEff for simple ligand" )
    kEffRep = kEff 
  kbT = data[ 'kbT' ]
  rNP = data[ 'rNP' ] 

  if data[ 'NP_type' ] == 'full':
    rNP = data[ 'rNP' ] + np.sqrt( 3.0 * kbT / kEff )
    data[ 'rNP' ] = rNP
    rMax = np.sqrt( np.pi * rNP**2 )
  elif data[ 'NP_type' ] == 'star_polymer':
    rNP = np.sqrt( 3.0 * kbT / min( kEff, kEffRep ) ) 
    data[ 'rNP' ] = rNP
    rMax = np.sqrt( np.pi * rNP**2 )
  elif data[ 'NP_type' ] == 'fixed':
    rMax = 3 * np.sqrt( 3.0 * kbT / kEff ) #This is basically 3 times the gyration radius
    data[ 'rNP' ] = rMax 

    #Old definition here was not really sustainable tbh, it basically modified rMax and thus other stuff according to the value of DG0, which kinda makes
    #no sense since truly this should be an excluded volume term only
    #rMax = np.sqrt( 2.0 * ( maxDG - DG0 ) / kEff ) + x0
  else:
    raise valueError('Value not recognized')
 
  data[ 'rMax' ] = rMax
  DGeff = -np.log( chi( rMax, z = 0, data = data ) ) 
  if verbose:
    print( f'rLimit is {rMax}' )
    print( f'RESETTING rLimit' )
    print( f'WARNING NEW rLimit to calculation based on size of site. New rLimit = {rMax} ' )
    print( f'DG at rLimit is {DGeff}' )

  tot = data[ 'nIntSamples' ]
  dr = rMax / tot
  data[ 'dr' ] = dr
  data[ 'rSamples' ] = np.array( range( tot ) * dr + epsMin )
  data[ 'zSamples' ] = data[ 'rSamples' ]

  data[ 'calculated' ] = True

  if verbose:
    aa = data[ 'rSamples' ][::10]
    print( f'(PRINTED EVERY TEN ONLY - rSamples {aa}' )
  return

def chi( r, z, data ):
  '''Chi is nothing but the bond strength, exp( - ( DG + DGcnf )/kbT )'''
  DG0 = data['DG0']
  kEff = data['kEff']
  x0 = data['x0']
  kbT = data['kbT']
  DGcnf = 0.5 * kEff * ( np.sqrt( r**2 + z**2 ) - x0 )**2 
  return np.exp(  -( DG0 + DGcnf ) / kbT )

def pureForce( r, z, data ):
  '''Returns the force exerted by a bond for a NP at position (r, z) from the surface,
  in cylindrical coordinates'''
  kEff = data['kEff']
  x0 = data['x0']
  kbT = data['kbT']
  return - kEff * np.abs( np.sqrt( r**2 + z**2 ) - x0 ) 

def integrand( r, z, sigma, pL, data ): 
  '''This is the integrand required for the calculation of the equilibrium value of 
  pLEq, Eq.4 in the paper'''
  NL = data[ 'NL' ]
  x = chi( r, z, data ) * pL
  return 2 * np.pi * r * sigma * x / ( 1.0 + NL * x )

def integral( pL, z, data ):
  '''Integral for the calculation of pLEq, Eq.4 in the paper'''
  kEff = data[ 'kEff' ]
  x0 = data[ 'x0' ]
  NL = data[ 'NL' ]
  sigma = data[ 'sigma' ]
  try:
    rSamples = data[ 'rSamples' ]
    dr = data[ 'dr' ]
  except:
    print( "Need to run rLimit once first" )
    raise ValueError

  #if data[ 'verbose' ]: #Additional lines only for debug mode 
  #  myIntegral = np.sum( integrand( rSamples, z, sigma, pL, data ) ) * dr  
  #  sol = quad( integrand, 0, np.inf, args = ( z, sigma, pL, data ) )[ 0 ]
  #  diff = abs( sol - myIntegral )
  #  print( f'Compare. Numerical {myIntegral}  semi-analitical {sol}, difference {diff}' )
  
  #sample at different values of r
  myIntegral = simpson( integrand( rSamples, z, sigma, pL, data ), rSamples )
  if data[ 'verbose' ]: #Additional lines only for debug mode 
    print( f'Calculated via Simpson: {myIntegral}' )

  return myIntegral

def pLEq( z, data ):
  '''Calculation of pLEq, Eq.4 in the paper, via bisection.
  In practice, we find the value of pL for which
  pL + f( pL ) - 1 = 0'''

  myChi = np.exp( data[ 'DG0' ] ) 
  epsilon = min( myChi**( -1.0 ), data[ 'epsilon_self' ] )
  #print( f"Epsilon value for convergence = {epsilon}")

  a = 1.0 #This is the value for which the function is surely positive
  b = 0.0 #This is the value for which the function is surely negative

  pLOut = bisect( function_pL, a, b, args = ( z, data ) )
  #print( f"Solution: {pLOut}" )

  if pLOut == 0.0:
    print( "pL too small?" ) 
     
  return pLOut

def function_pL( pL, z, data ):
  return pL + integral( pL, z, data ) - 1.0

def pREq( r, z, pLEq, data ):
  '''Value of pREq, given once the solution to pLEq exists, Eq. 5 in the paper'''
  NL = data[ 'NL' ]
  myChi = chi( r, z, data )
  
  return 1.0 / ( 1.0 + NL * pLEq * myChi )

def versor( r, z, direction ):
  '''Scaling factor to obtain the radial and z-directed force
  component from the force on a receptor at position (r,z) from the NP'''
  angle = np.arctan( r / z )

  #print( f"r: {r}, z {z[0:2]}" )

  if direction == 'r':
    versor = np.sin( angle )
  elif direction == 'z':
    versor = np.cos( angle )
  else:
    print( "Direction is wrong / undeclared" )
    raise ValueError
  return versor

def forceSingle( r, z, direction, data ):
  '''Eq 6 in the paper'''
  myChi = chi( r, z, data )
  myVersor = versor( r, z, direction = direction )
  #print( f"Direction {direction} versor {myVersor}" )
  myForce = pureForce( r, z, data ) * myVersor
  pL = pLEq( z, data )
  pR = pREq( r, z, pL, data )

  NL = data[ 'NL' ]
  result = NL * pR * pL * myChi * myForce  
 
  return result

def forceBound( r, direction, data ):
  '''Eq. 8, necessary to average force over all possible values of
  z in the bound state'''
  #Calculate first the integrand
  zSamples = data[ 'zSamples' ]
  myForce = forceSingle( r, zSamples, direction, data )
 

  #NEW PART
  try:
    myPBound = data[ 'PBoundZ' ]
  except KeyError:
    zSamples = data[ 'zSamples' ]
    myPBound = np.zeros( len( zSamples ) )
    for i in range( len( zSamples ) ):
      myPBound[ i ] = pCondBound( zSamples[ i ], data )
    #Now copy so you don't need to recalculate it next time
    data[ 'PBoundZ' ] = myPBound
  #END NEW PART 

  #myPBound = np.zeros( len( zSamples ) )
  #for i in range( len( zSamples ) ):
  #  myPBound[ i ] = pCondBound( zSamples[ i ], data )

  #print( f" force {myForce[::5]}" )
  #print( f" pBound {myPBound[::5]}" )
  myIntegrand = myForce * myPBound
  
  #Now integrate! 
  result = simpson( myIntegrand, zSamples )

  return result 

def averageForce( direction, data ):
  ''''This is the force given by Eq.7, that is, the force averaged over all receptors
  and including also the effect due to the number of bound particles and the total
  number of potential receptors'''
  frac = fractionBound( data )

  #This is the total number of receptors on a single adsorption site,
  #the denominator in Eq.7
  totRec = totRecAds( data ) 
 
  #Now we calculate the integrand
  rSamples = data[ 'rSamples' ]
  sigma = data[ 'sigma' ]
  myForce = np.zeros( len( rSamples ) )

  for i in range( len( myForce ) ):
    #print( f'rSamples: {rSamples[i]}' )
    myForce[ i ] = forceBound( rSamples[ i ], direction, data )
  #myForce = forceBound( rSamples, direction, data )
 
  myIntegrand = 2 * np.pi * rSamples * sigma * myForce

  #Now integrate! 
  finalIntegral = simpson( myIntegrand, rSamples )

  #print( f"Sum of forces {finalIntegral}, fraction bound particles {frac}, totRec {totRec}" )

  result = ( frac / totRec ) * finalIntegral 
  
  result2 = ( 1.0 /  totRec ) * finalIntegral 

  return result, result2

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

  recPerSite = data[ 'sigma' ] * As 
  #print( f"Receptor per site: {recPerSite}" )

  try:
    Kbind = data[ 'Kbind' ]
    #print( f"K bind is: {Kbind} ")
  except KeyError:
    Kbind = As * Omega( data )
    #print( f"Calculate K bind is: {Kbind} ")
  
  #Kbind = As * Omega( data )


  #Calculate the overall result
  num1 = ( cNP0 + cS0 ) * Kbind + 1.0
  num2 = 1.0 - np.sqrt( 1.0 - ( 4 * Kbind**2 * cNP0 * cS0 / num1**2 ) ) 
  den = 2.0 * Kbind * cS0
  result = num1 * num2 / den

  return result
  

def totRecAds( data ):
  #cV0 = data[ 'cV0'] #This is the molar concentration of viruses
  #cNP0 = data[ 'cNP0'] #This is the molar concentration of nanoparticles 
  rV = data[ 'rV' ] #This is the radius of a single virus 
  rNP = data[ 'rNP' ] #This is the radius of a Nanoparticle

  if data[ 'calculated' ] == False:
    raise ValueError( 'You need to call rLimit first' )
  
  #Nr of adsorption sites on a single virus
  nSites = max( 1, int( 4.0 * ( rV / rNP )**2 ) )
  #Area of an adsorption site
  data[ 'areaAds' ] = 4 * np.pi * rV**2 / nSites
  sigma = data[ 'sigma' ] #This is the grafting density of Receptors
  area = data[ 'areaAds' ] #This is the area of an adsorption site
  result = sigma * area
  return result 


def pCondBound( z, data ):
  '''Eq. 9 in the paper, p( be at z | your are in the bound state)''' 
  den = Omega( data )
  kbT = data[ 'kbT' ]
  unnormalised = np.exp( -ATot( z, data ) / kbT )
  
  result = unnormalised / den 

  return result

def integralForce( z, direction, data ):
  '''Force weighted by the number of bonds at a certain position
  Eq.6/7 in the paper'''
  rSamples = data[ 'rSamples' ]
  myIntegral = simpson( integrandForce( rSamples, z, direction, data ), rSamples )
  if data[ 'verbose' ]: #Additional lines only for debug mode 
    sol = quad( integrandForce, 0, np.inf, args = ( z, direction, data ) )[ 0 ]
    diff = abs( sol - myIntegral )
    print( f'Compare For integralForce. Numerical {myIntegral}  semi-analitical {sol}, difference {diff}' )
    #quit()
  return myIntegral 

#def integralOmega( z, data ):
#  '''Total number of bonds formed, necessary to calculate the average force per bond.
#  Eq.8 in the paper''' 
#  rSamples = data[ 'rSamples' ]
#  myIntegral = simpson( integrandOmega( rSamples, z, data ), rSamples )
#  if data[ 'verbose' ]: #Additional lines only for debug mode 
#    sol = quad( integrandOmega, 0, np.inf, args = ( z, data ) )[ 0 ]
#    diff = abs( sol - myIntegral )
#    print( f'Compare For integralOmega. Numerical {myIntegral}  semi-analitical {sol}, difference {diff}' )
#    #quit()
#  return myIntegral 

def integralPR( z, pLEq, data ):
  '''Integral to calculate the part of the free-energy dependent on receptors'''
  rSamples = data[ 'rSamples' ]
  myIntegral = simpson( integrandPR( rSamples, z, pLEq, data ), rSamples )
  if data[ 'verbose' ]: #Additional lines only for debug mode 
    sol = quad( integrandPR, 0, np.inf, args = ( z, pLEq, data ) )[ 0 ]
    diff = abs( sol - myIntegral )
    print( f'Compare For integralOmega. Numerical {myIntegral}  semi-analitical {sol}, difference {diff}' )
  return myIntegral 

#def forceR( z, data ):
#  '''Average force in the radial direction, Eq.6'''
#  sol = integralOmega( z, data )**(-1) * integralForce( z, direction = 'r', data = data )
#  return sol

#def forceZ( z, data ):
#  '''Average force in the perpendicular direction, Eq.7'''
#  sol = integralOmega( z, data )**(-1) * integralForce( z, direction = 'z', data = data )
#  return sol

def ABonds( z, data ):
  kbT = data[ 'kbT' ]
  #print( f"z here before: {z}" )
  pLequi = pLEq( z, data )
  NL = data[ 'NL' ]
  #print( f"pL at this distance: {pLequi}" )
  part1 = kbT * NL * ( np.log( pLequi ) + 0.5 * ( 1 - pLequi ) )
  part2 = integralPR( z, pLequi, data )


  qBound = np.exp( -( part1 + part2 ) / kbT ) - 1.0 #In practice, we count as bound only particles with at least a bond on the surface
  if qBound > 0:
    bound = -kbT * np.log( qBound )
  else:
    bound = +np.inf
  
  #print( f"z {z} , Bond energy part1 {part1}, part2 {part2}, Effective bound energy {bound}" )
  
  return bound

def RepIntegral( z, data ):
  '''This is the ratio of the partition function of a Gaussian chain with a confining surface at -z with respect to a pure
  Gaussian spring'''
  kEff = data['kEff']
  kEff = data['kEffRep']
  x0 = data['x0']
  assert x0 == 0, AssertionError( "Theory as written only works for Gaussian chains of zero mean" )
  integral = 1.0 / 2.0 * ( 1.0 + scipy.special.erf( np.sqrt( kEff ) * z ) ) 
 
  return integral 

def ARep( z, data ):
  kbT = data[ 'kbT' ]
  Nrep = data[ 'Nrep' ]
  steric = -kbT * Nrep * np.log( RepIntegral( z, data ) ) 
  return steric 

def integrandPR( r, z, pLEq, data ):
  '''Integrand of the function to calculate receptor contribution to bond energy'''
  sigma = data[ 'sigma' ]
  PR = pREq( r, z, pLEq, data )
  result = 2 * np.pi * sigma * r * ( np.log( PR ) + 0.5 * ( 1 - PR ) )

  return result

def ATot( z, data ):
  return ABonds( z, data ) + ARep( z, data )
  
def Omega( data ):
  '''Calculate Omega, the bound partition function, Eq. 10'''
  try:
    result = data[ 'boundPartition' ]
  except KeyError:
    zSamples = data[ 'zSamples' ]
    myA = np.zeros( len( zSamples ) )
    for i in range( len( zSamples ) ):
      myA[ i ] = np.exp( -ATot( zSamples[ i ], data ) ) 
    result = simpson( myA, zSamples )
    #print( f"Omega: {result}" )
    data[ 'boundPartition' ] = result

  return result 

