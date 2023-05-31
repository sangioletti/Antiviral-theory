#Python program implementing the theory for the future JCP paper

import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

data = { 'kEff': 4.0,
	 'x0' : 0,
         'sigma': 1.0,
         'NL' : 8.0,
         'kbT' : 1.0,  
         'DG0' : -5.0,
         'maxDG' : 50,
         'epsilon_self' : 10**(-7),
         'nIntSamples' : 10000, #Nr of points to sample numerical integral
         'countBeta' : 1000,
         'rhoNP' : 0.001,
         'verbose' : False 
	}

def rLimit( data ):
  '''Calculates the limit of integration for r(z). In
  practice, integrates only until the value of the bond energy is 
  +maxDG. Note that due to the symmetric role played we use the same 
  for zSamples'''
  try:
    verbose = data[ 'verbose' ]
  except KeyError:
    verbose = False
  x0 = data[ 'x0' ]
  maxDG = data[ 'maxDG' ]
  DG0 = data[ 'DG0' ]
  kEff = data[ 'kEff' ]
  rMax = np.sqrt( 2.0 * ( maxDG - DG0 ) / kEff ) + x0
  if verbose:
    print( f'rLimit is {rMax}' )
  DGeff = -np.log( chi( rMax, z = 0, data = data ) ) 
  if verbose:
    print( f'DG at rLimit is {DGeff}' )
  data[ 'rMax' ] = rMax
  tot = data[ 'nIntSamples' ]
  dr = rMax / tot
  data[ 'dr' ] = dr
  data[ 'rSamples' ] = np.array( range( tot ) * dr )
  data[ 'zSamples' ] = data[ 'rSamples' ][1::500] 
  aa = data[ 'rSamples' ][::10]
  if verbose:
    print( f'rSamples {aa}' )
  return

def chi( r, z, data ):
  '''Chi is nothing but the bond strength, exp( - ( DG + DGcnf )/kbT )'''
  DG0 = data['DG0']
  kEff = data['kEff']
  x0 = data['x0']
  kbT = data['kbT']
  DGcnf = 0.5 * kEff * ( np.sqrt( r**2 + z**2 ) - x0 )**2 
  return np.exp(  -( DG0 + DGcnf ) / kbT )

def singleForce( r, z, data ):
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

  #sample at different values of r
  myIntegral = scipy.integrate.simpson( integrand( rSamples, z, sigma, pL, data ), rSamples )
  
  if data[ 'verbose' ]: #Additional lines only for debug mode 
    myIntegral = np.sum( integrand( rSamples, z, sigma, pL, data ) ) * dr  
    sol = quad( integrand, 0, np.inf, args = ( z, sigma, pL, data ) )[ 0 ]
    diff = abs( sol - myIntegral )
    print( f'Compare. Numerical {myIntegral}  semi-analitical {sol}, difference {diff}' )

  return myIntegral

def pLEq( z, data, epsilon = 10**(-5), epsilon2 = 10**(-3) ):
  '''Self-consistent calculation of pLEq, Eq.4 in the paper'''
  epsilon = data[ 'epsilon_self' ]
  pLInput = min( 1.0, chi( r = 0, z = z, data = data )**(-1) )
  pLOut = max( 1 - integral( pLInput, z, data ), 0.0 )
  count = 0 
  count2 = 3.0 
  betaMix = 0.99
  countBeta = data[ 'countBeta' ]
  while abs( pLOut - pLInput ) > epsilon:
    #print( f"pLinput, pLoutput {pLInput}, {pLOut}" ) 
    pLInput = betaMix * pLInput + ( 1.0 - betaMix ) * pLOut
    pLOut = max( 1 - integral( pLInput, z, data ), 0.0 )
    count += 1
    if ( np.mod( count, countBeta ) == 0 ):
      #print( 'count, pLIn, plOut, betaMix', count, pLInput, pLOut, betaMix )
      betaMix = betaMix + 9 * 10**( -count2 )
      count2 += 1
      countBeta *= 10
      count = 0
      #print( f"New betaMix: {betaMix}" )

  if pLOut == 0.0:
    print( "pL too small?" ) 
     
  return pLOut

#NOTE: this can possibly be improved going to lower values of pL if 
# you are at the boundary, that is, if pL that gives the minimum is at pLInput[ 0 ]
#def pLEqOld( z, data, points = 10**4 ):
#  '''Calculation of pLEq via fixed sampling, Eq.4 in the paper'''
#  dpL = 1 / points
#  pLInput = np.array( range( 1, points + 1 ) ) * dpL
#  diffMin = 1000
#  for pL in pLInput:
#    pLOut = integral( pL, z, data )
#    diff = np.abs( pL - pLOut )
#    #print( "pL, pLOut, diff", pL, pLOut, diff )
#    if diff < diffMin:
#      diffMin = diff
#      pLEq = pLOut
#  print( "pLEq, Error:", pLEq, diffMin )
#     
#  return pLEq

def pREq( r, z, pLEq, data ):
  '''Value of pREq, given once the solution to pLEq exists, Eq. 5 in the paper'''
  NL = data[ 'NL' ]
  myChi = chi( r, z, data )
  
  return 1.0 / ( 1.0 + NL * pLEq * myChi )

def versor( r, z, direction ):
  '''Scaling factor to obtain the radial and z-directed force
  component from the force on a receptor at position (r,z) from the NP'''
  if z == 0:
    angle = np.pi/2
  else:
    angle = np.arctan( r / z )
  if direction == 'r':
    versor = np.sin( angle )
  elif direction == 'z':
    versor = np.cos( angle )
  else:
    print( "Direction is wrong / undeclared" )
    raise ValueError
  return versor

def forceSingle( r, z, data ):
  '''Eq 6 in the paper'''
  myChi = chi( r, z, data )
  
  #You need to multiply the force by the versor otherwise it will look like the two
  #components are always the same which is not
  myVersor = versor( r, z, direction = direction )
  myForce = singleForce( r, z, data ) * myVersor
  pL = pLEq( z, data )
  pR = pREq( r, z, pL, data ) 
  return NL * pR * pL * myChi * myForce 

def integrandForce( z, direction, data ):
  '''integrand of Eq. 7, necessary to calculate the average force exerted on a surface
  from a NP. (r,z) is the position of the receptor with respect to a NP at (0,0).
  Can be thought of as the force multiplied by the number of receptors at a 
  certain positions'''
  sigma = data[ 'sigma' ]
  myChi = chi( r, z, data )
  
  #You need to multiply the force by the versor otherwise it will look like the two
  #components are always the same which is not
  myVersor = versor( r, z, direction = direction )
  myForce = singleForce( r, z, data ) * myVersor
  pL = pLEq( z, data )
  pR = pREq( r, z, pL, data ) 
  return 2 * np.pi * r * sigma * forceBound 


def pCondBound( z, data ):
  '''Eq. 9 in the paper, p( be at z | your are in the bound state)''' 
  Omega = integralAOnly( data )
  unnormalised = A( z, data )
  result = unnormalised / Omega

  return result

def integralForce( z, direction, data ):
  '''Force weighted by the number of bonds at a certain position
  Eq.6/7 in the paper'''
  rSamples = data[ 'rSamples' ]
  myIntegral = scipy.integrate.simpson( integrandForce( rSamples, z, direction, data ), rSamples )
  if data[ 'verbose' ]: #Additional lines only for debug mode 
    sol = quad( integrandForce, 0, np.inf, args = ( z, direction, data ) )[ 0 ]
    diff = abs( sol - myIntegral )
    print( f'Compare For integralForce. Numerical {myIntegral}  semi-analitical {sol}, difference {diff}' )
    #quit()
  return myIntegral 

def integralOmega( z, data ):
  '''Total number of bonds formed, necessary to calculate the average force per bond.
  Eq.8 in the paper''' 
  rSamples = data[ 'rSamples' ]
  myIntegral = scipy.integrate.simpson( integrandOmega( rSamples, z, data ), rSamples )
  if data[ 'verbose' ]: #Additional lines only for debug mode 
    sol = quad( integrandOmega, 0, np.inf, args = ( z, data ) )[ 0 ]
    diff = abs( sol - myIntegral )
    print( f'Compare For integralOmega. Numerical {myIntegral}  semi-analitical {sol}, difference {diff}' )
    #quit()
  return myIntegral 

def integralPR( z, pL, data ):
  '''Integral to calculate the part of the free-energy dependent on receptors'''
  rSamples = data[ 'rSamples' ]
  myIntegral = scipy.integrate.simpson( integrandPR( rSamples, z, pL, data ), rSamples )
  if data[ 'verbose' ]: #Additional lines only for debug mode 
    sol = quad( integrandPR, 0, np.inf, args = ( z, pL, data ) )[ 0 ]
    diff = abs( sol - myIntegral )
    print( f'Compare For integralOmega. Numerical {myIntegral}  semi-analitical {sol}, difference {diff}' )
  return myIntegral 

def forceR( z, data ):
  '''Average force in the radial direction, Eq.6'''
  sol = integralOmega( z, data )**(-1) * integralForce( z, direction = 'r', data = data )
  return sol

def forceZ( z, data ):
  '''Average force in the perpendicular direction, Eq.7'''
  sol = integralOmega( z, data )**(-1) * integralForce( z, direction = 'z', data = data )
  return sol

def ABound( z, data ):
  kbT = data[ 'kbT' ]
  pL = pLEq( z, data )
  NL = data[ 'NL' ]
  #print( f"pL at this distance: {pL}" )
  part1 = kbT * NL * ( np.log( pL ) + 0.5 * ( 1 - pL ) )
  part2 = integralPR( z, pL, data )
  #print( f"Intermediate value of A: {part1+part2}" )
  bound = -kbT * np.log( np.exp( -( part1 + part2 ) / kbT ) - 1.0 ) #In practice, we count as bound only particles with at least a bond on the surface
  return bound

def A( z, data ):
  return ABound( z, data ) 
  
def integralAOnly( data ):
  zSamples = data[ 'zSamples' ]
  myIntegrand = []
  for z in zSamples:
    myA = np.exp( -A( z, data ) ) 
    myIntegrand.append( myA )
    print( f"Value of Qbound at z={z}: {myA}" )
  myIntegrand = np.array( myIntegrand )
  finalIntegral = scipy.integrate.simpson( myIntegrand, zSamples )
  return finalIntegral

def integrandAveForce( z, direction, data ):
  kbT = data[ 'kbT' ]
  rho = data[ 'rhoNP' ]
  if direction == 'r':
    sol = np.exp( -A( z, data ) / kbT ) * forceR( z, data )
  elif direction == 'z':
    sol = np.exp( -A( z, data ) / kbT ) * forceZ( z, data )
  return sol

def aveForce( direction, data ): 
  den =  integralAOnly( data ) 
  print( f"Denominator is {den}" )
  zSamples = data[ 'zSamples' ]
  myIntegrand = []
  for z in zSamples:
    forceAve = integrandAveForce( z, direction, data ) 
    myIntegrand.append( forceAve )
    aa = forceAve / den
    print( f"z {z} force {forceAve} and scaled contribution {aa:.5e}" )
  myIntegrand = np.array( myIntegrand )
  finalIntegral = scipy.integrate.simpson( myIntegrand, zSamples )
  return finalIntegral / den

#rLimit( data )
#print( " First Chi is: {0:.5e}".format( chi( 20, 1, data = data ) ) )
#print( "Integral is:", integral( pL = 0.5, z = 1.0, data = data ) ) 
#print( "pLEq is:", pLEq( z = 1.0, data = data ) ) 
#print( "pREq is:", pREq( r = 1.0, z = 1.0, pLEq = 0.5, data = data ) )
#print( "Integrand Omega:", integrandOmega( r = 1.0, z = 1.0, data = data  ) )
#print( "Integral Omega:", integralOmega( z = 1.0, data = data  ) )
myDG = range(5,-16,-1)
myNL = range(1,20,2)
mykEff = [ 1.0, 3.0, 5.0, 7.0 ] 
allData = []

for kEff in mykEff:
  for NL in myNL:
    force = []
    for DG in myDG:
      data[ 'DG0' ] = DG 
      data[ 'NL' ] = NL
      data[ 'kEff' ] = kEff
      rLimit( data )
      print( f"Calculation for DG = {DG}; NL = {NL}; kEff = {kEff}" )
      force.append( ( DG, aveForce( direction = 'r', data = data ), aveForce( direction = 'z', data = data ) ) )

    force = np.array( force )
    fileName = f"RESULTS_NL={NL}_kEFF={kEff}"

    with open( fileName, "w" ) as myF:
      myF.write( "DG( kbT ) Fr ( kbT / nm ) Fz (kbT / nm ) \n" )
      for dg, fr, fz in force:
        myF.write( f"{dg} {fr} {fz} \n" )

    figure = plt.plot( myDG, force[ :, 1], "r-", linewidth =2, label = 'Fr' )
    figure = plt.plot( myDG, force[ :, 2], "b-", linewidth =2, label = 'Fz' )
    plt.xlabel( "DG (kbT)" ) 
    plt.ylabel( "Force (kbT)" ) 
    plt.legend()
    plt.savefig( fileName + ".eps" )
    plt.close()
