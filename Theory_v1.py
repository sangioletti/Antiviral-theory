#Python program implementing the theory for the future JCP paper

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

data = { 'kEff': 2.0,
	 'x0' : 0,
         'sigma': 1.0,
         'NL' : 10.0,
         'kbT' : 1.0,  
         'DG0' : 1.0
	}

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
  return - kEff * ( np.sqrt( r**2 + z**2 ) - x0 ) 

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
  sol = quad( integrand, 0, np.inf, args = ( z, sigma, pL, data ) )[ 0 ]
  return sol

def pLEq( z, data, epsilon = 10**(-6) ):
  '''Self-consistent calculation of pLEq, Eq.4 in the paper'''
  pLInput = min( 1.0, chi( r = 0, z = z, data = data )**(-1) )
  pLOut = 1 - integral( pLInput, z, data )
  count = 0 
  count2 = 3.0 
  betaMix = 0.99
  while abs( pLOut - pLInput ) > epsilon:
    #print( f"pLinput, pLoutput {pLInput}, {pLOut}" ) 
    pLInput = betaMix * pLInput + ( 1.0 - betaMix ) * pLOut
    pLOut = 1 - integral( pLInput, z, data )
    count += 1
    if ( np.mod( count, 1000 ) == 0 ):
      print( 'count, pLIn, plOut, betaMix', count, pLInput, pLOut, betaMix )
      betaMix = betaMix + 9 * 10**( -count2 )
      count2 += 1
      print( f"New betaMix: {betaMix}" ) 
  #print( f"End of SCF at count {count}, pLEq = {pLOut}" )
     
  return pLOut

def pLEqOld( z, data, points = 10**4 ):
  '''Calculation of pLEq via fixed sampling, Eq.4 in the paper'''
  dpL = 1 / points
  pLInput = np.array( range( 1, points + 1 ) ) * dpL
  diffMin = 1000
  for pL in pLInput:
    pLOut = integral( pL, z, data )
    diff = np.abs( pL - pLOut )
    #print( "pL, pLOut, diff", pL, pLOut, diff )
    if diff < diffMin:
      diffMin = diff
      pLEq = pLOut
  print( "pLEq, Error:", pLEq, diffMin )
     
  return pLEq

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

def integrandForce( r, z, direction, data ):
  '''integrand necessary to calculate the average force exerted on a surface
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
  return -2 * np.pi * r * sigma * pR * pL * myChi * myForce 


def integrandOmega( r, z, data ):
  '''Integrand to calculate Omega, which is nothing but the total number of effective bonds formed
  Eq.8 in the paper''' 
  sigma = data[ 'sigma' ]
  myChi = chi( r, z, data )
  #You need to multiply the force by the versor otherwise it will look like the two
  #components are always the same which is not

  #print( f'r,z {r}, {z}' )
  pL = pLEq( z, data )
  pR = pREq( r, z, pL, data ) 
  return 2 * np.pi * r * sigma * pR * pL * myChi 


def integralForce( z, direction, data ):
  '''Force weighted by the number of bonds at a certain position
  Eq.6/7 in the paper'''
  sol = quad( integrandForce, 0, np.inf, args = ( z, direction, data ) )[ 0 ]
  return sol

def integralOmega( z, data ):
  '''Total number of bonds formed, necessary to calculate the average force per bond.
  Eq.8 in the paper''' 
  sol = quad( integrandOmega, 0, np.inf, args = ( z, data ) )[ 0 ]
  return sol

def forceR( z, data ):
  '''Average force in the radial direction, Eq.6'''
  sol = integralOmega( z, data )**(-1) * integralForce( z, direction = 'r', data = data )
  return sol

def forceZ( z, data ):
  '''Average force in the perpendicular direction, Eq.7'''
  sol = integralOmega( z, data )**(-1) * integralForce( z, direction = 'z', data = data )
  return sol

#def A( z, data ):
#  pL = pLEq( z, data ) 
#  pR = pREq( r, z, pLEq, data )


print( " First Chi is: {0:.5e}".format( chi( 20, 1, data = data ) ) )
print( "Integral is:", integral( pL = 0.5, z = 1.0, data = data ) ) 
print( "pLEq is:", pLEq( z = 1.0, data = data ) ) 
print( "pREq is:", pREq( r = 1.0, z = 1.0, pLEq = 0.5, data = data ) )
print( "Integrand Omega:", integrandOmega( r = 1.0, z = 1.0, data = data  ) )

#integralValues = np.zeros( 200 )
#dr = 0.05
#for i in range( 0, 200 ):
#  #print( f"i, {i}" )
#  integralValues[ i ] = integrandOmega( r = i*dr, z = 1.0, data = data  )
#print( integralValues[::10] )
#
#figure = plt.plot( integralValues ) 
#plt.show()

print( "Integral Omega:", integralOmega( z = 1.0, data = data  ) )
print( "IntegralForce", integralForce( z = 0.0, direction = 'r', data = data ) ) 
