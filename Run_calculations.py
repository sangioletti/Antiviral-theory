#Python program implementing the theory for the future JCP paper
import sys
sys.path.append("/Users/sangiole/Dropbox/Papers_data_live/Antiviral-theory")
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from Theory_v7 import *

data = { 'kEff': 1.0,
	 'x0' : 0,
         'sigma': 0.01,
         'NL' : 5,
         'N_inert' : 0,
         'kbT' : 1.0,  
         'DG0' : -5.0,
         'L0' : 1.0,
         'maxDG' : 20,
         'epsilon_self' : 10**(-7),
         'nIntSamples' : 200, #Nr of points to sample numerical integral
         'countBeta' : 1000,
         'rhoNP' : 0.001,
         'verbose' : False, 
         'cV0' :  10**(-9),  #This is the molar concentration of viruses
         'cNP0' : 10**(-6),  #This is the molar concentration of nanoparticles 
         'rV' : 100,#This is the radius of a single virus 
         'rNP': 10 #This is the radius of a Nanoparticle
	}

mySigma = [ 0.01 ]
myDG = [-10]
myNrep = range( 1,14,2) 
mykEff = [ 0.5, 1.0, 2.0, 4.0, 8.0 ] 
rNP = [ 10 ] 
allData = []

for sigma in mySigma:
  for kEff in mykEff:
    for Nrep in myNrep:
      force = []
      for DG in myDG:

        data[ 'sigma' ] = sigma 
        data[ 'DG0' ] = DG 
        data[ 'Nrep' ] = Nrep
        data[ 'kEff' ] = kEff
        NL = data[ 'NL' ]
        rLimit( data )
        print( f"Calculation for DG = {DG}; NL = {NL}; kEff = {kEff}" )
      
        force.append( 
                    ( 
                      DG, 
                      averageForce( direction = 'r', data = data ), 
                      averageForce( direction = 'z', data = data ) 
                    ) 
                   )

        data.pop( 'boundPartition' )
        data.pop( 'PBoundZ' )
 
      force = np.array( force )
      fileName = f"RESULTS_Nrep={Nrep}_kEFF={kEff}_SIGMA{sigma}"

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
