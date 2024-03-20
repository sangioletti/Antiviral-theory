#Python program implementing the theory for the future JCP paper
import sys
sys.path.append("/Users/sangiole/Dropbox/Papers_data_live/Antiviral-theory")
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from Theory_v7 import *

data = { 'kEff': 1.0,
         'kEffRep': 3.0,
	 'x0' : 0,
         'sigma': 0.01,
         'NL' : 5,
         'Nrep' : 10,
         'kbT' : 1.0,  
         'DG0' : -5.0,
         'L0' : 1.0,
         'maxDG' : 5,
         'epsilon_self' : 10**(-7),
         'nIntSamples' : 100, #200, #Nr of points to sample numerical integral
         'countBeta' : 1000,
         'rhoNP' : 0.001,
         'verbose' : False, 
         'cV0' :  10**(-9),  #This is the molar concentration of viruses
         'cNP0' : 10**(-6),  #This is the molar concentration of nanoparticles 
         'rV' : 100,#This is the radius of a single virus 
         'rNP': 10 #This is the radius of a Nanoparticle
	}

mySigma = np.logspace( -2, 0, 10 )
myDG =  range( -20, 5, 1 )
myNrep = [ 0, 3, 6, 9 ] #range( 0, 10, 2 ) 
mykEff = [ 0.025, 0.05, 0.1, 0.2 ] 
mykEffRep = [ 0.001, 0.01, 0.1 ] 
allData = []
convFact = 2.6 #Convert kbT/nm into pN assuming room temperature (300K) 


for sigma in mySigma:
  for kEff in mykEff:
    for kEffRep in mykEffRep:
      for Nrep in myNrep:
        force = []
        print( f"Calculation for sigma = {sigma}, kEff = {kEff}, kEffRep = {kEffRep}, NRep = {Nrep}" )
        for DG in myDG:
  
          rNP = np.sqrt( 3 * data[ 'kbT' ] / kEff ) 
          data[ 'sigma' ] = sigma 
          data[ 'DG0' ] = DG 
          data[ 'Nrep' ] = Nrep
          data[ 'kEff' ] = kEff
          data[ 'kEffRep' ] = kEffRep
          NL = data[ 'NL' ]
          rLimit( data )
          print( f"DG = {DG}" )
        
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
        fileName = f"RESULTS_sigma_{sigma}_kEff_{kEff}_kEffRep_{kEffRep}_NRep_{Nrep}" 
  
        with open( fileName, "w" ) as myF:
          myF.write( "DG( kbT ) Fr ( kbT / nm ) Fz (kbT / nm ) Fr ( pN ) Fz ( pN ) \n" )
          for dg, fr, fz in force:
            myF.write( f"{dg} {fr} {fz} {fr*convFact} {fz*convFact} \n" )
  
        figure = plt.plot( myDG, force[ :, 1], "r-", linewidth =2, label = 'Fr' )
        figure = plt.plot( myDG, force[ :, 2], "b-", linewidth =2, label = 'Fz' )
        plt.xlabel( "DG (kbT)" ) 
        plt.ylabel( "Force (kbT)" ) 
        plt.legend()
        plt.savefig( fileName + ".eps" )
        plt.close()
