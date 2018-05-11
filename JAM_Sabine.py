'''
#############################################################################
Aiming to create a code that is completely flexible. 

For this to be a completely flexible code, there are a few different configurations that need to be possible. 
________________
Configuration 1:
----------------
  potential geometry: gNFW dark matter halo + stellar light distribution
  free parameters: inner halo slope (gamma), inclination, beta, central halo density
    NB: The gamma calculated by this method is not the gamma_tot predicted by the simulations such as those of
        Remus et al. I will need to combine the stellar and DM components to actually calculate this value here.      

________________
Configuration 2:
----------------
  potential geometry: power law
  free parameters: total mass slope (gamma), inclination, beta, rho_s

For now, I will set the break radius to be 20 kpc for each of these configurations, 
which means that the number of free parameters will always be 4. 

These configurations are similar to models d) and e) from Mitzkus, Cappellari & Walcher (2016) arXiv:1610.04516

#############################################################################
REQUIRED ROUTINES:
  - JAM_Sabine_def.py
  - jam_axi_rms.py
  - JAM_Dictionary.py
  - galaxyParametersDictionary_v9.py

#############################################################################
author: Sabine Bellstedt <sbellstedt@swin.edu.au>
#############################################################################
'''
import numpy as np
import os, sys, pickle, time

DropboxDirectory = os.getcwd().split('Dropbox')[0]
lib_path = os.path.abspath(DropboxDirectory+'Dropbox/PhD_Analysis/Library') 
sys.path.append(lib_path)
from JAM_Sabine_def import *
from JAM_Dictionary import *
from galaxyParametersDictionary_v9 import *

GalName = raw_input('Galaxy Name: ')
Input_path = 'InputFiles/Original'
JAM_path = DropboxDirectory+'Dropbox/PhD_Analysis/Analysis/JAM_Modular'

Model = 'gNFW' # either gNFW or powerLaw
SLUGGS = True
ATLAS = True
GC = False

Inclination = True
Beta = True
Gamma = True
ScaleDensity = True
ML = False
ScaleRadius = False

SluggsWeight = True
AtlasWeight = True
GCWeight = False

FisedStellarMass = False

OutputFilename, ParamSymbol, ParameterNames, Data  = mainCall_modular(GalName, Input_path, JAM_path, Model, SLUGGS, ATLAS, GC, \
  Inclination, Beta, Gamma, ScaleDensity, ML, ScaleRadius, SluggsWeight, AtlasWeight, GCWeight, FixedStellarMass,\
  nwalkers = 50, burnSteps = 10, stepNumber = 2000)


# now to directly analyse the output

from chainconsumer import ChainConsumer

fileIn = open(OutputFilename, 'rb')
chain, flatchain, lnprobability, flatlnprobability, = pickle.load(fileIn) 
fileIn.close()

triangleFilename = OutputFilename.split('MCMCOutput_')[0]+OutputFilename.split('MCMCOutput_')[1].split('.')[0]+'_triangle.pdf'
      
c = ChainConsumer().add_chain(flatchain, parameters=ParamSymbol)
c.configure(statistics='max_shortest', summary=True)
fig = c.plotter.plot(figsize = 'PAGE', filename = triangleFilename)

FinalValues, Lower, Upper = [], [], []
for parameter in ParamSymbol:
  value, lower, upper = parameterExtractor(c.analysis.get_summary(), parameter)
  FinalValues.append(value)
  Lower.append(lower)
  Upper.append(upper)

X = np.zeros(np.array(ParameterNames).size, dtype=[('params', 'U22'), ('values', float), ('lower', float), ('upper', float)])
X['params'] = np.array(ParameterNames)
X['values'] = np.array(FinalValues)
X['lower']  = np.array(Lower)
X['upper']  = np.array(Upper)

np.savetxt(OutputFilename.split('MCMCOutput_')[0]+OutputFilename.split('MCMCOutput_')[1].split('.')[0]+'_parameters.txt', \
  X, fmt="%22s %10.3f %10.3f %10.3f", header = 'Parameter, Value, LowerErr, UpperErr')
