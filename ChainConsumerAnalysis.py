# code to run chainconsumer on a file if it has been terminated before completing the MCMC run
import numpy as np
import os, sys, pickle, time
import matplotlib
matplotlib.use('Agg') # stops the figure from being shown, which is what is needed to run on a supercomputer job. 

DropboxDirectory = os.getcwd().split('Dropbox')[0]
lib_path = os.path.abspath(DropboxDirectory+'Dropbox/PhD_Analysis/Library') 
sys.path.append(lib_path)
from JAM_Sabine_def import *

from chainconsumer import ChainConsumer

def parameterExtractor(inputDict, name):
	'''
	This function extracts the value and associated uncertainties from a specified parameter in a 
	chainconsumer dictionary. 
	'''
	value = inputDict[name][1]
	# try:
	lower = inputDict[name][1] - inputDict[name][0]
	upper = inputDict[name][2] - inputDict[name][1]
	# except:
	#         lower, upper = 0.0, 0.0
	# print 'Parameter extraction successful for', name
	return value, lower, upper

Galaxies = ['NGC821', 'NGC1023', 'NGC2549', 'NGC2699', 'NGC2768', 'NGC2974', 'NGC3377', 'NGC3379', 'NGC4111'\
, 'NGC4278', 'NGC4459', 'NGC4473', 'NGC4474', 'NGC4494', 'NGC4551', 'NGC4526', 'NGC4649', 'NGC4697', 'NGC5866', 'NGC7457']

SLUGGS = True
ATLAS = True
GC = True

Inclination = True
Beta = True
Gamma = True
ScaleDensity = True
ML = True
ScaleRadius = False

SluggsWeight = True
AtlasWeight = True
GCWeight = True

for GalName in Galaxies:
	try:

		ndim = 0
		ParameterNames, ParamSymbol = [], []
		if Inclination: # inclination 
			ndim += 1
			ParameterNames.append('Inclination')
			ParamSymbol.append(r"$i$")
		if Beta: # inclination 
			ndim += 1
			ParameterNames.append('Beta')
			ParamSymbol.append(r"$\beta$")
		if Gamma: # inclination 
			ndim += 1
			ParameterNames.append('Gamma')
			ParamSymbol.append(r"$\gamma$")
		if ScaleDensity: # inclination 
			ndim += 1
			ParameterNames.append('ScaleDensity')
			ParamSymbol.append(r"$\rho_S$")
		if ML: # inclination 
			ndim += 1
			ParameterNames.append('ML')
			ParamSymbol.append(r"$M/L$")
		
		if SluggsWeight:
			ndim += 1
			ParameterNames.append('SluggsWeight')
			ParamSymbol.append(r"$\omega_{\rm SLUGGS}$")
		if AtlasWeight:
			ndim += 1
			ParameterNames.append('AtlasWeight')
			ParamSymbol.append(r"$\omega_{\rm ATLAS}$")
		if GCWeight:
			ndim += 1
			ParameterNames.append('GCWeight')
			ParamSymbol.append(r"$\omega_{\rm GC}$")
		
		suffix = ''
		if SLUGGS:
		  suffix = suffix + 'SLUGGS_'
		if ATLAS:
		  suffix = suffix + 'ATLAS_'
		if GC:
		  suffix = suffix + 'GC_'
		suffix = suffix + 'FreeParam-'+str(ndim)
		
		JAM_path = DropboxDirectory+'Dropbox/PhD_Analysis/Analysis/JAM_Modular'
		
		OutputFilename = JAM_path+'/'+str(GalName)+'/'+suffix+'/'+str(GalName)+'_MCMCOutput_'+suffix+'.dat'
		
		
		fileIn = open(OutputFilename, 'rb')
		chain, flatchain, lnprobability, flatlnprobability, = pickle.load(fileIn) 
		fileIn.close()
		
		triangleFilename = OutputFilename.split('MCMCOutput_')[0]+OutputFilename.split('MCMCOutput_')[1].split('.')[0]+'_triangle.pdf'
		
		print 'number of steps', len(flatchain) / 200
		      
		c = ChainConsumer().add_chain(flatchain, parameters=ParamSymbol)
		c.configure(statistics='max', summary=True)
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
	except:
		print 'no file for ', GalName