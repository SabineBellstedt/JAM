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
commits of this code:
2016 Oct 27 - Initial version of this code, which implemented a single gNFW mass distribution
              with a uniform beta value throughout the value. R_s and rho_s values fixed.
              Scale here is still in arcseconds. 
2016 Oct 28 - Only have four free values now: inclincation, beta, density and gamma. 
              Break radius is fixed to 20 kpc. 
              All units are now in kpc. 
2016 Nov 07 - Code has been modified to include both configurations, one with stellar + gNFW halo, 
              the other with total mass being approximated by a power law. All units are in arcseconds, 
              to be converted into pc by the JAM code. MGE is in the original units
2016 Nov 15 - This version includes functioning implementations of both gNFW and power law mass distributions.
2017 Apr 21 - Option three has now been included, in which the two datasets are treated separately by JAM, 
              and the two weights for each data set form two extra free parameters. This version has 6 free 
              parameters. 
#############################################################################
'''
import numpy as np
import os, sys, pickle, time

lib_path = os.path.abspath('/Users/sbellstedt/Dropbox/PhD_Analysis/Library') #edit here
sys.path.append(lib_path)
from JAM_Sabine_def import *
from JAM_Dictionary import *
from galaxyParametersDictionary_v9 import *

GalName = raw_input('Galaxy Name: ')
MGE = MGE_Source[GalName]

#############################################################################
'''
Loading the MGE parameters for each galaxy. 
MGE source is determined in the JAM_Dictionary.py file, under the MGE_Source array
'''
surf_lum, sigma_lum, qobs_lum = MGEfinder(MGE, GalName, units = 'original') # specify whether the units of the MGE are in arcsec,
                                                                            # kpc, or in the original units.

#############################################################################  

sigmapsf = 0.6
pixsize = 0.8
filename='../Output/'+str(GalName)+'/JAM_'+str(GalName)
try:
  mbh = BlackHoleMass[GalName]
except:
  mbh = 0
  print 'No black hole mass specified for', GalName
distance = distMpc[GalName]
reff = Reff_Spitzer[GalName]
R_S = 20000 * 206265 / (distance * 1e6) # calculating a break radius of 20 kpc in terms of arcseconds. 
print 'Break radius (arcseconds):', R_S

config = 3 # select the kind of JAM configuration desired. Options are described at the top of the document. 

t0 = time.time()

if config == 1:
  '''
  Calling in the input file as determined by JAM_InputGenerator.py in the ../InputFiles folder. 
  This method of calling in the files doesn't affect the cleaning of the data at all. 
  All the cleaning is done in the previously mentioned python code. 
  '''
  input_filename = '../InputFiles/'+str(GalName)+'/JAM_Input_'+GalName+'.txt'
  xbin, ybin, rms, erms = np.loadtxt(input_filename, unpack = True, usecols = [0, 1, 2, 3], comments = '#')
  # setting up the MCMC so that rather than prodcing initial guesses, the walkers just sample from the given range. 
  print '4 parameters: Inclination, beta, density and gamma'
  ndim, nwalkers = 4, 100 # NUMBER OF WALKERS
  
  # setting up the walkers. 
  incTmp_lower, incTmp_upper = 40, 90
  betaTmp_lower, betaTmp_upper = -0.5, 0.5
  gamma_lower, gamma_upper = -3.5, 0
  log_rhoSTmp_lower, log_rhoSTmp_upper = np.log10(0.001), np.log10(1000)
  
  # setting up the initial positions of the walkers. 
  pos_gNFW = []
  for ii in np.arange(nwalkers):  
    incTmp = np.random.uniform(low=incTmp_lower, high=incTmp_upper)
    betaTmp = np.random.uniform(low=betaTmp_lower, high=betaTmp_upper)
    gamma = (np.random.uniform(low=gamma_lower, high=gamma_upper))  
    log_rhoSTmp = (np.random.uniform(low=log_rhoSTmp_lower, high=log_rhoSTmp_upper))  
    pos_gNFW.append([incTmp, betaTmp, log_rhoSTmp, gamma])
  
  boundaries = [incTmp_lower, incTmp_upper,  # with one fixed dimension
        betaTmp_lower, betaTmp_upper, log_rhoSTmp_lower, log_rhoSTmp_upper, gamma_lower, gamma_upper]
  
  # Setup MCMC sampler
  import emcee
  sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_gNFW,
                    args=(boundaries, xbin, ybin, rms, erms, mbh, distance,
                         reff, filename, 
                          surf_lum, sigma_lum, qobs_lum, R_S),
                    threads=8) #Threads gives the number of processors to use
  
  ############ implementing a burn-in period ###########
  burnSteps = 100
  pos, prob, state = sampler.run_mcmc(pos_gNFW, burnSteps)
  sampler.reset()
  
  stepNumber = 500
  outputMCMC = sampler.run_mcmc(pos, stepNumber) # uses the final position of the burn-in period as the starting point. 
  ######################################################
  
  
  t = time.time() - t0
  Date=time.strftime('%Y-%m-%d')
  Time = time.asctime().split()[3]
  print '########################################'
  print 'time elapsed:', float(t)/3600, 'hours'
  print '########################################'
  print 'Mean acceptance fraction: ', (np.mean(sampler.acceptance_fraction))
  print 'output filename: ', str(GalName)+'/JAM_'+str(GalName)+'_MCMC_gNFW_'+Date+':'+Time+'.dat'

  inputfile = open(input_filename, 'r')
  lines = inputfile.readlines()
  file.close()  
  
  fileOut = open('../Output/'+str(GalName)+'/JAM_'+str(GalName)+'_MCMC_gNFW_'+Date+':'+Time+'.dat', 'wb')
  pickle.dump([sampler.chain, sampler.flatchain, sampler.lnprobability, sampler.flatlnprobability], fileOut)
  fileOut.close()
  
  file=open('../Output/'+str(GalName)+'/JAM_'+str(GalName)+'_Notes_gNFW_'+Date+':'+Time+'.txt', 'w')
  file.write('Total mass distribution is in the form of stellar + gNFW halo. \n')
  file.write('Scale Radius has been fixed in this:\n')
  file.write('Number of walkers: '+str(nwalkers)+'\n')
  file.write('Number of steps: '+str(stepNumber)+'\n')
  file.write('Number of steps for burn-in period: '+str(burnSteps)+'\n')
  file.write('------------------------------------------------------------------\n')
  file.write('Initial walker range limits: \n')
  file.write('Inclination: '+str(incTmp_lower)+', '+str(incTmp_upper)+'\n')
  file.write('Beta: '+str(betaTmp_lower)+', '+str(betaTmp_upper)+'\n')
  file.write('log_rhoS: '+str(log_rhoSTmp_lower)+', '+str(log_rhoSTmp_upper)+'\n')
  file.write('log_R_S is fixed to 20 kpc. In arcseconds: '+str(R_S)+'\n')
  file.write('Gamma: '+str(gamma_lower)+', '+str(gamma_upper)+'\n')
  file.write('------------------------------------------------------------------\n')

  for line in lines:
    if line[0] == '#':
      file.write(line)

  file.close()


elif config == 2:
  '''
  Calling in the input file as determined by JAM_InputGenerator.py in the ../InputFiles folder. 
  This method of calling in the files doesn't affect the cleaning of the data at all. 
  All the cleaning is done in the previously mentioned python code. 
  '''
  input_filename = '../InputFiles/'+str(GalName)+'/JAM_Input_'+GalName+'.txt'
  xbin, ybin, rms, erms = np.loadtxt(input_filename, unpack = True, usecols = [0, 1, 2, 3], comments = '#')
  # setting up the MCMC so that rather than prodcing initial guesses, the walkers just sample from the given range. 
  print '4 parameters: Inclination, beta, density and total gamma'
  ndim, nwalkers = 4, 100 # NUMBER OF WALKERS

  # Gaussian = 0
  # Conversion = 206265 / (distMpc[GalName]*10**3)
  # for ii in range(len(surf_lum)):
  #   Gaussian += surf_lum[ii]*np.exp((-(R_S/20)**2)/(2* Conversion * sigma_lum[ii]**2)) # calculating the projected stellar density at 1 kpc. 
  
  # setting up the walkers.
  # min_inclination = degree(np.arccos(b_a[GalName]))
  min_inclination = degree(np.arccos(np.min(qobs_lum)))
  print 'minimum inclination:', min_inclination

  incTmp_lower, incTmp_upper = min_inclination, 90 # calculate minimum possible inclination
  betaTmp_lower, betaTmp_upper = 0.0, 0.5
  gamma_lower, gamma_upper = -2.4, -1.5
  log_rhoSTmp_lower, log_rhoSTmp_upper = np.log10(0.1), np.log10(100) # in this case, it wouldn't make any sense for the total
                                                                                   # mass density in the centre to be less than the 
                                                                                   # density of stellar mass. This will therefore feed
                                                                                   # back into the information about the priors. 
  
  # setting up the initial positions of the walkers. 
  pos_powerLaw = []
  for ii in np.arange(nwalkers):  
    incTmp = np.random.uniform(low=incTmp_lower, high=incTmp_upper)
    betaTmp = np.random.uniform(low=betaTmp_lower, high=betaTmp_upper)
    gamma = (np.random.uniform(low=gamma_lower, high=gamma_upper))  
    log_rhoSTmp = (np.random.uniform(low=log_rhoSTmp_lower, high=log_rhoSTmp_upper))  
    pos_powerLaw.append([incTmp, betaTmp, log_rhoSTmp, gamma])
  
  boundaries = [incTmp_lower, incTmp_upper,  # with one fixed dimension
        betaTmp_lower, betaTmp_upper, log_rhoSTmp_lower, log_rhoSTmp_upper, gamma_lower, gamma_upper]
  
  # Setup MCMC sampler
  import emcee
  sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_powerLaw,
                    args=(boundaries, xbin, ybin, rms, erms, mbh, distance,
                         reff, filename, 
                          surf_lum, sigma_lum, qobs_lum, R_S),
                    threads=8) #Threads gives the number of processors to use
  
  ############ implementing a burn-in period ###########
  burnSteps = 50
  pos, prob, state = sampler.run_mcmc(pos_powerLaw, burnSteps)
  sampler.reset()
  
  stepNumber = 500
  outputMCMC = sampler.run_mcmc(pos, stepNumber) # uses the final position of the burn-in period as the starting point. 
  ######################################################
  
  
  t = time.time() - t0
  Date=time.strftime('%Y-%m-%d')
  Time = time.asctime().split()[3]
  print '########################################'
  print 'time elapsed:', float(t)/3600, 'hours'
  print '########################################'
  print 'Mean acceptance fraction: ', (np.mean(sampler.acceptance_fraction))
  print 'output filename: ', str(GalName)+'/JAM_'+str(GalName)+'_MCMC_PowerLaw_'+Date+':'+Time+'.dat'

  if not os.path.exists('../Output/'+str(GalName)): # make a directory in which to store the output file for each galaxy
    os.mkdir('../Output/'+str(GalName))

  file = open(input_filename, 'r')
  lines = file.readlines()
  file.close()  

  fileOut = open('../Output/'+str(GalName)+'/JAM_'+str(GalName)+'_MCMC_PowerLaw_'+Date+':'+Time+'.dat', 'wb')
  pickle.dump([sampler.chain, sampler.flatchain, sampler.lnprobability, sampler.flatlnprobability], fileOut)
  fileOut.close()
  
  file=open('../Output/'+str(GalName)+'/JAM_'+str(GalName)+'_Notes_PowerLaw_'+Date+':'+Time+'.txt', 'w')
  file.write('Total mass distribution in the form of a broken power law. \n')
  file.write('Scale Radius has been fixed in this:\n')
  file.write('Number of walkers: '+str(nwalkers)+'\n')
  file.write('Number of steps: '+str(stepNumber)+'\n')
  file.write('Number of steps for burn-in period: '+str(burnSteps)+'\n')
  file.write('------------------------------------------------------------------\n')
  file.write('Initial walker range limits: \n')
  file.write('Inclination: '+str(incTmp_lower)+', '+str(incTmp_upper)+'\n')
  file.write('Beta: '+str(betaTmp_lower)+', '+str(betaTmp_upper)+'\n')
  file.write('log_rhoS: '+str(log_rhoSTmp_lower)+', '+str(log_rhoSTmp_upper)+'\n')
  file.write('log_R_S is fixed to 20 kpc. In arcseconds: '+str(R_S)+'\n')
  file.write('Gamma: '+str(gamma_lower)+', '+str(gamma_upper)+'\n')
  file.write('------------------------------------------------------------------\n')

  for line in lines:
    if line[0] == '#':
      file.write(line)

  file.close()


# The third option has been designed in order to run the emcee process with SLUGGS and ATLAS3D data separately, 
# then using two free parameters by which to weight the likelihoods calculated for each data set. 
# This is essentially a test, to see whether this more rigorous approach to combining datasets is better, 
# or produces better results. 

elif config == 3:
  input_filename_sluggs = '../InputFiles/'+str(GalName)+'/JAM_Input_'+GalName+'_SLUGGS.txt'
  input_filename_atlas = '../InputFiles/'+str(GalName)+'/JAM_Input_'+GalName+'_ATLAS3D.txt'
  xbin_sluggs, ybin_sluggs, rms_sluggs, erms_sluggs = np.loadtxt(input_filename_sluggs, unpack = True, usecols = [0, 1, 2, 3], comments = '#')
  xbin_atlas, ybin_atlas, rms_atlas, erms_atlas = np.loadtxt(input_filename_atlas, unpack = True, usecols = [0, 1, 2, 3], comments = '#')
  # setting up the MCMC so that rather than prodcing initial guesses, the walkers just sample from the given range. 
  print '6 parameters: Inclination, beta, density, total gamma and the data set weight.'
  ndim, nwalkers = 6, 100 # NUMBER OF WALKERS
 
  # setting up the walkers.
  min_inclination = degree(np.arccos(np.min(qobs_lum))) # using the MGE to calculate the minimum possible inclination 
  print 'minimum inclination:', min_inclination
 
  incTmp_lower, incTmp_upper = min_inclination, 90 # calculate minimum possible inclination
  betaTmp_lower, betaTmp_upper = -0.2, 0.5
  gamma_lower, gamma_upper = -2.4, -1.5
  log_rhoSTmp_lower, log_rhoSTmp_upper = np.log10(010), np.log10(100000000) 
  sluggsWeight_lower, sluggsWeight_upper = 0, 10
  atlasWeight_lower, atlasWeight_upper = 0, 10
  sluggsWeightPrior_lower, sluggsWeightPrior_upper = np.exp(-sluggsWeight_upper), np.exp(sluggsWeight_lower)
  atlasWeightPrior_lower, atlasWeightPrior_upper = np.exp(-atlasWeight_upper), np.exp(atlasWeight_lower)
  
  # setting up the initial positions of the walkers. 
  pos_powerLaw = []
  for ii in np.arange(nwalkers):  
    incTmp = np.random.uniform(low=incTmp_lower, high=incTmp_upper) # uniform sampling in linear space
    betaTmp = np.random.uniform(low=betaTmp_lower, high=betaTmp_upper) # uniform sampling in linear space
    gamma = (np.random.uniform(low=gamma_lower, high=gamma_upper))   # uniform sampling in linear space
    log_rhoSTmp = (np.random.uniform(low=log_rhoSTmp_lower, high=log_rhoSTmp_upper))  # uniform sampling in log space
    sluggsWeightPrior = (np.random.uniform(low=sluggsWeightPrior_lower, high=sluggsWeightPrior_upper))   # uniform sampling in expoential space
    atlasWeightPrior = (np.random.uniform(low=atlasWeightPrior_lower, high=atlasWeightPrior_upper)) # uniform sampling in exponential space  
    sluggsWeight = -np.log(sluggsWeightPrior)
    atlasWeight = -np.log(atlasWeightPrior)
    pos_powerLaw.append([incTmp, betaTmp, log_rhoSTmp, gamma, sluggsWeight, atlasWeight])
   
  boundaries = [incTmp_lower, incTmp_upper,  # with one fixed dimension
        betaTmp_lower, betaTmp_upper, log_rhoSTmp_lower, log_rhoSTmp_upper, gamma_lower, gamma_upper, 
        sluggsWeight_lower, sluggsWeight_upper, atlasWeight_lower, atlasWeight_upper]
   
  # Setup MCMC sampler
  import emcee
  sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_powerLaw_separateDatasets,
                    args=(boundaries, xbin_sluggs, ybin_sluggs, rms_sluggs, erms_sluggs, 
                      xbin_atlas, ybin_atlas, rms_atlas, erms_atlas, mbh, distance,
                         reff, filename, 
                          surf_lum, sigma_lum, qobs_lum, R_S),
                    threads=8) #Threads gives the number of processors to use
  
  ############ implementing a burn-in period ###########
  burnSteps = 50
  pos, prob, state = sampler.run_mcmc(pos_powerLaw, burnSteps)
  sampler.reset()
  
  stepNumber = 500
  outputMCMC = sampler.run_mcmc(pos, stepNumber) # uses the final position of the burn-in period as the starting point. 
  ######################################################
  
  
  t = time.time() - t0
  Date=time.strftime('%Y-%m-%d')
  Time = time.asctime().split()[3]
  print '########################################'
  print 'time elapsed:', float(t)/3600, 'hours'
  print '########################################'
  print 'Mean acceptance fraction: ', (np.mean(sampler.acceptance_fraction))
  print 'output filename: ', str(GalName)+'/JAM_'+str(GalName)+'_MCMC_PowerLaw_'+Date+':'+Time+'.dat'

  if not os.path.exists('../Output/'+str(GalName)): # make a directory in which to store the output file for each galaxy
    os.mkdir('../Output/'+str(GalName))

  # file = open(input_filename, 'r')
  # lines = file.readlines()
  # file.close()  

  fileOut = open('../Output/'+str(GalName)+'/JAM_'+str(GalName)+'_MCMC_PowerLaw_'+Date+':'+Time+'.dat', 'wb')
  pickle.dump([sampler.chain, sampler.flatchain, sampler.lnprobability, sampler.flatlnprobability], fileOut)
  fileOut.close()
  
  file=open('../Output/'+str(GalName)+'/JAM_'+str(GalName)+'_Notes_PowerLaw_'+Date+':'+Time+'.txt', 'w')
  file.write('Total mass distribution in the form of a broken power law. \n')
  file.write('SLUGGS and ATLAS3D datasets have been processed separately. \n')
  file.write('Scale Radius has been fixed in this:\n')
  file.write('Number of walkers: '+str(nwalkers)+'\n')
  file.write('Number of steps: '+str(stepNumber)+'\n')
  file.write('Number of steps for burn-in period: '+str(burnSteps)+'\n')
  file.write('------------------------------------------------------------------\n')
  file.write('Initial walker range limits: \n')
  file.write('Inclination: '+str(incTmp_lower)+', '+str(incTmp_upper)+'\n')
  file.write('Beta: '+str(betaTmp_lower)+', '+str(betaTmp_upper)+'\n')
  # file.write('log_rhoS: '+str(log_rhoSTmp_lower)+', '+str(log_rhoSTmp_upper)+'\n')
  file.write('log_rhoS is fixed to '+str(logRho)+'\n')
  file.write('log_R_S is fixed to 20 kpc. In arcseconds: '+str(R_S)+'\n')
  file.write('Gamma: '+str(gamma_lower)+', '+str(gamma_upper)+'\n')
  file.write('SLUGGS Weight: '+str(sluggsWeight_lower)+', '+str(sluggsWeight_upper)+'\n')
  file.write('ATLAS3D Weight: '+str(atlasWeight_lower)+', '+str(atlasWeight_upper)+'\n')
  file.write('------------------------------------------------------------------\n')

  # for line in lines:
  #   if line[0] == '#':
  #     file.write(line)

  file.close()
