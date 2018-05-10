import mge_fit_1d as mge
import numpy as np
import jam_axi_rms as Jrms
import matplotlib.pyplot as plt
import os, sys, time
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from scipy.misc import imread
import matplotlib.cbook as cbook

from galaxyParametersDictionary_v9 import *
from Sabine_Define import *
from JAM_Analysis_def import *

def gNFW(radius, rho_s, R_s, gamma): # From Cappellari+13
  r_tmp = radius/R_s
  return rho_s*(r_tmp**gamma)*(0.5+0.5*r_tmp)**(-gamma-3)

def gNFW_RelocatedDensity(x, radius, rho_x, R_s, gamma): # From Cappellari+13
  return rho_x * (radius/x)**gamma * ((R_s+radius)/(R_s+x))**(-gamma-3)


def powerLaw(radius, rho_s, R_s, gamma, outer): # here we use a broken power law. 
  '''
  I have realised that the initial power law parametrisation wasn't doing what I thoguht it was
  therefore I will change the parametrisation to be the same as the gNFW, in the same manner was
  was done by Poci+17
  '''
  x = R_s/20.
  return rho_s * (radius/x)**gamma * ((R_s+radius)/(R_s+x))**(-gamma+outer)



def CheckBoundaryConditions(Parameters, ParameterBoundaries):
  ParameterNumber = len(Parameters)

  # separate the boundaries array into a lower and an upper array
  # first selecting the lower bounds, which are indices 0, 2, 4...
  Indices_lower = np.arange(0, 2*ParameterNumber, 2)
  # then selecting the lower bounds, which are indices 1, 3, 5...
  Indices_upper = np.arange(1, 2*ParameterNumber, 2)

  ParameterBoundaries_lower = ParameterBoundaries[Indices_lower]
  ParameterBoundaries_upper = ParameterBoundaries[Indices_upper]

  Check = True
  for ii in range(ParameterNumber):
    if not ((Parameters[ii] <= ParameterBoundaries_upper[ii]) & (Parameters[ii] >= ParameterBoundaries_lower[ii])):
      Check = False
  return Check


def lnprior(theta, *args): 
  tmpInputArgs = args[0] 

  if CheckBoundaryConditions(np.array(theta), np.array(tmpInputArgs)):
    return 0.
  else:
    return -np.inf

def lnprob(theta, *args):
  boundaries, ParameterNames, Data, model, mbh, distance, reff, filename, surf_lum, sigma_lum, qobs_lum = args

  lp = lnprior(theta, boundaries)
  if not np.isfinite(lp):
      return -np.inf
  return lp + lnlike(theta, ParameterNames, Data, model, mbh, distance, reff, filename, surf_lum, sigma_lum, qobs_lum) #In logarithmic space, the multiplication becomes sum.

def lnlike(theta, ParameterNames, Data, model, mbh, distance, reff, filename, surf_lum, sigma_lum, qobs_lum):
  ParameterNames = np.array(ParameterNames)

  if 'Inclination' in ParameterNames:
    inc = np.array(theta)[np.where(ParameterNames == 'Inclination')]
  else:
    raise ValueError('Inclination not set as a free parameter. ')

  if 'Beta' in ParameterNames:
    beta = np.array(theta)[np.where(ParameterNames == 'Beta')]
  else:
    raise ValueError('Beta not set as a free parameter. ')

  if 'ScaleDensity' in ParameterNames:
    log_rho_s = np.array(theta)[np.where(ParameterNames == 'ScaleDensity')]
  else:
    raise ValueError('Scale density not set as a free parameter. ')

  if 'Gamma' in ParameterNames:
    gamma = np.array(theta)[np.where(ParameterNames == 'Gamma')]
  else:
    raise ValueError('Gamma not set as a free parameter. ')

  if 'ML' in ParameterNames:
    ml = np.array(theta)[np.where(ParameterNames == 'ML')]
  else:
    ml = 1

  if 'ScaleRadius' in ParameterNames:
    R_S = np.array(theta)[np.where(ParameterNames == 'ScaleRadius')]
  else:
    R_S = 20000

  if len(Data) == 4:
    if 'AtlasWeight' in ParameterNames:
      raise ValueError("ATLAS^3D hyperparameter set, but ATLAS^3D data not provided.")
    else:
      xbin1, ybin1, rms1, erms1 = Data
      DatasetNumber = 1
  elif len(Data) == 8:
    if not 'AtlasWeight' in ParameterNames:
      raise ValueError("ATLAS^3D hyperparameter not not provided.")
    if not 'SluggsWeight' in ParameterNames:
      raise ValueError("SLUGGS hyperparameter not not provided.")
    sluggsWeight = np.array(theta)[np.where(ParameterNames == 'SluggsWeight')]
    atlasWeight = np.array(theta)[np.where(ParameterNames == 'AtlasWeight')]
    xbin1, ybin1, rms1, erms1, xbin2, ybin2, rms2, erms2 = Data
    DatasetNumber = 2
  elif len(Data) == 12:
    if not 'AtlasWeight' in ParameterNames:
      raise ValueError("ATLAS^3D hyperparameter not not provided.")
    if not 'SluggsWeight' in ParameterNames:
      raise ValueError("SLUGGS hyperparameter not not provided.")
    if not 'GCWeight' in ParameterNames:
      raise ValueError("GC hyperparameter not not provided.")
    sluggsWeight = np.array(theta)[np.where(ParameterNames == 'SluggsWeight')]
    atlasWeight = np.array(theta)[np.where(ParameterNames == 'AtlasWeight')]
    gcWeight = np.array(theta)[np.where(ParameterNames == 'GCWeight')]
    xbin1, ybin1, rms1, erms1, xbin2, ybin2, rms2, erms2, xbin3, ybin3, rms3, erms3  = Data
    DatasetNumber = 3


  beta_array = np.ones(len(surf_lum))*beta
  
  n_MGE = 300.                                                                 # logarithmically spaced radii in parsec
  x_MGE = np.logspace(np.log10(1), np.log10(30000), n_MGE)                    # the units of the density are now mass/pc^2

  if model == 'gNFW':
    y_MGE = gNFW_RelocatedDensity(R_S/20, x_MGE, 10.**log_rho_s, R_S, gamma)    
    p = mge.mge_fit_1d(x_MGE, y_MGE, ngauss=10, quiet=True)
    
    surf_dm = p.sol[0]                                # here implementing a potential
    sigma_dm = p.sol[1]  * 206265 / (distance * 1e6)  # configuration which includes the 
    qobs_dm = np.ones(len(surf_dm))                   # stellar mass and also a dark matter    
                                                      # halo in the form of a gNFW distribution. 
  
    surf_pot = np.append(ml * surf_lum, surf_dm)      #  I'm going to need to be careful here to account for the stellar M/L,                                               
    sigma_pot = np.append(sigma_lum, sigma_dm)        #  since the scaling between the two mass distributions needs to be right. 
    qobs_pot = np.append(qobs_lum, qobs_dm)           #    

    if DatasetNumber == 1:
      try:
        rmsModel, ml, chi2, flux = Jrms.jam_axi_rms(
                     surf_lum, sigma_lum, qobs_lum, surf_pot,
                     sigma_pot, qobs_pot,
                     inc, mbh, distance,
                     xbin1, ybin1, filename, 
                     rms=rms1, erms=erms1,
                      plot=False, beta=beta_array, 
                     tensor='zz', quiet=True)
        realChi2 = np.sum(np.log(2*np.pi*erms1**2) + ((rms1 - rmsModel)/erms1)**2.)
      except:
        realChi2 = np.inf 
  
    elif DatasetNumber == 2:
      try:
        rmsModel, ml, chi2, flux = Jrms.jam_axi_rms(
                     surf_lum, sigma_lum, qobs_lum, surf_pot,
                     sigma_pot, qobs_pot,
                     inc, mbh, distance,
                     xbin1, ybin1, filename, 
                     rms=rms1, erms=erms1,
                      plot=False, beta=beta_array, 
                     tensor='zz', quiet=True)
        realChi2_sluggs = np.sum(np.log(2*np.pi*erms1**2/sluggsWeight) + (sluggsWeight*(rms1 - rmsModel)/erms1)**2.)
        
        rmsModel, ml, chi2, flux = Jrms.jam_axi_rms(
                     surf_lum, sigma_lum, qobs_lum, surf_pot,
                     sigma_pot, qobs_pot,
                     inc, mbh, distance,
                     xbin2, ybin2, filename, #ml,
                     rms=rms2, erms=erms2,
                      plot=False, beta=beta_array, 
                     tensor='zz', quiet=True)
        realChi2_atlas = np.sum(np.log(2*np.pi*erms2**2/atlasWeight) + (atlasWeight*(rms2 - rmsModel)/erms2)**2.)
        
        realChi2 = realChi2_sluggs + realChi2_atlas
      except:
        realChi2 = np.inf 

    elif DatasetNumber == 3:
      try:
        rmsModel, ml, chi2, flux = Jrms.jam_axi_rms(
                     surf_lum, sigma_lum, qobs_lum, surf_pot,
                     sigma_pot, qobs_pot,
                     inc, mbh, distance,
                     xbin1, ybin1, filename, 
                     rms=rms1, erms=erms1,
                      plot=False, beta=beta_array, 
                     tensor='zz', quiet=True)
        realChi2_sluggs = np.sum(np.log(2*np.pi*erms1**2/sluggsWeight) + (sluggsWeight*(rms1 - rmsModel)/erms1)**2.)
        
        rmsModel, ml, chi2, flux = Jrms.jam_axi_rms(
                     surf_lum, sigma_lum, qobs_lum, surf_pot,
                     sigma_pot, qobs_pot,
                     inc, mbh, distance,
                     xbin2, ybin2, filename, #ml,
                     rms=rms2, erms=erms2,
                      plot=False, beta=beta_array, 
                     tensor='zz', quiet=True)
        realChi2_atlas = np.sum(np.log(2*np.pi*erms2**2/atlasWeight) + (atlasWeight*(rms2 - rmsModel)/erms2)**2.)

        rmsModel, ml, chi2, flux = Jrms.jam_axi_rms(
                     surf_lum, sigma_lum, qobs_lum, surf_pot,
                     sigma_pot, qobs_pot,
                     inc, mbh, distance,
                     xbin3, ybin3, filename, #ml,
                     rms=rms3, erms=erms3,
                      plot=False, beta=beta_array, 
                     tensor='zz', quiet=True)
        realChi2_gc = np.sum(np.log(2*np.pi*erms3**2/gcWeight) + (gcWeight*(rms3 - rmsModel)/erms3)**2.)
        
        realChi2 = realChi2_sluggs + realChi2_atlas + realChi2_gc
      except:
        realChi2 = np.inf 
  

  elif model == 'powerLaw':
    y_MGE = powerLaw(x_MGE, 10.**log_rho_s, R_S, gamma, -3)    
    p = mge.mge_fit_1d(x_MGE, y_MGE, ngauss=10, quiet=True)
  
    surf_pot = p.sol[0]                               
    sigma_pot = p.sol[1] * 206265 / (distance * 1e6)            # converting the dispersions into arcseconds because it wants only                           
    qobs_pot = np.ones(len(surf_pot))                           # the radial dimension in these units. 

    if DatasetNumber == 1:
      try:
        rmsModel, ml, chi2, flux = Jrms.jam_axi_rms(
                     surf_lum, sigma_lum, qobs_lum, surf_pot,
                     sigma_pot, qobs_pot,
                     inc, mbh, distance,
                     xbin1, ybin1, filename, ml, 
                     rms=rms1, erms=erms1,
                      plot=False, beta=beta_array, 
                     tensor='zz', quiet=True)
        realChi2 = np.sum(np.log(2*np.pi*erms1**2) + ((rms1 - rmsModel)/erms1)**2.)
      except:
        realChi2 = np.inf 
  
    elif DatasetNumber == 2:
      try:
        rmsModel, ml, chi2, flux = Jrms.jam_axi_rms(
                     surf_lum, sigma_lum, qobs_lum, surf_pot,
                     sigma_pot, qobs_pot,
                     inc, mbh, distance,
                     xbin1, ybin1, filename,  ml, 
                     rms=rms1, erms=erms1,
                      plot=False, beta=beta_array, 
                     tensor='zz', quiet=True)
        realChi2_sluggs = np.sum(np.log(2*np.pi*erms1**2/sluggsWeight) + (sluggsWeight*(rms1 - rmsModel)/erms1)**2.)
        
        rmsModel, ml, chi2, flux = Jrms.jam_axi_rms(
                     surf_lum, sigma_lum, qobs_lum, surf_pot,
                     sigma_pot, qobs_pot,
                     inc, mbh, distance,
                     xbin2, ybin2, filename, ml,
                     rms=rms2, erms=erms2,
                      plot=False, beta=beta_array, 
                     tensor='zz', quiet=True)
        realChi2_atlas = np.sum(np.log(2*np.pi*erms2**2/atlasWeight) + (atlasWeight*(rms2 - rmsModel)/erms2)**2.)
        
        realChi2 = realChi2_sluggs + realChi2_atlas
      except:
        realChi2 = np.inf 

    elif DatasetNumber == 3:
      try:
        rmsModel, ml, chi2, flux = Jrms.jam_axi_rms(
                     surf_lum, sigma_lum, qobs_lum, surf_pot,
                     sigma_pot, qobs_pot,
                     inc, mbh, distance,
                     xbin1, ybin1, filename,  ml, 
                     rms=rms1, erms=erms1,
                      plot=False, beta=beta_array, 
                     tensor='zz', quiet=True)
        realChi2_sluggs = np.sum(np.log(2*np.pi*erms1**2/sluggsWeight) + (sluggsWeight*(rms1 - rmsModel)/erms1)**2.)
        
        rmsModel, ml, chi2, flux = Jrms.jam_axi_rms(
                     surf_lum, sigma_lum, qobs_lum, surf_pot,
                     sigma_pot, qobs_pot,
                     inc, mbh, distance,
                     xbin2, ybin2, filename, ml,
                     rms=rms2, erms=erms2,
                      plot=False, beta=beta_array, 
                     tensor='zz', quiet=True)
        realChi2_atlas = np.sum(np.log(2*np.pi*erms2**2/atlasWeight) + (atlasWeight*(rms2 - rmsModel)/erms2)**2.)

        rmsModel, ml, chi2, flux = Jrms.jam_axi_rms(
                     surf_lum, sigma_lum, qobs_lum, surf_pot,
                     sigma_pot, qobs_pot,
                     inc, mbh, distance,
                     xbin3, ybin3, filename, ml,
                     rms=rms3, erms=erms3,
                      plot=False, beta=beta_array, 
                     tensor='zz', quiet=True)
        realChi2_gc = np.sum(np.log(2*np.pi*erms3**2/gcWeight) + (gcWeight*(rms3 - rmsModel)/erms3)**2.)
        
        realChi2 = realChi2_sluggs + realChi2_atlas + realChi2_gc
      except:
        realChi2 = np.inf 
  
  else:
    raise ValueError('Model is unrecognisable. Should be either "gNFW" or "powerLaw"')

     
    
    
  return -realChi2/2



def InitialWalkerPosition(WalkerNumber, LowerBounds, UpperBounds, PriorType):
  # wiriting up a code that, for an arbitrary number of parameters and prior bounds, will set up the initial walker positions. 

  InitialPosition = []
  for ii in np.arange(WalkerNumber): 
    WalkerPosition = [] 
    for jj in range(len(LowerBounds)):
      if PriorType[jj] == 'uniform':
        WalkerPosition.append(np.random.uniform(low=LowerBounds[jj], high=UpperBounds[jj]) )
      elif PriorType[jj] == 'exponential':
        ExpInitPosition = np.random.uniform(low=np.exp(-UpperBounds[jj]), high=np.exp(LowerBounds[jj])) 
        WalkerPosition.append(-np.log(ExpInitPosition))
  
    InitialPosition.append(WalkerPosition)

  Boundaries = []
  for ii in range(len(LowerBounds)):
    Boundaries.append(LowerBounds[ii])
    Boundaries.append(UpperBounds[ii])

  return InitialPosition, Boundaries

def mainCall_modular(GalName, Input_path, JAM_path, Model = 'gNFW', SLUGGS = True, ATLAS = True, GC = False,\
  Inclination = True, Beta = True, Gamma = True, ScaleDensity = True, ML = False, ScaleRadius = False, \
  SluggsWeight = False, AtlasWeight = False, GCWeight = False, \
  nwalkers = 2000, burnSteps = 1000, stepNumber = 4000):

  MGE = MGE_Source[GalName]
  
  '''
  Loading the MGE parameters for each galaxy. 
  MGE source is determined in the JAM_Dictionary.py file, under the MGE_Source array
  '''
  surf_lum, sigma_lum, qobs_lum = MGEfinder(MGE, GalName, units = 'original') # specify whether the units of the MGE are in arcsec,
                                                                              # kpc, or in the original units.
  sigmapsf = 0.6
  pixsize = 0.8
  filename=JAM_path+'JAM/Output/'+str(GalName)+'/JAM_'+str(GalName)
  try:
    mbh = BlackHoleMass[GalName]
  except:
    mbh = 0
    print 'No black hole mass specified for', GalName
  distance = distMpc[GalName]
  reff = Reff_Spitzer[GalName] * (distance * 1e6) / 206265 # defining the effective radius in pc. 
  # R_S = 20000 # leaving the input scale radius in parsec. 
  # print 'Break radius (arcseconds):', R_S
  
  
  t0 = time.time()

  '''
  Calling in the input file as determined by JAM_InputGenerator.py in the ../InputFiles folder. 
  This method of calling in the files doesn't affect the cleaning of the data at all. 
  All the cleaning is done in the previously mentioned python code. 
  '''
  Data = []
  if SLUGGS:
    input_filename_sluggs = JAM_path+'/'+Input_path+'/'+str(GalName)+'/JAM_Input_'+GalName+'_SLUGGS.txt'
    xbin_sluggs, ybin_sluggs, rms_sluggs, erms_sluggs = np.loadtxt(input_filename_sluggs, unpack = True, usecols = [0, 1, 2, 3], comments = '#')
    Data.append(xbin_sluggs)
    Data.append(ybin_sluggs)
    Data.append(rms_sluggs)
    Data.append(erms_sluggs)
  if ATLAS:
    input_filename_atlas = JAM_path+'/'+Input_path+'/'+str(GalName)+'/JAM_Input_'+GalName+'_ATLAS3D.txt'
    xbin_atlas, ybin_atlas, rms_atlas, erms_atlas = np.loadtxt(input_filename_atlas, unpack = True, usecols = [0, 1, 2, 3], comments = '#')
    Data.append(xbin_atlas)
    Data.append(ybin_atlas)
    Data.append(rms_atlas)
    Data.append(erms_atlas)
  if GC:
    input_filename_gc = JAM_path+'/'+Input_path+'/'+str(GalName)+'/JAM_Input_'+GalName+'_GC.txt'
    xbin_gc, ybin_gc, rms_gc, erms_gc = np.loadtxt(input_filename_gc, unpack = True, usecols = [0, 1, 2, 3], comments = '#')
    Data.append(xbin_gc)
    Data.append(ybin_gc)
    Data.append(rms_gc)
    Data.append(erms_gc)


  min_inclination = degree(np.arccos(np.min(qobs_lum))) # using the MGE to calculate the minimum possible inclination 
 
  ndim = 0
  LowerBounds, UpperBounds, PriorType, ParameterNames, ParamSymbol = [], [], [], [], []
  # selecting all possible free parameters, with their prior bounds:
  if Inclination: # inclination 
    ndim += 1
    LowerBounds.append(min_inclination)
    UpperBounds.append(90)
    PriorType.append('uniform')
    ParameterNames.append('Inclination')
    ParamSymbol.append(r"$i$")
  if Beta: # inclination 
    ndim += 1
    LowerBounds.append(-0.2)
    UpperBounds.append(0.5)
    PriorType.append('uniform')
    ParameterNames.append('Beta')
    ParamSymbol.append(r"$\beta$")
  if Gamma: # inclination 
    ndim += 1
    LowerBounds.append(-2.4)
    UpperBounds.append(-1.5)
    PriorType.append('uniform')
    ParameterNames.append('Gamma')
    ParamSymbol.append(r"$\gamma$")
  if ScaleDensity: # inclination 
    ndim += 1
    LowerBounds.append(np.log10(0.1))
    UpperBounds.append(np.log10(100000))
    PriorType.append('uniform')
    ParameterNames.append('ScaleDensity')
    ParamSymbol.append(r"$\rho_S$")
  if ML: # inclination 
    ndim += 1
    LowerBounds.append(1)
    UpperBounds.append(5)
    PriorType.append('uniform')
    ParameterNames.append('ML')
    ParamSymbol.append(r"$M/L$")
  if ScaleRadius: # inclination 
    ndim += 1
    LowerBounds.append(10000)
    UpperBounds.append(40000)
    PriorType.append('uniform')
    ParameterNames.append('ScaleRadius')
    ParamSymbol.append(r"$R_S$")

  if SluggsWeight:
    ndim += 1
    LowerBounds.append(0)
    UpperBounds.append(10)
    PriorType.append('exponential')
    ParameterNames.append('SluggsWeight')
    ParamSymbol.append(r"$\omega_{\rm SLUGGS}$")
  if AtlasWeight:
    ndim += 1
    LowerBounds.append(0)
    UpperBounds.append(10)
    PriorType.append('exponential')
    ParameterNames.append('AtlasWeight')
    ParamSymbol.append(r"$\omega_{\rm ATLAS}$")
  if GCWeight:
    ndim += 1
    LowerBounds.append(0)
    UpperBounds.append(10)
    PriorType.append('exponential')
    ParameterNames.append('GCWeight')
    ParamSymbol.append(r"$\omega_{\rm GC}$")

  pos, boundaries = InitialWalkerPosition(nwalkers, LowerBounds, UpperBounds, PriorType)
  
  # Setup MCMC sampler
  import emcee
  sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                    args=(boundaries, ParameterNames, Data, Model, mbh, distance,
                         reff, filename, surf_lum, sigma_lum, qobs_lum),
                    threads=16) #Threads gives the number of processors to use
  
  ############ implementing a burn-in period ###########
  pos_afterBurn, prob, state = sampler.run_mcmc(pos, burnSteps)
  sampler.reset()
  
  suffix = ''
  if SLUGGS:
    suffix = suffix + 'SLUGGS_'
  if ATLAS:
    suffix = suffix + 'ATLAS_'
  if GC:
    suffix = suffix + 'GC_'
  suffix = suffix + 'FreeParam-'+str(ndim)

  if not os.path.exists(JAM_path+'/'+str(GalName)): 
    os.mkdir(JAM_path+'/'+str(GalName))

  if not os.path.exists(JAM_path+'/'+str(GalName)+'/'+suffix): 
    os.mkdir(JAM_path+'/'+str(GalName)+'/'+suffix)

  OutputFilename = JAM_path+'/'+str(GalName)+'/'+suffix+'/'+str(GalName)+'_MCMCOutput_'+suffix+'.dat'
  
  stepsBetweenIterations = stepNumber / 100
  iterationNumber = stepNumber / stepsBetweenIterations

  import pickle

  for iteration in range(iterationNumber):
    pos_afterBurn, prob, state = sampler.run_mcmc(pos_afterBurn, stepsBetweenIterations) # uses the final position of the burn-in period as the starting point. 
    fileOut = open(OutputFilename, 'wb')
    pickle.dump([sampler.chain, sampler.flatchain, sampler.lnprobability, sampler.flatlnprobability], fileOut)
    fileOut.close()
    print 'Number of steps completed:', (iteration+1)*stepsBetweenIterations, len(sampler.chain[0])
  ######################################################
  
  
  t = time.time() - t0
  Date=time.strftime('%Y-%m-%d')
  Time = time.asctime().split()[3]
  print '########################################'
  print 'time elapsed:', float(t)/3600, 'hours'
  print '########################################'
  print 'Mean acceptance fraction: ', (np.mean(sampler.acceptance_fraction))
  print 'output filename: ', OutputFilename

  fileOut = open(OutputFilename, 'wb')
  pickle.dump([sampler.chain, sampler.flatchain, sampler.lnprobability, sampler.flatlnprobability], fileOut)
  fileOut.close()

  return OutputFilename, ParamSymbol, ParameterNames, Data 
