import mge_fit_1d as mge
import numpy as np
import jam_axi_rms as Jrms
import matplotlib.pyplot as plt
import os, sys
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from scipy.misc import imread
import matplotlib.cbook as cbook

from galaxyParametersDictionary_v9 import *
from Sabine_Define import *

def gNFW(radius, rho_s, R_s, gamma): # From Cappellari+13
  r_tmp = radius/R_s
  return rho_s*(r_tmp**gamma)*(0.5+0.5*r_tmp)**(-gamma-3)

def gNFW_RelocatedDensity(x, radius, rho_x, R_s, gamma): # From Cappellari+13
  return rho_x * (radius/x)**gamma * ((R_s+radius)/(R_s+x))**(-gamma-3)

def lnlike_gNFW(theta, *args):
  try:
    inc, beta, log_rho_s, gamma, sluggsWeight, atlasWeight  = theta
    tmpInputArgs = args[0]
    (xbin1, ybin1, rms1, erms1, xbin2, ybin2, rms2, erms2, mbh, distance, reff, filename, ml, 
      surf_lum, sigma_lum, qobs_lum, R_S) = tmpInputArgs
    beta_array = np.ones(len(surf_lum))*beta
    
    n_MGE = 10.
    x_MGE = np.logspace(np.log10(0.01), np.log10(500), n_MGE)
    y_MGE = gNFW_RelocatedDensity(R_S/20, x_MGE, 10.**log_rho_s, R_S, gamma)

    p = mge.mge_fit_1d(x_MGE, y_MGE, ngauss=10, quiet=True)
    
    surf_dm = p.sol[0]                                # here implementing a potential
    sigma_dm = p.sol[1]                               # configuration which includes the 
    qobs_dm = np.ones(len(surf_dm))                   # stellar mass and also a dark matter    
                                                      # halo in the form of a gNFW distribution. 
    surf_pot = np.append(surf_lum, surf_dm)           #                                                 
    sigma_pot = np.append(sigma_lum, sigma_dm)        #                                                   
    qobs_pot = np.append(qobs_lum, qobs_dm)           #     
   
    rmsModel, ml, chi2, flux = Jrms.jam_axi_rms(
                 surf_lum, sigma_lum, qobs_lum, surf_pot,
                 sigma_pot, qobs_pot,
                 inc, mbh, distance,
                 xbin1, ybin1, filename, ml,
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
  return -realChi2 

## Defining (uniform) priors on the probability of the parameters
def lnprior(theta): #p(inc, beta_in, beta, log_rho_s, R_s, gamma)
    incl, beta, log_rho_0, log_R_s = theta
    if ((60 < incl <= 90) and (-0.5 < beta < 0.5) #and (-6 < log_rho_0 <0)
       and (np.log10(100)<R_s<np.log10(4000))): #Roughly Cappellari's 10<R_s<50 kpc
        return 0.0
    else:
        return -np.inf

def lnprob(theta, *args):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, *args) #In logarithmic space, the multiplication becomes sum.


def lnprior_gNFW(theta, *args): #p(m, b, f)
    incl, beta, log_rho_s, gamma = theta

    tmpInputArgs = args[0] 
    (incTmp_lower, incTmp_upper, betaoutTmp_lower, betaoutTmp_upper, log_rhoSTmp_lower, log_rhoSTmp_upper, gamma_lower, gamma_upper, 
      sluggsWeight_lower, sluggsWeight_upper, atlasWeight_lower, atlasWeight_upper) = tmpInputArgs

    if ((incTmp_lower < incl <= incTmp_upper) and 
       (betaoutTmp_lower < beta < betaoutTmp_upper)  and 
       (log_rhoSTmp_lower < log_rho_s < log_rhoSTmp_upper) and (gamma_lower < gamma < gamma_upper)
       and (sluggsWeight_lower < sluggsWeight <= sluggsWeight_upper) and (atlasWeight_lower < atlasWeight <= atlasWeight_upper)):
        return 0.
    else:
        return -np.inf

def lnprob_gNFW(theta, *args):
    arg1 = args[0]
    arg2 = args[1:]

    lp = lnprior_gNFW(theta, arg1)
    # print lp
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_gNFW(theta, arg2) #In logarithmic space, the multiplication becomes sum.

def powerLaw(radius, rho_s, R_s, gamma, outer): # here we use a broken power law. 
  # y = []
  # for ii in radius:
  #   if ii < R_s:
  #     y.append(rho_s * ii**(gamma))
  #   elif ii >= R_s:
  #     y.append(rho_s * R_s**(gamma-outer) * ii**(outer))
  # return np.array(y)
  '''
  I have realised that the initial power law parametrisation wasn't doing what I thoguht it was
  therefore I will change the parametrisation to be the same as the gNFW, in the same manner was
  was done by Poci+17
  '''
  x = R_s/20.
  return rho_s * (radius/x)**gamma * ((R_s+radius)/(R_s+x))**(-gamma+outer)


def lnlike_powerLaw(theta, *args):
  try:
    inc, beta, log_rho_s, gamma = theta
    tmpInputArgs = args[0]
    (xbin, ybin, rms, erms, mbh, distance, reff, filename, #ml, 
      surf_lum, sigma_lum, qobs_lum, R_S) = tmpInputArgs
    innerIndex = np.where(sigma_lum <= reff)
    beta_array = np.ones(len(surf_lum))*beta
  
    n_MGE = 300.
    x_MGE = np.logspace(np.log10(1), np.log10(30000), n_MGE)    # logarithmically spaced radii in parsec
                                                                # the units of the density are now mass/pc^2
    y_MGE = powerLaw(x_MGE, 10.**log_rho_s, R_S, gamma, -3)     # The profile should be logarithmically sampled! The break radius is
                                                                # set to 20 kpc. 
    p = mge.mge_fit_1d(x_MGE, y_MGE, ngauss=10, quiet=True)
  
    surf_pot = p.sol[0]                               
    sigma_pot = p.sol[1] * 206265 / (distance * 1e6)            # converting the dispersions into arcseconds because it wants only                           
    qobs_pot = np.ones(len(surf_pot))                           # the radial dimension in these units. 
                                            
    # print 'MGE parameters:', surf_lum, sigma_lum, qobs_lum
  
    # print 'calculating rmsModel'
    rmsModel, ml, chi2, flux = Jrms.jam_axi_rms(
                 surf_lum, sigma_lum, qobs_lum, surf_pot,
                 sigma_pot, qobs_pot,
                 inc, mbh, distance,
                 xbin, ybin, filename, #ml,
                 rms=rms, erms=erms,
                  plot=False, beta=beta_array, #ml=-1,
                #sigmapsf=0.1,
                #pixsize=0.1, #Prevent convolution by PSF
                 tensor='zz', quiet=True)
    realChi2 = np.sum(((rms - rmsModel)/erms)**2.)
    # print realChi2
    # print rmsModel
    # print -realChi2/2.

  except:
    realChi2 = np.inf #take it off (lnprior already takes care of it)
  return -realChi2/2. #Cappellari13


def lnprior_powerLaw(theta, *args): #p(m, b, f)
    incl, beta, log_rho_s, gamma, sluggsWeight, atlasWeight = theta

    tmpInputArgs = args[0] 
    (incTmp_lower, incTmp_upper, 
      betaoutTmp_lower, betaoutTmp_upper, log_rhoSTmp_lower, log_rhoSTmp_upper,gamma_lower, gamma_upper) = tmpInputArgs

    if ((incTmp_lower < incl <= incTmp_upper) and 
       (betaoutTmp_lower < beta < betaoutTmp_upper)  and 
       (log_rhoSTmp_lower < log_rho_s < log_rhoSTmp_upper) and (gamma_lower < gamma < gamma_upper)):
        return 0.
    else:
        return -np.inf

def lnprob_powerLaw(theta, *args):
    arg1 = args[0]
    arg2 = args[1:]

    lp = lnprior_powerLaw(theta, arg1)
    # print 'lnprob output', lp, lnlike_powerLaw(theta, arg2)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_powerLaw(theta, arg2) #In logarithmic space, the multiplication becomes sum.

def lnlike_powerLaw_separateDatasets(theta, *args):
  try:
    inc, beta, log_rho_s, gamma, sluggsWeight, atlasWeight = theta
    tmpInputArgs = args[0]
    (xbin1, ybin1, rms1, erms1, xbin2, ybin2, rms2, erms2, mbh, distance, reff, filename, ml, 
      surf_lum, sigma_lum, qobs_lum, R_S) = tmpInputArgs
    # innerIndex = np.where(sigma_lum <= reff)
    beta_array = np.ones(len(surf_lum))*beta

    
    n_MGE = 300.
    x_MGE = np.logspace(np.log10(1), np.log10(30000), n_MGE)    # logarithmically spaced radii in parsec
                                                                # the units of the density are now mass/pc^2
    y_MGE = powerLaw(x_MGE, 10.**log_rho_s, R_S, gamma, -3)     # The profile should be logarithmically sampled! The break radius is
                                                                # set to 20 kpc. 
    p = mge.mge_fit_1d(x_MGE, y_MGE, ngauss=10, quiet=True)
  
    surf_pot = p.sol[0]                               
    sigma_pot = p.sol[1] * 206265 / (distance * 1e6)            # converting the dispersions into arcseconds because it wants only                           
    qobs_pot = np.ones(len(surf_pot))                           # the radial dimension in these units. 

    rmsModel, ml, chi2, flux = Jrms.jam_axi_rms(
                 surf_lum, sigma_lum, qobs_lum, surf_pot,
                 sigma_pot, qobs_pot,
                 inc, mbh, distance,
                 xbin1, ybin1, filename, ml,
                 rms=rms1, erms=erms1,
                  plot=False, beta=beta_array, 
                 tensor='zz', quiet=True)
    realChi2_sluggs = np.sum(np.log(2*np.pi*erms1**2/sluggsWeight) + (sluggsWeight*(rms1 - rmsModel)/erms1)**2.)

    # print rmsModel
    rmsModel, ml, chi2, flux = Jrms.jam_axi_rms(
                 surf_lum, sigma_lum, qobs_lum, surf_pot,
                 sigma_pot, qobs_pot,
                 inc, mbh, distance,
                 xbin2, ybin2, filename, ml,
                 rms=rms2, erms=erms2,
                  plot=False, beta=beta_array, 
                 tensor='zz', quiet=True)
    realChi2_atlas = np.sum(np.log(2*np.pi*erms2**2/atlasWeight) + (atlasWeight*(rms2 - rmsModel)/erms2)**2.)

    # print rmsModel
    
    realChi2 = realChi2_sluggs + realChi2_atlas
  except:
    realChi2 = np.inf #take it off (lnprior already takes care of it)
  return -realChi2 #Cappellari13

def lnprior_powerLaw_separateDatasets(theta, *args): #p(m, b, f)
    incl, beta, log_rho_s, gamma, sluggsWeight, atlasWeight = theta
 
    tmpInputArgs = args[0] 
    (incTmp_lower, incTmp_upper, betaoutTmp_lower, betaoutTmp_upper, log_rhoSTmp_lower, log_rhoSTmp_upper,
        gamma_lower, gamma_upper, sluggsWeight_lower, sluggsWeight_upper, atlasWeight_lower, atlasWeight_upper) = tmpInputArgs
 
    if ((incTmp_lower < incl <= incTmp_upper) and
       (betaoutTmp_lower < beta <= betaoutTmp_upper)  and
       (log_rhoSTmp_lower < log_rho_s <= log_rhoSTmp_upper) and (gamma_lower < gamma <= gamma_upper)
        and (sluggsWeight_lower < sluggsWeight <= sluggsWeight_upper) and (atlasWeight_lower < atlasWeight <= atlasWeight_upper)):
        return 0.
        # print 'boundary conditions not satisfied!'
    else:
        return -np.inf

def lnprob_powerLaw_separateDatasets(theta, *args):
    arg1 = args[0]
    arg2 = args[1:]

    lp = lnprior_powerLaw_separateDatasets(theta, arg1)
    # print lp, lnlike_powerLaw_separateDatasets(theta, arg2)
    # print 'lnprob output', lp, lnlike_powerLaw(theta, arg2)
    if not np.isfinite(lp):
        return -np.inf
    #     print 'lp is infinite'
    # print lp, lnlike_powerLaw_separateDatasets(theta, arg2)
    return lp + lnlike_powerLaw_separateDatasets(theta, arg2) #In logarithmic space, the multiplication becomes sum.

def lnlike_gNFW_ml(theta, *args):
  try:
    inc, beta, log_rho_s, gamma, ml, sluggsWeight, atlasWeight  = theta
    tmpInputArgs = args[0]
    (xbin1, ybin1, rms1, erms1, xbin2, ybin2, rms2, erms2, mbh, distance, reff, filename, 
      surf_lum, sigma_lum, qobs_lum, R_S) = tmpInputArgs
    beta_array = np.ones(len(surf_lum))*beta
    
    n_MGE = 300.                                                                 # logarithmically spaced radii in parsec
    x_MGE = np.logspace(np.log10(1), np.log10(30000), n_MGE)                    # the units of the density are now mass/pc^2
    y_MGE = gNFW_RelocatedDensity(R_S/20, x_MGE, 10.**log_rho_s, R_S, gamma)    

    p = mge.mge_fit_1d(x_MGE, y_MGE, ngauss=10, quiet=True)
    
    surf_dm = p.sol[0]                                # here implementing a potential
    sigma_dm = p.sol[1]  * 206265 / (distance * 1e6)  # configuration which includes the 
    qobs_dm = np.ones(len(surf_dm))                   # stellar mass and also a dark matter    
                                                      # halo in the form of a gNFW distribution. 
  
    surf_pot = np.append(ml * surf_lum, surf_dm)      #  I'm going to need to be careful here to account for the stellar M/L,                                               
    sigma_pot = np.append(sigma_lum, sigma_dm)        #  since the scaling between the two mass distributions needs to be right. 
    qobs_pot = np.append(qobs_lum, qobs_dm)           #     
    
    rmsModel, ml, chi2, flux = Jrms.jam_axi_rms(
                 surf_lum, sigma_lum, qobs_lum, surf_pot,
                 sigma_pot, qobs_pot,
                 inc, mbh, distance,
                 xbin1, ybin1, filename, #ml,
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
  return -realChi2/2. 

def lnprior_gNFW_ml(theta, *args): #p(m, b, f)
  incl, beta, log_rho_s, gamma, ml, sluggsWeight, atlasWeight = theta

  tmpInputArgs = args[0] 
  (incTmp_lower, incTmp_upper, betaoutTmp_lower, betaoutTmp_upper, log_rhoSTmp_lower, log_rhoSTmp_upper, gamma_lower, gamma_upper, 
    ml_lower, ml_upper, 
    sluggsWeight_lower, sluggsWeight_upper, atlasWeight_lower, atlasWeight_upper) = tmpInputArgs

  if ((incTmp_lower < incl <= incTmp_upper) and 
     (betaoutTmp_lower < beta < betaoutTmp_upper)  and 
     (log_rhoSTmp_lower < log_rho_s < log_rhoSTmp_upper) and (gamma_lower < gamma < gamma_upper)
     and (sluggsWeight_lower < sluggsWeight <= sluggsWeight_upper) and (atlasWeight_lower < atlasWeight <= atlasWeight_upper)
      and (ml_lower < ml <= ml_upper)):
      return 0.
  else:
      return -np.inf

def lnprob_gNFW_ml(theta, *args):
    arg1 = args[0]
    arg2 = args[1:]

    lp = lnprior_gNFW_ml(theta, arg1)
    # print lp
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_gNFW_ml(theta, arg2) #In logarithmic space, the multiplication becomes sum.

# Analysis functions:

def customCorner(GalName, parameterArrays, MaximumProbabilityValues, MaximumProbabilityValueErrors, filename, totalMassDistribution, figure, chi2):
    
    
    IncRange = [np.min(parameterArrays[0]), np.max(parameterArrays[0])]
    BetaRange = [np.min(parameterArrays[1]), np.max(parameterArrays[1])]
    RhoRange = [np.min(parameterArrays[2]), np.max(parameterArrays[2])]
    GammaRange = [np.min(parameterArrays[3]), np.max(parameterArrays[3])]
    
    # making my own corner plot that looks better. 
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace = 0.04, hspace = 0.03)
    
    
    fig = plt.figure(figsize = (10, 10)) 
    
    ax1 =  plt.subplot(gs1[0, 0])
    ax2 =  plt.subplot(gs1[1, 0])
    plt.yticks(rotation=30)
    ax3 =  plt.subplot(gs1[1, 1])
    ax4 =  plt.subplot(gs1[2, 0])
    plt.yticks(rotation=30)
    ax5 =  plt.subplot(gs1[2, 1])
    ax6 =  plt.subplot(gs1[2, 2])
    ax7 =  plt.subplot(gs1[3, 0])
    plt.xticks(rotation=30)
    plt.yticks(rotation=30)
    ax8 =  plt.subplot(gs1[3, 1])
    plt.xticks(rotation=30)
    ax9 =  plt.subplot(gs1[3, 2])
    plt.xticks(rotation=30)
    ax10 = plt.subplot(gs1[3, 3])
    plt.xticks(rotation=30)
    
    ax11 = plt.subplot(222, aspect = 'equal') # an additional subplot for the kinematic fit diagram. 
    datafile = cbook.get_sample_data(figure)
    img = imread(datafile)
    # img = np.flipud(imread(datafile))
    # ax2 = fig.add_subplot(312, aspect = 'equal')
    ax11.imshow(img, zorder=0, interpolation = 'None')
    ax11.set_xticks([])
    ax11.set_yticks([])
    ax11.set_xticklabels([])
    ax11.set_yticklabels([])
    ax11.axis('off')
    
    
    ax1.hist(parameterArrays[0], bins = 30, histtype='stepfilled', color = 'white')
    
    counts,xbins,ybins=np.histogram2d(parameterArrays[0], parameterArrays[1],bins=30,normed=LogNorm())
    ax2.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
    # ax2.scatter(Inclination_all, Beta_all, s = 0.5, c = 'k', alpha = 0.9, edgecolor='None')
    
    counts,xbins,ybins=np.histogram2d(parameterArrays[0], parameterArrays[2],bins=30,normed=LogNorm())
    ax4.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
    # ax4.scatter(Inclination_all, Rho_All, s = 0.5, c = 'k', alpha = 0.9, edgecolor='None')
    
    counts,xbins,ybins=np.histogram2d(parameterArrays[0], parameterArrays[3],bins=30,normed=LogNorm())
    ax7.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
    # ax7.scatter(Inclination_all, Gamma_All, s = 0.5, c = 'k', alpha = 0.9, edgecolor='None')
    
    
    ax3.hist(parameterArrays[1], bins = 30, histtype='stepfilled', color = 'white')
    
    counts,xbins,ybins=np.histogram2d(parameterArrays[1], parameterArrays[2],bins=30,normed=LogNorm())
    ax5.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
    # ax5.scatter(Beta_all, Rho_All, s = 0.5, c = 'k', alpha = 0.9, edgecolor='None')
    
    counts,xbins,ybins=np.histogram2d(parameterArrays[1], parameterArrays[3],bins=30,normed=LogNorm())
    ax8.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
    # ax8.scatter(Beta_all, Gamma_All, s = 0.5, c = 'k', alpha = 0.9, edgecolor='None')
    
    ax6.hist(parameterArrays[2], bins = 30, histtype='stepfilled', color = 'white')
    
    counts,xbins,ybins=np.histogram2d(parameterArrays[2], parameterArrays[3],bins=30,normed=LogNorm())
    ax9.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
    # ax9.scatter(Rho_All, Gamma_All, s = 0.5, c = 'k', alpha = 0.9, edgecolor='None')
    
    ax10.hist(parameterArrays[3], bins = 30, histtype='stepfilled', color = 'white')
    
    ax1.axvline(MaximumProbabilityValues[0])
    ax2.axvline(MaximumProbabilityValues[0], c = 'r')
    ax4.axvline(MaximumProbabilityValues[0], c = 'r')
    ax7.axvline(MaximumProbabilityValues[0], c = 'r')
    ax3.axvline(MaximumProbabilityValues[1])
    ax5.axvline(MaximumProbabilityValues[1], c = 'r')
    ax8.axvline(MaximumProbabilityValues[1], c = 'r')
    ax6.axvline(MaximumProbabilityValues[2])
    ax9.axvline(MaximumProbabilityValues[2], c = 'r')

    ax10.axvline(MaximumProbabilityValues[3])
    
    ax7.axhline(MaximumProbabilityValues[3], c = 'r')
    ax8.axhline(MaximumProbabilityValues[3], c = 'r')
    ax9.axhline(MaximumProbabilityValues[3], c = 'r')
    ax4.axhline(MaximumProbabilityValues[2], c = 'r')
    ax5.axhline(MaximumProbabilityValues[2], c = 'r')
    ax2.axhline(MaximumProbabilityValues[1], c = 'r')
    
    ax1.axvline(MaximumProbabilityValues[0]+MaximumProbabilityValueErrors[0], linestyle = '--')
    ax1.axvline(MaximumProbabilityValues[0]-MaximumProbabilityValueErrors[1], linestyle = '--')
    ax3.axvline(MaximumProbabilityValues[1]+MaximumProbabilityValueErrors[2], linestyle = '--')
    ax3.axvline(MaximumProbabilityValues[1]-MaximumProbabilityValueErrors[3], linestyle = '--')
    ax6.axvline(MaximumProbabilityValues[2]+MaximumProbabilityValueErrors[4], linestyle = '--')
    ax6.axvline(MaximumProbabilityValues[2]-MaximumProbabilityValueErrors[5], linestyle = '--')

    ax10.axvline(MaximumProbabilityValues[3]+MaximumProbabilityValueErrors[6], linestyle = '--')
    ax10.axvline(MaximumProbabilityValues[3]-MaximumProbabilityValueErrors[7], linestyle = '--')
    
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    ax4.set_xticklabels([])
    ax5.set_xticklabels([])
    ax6.set_xticklabels([])
    # ax10.set_xticklabels([])
    
    ax1.set_yticklabels([])
    ax3.set_yticklabels([])
    ax5.set_yticklabels([])
    ax6.set_yticklabels([])
    ax8.set_yticklabels([])
    ax9.set_yticklabels([])
    ax10.set_yticklabels([])
    
    ax1.set_xlim(IncRange)
    ax2.set_xlim(IncRange)
    ax3.set_xlim(BetaRange)
    ax4.set_xlim(IncRange)
    ax5.set_xlim(BetaRange)
    ax6.set_xlim(RhoRange)
    ax7.set_xlim(IncRange)
    ax8.set_xlim(BetaRange)
    ax9.set_xlim(RhoRange)
    ax10.set_xlim(GammaRange)
    
    ax2.set_ylim(BetaRange)
    ax4.set_ylim(RhoRange)
    ax5.set_ylim(RhoRange)
    ax7.set_ylim(GammaRange)
    ax8.set_ylim(GammaRange)
    ax9.set_ylim(GammaRange)
    
    ax2.set_ylabel(r"$\beta$", fontsize = 16)
    ax4.set_ylabel(r"$\log_{10}\rho _{s}$", fontsize = 16)
    ax7.set_ylabel('$\gamma_{tot}$', fontsize = 16)
    
    ax7.set_xlabel('$i$', fontsize = 16)
    ax8.set_xlabel(r"$\beta$", fontsize = 16)
    ax9.set_xlabel(r"$\log_{10}\rho _{s}$", fontsize = 16)
    ax10.set_xlabel('$\gamma_{tot}$', fontsize = 16)
    
    ax1.set_title('$i = '+str(round(MaximumProbabilityValues[0], 2))+\
      '^{+'+str(round(MaximumProbabilityValueErrors[0], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[1], 2))+'}$', fontsize = 16)
    ax3.set_title(r'$\beta = '+str(round(MaximumProbabilityValues[1], 2))+\
      '^{+'+str(round(MaximumProbabilityValueErrors[2], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[3], 2))+'}$', fontsize = 16)
    ax6.set_title(r'$\log_{10}\rho _{s} = '+str(round(MaximumProbabilityValues[2], 2))+\
      '^{+'+str(round(MaximumProbabilityValueErrors[4], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[5], 2))+'}$', fontsize = 16)
    ax10.set_title('$\gamma_{tot} = '+str(round(MaximumProbabilityValues[3], 2))+\
      '^{+'+str(round(MaximumProbabilityValueErrors[6], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[7], 2))+'}$', fontsize = 16)
    
    ax3.text(-0.5, 3, 'NGC '+GalName.split('C')[1], transform=ax6.transAxes,
          fontsize=16, fontweight='bold', va='top', ha='left')
    
    ax3.text(-0.5, 2.8, 'total mass distribution: '+str(totalMassDistribution), transform=ax6.transAxes,
          fontsize=14, fontweight='bold', va='top', ha='left')
    ax3.text(-0.5, 2.6, r'$\chi^2$/DOF: '+str(round(chi2, 2)), transform=ax6.transAxes,
          fontsize=14, fontweight='bold', va='top', ha='left')
    plt.savefig(filename)

def customCorner_ml(GalName, parameterArrays, MaximumProbabilityValues, MaximumProbabilityValueErrors, filename, totalMassDistribution, figure, chi2):
    
    
    IncRange = [np.min(parameterArrays[0]), np.max(parameterArrays[0])]
    BetaRange = [np.min(parameterArrays[1]), np.max(parameterArrays[1])]
    RhoRange = [np.min(parameterArrays[2]), np.max(parameterArrays[2])]
    GammaRange = [np.min(parameterArrays[3]), np.max(parameterArrays[3])]
    MlRange = [np.min(parameterArrays[4]), np.max(parameterArrays[4])]
    
    # making my own corner plot that looks better. 
    gs1 = gridspec.GridSpec(5, 5)
    gs1.update(wspace = 0.04, hspace = 0.03)
    
    
    fig = plt.figure(figsize = (10, 10)) 
    
    ax1 =  plt.subplot(gs1[0, 0])
    ax2 =  plt.subplot(gs1[1, 0])
    plt.yticks(rotation=30)
    ax3 =  plt.subplot(gs1[1, 1])
    ax4 =  plt.subplot(gs1[2, 0])
    plt.yticks(rotation=30)
    ax5 =  plt.subplot(gs1[2, 1])
    ax6 =  plt.subplot(gs1[2, 2])
    ax7 =  plt.subplot(gs1[3, 0])
    plt.yticks(rotation=30)
    ax8 =  plt.subplot(gs1[3, 1])
    ax9 =  plt.subplot(gs1[3, 2])
    ax10 = plt.subplot(gs1[3, 3])
    ax11 = plt.subplot(gs1[4, 0])
    plt.yticks(rotation=30)
    plt.xticks(rotation=30)
    ax12 = plt.subplot(gs1[4, 1])
    plt.xticks(rotation=30)
    ax13 = plt.subplot(gs1[4, 2])
    plt.xticks(rotation=30)
    ax14 = plt.subplot(gs1[4, 3])
    plt.xticks(rotation=30)
    ax15 = plt.subplot(gs1[4, 4])
    plt.xticks(rotation=30)
   
    ax16 = plt.subplot(233, aspect = 'equal') # an additional subplot for the kinematic fit diagram. 
    datafile = cbook.get_sample_data(figure)
    img = imread(datafile)
    # img = np.flipud(imread(datafile))
    # ax2 = fig.add_subplot(312, aspect = 'equal')
    ax16.imshow(img, zorder=0, interpolation = 'None')
    ax16.set_xticks([])
    ax16.set_yticks([])
    ax16.set_xticklabels([])
    ax16.set_yticklabels([])
    ax16.axis('off')
    
    
    # column 1
    ax1.hist(parameterArrays[0], bins = 30, histtype='stepfilled', color = 'white')
      
    counts,xbins,ybins=np.histogram2d(parameterArrays[0], parameterArrays[1],bins=30,normed=LogNorm())
    ax2.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
      
    counts,xbins,ybins=np.histogram2d(parameterArrays[0], parameterArrays[2],bins=30,normed=LogNorm())
    ax4.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
      
    counts,xbins,ybins=np.histogram2d(parameterArrays[0], parameterArrays[3],bins=30,normed=LogNorm())
    ax7.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
  
    counts,xbins,ybins=np.histogram2d(parameterArrays[0], parameterArrays[4],bins=30,normed=LogNorm())
    ax11.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
      
    # column 2
    ax3.hist(parameterArrays[1], bins = 30, histtype='stepfilled', color = 'white')
      
    counts,xbins,ybins=np.histogram2d(parameterArrays[1], parameterArrays[2],bins=30,normed=LogNorm())
    ax5.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
      
    counts,xbins,ybins=np.histogram2d(parameterArrays[1], parameterArrays[3],bins=30,normed=LogNorm())
    ax8.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
  
    counts,xbins,ybins=np.histogram2d(parameterArrays[1], parameterArrays[4],bins=30,normed=LogNorm())
    ax12.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
  
    # column 3
    ax6.hist(parameterArrays[2], bins = 30, histtype='stepfilled', color = 'white')
      
    counts,xbins,ybins=np.histogram2d(parameterArrays[2], parameterArrays[3],bins=30,normed=LogNorm())
    ax9.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
  
    counts,xbins,ybins=np.histogram2d(parameterArrays[2], parameterArrays[4],bins=30,normed=LogNorm())
    ax13.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
      
    # column 4
    ax10.hist(parameterArrays[3], bins = 30, histtype='stepfilled', color = 'white')
  
    counts,xbins,ybins=np.histogram2d(parameterArrays[3], parameterArrays[4],bins=30,normed=LogNorm())
    ax14.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
  
    # column 5
    ax15.hist(parameterArrays[4], bins = 30, histtype='stepfilled', color = 'white')
  
    
    ax1.axvline(MaximumProbabilityValues[0])
    ax2.axvline(MaximumProbabilityValues[0], c = 'r')
    ax4.axvline(MaximumProbabilityValues[0], c = 'r')
    ax7.axvline(MaximumProbabilityValues[0], c = 'r')
    ax11.axvline(MaximumProbabilityValues[0], c = 'r')
    ax3.axvline(MaximumProbabilityValues[1])
    ax5.axvline(MaximumProbabilityValues[1], c = 'r')
    ax8.axvline(MaximumProbabilityValues[1], c = 'r')
    ax12.axvline(MaximumProbabilityValues[1], c = 'r')
    ax6.axvline(MaximumProbabilityValues[2])
    ax9.axvline(MaximumProbabilityValues[2], c = 'r')
    ax13.axvline(MaximumProbabilityValues[2], c = 'r')
    ax10.axvline(MaximumProbabilityValues[3])
    ax14.axvline(MaximumProbabilityValues[3], c = 'r')
    ax15.axvline(MaximumProbabilityValues[4])

    ax11.axhline(MaximumProbabilityValues[4], c = 'r')
    ax12.axhline(MaximumProbabilityValues[4], c = 'r')
    ax13.axhline(MaximumProbabilityValues[4], c = 'r')
    ax14.axhline(MaximumProbabilityValues[4], c = 'r')
    ax7.axhline(MaximumProbabilityValues[3], c = 'r')
    ax8.axhline(MaximumProbabilityValues[3], c = 'r')
    ax9.axhline(MaximumProbabilityValues[3], c = 'r')
    ax4.axhline(MaximumProbabilityValues[2], c = 'r')
    ax5.axhline(MaximumProbabilityValues[2], c = 'r')
    ax2.axhline(MaximumProbabilityValues[1], c = 'r')
      
    ax1.axvline(MaximumProbabilityValues[0]+MaximumProbabilityValueErrors[0], linestyle = '--')
    ax1.axvline(MaximumProbabilityValues[0]-MaximumProbabilityValueErrors[1], linestyle = '--')
    ax3.axvline(MaximumProbabilityValues[1]+MaximumProbabilityValueErrors[2], linestyle = '--')
    ax3.axvline(MaximumProbabilityValues[1]-MaximumProbabilityValueErrors[3], linestyle = '--')
    ax6.axvline(MaximumProbabilityValues[2]+MaximumProbabilityValueErrors[4], linestyle = '--')
    ax6.axvline(MaximumProbabilityValues[2]-MaximumProbabilityValueErrors[5], linestyle = '--')
    ax10.axvline(MaximumProbabilityValues[3]+MaximumProbabilityValueErrors[6], linestyle = '--')
    ax10.axvline(MaximumProbabilityValues[3]-MaximumProbabilityValueErrors[7], linestyle = '--')
    ax15.axvline(MaximumProbabilityValues[4]+MaximumProbabilityValueErrors[8], linestyle = '--')
    ax15.axvline(MaximumProbabilityValues[4]-MaximumProbabilityValueErrors[9], linestyle = '--')
      
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    ax4.set_xticklabels([])
    ax5.set_xticklabels([])
    ax6.set_xticklabels([])
    ax7.set_xticklabels([])
    ax8.set_xticklabels([])
    ax9.set_xticklabels([])
    ax10.set_xticklabels([])
      
    ax1.set_yticklabels([])
    ax3.set_yticklabels([])
    ax5.set_yticklabels([])
    ax6.set_yticklabels([])
    ax8.set_yticklabels([])
    ax9.set_yticklabels([])
    ax10.set_yticklabels([])
    ax12.set_yticklabels([])
    ax13.set_yticklabels([])
    ax14.set_yticklabels([])
    ax15.set_yticklabels([])
     
    ax1.set_xlim(IncRange)
    ax2.set_xlim(IncRange)
    ax3.set_xlim(BetaRange)
    ax4.set_xlim(IncRange)
    ax5.set_xlim(BetaRange)
    ax6.set_xlim(RhoRange)
    ax7.set_xlim(IncRange)
    ax8.set_xlim(BetaRange)
    ax9.set_xlim(RhoRange)
    ax10.set_xlim(GammaRange)
    ax11.set_xlim(IncRange)
    ax12.set_xlim(BetaRange)
    ax13.set_xlim(RhoRange)
    ax14.set_xlim(GammaRange)
    ax15.set_xlim(MlRange)
      
    ax2.set_ylim(BetaRange)
    ax4.set_ylim(RhoRange)
    ax5.set_ylim(RhoRange)
    ax7.set_ylim(GammaRange)
    ax8.set_ylim(GammaRange)
    ax9.set_ylim(GammaRange)
    ax11.set_ylim(MlRange)
    ax12.set_ylim(MlRange)
    ax13.set_ylim(MlRange)
    ax14.set_ylim(MlRange)
      
    ax2.set_ylabel(r"$\beta$", fontsize = 16)
    ax4.set_ylabel(r"$\log_{10}\rho _{s} (M_{\odot}{\rm pc}^{-3})$", fontsize = 12)
    ax7.set_ylabel('$\gamma_{tot}$', fontsize = 16)
    ax11.set_ylabel('M/L', fontsize = 12)
      
    ax11.set_xlabel('$i$', fontsize = 16)
    ax12.set_xlabel(r"$\beta$", fontsize = 16)
    ax13.set_xlabel(r"$\log_{10}\rho _{s} (M_{\odot}{\rm pc}^{-3})$", fontsize = 12)
    ax14.set_xlabel('$\gamma_{tot}$', fontsize = 16)
    ax15.set_xlabel('M/L', fontsize = 12)
      
    ax1.set_title('$i = '+str(round(MaximumProbabilityValues[0], 2))+\
      '^{+'+str(round(MaximumProbabilityValueErrors[0], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[1], 2))+'}$', fontsize = 12)
    ax3.set_title(r'$\beta = '+str(round(MaximumProbabilityValues[1], 2))+\
      '^{+'+str(round(MaximumProbabilityValueErrors[2], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[3], 2))+'}$', fontsize = 12)
    ax6.set_title(r'$\log_{10}\rho _{s} = '+str(round(MaximumProbabilityValues[2], 2))+\
      '^{+'+str(round(MaximumProbabilityValueErrors[4], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[5], 2))+'}$', fontsize = 12)
    ax10.set_title('$\gamma_{tot} = '+str(round(MaximumProbabilityValues[3], 2))+\
      '^{+'+str(round(MaximumProbabilityValueErrors[6], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[7], 2))+'}$', fontsize = 12)
    ax15.set_title('M/L$ = '+str(round(MaximumProbabilityValues[4], 2))+\
      '^{+'+str(round(MaximumProbabilityValueErrors[8], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[9], 2))+'}$', fontsize = 12)
    
    ax3.text(-0.5, 3, 'NGC '+GalName.split('C')[1], transform=ax6.transAxes,
          fontsize=16, fontweight='bold', va='top', ha='left')
    
    ax3.text(-0.5, 2.8, 'total mass distribution: '+str(totalMassDistribution), transform=ax6.transAxes,
          fontsize=14, fontweight='bold', va='top', ha='left')
    ax3.text(-0.5, 2.6, r'$\chi^2$/DOF: '+str(round(chi2, 2)), transform=ax6.transAxes,
          fontsize=14, fontweight='bold', va='top', ha='left')
    plt.savefig(filename)

def customCorner_ExtraWeights(GalName, parameterArrays, MaximumProbabilityValues, MaximumProbabilityValueErrors, filename, totalMassDistribution):
     
     
    IncRange = [np.min(parameterArrays[0]), np.max(parameterArrays[0])]
    BetaRange = [np.min(parameterArrays[1]), np.max(parameterArrays[1])]
    GammaRange = [np.min(parameterArrays[2]), np.max(parameterArrays[2])]
    SluggsWeightRange = [np.min(parameterArrays[3]), np.max(parameterArrays[3])]
    AtlasWeightRange = [np.min(parameterArrays[4]), np.max(parameterArrays[4])]
     
    # making my own corner plot that looks better. 
    gs1 = gridspec.GridSpec(6, 6)
    gs1.update(wspace = 0.04, hspace = 0.03)
     
     
    fig = plt.figure(figsize = (10, 10)) 
     
    ax1 =  plt.subplot(gs1[0, 0])
    ax2 =  plt.subplot(gs1[1, 0])
    plt.yticks(rotation=30)
    ax3 =  plt.subplot(gs1[1, 1])
    ax4 =  plt.subplot(gs1[2, 0])
    plt.yticks(rotation=30)
    ax5 =  plt.subplot(gs1[2, 1])
    ax6 =  plt.subplot(gs1[2, 2])
    ax7 =  plt.subplot(gs1[3, 0])
    plt.yticks(rotation=30)
    ax8 =  plt.subplot(gs1[3, 1])
    ax9 =  plt.subplot(gs1[3, 2])
    ax10 = plt.subplot(gs1[3, 3])
    ax11 = plt.subplot(gs1[4, 0])
    plt.yticks(rotation=30)
    ax12 = plt.subplot(gs1[4, 1])
    ax13 = plt.subplot(gs1[4, 2])
    ax14 = plt.subplot(gs1[4, 3])
    ax15 = plt.subplot(gs1[4, 4])
    
     
    # column 1
    ax1.hist(parameterArrays[0], bins = 30, histtype='stepfilled', color = 'white')
     
    counts,xbins,ybins=np.histogram2d(parameterArrays[0], parameterArrays[1],bins=30,normed=LogNorm())
    ax2.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
     
    counts,xbins,ybins=np.histogram2d(parameterArrays[0], parameterArrays[2],bins=30,normed=LogNorm())
    ax4.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
     
    counts,xbins,ybins=np.histogram2d(parameterArrays[0], parameterArrays[3],bins=30,normed=LogNorm())
    ax7.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
 
    counts,xbins,ybins=np.histogram2d(parameterArrays[0], parameterArrays[4],bins=30,normed=LogNorm())
    ax11.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
     
    
    # column 2
    ax3.hist(parameterArrays[1], bins = 30, histtype='stepfilled', color = 'white')
     
    counts,xbins,ybins=np.histogram2d(parameterArrays[1], parameterArrays[2],bins=30,normed=LogNorm())
    ax5.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
     
    counts,xbins,ybins=np.histogram2d(parameterArrays[1], parameterArrays[3],bins=30,normed=LogNorm())
    ax8.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
 
    counts,xbins,ybins=np.histogram2d(parameterArrays[1], parameterArrays[4],bins=30,normed=LogNorm())
    ax12.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
     
    # column 3
    ax6.hist(parameterArrays[2], bins = 30, histtype='stepfilled', color = 'white')
     
    counts,xbins,ybins=np.histogram2d(parameterArrays[2], parameterArrays[3],bins=30,normed=LogNorm())
    ax9.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
 
    counts,xbins,ybins=np.histogram2d(parameterArrays[2], parameterArrays[4],bins=30,normed=LogNorm())
    ax13.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
    
    # column 4
    ax10.hist(parameterArrays[3], bins = 30, histtype='stepfilled', color = 'white')
 
    counts,xbins,ybins=np.histogram2d(parameterArrays[3], parameterArrays[4],bins=30,normed=LogNorm())
    ax14.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
 
    # column 5
    ax15.hist(parameterArrays[4], bins = 30, histtype='stepfilled', color = 'white')
 
  
    ax1.axvline(MaximumProbabilityValues[0])
    ax2.axvline(MaximumProbabilityValues[0], c = 'r')
    ax4.axvline(MaximumProbabilityValues[0], c = 'r')
    ax7.axvline(MaximumProbabilityValues[0], c = 'r')
    ax11.axvline(MaximumProbabilityValues[0], c = 'r')
    ax3.axvline(MaximumProbabilityValues[1])
    ax5.axvline(MaximumProbabilityValues[1], c = 'r')
    ax8.axvline(MaximumProbabilityValues[1], c = 'r')
    ax12.axvline(MaximumProbabilityValues[1], c = 'r')
    ax6.axvline(MaximumProbabilityValues[2])
    ax9.axvline(MaximumProbabilityValues[2], c = 'r')
    ax13.axvline(MaximumProbabilityValues[2], c = 'r')
    ax10.axvline(MaximumProbabilityValues[3])
    ax14.axvline(MaximumProbabilityValues[3], c = 'r')
    ax15.axvline(MaximumProbabilityValues[4])
     
    ax11.axhline(MaximumProbabilityValues[4], c = 'r')
    ax12.axhline(MaximumProbabilityValues[4], c = 'r')
    ax13.axhline(MaximumProbabilityValues[4], c = 'r')
    ax14.axhline(MaximumProbabilityValues[4], c = 'r')
    ax7.axhline(MaximumProbabilityValues[3], c = 'r')
    ax8.axhline(MaximumProbabilityValues[3], c = 'r')
    ax9.axhline(MaximumProbabilityValues[3], c = 'r')
    ax4.axhline(MaximumProbabilityValues[2], c = 'r')
    ax5.axhline(MaximumProbabilityValues[2], c = 'r')
    ax2.axhline(MaximumProbabilityValues[1], c = 'r')
     
    ax1.axvline(MaximumProbabilityValues[0]+MaximumProbabilityValueErrors[0], linestyle = '--')
    ax1.axvline(MaximumProbabilityValues[0]-MaximumProbabilityValueErrors[1], linestyle = '--')
    ax3.axvline(MaximumProbabilityValues[1]+MaximumProbabilityValueErrors[2], linestyle = '--')
    ax3.axvline(MaximumProbabilityValues[1]-MaximumProbabilityValueErrors[3], linestyle = '--')
    ax6.axvline(MaximumProbabilityValues[2]+MaximumProbabilityValueErrors[4], linestyle = '--')
    ax6.axvline(MaximumProbabilityValues[2]-MaximumProbabilityValueErrors[5], linestyle = '--')
    ax10.axvline(MaximumProbabilityValues[3]+MaximumProbabilityValueErrors[6], linestyle = '--')
    ax10.axvline(MaximumProbabilityValues[3]-MaximumProbabilityValueErrors[7], linestyle = '--')
    ax15.axvline(MaximumProbabilityValues[4]+MaximumProbabilityValueErrors[8], linestyle = '--')
    ax15.axvline(MaximumProbabilityValues[4]-MaximumProbabilityValueErrors[9], linestyle = '--')
     
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    ax4.set_xticklabels([])
    ax5.set_xticklabels([])
    ax6.set_xticklabels([])
    ax7.set_xticklabels([])
    ax8.set_xticklabels([])
    ax9.set_xticklabels([])
    ax10.set_xticklabels([])
    ax11.set_xticklabels([])
    ax12.set_xticklabels([])
    ax13.set_xticklabels([])
    ax14.set_xticklabels([])
    ax15.set_xticklabels([])
    # ax10.set_xticklabels([])
     
    ax1.set_yticklabels([])
    ax3.set_yticklabels([])
    ax5.set_yticklabels([])
    ax6.set_yticklabels([])
    ax8.set_yticklabels([])
    ax9.set_yticklabels([])
    ax10.set_yticklabels([])
    ax12.set_yticklabels([])
    ax13.set_yticklabels([])
    ax14.set_yticklabels([])
    ax15.set_yticklabels([])
     
    ax1.set_xlim(IncRange)
    ax2.set_xlim(IncRange)
    ax3.set_xlim(BetaRange)
    ax4.set_xlim(IncRange)
    ax5.set_xlim(BetaRange)
    ax6.set_xlim(GammaRange)
    ax7.set_xlim(IncRange)
    ax8.set_xlim(BetaRange)
    ax9.set_xlim(GammaRange)
    ax10.set_xlim(SluggsWeightRange)
    ax11.set_xlim(IncRange)
    ax12.set_xlim(BetaRange)
    ax13.set_xlim(GammaRange)
    ax14.set_xlim(SluggsWeightRange)
    ax15.set_xlim(AtlasWeightRange)
     
    ax2.set_ylim(BetaRange)
    ax4.set_ylim(GammaRange)
    ax5.set_ylim(GammaRange)
    ax7.set_ylim(SluggsWeightRange)
    ax8.set_ylim(SluggsWeightRange)
    ax9.set_ylim(SluggsWeightRange)
    ax11.set_ylim(AtlasWeightRange)
    ax12.set_ylim(AtlasWeightRange)
    ax13.set_ylim(AtlasWeightRange)
    ax14.set_ylim(AtlasWeightRange)
     
    ax2.set_ylabel(r"$\beta$", fontsize = 16)
    ax4.set_ylabel('$\gamma_{tot}$', fontsize = 16)
    ax7.set_ylabel('SLUGGS Weight', fontsize = 12)
    ax11.set_ylabel('ATLAS Weight', fontsize = 12)
     
    ax11.set_xlabel('$i$', fontsize = 16)
    ax12.set_xlabel(r"$\beta$", fontsize = 16)
    ax13.set_xlabel('$\gamma_{tot}$', fontsize = 16)
    ax14.set_xlabel('SLUGGS Weight', fontsize = 12)
    ax15.set_xlabel('ATLAS Weight', fontsize = 12)
     
    ax1.set_title('$i = '+str(round(MaximumProbabilityValues[0], 2))+'^{+'+str(round(MaximumProbabilityValueErrors[0], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[1], 2))+'}$', fontsize = 12)
    ax3.set_title(r'$\beta = '+str(round(MaximumProbabilityValues[1], 2))+'^{+'+str(round(MaximumProbabilityValueErrors[2], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[3], 2))+'}$', fontsize = 12)
    ax6.set_title('$\gamma_{tot} = '+str(round(MaximumProbabilityValues[2], 2))+'^{+'+str(round(MaximumProbabilityValueErrors[4], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[5], 2))+'}$', fontsize = 12)
    ax10.set_title('SW$ = '+str(round(MaximumProbabilityValues[3], 2))+'^{+'+str(round(MaximumProbabilityValueErrors[6], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[7], 2))+'}$', fontsize = 12)
    ax15.set_title('AW$ = '+str(round(MaximumProbabilityValues[4], 2))+'^{+'+str(round(MaximumProbabilityValueErrors[8], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[9], 2))+'}$', fontsize = 12)
     
    ax3.text(-0.5, 3, 'NGC '+GalName.split('C')[1], transform=ax6.transAxes,
          fontsize=16, fontweight='bold', va='top', ha='left')
     
    ax3.text(-0.5, 2.8, 'total mass distribution: '+str(totalMassDistribution), transform=ax6.transAxes,
          fontsize=14, fontweight='bold', va='top', ha='left')
    plt.savefig(filename)

def customCorner_Kinematics(GalName, parameterArrays, MaximumProbabilityValues, MaximumProbabilityValueErrors, filename, totalMassDistribution, figure, chi2):
    IncRange = [np.min(parameterArrays[0]), np.max(parameterArrays[0])]
    BetaRange = [np.min(parameterArrays[1]), np.max(parameterArrays[1])]
    RhoRange = [np.min(parameterArrays[2]), np.max(parameterArrays[2])]
    GammaRange = [np.min(parameterArrays[3]), np.max(parameterArrays[3])]
    SluggsWeightRange = [np.min(parameterArrays[4]), np.max(parameterArrays[4])]
    AtlasWeightRange = [np.min(parameterArrays[5]), np.max(parameterArrays[5])]
      
    # making my own corner plot that looks better. 
    gs1 = gridspec.GridSpec(6, 6)
    gs1.update(wspace = 0.04, hspace = 0.03)
      
      
    fig = plt.figure(figsize = (10, 10)) 
      
    ax1 =  plt.subplot(gs1[0, 0])
    ax2 =  plt.subplot(gs1[1, 0])
    plt.yticks(rotation=30)
    ax3 =  plt.subplot(gs1[1, 1])
    ax4 =  plt.subplot(gs1[2, 0])
    plt.yticks(rotation=30)
    ax5 =  plt.subplot(gs1[2, 1])
    ax6 =  plt.subplot(gs1[2, 2])
    ax7 =  plt.subplot(gs1[3, 0])
    plt.yticks(rotation=30)
    ax8 =  plt.subplot(gs1[3, 1])
    ax9 =  plt.subplot(gs1[3, 2])
    ax10 = plt.subplot(gs1[3, 3])
    ax11 = plt.subplot(gs1[4, 0])
    plt.yticks(rotation=30)
    ax12 = plt.subplot(gs1[4, 1])
    ax13 = plt.subplot(gs1[4, 2])
    ax14 = plt.subplot(gs1[4, 3])
    ax15 = plt.subplot(gs1[4, 4])
    ax16 = plt.subplot(gs1[5, 0])
    plt.yticks(rotation=30)
    plt.xticks(rotation=30)
    ax17 = plt.subplot(gs1[5, 1])
    plt.xticks(rotation=30)
    ax18 = plt.subplot(gs1[5, 2])
    plt.xticks(rotation=30)
    ax19 = plt.subplot(gs1[5, 3])
    plt.xticks(rotation=30)
    ax20 = plt.subplot(gs1[5, 4])
    plt.xticks(rotation=30)
    ax21 = plt.subplot(gs1[5, 5])
    plt.xticks(rotation=30)

    ax22 = plt.subplot(222, aspect = 'equal') # an additional subplot for the kinematic fit diagram. 
    datafile = cbook.get_sample_data(figure)
    img = imread(datafile)
    # img = np.flipud(imread(datafile))
    # ax2 = fig.add_subplot(312, aspect = 'equal')
    ax22.imshow(img, zorder=0, interpolation = 'None')
    ax22.set_xticks([])
    ax22.set_yticks([])
    ax22.set_xticklabels([])
    ax22.set_yticklabels([])
    ax22.axis('off')
      
    # column 1
    ax1.hist(parameterArrays[0], bins = 30, histtype='stepfilled', color = 'white')
      
    counts,xbins,ybins=np.histogram2d(parameterArrays[0], parameterArrays[1],bins=30,normed=LogNorm())
    ax2.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
      
    counts,xbins,ybins=np.histogram2d(parameterArrays[0], parameterArrays[2],bins=30,normed=LogNorm())
    ax4.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
      
    counts,xbins,ybins=np.histogram2d(parameterArrays[0], parameterArrays[3],bins=30,normed=LogNorm())
    ax7.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
  
    counts,xbins,ybins=np.histogram2d(parameterArrays[0], parameterArrays[4],bins=30,normed=LogNorm())
    ax11.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
      
    counts,xbins,ybins=np.histogram2d(parameterArrays[0], parameterArrays[5],bins=30,normed=LogNorm())
    ax16.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
      
    # column 2
    ax3.hist(parameterArrays[1], bins = 30, histtype='stepfilled', color = 'white')
      
    counts,xbins,ybins=np.histogram2d(parameterArrays[1], parameterArrays[2],bins=30,normed=LogNorm())
    ax5.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
      
    counts,xbins,ybins=np.histogram2d(parameterArrays[1], parameterArrays[3],bins=30,normed=LogNorm())
    ax8.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
  
    counts,xbins,ybins=np.histogram2d(parameterArrays[1], parameterArrays[4],bins=30,normed=LogNorm())
    ax12.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
      
    counts,xbins,ybins=np.histogram2d(parameterArrays[1], parameterArrays[5],bins=30,normed=LogNorm())
    ax17.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
  
  
    # column 3
    ax6.hist(parameterArrays[2], bins = 30, histtype='stepfilled', color = 'white')
      
    counts,xbins,ybins=np.histogram2d(parameterArrays[2], parameterArrays[3],bins=30,normed=LogNorm())
    ax9.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
  
    counts,xbins,ybins=np.histogram2d(parameterArrays[2], parameterArrays[4],bins=30,normed=LogNorm())
    ax13.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
  
    counts,xbins,ybins=np.histogram2d(parameterArrays[2], parameterArrays[5],bins=30,normed=LogNorm())
    ax18.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
      
    # column 4
    ax10.hist(parameterArrays[3], bins = 30, histtype='stepfilled', color = 'white')
  
    counts,xbins,ybins=np.histogram2d(parameterArrays[3], parameterArrays[4],bins=30,normed=LogNorm())
    ax14.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
  
    counts,xbins,ybins=np.histogram2d(parameterArrays[3], parameterArrays[5],bins=30,normed=LogNorm())
    ax19.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
  
  
    # column 5
    ax15.hist(parameterArrays[4], bins = 30, histtype='stepfilled', color = 'white')
  
    counts,xbins,ybins=np.histogram2d(parameterArrays[4], parameterArrays[5],bins=30,normed=LogNorm())
    ax20.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
  
    # column 6
    ax21.hist(parameterArrays[5], bins = 30, histtype='stepfilled', color = 'white')
  
  
  
      
    ax1.axvline(MaximumProbabilityValues[0])
    ax2.axvline(MaximumProbabilityValues[0], c = 'r')
    ax4.axvline(MaximumProbabilityValues[0], c = 'r')
    ax7.axvline(MaximumProbabilityValues[0], c = 'r')
    ax11.axvline(MaximumProbabilityValues[0], c = 'r')
    ax16.axvline(MaximumProbabilityValues[0], c = 'r')
    ax3.axvline(MaximumProbabilityValues[1])
    ax5.axvline(MaximumProbabilityValues[1], c = 'r')
    ax8.axvline(MaximumProbabilityValues[1], c = 'r')
    ax12.axvline(MaximumProbabilityValues[1], c = 'r')
    ax17.axvline(MaximumProbabilityValues[1], c = 'r')
    ax6.axvline(MaximumProbabilityValues[2])
    ax9.axvline(MaximumProbabilityValues[2], c = 'r')
    ax13.axvline(MaximumProbabilityValues[2], c = 'r')
    ax18.axvline(MaximumProbabilityValues[2], c = 'r')
    ax10.axvline(MaximumProbabilityValues[3])
    ax14.axvline(MaximumProbabilityValues[3], c = 'r')
    ax19.axvline(MaximumProbabilityValues[3], c = 'r')
    ax15.axvline(MaximumProbabilityValues[4])
    ax20.axvline(MaximumProbabilityValues[4], c = 'r')
    ax21.axvline(MaximumProbabilityValues[5])
      
    ax16.axhline(MaximumProbabilityValues[5], c = 'r')
    ax17.axhline(MaximumProbabilityValues[5], c = 'r')
    ax18.axhline(MaximumProbabilityValues[5], c = 'r')
    ax19.axhline(MaximumProbabilityValues[5], c = 'r')
    ax20.axhline(MaximumProbabilityValues[5], c = 'r')
    ax11.axhline(MaximumProbabilityValues[4], c = 'r')
    ax12.axhline(MaximumProbabilityValues[4], c = 'r')
    ax13.axhline(MaximumProbabilityValues[4], c = 'r')
    ax14.axhline(MaximumProbabilityValues[4], c = 'r')
    ax7.axhline(MaximumProbabilityValues[3], c = 'r')
    ax8.axhline(MaximumProbabilityValues[3], c = 'r')
    ax9.axhline(MaximumProbabilityValues[3], c = 'r')
    ax4.axhline(MaximumProbabilityValues[2], c = 'r')
    ax5.axhline(MaximumProbabilityValues[2], c = 'r')
    ax2.axhline(MaximumProbabilityValues[1], c = 'r')
      
    ax1.axvline(MaximumProbabilityValues[0]+MaximumProbabilityValueErrors[0], linestyle = '--')
    ax1.axvline(MaximumProbabilityValues[0]-MaximumProbabilityValueErrors[1], linestyle = '--')
    ax3.axvline(MaximumProbabilityValues[1]+MaximumProbabilityValueErrors[2], linestyle = '--')
    ax3.axvline(MaximumProbabilityValues[1]-MaximumProbabilityValueErrors[3], linestyle = '--')
    ax6.axvline(MaximumProbabilityValues[2]+MaximumProbabilityValueErrors[4], linestyle = '--')
    ax6.axvline(MaximumProbabilityValues[2]-MaximumProbabilityValueErrors[5], linestyle = '--')
    ax10.axvline(MaximumProbabilityValues[3]+MaximumProbabilityValueErrors[6], linestyle = '--')
    ax10.axvline(MaximumProbabilityValues[3]-MaximumProbabilityValueErrors[7], linestyle = '--')
    ax15.axvline(MaximumProbabilityValues[4]+MaximumProbabilityValueErrors[8], linestyle = '--')
    ax15.axvline(MaximumProbabilityValues[4]-MaximumProbabilityValueErrors[9], linestyle = '--')
    ax21.axvline(MaximumProbabilityValues[5]+MaximumProbabilityValueErrors[10], linestyle = '--')
    ax21.axvline(MaximumProbabilityValues[5]-MaximumProbabilityValueErrors[11], linestyle = '--')
      
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    ax4.set_xticklabels([])
    ax5.set_xticklabels([])
    ax6.set_xticklabels([])
    ax7.set_xticklabels([])
    ax8.set_xticklabels([])
    ax9.set_xticklabels([])
    ax10.set_xticklabels([])
    ax11.set_xticklabels([])
    ax12.set_xticklabels([])
    ax13.set_xticklabels([])
    ax14.set_xticklabels([])
    ax15.set_xticklabels([])
    # ax10.set_xticklabels([])
      
    ax1.set_yticklabels([])
    ax3.set_yticklabels([])
    ax5.set_yticklabels([])
    ax6.set_yticklabels([])
    ax8.set_yticklabels([])
    ax9.set_yticklabels([])
    ax10.set_yticklabels([])
    ax12.set_yticklabels([])
    ax13.set_yticklabels([])
    ax14.set_yticklabels([])
    ax15.set_yticklabels([])
    ax17.set_yticklabels([])
    ax18.set_yticklabels([])
    ax19.set_yticklabels([])
    ax20.set_yticklabels([])
    ax21.set_yticklabels([])
      
    ax1.set_xlim(IncRange)
    ax2.set_xlim(IncRange)
    ax3.set_xlim(BetaRange)
    ax4.set_xlim(IncRange)
    ax5.set_xlim(BetaRange)
    ax6.set_xlim(RhoRange)
    ax7.set_xlim(IncRange)
    ax8.set_xlim(BetaRange)
    ax9.set_xlim(RhoRange)
    ax10.set_xlim(GammaRange)
    ax11.set_xlim(IncRange)
    ax12.set_xlim(BetaRange)
    ax13.set_xlim(RhoRange)
    ax14.set_xlim(GammaRange)
    ax15.set_xlim(SluggsWeightRange)
    ax16.set_xlim(IncRange)
    ax17.set_xlim(BetaRange)
    ax18.set_xlim(RhoRange)
    ax19.set_xlim(GammaRange)
    ax20.set_xlim(SluggsWeightRange)
    ax21.set_xlim(AtlasWeightRange)
      
    ax2.set_ylim(BetaRange)
    ax4.set_ylim(RhoRange)
    ax5.set_ylim(RhoRange)
    ax7.set_ylim(GammaRange)
    ax8.set_ylim(GammaRange)
    ax9.set_ylim(GammaRange)
    ax11.set_ylim(SluggsWeightRange)
    ax12.set_ylim(SluggsWeightRange)
    ax13.set_ylim(SluggsWeightRange)
    ax14.set_ylim(SluggsWeightRange)
    ax16.set_ylim(AtlasWeightRange)
    ax17.set_ylim(AtlasWeightRange)
    ax18.set_ylim(AtlasWeightRange)
    ax19.set_ylim(AtlasWeightRange)
    ax20.set_ylim(AtlasWeightRange)
      
    ax2.set_ylabel(r"$\beta$", fontsize = 16)
    ax4.set_ylabel(r"$\log_{10}\rho _{s} (M_{\odot}{\rm pc}^{-3})$", fontsize = 12)
    ax7.set_ylabel('$\gamma_{tot}$', fontsize = 16)
    ax11.set_ylabel('SLUGGS Weight', fontsize = 12)
    ax16.set_ylabel('ATLAS Weight', fontsize = 12)
      
    ax16.set_xlabel('$i$', fontsize = 16)
    ax17.set_xlabel(r"$\beta$", fontsize = 16)
    ax18.set_xlabel(r"$\log_{10}\rho _{s} (M_{\odot}{\rm pc}^{-3})$", fontsize = 12)
    ax19.set_xlabel('$\gamma_{tot}$', fontsize = 16)
    ax20.set_xlabel('SLUGGS Weight', fontsize = 12)
    ax21.set_xlabel('ATLAS Weight', fontsize = 12)
      
    ax1.set_title('$i = '+str(round(MaximumProbabilityValues[0], 2))+\
      '^{+'+str(round(MaximumProbabilityValueErrors[0], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[1], 2))+'}$', fontsize = 12)
    ax3.set_title(r'$\beta = '+str(round(MaximumProbabilityValues[1], 2))+\
      '^{+'+str(round(MaximumProbabilityValueErrors[2], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[3], 2))+'}$', fontsize = 12)
    ax6.set_title(r'$\log_{10}\rho _{s} = '+str(round(MaximumProbabilityValues[2], 2))+\
      '^{+'+str(round(MaximumProbabilityValueErrors[4], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[5], 2))+'}$', fontsize = 12)
    ax10.set_title('$\gamma_{tot} = '+str(round(MaximumProbabilityValues[3], 2))+\
      '^{+'+str(round(MaximumProbabilityValueErrors[6], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[7], 2))+'}$', fontsize = 12)
    ax15.set_title('SW$ = '+str(round(MaximumProbabilityValues[4], 2))+\
      '^{+'+str(round(MaximumProbabilityValueErrors[8], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[9], 2))+'}$', fontsize = 12)
    ax21.set_title('AW$ = '+str(round(MaximumProbabilityValues[5], 2))+\
      '^{+'+str(round(MaximumProbabilityValueErrors[10], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[11], 2))+'}$', fontsize = 12)
      
    ax3.text(-0.5, 3.5, 'NGC '+GalName.split('C')[1], transform=ax6.transAxes,
          fontsize=16, fontweight='bold', va='top', ha='left')
      
    ax3.text(-0.5, 3.3, 'Total mass distribution: '+str(totalMassDistribution), transform=ax6.transAxes,
          fontsize=14, fontweight='bold', va='top', ha='left')
    ax3.text(-0.5, 3.1, r'$\chi^2$/DOF: '+str(round(chi2, 2)), transform=ax6.transAxes,
          fontsize=14, fontweight='bold', va='top', ha='left')
    plt.savefig(filename)

def customCorner_Kinematics_ml(GalName, parameterArrays, MaximumProbabilityValues, MaximumProbabilityValueErrors, filename, totalMassDistribution, figure, chi2):
    IncRange = [np.min(parameterArrays[0]), np.max(parameterArrays[0])]
    BetaRange = [np.min(parameterArrays[1]), np.max(parameterArrays[1])]
    RhoRange = [np.min(parameterArrays[2]), np.max(parameterArrays[2])]
    GammaRange = [np.min(parameterArrays[3]), np.max(parameterArrays[3])]
    MlRange = [np.min(parameterArrays[4]), np.max(parameterArrays[4])]
    SluggsWeightRange = [np.min(parameterArrays[5]), np.max(parameterArrays[5])]
    AtlasWeightRange = [np.min(parameterArrays[6]), np.max(parameterArrays[6])]
      
    # making my own corner plot that looks better. 
    gs1 = gridspec.GridSpec(7, 7)
    gs1.update(wspace = 0.04, hspace = 0.03)
      
      
    fig = plt.figure(figsize = (10, 10)) 
      
    ax1 =  plt.subplot(gs1[0, 0])
    ax2 =  plt.subplot(gs1[1, 0])
    plt.yticks(rotation=30)
    ax3 =  plt.subplot(gs1[1, 1])
    ax4 =  plt.subplot(gs1[2, 0])
    plt.yticks(rotation=30)
    ax5 =  plt.subplot(gs1[2, 1])
    ax6 =  plt.subplot(gs1[2, 2])
    ax7 =  plt.subplot(gs1[3, 0])
    plt.yticks(rotation=30)
    ax8 =  plt.subplot(gs1[3, 1])
    ax9 =  plt.subplot(gs1[3, 2])
    ax10 = plt.subplot(gs1[3, 3])
    ax11 = plt.subplot(gs1[4, 0])
    plt.yticks(rotation=30)
    ax12 = plt.subplot(gs1[4, 1])
    ax13 = plt.subplot(gs1[4, 2])
    ax14 = plt.subplot(gs1[4, 3])
    ax15 = plt.subplot(gs1[4, 4])
    ax16 = plt.subplot(gs1[5, 0])
    plt.yticks(rotation=30)
    ax17 = plt.subplot(gs1[5, 1])
    ax18 = plt.subplot(gs1[5, 2])
    ax19 = plt.subplot(gs1[5, 3])
    ax20 = plt.subplot(gs1[5, 4])
    ax21 = plt.subplot(gs1[5, 5])
    ax22 = plt.subplot(gs1[6, 0])
    plt.yticks(rotation=30)
    plt.xticks(rotation=30)
    ax23 = plt.subplot(gs1[6, 1])
    plt.xticks(rotation=30)
    ax24 = plt.subplot(gs1[6, 2])
    plt.xticks(rotation=30)
    ax25 = plt.subplot(gs1[6, 3])
    plt.xticks(rotation=30)
    ax26 = plt.subplot(gs1[6, 4])
    plt.xticks(rotation=30)
    ax27 = plt.subplot(gs1[6, 5])
    plt.xticks(rotation=30)
    ax28 = plt.subplot(gs1[6, 6])
    plt.xticks(rotation=30)

    ax29 = plt.subplot(233, aspect = 'equal') # an additional subplot for the kinematic fit diagram. 
    datafile = cbook.get_sample_data(figure)
    img = imread(datafile)
    # img = np.flipud(imread(datafile))
    # ax2 = fig.add_subplot(312, aspect = 'equal')
    ax29.imshow(img, zorder=0, interpolation = 'None')
    ax29.set_xticks([])
    ax29.set_yticks([])
    ax29.set_xticklabels([])
    ax29.set_yticklabels([])
    ax29.axis('off')
      
    # column 1
    ax1.hist(parameterArrays[0], bins = 30, histtype='stepfilled', color = 'white')
      
    counts,xbins,ybins=np.histogram2d(parameterArrays[0], parameterArrays[1],bins=30,normed=LogNorm())
    ax2.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
      
    counts,xbins,ybins=np.histogram2d(parameterArrays[0], parameterArrays[2],bins=30,normed=LogNorm())
    ax4.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
      
    counts,xbins,ybins=np.histogram2d(parameterArrays[0], parameterArrays[3],bins=30,normed=LogNorm())
    ax7.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
  
    counts,xbins,ybins=np.histogram2d(parameterArrays[0], parameterArrays[4],bins=30,normed=LogNorm())
    ax11.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
      
    counts,xbins,ybins=np.histogram2d(parameterArrays[0], parameterArrays[5],bins=30,normed=LogNorm())
    ax16.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')

    counts,xbins,ybins=np.histogram2d(parameterArrays[0], parameterArrays[6],bins=30,normed=LogNorm())
    ax22.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
      
    # column 2
    ax3.hist(parameterArrays[1], bins = 30, histtype='stepfilled', color = 'white')
      
    counts,xbins,ybins=np.histogram2d(parameterArrays[1], parameterArrays[2],bins=30,normed=LogNorm())
    ax5.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
      
    counts,xbins,ybins=np.histogram2d(parameterArrays[1], parameterArrays[3],bins=30,normed=LogNorm())
    ax8.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
  
    counts,xbins,ybins=np.histogram2d(parameterArrays[1], parameterArrays[4],bins=30,normed=LogNorm())
    ax12.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
      
    counts,xbins,ybins=np.histogram2d(parameterArrays[1], parameterArrays[5],bins=30,normed=LogNorm())
    ax17.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')

    counts,xbins,ybins=np.histogram2d(parameterArrays[1], parameterArrays[6],bins=30,normed=LogNorm())
    ax23.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
  
  
    # column 3
    ax6.hist(parameterArrays[2], bins = 30, histtype='stepfilled', color = 'white')
      
    counts,xbins,ybins=np.histogram2d(parameterArrays[2], parameterArrays[3],bins=30,normed=LogNorm())
    ax9.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
  
    counts,xbins,ybins=np.histogram2d(parameterArrays[2], parameterArrays[4],bins=30,normed=LogNorm())
    ax13.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
  
    counts,xbins,ybins=np.histogram2d(parameterArrays[2], parameterArrays[5],bins=30,normed=LogNorm())
    ax18.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')

    counts,xbins,ybins=np.histogram2d(parameterArrays[2], parameterArrays[6],bins=30,normed=LogNorm())
    ax24.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
      
    # column 4
    ax10.hist(parameterArrays[3], bins = 30, histtype='stepfilled', color = 'white')
  
    counts,xbins,ybins=np.histogram2d(parameterArrays[3], parameterArrays[4],bins=30,normed=LogNorm())
    ax14.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
  
    counts,xbins,ybins=np.histogram2d(parameterArrays[3], parameterArrays[5],bins=30,normed=LogNorm())
    ax19.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')

    counts,xbins,ybins=np.histogram2d(parameterArrays[3], parameterArrays[6],bins=30,normed=LogNorm())
    ax25.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
  
  
    # column 5
    ax15.hist(parameterArrays[4], bins = 30, histtype='stepfilled', color = 'white')
  
    counts,xbins,ybins=np.histogram2d(parameterArrays[4], parameterArrays[5],bins=30,normed=LogNorm())
    ax20.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')

    counts,xbins,ybins=np.histogram2d(parameterArrays[4], parameterArrays[6],bins=30,normed=LogNorm())
    ax26.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
  
    # column 6
    ax21.hist(parameterArrays[5], bins = 30, histtype='stepfilled', color = 'white')

    counts,xbins,ybins=np.histogram2d(parameterArrays[5], parameterArrays[6],bins=30,normed=LogNorm())
    ax27.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
        ybins.min(),ybins.max()],linewidths=2,
        linestyles='solid')
  
    # column 7
    ax28.hist(parameterArrays[6], bins = 30, histtype='stepfilled', color = 'white') 
  
      
    ax1.axvline(MaximumProbabilityValues[0])
    ax2.axvline(MaximumProbabilityValues[0], c = 'r')
    ax4.axvline(MaximumProbabilityValues[0], c = 'r')
    ax7.axvline(MaximumProbabilityValues[0], c = 'r')
    ax11.axvline(MaximumProbabilityValues[0], c = 'r')
    ax16.axvline(MaximumProbabilityValues[0], c = 'r')
    ax22.axvline(MaximumProbabilityValues[0], c = 'r')
    ax3.axvline(MaximumProbabilityValues[1])
    ax5.axvline(MaximumProbabilityValues[1], c = 'r')
    ax8.axvline(MaximumProbabilityValues[1], c = 'r')
    ax12.axvline(MaximumProbabilityValues[1], c = 'r')
    ax17.axvline(MaximumProbabilityValues[1], c = 'r')
    ax23.axvline(MaximumProbabilityValues[1], c = 'r')
    ax6.axvline(MaximumProbabilityValues[2])
    ax9.axvline(MaximumProbabilityValues[2], c = 'r')
    ax13.axvline(MaximumProbabilityValues[2], c = 'r')
    ax18.axvline(MaximumProbabilityValues[2], c = 'r')
    ax24.axvline(MaximumProbabilityValues[2], c = 'r')
    ax10.axvline(MaximumProbabilityValues[3])
    ax14.axvline(MaximumProbabilityValues[3], c = 'r')
    ax19.axvline(MaximumProbabilityValues[3], c = 'r')
    ax25.axvline(MaximumProbabilityValues[3], c = 'r')
    ax15.axvline(MaximumProbabilityValues[4])
    ax20.axvline(MaximumProbabilityValues[4], c = 'r')
    ax26.axvline(MaximumProbabilityValues[4], c = 'r')
    ax21.axvline(MaximumProbabilityValues[5])
    ax27.axvline(MaximumProbabilityValues[5], c = 'r')
    ax28.axvline(MaximumProbabilityValues[6])

    ax22.axhline(MaximumProbabilityValues[6], c = 'r')
    ax23.axhline(MaximumProbabilityValues[6], c = 'r')
    ax24.axhline(MaximumProbabilityValues[6], c = 'r')
    ax25.axhline(MaximumProbabilityValues[6], c = 'r')
    ax26.axhline(MaximumProbabilityValues[6], c = 'r')
    ax27.axhline(MaximumProbabilityValues[6], c = 'r')          
    ax16.axhline(MaximumProbabilityValues[5], c = 'r')
    ax17.axhline(MaximumProbabilityValues[5], c = 'r')
    ax18.axhline(MaximumProbabilityValues[5], c = 'r')
    ax19.axhline(MaximumProbabilityValues[5], c = 'r')
    ax20.axhline(MaximumProbabilityValues[5], c = 'r')
    ax11.axhline(MaximumProbabilityValues[4], c = 'r')
    ax12.axhline(MaximumProbabilityValues[4], c = 'r')
    ax13.axhline(MaximumProbabilityValues[4], c = 'r')
    ax14.axhline(MaximumProbabilityValues[4], c = 'r')
    ax7.axhline(MaximumProbabilityValues[3], c = 'r')
    ax8.axhline(MaximumProbabilityValues[3], c = 'r')
    ax9.axhline(MaximumProbabilityValues[3], c = 'r')
    ax4.axhline(MaximumProbabilityValues[2], c = 'r')
    ax5.axhline(MaximumProbabilityValues[2], c = 'r')
    ax2.axhline(MaximumProbabilityValues[1], c = 'r')
      
    ax1.axvline(MaximumProbabilityValues[0]+MaximumProbabilityValueErrors[0], linestyle = '--')
    ax1.axvline(MaximumProbabilityValues[0]-MaximumProbabilityValueErrors[1], linestyle = '--')
    ax3.axvline(MaximumProbabilityValues[1]+MaximumProbabilityValueErrors[2], linestyle = '--')
    ax3.axvline(MaximumProbabilityValues[1]-MaximumProbabilityValueErrors[3], linestyle = '--')
    ax6.axvline(MaximumProbabilityValues[2]+MaximumProbabilityValueErrors[4], linestyle = '--')
    ax6.axvline(MaximumProbabilityValues[2]-MaximumProbabilityValueErrors[5], linestyle = '--')
    ax10.axvline(MaximumProbabilityValues[3]+MaximumProbabilityValueErrors[6], linestyle = '--')
    ax10.axvline(MaximumProbabilityValues[3]-MaximumProbabilityValueErrors[7], linestyle = '--')
    ax15.axvline(MaximumProbabilityValues[4]+MaximumProbabilityValueErrors[8], linestyle = '--')
    ax15.axvline(MaximumProbabilityValues[4]-MaximumProbabilityValueErrors[9], linestyle = '--')
    ax21.axvline(MaximumProbabilityValues[5]+MaximumProbabilityValueErrors[10], linestyle = '--')
    ax21.axvline(MaximumProbabilityValues[5]-MaximumProbabilityValueErrors[11], linestyle = '--')
    ax28.axvline(MaximumProbabilityValues[6]+MaximumProbabilityValueErrors[12], linestyle = '--')
    ax28.axvline(MaximumProbabilityValues[6]-MaximumProbabilityValueErrors[13], linestyle = '--')
      
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    ax4.set_xticklabels([])
    ax5.set_xticklabels([])
    ax6.set_xticklabels([])
    ax7.set_xticklabels([])
    ax8.set_xticklabels([])
    ax9.set_xticklabels([])
    ax10.set_xticklabels([])
    ax11.set_xticklabels([])
    ax12.set_xticklabels([])
    ax13.set_xticklabels([])
    ax14.set_xticklabels([])
    ax15.set_xticklabels([])
    ax16.set_xticklabels([])
    ax17.set_xticklabels([])
    ax18.set_xticklabels([])
    ax19.set_xticklabels([])
    ax20.set_xticklabels([])
    ax21.set_xticklabels([])
      
    ax1.set_yticklabels([])
    ax3.set_yticklabels([])
    ax5.set_yticklabels([])
    ax6.set_yticklabels([])
    ax8.set_yticklabels([])
    ax9.set_yticklabels([])
    ax10.set_yticklabels([])
    ax12.set_yticklabels([])
    ax13.set_yticklabels([])
    ax14.set_yticklabels([])
    ax15.set_yticklabels([])
    ax17.set_yticklabels([])
    ax18.set_yticklabels([])
    ax19.set_yticklabels([])
    ax20.set_yticklabels([])
    ax21.set_yticklabels([])
    ax23.set_yticklabels([])
    ax24.set_yticklabels([])
    ax25.set_yticklabels([])
    ax26.set_yticklabels([])
    ax27.set_yticklabels([])
    ax28.set_yticklabels([])
      
    ax1.set_xlim(IncRange)
    ax2.set_xlim(IncRange)
    ax3.set_xlim(BetaRange)
    ax4.set_xlim(IncRange)
    ax5.set_xlim(BetaRange)
    ax6.set_xlim(RhoRange)
    ax7.set_xlim(IncRange)
    ax8.set_xlim(BetaRange)
    ax9.set_xlim(RhoRange)
    ax10.set_xlim(GammaRange)
    ax11.set_xlim(IncRange)
    ax12.set_xlim(BetaRange)
    ax13.set_xlim(RhoRange)
    ax14.set_xlim(GammaRange)
    ax15.set_xlim(MlRange)
    ax16.set_xlim(IncRange)
    ax17.set_xlim(BetaRange)
    ax18.set_xlim(RhoRange)
    ax19.set_xlim(GammaRange)
    ax20.set_xlim(MlRange)
    ax21.set_xlim(SluggsWeightRange)
    ax22.set_xlim(IncRange)
    ax23.set_xlim(BetaRange)
    ax24.set_xlim(RhoRange)
    ax25.set_xlim(GammaRange)
    ax26.set_xlim(MlRange)
    ax27.set_xlim(SluggsWeightRange)
    ax28.set_xlim(AtlasWeightRange)
      
    ax2.set_ylim(BetaRange)
    ax4.set_ylim(RhoRange)
    ax5.set_ylim(RhoRange)
    ax7.set_ylim(GammaRange)
    ax8.set_ylim(GammaRange)
    ax9.set_ylim(GammaRange)
    ax11.set_ylim(MlRange)
    ax12.set_ylim(MlRange)
    ax13.set_ylim(MlRange)
    ax14.set_ylim(MlRange)
    ax16.set_ylim(SluggsWeightRange)
    ax17.set_ylim(SluggsWeightRange)
    ax18.set_ylim(SluggsWeightRange)
    ax19.set_ylim(SluggsWeightRange)
    ax20.set_ylim(SluggsWeightRange)
    ax22.set_ylim(AtlasWeightRange)
    ax23.set_ylim(AtlasWeightRange)
    ax24.set_ylim(AtlasWeightRange)
    ax25.set_ylim(AtlasWeightRange)
    ax26.set_ylim(AtlasWeightRange)
    ax27.set_ylim(AtlasWeightRange)
      
    ax2.set_ylabel(r"$\beta$", fontsize = 16)
    ax4.set_ylabel(r"$\log_{10}\rho _{s} (M_{\odot}{\rm pc}^{-3})$", fontsize = 12)
    ax7.set_ylabel('$\gamma_{tot}$', fontsize = 16)
    ax11.set_ylabel('M/L', fontsize = 12)
    ax16.set_ylabel('SLUGGS Weight', fontsize = 12)
    ax22.set_ylabel('ATLAS Weight', fontsize = 12)
      
    ax22.set_xlabel('$i$', fontsize = 16)
    ax23.set_xlabel(r"$\beta$", fontsize = 16)
    ax24.set_xlabel(r"$\log_{10}\rho _{s} (M_{\odot}{\rm pc}^{-3})$", fontsize = 12)
    ax25.set_xlabel('$\gamma_{tot}$', fontsize = 16)
    ax26.set_xlabel('M/L', fontsize = 12)
    ax27.set_xlabel('SLUGGS Weight', fontsize = 12)
    ax28.set_xlabel('ATLAS Weight', fontsize = 12)
      
    ax1.set_title('$i = '+str(round(MaximumProbabilityValues[0], 2))+\
      '^{+'+str(round(MaximumProbabilityValueErrors[0], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[1], 2))+'}$', fontsize = 12)
    ax3.set_title(r'$\beta = '+str(round(MaximumProbabilityValues[1], 2))+\
      '^{+'+str(round(MaximumProbabilityValueErrors[2], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[3], 2))+'}$', fontsize = 12)
    ax6.set_title(r'$\log_{10}\rho _{s} = '+str(round(MaximumProbabilityValues[2], 2))+\
      '^{+'+str(round(MaximumProbabilityValueErrors[4], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[5], 2))+'}$', fontsize = 12)
    ax10.set_title('$\gamma_{tot} = '+str(round(MaximumProbabilityValues[3], 2))+\
      '^{+'+str(round(MaximumProbabilityValueErrors[6], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[7], 2))+'}$', fontsize = 12)
    ax15.set_title('M/L$ = '+str(round(MaximumProbabilityValues[4], 2))+\
      '^{+'+str(round(MaximumProbabilityValueErrors[8], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[9], 2))+'}$', fontsize = 12)
    ax21.set_title('SW$ = '+str(round(MaximumProbabilityValues[5], 2))+\
      '^{+'+str(round(MaximumProbabilityValueErrors[10], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[11], 2))+'}$', fontsize = 12)
    ax28.set_title('AW$ = '+str(round(MaximumProbabilityValues[6], 2))+\
      '^{+'+str(round(MaximumProbabilityValueErrors[12], 2))+'}_{-'+str(round(MaximumProbabilityValueErrors[13], 2))+'}$', fontsize = 12)
      
    ax3.text(-0.5, 3.5, 'NGC '+GalName.split('C')[1], transform=ax6.transAxes,
          fontsize=16, fontweight='bold', va='top', ha='left')
      
    ax3.text(-0.5, 3.3, 'Total mass distribution: '+str(totalMassDistribution), transform=ax6.transAxes,
          fontsize=14, fontweight='bold', va='top', ha='left')
    ax3.text(-0.5, 3.1, r'$\chi^2$/DOF: '+str(round(chi2, 2)), transform=ax6.transAxes,
          fontsize=14, fontweight='bold', va='top', ha='left')

    fig.subplots_adjust(hspace = 0.0, wspace = 0.0)
    plt.savefig(filename)

def powerLawFit(Radius_array, fitGaussian, MinRadius, MaxRadius, SamplingNumber=250, 
        slopeLowerLog=-4.0, slopeUpperLog=-0.5):
  # r_gaussian = np.logspace(np.log10(0.01), np.log10(3000), RadialNumber) # sampled points in radius space
  a = np.logspace(np.log10(1e-1), np.log10(1e8), SamplingNumber) # y-intercept range and sampling
  k = np.linspace(slopeLowerLog, slopeUpperLog, SamplingNumber) # slope range and sampling

  # identifying the gaussian points closest to the minimum and maximum radial range
  MinGaussian, MaxGaussian = 0, 0
  for ii in range(len(Radius_array)):
    if (Radius_array[ii] > MinRadius) & (MinGaussian == 0):
      MinGaussian = ii
    if (Radius_array[ii] < MaxRadius):
      MaxGaussian = ii

  
  chi2_min=1e20
  a_min, k_min = 0, 0
  for ii in range(len(a)):
    for jj in range(len(k)):
      # plot the power law for the specific parameters
      y_power = a[ii]*Radius_array**(k[jj])
      #calculate the chi^2 for this distribution of plots. 
      chi2=0
      for kk in range(MinGaussian, MaxGaussian): # the ranges here determine over which radial range the power law should be fitted. 
        # print y_power[kk]
        # print fitGaussian[kk]
        chi2 += (np.log10(y_power[kk])-np.log10(fitGaussian[kk]))**2
      if chi2 < chi2_min:
        chi2_min = chi2
        a_min = a[ii]
        k_min = k[jj]
  power_law = a_min * Radius_array**(k_min)
  return [power_law, k_min]

def ParameterRangeSelection(parameterArray, parameterMax, GalName, parameterName, xScale='lin'):
  parameterArray = np.array(parameterArray)
  parameterSort = parameterArray[np.argsort(parameterArray)]
  for ii in range(len(parameterSort)):
    if parameterSort[ii] == parameterMax:
      medianArg = ii
  
  if (medianArg > int(len(parameterSort)*0.34)) & (medianArg + int(len(parameterSort)*0.34) < len(parameterSort)):
    lowerArg = medianArg-int(len(parameterSort)*0.34)
    upperArg = medianArg+int(len(parameterSort)*0.34)
    # print 'even'
  elif medianArg + int(len(parameterSort)*0.34) > len(parameterSort):
    upperArg = len(parameterSort)
    lowerArg = len(parameterSort) - int(len(parameterSort)*0.68)
    # print 'uneven upper'
  else:
    lowerArg = 0
    upperArg = int(len(parameterSort)*0.68)
    # print 'uneven lower'
  
  parameterSelected = []
  for ii in range(lowerArg, upperArg):
    parameterSelected.append(parameterSort[ii])
  if xScale == 'lin':
    fig=plt.figure(figsize=(6, 4))
    ax1=fig.add_subplot(111)
    n, bins, patches = ax1.hist(parameterSelected, 50, facecolor='green')
    ax1.set_ylabel('Number')
    ax1.set_title(parameterName)
    ymin, ymax = ax1.get_ylim()
    ax1.plot((parameterMax, parameterMax), (ymin, ymax), 'r-')
    plt.savefig('../Output/'+str(GalName)+'/'+str(GalName)+parameterName+'MedianDistribution.pdf')
    plt.close()
  elif xScale == 'log':
    for ii in range(len(parameterSelected)):
      parameterSelected[ii] = 10**parameterSelected[ii]
    fig=plt.figure(figsize=(6, 4))
    ax1=fig.add_subplot(111)
    n, bins, patches = ax1.hist(parameterSelected, 50, facecolor='green')
    ax1.set_ylabel('Number')
    ax1.set_title(parameterName)
    ax1.set_xscale("log")
    ymin, ymax = ax1.get_ylim()
    ax1.plot((10**parameterMax, 10**parameterMax), (ymin, ymax), 'r-')
    plt.savefig('../Output/'+str(GalName)+'/'+str(GalName)+parameterName+'MedianDistribution.pdf')
    plt.close()
  return parameterSelected

def ProbabilityPlot(ParameterArray, ProbabilityArray, GalName, parameterMax, ParameterName='Parameter'):
  fig=plt.figure(figsize=(8, 6))
  ax1=fig.add_subplot(211)
  ax1.scatter(ParameterArray, ProbabilityArray, marker='o', s=1, edgecolor='None')
  ax1.set_ylabel('ln(Likelihood)')
  # ax1.set_xticks([])
  ymin, ymax = ax1.get_ylim()
  ax1.plot((parameterMax, parameterMax), (ymin, ymax), 'r-')
  ax2=fig.add_subplot(212, sharex=ax1)
  n, bins, patches = ax2.hist(ParameterArray, 50, facecolor='green')
  ax2.set_ylabel('Number')
  ax2.set_xlabel(ParameterName)
  ymin, ymax = ax2.get_ylim()
  ax2.plot((parameterMax, parameterMax), (ymin, ymax), 'r-')
  plt.subplots_adjust(hspace = 0.15, wspace=0.2)
  plt.savefig('../Output/'+str(GalName)+'/'+str(GalName)+ParameterName+'_ProbabilityDistribution.pdf')
  plt.close()

# plot the cumulative dark matter fraction with radius
def CumulativeDMFraction(Radius, Fraction_DM, GalName):
  fig=plt.figure()
  ax1=fig.add_subplot(111)
  ax1.errorbar(np.log10(Radius), Fraction_DM, None, None, c='r', label='DM fraction', linestyle = '-', color = 'r')
  ax1.set_xlabel('log(radius) [arcsec]')
  ax1.set_ylabel('DM fraction')
  handles, labels=ax1.get_legend_handles_labels()
  ax1.legend(handles, labels, loc=2, fontsize='small', scatterpoints=1) 
  
  plt.title(str(GalName)+' DM fraction ')
  plt.savefig('../Output/'+str(GalName)+'/'+str(GalName)+'DMFraction.pdf')
  plt.close()
  return

def MGEfinder(MGE, GalName, units = 'original'):
  DropboxDirectory = os.getcwd().split('Dropbox')[0]
  MGE_path = DropboxDirectory+'Dropbox/PhD_Analysis/Analysis/JAM Modelling/mge_parameters'

  if GalName == 'NGC0821':
      GalName = 'NGC821'

  def ArrayExtractorArcsec(filename):
    file=open(filename, 'r')
    lines=file.readlines()
    file.close()
    # need to calculate a conversion factor between L/pc^2 and L/"^2. 
    try:
     Conversion = ((distMpc[GalName]*10**6) / (206265))**2
    except:
      Conversion = ((distMpc_ATLAS[GalName]*10**6) / (206265))**2
 
    surf, sigma, qobs = [], [], []
     
    i=0
    for line in lines:
        if i != 0 and line[0] != '#':
            surf.append(10**float(line.split()[0]) * Conversion)
            sigma.append(10**float(line.split()[1]))
            qobs.append(float(line.split()[2]))
            i += 1
        else:
            i += 1
    return np.array(surf), np.array(sigma), np.array(qobs)

  def ArrayExtractorKpc(filename):
    file=open(filename, 'r')
    lines=file.readlines()
    file.close()
    # need to calculate a conversion factor between L/pc^2 and L/kpc^2. 
    Conversion = 10**6

    surf, sigma, qobs = [], [], []
    
    i=0
    for line in lines:
        if i != 0 and line[0] != '#':
            surf.append(float(line.split()[0]) * Conversion)# converting this value from L/pc^2 to L/kpc^2. 
            sigma.append(arcsecToKpc(float(line.split()[1]), distMpc[GalName]*1000)) # converting from arcseconds to kpc
            qobs.append(float(line.split()[2]))
            i += 1
        else:
            i += 1
    return np.array(surf), np.array(sigma), np.array(qobs)

  def ArrayExtractorOriginal(filename):
    file=open(filename, 'r')
    lines=file.readlines()
    file.close()
    surf, sigma, qobs = [], [], []
    
    i=0
    for line in lines:
        if i != 0 and line[0] != '#':
            surf.append(float(line.split()[0]))
            sigma.append(float(line.split()[1]))
            qobs.append(float(line.split()[2]))
            i += 1
        else:
            i += 1
    return np.array(surf), np.array(sigma), np.array(qobs)

  if MGE == 'S13':
    filename = MGE_path+'/mge_parameters_Scott13/mge_'+GalName+'.txt'

  elif MGE == 'S09':
    filename = MGE_path+'/mge_parameters_Scott09/mge_'+GalName+'_Scott09.txt'

  elif MGE == 'C06':
    filename = MGE_path+'/mge_parameters_Cappellari06/mge_'+GalName+'_C06.txt'

  elif MGE == 'E99':
    filename = MGE_path+'/mge_parameters_Emsellem99/mge_'+GalName+'.txt'

  elif MGE == 'Sabine':
    filename = MGE_path+'/mge_parameters_Sabine/mge_'+GalName+'.txt'

  if units == 'arcsec':
    print 'MGE used with units of arcsec'
    return ArrayExtractorArcsec(filename)
  elif units == 'kpc':
    print 'MGE used with units of kpc'
    return ArrayExtractorKpc(filename)
  elif units == 'original':
    print 'MGE used with original units'
    return ArrayExtractorOriginal(filename)
