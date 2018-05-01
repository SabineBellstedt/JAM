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
