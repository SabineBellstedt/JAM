import numpy as np
import pickle, time, glob, sys, os
import matplotlib.pyplot as plt
import matplotlib as mpl
from progressbar import ProgressBar, Percentage, Bar
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

DropboxDirectory = os.getcwd().split('Dropbox')[0]
lib_path = os.path.abspath(DropboxDirectory+'Dropbox/PhD_Analysis/Library') 
sys.path.append(lib_path)
from JAM_Sabine_def import *
from JAM_Dictionary import *
from JAM_Analysis_def import *
from Sabine_Define import *
from galaxyParametersDictionary_v9 import *
lib_path = os.path.abspath(DropboxDirectory+'Dropbox/PhD_Analysis/Analysis/JAM_Modular/jam') 
sys.path.append(lib_path)
import jam_axi_rms as Jrms


JAM_path = DropboxDirectory+'Dropbox/PhD_Analysis/Analysis/JAM_Modular'
Input_path = 'InputFiles/Original'

model = 'gNFW'

Galaxies = ['NGC821', 'NGC1023', 'NGC2549', 'NGC2699', 'NGC2768', 'NGC2974', 'NGC3377', 'NGC3379', 'NGC4111'\
, 'NGC4278', 'NGC4459', 'NGC4473', 'NGC4474', 'NGC4494', 'NGC4551', 'NGC4526', 'NGC4649', 'NGC4697', 'NGC5866', 'NGC7457']

for GalName in Galaxies:
	MGE = MGE_Source[GalName]

	for SLUGGS in [True, False]:
		for ATLAS in [True, False]:
			for GC in [True, False]:
				for ndim in np.arange(5, 9, 1):
					suffix = ''
  					if SLUGGS:
  					  suffix = suffix + 'SLUGGS_'
  					if ATLAS:
  					  suffix = suffix + 'ATLAS_'
  					if GC:
  					  suffix = suffix + 'GC_'
  					suffix = suffix + 'FreeParam-'+str(ndim)

  					# test to see if an output exists of this configuration.
  					ParameterFilename = GalName+'/'+suffix+'/'+GalName+'_'+suffix+'_parameters.txt' 
  					if os.path.isfile(ParameterFilename):
  						# now we can run the code

						if SLUGGS:
							input_filename_sluggs = JAM_path+'/'+Input_path+'/'+str(GalName)+'/JAM_Input_'+GalName+'_SLUGGS.txt'
							xbin_sluggs, ybin_sluggs, rms_sluggs, erms_sluggs = np.loadtxt(input_filename_sluggs, unpack = True, usecols = [0, 1, 2, 3], comments = '#')
						if ATLAS:
							input_filename_atlas = JAM_path+'/'+Input_path+'/'+str(GalName)+'/JAM_Input_'+GalName+'_ATLAS3D.txt'
							xbin_atlas, ybin_atlas, rms_atlas, erms_atlas = np.loadtxt(input_filename_atlas, unpack = True, usecols = [0, 1, 2, 3], comments = '#')
						if GC:
							input_filename_gc = JAM_path+'/'+Input_path+'/'+str(GalName)+'/JAM_Input_'+GalName+'_GC.txt'
							xbin_gc, ybin_gc, rms_gc, erms_gc = np.loadtxt(input_filename_gc, unpack = True, usecols = [0, 1, 2, 3], comments = '#')

						if not ATLAS and not GC:
							xbin = xbin_sluggs
							ybin =ybin_sluggs
							rms = rms_sluggs
							erms = erms_sluggs
						elif not GC:
							xbin = np.concatenate((xbin_sluggs, xbin_atlas))
							ybin = np.concatenate((ybin_sluggs, ybin_atlas))
							rms = np.concatenate((rms_sluggs, rms_atlas))
							erms = np.concatenate((erms_sluggs, erms_atlas))
						else:
							xbin = np.concatenate((xbin_sluggs, xbin_atlas, xbin_gc))
							ybin = np.concatenate((ybin_sluggs, ybin_atlas, ybin_gc))
							rms = np.concatenate((rms_sluggs, rms_atlas, rms_gc))
							erms = np.concatenate((erms_sluggs, erms_atlas, erms_gc))
						
						sigmapsf = 0.6
						pixsize = 0.8
						distance = distMpc[GalName]
						R_S = 20000.
						Reff = Reff_Spitzer[GalName] * ((distance * 1e6) / 206265)
						try:
							mbh = BlackHoleMass[GalName]
						except:
							mbh = 0
							print 'No black hole mass specified for', GalName

						# finding the MGE parameters based on the literature source determined in the dictionary file
						surf, sigma, qobs = MGEfinder(MGE, GalName, units = 'original')
						
						surf_lum, sigma_lum, qobs_lum = surf.copy(), sigma.copy(), qobs.copy()
						sigma_lum_pc = sigma_lum  * (distance * 1e6) / 206265 # since the input sigma_lum is in arcseconds. 
						
						

						# now reading in the best-fitting free parameters
						ParameterNames = np.loadtxt(ParameterFilename, comments = '#', unpack = True, usecols = [0], dtype = str)
						Value, LowerErr, UpperErr = np.loadtxt(ParameterFilename, comments = '#', unpack = True, usecols = [1, 2, 3])
						
						# automatically allocate parameter names based on what is in the parameter file
						if 'Inclination' in ParameterNames:
							inc = Value[np.where(ParameterNames == 'Inclination')]
							incUpper = UpperErr[np.where(ParameterNames == 'Inclination')]
							incLower = LowerErr[np.where(ParameterNames == 'Inclination')]
						else:
							raise ValueError('Inclination not set as a free parameter. ')

						if 'Beta' in ParameterNames:
							beta = Value[np.where(ParameterNames == 'Beta')]
							betaUpper = UpperErr[np.where(ParameterNames == 'Beta')]
							betaLower = LowerErr[np.where(ParameterNames == 'Beta')]
						else:
							raise ValueError('Beta not set as a free parameter. ')

						if 'ScaleDensity' in ParameterNames:
							log_rho_s = Value[np.where(ParameterNames == 'ScaleDensity')]
							log_rho_sUpper = UpperErr[np.where(ParameterNames == 'ScaleDensity')]
							log_rho_sLower = LowerErr[np.where(ParameterNames == 'ScaleDensity')]
						else:
							raise ValueError('Scale density not set as a free parameter. ')

						if 'Gamma' in ParameterNames:
							gamma = Value[np.where(ParameterNames == 'Gamma')]
							gammaUpper = UpperErr[np.where(ParameterNames == 'Gamma')]
							gammaLower = LowerErr[np.where(ParameterNames == 'Gamma')]
						else:
							raise ValueError('Gamma not set as a free parameter. ')

						if 'ML' in ParameterNames:
							ml = Value[np.where(ParameterNames == 'ML')][0]
							mlUpper = UpperErr[np.where(ParameterNames == 'ML')]
							mlLower = LowerErr[np.where(ParameterNames == 'ML')]
						else:
							ml = 1

						n_MGE = 300.
						x_MGE = np.logspace(np.log10(1), np.log10(30000), n_MGE) # logarithmically spaced radii

						beta_array = np.ones(len(surf_lum))*beta
						
						if model == 'powLaw':
							y_MGE = powerLaw(x_MGE, 10.**log_rho_s, R_S, gamma, -3) 
						elif model == 'gNFW':
							y_MGE = gNFW_RelocatedDensity(R_S/20, x_MGE, 10.**log_rho_s, R_S, gamma)
						
						p = mge.mge_fit_1d(x_MGE, y_MGE, ngauss=16, quiet=True, plot=True) # the mge_fit_1d function will produce a projected, 2d distribution 
																	   # from the analytic 1d function, so that it is the appropriate format
																	   # as an input for JAM. 

						if model == 'powLaw':
							surf_tot = p.sol[0]                                # here implementing a potential
							sigma_tot = p.sol[1]                               # configuration which includes the 
							sigma_tot_arcsec = p.sol[1]  * 206265 / (distance * 1e6) 
							qobs_tot = np.ones(len(surf_tot))  
						elif model == 'gNFW':
							surf_dm = p.sol[0]                                # here implementing a potential
							sigma_dm = p.sol[1]                               # configuration which includes the 
							qobs_dm = np.ones(len(surf_dm))                   # stellar mass and also a dark matter    
							                                                  # halo in the form of a gNFW distribution. 
						
						 	surf_tot = np.append(ml * surf_lum, surf_dm)    #    
						 	sigma_tot_arcsec = np.append(sigma_lum, sigma_dm * 206265 / (distance * 1e6) )
							sigma_tot = np.append(sigma_lum, sigma_dm) #                                                   
							qobs_tot = np.append(qobs_lum, qobs_dm)    #  


						jamFigureFilename = JAM_path+'/'+str(GalName)+'/'+suffix+'/'+str(GalName)+'_'+suffix+'_Fig'

						rmsModel, ml, chi2, flux = Jrms.jam_axi_rms(
						             surf_lum, sigma_lum, qobs_lum, surf_tot,
						             sigma_tot_arcsec, qobs_tot,
						             inc, mbh, distance,
						             xbin, ybin, jamFigureFilename, #MassToLightRatio,
						             rms=rms, erms=erms,
						              plot=True, beta=beta_array, 
						             tensor='zz', quiet=False)
						plt.close()  


						Radius_MassProfile = np.logspace(np.log10(10), np.log10(30000), 100) 

						dataRadiusMaximum = np.sqrt(np.max(xbin)**2 + np.max(ybin)**2) * (distance * 1e6) / 206265 # the value needs to be converted from arcseconds into parsec

						if model == 'gNFW':
							sigma_dm_pc = sigma_dm  # with the current unit system, the MGE dispersion is already in pc. 
							sigma_tot_pc = sigma_tot  # with the current unit system, the MGE dispersion is already in pc. 

							DM_mass_Re = MGE_mass_total(surf_dm, sigma_dm_pc, qobs_dm, Reff)
							StellarMass_Re = MGE_mass_total(surf_lum, sigma_lum_pc, qobs_lum, Reff, radian(inc), ml)
							TotalMass_Re = DM_mass_Re + StellarMass_Re

							fractionDM = DM_mass_Re / TotalMass_Re

							# I need to calculate the total density profile here. 
							TotalStellarDensityProfile  = ml * MGE_deprojector(Radius_MassProfile, surf_lum, sigma_lum_pc, qobs_lum, radian(inc), ml)
							DeprojectedDMProfile = MGE_deprojector(Radius_MassProfile, surf_dm, sigma_dm_pc, qobs_dm)
							TotalDensityProfile = TotalStellarDensityProfile + DeprojectedDMProfile

						
							# StellarMass_gNFW = MGE_mass_total(surf_lum, sigma_lum_pc, qobs_lum, 100*Reff, radian(Inc_gNFW), ML_gNFW)
							# TotalMass_gNFW= StellarMass_gNFW + MGE_mass_total(surf_gNFW_dm, sigma_gNFW_dm, qobs_gNFW_dm, 100*Reff)
							


							# now calculating the associated profiles at the limits of the error range:
							n_MGE = 300.
							x_MGE = np.logspace(np.log10(10), np.log10(30000), n_MGE) # range sampled in pc. 
							y_MGE = gNFW_RelocatedDensity(R_S/20, x_MGE, 10.**log_rho_s, R_S, gamma+gammaUpper)	
							try:
								p = mge.mge_fit_1d(x_MGE, y_MGE, ngauss=16 , quiet=True, plot=True)		
							except:
								p = mge.mge_fit_1d(x_MGE, y_MGE, ngauss=5 , quiet=True, plot=True)	
							surf_dm_upper = p.sol[0]                               
							sigma_dm_upper = p.sol[1]                              
							qobs_dm_upper = np.ones(len(surf_dm_upper)) 
							TotalDensityProfile_ErrorAnalysisUpper = ml * MGE_deprojector(Radius_MassProfile, surf_lum, sigma_lum_pc, qobs_lum, radian(inc - incLower), ml)\
							 + MGE_deprojector(Radius_MassProfile, surf_dm_upper, sigma_dm_upper, qobs_dm_upper)
						
							y_MGE = gNFW_RelocatedDensity(R_S/20, x_MGE, 10.**log_rho_s, R_S, gamma-gammaLower)	
							try:
								p = mge.mge_fit_1d(x_MGE, y_MGE, ngauss=16 , quiet=True, plot=True)		
							except:
								p = mge.mge_fit_1d(x_MGE, y_MGE, ngauss=5 , quiet=True, plot=True)		
							surf_dm_lower = p.sol[0]                               
							sigma_dm_lower = p.sol[1]                              
							qobs_dm_lower = np.ones(len(surf_dm_lower)) 
							TotalDensityProfile_ErrorAnalysisLower = ml * MGE_deprojector(Radius_MassProfile, surf_lum, sigma_lum_pc, qobs_lum, radian(inc + incUpper), ml)\
							 + MGE_deprojector(Radius_MassProfile, surf_dm_lower, sigma_dm_lower, qobs_dm_lower)
						
						
							TotalStellarMass = np.sum(2*np.pi*ml * surf_lum*qobs_lum*(sigma_lum_pc)**2)

							fig=plt.figure(figsize=(5, 5))
							ax1=fig.add_subplot(111)
							from matplotlib.patches import Rectangle

							ax1.add_patch(Rectangle((1e1, -5.95), dataRadiusMaximum , (3.9+5.95), facecolor="#E6E6E6", edgecolor = 'None'))
							print 'maximum radius (pc)', dataRadiusMaximum
							# ax1.plot(Radius_MassProfile, TotalDensityProfile, color = 'r', label = 'Total (DM + stellar) deprojected', linewidth = 2.)

							FittedGamma1, FittedGamma1ErrUpper, FittedGamma1ErrLower, x_fitted, y_fitted  = \
								fittedgammaCalculator(0.1*Reff, 1.*Reff, Radius_MassProfile, TotalDensityProfile, \
								TotalDensityProfile_ErrorAnalysisUpper, TotalDensityProfile_ErrorAnalysisLower)
							# ax1.plot(x_fitted, y_fitted, color = 'g', linestyle = '--', dashes = [2, 2], linewidth = 3.)
						
							FittedGamma2, FittedGamma2ErrUpper, FittedGamma2ErrLower, x_fitted, y_fitted  = \
								fittedgammaCalculator(0, min(dataRadiusMaximum, 4*Reff), Radius_MassProfile, TotalDensityProfile, \
									TotalDensityProfile_ErrorAnalysisUpper, TotalDensityProfile_ErrorAnalysisLower)
							# ax1.plot(x_fitted, y_fitted, color = 'g', linestyle = '--', dashes = [2, 2], linewidth = 3.)
						
							FittedGamma3, FittedGamma3ErrUpper, FittedGamma3ErrLower, x_fitted, y_fitted  = \
								fittedgammaCalculator(0.1*Reff, min(dataRadiusMaximum, 4*Reff), Radius_MassProfile, TotalDensityProfile, \
								TotalDensityProfile_ErrorAnalysisUpper, TotalDensityProfile_ErrorAnalysisLower)
							# ax1.plot(x_fitted, y_fitted, color = 'orange', linestyle = '--', dashes = [3, 3], linewidth = 4.) # plotting the 0.1 - R_max line

							ax1.plot(Radius_MassProfile, np.log10(DeprojectedDMProfile), color = 'b', label = 'Model 2 DM', linewidth = 1.2)
							ax1.plot(Radius_MassProfile, np.log10(TotalStellarDensityProfile), color = 'orange', label = 'Model 2 Stellar Mass', linewidth = 1.2)
							ax1.plot(Radius_MassProfile, np.log10(TotalDensityProfile), color = 'k', label = 'Model 2 Total Mass', linestyle = '--', linewidth = 2.0, dashes = [6, 3, 3, 3], )
							
							# ax1.plot(Radius_MassProfile, DeprojectedDMProfile, color = 'k', label = 'Deprojected DM', linewidth = 1.)
							# ax1.plot(Radius_MassProfile, TotalStellarDensityProfile, color = 'b', label = 'Deprojected Stellar Brightness', linewidth = 2.)
						
							ax1.axvline(Reff, color = 'c') # plot the effective radius in pc
							ax1.text(Reff, np.max(TotalStellarDensityProfile)/10, '$R_e$', color = 'c', rotation = 'vertical')
							ax1.axvline(R_S, color = 'g')
							ax1.text(R_S, np.max(TotalStellarDensityProfile)/10, '$R_S$', color = 'g', rotation = 'vertical')
						
							ax1.set_xscale('log')
							# ax1.set_yscale('log')
							ax1.set_xlabel('Radius [pc]')

							ax1.set_ylabel(r'$\log(\rho)\,[\rm{M}_{\odot} / \rm{pc}^3]$')
							ax1.set_yticks([-4, -2, 0, 2, 4])	
							ax1.set_xlim([90, 30000])
							ax1.set_ylim([-5.95, 3.5])
	
							handles, labels = ax1.get_legend_handles_labels()
							ax1.legend(handles, labels, loc = 3, fontsize = 10, scatterpoints = 1)
							
							plt.subplots_adjust(left = 0.15)
							plt.savefig(JAM_path+'/'+str(GalName)+'/'+suffix+'/'+str(GalName)+'_'+suffix+'_MassProfile.pdf')
							plt.close()

							# saving all the extra values to file
							ParameterNames = ['Gamma 0.1-1', 'Gamma 0.0-4', 'Gamma 0.1-4', 'DM Fraction (<1Re)']
							FinalValues = [FittedGamma1, FittedGamma2, FittedGamma3, fractionDM]
							Lower = [FittedGamma1ErrLower, FittedGamma2ErrLower, FittedGamma3ErrLower, 0.0]
							Upper = [FittedGamma1ErrUpper, FittedGamma2ErrUpper, FittedGamma3ErrUpper, 0.0]
							X = np.zeros(np.array(ParameterNames).size, dtype=[('params', 'U22'), ('values', float), ('lower', float), ('upper', float)])
							X['params'] = np.array(ParameterNames)
							X['values'] = np.array(FinalValues)
							X['lower']  = np.array(Lower)
							X['upper']  = np.array(Upper)
							
							np.savetxt(JAM_path+'/'+str(GalName)+'/'+suffix+'/'+str(GalName)+'_'+suffix+'_AdditionalParameters.txt', \
							  X, fmt="%22s %10.3f %10.3f %10.3f", header = 'Parameter, Value, LowerErr, UpperErr')

					# else:
					# 	print 'no filename ', suffix
					