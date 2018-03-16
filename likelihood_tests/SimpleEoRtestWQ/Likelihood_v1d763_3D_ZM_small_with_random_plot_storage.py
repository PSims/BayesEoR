
###
# Imports
###
import matplotlib
matplotlib.use('pdf')

# use_MultiNest = False
use_MultiNest = True

import numpy as np
import numpy
from numpy import arange, shape, log10, pi
import scipy
from subprocess import os
import sys
# sys.path.append(os.path.expanduser('~/EoR/PolyChord1d9/PolyChord_WorkingInitSetup_Altered/'))
sys.path.append(os.path.expanduser('~/EoR/Python_Scripts/BayesEoR/SpatialPS/PolySpatialPS/'))
if use_MultiNest:
	from pymultinest.solve import solve
else:
	import PyPolyChord.PyPolyChord as PolyChord
import pylab
import time
from scipy.linalg import block_diag
from pprint import pprint

from Linalg_v1d1 import IDFT_Array_IDFT_2D_ZM_SH, makeGaussian, Produce_Full_Coordinate_Arrays, Produce_Coordinate_Arrays_ZM
from Linalg_v1d1 import Produce_Coordinate_Arrays_ZM_Coarse, Produce_Coordinate_Arrays_ZM_SH, Calc_Coords_High_Res_Im_to_Large_uv
from Linalg_v1d1 import Calc_Coords_Large_Im_to_High_Res_uv, Restore_Centre_Pixel, Calc_Indices_Centre_3x3_Grid
from Linalg_v1d1 import Delete_Centre_3x3_Grid, Delete_Centre_Pix, N_is_Odd, Calc_Indices_Centre_NxN_Grid, Obtain_Centre_NxN_Grid
from Linalg_v1d1 import Restore_Centre_3x3_Grid, Restore_Centre_NxN_Grid, Generate_Combined_Coarse_plus_Subharmic_uv_grids
from Linalg_v1d1 import IDFT_Array_IDFT_2D_ZM_SH, DFT_Array_DFT_2D, IDFT_Array_IDFT_2D, DFT_Array_DFT_2D_ZM, IDFT_Array_IDFT_2D_ZM
from Linalg_v1d1 import Construct_Hermitian, Construct_Hermitian_Gridding_Matrix, Construct_Hermitian_Gridding_Matrix_CosSin
from Linalg_v1d1 import Construct_Hermitian_Gridding_Matrix_CosSin_SH_v4, IDFT_Array_IDFT_1D, IDFT_Array_IDFT_1D
from Linalg_v1d1 import generate_gridding_matrix_vis_ordered_to_chan_ordered

from SimData_v1d4 import generate_test_sim_signal, map_out_bins_for_power_spectral_coefficients
from SimData_v1d4 import generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector
from SimData_v1d4 import generate_test_sim_signal_with_large_spectral_scales_1
from SimData_v1d4 import map_out_bins_for_power_spectral_coefficients_WQ_v2, generate_k_cube_in_physical_coordinates

from SimData_v1d4 import map_out_bins_for_power_spectral_coefficients_HERA_Binning
from SimData_v1d4 import map_out_bins_for_power_spectral_coefficients_WQ_v2_HERA_Binning
from SimData_v1d4 import generate_test_sim_signal_with_large_spectral_scales_1_HERA_Binning
from SimData_v1d4 import generate_test_sim_signal_with_large_spectral_scales_2_HERA_Binning
from SimData_v1d4 import generate_k_cube_in_physical_coordinates_21cmFAST
from SimData_v1d4 import generate_test_sim_signal_with_large_spectral_scales_2_21cmFAST_Binning
from SimData_v1d4 import GenerateForegroundCube
from SimData_v1d4 import generate_masked_coordinate_cubes, generate_k_cube_model_cylindrical_binning
from SimData_v1d4 import generate_k_cube_model_spherical_binning, construct_GRN_unitary_hermitian_k_cube

from Linalg_v1d1 import IDFT_Array_IDFT_1D_WQ, generate_gridding_matrix_vis_ordered_to_chan_ordered_WQ
from Linalg_v1d1 import IDFT_Array_IDFT_1D_WQ_ZM, generate_gridding_matrix_vis_ordered_to_chan_ordered_ZM

from Generate_matrix_stack_v1d1 import BuildMatrices

from Utils_v1d0 import PriorC, ParseCommandLineArguments, DataUnitConversionmkandJyperpix, WriteDataToFits
from Utils_v1d0 import ExtractDataFrom21cmFASTCube, plot_signal_vs_MLsignal_residuals

from GenerateForegroundCube_v1d0 import generate_Jelic_cube, generate_data_from_loaded_EoR_cube
from GenerateForegroundCube_v1d0 import generate_test_signal_from_image_cube

###
# Set analysis parameters
###
argv = sys.argv[1:]
PCLA = ParseCommandLineArguments()
nq = PCLA.parse(argv) #defaults to 2 (i.e. jointly fits for a second order quadratic)
# nq = 2
# nq = 0

nf=38
neta=38
neta = neta -nq
	
nu=3
nv=3
nx=3
ny=3

sigma=20.e-1
# sigma=80.e-1
#sigma=100.e-1
sub_quad = False

small_cube = nu<=7 and nv<=7
nuv = (nu*nv-1)
Show=False
current_file_version = 'Likelihood_v1d76_3D_ZM'
array_save_directory = 'array_storage/{}_{}_{}_{}_{}_{:.1E}/'.format(current_file_version,nu,nv,neta,nq,sigma).replace('.','d')

BM = BuildMatrices(array_save_directory, nu, nv, nx, ny, neta, nf, nq, sigma)
BM.build_minimum_sufficient_matrix_stack()

chan_selection=''

Fz_normalisation = nf**0.5
DFT2D_Fz_normalisation = (nu*nv*nf)**0.5
n_Fourier = (nu*nv-1)*nf
n_quad = (nu*nv-1)*nq
n_model = n_Fourier+n_quad
n_dat = n_Fourier


#----------------------
#----------------------


import numpy as np
from numpy import *
import pylab

dat1 = np.loadtxt('Stats/SQ-N0d2-WN0-Cyl14-EoRnf0d422Fg-FgP-18f-163Comb-v1d0-_posterior_weighted_means_and_standard_deviations.dat')[:-1]
# dat1 = np.loadtxt('Stats/SQ-N0d2-WN0-Cyl14-163JpEGS-18f-z7d6-v1d0-_posterior_weighted_means_and_standard_deviations.dat')
dat2 = np.loadtxt('Stats/SQ-N0d2-WN0-Cyl14-EoRnf0d422-18f-v1d0-_posterior_weighted_means_and_standard_deviations.dat')
# dat2 = np.loadtxt('Stats/SQ-N0d2-WN0-Cyl14-163JpEGS-18f-z7d6-nb2-v1d0-_posterior_weighted_means_and_standard_deviations.dat')
# dat2 = np.loadtxt('Stats/SQ-N0d2-WN0-Cyl14-EoRnf0d422Fg-FgP-18f-163Comb-kp10-_posterior_weighted_means_and_standard_deviations.dat')
# dat2 = np.loadtxt('Stats/SQ-N0d2-WN0-Cyl14-EoRnf0d444-18f-v1d0-_posterior_weighted_means_and_standard_deviations.dat')
dat3 = np.loadtxt('Stats/SQ-N0d2-WN0-Cyl14-163JpEGS-18f-z7d6-v1d0-_posterior_weighted_means_and_standard_deviations.dat')
SQ-N0d2-WN0-OCB2-EoRnf0d424-18f-v2d0-_posterior_weighted_means_and_standard_deviations.dat
Nrows = 7
pylab.figure()
pylab.errorbar(np.arange(len(dat1)-1), dat1[:-1,0], yerr=dat1[:-1,1], fmt='+')
pylab.errorbar(np.arange(len(dat2)), dat2[:,0], yerr=dat2[:,1], fmt='+')
pylab.show()

fig,ax=pylab.subplots(nrows=2,ncols=1)
ax[0].errorbar(np.arange(len(dat1)-1), dat2[:,0]-dat1[:-1,0], yerr=dat1[:-1,1], fmt='+')
ax[1].errorbar(np.arange(len(dat1)-1), (dat2[:,0]-dat1[:-1,0])/dat1[:-1,1], fmt='+')
fig.show()

OutPath = 'plots_cylindrical_power_specs/'
fig,ax=pylab.subplots(figsize=(10,10))
cax=pylab.imshow(dat1[:,0].reshape(-1,Nrows).T[::-1], vmin=0.0, vmax=2.5, cmap=pylab.get_cmap('jet'),extent=(0.0,0.05207485,-1.337,-0.1662), aspect=((Nrows/14.)*0.052/(-0.1662--1.337)))
cbar=pylab.colorbar(cax, fraction=0.024, pad=0.01)
cbar.ax.tick_params(labelsize=17) 
pylab.xlabel('$k_{\perp}\ [h\\mathrm{Mpc}^{-1}]$', fontsize=20)
pylab.ylabel('$\log_{10}(k_{\parallel}\ [h\\mathrm{Mpc}^{-1}])$', fontsize=20)
pylab.tick_params(labelsize=16)
fig.tight_layout()
print 'Outputting to: '+OutPath+'EoRnf0d422Fg-FgP_IntrinsicCylindricalPS.png'
pylab.savefig(OutPath+'EoRnf0d422Fg-FgP_IntrinsicCylindricalPS.png')
# pylab.show()


fig,ax=pylab.subplots(figsize=(10,10))
cax=pylab.imshow(dat2[:,0].reshape(-1,Nrows).T[::-1], cmap=pylab.get_cmap('jet'),extent=(0.0,0.05207485,-1.337,-0.1662), aspect=((Nrows/14.)*0.052/(-0.1662--1.337)))
cbar=pylab.colorbar(cax, fraction=0.024, pad=0.01)
cbar.ax.tick_params(labelsize=17) 
pylab.xlabel('$k_{\perp}\ [h\\mathrm{Mpc}^{-1}]$', fontsize=20)
pylab.ylabel('$\log_{10}(k_{\parallel}\ [h\\mathrm{Mpc}^{-1}])$', fontsize=20)
pylab.tick_params(labelsize=16)
fig.tight_layout()
print 'Outputting to: '+OutPath+'EoRnf0d422_IntrinsicCylindricalPS.png'
# pylab.savefig(OutPath+'EoRnf0d422_IntrinsicCylindricalPS.png')
pylab.show()


bias_ps = dat1[:,0].reshape(-1,Nrows).T[::-1] - dat2[:,0].reshape(-1,Nrows).T[::-1]
mask = (bias_ps - 0.5*dat2[:,1].reshape(-1,Nrows).T[::-1]) < 0
bias_ps[mask] = np.nan

fig,ax=pylab.subplots(figsize=(10,10))
cax=pylab.imshow(bias_ps, cmap=pylab.get_cmap('jet'),extent=(0.0,0.05207485,-1.337,-0.1662), aspect=((Nrows/14.)*0.052/(-0.1662--1.337)))
cbar=pylab.colorbar(cax, fraction=0.024, pad=0.01)
cbar.ax.tick_params(labelsize=17) 
pylab.xlabel('$k_{\perp}\ [h\\mathrm{Mpc}^{-1}]$', fontsize=20)
pylab.ylabel('$\log_{10}(k_{\parallel}\ [h\\mathrm{Mpc}^{-1}])$', fontsize=20)
pylab.tick_params(labelsize=16)
fig.tight_layout()
print 'Outputting to: '+OutPath+'bias_ps.png'
# pylab.savefig(OutPath+'EoRnf0d422_IntrinsicCylindricalPS.png')
pylab.show()


s2n = (10.**dat1[:,0].reshape(-1,Nrows).T[::-1]/(10.**dat1[:,0].reshape(-1,Nrows).T[::-1]*log(10)*dat1[:,1].reshape(-1,Nrows).T[::-1]))
fig,ax=pylab.subplots(figsize=(10,10))
cax=pylab.imshow(s2n, cmap=pylab.get_cmap('jet'),extent=(0.0,0.05207485,-1.337,-0.1662), aspect=((Nrows/14.)*0.052/(-0.1662--1.337)))
cbar=pylab.colorbar(cax, fraction=0.024, pad=0.01)
cbar.ax.tick_params(labelsize=17) 
pylab.xlabel('$k_{\perp}\ [h\\mathrm{Mpc}^{-1}]$', fontsize=20)
pylab.ylabel('$\log_{10}(k_{\parallel}\ [h\\mathrm{Mpc}^{-1}])$', fontsize=20)
pylab.tick_params(labelsize=16)
fig.tight_layout()
print 'Outputting to: '+OutPath+'EoRnf0d422Fg-FgP_IntrinsicCylindricalPS.png'
pylab.savefig(OutPath+'EoRnf0d422Fg-FgP_SignalToNoise.png')
# pylab.show()


comb = log10(10.**dat1[:,0].reshape(-1,Nrows).T[::-1] + 10.**dat3[:,0].reshape(-1,8).T[:-1][::-1])
fig,ax=pylab.subplots(figsize=(10,10))
cax=pylab.imshow(comb, vmin=0.0, vmax=2.5, cmap=pylab.get_cmap('jet'),extent=(0.0,0.05207485,-1.337,-0.1662), aspect=((Nrows/14.)*0.052/(-0.1662--1.337)))
cbar=pylab.colorbar(cax, fraction=0.024, pad=0.01)
cbar.ax.tick_params(labelsize=17) 
pylab.xlabel('$k_{\perp}\ [h\\mathrm{Mpc}^{-1}]$', fontsize=20)
pylab.ylabel('$\log_{10}(k_{\parallel}\ [h\\mathrm{Mpc}^{-1}])$', fontsize=20)
pylab.tick_params(labelsize=16)
fig.tight_layout()
print 'Outputting to: '+OutPath+'EoRnf0d422Fg_IntrinsicCylindricalPS.png'
pylab.savefig(OutPath+'EoRnf0d422Fg_IntrinsicCylindricalPS.png')
pylab.show()


fig,ax=pylab.subplots(figsize=(10,10))
cax=pylab.imshow(dat3[:,0].reshape(-1,8).T[:][::-1], cmap=pylab.get_cmap('jet'),extent=(0.0,0.05207485,-1.337,-0.1662), aspect=((Nrows/14.)*0.052/(-0.1662--1.337)))
cbar=pylab.colorbar(cax, fraction=0.024, pad=0.01)
cbar.ax.tick_params(labelsize=17) 
pylab.xlabel('$k_{\perp}\ [h\\mathrm{Mpc}^{-1}]$', fontsize=20)
pylab.ylabel('$\log_{10}(k_{\parallel}\ [h\\mathrm{Mpc}^{-1}])$', fontsize=20)
pylab.tick_params(labelsize=16)
fig.tight_layout()
print 'Outputting to: '+OutPath+'163JpEGS_IntrinsicCylindricalPS.png'
pylab.savefig(OutPath+'163JpEGS_IntrinsicCylindricalPS.png')
pylab.show()

new_ocb = np.zeros([18,15])
new_ocb[0,:-1] = dat3[:,0].reshape(-1,8).T[:][0]
new_ocb[1,:-1] = dat3[:,0].reshape(-1,8).T[:][2]
new_ocb[2:4,:-1] = dat3[:,0].reshape(-1,8).T[:][3]
new_ocb[4:6,:-1] = dat3[:,0].reshape(-1,8).T[:][4]
new_ocb[6:9,:-1] = dat3[:,0].reshape(-1,8).T[:][5]
new_ocb[9:14,:-1] = dat3[:,0].reshape(-1,8).T[:][6]
new_ocb[14:18,:-1] = dat3[:,0].reshape(-1,8).T[:][7]
new_ocb[:,14]=new_ocb[:,13]
# threshold = sorted(dat3[:,0])[70] #median power
threshold = sorted(dat3[:,0])[49] #median power
new_ocb_params = np.zeros([18,15])
p_count=1
for i in range(len(new_ocb_params)):
	for j in range(len(new_ocb_params[0])):
		if new_ocb[i,j]>=threshold:
			new_ocb_params[i,j] = p_count
			p_count+=1
block_thresholds = log10(linspace(10.**new_ocb.min(), 10.**threshold, 5))
for i in range(len(new_ocb_params)):
	for j in range(len(new_ocb_params[0])):
		for k in range(len(block_thresholds)-1):
			if block_thresholds[k+1]>new_ocb[i,j]>=block_thresholds[k]:
				new_ocb_params[i,j] = p_count+k
pylab.imshow(new_ocb_params[::-1])
pylab.show()
# np.savetxt('OptimalCylBinning2.dat', new_ocb_params, fmt='%d')
np.savetxt('OptimalCylBinning32.dat', new_ocb_params, fmt='%d')

   


threshold = sorted(dat3[:,0])[49] #median power
threshold_SubCylindrical = sorted(dat3[:,0])[-25] #Split the highest power modes into sub-cylindrical bins
new_ocb_params_SubCylindrical = np.zeros([18,15,2])
p_count=1
for i in range(len(new_ocb_params_SubCylindrical)):
	for j in range(len(new_ocb_params_SubCylindrical[0])):
		if new_ocb[i,j]>=threshold:
			if  new_ocb[i,j]>=threshold_SubCylindrical:
				for k_sc in range(len(new_ocb_params_SubCylindrical[0][0])):
					new_ocb_params_SubCylindrical[i,j,k_sc] = p_count
					p_count+=1
			else:
					new_ocb_params_SubCylindrical[i,j,:] = p_count
					p_count+=1				
block_thresholds = log10(linspace(10.**new_ocb.min(), 10.**threshold, 5))
for i in range(len(new_ocb_params_SubCylindrical)):
	for j in range(len(new_ocb_params_SubCylindrical[0])):
		for k in range(len(block_thresholds)-1):
			if block_thresholds[k+1]>new_ocb[i,j]>=block_thresholds[k]:
				new_ocb_params_SubCylindrical[i,j,:] = p_count+k
pylab.imshow(new_ocb_params_SubCylindrical[::-1][:,:,0])
pylab.show()
new_ocb_params_SubCylindrical_stacked = np.concatenate((new_ocb_params_SubCylindrical[:,:,0], new_ocb_params_SubCylindrical[:,:,1]))
# np.savetxt('OptimalCylBinning32_SubCylindrical.dat', new_ocb_params_SubCylindrical, fmt='%d')
np.savetxt('OptimalCylBinning32_SubCylindrical.dat', new_ocb_params_SubCylindrical_stacked, fmt='%d')




import numpy as np
from numpy import *
import pylab
dat3 = np.loadtxt('Stats/SQ-N0d2-WN0-OCB2-EoRnf0d424-18f-v2d0-_posterior_weighted_means_and_standard_deviations.dat')
cylindrical_mapping = np.loadtxt('../../../../OptimalCylBinning32.dat')
OCMP = np.zeros_like(cylindrical_mapping)

for i in range(len(dat3)):
    OCMP[cylindrical_mapping==(i+1)]=dat3[i][0]
 
pylab.imshow(OCMP[::-1])
pylab.colorbar()
pylab.show()




pylab.figure()
pylab.imshow(dat1[:,0].reshape(-1,Nrows).T[::-1], vmin=0.0, vmax=2.0, cmap=pylab.get_cmap('jet'))
pylab.colorbar()
# pylab.show()

pylab.figure()
pylab.imshow(dat2[:,0].reshape(-1,Nrows).T[::-1], vmin=0.0, vmax=2.0, cmap=pylab.get_cmap('jet'))
pylab.colorbar()
pylab.show()

pylab.figure()
pylab.imshow(log10(10.**dat1+10.**dat2)[:,0].reshape(-1,Nrows).T[::-1], vmin=0.0, vmax=2.0, cmap=pylab.get_cmap('jet'))
pylab.colorbar()
pylab.show()


fig,ax=pylab.subplots(nrows=3,ncols=1)
cax0 = ax[0].imshow(dat2[:,0].reshape(-1,7).T[::-1], vmin=0.6)
fig.colorbar(cax0, ax=ax[0])
cax1 = ax[1].imshow(dat1[:-1,0].reshape(-1,7).T[::-1], vmin=0.6)
fig.colorbar(cax1, ax=ax[1])
cax2 = ax[2].imshow((dat2[:,0].reshape(-1,7).T[::-1]-dat1[:-1,0].reshape(-1,7).T[::-1])/dat1[:-1,1].reshape(-1,7).T[::-1])
fig.colorbar(cax2, ax=ax[2])
fig.show()

pylab.figure()
# pylab.imshow(dat1[:-1,1].reshape(-1,7).T[::-1])
pylab.imshow(dat1[:-1,1].reshape(-1,7).T[::-1], vmax=0.15)
pylab.colorbar()
# pylab.show()

pylab.figure()
# pylab.imshow(dat2[:,1].reshape(-1,7).T[::-1])
pylab.imshow(dat2[:,1].reshape(-1,7).T[::-1], vmax=0.15)
pylab.colorbar()
pylab.show()

pylab.figure()
pylab.imshow((dat1[:-1,1]>0.30).reshape(-1,7).T[::-1], vmax=1.0)
pylab.colorbar()
# pylab.show()



###
# EoRFg
###
dat1_errors = dat1[:,1].copy()
provisional_linear_space_propagated_error_EoRFg = (10.**dat1[:,0].reshape(-1,7).T)*log(10)*dat1_errors.reshape(-1,7).T

Signal_to_noise = (10.**dat1[:,0].reshape(-1,7).T[::-1]/provisional_linear_space_propagated_error_EoRFg[::-1])
undetected_coefficients_selector_mask = Signal_to_noise<1.0
pylab.close('all')
pylab.figure()
# pylab.imshow(Signal_to_noise)
pylab.imshow(undetected_coefficients_selector_mask)
pylab.colorbar()
pylab.show()

dat1_errors[undetected_coefficients_selector_mask[::-1].T.flatten()]=1000.0 #Undetected coefficients are prior dominated and therefore should be zero-weighted to avoid biasing the mean
linear_space_propagated_error_EoRFg = (10.**dat1[:,0].reshape(-1,7).T)*log(10)*dat1_errors.reshape(-1,7).T
# linear_space_propagated_error_on_mean = np.sum(linear_space_propagated_error**2., axis=1)**0.5/14.
linear_space_propagated_error_on_mean_EoRFg = np.sum(1./linear_space_propagated_error_EoRFg**2., axis=1)**0.5
log_space_propagated_error_EoRFg = linear_space_propagated_error_on_mean_EoRFg/(log(10)*np.mean(10.**dat1[:,0].reshape(-1,7).T, axis=1))

linear_space_powers_EoRFg = 10.**dat1[:,0].reshape(-1,7).T
weighted_mean_linear_space_powers_EoRFg = (np.sum(linear_space_powers_EoRFg/linear_space_propagated_error_EoRFg**2., axis=1)/np.sum(1./linear_space_propagated_error_EoRFg**2., axis=1))
log_weighted_mean_linear_space_powers_EoRFg = log10(weighted_mean_linear_space_powers_EoRFg)
print log_weighted_mean_linear_space_powers_EoRFg
print log_space_propagated_error_EoRFg

###
# EoR
###
dat2_errors = dat2[:,1].copy()
# dat2_errors[dat2[:,1]>0.10]=1000.1
linear_space_propagated_error_EoR = (10.**dat2[:,0].reshape(-1,7).T)*log(10)*dat2_errors.reshape(-1,7).T
# linear_space_propagated_error_on_mean = np.sum(linear_space_propagated_error**2., axis=1)**0.5/14.
linear_space_propagated_error_on_mean_EoR = np.sum(1./linear_space_propagated_error_EoR**2., axis=1)**0.5
log_space_propagated_error_EoR = linear_space_propagated_error_on_mean_EoR/(log(10)*np.mean(10.**dat2[:,0].reshape(-1,7).T, axis=1))

linear_space_powers_EoR = 10.**dat2[:,0].reshape(-1,7).T
weighted_mean_linear_space_powers_EoR = (np.sum(linear_space_powers_EoR/linear_space_propagated_error_EoR**2., axis=1)/np.sum(1./linear_space_propagated_error_EoR**2., axis=1))
log_weighted_mean_linear_space_powers_EoR = log10(weighted_mean_linear_space_powers_EoR)
print log_weighted_mean_linear_space_powers_EoR
print log_space_propagated_error_EoR



linear_space_powers_EoR = 10.**dat2[:,0].reshape(-1,7).T
weighted_mean_linear_space_powers_EoR = (np.sum(linear_space_powers/linear_space_propagated_error**2., axis=1)/np.sum(1./linear_space_propagated_error**2., axis=1))

print np.log10(np.mean(10.**dat1[:,0].reshape(-1,7).T, axis=1))
print np.log10(np.mean(10.**dat2[:,0].reshape(-1,7).T, axis=1))
print log_space_propagated_error


k_par_vals = linspace(-1.337,-0.1662, 7)

comb_sph = np.log10(np.mean(10.**dat3[:,0].reshape(-1,8).T[:-1], axis=1)+10.**log_weighted_mean_linear_space_powers_EoR)
print comb_sph


pylab.figure(figsize=(10,10))
pylab.axis(xmin=-1.4,xmax=0.0,ymin=-0.5,ymax=3.5)
pylab.errorbar(k_par_vals, log_weighted_mean_linear_space_powers_EoR, fmt='--', color='black', linewidth=1.5)
pylab.errorbar(k_par_vals, log_weighted_mean_linear_space_powers_EoRFg, log_space_propagated_error_EoRFg, fmt='+', markeredgewidth=2, color='blue', linewidth=1.5)
# pylab.errorbar(TwoSigmaUpperLimits_K_ArrayFG, TwoSigmaUpperLimitsFG+0.4, yerr=[0.4*np.ones(len(TwoSigmaUpperLimitsFG)), 0.0*np.ones(len(TwoSigmaUpperLimitsFG))], color='darkblue', fmt='-', lolims=TwoSigmaUpperLimitsFG, linestyle = '', linewidth=1.5, capsize=5)
pylab.errorbar(k_par_vals, comb_sph, fmt='--', markeredgewidth=2, color='red', linewidth=1.5)
pylab.errorbar(k_par_vals, comb_sph, fmt='+', markeredgewidth=2, color='red', linewidth=1.5)
#
pylab.xlabel('$\\log_{10}(k[h\\mathrm{Mpc}^{-1}])$', size='23')
pylab.ylabel('$\\log_{10}(P(k)[\mathrm{mK}^{2}])$', size='23')
#
print 'Outputting to: '+OutPath+'EoRnf0d422Fg-FgP_SphericallyAveraged.png'
pylab.savefig(OutPath+'EoRnf0d422Fg-FgP_SphericallyAveraged.png')
pylab.show()




import subprocess
import numpy as np

Files_and_Subdirectories_list=subprocess.os.listdir('Stats/') 
subsets_EoRFg_Fgp = [file for file in Files_and_Subdirectories_list if file.count('sub') and file.count('Fg')]
subsets_EoR = [file for file in Files_and_Subdirectories_list if file.count('sub') and not file.count('Fg')]
subsets_EoRFg_Fgp_dat = [np.loadtxt('Stats/'+file) for file in subsets_EoRFg_Fgp]
# for i in range(len(subsets_EoRFg_Fgp_dat)): subsets_EoRFg_Fgp_dat[i][1][0] -= 0.5
subsets_EoR_dat = [np.loadtxt('Stats/'+file) for file in subsets_EoR]
Fg_dat = np.loadtxt('Stats/'+'SQ-N0d2-WN0-Sph-Fg-18f-163Jelic-v1d1-_posterior_weighted_means_and_standard_deviations.dat')

k_par_vals = linspace(-1.337,-0.1662, 8)
OutPath = 'plots_cylindrical_power_specs/'
pylab.figure(figsize=(10,10))
pylab.axis(xmin=-1.4,xmax=0.0,ymin=0.5,ymax=1.75)
# pylab.errorbar(linspace(-1.337,-0.1662, 7), Fg_dat[:,0], fmt='--', linewidth=1.5)
for i in range(len(subsets_EoR_dat)):pylab.errorbar(k_par_vals, subsets_EoR_dat[i][:,0], fmt='--', linewidth=1.5)
for i in range(len(subsets_EoRFg_Fgp_dat)):pylab.errorbar(k_par_vals, subsets_EoRFg_Fgp_dat[i][:-1,0], subsets_EoRFg_Fgp_dat[i][:-1,1], fmt='+', markeredgewidth=2, linewidth=1.5)
# #
pylab.xlabel('$\\log_{10}(k[h\\mathrm{Mpc}^{-1}])$', size='23')
pylab.ylabel('$\\log_{10}(P(k)[\mathrm{mK}^{2}])$', size='23')
#
# print 'Outputting to: '+OutPath+'EoRFgSubplots.png'
# pylab.savefig(OutPath+'EoRFgSubplots.png')
pylab.show()



import numpy as np

# dat1 = np.loadtxt('SQ-N0d05-WN0-Sph-GRNEoR-18f-nq-v1d0-.txt')
# dat2 = np.loadtxt('SQ-N0d2-WN0-Sph-GRNEoR2-18f-nq-v1d0-.txt')
# dat3  = np.loadtxt('SQ-N0d2-WN0-Sph-GRNEoRGRNEoR2-EoR2P-18f-nq-v1d0-.txt')
# dat1 = np.loadtxt('SQ-N0d2-WN0-Sph-GRNEoR-18f-v1d0-.txt')
# dat2 = np.loadtxt('SQ-N0d2-WN0-Sph-GRNFg-18f-v1d0-.txt')
# dat3  = np.loadtxt('SQ-N0d2-WN0-Sph-GRNEoRGRNFg-18f-v1d1-.txt')
# dat1 = np.loadtxt('SQ-N0d2-WN0-Sph-GRNEoR-18f-v1d0-.txt')
# dat2 = np.loadtxt('SQ-N0d2-WN0-Sph-JpEGS-18f-v1d0-.txt')
# dat3  = np.loadtxt('SQ-N0d2-WN0-Sph-GRNEoRFg-18f-v1d1-.txt')
# dat1 = np.loadtxt('SQ-N0d2-WN0-Sph-EoRnf0d424-18f-sub0-.txt')
# dat2 = np.loadtxt('SQ-N0d2-WN0-Sph-GRNFg-18f-v1d0-.txt')
# dat3  = np.loadtxt('SQ-N0d2-WN0-Sph-EoRnf0d424Fg-18f-v1d0-.txt')
dat1 = np.loadtxt('SQ-N0d2-WN0-Sph-EoRnf0d424-18f-v1d0-.txt')
dat2 = np.loadtxt('SQ-N0d2-WN0-Sph-GRNFg-18f-v1d0-.txt')
dat3  = np.loadtxt('SQ-N0d2-WN0-Sph-EoRnf0d424GRNFg-18f-v1d0-.txt')



expected_combPS_vals = np.log10(10.**np.sum(dat1[:,2:]*dat1[:,0].reshape(-1,1), axis=0)+10.**np.sum(dat2[:,2:]*dat2[:,0].reshape(-1,1), axis=0))

import pylab
OutPath = 'Plots/'
Npar=8
Fig,ax = pylab.subplots(Npar,2, figsize=(10,10))
for i_par in range(Npar):
        hist1,bin_edges1 = np.histogram(dat1[:,2+i_par], weights=dat1[:,0], bins=20, normed=True)
        bin_centres1 = (bin_edges1[:-1]+bin_edges1[1:])/2.0
        hist2,bin_edges2 = np.histogram(dat2[:,2+i_par], weights=dat2[:,0], bins=20, normed=True)
        bin_centres2 = (bin_edges2[:-1]+bin_edges2[1:])/2.0
        hist3,bin_edges3 = np.histogram(dat3[:,2+i_par], weights=dat3[:,0], bins=20, normed=True)
        bin_centres3 = (bin_edges3[:-1]+bin_edges3[1:])/2.0
        ax[i_par,0].errorbar(bin_centres1, hist1, linestyle='-', drawstyle='steps', color='red', linewidth=1.5)
        ax[i_par,0].errorbar(bin_centres2, hist2, linestyle='-', drawstyle='steps', color='blue', linewidth=1.5)
        ax[i_par,0].errorbar(bin_centres3, hist3, linestyle='-', drawstyle='steps', color='black', linewidth=1.5)
        ax[i_par,0].axvline(expected_combPS_vals[i_par], color='green', linestyle='-')
        ax[i_par,1].errorbar(bin_centres3-expected_combPS_vals[i_par], hist3, linestyle='-', drawstyle='steps', color='orange', linewidth=1.5)
        # ax[i_par].set_xlim([bin_centres[0], bin_centres[-1]])
        for j in range(2):ax[i_par,j].set_yticklabels(['C$_{%s}$'%str(i_par+1)],rotation='vertical')
        for j in range(2):ax[i_par,j].locator_params(axis = 'x', nbins = 5)
        for j in range(2):ax[i_par,0].set_ylabel('C$_{%s}$'%str(i_par+1), size='23',rotation='vertical')
        if i_par==Npar-1:
        	for j in range(2):ax[i_par, j].set_xlabel('$\\log_{10}(\Delta^{2}(k)~[\\mathrm{mK}^{2}])$', size='23', labelpad=15)
        #
        pylab.rc('text', usetex=True)
        pylab.rc('font', family='serif')
        pylab.rc('font', size='18')
        pylab.tight_layout()
        #                       
        for j in range(2):ax[i_par,j].tick_params(\
        axis='y',          # changes apply to the y-axis
        labelleft='on') # labels along the bottom edge are off
output_filename = 'EoRGRNFg.png'
# output_filename = 'GRNEoRGRNFg.png'
# output_filename = 'EoRFg.png'
print 'Outputting to: '+OutPath+output_filename
Fig.savefig(OutPath+output_filename)
Fig.show()




import numpy as np

dat1 = np.loadtxt('EoR2-.txt')
dat31  = np.loadtxt('EoR2SimEoR-.txt')
# dat32 = np.loadtxt('EoR2SimEoR-v2d0-.txt')
dat32 = np.loadtxt('EoR2SimEoR-v2d2-.txt')
dat33 = np.loadtxt('EoR2SimEoR-v2d3-.txt')

dat = [dat1,dat31,dat32,dat33]



fortran_PS_params = np.array([0.55,0.95,1.22,1.31,1.29,1.16,0.73])
expected_combPS_vals_f = np.log10(10.**np.sum(dat1[:,2:-1]*dat1[:,0].reshape(-1,1), axis=0)+10.**fortran_PS_params)

import pylab
OutPath = 'Plots/'
Npar=7
colors=['red']+['black']*10
Fig,ax = pylab.subplots(Npar,2, figsize=(10,10))
for i_par in range(Npar):
	for i_dat in range(len(dat)):
		hist1,bin_edges1 = np.histogram(dat[i_dat][:,2+i_par], weights=dat[i_dat][:,0], bins=20, normed=True)
		bin_centres1 = (bin_edges1[:-1]+bin_edges1[1:])/2.0
		ax[i_par,0].errorbar(bin_centres1, hist1, linestyle='-', drawstyle='steps', color=colors[i_dat], linewidth=1.5)
		ax[i_par,0].axvline(expected_combPS_vals_f[i_par], color='green', linestyle='-')
		if i_dat==2: ax[i_par,1].errorbar(bin_centres1-expected_combPS_vals_f[i_par], hist1, linestyle='-', drawstyle='steps', color='orange', linewidth=1.5)
		for j in range(2):ax[i_par,j].set_yticklabels(['C$_{%s}$'%str(i_par+1)],rotation='vertical')
		for j in range(2):ax[i_par,j].locator_params(axis = 'x', nbins = 5)
		for j in range(2):ax[i_par,0].set_ylabel('C$_{%s}$'%str(i_par+1), size='23',rotation='vertical')
		if i_par==Npar-1:
			for j in range(2):ax[i_par, j].set_xlabel('$\\log_{10}(\Delta^{2}(k)~[\\mathrm{mK}^{2}])$', size='23', labelpad=15)
		#
		pylab.rc('text', usetex=True)
		pylab.rc('font', family='serif')
		pylab.rc('font', size='18')
		pylab.tight_layout()
		#                       
		for j in range(2):ax[i_par,j].tick_params(\
		axis='y',          # changes apply to the y-axis
		labelleft='on') # labels along the bottom edge are off
output_filename = 'EoRGRNEoR-L.png'
print 'Outputting to: '+OutPath+output_filename
Fig.savefig(OutPath+output_filename)
Fig.show()

	



import numpy as np
from numpy import *
dat1 = np.loadtxt('SQ-N0d2-WN0-Cyl14-163JpEGS-18f-z7d6-v1d0--MLuveta.dat')
chanv1_real = dat1[dat1[:,2]==dat1[18][2]][::2][:960]
chanv1_imag = dat1[dat1[:,2]==dat1[18][2]][::2][960:]
chanv1 = chanv1_real + 1j*chanv1_imag
chan1 = np.concatenate((chanv1[:480][:,3],[1],chanv1[480:][:,3]))

fig,ax=pylab.subplots(figsize=(10,10))
cax=pylab.imshow(np.log10(abs(chan1.reshape(31,31))), cmap=pylab.get_cmap('jet'),vmin=2.5,vmax=4.0)
cbar=pylab.colorbar(cax, fraction=0.024, pad=0.01)
cbar.ax.tick_params(labelsize=17) 
pylab.xlabel('$k_{\perp}\ [h\\mathrm{Mpc}^{-1}]$', fontsize=20)
pylab.ylabel('$\log_{10}(k_{\parallel}\ [h\\mathrm{Mpc}^{-1}])$', fontsize=20)
pylab.tick_params(labelsize=16)
spacing = 15*2**0.5 / 14
bin_maxima = 1+np.arange(1,14,1)*spacing
for bin_limit in bin_maxima:
	circle = pylab.Circle((15,15), bin_limit, color='black', fill=False, linewidth=3.5)
	ax.add_patch(circle)
fig.tight_layout()
# pylab.show()


import numpy as np
dat2 = np.loadtxt('SQ-N0d2-WN0-Sph-EoRnf0d424-18f-v1d0--MLuveta.dat')
chanv2_real = dat2[dat1[:,2]==dat2[18][2]][::2][:960]
chanv2_imag = dat2[dat1[:,2]==dat2[18][2]][::2][960:]
chanv2 = chanv2_real + 1j*chanv2_imag
chan2 = np.concatenate((chanv2[:480][:,3],[1],chanv2[480:][:,3]))

fig,ax=pylab.subplots(figsize=(10,10))
cax=pylab.imshow(np.log10(abs(chan2.reshape(31,31))), cmap=pylab.get_cmap('jet'),vmin=2.5,vmax=4.0)
cbar=pylab.colorbar(cax, fraction=0.024, pad=0.01)
cbar.ax.tick_params(labelsize=17) 
pylab.xlabel('$k_{\perp}\ [h\\mathrm{Mpc}^{-1}]$', fontsize=20)
pylab.ylabel('$\log_{10}(k_{\parallel}\ [h\\mathrm{Mpc}^{-1}])$', fontsize=20)
pylab.tick_params(labelsize=16)
bin_maxima = 1+np.arange(1,14,1)*spacing
for bin_limit in bin_maxima:
	circle = pylab.Circle((15,15), bin_limit, color='black', fill=False, linewidth=3.5)
	ax.add_patch(circle)
fig.tight_layout()
pylab.show()




xx,yy=np.mgrid[-15:16,-15:16]
radius = (xx**2.+yy**2.)**0.5
bin_limits = np.concatenate(([0.5],bin_maxima))

bin1=np.logical_and(bin_limits[0]<radius, radius<bin_limits[1])
print log10(chan1.reshape(31,31)[bin1].var())
print log10(chan2.reshape(31,31)[bin1].var())
print log10((chan1+chan2).reshape(31,31)[bin1].var())
print log10((chan1+chan2).reshape(31,31)[bin1].var() - chan1.reshape(31,31)[bin1].var())

print log10(chan1.reshape(31,31).var())
print log10(chan2.reshape(31,31).var())
print log10((chan1+chan2).reshape(31,31).var())
print log10((chan1+chan2).reshape(31,31).var() - chan1.reshape(31,31).var())

for i in range(13):
	bin1=np.logical_and(bin_limits[i]<radius, radius<bin_limits[i+1])
	print i
	# print log10(chan1.reshape(31,31)[bin1].var())
	print log10(chan2.reshape(31,31)[bin1].var())
	# print log10((chan1+chan2).reshape(31,31)[bin1].var())
	print log10((chan1+chan2).reshape(31,31)[bin1].var() - chan1.reshape(31,31)[bin1].var())
	print log10((chan1+chan2).reshape(31,31)[bin1].var() - chan1.reshape(31,31)[bin1].var()) - log10(chan2.reshape(31,31)[bin1].var()), '\n'


print log10(chan2.reshape(31,31).var())
print log10((chan1+chan2).reshape(31,31).var() - chan1.reshape(31,31).var())
print log10((chan1+chan2).reshape(31,31).var() - chan1.reshape(31,31).var()) - log10(chan2.reshape(31,31).var()), '\n'




###
# Load in data
###
import numpy as np

fg_mluveta = np.loadtxt('SQ-N0d2-WN0-OCB2-JpEGS-18f-v1d1--MLuveta.dat')
EoR_mluveta = np.loadtxt('SQ-N0d2-WN0-OCB2-EoRnf0d424-18f-v2d0--MLuveta.dat')

###
# Foregrounds
###
fg_mluveta_complex = fg_mluveta[:,3].reshape(-1,36)[0:960] +1j*fg_mluveta[:,3].reshape(-1,36)[960:]
fg_mluveta_ks = (fg_mluveta[:,0]**2. + fg_mluveta[:,1]**2. + fg_mluveta[:,2]**2.)**0.5
fg_mluveta_ks_reshaped = fg_mluveta_ks.reshape(-1,36)[0:960]
fgc = fg_mluveta_complex.copy()
fgc_cube = np.array([np.concatenate((fgc[:,i][:480],[0],fgc[:,i][480:])).reshape(31,31) for i in range(fg_mluveta_complex.shape[1])])

import pylab
fig,ax=pylab.subplots(figsize=(10,10))
cax=pylab.imshow(np.log10(abs(fgc_cube[1])**2.), cmap=pylab.get_cmap('jet'))
# cax=pylab.imshow(abs(fgc_cube[1])**2., cmap=pylab.get_cmap('jet'))
cbar=pylab.colorbar(cax, fraction=0.024, pad=0.01)
cbar.ax.tick_params(labelsize=17) 
# pylab.xlabel('$k_{\perp}\ [h\\mathrm{Mpc}^{-1}]$', fontsize=20)
# pylab.ylabel('$\log_{10}(k_{\parallel}\ [h\\mathrm{Mpc}^{-1}])$', fontsize=20)
pylab.tick_params(labelsize=16)
fig.tight_layout()
pylab.show()



###
# EoR
###
EoR_mluveta_complex = EoR_mluveta[:,3].reshape(-1,36)[0:960] +1j*EoR_mluveta[:,3].reshape(-1,36)[960:]
EoR_mluveta_ks = (EoR_mluveta[:,0]**2. + EoR_mluveta[:,1]**2. + EoR_mluveta[:,2]**2.)**0.5
EoR_mluveta_ks_reshaped = EoR_mluveta_ks.reshape(-1,36)[0:960]
EoRc = EoR_mluveta_complex.copy()
EoRc_cube = np.array([np.concatenate((EoRc[:,i][:480],[0],EoRc[:,i][480:])).reshape(31,31) for i in range(EoR_mluveta_complex.shape[1])])


###
# Efficacy of quadratic model for foreground emission as a function of spectral range
###
# cd /oasis/projects/nsf/brn112/psims/Fortran/Polychord/PolyChord1d10/PPSimpleEor_OCB2/Results/RA0/Cyl/OCB/Stats/
import numpy as np
from numpy import *
k = [4.39E-02, 6.59E-02, 9.89E-02, 1.48E-01, 2.22E-01, 3.34E-01, 5.00E-01, 7.51E-01][:-2]
dat103 = np.loadtxt('SQ-N0d02-WN0-Sph-103Fg85db0d02-18f-v1d0-_posterior_weighted_means_and_standard_deviations.dat')[:-2]
dat163 = np.loadtxt('SQ-N0d02-WN0-Sph-163Fg85db0d02-18f-v3d0-_posterior_weighted_means_and_standard_deviations.dat')[:-2]
dat223 = np.loadtxt('SQ-N0d02-WN0-Sph-223Fg85db0d02-18f-v3d0-_posterior_weighted_means_and_standard_deviations.dat')[:-2]

import pylab
pylab.errorbar(log10(k), dat103[:,0],dat103[:,1])
pylab.errorbar(log10(k), dat163[:,0],dat163[:,1])
pylab.errorbar(log10(k), dat223[:,0],dat223[:,1])
pylab.legend(['119-127 MHz','159-167 MHz','219-227 MHz'])
pylab.xlabel('$\\log_{10}(k[h\\mathrm{Mpc}^{-1}])$', size='18')
pylab.ylabel('$\\log_{10}(\Delta^{2}(k)~[\\mathrm{mK}^{2}])$', size='18')
pylab.tight_layout()
pylab.savefig('Foreground_contamination_as_a_function_of_frequency.png')
pylab.show()


dat_EoR = np.loadtxt('SQ-N0d02-WN1-cyl15-EoR-18f-v2d0-_posterior_weighted_means_and_standard_deviations.dat')
dat_Fg = np.loadtxt('SQ-N0d02-WN1-cyl15-163CombFg62db0d1-18f-v2d0-_posterior_weighted_means_and_standard_deviations.dat')
dat_EoRFg = np.loadtxt('SQ-N0d02-WN1-cyl15-EoR163CombFg62db0d1-18f-v2d0-_posterior_weighted_means_and_standard_deviations.dat')
dat_Fg2 = np.loadtxt('SQ-N0d02-WN1-cyl15-EoR163CombFg62db0d1-18f-v2d1-_posterior_weighted_means_and_standard_deviations.dat') #Misnamed file?

pylab.figure()
pylab.imshow(dat_EoR[:,0][:-1].reshape(15,10).T[::-1])
pylab.colorbar()
pylab.figure()
pylab.imshow(dat_Fg[:,0][:-1].reshape(15,10).T[::-1])
pylab.colorbar()
pylab.figure()
pylab.imshow(dat_EoRFg[:,0][:-1].reshape(15,10).T[::-1])
pylab.colorbar()
pylab.figure()
pylab.imshow(dat_Fg2[:,0][:-1].reshape(15,10).T[::-1], vmax=1.3)
pylab.colorbar()
pylab.show()


###
# Log-quadratic spectral model with OCB foreground power spectra
###
import numpy as np
from numpy import *
dat225 = np.loadtxt('SQ-N0d02-WN1-OCB2-225CombFg62db0d1-18f-v1d0-_posterior_weighted_means_and_standard_deviations.dat')
dat215 = np.loadtxt('SQ-N0d02-WN1-OCB2-215CombFg62db0d1-18f-v1d0-_posterior_weighted_means_and_standard_deviations.dat')
dat205 = np.loadtxt('SQ-N0d02-WN1-OCB2-205CombFg62db0d1-18f-v1d0-_posterior_weighted_means_and_standard_deviations.dat')
dat163 = np.loadtxt('SQ-N0d02-WN1-OCB2-163CombFg62db0d1-18f-v1d2-_posterior_weighted_means_and_standard_deviations.dat')
multi_frequency_ps_data = np.vstack((dat225[:,0],dat205[:,0])).T
extrapolated_ps_163 = [np.polyval(np.polyfit([229,209],multi_frequency_ps_data[i],1), 167) for i in range(len(multi_frequency_ps_data))]

print (10.**np.array(extrapolated_ps_163) - 10.**dat163[:,0])[:]
print (10.**np.array(extrapolated_ps_163) - 10.**dat163[:,0])[0:15]
print (10.**np.array(extrapolated_ps_163) - 10.**dat163[:,0])[0:15].mean()


multi_frequency_ps_data = np.vstack((dat225[:,0],dat215[:,0],dat205[:,0])).T
extrapolated_ps_163 = [np.polyval(np.polyfit([229,219,209],multi_frequency_ps_data[i],2), 167) for i in range(len(multi_frequency_ps_data))]
extrapolated_ps_163 = np.array(extrapolated_ps_163)

print (10.**np.array(extrapolated_ps_163) - 10.**dat163[:,0])[:]
print (10.**np.array(extrapolated_ps_163) - 10.**dat163[:,0])[0:15]
print (10.**np.array(extrapolated_ps_163) - 10.**dat163[:,0])[0:15].mean()

# extrapolated_ps_163_mod = np.log10((10.**extrapolated_ps_163)+2.0) #Running the power spectrum with this prior had the expected effect - the +2 is effectively adding an additional sherically averaged power spectral component and decreases the recovered EoR spherical power spectrum by the equivalent amount. This is in contrast to increasing the power in the cylindrical prior by a scale factor, in which case spherically averaged EoR power spectrum doesn't decrease by the mean power added but instead seems to be determined by increasing the power of any underestmated coefficients to a sufficient level to match the EoR+Fg power spectrum in that bin (which still results in positive bias).

data_mask = dat225[:,0]<-2
extrapolated_ps_163[data_mask]=-5.0
# np.savetxt('extrapolated_ps_163.dat',extrapolated_ps_163)
np.savetxt('extrapolated_ps_163_OCB_component.dat',extrapolated_ps_163)
# extrapolated_ps_163_mod[data_mask]=-5.0
# np.savetxt('extrapolated_ps_163.dat',extrapolated_ps_163_mod)



###
# Log-quadratic spectral model with Sph+OCB foreground power spectra
###
import numpy as np
from numpy import *
# dat225 = np.loadtxt('SQ-N4d0-WN1-Sph-230CombFg62-18f-10B-v1d0-_posterior_weighted_means_and_standard_deviations.dat')
# dat215 = np.loadtxt('SQ-N4d5-WN1-Sph-220CombFg62-18f-10B-v1d0-_posterior_weighted_means_and_standard_deviations.dat')
# dat205 = np.loadtxt('SQ-N5d1-WN1-Sph-210CombFg62-18f-10B-v1d0-_posterior_weighted_means_and_standard_deviations.dat')

dat225 = np.loadtxt('SQ-N4d3-WN0-Sph-225CombFg62-18f-10B-v1d0-_posterior_weighted_means_and_standard_deviations.dat')
dat215 = np.loadtxt('SQ-N4d8-WN0-Sph-215CombFg62-18f-10B-v1d0-_posterior_weighted_means_and_standard_deviations.dat')
dat205 = np.loadtxt('SQ-N5d5-WN0-Sph-205CombFg62-18f-10B-v1d0-_posterior_weighted_means_and_standard_deviations.dat')

n_poly = 2
# freqs = [234,224,214]
freqs = [229,219,209]
fiducial_freq  = 163
use_log_freq = True
# use_log_freq = False
if use_log_freq:
	freqs = np.log10(freqs)
	fiducial_freq = np.log10(fiducial_freq)
# multi_frequency_ps_data = np.vstack((dat225[:,0],dat215[:,0])).T
multi_frequency_ps_data = np.vstack((dat225[:,0],dat215[:,0],dat205[:,0])).T
extrapolated_ps_163 = [np.polyval(np.polyfit(freqs,multi_frequency_ps_data[i],n_poly), fiducial_freq) for i in range(len(multi_frequency_ps_data))]
extrapolated_ps_163 = np.array(extrapolated_ps_163)
print extrapolated_ps_163

data_mask = np.logical_or(dat225[:,0]<-1.5, dat225[:,1]>1.5)
# data_mask = dat225[:,0]<-2
extrapolated_ps_163[data_mask]=-5.0
extrapolated_ps_163_sph = extrapolated_ps_163[0:10]
extrapolated_ps_163_OCB = extrapolated_ps_163[10:10+82]

n_sigma = 1
# multi_frequency_ps_data_min = np.vstack((dat225[:,0]+n_sigma*dat225[:,1],dat215[:,0])).T
multi_frequency_ps_data_min = np.vstack((dat225[:,0]+n_sigma*dat225[:,1],dat215[:,0],dat205[:,0]-n_sigma*dat205[:,1])).T
extrapolated_ps_163_min = 	[np.polyval(np.polyfit(freqs,multi_frequency_ps_data_min[i],n_poly), fiducial_freq) for i in range(len(multi_frequency_ps_data_min))]
extrapolated_ps_163_min = np.array(extrapolated_ps_163_min)

# multi_frequency_ps_data_max = np.vstack((dat225[:,0]-n_sigma*dat225[:,1],dat215[:,0])).T
multi_frequency_ps_data_max = np.vstack((dat225[:,0]-n_sigma*dat225[:,1],dat215[:,0],dat205[:,0]+n_sigma*dat205[:,1])).T
extrapolated_ps_163_max = [np.polyval(np.polyfit(freqs,multi_frequency_ps_data_max[i],n_poly), fiducial_freq) for i in range(len(multi_frequency_ps_data_max))]
extrapolated_ps_163_max = np.array(extrapolated_ps_163_max)

extrapolated_ps_163_min[data_mask]=-5.0
extrapolated_ps_163_max[data_mask]=-5.0
delta_extrapolated_ps_163_plus = extrapolated_ps_163_max - extrapolated_ps_163
delta_extrapolated_ps_163_minus = extrapolated_ps_163_min - extrapolated_ps_163



import pylab as P
from matplotlib import pyplot as plt



multi_frequency_ps_data_errors = np.vstack((dat225[:,1],dat215[:,1],dat205[:,1])).T
freqs_range_225_to_163 = np.linspace(log10(229), np.log10(163), 100)
extrapolated_ps_225_to_163 = np.array([np.polyval(np.polyfit(freqs,multi_frequency_ps_data[i],n_poly), freqs_range_225_to_163) for i in range(1)])
extrapolated_ps_225_to_163_min = np.squeeze([np.polyval(np.polyfit(freqs,multi_frequency_ps_data_min[i],n_poly), freqs_range_225_to_163) for i in range(1)])
extrapolated_ps_225_to_163_max = np.squeeze([np.polyval(np.polyfit(freqs,multi_frequency_ps_data_max[i],n_poly), freqs_range_225_to_163) for i in range(1)])


fig = P.figure(figsize=(10,10))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
ax1.errorbar(freqs, multi_frequency_ps_data[0], multi_frequency_ps_data_errors[0]*1, fmt='o', color='black', linewidth=1.5, label='Measured low-$z$ foreground contamination')
ax1.errorbar(freqs_range_225_to_163, extrapolated_ps_225_to_163, fmt='--', color='red', linewidth=1.5, label='Log-quadratic model for foreground \ncontamination')
ax1.errorbar(fiducial_freq, extrapolated_ps_163[0], delta_extrapolated_ps_163_plus[0]*1, fmt='o', color='red', linewidth=1.5, label='Model for contamination of the EoR \npower spectrum by foregrounds at $z=7.6$')
# P.errorbar(freqs_range_225_to_163, extrapolated_ps_225_to_163_min, fmt='--', color='red', linewidth=1.5)
# P.errorbar(freqs_range_225_to_163, extrapolated_ps_225_to_163_max, fmt='--', color='red', linewidth=1.5)
ax1.fill_between(freqs_range_225_to_163, extrapolated_ps_225_to_163_max, extrapolated_ps_225_to_163_min, color='gray', alpha=0.5)
ax1.tick_params(labelsize=20)
ax1.set_xlabel('$\\log_{10}(\\nu[\mathrm{MHz}])$', size='23')
ax1.set_ylabel('$\\log_{10}(\Delta_{k}^{2}[\mathrm{mK}^{2}])$', size='23')
ax1.invert_xaxis()

def tick_function(nu):
    redshifts = (1400./10.**nu)-1.0
    return ["%.2f" % z for z in redshifts]

ax2.set_xlim(ax1.get_xlim())
ax2.set_xticklabels(tick_function(P.xticks()[0]))
ax2.set_xlabel(r"Redshift", size='23')
ax2.tick_params(labelsize=20)
# ax2.set_xscale('log')
# ax1.set_yscale('log')
ax1.legend(loc='upper left', bbox_to_anchor=(0.0,0.9), fontsize=16)

bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
ax1.text(2.325, 0.65, "$\\log_{10}(k[\mathrm{h^{-1}Mpc}])=-1.38$", ha="center", va="center", size=20,
        bbox=bbox_props)
P.tight_layout()
print 'Outputting to:', 'Log_quadratic_foregroung_contamination_model.png'
P.savefig('Log_quadratic_foregroung_contamination_model.png')
# 
P.show()




import pylab as P
from matplotlib import pyplot as plt



multi_frequency_ps_data_errors = np.vstack((dat225[:,1],dat215[:,1],dat205[:,1])).T
freqs_range_225_to_163 = np.linspace(log10(229), np.log10(163), 100)
extrapolated_ps_225_to_163 = np.array([np.polyval(np.polyfit(freqs,multi_frequency_ps_data[i],n_poly), freqs_range_225_to_163) for i in range(1)])
extrapolated_ps_225_to_163_min = np.squeeze([np.polyval(np.polyfit(freqs,multi_frequency_ps_data_min[i],n_poly), freqs_range_225_to_163) for i in range(1)])
extrapolated_ps_225_to_163_max = np.squeeze([np.polyval(np.polyfit(freqs,multi_frequency_ps_data_max[i],n_poly), freqs_range_225_to_163) for i in range(1)])

multi_frequency_ps_data_errors = 10.**multi_frequency_ps_data*log(10.0)*multi_frequency_ps_data_errors
extrapolated_ps_163_uncertainty = 10.**(extrapolated_ps_163+delta_extrapolated_ps_163_plus)*1-10.**extrapolated_ps_163

def tick_function(nu):
    redshifts = (1400./nu)-1.0
    return ["%.2f" % z for z in redshifts]


fig = P.figure(figsize=(10,10))
ax1 = fig.add_subplot(111)
ax1.errorbar(10.**freqs, 10.**multi_frequency_ps_data[0], multi_frequency_ps_data_errors[0]*1, fmt='o', color='black', linewidth=1.5, label='Measured low-$z$ foreground contamination')
ax1.errorbar(10.**freqs_range_225_to_163, 10.**extrapolated_ps_225_to_163, fmt='--', color='red', linewidth=1.5, label='Log-quadratic model for foreground \ncontamination')
ax1.errorbar(10.**fiducial_freq, 10.**extrapolated_ps_163[0], extrapolated_ps_163_uncertainty[0], fmt='o', color='red', linewidth=1.5, label='Model for contamination of the EoR \npower spectrum by foregrounds at $z=7.6$')
ax1.fill_between(10.**freqs_range_225_to_163, 10.**extrapolated_ps_225_to_163_max, 10.**extrapolated_ps_225_to_163_min, color='gray', alpha=0.5)
ax1.set_xlabel('$\\nu[\mathrm{MHz}]$', size='23')
ax1.set_ylabel('$\Delta_{k}^{2}[\mathrm{mK}^{2}]$', size='23')
ax1.invert_xaxis()
ax1.set_xscale('log')
ax1.set_yscale('log')
xticks = np.round(np.linspace((224), (163), 6),0)
ax1.set_xticks(xticks)
ax1.get_xaxis().set_major_formatter(P.matplotlib.ticker.FormatStrFormatter('%0.1f'))
ax1.get_yaxis().set_major_formatter(P.matplotlib.ticker.FormatStrFormatter('%0.1f'))
P.matplotlib.pyplot.grid(True, which="both")
ax2 = ax1.twiny()
ax2.set_xscale('log')
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(xticks)
ax2.set_xticklabels(tick_function(xticks))
ax2.set_xlabel(r"Redshift", size='23')
ax1.xaxis.set_minor_formatter(P.matplotlib.ticker.NullFormatter())
ax1.yaxis.set_minor_formatter(P.matplotlib.ticker.NullFormatter())
ax2.xaxis.set_minor_formatter(P.matplotlib.ticker.NullFormatter())
ax1.tick_params(axis='both', labelsize=20)
ax2.tick_params(axis='both', labelsize=20)
ylim = ax1.get_ylim()
ax1.vlines([200.0], *ax1.get_ylim())
ax1.set_ylim(ylim)
# bbox_props = dict(boxstyle="larrow", fc=(0.8,0.9,0.9), ec="black", lw=2)
# t = ax1.text(200, 1.0, "Post-reionization", ha="center", va="center", rotation=0, size=15, bbox=bbox_props, fontsize=16)

ax1.legend(loc='upper left', bbox_to_anchor=(0.0,0.9), fontsize=17)
bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
ax1.text(214, 4.0, "$\\log_{10}(k[\mathrm{h^{-1}Mpc}])=-1.38$", ha="center", va="center", size=20,
        bbox=bbox_props)
P.tight_layout()
print 'Outputting to:', 'Log_quadratic_foregroung_contamination_model_log_axes.png'
P.savefig('Log_quadratic_foregroung_contamination_model_log_axes.png')
# 
P.show()




np.savetxt('delta_extrapolated_ps_163_plus.dat',delta_extrapolated_ps_163_plus)
np.savetxt('delta_extrapolated_ps_163_minus.dat',delta_extrapolated_ps_163_minus)

np.savetxt('extrapolated_ps_163_Sph_component.dat',extrapolated_ps_163_sph)
np.savetxt('extrapolated_ps_163_OCB_component.dat',extrapolated_ps_163_OCB)




###
# Log-quadratic spectral model with Sph+OCB foreground power spectra
###
import numpy as np
from numpy import *

k_vals = np.loadtxt('/gpfs/data/jpober/psims/EoR/Fortran/PolyChord/PolyChord1d10/PPSimpleEor_OCB3_Comet/k_vals')
EoR = np.loadtxt('SQ-N10d0-WN0-Sph-EoR-18f-10B-v1d0-_posterior_weighted_means_and_standard_deviations.dat')
EoRFg = np.loadtxt('SQ-N10d0-WN0-Sph-EoR163CombFg62-18f-10B-v1d0-_posterior_weighted_means_and_standard_deviations.dat')
EoRFg_FgP = np.loadtxt('SQ-N10d0-WN0-Sph-EoR159CombFg66-LQSphP-18f-10B-v1d0-_posterior_weighted_means_and_standard_deviations.dat')

log_extrapolated_ps_163_uncertainty = EoRFg_FgP[0,0]-log10(10.**EoRFg_FgP[0,0]-extrapolated_ps_163_uncertainty[0])
EoRFg_FgP[0,1] = (EoRFg_FgP[0,1]**2.0 + log_extrapolated_ps_163_uncertainty**2.0)**0.5 #Add errors in quadrature

import pylab as P
from matplotlib import pyplot as plt

P.figure(figsize=(10,10))
P.errorbar(log10(k_vals)[:], EoR[:,0], fmt='--', color='blue', linewidth=1.5, label='Simulated EoR power spectrum at $z=7.6$')
# P.errorbar(log10(k_vals)[:], EoR[:,0], EoR[:,1]*3, fmt='o', color='blue', linewidth=1.5, label='Simulated EoR power spectrum at $z=7.6$')
P.errorbar(log10(k_vals)[:]-0.01, EoRFg[:,0], EoRFg[:,1]*3, fmt='o', color='black', linewidth=1.5, label='Recovered EoR power without foreground \npower spectrum modelling')
# P.errorbar(log10(k_vals)[:]-0.02, EoRFg_FgP[:,0], EoRFg_FgP[:,1], fmt='o', color='red', linewidth=1.5)
#
P.tick_params(labelsize=20)
plt.xlabel('$\\log_{10}(k[h\\mathrm{Mpc}^{-1}])$', size='23')
plt.ylabel('$\\log_{10}(\Delta_{k}^{2}[\mathrm{mK}^{2}])$', size='23')
#
P.ylim([0.4,1.5])
P.legend(loc='lower right', fontsize=17)
print 'Outputting to:', 'EoR159CombFg66.png'
P.savefig('EoR159CombFg66.png')
P.show()


P.figure(figsize=(10,10))
P.errorbar(log10(k_vals)[:], EoR[:,0], fmt='--', color='blue', linewidth=1.5, label='Simulated EoR power spectrum at $z=7.6$')
# P.errorbar(log10(k_vals)[:], EoR[:,0], EoR[:,1]*3, fmt='o', color='blue', linewidth=1.5, label='Simulated EoR power spectrum at $z=7.6$')
P.errorbar(log10(k_vals)[:]-0.01, EoRFg[:,0], EoRFg[:,1]*3, fmt='o', color='black', linewidth=1.5, label='Recovered EoR power without foreground \npower spectrum modelling')
P.errorbar(log10(k_vals)[:]-0.02, EoRFg_FgP[:,0], EoRFg_FgP[:,1]*3, fmt='o', color='red', linewidth=1.5, label='Recovered EoR power with foreground \npower spectrum modelling')
#
P.tick_params(labelsize=20)
plt.xlabel('$\\log_{10}(k[h\\mathrm{Mpc}^{-1}])$', size='23')
plt.ylabel('$\\log_{10}(\Delta_{k}^{2}[\mathrm{mK}^{2}])$', size='23')
P.ylim([0.4,1.5])
P.legend(loc='lower right', fontsize=17)
#
print 'Outputting to:', 'EoR159CombFg66_LQSphP.png'
P.savefig('EoR159CombFg66_LQSphP.png')
P.show()








import numpy as np
from numpy import *

k_vals = np.loadtxt('/gpfs/data/jpober/psims/EoR/Fortran/PolyChord/PolyChord1d10/PPSimpleEor_OCB3_Comet/k_vals')
EoRnlf1 = np.loadtxt('SQ-N10d0-WN0-Sph-EoR-18f-10B-nlf1-v1d0-_posterior_weighted_means_and_standard_deviations.dat')
EoRnlf3 = np.loadtxt('SQ-N10d0-WN0-Sph-EoR-18f-10B-v1d0-_posterior_weighted_means_and_standard_deviations.dat')


import pylab as P
from matplotlib import pyplot as plt

P.figure(figsize=(10,10))
P.errorbar(log10(k_vals)[:], EoRnlf1[:,0], EoRnlf1[:,1], fmt='o', color='blue', linewidth=1.5, label='nlf1')
P.errorbar(log10(k_vals)[:], EoRnlf3[:,0], EoRnlf3[:,1], fmt='o', color='black', linewidth=1.5, label='nlf3')
P.errorbar(log10(k_vals)[:], EoRnlf1[:,0], fmt='--', color='blue', linewidth=1.5)
P.errorbar(log10(k_vals)[:], EoRnlf3[:,0], fmt='--', color='black', linewidth=1.5)
P.tick_params(labelsize=20)
plt.xlabel('$\\log_{10}(k[h\\mathrm{Mpc}^{-1}])$', size='23')
plt.ylabel('$\\log_{10}(\Delta_{k}^{2}[\mathrm{mK}^{2}])$', size='23')
#
P.ylim([0.4,1.5])
P.legend(loc='lower right', fontsize=17)
# print 'Outputting to:', 'EoR159CombFg66.png'
# P.savefig('EoR159CombFg66.png')
P.show()








import numpy as np
from numpy import *

k_vals = np.loadtxt('/users/psims/EoR/Python_Scripts/BayesEoR/SpatialPS/PolySpatialPS/likelihood_tests/SimpleEoRtestWQ/k_vals.txt')
EoR_nq2 = np.loadtxt('EoR_z10d2_nq2-_posterior_weighted_means_and_standard_deviations.dat')
EoR_GalFg_nq2 = np.loadtxt('EoR_z10d2_GalFg_beta_2d63_dbeta_0d1_nq2-_posterior_weighted_means_and_standard_deviations.dat')
EoR_GalFg_nq1_pl1 = np.loadtxt('EoR_z10d2_GalFg_beta_2d63_dbeta_0d02_nq1_pl1-_posterior_weighted_means_and_standard_deviations.dat')

import pylab as P
from matplotlib import pyplot as plt

P.figure(figsize=(10,10))
P.errorbar(log10(k_vals)[:], EoR_nq2[:,0], fmt='--', color='black', linewidth=1.5)
P.errorbar(log10(k_vals)[:], EoR_GalFg_nq2[:,0], EoR_GalFg_nq2[:,1], fmt='o', color='blue', linewidth=1.5, label='nq2')
P.errorbar(log10(k_vals)[:], EoR_GalFg_nq1_pl1[:,0], EoR_GalFg_nq1_pl1[:,1], fmt='o', color='red', linewidth=1.5, label='nq1pl1')
P.errorbar(log10(k_vals)[:], EoR_GalFg_nq2[:,0], fmt='--', color='blue', linewidth=1.5)
P.errorbar(log10(k_vals)[:], EoR_GalFg_nq1_pl1[:,0], fmt='--', color='red', linewidth=1.5)
P.tick_params(labelsize=20)
plt.xlabel('$\\log_{10}(k[h\\mathrm{Mpc}^{-1}])$', size='23')
plt.ylabel('$\\log_{10}(\Delta_{k}^{2}[\mathrm{mK}^{2}])$', size='23')
#
P.legend(loc='lower right', fontsize=17)
print 'Outputting to:', 'EoR_159_Fg66_quad_pl_comparison.png'
P.savefig('EoR_159_Fg66_quad_pl_comparison.png')
P.show()



Tb_nu = Tb_nu[:,:496,:496]

import numpy
axes_tuple = (1,2)
vfft1=numpy.fft.ifftshift(Tb_nu[0:nf]-Tb_nu[0].mean()+0j, axes=axes_tuple)
vfft1=numpy.fft.fftn(vfft1, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
vfft1=numpy.fft.fftshift(vfft1, axes=axes_tuple)

sci_f, sci_v, sci_u = vfft1.shape
sci_v_centre = sci_v/2
sci_u_centre = sci_u/2
vfft1_subset = vfft1[0:nf,sci_u_centre-nu/2:sci_u_centre+nu/2+1,sci_v_centre-nv/2:sci_v_centre+nv/2+1]





bl_size = Tb_nu[0].shape[0]/31
dat_subset = 31*bl_size
averaged_cube = []
for i_freq in range(len(Tb_nu)):
	# Tb_nu_bls = np.array([Tb_nu[0][:496,:496][i*bl_size:(i+1)*bl_size,j*bl_size:(j+1)*bl_size] for i in range(31) for j in range(31)])
	Tb_nu_bls = np.array([Tb_nu[i_freq][:dat_subset,:dat_subset][i*bl_size:(i+1)*bl_size,j*bl_size:(j+1)*bl_size] for i in range(31) for j in range(31)])
	Tb_nu_bls_means = np.array([x.mean() for x in Tb_nu_bls]).reshape(31,31)
	averaged_cube.append(Tb_nu_bls_means)

averaged_cube = np.array(averaged_cube)

import numpy
axes_tuple = (1,2)
vfft2=numpy.fft.ifftshift(averaged_cube[0:nf]-averaged_cube[0].mean()+0j, axes=axes_tuple)
vfft2=numpy.fft.fftn(vfft2, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
vfft2=numpy.fft.fftshift(vfft2, axes=axes_tuple)

sci_f, sci_v, sci_u = vfft2.shape
sci_v_centre = sci_v/2
sci_u_centre = sci_u/2
vfft2_subset = vfft2[0:nf,sci_u_centre-nu/2:sci_u_centre+nu/2+1,sci_v_centre-nv/2:sci_v_centre+nv/2+1]





import numpy as np
from numpy import *

k_vals = np.loadtxt('/users/psims/EoR/Python_Scripts/BayesEoR/SpatialPS/PolySpatialPS/likelihood_tests/SimpleEoRtestWQ/k_vals.txt')
GalFg_beta_2d63_dbeta_0d002_pl1 = np.loadtxt('GalFg_beta_2d63_dbeta_0d002_pl1-_posterior_weighted_means_and_standard_deviations.dat')
GalFg_beta_2d63_dbeta_0d01_pl1 = np.loadtxt('GalFg_beta_2d63_dbeta_0d01_pl1-_posterior_weighted_means_and_standard_deviations.dat')
GalFg_beta_2d63_dbeta_0d02_pl1 = np.loadtxt('GalFg_beta_2d63_dbeta_0d02_pl1-_posterior_weighted_means_and_standard_deviations.dat')
GalFg_beta_2d63_dbeta_0d1_pl1 = np.loadtxt('GalFg_beta_2d63_dbeta_0d1_pl1-_posterior_weighted_means_and_standard_deviations.dat')



import pylab as P
from matplotlib import pyplot as plt

P.figure(figsize=(10,10))
# P.errorbar(log10(k_vals)[:], GalFg_beta_2d63_dbeta_0d002_pl1[:,0], GalFg_beta_2d63_dbeta_0d002_pl1[:,1], fmt='o', color='blue', linewidth=1.5, label='nq2')
# P.errorbar(log10(k_vals)[:], GalFg_beta_2d63_dbeta_0d02_pl1[:,0], GalFg_beta_2d63_dbeta_0d02_pl1[:,1], fmt='o', color='red', linewidth=1.5, label='nq1pl1')
# P.errorbar(log10(k_vals)[:], GalFg_beta_2d63_dbeta_0d02_pl1[:,0]+0.4, yerr=[0.4*np.ones(len(GalFg_beta_2d63_dbeta_0d02_pl1[:,0])), 0.0*np.ones(len(GalFg_beta_2d63_dbeta_0d02_pl1[:,0]))], color='darkblue', fmt='-', lolims=GalFg_beta_2d63_dbeta_0d02_pl1[:,0], linestyle = '', linewidth=1.5, capsize=5)
P.errorbar(log10(k_vals)[:], GalFg_beta_2d63_dbeta_0d02_pl1[:,0]+0.4, yerr=0.4, uplims=True, linestyle='')

P.tick_params(labelsize=20)
plt.xlabel('$\\log_{10}(k[h\\mathrm{Mpc}^{-1}])$', size='23')
plt.ylabel('$\\log_{10}(\Delta_{k}^{2}[\mathrm{mK}^{2}])$', size='23')
#
# P.ylim([0.4,1.5])
P.legend(loc='lower right', fontsize=17)
# print 'Outputting to:', 'EoR_159_Fg66_quad_pl_comparison.png'
# P.savefig('EoR_159_Fg66_quad_pl_comparison.png')
P.show()





import numpy as np
from numpy import *
import pylab as P

PS_21cmFast_data = np.loadtxt('/users/psims/EoR/EoR_simulations/21cmFAST_512MPc_512pix_128pix/Output_files/Deldel_T_power_spec/ps_no_halos_z010.20_nf0.841350_useTs0_zetaX-1.0e+00_alphaX-1.0_TvirminX-1.0e+00_aveTb022.72_Pop-1_128_512Mpc_v3')
PS_21cmFast_k_vals = np.log10(PS_21cmFast_data[:,0])
PS_21cmFast_PS = np.log10(PS_21cmFast_data[:,1])
PS_21cmFast_uncertainties = PS_21cmFast_data[:,2]/(PS_21cmFast_data[:,1]*log(10.0))

k_vals_nf128 = np.loadtxt('/users/psims/EoR/Python_Scripts/BayesEoR/SpatialPS/PolySpatialPS/likelihood_tests/SimpleEoRtestWQ/k_vals_nf128.txt')
EoR_nf128_nq0 = np.loadtxt('/users/psims/EoR/Python_Scripts/BayesEoR/SpatialPS/PolySpatialPS/likelihood_tests/SimpleEoRtestWQ/chains/Stats/EoR_nf128_nq0-_posterior_weighted_means_and_standard_deviations.dat')
EoR_nf128_nq2 = np.loadtxt('/users/psims/EoR/Python_Scripts/BayesEoR/SpatialPS/PolySpatialPS/likelihood_tests/SimpleEoRtestWQ/chains/Stats/EoR_nf128_nq2-_posterior_weighted_means_and_standard_deviations.dat')


P.figure(figsize=(10,10))
P.errorbar(PS_21cmFast_k_vals, PS_21cmFast_PS, PS_21cmFast_uncertainties, fmt='o', color='red', linewidth=1.5, label='21cmFast')
P.errorbar(PS_21cmFast_k_vals, PS_21cmFast_PS, fmt='--', color='red', linewidth=1.5)
P.errorbar(np.log10(k_vals_nf128), EoR_nf128_nq0[:,0], EoR_nf128_nq0[:,1], fmt='o', color='blue', linewidth=1.5, label='nq0')
P.errorbar(np.log10(k_vals_nf128)[1:], EoR_nf128_nq2[:,0][1:], EoR_nf128_nq2[:,1][1:], fmt='o', color='black', linewidth=1.5, label='nq2')

P.tick_params(labelsize=20)
P.xlabel('$\\log_{10}(k[h\\mathrm{Mpc}^{-1}])$', size='23')
P.ylabel('$\\log_{10}(\Delta_{k}^{2}[\mathrm{mK}^{2}])$', size='23')
#
P.legend(loc='lower right', fontsize=17)
P.show()







import numpy as np
from numpy import *
import pylab as P

PS_21cmFast_data = np.loadtxt('/users/psims/EoR/EoR_simulations/21cmFAST_512MPc_128pix_32pix/Output_files/Deldel_T_power_spec/ps_no_halos_z007.60_nf0.422668_useTs0_zetaX-1.0e+00_alphaX-1.0_TvirminX-1.0e+00_aveTb009.94_Pop-1_32_512Mpc_v3')
PS_21cmFast_k_vals = np.log10(PS_21cmFast_data[:,0])
PS_21cmFast_PS = np.log10(PS_21cmFast_data[:,1])
PS_21cmFast_uncertainties = PS_21cmFast_data[:,2]/(PS_21cmFast_data[:,1]*log(10.0))

k_vals_nu_31_nv_31_nf_32_nq_0 = np.loadtxt('/users/psims/EoR/Python_Scripts/BayesEoR/git_version/BayesEoR/spec_model_tests/k_vals/k_vals_nu_31_nv_31_nf_32_nq_0.txt')
EoR_nu_31_nv_31_nf32_nq0 = np.loadtxt('EoR_nu_31_nv_31_nf32_nq0-_posterior_weighted_means_and_standard_deviations.dat')
EoR_nu_31_nv_31_nf32_nq2 = np.loadtxt('EoR_nu_31_nv_31_nf32_nq2-_posterior_weighted_means_and_standard_deviations.dat')

P.figure(figsize=(10,10))
P.errorbar(PS_21cmFast_k_vals, PS_21cmFast_PS, PS_21cmFast_uncertainties, fmt='o', color='red', linewidth=1.5, label='21cmFast')
P.errorbar(PS_21cmFast_k_vals, PS_21cmFast_PS, fmt='--', color='red', linewidth=1.5)
P.errorbar(np.log10(k_vals_nu_31_nv_31_nf_32_nq_0), EoR_nu_31_nv_31_nf32_nq0[:,0], EoR_nu_31_nv_31_nf32_nq0[:,1], fmt='o', color='blue', linewidth=1.5, label='nq0')
P.errorbar(np.log10(k_vals_nu_31_nv_31_nf_32_nq_0)[1:], EoR_nu_31_nv_31_nf32_nq2[:,0][1:], EoR_nu_31_nv_31_nf32_nq2[:,1][1:], fmt='o', color='black', linewidth=1.5, label='nq2')

P.tick_params(labelsize=20)
P.xlabel('$\\log_{10}(k[h\\mathrm{Mpc}^{-1}])$', size='23')
P.ylabel('$\\log_{10}(\Delta_{k}^{2}[\mathrm{mK}^{2}])$', size='23')
#
P.legend(loc='lower right', fontsize=17)
P.show()






#----------------------
#----------------------




	


###
# Generate data = s+n (with known s and n), to estimate the power spectrum of
###
test_sim_out = generate_test_sim_signal_with_large_spectral_scales_2_21cmFAST_Binning(nu,nv,nx,ny,nf,neta,nq)
if small_cube or True:
	s, s_im, s_quad_only, s_im_quad_only, s_fourier_only, s_im_fourier_only, bin_selector_in_k_cube_mask, high_spatial_frequency_selector_mask, k_cube_signal, k_sigma, ZM_mask, k_z_mean_mask = test_sim_out
	bin_selector_cube_ordered_list = bin_selector_in_k_cube_mask
else:
	s=test_sim_out[0]
	bin_selector_cube_ordered_list = test_sim_out[6]
	high_spatial_frequency_selector_mask = test_sim_out[7]
	k_sigma = test_sim_out[9]
	ZM_mask = test_sim_out[	10]
	k_z_mean_mask = test_sim_out[11]
test_sim_out=0

if nq==0:
	map_bins_out = map_out_bins_for_power_spectral_coefficients_HERA_Binning(nu,nv,nx,ny,nf,neta,nq, bin_selector_cube_ordered_list,high_spatial_frequency_selector_mask, ZM_mask, k_z_mean_mask)
	bin_selector_in_model_mask_vis_ordered = map_bins_out
	bin_selector_vis_ordered_list = bin_selector_in_model_mask_vis_ordered
else:
	map_bins_out = map_out_bins_for_power_spectral_coefficients_WQ_v2_HERA_Binning(nu,nv,nx,ny,nf,neta,nq, bin_selector_cube_ordered_list,high_spatial_frequency_selector_mask, ZM_mask, k_z_mean_mask)
	bin_selector_in_model_mask_vis_ordered_WQ, Quad_modes_only_boolean_array_vis_ordered = map_bins_out
	bin_selector_vis_ordered_list = bin_selector_in_model_mask_vis_ordered_WQ
map_bins_out=0

k = generate_k_cube_in_physical_coordinates_21cmFAST(nu,nv,nx,ny,nf,neta)[0]
k_vis_ordered = k.T.flatten()
modk_vis_ordered_list = [k_vis_ordered[bin_selector.T.flatten()] for bin_selector in bin_selector_cube_ordered_list]


###
# Map out power spectrum bins
###
mod_k, k_x, k_y, k_z, deltakperp, deltakpara, x, y, z = generate_k_cube_in_physical_coordinates_21cmFAST(nu,nv,nx,ny,nf,neta)
k_x_masked = generate_masked_coordinate_cubes(k_x, nu,nv,nx,ny,nf,neta,nq)
k_y_masked = generate_masked_coordinate_cubes(k_y, nu,nv,nx,ny,nf,neta,nq)
k_z_masked = generate_masked_coordinate_cubes(k_z, nu,nv,nx,ny,nf,neta,nq)
mod_k_masked = generate_masked_coordinate_cubes(mod_k, nu,nv,nx,ny,nf,neta,nq)

k_cube_voxels_in_bin, modkbins_containing_voxels = generate_k_cube_model_spherical_binning(mod_k_masked, k_z_masked, nu,nv,nx,ny,nf,neta,nq)

# do_cylindrical_binning = True
do_cylindrical_binning = False
if do_cylindrical_binning:
	n_k_perp_bins=2
	k_cube_voxels_in_bin, modkbins_containing_voxels, k_perp_bins = generate_k_cube_model_cylindrical_binning(mod_k_masked, k_z_masked, k_y_masked, k_x_masked, n_k_perp_bins, nu,nv,nx,ny,nf,neta,nq)





#----------------------

use_foreground_cube = True
# use_foreground_cube = False

if use_foreground_cube:
	###
	# beta_experimental_mean = 2.55+0 #Used in my `Jelic' GDSE models before 5th October 2017
	beta_experimental_mean = 2.63+0   #Matches beta_150_408 in Mozden, Bowman et al. 2016
	# beta_experimental_std  = 0.1    #Used in my `Jelic' GDSE models before 5th October 2017
	beta_experimental_std  = 0.02      #A conservative over-estimate of the dbeta_150_408=0.01 (dbeta_90_190=0.02) in Mozden, Bowman et al. 2016
	gamma_mean             = -2.7     #Revise to match published values
	gamma_sigma            = 0.3      #Revise to match published values
	Tb_experimental_mean_K = 194.0    #Matches GSM mean in region A (see /users/psims/Cav/EoR/Missing_Radio_Flux/Surveys/Convert_GSM_to_HEALPIX_Map_and_Cartesian_Projection_Fits_File_v6d0_pygsm.py)
	# Tb_experimental_std_K  = 85.0     #65th percentile std at at 0.333 degree resolution in 50 deg by 50 deg maps centered on Dec=-30.0
	Tb_experimental_std_K  = 62.0   #Median std at at 0.333 degree resolution in 50 deg by 50 deg maps centered on Dec=-30.0
	# Tb_experimental_std_K  = 23.0   #Matches GSM in region A at 0.333 degree resolution (i.e. for a 50 degree map 150 pixels across). Note: std. is a function of resultion so the foreground map should be made at the same resolution for this std normalisation to be accurate
	# Tb_experimental_mean_K = 240.0  #Revise to match published values
	# Tb_experimental_std_K  = 4.0    #Revise to match published values
	nu_min_MHz             = 163.0-4.0
	# nu_min_MHz             = 163.0
	# nu_min_MHz             = 225.0
	Tb_experimental_std_K = Tb_experimental_std_K*(nu_min_MHz/163.)**-beta_experimental_mean
	channel_width_MHz      = 0.2
	# fits_storage_dir = 'fits_storage/Jelic_TbStd_{}_beta_{}_dbeta{}/'.format(Tb_experimental_std_K, beta_experimental_mean, beta_experimental_std).replace('.','d')
	# fits_storage_dir = 'fits_storage/fractional_frequency_band_test/Jelic_nu_min_MHz_{}_TbStd_{}_beta_{}_dbeta{}/'.format(nu_min_MHz, Tb_experimental_std_K, beta_experimental_mean, beta_experimental_std).replace('.','d')
	fits_storage_dir = 'fits_storage/multi_frequency_band_test3/Jelic_nu_min_MHz_{}_TbStd_{}_beta_{}_dbeta{}/'.format(nu_min_MHz, Tb_experimental_std_K, beta_experimental_mean, beta_experimental_std).replace('.','d')
	###
	HF_nu_min_MHz_array = [210,220,230]
	# HF_nu_min_MHz_array = [205,215,225]
	foreground_outputs = generate_Jelic_cube(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show, beta_experimental_mean,beta_experimental_std,gamma_mean,gamma_sigma,Tb_experimental_mean_K,Tb_experimental_std_K,nu_min_MHz,channel_width_MHz, generate_additional_extrapolated_HF_foreground_cube=True, fits_storage_dir=fits_storage_dir, HF_nu_min_MHz_array=HF_nu_min_MHz_array)
	# foreground_outputs = generate_Jelic_cube(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show, beta_experimental_mean,beta_experimental_std,gamma_mean,gamma_sigma,Tb_experimental_mean_K,Tb_experimental_std_K,nu_min_MHz,channel_width_MHz)
	fg, s, Tb_nu, beta_experimental_mean,beta_experimental_std,gamma_mean,gamma_sigma,Tb_experimental_mean_K,Tb_experimental_std_K,nu_min_MHz,channel_width_MHz, HF_Tb_nu = foreground_outputs
	# fg, s, Tb_nu, z = foreground_outputs[0:3]
	foreground_outputs = []



print (Tb_nu[0]/HF_Tb_nu[0]).std()
print (Tb_nu[0]/(HF_Tb_nu[0]*(nu_min_MHz/225.)**-beta_experimental_mean)).std()
residuals = Tb_nu[0] - (HF_Tb_nu[0]*(nu_min_MHz/225.)**-beta_experimental_mean)

Show=False
pylab.close('all')
pylab.figure()
pylab.imshow(Tb_nu[0])
if Show: pylab.colorbar()

pylab.figure()
pylab.imshow(residuals)
pylab.colorbar()
if Show: pylab.show()

print 100.*(residuals/Tb_nu[0]).mean()








###
# GRN foreground test
###
# random_seed = 314159   #For Fg3
# random_seed = 3141591  #For Fg4
random_seed = 31415911 #For Fg5
GFC = GenerateForegroundCube(nu,nv,neta,nq, beta_experimental_mean, beta_experimental_std, gamma_mean, gamma_sigma, Tb_experimental_mean_K, Tb_experimental_std_K, nu_min_MHz, channel_width_MHz, random_seed=random_seed, k_parallel_scaling=True)

# gamma_m, gamma_s = (-6, 1.e-3)
gamma_m, gamma_s = (-3.0, 1.e-3)
impix=31
foreground_k_space_amplitudes_cube = GFC.generate_GRN_for_A_and_beta_fields(impix,impix,impix,impix,nf,neta,nq, gamma_m, gamma_s)

axes_tuple=(0,1,2)
unnormalised_A=numpy.fft.ifftshift(foreground_k_space_amplitudes_cube+0j, axes=axes_tuple)
unnormalised_A=numpy.fft.fftn(unnormalised_A, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
unnormalised_A=numpy.fft.fftshift(unnormalised_A, axes=axes_tuple)


pylab.figure()
pylab.imshow(log10(abs(foreground_k_space_amplitudes_cube[10])), vmin=-3,vmax=1.5)
pylab.colorbar()

pylab.figure()
pylab.imshow(log10(abs(foreground_k_space_amplitudes_cube[:,16,16-16:16+15])), vmin=-3,vmax=1.5)
pylab.colorbar()
# pylab.show()


# GRN_total_power = 3.e6 #Note, this was 3.e6 for original Jelic cube
GRN_total_power = 1.e6 #Note, this was 3.e6 for original Jelic cube
# GRN_total_power = 1.e2
GRN_fg = unnormalised_A.real*(GRN_total_power**0.5)/unnormalised_A.real.std()

pylab.figure()
pylab.imshow(log10(abs(GRN_fg[10]))*2)
pylab.colorbar()
pylab.show()

pylab.figure()
for i in range(5):pylab.errorbar(arange(38),GRN_fg[:,16,16+i])
pylab.show()

pylab.hist(GRN_fg.flatten())
pylab.show()

WD2F = WriteDataToFits()
WD2F.write_data(GRN_fg, 'fits_storage/GRN/Fg5/GRN_foreground_cube_v1d0.fits', Box_Side_cMpc=2048, simulation_redshift=7.6)
# WD2F.write_data(GRN_fg, 'fits_storage/GRN/Fg2/GRN_foreground_cube_v1d0.fits', Box_Side_cMpc=2048, simulation_redshift=7.6)
# WD2F.write_data(GRN_fg, 'fits_storage/GRN/EoR2/GRN_EoR_cube_v1d1.fits', Box_Side_cMpc=2048, simulation_redshift=7.6)



NGRN_fg = (GRN_fg+GRN_fg**2.-(GRN_fg**2.).mean())
NGRN_fg = NGRN_fg*GRN_fg.std()/NGRN_fg.std()

pylab.figure()
pylab.imshow(log10(abs(NGRN_fg[10]))*2)
pylab.colorbar()
pylab.show()

pylab.figure()
for i in range(5):pylab.errorbar(arange(38),NGRN_fg[:,16,16+i])
pylab.show()

pylab.hist((GRN_fg+GRN_fg**2.-(GRN_fg**2.).mean()).flatten())
pylab.show()

WD2F.write_data(NGRN_fg, 'fits_storage/NGRN/Fg5/NGRN_foreground_cube_v1d0.fits', Box_Side_cMpc=2048, simulation_redshift=7.6)


###






# mod_k_big = generate_k_cube_in_physical_coordinates_21cmFAST(129,129,129,129,nf,neta)[0]
# GRNH = construct_GRN_unitary_hermitian_k_cube(129,129,neta,nq)
# SGRNH = GRNH/(1.e-10+mod_k_big**3.)
# axes_tuple=(0,1,2)
# vfft1=numpy.fft.ifftshift(SGRNH[0:nf], axes=axes_tuple)
# vfft1=numpy.fft.ifftn(vfft1, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
# vfft1=numpy.fft.fftshift(vfft1, axes=axes_tuple)
# vfft1 = vfft1.real - vfft1.real.mean()
# vfft1 = vfft1*10.0*Tb_nu.std()/vfft1.std()
# # s = generate_test_signal_from_image_cube(nu,nv,nx,ny,nf,neta,nq,vfft1.real)
# image_cube_mK = vfft1
# output_fits_file_name = 'mock_EoR_cube_v1d0.fits'
# generate_test_signal_from_image_cube(129,129,129,129,nf,neta,nq,image_cube_mK,output_fits_file_name)


# use_EoR_cube = True
use_EoR_cube = False
if use_EoR_cube:
	print 'Using use_EoR_cube data'
	#----------------------
	###
	# Replace Gaussian signal with EoR cube
	###
	s, abc, scidata1 = generate_data_from_loaded_EoR_cube(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show,chan_selection)


	#----------------------



if use_EoR_cube:
	d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector(sigma, abc)[0]
elif use_foreground_cube:
	d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector(sigma, fg)[0]
else:
	d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector(sigma, s)[0]




###
# Load base matrices used in the likelihood
###

T_Ninv_T = BM.read_data_from_hdf5(array_save_directory+'T_Ninv_T.h5', 'T_Ninv_T')
T = BM.read_data_from_hdf5(array_save_directory+'T.h5', 'T')
block_T_Ninv_T = BM.read_data_from_hdf5(array_save_directory+'block_T_Ninv_T.h5', 'block_T_Ninv_T')
Ninv = BM.read_data_from_hdf5(array_save_directory+'Ninv.h5', 'Ninv')
Ninv_d = np.dot(Ninv,d)
dbar = np.dot(T.conjugate().T,Ninv_d)

Ninv=[]





###
# Define some helper variables and the posterior probability class
###
Sigma_Diag_Indices=np.diag_indices(shape(T_Ninv_T)[0])
Npar = shape(T_Ninv_T)[0]
x=[100.e0]*len(bin_selector_vis_ordered_list)
nuv = (nu*nv-1)
block_T_Ninv_T = np.array([np.hsplit(block,nuv) for block in np.vsplit(T_Ninv_T,nuv)])
if nq==0:
	masked_power_spectral_modes = np.logical_not(np.logical_or.reduce(bin_selector_in_model_mask_vis_ordered).reshape(-1,neta)).flatten()
else:
	masked_power_spectral_modes = Quad_modes_only_boolean_array_vis_ordered
from numpy import real



indices_x, indices_y = np.mgrid[0:nuv*nf,0:nuv*nf]
block_diag_indices_x = np.array([np.hsplit(block,nuv) for block in np.vsplit(indices_x,nuv)])[np.diag_indices(nuv)]
block_diag_indices_y = np.array([np.hsplit(block,nuv) for block in np.vsplit(indices_y,nuv)])[np.diag_indices(nuv)]
indices_x=[]
indices_y=[]



class PowerSpectrumPosteriorProbability(object):
	def __init__(self, T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, fit_single_elems=False, **kwargs):
		##===== Defaults =======
		default_diagonal_sigma = False
		default_block_T_Ninv_T=[]
		default_log_priors = False
		default_dimensionless_PS=False
		default_inverse_quadratic_power= 0.0
		default_Print=False
		
		##===== Inputs =======
		self.diagonal_sigma=kwargs.pop('diagonal_sigma',default_diagonal_sigma)
		self.block_T_Ninv_T=kwargs.pop('block_T_Ninv_T',default_block_T_Ninv_T)
		self.log_priors=kwargs.pop('log_priors',default_log_priors)
		if self.log_priors:print 'Using log-priors'
		self.dimensionless_PS=kwargs.pop('dimensionless_PS',default_dimensionless_PS)
		if self.dimensionless_PS:print 'Calculating dimensionless_PS'
		self.inverse_quadratic_power=kwargs.pop('inverse_quadratic_power',default_inverse_quadratic_power)
		self.Print=kwargs.pop('Print',default_Print)

		self.fit_single_elems      = fit_single_elems
		self.T_Ninv_T              = T_Ninv_T
		self.dbar                  = dbar
		self.Sigma_Diag_Indices    = Sigma_Diag_Indices
		self.diagonal_sigma        = False
		self.block_diagonal_sigma  = False
		self.instantiation_time    = time.time()
		self.count                 = 0
		self.Npar                  = Npar
		self.k_cube_voxels_in_bin  = k_cube_voxels_in_bin
		self.nuv  = nuv
		self.nu  = nu
		self.nv  = nv
		self.nx  = nx
		self.ny  = ny
		self.neta  = neta
		self.nf  = nf
		self.nq  = nq
		self.masked_power_spectral_modes  = masked_power_spectral_modes
		self.modk_vis_ordered_list = modk_vis_ordered_list

	def add_power_to_diagonals(self, T_Ninv_T_block, PhiI_block, **kwargs):
		return T_Ninv_T_block+np.diag(PhiI_block)

	def calc_Sigma_block_diagonals(self, T_Ninv_T, PhiI, **kwargs):
		PhiI_blocks = np.split(PhiI, self.nuv)
		Sigma_block_diagonals = np.array([self.add_power_to_diagonals(T_Ninv_T[(self.neta+self.nq)*i_block:(self.neta+self.nq)*(i_block+1),(self.neta+self.nq)*i_block:(self.neta+self.nq)*(i_block+1)], PhiI_blocks[i_block]) for i_block in range(self.nuv)])
		return Sigma_block_diagonals

	def calc_SigmaI_dbar_wrapper(self, x, T_Ninv_T, dbar, **kwargs):
		block_T_Ninv_T=[]

		##===== Inputs =======
		if 'block_T_Ninv_T' in kwargs:
			block_T_Ninv_T=kwargs['block_T_Ninv_T']
		
		start = time.time()
		PowerI = self.calc_PowerI(x)
		PhiI=PowerI
		if self.Print:print 'Time taken: {}'.format(time.time()-start)

		do_block_diagonal_inversion = len(shape(block_T_Ninv_T))>1
		if do_block_diagonal_inversion:
			if self.Print:print 'Using block-diagonal inversion'
			start = time.time()
			Sigma_block_diagonals = self.calc_Sigma_block_diagonals(T_Ninv_T, PhiI)
			if self.Print:print 'Time taken: {}'.format(time.time()-start)
			if self.Print:print 'nuv', self.nuv

			start = time.time()
			dbar_blocks = np.split(dbar, self.nuv)
			SigmaI_dbar_blocks = np.array([self.calc_SigmaI_dbar(Sigma_block_diagonals[i_block], dbar_blocks[i_block])  for i_block in range(self.nuv)])
			if self.Print:print 'Time taken: {}'.format(time.time()-start)
			
			SigmaI_dbar = SigmaI_dbar_blocks.flatten()
			dbarSigmaIdbar=np.dot(dbar.conjugate().T,SigmaI_dbar)
			if self.Print:print 'Time taken: {}'.format(time.time()-start)

			logSigmaDet=np.sum([np.linalg.slogdet(Sigma_block)[1] for Sigma_block in Sigma_block_diagonals])
			if self.Print:print 'Time taken: {}'.format(time.time()-start)

		else:
			print 'Not using block-diagonal inversion'
			start = time.time()
			###
			# Note: the following two lines can probably be speeded up by adding T_Ninv_T and np.diag(PhiI). (Should test this!) but this else statement only occurs on the GPU inversion so will deal with it later.
			###
			Sigma=T_Ninv_T.copy()
			Sigma[self.Sigma_Diag_Indices]+=PhiI
			if self.Print:print 'Time taken: {}'.format(time.time()-start)

			start = time.time()
			SigmaI_dbar = self.calc_SigmaI_dbar(Sigma, dbar)
			if self.Print:print 'Time taken: {}'.format(time.time()-start)

			dbarSigmaIdbar=np.dot(dbar.conjugate().T,SigmaI_dbar)
			if self.Print:print 'Time taken: {}'.format(time.time()-start)

			logSigmaDet=np.linalg.slogdet(Sigma)[1]
			if self.Print:print 'Time taken: {}'.format(time.time()-start)
			# logSigmaDet=2.*np.sum(np.log(np.diag(Sigmacho)))

		return SigmaI_dbar, dbarSigmaIdbar, PhiI, logSigmaDet

	def calc_SigmaI_dbar(self, Sigma, dbar, **kwargs):
		##===== Defaults =======
		block_T_Ninv_T=[]

		##===== Inputs =======
		if 'block_T_Ninv_T' in kwargs:
			block_T_Ninv_T=kwargs['block_T_Ninv_T']
		
		# Sigmacho = scipy.linalg.cholesky(Sigma, lower=True).astype(np.complex256)
		# SigmaI_dbar = scipy.linalg.cho_solve((Sigmacho,True), dbar)
		SigmaI = scipy.linalg.inv(Sigma)
		SigmaI_dbar = np.dot(SigmaI, dbar)
		return SigmaI_dbar
		
	def calc_dimensionless_power_spectral_normalisation_ltl(self, i_bin, **kwargs):
		EoRVolume = 770937185.063917
		Omega_map = ((12*np.pi/180.)**2)
		dimensionless_PS_scaling = (EoRVolume*self.modk_vis_ordered_list[i_bin]**3.)/(Omega_map**4 * (self.nu*self.nv*self.nf)**4)
		return dimensionless_PS_scaling

	def calc_dimensionless_power_spectral_normalisation_21cmFAST(self, i_bin, **kwargs):
		###
		# NOTE: the physical size of the cosmological box is simulation dependent. The values here are matched to the following 21cmFAST simulation:
		# /users/psims/EoR/EoR_simulations/21cmFAST_512MPc_512pix_128pix/ and will require updating if the input signal is changed to a new source
		# 21cmFAST normalisation:
		#define BOX_LEN (float) 512 // in Mpc
		#define VOLUME (BOX_LEN*BOX_LEN*BOX_LEN) // in Mpc^3
		# p_box[ct] += pow(k_mag,3)*pow(cabs(deldel_T[HII_C_INDEX(n_x, n_y, n_z)]), 2)/(2.0*PI*PI*VOLUME);
		###
		EoR_x_full_pix = 128 #pix (defined by input to 21cmFAST simulation)
		EoR_y_full_pix = 128 #pix (defined by input to 21cmFAST simulation)
		EoR_z_full_pix = 128 #pix (defined by input to 21cmFAST simulation)
		EoR_x_full_Mpc = 512. #Mpc (defined by input to 21cmFAST simulation)
		EoR_y_full_Mpc = 512. #Mpc (defined by input to 21cmFAST simulation)
		EoR_z_full_Mpc = 512. #Mpc (defined by input to 21cmFAST simulation)
		# EoR_analysis_cube_x_pix = EoR_x_full_pix #Mpc Analysing the full FoV in x
		# EoR_analysis_cube_y_pix = EoR_y_full_pix #Mpc Analysing the full FoV in y
		# EoR_analysis_cube_z_pix = 38 #Mpc Analysing 38 of the 128 channels of the full EoR_simulations 
		EoR_analysis_cube_x_pix = 128 #pix Analysing the full FoV in x
		EoR_analysis_cube_y_pix = 128 #pix Analysing the full FoV in y
		EoR_analysis_cube_z_pix = self.nf #Mpc Analysing 38 of the 128 channels of the full EoR_simulations 
		EoR_analysis_cube_x_Mpc = EoR_x_full_Mpc * (124./128) #Mpc Analysing the full FoV in x
		EoR_analysis_cube_y_Mpc = EoR_y_full_Mpc * (124./128) #Mpc Analysing the full FoV in y
		EoR_analysis_cube_z_Mpc = EoR_z_full_Mpc*(float(EoR_analysis_cube_z_pix)/EoR_z_full_pix) #Mpc Analysing 38 of the 128 channels of the full simulation
		EoRVolume = EoR_analysis_cube_x_Mpc*EoR_analysis_cube_y_Mpc*EoR_analysis_cube_z_Mpc
		pixel_volume = EoR_analysis_cube_x_pix*EoR_analysis_cube_y_pix*EoR_analysis_cube_z_pix
		dimensionless_PS_scaling = (self.modk_vis_ordered_list[i_bin]**3.)*(EoRVolume**1.0)/(2.*(np.pi**2)*pixel_volume)

		return dimensionless_PS_scaling

	def calc_PowerI(self, x, **kwargs):
		
		if self.dimensionless_PS:
			PowerI=np.zeros(self.Npar)+self.inverse_quadratic_power #set to zero for a uniform distribution
			for i_bin in range(len(self.k_cube_voxels_in_bin)):
				dimensionless_PS_scaling = self.calc_dimensionless_power_spectral_normalisation_21cmFAST(i_bin)
				PowerI[self.k_cube_voxels_in_bin[i_bin]] = dimensionless_PS_scaling/x[i_bin] #NOTE: fitting for power not std here
		else:
			PowerI=np.zeros(self.Npar)+self.inverse_quadratic_power #set to zero for a uniform distribution
			for i_bin in range(len(self.k_cube_voxels_in_bin)):
				PowerI[self.k_cube_voxels_in_bin[i_bin]] = 1./x[i_bin]  #NOTE: fitting for power not std here

		return PowerI

	def posterior_probability(self, x, **kwargs):
		##===== Defaults =======
		block_T_Ninv_T=self.block_T_Ninv_T
		fit_single_elems = self.fit_single_elems
		T_Ninv_T = self.T_Ninv_T
		dbar = self.dbar

		##===== Inputs =======
		if 'block_T_Ninv_T' in kwargs:
			block_T_Ninv_T=kwargs['block_T_Ninv_T']

		if self.log_priors:
			# print 'Using log-priors'
			x = 10.**np.array(x)

		phi = [0.0]
		do_block_diagonal_inversion = len(shape(block_T_Ninv_T))>1
		self.count+=1
		start = time.time()
		try:
			if do_block_diagonal_inversion:
				if self.Print:print 'Using block-diagonal inversion'
				SigmaI_dbar, dbarSigmaIdbar, PhiI, logSigmaDet = self.calc_SigmaI_dbar_wrapper(x, T_Ninv_T, dbar, block_T_Ninv_T=block_T_Ninv_T)
			else:
				if self.Print:print 'Not using block-diagonal inversion'
				SigmaI_dbar, dbarSigmaIdbar, PhiI, logSigmaDet = self.calc_SigmaI_dbar_wrapper(x, T_Ninv_T, dbar)

			logPhiDet=-1*np.sum(np.log(PhiI[np.logical_not(self.masked_power_spectral_modes)])).real #Only possible because Phi is diagonal (otherwise would need to calc np.linalg.slogdet(Phi)). -1 factor is to get logPhiDet from logPhiIDet. Note: the real part of this calculation matches the solution given by np.linalg.slogdet(Phi))

			MargLogL =  -0.5*logSigmaDet -0.5*logPhiDet + 0.5*dbarSigmaIdbar
			vals = map(real, (-0.5*logSigmaDet, -0.5*logPhiDet, 0.5*dbarSigmaIdbar, MargLogL))
			MargLogL =  MargLogL.real
			print_rate = 10000
			# brk()
			
			if self.nu>10:
				print_rate=100
			if self.count%print_rate==0:
				print 'count', self.count
				print 'Time since class instantiation: %f'%(time.time()-self.instantiation_time)
				print 'Time for this likelihood call: %f'%(time.time()-start)
			return (MargLogL.squeeze())*1.0, phi
		except Exception as e:
			print 'Exception encountered...'
			print e
			return -np.inf, phi






###
# Instantiate class and check that posterior_probability returns a likelihood rather than inf for the correct answer
###
# self = PowerSpectrumPosteriorProbability(block_T_Ninv_T=block_T_Ninv_T)
if small_cube: PSPP = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list)
PSPP_block_diag = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, block_T_Ninv_T=block_T_Ninv_T, Print=True)



self = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, block_T_Ninv_T=block_T_Ninv_T, Print=True)





start = time.time()

if small_cube:
	print PSPP.posterior_probability([1.e0]*len(bin_selector_vis_ordered_list), diagonal_sigma=False)[0]
	print PSPP_block_diag.posterior_probability([1.e0]*len(bin_selector_vis_ordered_list), diagonal_sigma=False, block_T_Ninv_T=block_T_Ninv_T)[0]






if sub_quad:
	maxL_quad_fit = PSPP_block_diag.calc_SigmaI_dbar_wrapper([1.e-20]*len(bin_selector_vis_ordered_list), T_Ninv_T, dbar, block_T_Ninv_T=block_T_Ninv_T)[0]
	maxL_quad_signal = np.dot(T,maxL_quad_fit)
	Ninv = BM.read_data_from_hdf5(array_save_directory+'Ninv.h5', 'Ninv')
	Ninv_maxL_quad_signal = np.dot(Ninv,maxL_quad_signal)
	ML_qbar = np.dot(T.conjugate().T,Ninv_maxL_quad_signal)
	q_sub_dbar = dbar-ML_qbar
	if small_cube: PSPP.dbar = q_sub_dbar
	PSPP_block_diag.dbar = q_sub_dbar
	if small_cube:
		print PSPP.posterior_probability([1.e0]*len(bin_selector_vis_ordered_list), diagonal_sigma=False)[0]
		print PSPP_block_diag.posterior_probability([1.e0]*len(bin_selector_vis_ordered_list), diagonal_sigma=False, block_T_Ninv_T=block_T_Ninv_T)[0]

print 'Time taken: %f'%(time.time()-start)

start = time.time()
print PSPP_block_diag.posterior_probability([1.e0]*len(bin_selector_vis_ordered_list), diagonal_sigma=False, block_T_Ninv_T=block_T_Ninv_T)[0]
print 'Time taken: %f'%(time.time()-start)

start = time.time()
print PSPP_block_diag.posterior_probability([1.e0]*len(bin_selector_vis_ordered_list))[0]
print 'Time taken: %f'%(time.time()-start)



###
# Plot the total, quad component, and Fourier components of the signal vs. their maximum likelihood fitted equivalents, along with the fit residuals. Save the plots to file.
###
base_dir = 'Plots'
save_dir = base_dir+'/Likelihood_v1d73_3D_ZM/'
if not os.path.isdir(save_dir):
		os.makedirs(save_dir)


# Show=True
Show=False
if not use_EoR_cube:
	maxL_k_cube_signal = PSPP_block_diag.calc_SigmaI_dbar_wrapper(np.array(k_sigma)**2., T_Ninv_T, dbar, block_T_Ninv_T=block_T_Ninv_T)[0]
else:
	maxL_k_cube_signal = PSPP_block_diag.calc_SigmaI_dbar_wrapper(np.array([100.0]*len(bin_selector_vis_ordered_list))**2, T_Ninv_T, dbar, block_T_Ninv_T=block_T_Ninv_T)[0]

maxL_f_plus_q_signal = np.dot(T,maxL_k_cube_signal)
save_path = save_dir+'Total_signal_model_fit_and_residuals.png'
if use_EoR_cube:save_path=save_path.replace('.png', '_EoR_cube.png')
plot_signal_vs_MLsignal_residuals(s, maxL_f_plus_q_signal, sigma, save_path)

if small_cube and not use_EoR_cube and nq==2:
	maxL_k_cube_quad_modes = maxL_k_cube_signal.copy()
	maxL_k_cube_quad_modes[np.logical_not(Quad_modes_only_boolean_array_vis_ordered)] = 0.0
	q_component_of_maxL_of_f_plus_q_signal = np.dot(T,maxL_k_cube_quad_modes)
	plot_signal_vs_MLsignal_residuals(s_quad_only, q_component_of_maxL_of_f_plus_q_signal, sigma, save_dir+'Quad_component_of_signal_model_fit_and_residuals.png')

	maxL_k_cube_fourier_modes = maxL_k_cube_signal.copy()
	maxL_k_cube_fourier_modes[Quad_modes_only_boolean_array_vis_ordered] = 0.0
	f_component_of_maxL_of_f_plus_q_signal = np.dot(T,maxL_k_cube_fourier_modes)
	plot_signal_vs_MLsignal_residuals(s_fourier_only, f_component_of_maxL_of_f_plus_q_signal, sigma, save_dir+'Fourier_component_of_signal_model_fit_and_residuals.png')


# No_large_spectral_scale_model_fit = True
No_large_spectral_scale_model_fit = False
if No_large_spectral_scale_model_fit:
	PSPP_block_diag.inverse_quadratic_power=1.e10
	maxL_k_cube_signal = PSPP_block_diag.calc_SigmaI_dbar_wrapper(np.array([100.0]*len(bin_selector_vis_ordered_list))**2, T_Ninv_T, dbar, block_T_Ninv_T=block_T_Ninv_T)[0]
	# maxL_k_cube_signal = PSPP_block_diag.calc_SigmaI_dbar_wrapper(np.array(k_sigma)**2., T_Ninv_T, q_sub_dbar, block_T_Ninv_T=block_T_Ninv_T)[0]

	maxL_f_plus_q_signal = np.dot(T,maxL_k_cube_signal)
	save_path = save_dir+'Total_signal_model_fit_and_residuals_NQ.png'
	if use_EoR_cube:save_path=save_path.replace('.png', '_EoR_cube.png')
	plot_signal_vs_MLsignal_residuals(s-maxL_quad_signal, maxL_f_plus_q_signal, sigma, save_path)

	if small_cube and not use_EoR_cube:
		maxL_k_cube_fourier_modes = maxL_k_cube_signal.copy()
		maxL_k_cube_fourier_modes[Quad_modes_only_boolean_array_vis_ordered] = 0.0
		f_component_of_maxL_of_f_plus_q_signal = np.dot(T,maxL_k_cube_fourier_modes)
		plot_signal_vs_MLsignal_residuals(s_fourier_only, f_component_of_maxL_of_f_plus_q_signal, sigma, save_dir+'Fourier_component_of_signal_model_fit_and_residuals_NQ.png')










###
# Calculate the power spectrum using PolyChord
###

#######################
###
# PolyChord setup
# priors_min_max = np.array([[0.0, 35.0] for _ in range(len(bin_selector_vis_ordered_list))])**2.0
log_priors_min_max = [[-5.0, 2.0] for _ in range(len(bin_selector_vis_ordered_list))]
# log_priors_min_max = [[-2.0, 2.0] for _ in range(len(bin_selector_vis_ordered_list))]
log_priors_min_max[1][1]=4.0

# prior_c = PriorC(priors_min_max)
prior_c = PriorC(log_priors_min_max)
nDerived = 0
nDims = len(k_cube_voxels_in_bin)

base_dir = 'chains/clusters/'
if not os.path.isdir(base_dir):
		os.makedirs(base_dir)

log_priors = True
dimensionless_PS = True
# dimensionless_PS = False
zero_the_quadratic_modes = False

file_root = 'Test_mini-sigma_{:.1E}-lp_F-dPS_F-v1-'.format(sigma).replace('.','d')
# file_root = 'Quad_only_Test_mini-sigma_{:.1E}-lp_F-dPS_F-v1-'.format(sigma).replace('.','d')
if chan_selection!='':file_root=chan_selection+file_root
if log_priors:
	file_root=file_root.replace('lp_F', 'lp_T')
if dimensionless_PS:
	file_root=file_root.replace('dPS_F', 'dPS_T')
if zero_the_quadratic_modes or nq==0:
	file_root=file_root.replace('mini-', 'mini-NQ-')
if use_EoR_cube:
	file_root=file_root.replace('Test_mini', 'EoR_mini')
if use_MultiNest:
	file_root='MN-'+file_root

version_number='1'
file_name_exists = os.path.isfile('chains/'+file_root+'_phys_live.txt') or os.path.isfile('chains/'+file_root+'.resume') or os.path.isfile('chains/'+file_root+'resume.dat')
while file_name_exists:
	fr1,fr2 = file_root.split('-v')
	fr21,fr22 = fr2.split('-')
	next_version_number = str(int(fr21)+1)
	file_root=file_root.replace('v'+version_number+'-', 'v'+next_version_number+'-')
	version_number = next_version_number
	file_name_exists = os.path.isfile('chains/'+file_root+'_phys_live.txt') or os.path.isfile('chains/'+file_root+'.resume') or os.path.isfile('chains/'+file_root+'resume.dat')


# file_root='MN-Test_mini-sigma_2d0E+00-lp_T-dPS_T-v18-'
print 'Output file_root = ', file_root

PSPP_block_diag_Polychord = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, block_T_Ninv_T=block_T_Ninv_T, log_priors=log_priors, dimensionless_PS=dimensionless_PS, Print=True)

if zero_the_quadratic_modes: PSPP_block_diag_Polychord.inverse_quadratic_power=1.e10
if sub_quad: PSPP_block_diag_Polychord.dbar = q_sub_dbar

# T_Ninv_T=[]
# block_T_Ninv_T=[]

start = time.time()
PSPP_block_diag_Polychord.Print=False
Nit=20
for _ in range(Nit):
	L =  PSPP_block_diag_Polychord.posterior_probability([1.e0]*nDims)[0]
# PSPP_block_diag_Polychord.Print=True
PSPP_block_diag_Polychord.Print=False
print 'Average evaluation time: %f'%((time.time()-start)/float(Nit))

# print PSPP_block_diag_Polychord.block_T_Ninv_T


def likelihood(theta, calc_likelihood=PSPP_block_diag_Polychord.posterior_probability):
	# return PSPP_block_diag_Polychord.posterior_probability(theta)
	return calc_likelihood(theta)

def MultiNest_likelihood(theta, calc_likelihood=PSPP_block_diag_Polychord.posterior_probability):
	return calc_likelihood(theta)[0]


if use_MultiNest:
	MN_nlive = nDims*25
	# run MultiNest
	result = solve(LogLikelihood=MultiNest_likelihood, Prior=prior_c.prior_func, n_dims=nDims, outputfiles_basename="chains/"+file_root, n_live_points=MN_nlive)
else:
	precision_criterion = 0.05
	#precision_criterion = 0.001 #The default value
	nlive=nDims*10
	#nlive=nDims*25 #The default value
	PolyChord.mpi_notification()
	PolyChord.run_nested_sampling(PSPP_block_diag_Polychord.posterior_probability, nDims, nDerived, file_root=file_root, read_resume=False, prior=prior_c.prior_func, precision_criterion=precision_criterion, nlive=nlive)





print 'Sampling complete!'
#######################









