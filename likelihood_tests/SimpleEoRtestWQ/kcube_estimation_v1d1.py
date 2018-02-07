
###
# Imports
###

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
	Tb_experimental_std_K  = 23.0     #Matches GSM in region A at 0.333 degree resolution (i.e. for a 50 degree map 150 pixels across). Note: std. is a function of resultion so the foreground map should be made at the same resolution for this std normalisation to be accurate
	# Tb_experimental_mean_K = 240.0  #Revise to match published values
	# Tb_experimental_std_K  = 4.0    #Revise to match published values
	nu_min_MHz             = 163.0
	channel_width_MHz      = 0.2
	fits_storage_dir = 'fits_storage/Jelic_beta_{}_dbeta{}/'.format(beta_experimental_mean, beta_experimental_std).replace('.','d')
	###
	foreground_outputs = generate_Jelic_cube(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show, beta_experimental_mean,beta_experimental_std,gamma_mean,gamma_sigma,Tb_experimental_mean_K,Tb_experimental_std_K,nu_min_MHz,channel_width_MHz, generate_additional_extrapolated_HF_foreground_cube=True, fits_storage_dir=fits_storage_dir)
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





# use_EoR_signal_instead = True
use_EoR_signal_instead = False
if use_EoR_signal_instead:
	s, abc, scidata1 = generate_data_from_loaded_EoR_cube(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show,chan_selection)
	###
	# Overwrite Tb_nu with 38 channels of the EoR cube
	###
	Tb_nu = scidata1[0:38]





axes_tuple=(1,2)
fftTb_nu=numpy.fft.ifftshift(Tb_nu+0j, axes=axes_tuple)
fftTb_nu=numpy.fft.fftn(fftTb_nu, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
fftTb_nu=numpy.fft.fftshift(fftTb_nu, axes=axes_tuple)/(Tb_nu.size)**0.5

n=Tb_nu.shape[1]/2-1
# n=255
pylab.errorbar(arange(38), Tb_nu[:,n,n]/abs(Tb_nu[:,n,n]).max(),color='black')
pylab.errorbar(arange(38), fftTb_nu[:,n,n].real/abs(fftTb_nu[:,n,n].real).max(),color='red')
pylab.errorbar(arange(38), -fftTb_nu[:,n,n].imag/abs(fftTb_nu[:,n,n].imag).max(),color='blue')
if Show: pylab.show()






pylab.close('all')
x=arange(38)+1
d=Tb_nu[:,n,n]
# m=np.polyval(np.polyfit(x,d,2),x)
# pylab.errorbar(arange(38), (d-m),color='black')
d=fftTb_nu[:,n,n].real
m=np.polyval(np.polyfit(x,d,2),x)
pylab.errorbar(x, (d-m),color='red')
d=fftTb_nu[:,n,n].imag
m=np.polyval(np.polyfit(x,d,2),x)
pylab.errorbar(x, (d-m),color='blue')
pylab.show()

print d/(d-m)/1.e5


pylab.close('all')
x=arange(38)+1
d=fftTb_nu[:,n,n].real
m=np.polyval(np.polyfit(x,d,2),x)
pylab.errorbar(x, d,color='red',fmt='--')
pylab.errorbar(x, m,color='red',fmt='-')
# pylab.errorbar(x, (d-m),color='red')
# d=fftTb_nu[:,n,n].imag
# m=np.polyval(np.polyfit(x,d,2),x)
# pylab.errorbar(x, (d-m),color='blue')
pylab.show()





nu_array_MHz = nu_min_MHz+np.arange(nf)*channel_width_MHz
def model(x,a):
	return a * (nu_array_MHz/nu_min_MHz)**-2.6


# d=Tb_nu[:,n,n]
# d=fftTb_nu[:,n,n].real
d=fftTb_nu[:,n,n].imag
from scipy.optimize import curve_fit
popt, pcov = scipy.optimize.curve_fit(model, x, d)
fig,ax=pylab.subplots(nrows=2,ncols=1)
ax[0].plot(x, d, 'b-', label='data')
ax[0].plot(x, model(x, *popt), 'r-', label='fit')
ax[1].plot(x, d-model(x, *popt), 'b-', label='residuals')
for axis in ax:axis.legend()
fig.show()

r=d-model(x, *popt)
m2=np.polyval(np.polyfit(x,r,2),x)
fig,ax=pylab.subplots(nrows=2,ncols=1)
ax[0].errorbar(x, (d-m),color='blue')
ax[0].errorbar(x, (r-m2),color='red')
ax[1].errorbar(x, (r-m2),color='red')
fig.show()






nu_array_MHz = nu_min_MHz+np.arange(nf)*channel_width_MHz
def model_q(x,q1,q2,q3):
	q_val = (q1 + q2*x + q3*x**2)/float(nf)
	return q_val

def model_pl(x,p1):
	p_val = p1 * (nu_array_MHz/nu_min_MHz)**-beta_experimental_mean
	# p_val = p1 * (nu_array_MHz/nu_min_MHz)**-2.6
	return p_val

def model_q_plus_pl(x,q1,q2,q3,p1):
	p_val = model_pl(x,p1)
	q_val = model_q(x,q1,q2,q3)
	m=p_val+q_val
	return m

def model_2nd_order_pl(x,p1,p2,p3):
	###
	# log10(d) = p11 + p2*log10(x_nu) + p3*log10(x_nu)**2.0
	# d = 10.**(p11 + p2*log10(x_nu) + p3*log10(x_nu)**2.0)
	# d = 10.**p11 * 10.**(p2*log10(x_nu)) * 10.**(p3*log10(x_nu)**2.0)
	# d = p1 * x_nu**p2 * x_nu**(p3*log10(x_nu) with p1=10.**p11
	###
	# p2_min = -2.63-0.5
	# p2_max = 2.0
	# if p2<p2_min: p2=p2_min
	# if p2>p2_max: p2=p2_max
	nu_array_MHz = nu_min_MHz+np.arange(nf)*channel_width_MHz
	x_nu = (nu_array_MHz/nu_min_MHz)
	p_val = p1 * x_nu**p2 * x_nu**(p3*log10(x_nu))
	m=p_val
	return m


d=fftTb_nu[:,n,n].imag
from scipy.optimize import curve_fit
popt, pcov = scipy.optimize.curve_fit(model_2nd_order_pl, x, d)
m_2nd_order_pl = model_2nd_order_pl(x, *popt)
print (d - m_2nd_order_pl)
print (m_2nd_order_pl)


from pymultinest.solve import solve

#######################
###
nDims = 3
priors_min_max = [[-5.0, 2.0] for _ in range(nDims)]
priors_min_max[0][0]=-500.
priors_min_max[0][1]= 500.
priors_min_max[1][0]=-2.63-0.5
priors_min_max[1][1]=-2.63+0.5
priors_min_max[2][0]=-1.
priors_min_max[2][1]= 1.

prior_c = PriorC(priors_min_max)
nDerived = 0

base_dir = 'chains/clusters/'
if not os.path.isdir(base_dir):
		os.makedirs(base_dir)

file_root='MN-Test-v1d0-'
print 'Output file_root = ', file_root

start = time.time()
def calc_likelihood(theta):
		p1,p2,p3=theta
		m = model_2nd_order_pl(x,p1,p2,p3)
		sigma=1.e-3
		logL = -np.sum((d-m)**2.0/sigma**2.0)
		return logL


MN_nlive = nDims*25
# run MultiNest
result = solve(LogLikelihood=calc_likelihood, Prior=prior_c.prior_func, n_dims=nDims, outputfiles_basename="chains/"+file_root, n_live_points=MN_nlive, resume=False)


print np.mean(result['samples'], axis=0)



import numpy as np
from numpy import *

ML_q_plus_PL = np.loadtxt('ML_q_plus_PL.dat')
uvf_presub = np.loadtxt('uvf_presub.dat')
uvf_postsub = np.loadtxt('uvf_postsub.dat')
import pylab
x=arange(38)+1.
pylab.errorbar(x,uvf_presub[:,2][38*1:38*2])
pylab.errorbar(x,3.e4*((163.+arange(38)*0.2)/163.)**-2.6 - 3.e4/2-12500.)
pylab.show()
# ZeroNoisePSModel = np.loadtxt('/home/psims/Fortran/Polychord/Data/163MHz_Foregrounds2/Jelic_beta_2d63_dbeta0d1/ZeroNoisePSModel_v2.dat') 
# for i in range (50): 
# 	pylab.errorbar(x,ZeroNoisePSModel[:,2][i::960*2]/abs(ZeroNoisePSModel[:,2][i::960*2]).max())
# pylab.show()
F=np.zeros([38,4])
for i in range(3):F[:,i]=x**i/38.
F[:,3]=((163.+(x*0.2))/163.)**-2.63
FNF = np.dot(F.T,F)
FNF_inv = np.linalg.inv(FNF)
d=uvf_presub[:,2][38*481:38*482]
FNd = np.dot(F.T,d.reshape(-1,1))
ahat = np.dot(FNF_inv,FNd)
m=np.dot(F,ahat) 
r1 = (d-m.flatten()).copy()
print r1

F=np.zeros([38,3])
for i in range(3):F[:,i]=x**i
FNF = np.dot(F.T,F)
FNF_inv = np.linalg.inv(FNF)
d=uvf_presub[:,2][38*481:38*482]
FNd = np.dot(F.T,d.reshape(-1,1))
ahat = np.dot(FNF_inv,FNd)
m=np.dot(F,ahat) 
r2 = (d-m.flatten()).copy()
print r2








d=fftTb_nu[:,n,n].imag
from scipy.optimize import curve_fit
popt, pcov = scipy.optimize.curve_fit(model_q_plus_pl, x, d)
m_q = model_q(x, *popt[0:3])
m_pl = model_pl(x, *popt[3:4])
r_joint=d-model_q_plus_pl(x, *popt)
fig,ax=pylab.subplots(nrows=4,ncols=1)
ax[0].errorbar(x,r_joint,color='red')
ax[1].errorbar(x,m_q,color='black')
ax[2].errorbar(x,m_pl,color='red')
ax[3].errorbar(x, (r-m2),color='red')
fig.show()


fig,ax=pylab.subplots(nrows=4,ncols=1, figsize=(5,10))
ax[0].errorbar(x, m_q,color='orange',fmt='--', label='$m_q$')
ax[0].errorbar(x, m_pl,color='blue',fmt='--', label='$m_{pl}$')
ax[0].errorbar(x, m_q+m_pl,color='red',fmt='--', label='$m$')
ax[0].errorbar(x, d,color='black',fmt='-', label='$d$')
ax[1].errorbar(x, m_pl,color='blue',fmt='--', label='$m_{pl}$')
ax[2].errorbar(x, m_q,color='orange',fmt='--', label='$m_q$')
ax[3].errorbar(x, d-m_q-m_pl,color='red',fmt='-', label='residuals')
fig.tight_layout()
for axis in ax:axis.legend(loc="upper right")
fig.show()




d=fftTb_nu[:,n,n].imag
from scipy.optimize import curve_fit
popt11, pcov11 = scipy.optimize.curve_fit(model_q, x, d)
r11=d-model_q(x, *popt11)
popt12, pcov12 = scipy.optimize.curve_fit(model_pl, x, r11)
r12=r11-model_pl(x, *popt12)

popt13, pcov13 = scipy.optimize.curve_fit(model_pl, x, d)
r13=d-model_pl(x, *popt13)
popt14, pcov14 = scipy.optimize.curve_fit(model_q, x, r13)
r14=r13-model_q(x, *popt14)

fig,ax=pylab.subplots(nrows=6,ncols=1, figsize=(5,10))
ax[0].errorbar(x,r_joint,color='red', label='residuals $r_\mathrm{q,pl}d-m_\mathrm{q,pl}$')
ax[1].errorbar(x, (r-m2),color='red', label='residuals $d-m_\mathrm{pl}(d)-m_\mathrm{q}(r_\mathrm{pl})$')
ax[2].errorbar(x,r11,color='red', label='residuals $d-m_\mathrm{q}(d)$')
ax[3].errorbar(x,r12,color='red', label='residuals $d-m_\mathrm{q}(d)-m_\mathrm{pl}(r_\mathrm{q})$')
ax[4].errorbar(x,r13,color='red', label='residuals $d-m_\mathrm{pl}(d)-m_\mathrm{q}(r_\mathrm{pl})$')
ax[5].errorbar(x,r14,color='red', label='residuals $d-m_\mathrm{pl}(d)-m_\mathrm{q}(r_\mathrm{pl})$')
fig.tight_layout()
for axis in ax:axis.legend(loc="upper right")
fig.show()




print ','.join(['a{}'.format(i) for i in range(1,39,1)])
def model_Fourier(x,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,a33,a34,a35,a36,a37,a38):
	a = (a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,a33,a34,a35,a36,a37,a38)
	m_f = 0.0
	numfreqs=19
	numchan=38.
	for m2 in range(numfreqs):
		m_f = m_f + a[m2]*(2**0.5)*np.sin(2*pi*((m2+1.)/numchan)*x)/numchan
		m_f = m_f + a[m2+numfreqs]*(2**0.5)*np.cos(2*pi*((m2+1.)/numchan)*x)/numchan

    # do m2=1,numfreqs
    #         FMatrix(m1,nlfterms+m2) = sqrt(2d0)*sin(2*Pi*(dble(m2)/numchan)*m1)/numchan
    #         FMatrix(m1,nlfterms+m2+numfreqs) = sqrt(2d0)*cos(2*Pi*(dble(m2)/numchan)*m1)/numchan
    # end do
	return m_f


def model_q_plus_Fourier(x,q1,q2,q3,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,a33,a34):
	q_val = model_q(x,q1,q2,q3)
	F_val = model_Fourier(x,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,a33,a34)
	m=q_val+F_val
	return m


def model_q_plus_pl_plus_Fourier(x,q1,q2,q3,p1,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,a33,a34):
	p_val = model_pl(x,p1)
	q_val = model_q(x,q1,q2,q3)
	F_val = model_Fourier(x,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,a33,a34)
	m=p_val+q_val+F_val
	return m

d=fftTb_nu[:,n,n].imag
from scipy.optimize import curve_fit
popt, pcov = scipy.optimize.curve_fit(model_Fourier, x, d, p0=np.ones(38)*1)
# m_q = model_q(x, *popt[0:3])
# m_pl = model_pl(x, *popt[3:4])
m_F = model_Fourier(x, *popt[:])
r_joint2=d-model_Fourier(x, *popt)
fig,ax=pylab.subplots(nrows=3,ncols=1)
ax[0].errorbar(x,r_joint,color='black')
ax[0].errorbar(x,r_joint2,color='red')
ax[1].errorbar(x, r_joint,color='black')
ax[2].errorbar(x, r_joint2,color='red')
fig.show()


fig,ax=pylab.subplots(nrows=4,ncols=1)
ax[0].errorbar(x,m_q,color='black')
ax[1].errorbar(x,m_pl,color='red')
ax[2].errorbar(x, m_F,color='black')
ax[3].errorbar(x, m_q+m_pl+m_F,color='red')
fig.show()

fig,ax=pylab.subplots(nrows=5,ncols=1, figsize=(5,10))
ax[0].errorbar(x, m_q,color='orange',fmt='--', label='$m_q$')
ax[0].errorbar(x, m_pl,color='blue',fmt='--', label='$m_{pl}$')
ax[0].errorbar(x, m_F,color='green',fmt='--', label='$m_{F}$')
ax[0].errorbar(x, m_q+m_pl+m_F,color='red',fmt='--', label='$m$')
ax[0].errorbar(x, d,color='black',fmt='-', label='$d$')
ax[1].errorbar(x, m_pl,color='blue',fmt='--', label='$m_{pl}$')
ax[2].errorbar(x, m_q,color='orange',fmt='--', label='$m_q$')
ax[3].errorbar(x, m_F,color='green',fmt='--', label='$m_{F}$')
ax[4].errorbar(x, d-(m_q+m_pl+m_F),color='green',fmt='--', label='$m_{F}$')
fig.tight_layout()
for axis in ax:axis.legend(loc="upper right")
fig.show()


def likelihood31(theta, return_model=False):
	m_params = theta[:-1]
	a_params = m_params[4:]
	m=model_q_plus_pl_plus_Fourier(x,*m_params)
	power_amp=theta[-1]
	sigma=1.e-1
	chisq = np.sum((d-m)**2.0/sigma**2.0)
	aphia = np.sum((a_params)**2.0/power_amp**2.0)
	logdetphi = np.sum(np.log(power_amp**2.0))
	loglikelihood = chisq+aphia+logdetphi
	if return_model:
		return m
	else:
		return loglikelihood


def likelihood32(theta, return_model=False):
	m_params = theta[:-2]
	a_params = m_params[4:]
	m=model_q_plus_pl_plus_Fourier(x,*m_params)
	power_amp=theta[-2]
	a_mean=theta[-1]
	sigma=1.e-4
	chisq = np.sum((d-m)**2.0/sigma**2.0)
	aphia = np.sum((a_params-a_mean)**2.0/power_amp**2.0)
	logdetphi = np.sum(np.log(power_amp**2.0))
	loglikelihood = chisq+aphia+logdetphi
	# print loglikelihood
	if return_model:
		return m
	else:
		if np.isinf(loglikelihood):
			return np.inf #if loglikelihood is -inf return inf to stop the sampler using those values (since the sampler actaually likes -inf because it's minimising)
		else:
			return loglikelihood


def likelihood33(theta, return_model=False):
	m=model_q_plus_pl(x,*theta)
	sigma=1.e-1
	chisq = np.sum((d-m)**2.0/sigma**2.0)
	loglikelihood = chisq
	if return_model:
		return m
	else:
		return loglikelihood


x031=np.ones(37)
x031[0:4]=popt[0:4]
x031[-1]=100
out31 = scipy.optimize.minimize(likelihood31, x031)
x032=np.ones(38)
x032[0:4]=popt[0:4]
x032[-2]=100
out32 = scipy.optimize.minimize(likelihood32, x032, method='Powell')
x033=np.ones(4)
# x033=popt
out33 = scipy.optimize.minimize(likelihood33, x033, method='Powell')

return_model = True
m31 = likelihood31(out31['x'], return_model)
m32 = likelihood32(out32['x'], return_model)
m33 = likelihood33(out33['x'], return_model)


fig,ax=pylab.subplots(nrows=5,ncols=1,figsize=(5,10))
ax[0].errorbar(x, (d-m31),color='red',fmt='-', label='residuals')
ax[1].errorbar(x, (d-m32),color='red',fmt='-', label='residuals')
ax[2].errorbar(x, (d-m33),color='red',fmt='-', label='residuals')
ax[3].errorbar(x, r_joint,color='red',fmt='-', label='residuals')
ax[4].errorbar(x, r_joint2,color='red',fmt='-', label='residuals')
fig.tight_layout()
# ax.legend(loc="upper right")
fig.show()





###
# Quad + power law + Fourier
###
import time
u_c,v_c = np.array(Tb_nu.shape[1:])/2
# u_c,v_c = 256,256
fftTb_nu_subset = fftTb_nu[:,u_c-15:u_c+16,v_c-15:v_c+16]
real_popts_q_plus_pl = []
imag_popts_q_plus_pl = []
d_reals = []
d_imags = []
start=time.time()
nu=8
nv=1
# nv=31
for i in range(15,15+nu,1):
	print i,'Time since start: {}'.format(time.time()-start)
	# for j in range(nv):
	for j in range(15,15+nv,1):
		if not (i==15 and j==15):
			d_real = fftTb_nu_subset[:,i,j].real
			d_imag = fftTb_nu_subset[:,i,j].imag
			popt_r, pcov_r = scipy.optimize.curve_fit(model_q_plus_pl, x, d_real)
			popt_i, pcov_i = scipy.optimize.curve_fit(model_q_plus_pl, x, d_imag)
			real_popts_q_plus_pl.append(popt_r)
			imag_popts_q_plus_pl.append(popt_i)
			d_reals.append(d_real)
			d_imags.append(d_imag)



d_big_real = np.array(d_reals)
d_big_imag = np.array(d_imags)
# d_big = np.concatenate((d_big_real,d_big_imag))
d_big = d_big_imag

# q_plus_pl_start_points = real_popts_q_plus_pl
q_plus_pl_start_points = imag_popts_q_plus_pl
# q_plus_pl_start_points = np.concatenate((real_popts_q_plus_pl,imag_popts_q_plus_pl))
# q_plus_pl_start_points = imag_popts_q_plus_pl

from pdb import set_trace as brk
n_fg_params = 4

def likelihood32_big(theta, return_model=False):
	abs_power_squared = abs(theta[0:16]) #Use abs_power rather than power_amps**2. to avoid numerical issues related to aphia being much larger if 1.) a[:]=1.e-11 and phi[:]=1.e-11 than if e.g. 2.) a[:]=1.e-21 and phi[:]=1.e-11, but 2.) is never attempted due to power_amps being squared. i.e. there needs to be an option for phi[:] to be extremely small but for a[:] to be smaller still.
	a_means    = theta[16:32]
	m_params   = theta[32:]
	m_param_sets = m_params.reshape(-1,36)
	a_param_sets = np.array([m_param_set[n_fg_params:] for m_param_set in m_param_sets])
	# a_param_sets = np.array([m_param_set[4:] for m_param_set in m_param_sets])
	m=np.array([model_q_plus_pl_plus_Fourier(x,*m_param_set) for m_param_set in m_param_sets])
	sigma=1.e-3
	chisq = np.sum((d_big-m)**2.0/sigma**2.0)

	aphia = np.sum([np.sum((a_param_sets[:,i])**2.0/abs_power_squared[i]**0.5) for i in range(len(abs_power_squared)) ]) #for the sine wave modes
	aphia = aphia + np.sum([np.sum((a_param_sets[:,i+(32/2)])**2.0/abs_power_squared[i]**0.5) for i in range(len(abs_power_squared)) ]) #for the cosine wave modes (which fill the second half of each a_param_set i.e. a_param_sets[:,(32/2):])
	# aphia = np.sum([np.sum((a_param_sets[:,i]-a_means[i])**2.0/power_amps[i]**2.0) for i in range(len(power_amps)) ])
	logdetphi = np.sum(np.log(abs_power_squared**0.5))
	loglikelihood = chisq+aphia+logdetphi
	Print=True
	if Print: print 'loglikelihood,chisq,aphia,logdetphi', loglikelihood,chisq,aphia,logdetphi
	# print 'Time since start: {}'.format(time.time()-start)
	# brk()
	if return_model:
		return m
	else:
		return loglikelihood

# n_par = 32+1*(nu*nv-1)*36 #16 means, 16 power_amps, (nu*nv-1)*4 pl and quad params and (nu*nv-1)*32 Fourier amps
n_par = 32+len(q_plus_pl_start_points)*36 #16 means, 16 power_amps, (nu*nv-1)*4 pl and quad params and (nu*nv-1)*32 Fourier amps
x032_big=np.ones(n_par)
x032_big[0:32]=100.0
x032_big_m_params = x032_big[32:]
x032_big_m_params = x032_big_m_params.reshape(-1,36)
for i in range(len(q_plus_pl_start_points)):
	x032_big_m_params[i][0:4] = q_plus_pl_start_points[i]


start=time.time()
out32_big = scipy.optimize.minimize(likelihood32_big, x032_big, method='Powell')
print 'Time since start: {}'.format(time.time()-start)



print likelihood32_big(out32_big['x'])
residuals_big2 = d_big-likelihood32_big(out32_big['x'],True)

print (out32_big['x'][0:16]*1.e3)**2.

# 8  - Time since start: 34.3494849205
# 16 - Time since start: 141.838307142
# 32 - Time since start: 564.353577852

test = out32_big['x'].copy()
print likelihood32_big(test)

test[0:32]=1.e-11
test_m_params = test[32:].reshape(-1,36)
for i in range(len(q_plus_pl_start_points)):
	test_m_params[i][0:4] = q_plus_pl_start_points[i]
	test_m_params[i][4:] = 1.e-21

print likelihood32_big(test)



# n_par = 32+len(q_plus_pl_start_points)*36 #16 means, 16 power_amps, (nu*nv-1)*4 pl and quad params and (nu*nv-1)*32 Fourier amps
# x032_big=np.ones(n_par)
# x032_big_min=np.array([None]*n_par)
# x032_big_max=np.array([None]*n_par)
# x032_big[0:32]=100.0
# x032_big_m_params = x032_big[32:].reshape(-1,36)
# x032_big_min_m_params = x032_big_min[32:].reshape(-1,36)
# x032_big_max_m_params = x032_big_max[32:].reshape(-1,36)
# for i in range(len(q_plus_pl_start_points)):
# 	x032_big_m_params[i][0:4] = q_plus_pl_start_points[i]
# 	x032_big_min_m_params[i][0:4] = None
# 	x032_big_max_m_params[i][0:4] = None


# x032_big_bounds = np.vstack((x032_big_min, x032_big_max)).T

# start=time.time()
# # out32_big = scipy.optimize.minimize(likelihood32_big, x032_big, method='TNC', bounds=x032_big_bounds)
# out32_big = scipy.optimize.minimize(likelihood32_big, x032_big, method='SLSQP', bounds=x032_big_bounds)
# print 'Time since start: {}'.format(time.time()-start)









###
# Quad + Fourier
###
import time
u_c,v_c = np.array(Tb_nu.shape[1:])/2
# u_c,v_c = 256,256
fftTb_nu_subset = fftTb_nu[:,u_c-15:u_c+16,v_c-15:v_c+16]
real_popts_q = []
imag_popts_q = []
d_reals = []
d_imags = []
start=time.time()
nu=8
nv=1
# nv=31
for i in range(15,15+nu,1):
	print i,'Time since start: {}'.format(time.time()-start)
	# for j in range(nv):
	for j in range(15,15+nv,1):
		if not (i==15 and j==15):
			d_real = fftTb_nu_subset[:,i,j].real
			d_imag = fftTb_nu_subset[:,i,j].imag
			popt_r, pcov_r = scipy.optimize.curve_fit(model_q, x, d_real)
			popt_i, pcov_i = scipy.optimize.curve_fit(model_q, x, d_imag)
			real_popts_q.append(popt_r)
			imag_popts_q.append(popt_i)
			d_reals.append(d_real)
			d_imags.append(d_imag)


d_big_real = np.array(d_reals)
d_big_imag = np.array(d_imags)
# d_big = np.concatenate((d_big_real,d_big_imag))
d_big = d_big_imag

# q_start_points = real_popts_q
q_start_points = imag_popts_q
# q_start_points = np.concatenate((real_popts_q,imag_popts_q))
# q_start_points = imag_popts_q

from pdb import set_trace as brk
n_fg_params = 3

def likelihood32_big_qonly(theta, return_model=False):
	abs_power_squared = abs(theta[0:16]) #Use abs_power rather than power_amps**2. to avoid numerical issues related to aphia being much larger if 1.) a[:]=1.e-11 and phi[:]=1.e-11 than if e.g. 2.) a[:]=1.e-21 and phi[:]=1.e-11, but 2.) is never attempted due to power_amps being squared. i.e. there needs to be an option for phi[:] to be extremely small but for a[:] to be smaller still.
	a_means    = theta[16:32]
	m_params   = theta[32:]
	m_param_sets = m_params.reshape(-1,35)
	a_param_sets = np.array([m_param_set[n_fg_params:] for m_param_set in m_param_sets])
	m=np.array([model_q_plus_Fourier(x,*m_param_set) for m_param_set in m_param_sets])
	sigma=1.e-3
	chisq = np.sum((d_big-m)**2.0/sigma**2.0)

	power_amp=theta[-2]
	a_mean=theta[-1]
	aphia = np.sum([np.sum((a_param_sets[:,i])**2.0/abs_power_squared[i]**0.5) for i in range(len(abs_power_squared)) ]) #for the sine wave modes
	aphia = aphia + np.sum([np.sum((a_param_sets[:,i+(32/2)])**2.0/abs_power_squared[i]**0.5) for i in range(len(abs_power_squared)) ]) #for the cosine wave modes (which fill the second half of each a_param_set i.e. a_param_sets[:,(32/2):])
	# aphia = np.sum([np.sum((a_param_sets[:,i]-a_means[i])**2.0/power_amps[i]**2.0) for i in range(len(power_amps)) ])
	logdetphi = np.sum(np.log(abs_power_squared**0.5))
	loglikelihood = chisq+aphia+logdetphi
	Print=False
	if Print: print 'loglikelihood,chisq,aphia,logdetphi', loglikelihood,chisq,aphia,logdetphi
	# print 'Time since start: {}'.format(time.time()-start)
	# brk()
	if return_model:
		return m
	else:
		return loglikelihood

# n_par = 32+1*(nu*nv-1)*35 #16 means, 16 power_amps, (nu*nv-1)*4 pl and quad params and (nu*nv-1)*32 Fourier amps
n_par = 32+len(q_start_points)*35 #16 means, 16 power_amps, (nu*nv-1)*4 pl and quad params and (nu*nv-1)*32 Fourier amps
x032_big_qonly=np.ones(n_par)
x032_big_qonly[0:32]=100.0
x032_big_qonly_m_params = x032_big_qonly[32:]
x032_big_qonly_m_params = x032_big_qonly_m_params.reshape(-1,35)
for i in range(len(q_start_points)):
	x032_big_qonly_m_params[i][0:3] = q_start_points[i]


start=time.time()
out32_big_qonly = scipy.optimize.minimize(likelihood32_big_qonly, x032_big_qonly, method='Powell')
print 'Time since start: {}'.format(time.time()-start)



print likelihood32_big_qonly(out32_big_qonly['x'])
residuals_big_qonly2 = d_big-likelihood32_big_qonly(out32_big_qonly['x'],True)

print (out32_big_qonly['x'][0:16]*1.e3)**2.
print (abs(out32_big_qonly['x'][0:16])**0.5)*1.e3






###
# Cubic + Fourier
###
def model_cubic(x,q1,q2,q3,q4):
	cubic_val = (q1 + q2*x + q3*x**2 + q4*x**3)/float(nf)
	return cubic_val


def model_cubic_plus_Fourier(x,q1,q2,q3,q4,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,a33,a34):
	cubic_val = model_cubic(x,q1,q2,q3,q4)
	F_val = model_Fourier(x,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,a33,a34)
	m=cubic_val+F_val
	return m


import time
u_c,v_c = np.array(Tb_nu.shape[1:])/2
# u_c,v_c = 256,256
fftTb_nu_subset = fftTb_nu[:,u_c-15:u_c+16,v_c-15:v_c+16]
real_popts_cubic = []
imag_popts_cubic = []
d_reals = []
d_imags = []
start=time.time()
nu=8
nv=1
# nv=31
for i in range(15,15+nu,1):
	print i,'Time since start: {}'.format(time.time()-start)
	# for j in range(nv):
	for j in range(15,15+nv,1):
		if not (i==15 and j==15):
			d_real = fftTb_nu_subset[:,i,j].real
			d_imag = fftTb_nu_subset[:,i,j].imag
			popt_r, pcov_r = scipy.optimize.curve_fit(model_cubic, x, d_real)
			popt_i, pcov_i = scipy.optimize.curve_fit(model_cubic, x, d_imag)
			real_popts_cubic.append(popt_r)
			imag_popts_cubic.append(popt_i)
			d_reals.append(d_real)
			d_imags.append(d_imag)


d_big_real = np.array(d_reals)
d_big_imag = np.array(d_imags)
# d_big = np.concatenate((d_big_real,d_big_imag))
d_big = d_big_imag

# cubic_start_points = real_popts_cubic
cubic_start_points = imag_popts_cubic
# cubic_start_points = np.concatenate((real_popts_cubic,imag_popts_cubic))
# cubic_start_points = imag_popts_cubic

from pdb import set_trace as brk
n_fg_params = 4

def likelihood32_big_cubiconly(theta, return_model=False):
	power_amps = theta[0:16]
	a_means    = theta[16:32]
	m_params   = theta[32:]
	m_param_sets = m_params.reshape(-1,36)
	a_param_sets = np.array([m_param_set[n_fg_params:] for m_param_set in m_param_sets])
	m=np.array([model_cubic_plus_Fourier(x,*m_param_set) for m_param_set in m_param_sets])
	sigma=1.e-3
	chisq = np.sum((d_big-m)**2.0/sigma**2.0)

	power_amp=theta[-2]
	a_mean=theta[-1]
	aphia = np.sum([np.sum((a_param_sets[:,i])**2.0/power_amps[i]**2.0) for i in range(len(power_amps)) ]) #for the sine wave modes
	aphia = aphia + np.sum([np.sum((a_param_sets[:,i+(32/2)])**2.0/power_amps[i]**2.0) for i in range(len(power_amps)) ]) #for the cosine wave modes (which fill the second half of each a_param_set i.e. a_param_sets[:,(32/2):])
	# aphia = np.sum([np.sum((a_param_sets[:,i]-a_means[i])**2.0/power_amps[i]**2.0) for i in range(len(power_amps)) ])
	logdetphi = np.sum(np.log(power_amps**2.0))
	loglikelihood = chisq+aphia+logdetphi
	Print=False
	if Print: print 'loglikelihood,chisq,aphia,logdetphi', loglikelihood,chisq,aphia,logdetphi
	# print 'Time since start: {}'.format(time.time()-start)
	# brk()
	if return_model:
		return m
	else:
		return loglikelihood

# n_par = 32+1*(nu*nv-1)*35 #16 means, 16 power_amps, (nu*nv-1)*4 pl and quad params and (nu*nv-1)*32 Fourier amps
n_par = 32+len(cubic_start_points)*36 #16 means, 16 power_amps, (nu*nv-1)*4 pl and quad params and (nu*nv-1)*32 Fourier amps
x032_big_cubiconly=np.ones(n_par)
x032_big_cubiconly[0:32]=100.0
x032_big_cubiconly_m_params = x032_big_cubiconly[32:]
x032_big_cubiconly_m_params = x032_big_cubiconly_m_params.reshape(-1,36)
for i in range(len(cubic_start_points)):
	x032_big_cubiconly_m_params[i][0:4] = cubic_start_points[i]


start=time.time()
out32_big_cubiconly = scipy.optimize.minimize(likelihood32_big_cubiconly, x032_big_cubiconly, method='Powell')
print 'Time since start: {}'.format(time.time()-start)



print likelihood32_big_cubiconly(out32_big_cubiconly['x'])
residuals_big_cubiconly2 = d_big-likelihood32_big_cubiconly(out32_big_cubiconly['x'],True)

print (out32_big_cubiconly['x'][0:16]*1.e3)**2.







###
# Cubic + pl + Fourier
###
def model_cubic_plus_pl(x,q1,q2,q3,q4,p1):
	p_val = model_pl(x,p1)
	cubic_val = model_cubic(x,q1,q2,q3,q4)
	m=p_val+cubic_val
	return m

def model_cubic_plus_pl_plus_Fourier(x,q1,q2,q3,q4,p1,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,a33,a34):
	p_val = model_pl(x,p1)
	cubic_val = model_cubic(x,q1,q2,q3,q4)
	F_val = model_Fourier(x,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,a33,a34)
	m=p_val+cubic_val+F_val
	return m


import time
u_c,v_c = np.array(Tb_nu.shape[1:])/2
# u_c,v_c = 256,256
fftTb_nu_subset = fftTb_nu[:,u_c-15:u_c+16,v_c-15:v_c+16]
real_popts_cubic_pl = []
imag_popts_cubic_pl = []
d_reals = []
d_imags = []
start=time.time()
nu=8
nv=1
# nv=31
for i in range(15,15+nu,1):
	print i,'Time since start: {}'.format(time.time()-start)
	# for j in range(nv):
	for j in range(15,15+nv,1):
		if not (i==15 and j==15):
			d_real = fftTb_nu_subset[:,i,j].real
			d_imag = fftTb_nu_subset[:,i,j].imag
			popt_r, pcov_r = scipy.optimize.curve_fit(model_cubic_plus_pl, x, d_real)
			popt_i, pcov_i = scipy.optimize.curve_fit(model_cubic_plus_pl, x, d_imag)
			real_popts_cubic_pl.append(popt_r)
			imag_popts_cubic_pl.append(popt_i)
			d_reals.append(d_real)
			d_imags.append(d_imag)


d_big_real = np.array(d_reals)
d_big_imag = np.array(d_imags)
# d_big = np.concatenate((d_big_real,d_big_imag))
d_big = d_big_imag

# cubic_pl_start_points = real_popts_cubic_pl
cubic_pl_start_points = imag_popts_cubic_pl
# cubic_pl_start_points = np.concatenate((real_popts_cubic_pl,imag_popts_cubic_pl))
# cubic_pl_start_points = imag_popts_cubic_pl

from pdb import set_trace as brk
n_fg_params = 5

def likelihood32_big_cubic_pl(theta, return_model=False):
	power_amps = theta[0:16]
	a_means    = theta[16:32]
	m_params   = theta[32:]
	m_param_sets = m_params.reshape(-1,37)
	a_param_sets = np.array([m_param_set[n_fg_params:] for m_param_set in m_param_sets])
	m=np.array([model_cubic_plus_pl_plus_Fourier(x,*m_param_set) for m_param_set in m_param_sets])
	sigma=1.e-3
	chisq = np.sum((d_big-m)**2.0/sigma**2.0)

	power_amp=theta[-2]
	a_mean=theta[-1]
	aphia = np.sum([np.sum((a_param_sets[:,i])**2.0/power_amps[i]**2.0) for i in range(len(power_amps)) ]) #for the sine wave modes
	aphia = aphia + np.sum([np.sum((a_param_sets[:,i+(32/2)])**2.0/power_amps[i]**2.0) for i in range(len(power_amps)) ]) #for the cosine wave modes (which fill the second half of each a_param_set i.e. a_param_sets[:,(32/2):])
	# aphia = np.sum([np.sum((a_param_sets[:,i]-a_means[i])**2.0/power_amps[i]**2.0) for i in range(len(power_amps)) ]) #for the sine wave modes
	logdetphi = np.sum(np.log(power_amps**2.0))
	loglikelihood = chisq+aphia+logdetphi
	Print=False
	if Print: print 'loglikelihood,chisq,aphia,logdetphi', loglikelihood,chisq,aphia,logdetphi
	# print 'Time since start: {}'.format(time.time()-start)
	# brk()
	if return_model:
		return m
	else:
		return loglikelihood

# n_par = 32+1*(nu*nv-1)*35 #16 means, 16 power_amps, (nu*nv-1)*4 pl and quad params and (nu*nv-1)*32 Fourier amps
n_par = 32+len(cubic_pl_start_points)*37 #16 means, 16 power_amps, (nu*nv-1)*4 pl and quad params and (nu*nv-1)*32 Fourier amps
x032_big_cubic_pl=np.ones(n_par)
x032_big_cubic_pl[0:32]=100.0
x032_big_cubic_pl_m_params = x032_big_cubic_pl[32:]
x032_big_cubic_pl_m_params = x032_big_cubic_pl_m_params.reshape(-1,37)
for i in range(len(cubic_pl_start_points)):
	x032_big_cubic_pl_m_params[i][0:5] = cubic_pl_start_points[i]

###
# Generate start point for cubic_plus_pl_plus_Fourier from the current best likelihood - quadratic_plus_pl_plus_Fourier ML coefficients
###
n_par = 32+len(cubic_pl_start_points)*37 #16 means, 16 power_amps, (nu*nv-1)*4 pl and quad params and (nu*nv-1)*32 Fourier amps
x032_big_cubic_pl=np.zeros(n_par)
x032_big_cubic_pl[0:16]=out32_big['x'][0:16].copy()
x032_big_cubic_pl[16:32]=100.0
x032_big_cubic_pl_m_params = x032_big_cubic_pl[32:]
x032_big_cubic_pl_m_params = x032_big_cubic_pl_m_params.reshape(-1,37)
for i in range(len(cubic_pl_start_points)):
	x032_big_cubic_pl_m_params[i][0:3] = out32_big['x'][32:].reshape(-1,36)[i][0:3].copy()#Quadratic vals
	x032_big_cubic_pl_m_params[i][4] = out32_big['x'][32:].reshape(-1,36)[i][3].copy() #Power law vals
	x032_big_cubic_pl_m_params[i][4:] = out32_big['x'][32:].reshape(-1,36)[i][3:].copy() #Fourier vals

	
print likelihood32_big_cubic_pl(x032_big_cubic_pl)


start=time.time()
out32_big_cubic_pl = scipy.optimize.minimize(likelihood32_big_cubic_pl, x032_big_cubic_pl, method='Powell')
print 'Time since start: {}'.format(time.time()-start)



print likelihood32_big_cubic_pl(out32_big_cubic_pl['x'])
residuals_big_cubic_pl2 = d_big-likelihood32_big_cubic_pl(out32_big_cubic_pl['x'],True)

print (out32_big_cubic_pl['x'][0:16]*1.e3)**2.










###
# quadratic + variable pl + Fourier
###


def model_variable_pl(x,p1, delta_beta):
	p_val = p1 * (nu_array_MHz/nu_min_MHz)**(-2.6-(3/(1+delta_beta)))
	return p_val


def model_q_plus_variable_pl(x,q1,q2,q3,p1,delta_beta):
	p_val = model_variable_pl(x,p1,delta_beta)
	q_val = model_q(x,q1,q2,q3)
	m=p_val+q_val
	return m


def model_q_plus_variable_pl_plus_Fourier(x,q1,q2,q3,p1,delta_beta,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,a33,a34):
	p_val = model_variable_pl(x,p1,delta_beta)
	q_val = model_q(x,q1,q2,q3)
	F_val = model_Fourier(x,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30,a31,a32,a33,a34)
	m=p_val+q_val+F_val
	return m


def likelihood32_little_variable_pl_No_Fourier(theta, d, return_model=False):
	p1,delta_beta = theta
	m = model_variable_pl(x,p1,delta_beta)

	sigma=1.0
	chisq = np.sum((d-m)**2.0/sigma**2.0)
	loglikelihood = chisq
	Print=False
	if Print: print 'loglikelihood', loglikelihood
	# brk()
	if return_model:
		return m
	else:
		return loglikelihood


def likelihood32_little_q_plus_variable_pl_No_Fourier(theta, d, return_model=False):
	q1,q2,q3,p1,delta_beta = theta
	m = model_q_plus_variable_pl(x,q1,q2,q3,p1,delta_beta)

	sigma=1.0
	chisq = np.sum((d-m)**2.0/sigma**2.0)
	loglikelihood = chisq
	Print=False
	if Print: print 'loglikelihood', loglikelihood
	# brk()
	if return_model:
		return m
	else:
		return loglikelihood


start=time.time()
out32_little_q_plus_variable_pl_No_Fourier = scipy.optimize.minimize(likelihood32_little_q_plus_variable_pl_No_Fourier, np.ones(5), args=(d_real,False), method='Powell')
# out32_little_variable_pl_No_Fourier = scipy.optimize.minimize(likelihood32_little_variable_pl_No_Fourier, np.ones(2), args=(d_real,False), method='Powell')
print 'Time since start: {}'.format(time.time()-start)

print model_q_plus_variable_pl(x,*out32_little_q_plus_variable_pl_No_Fourier['x']) - d_real


import time
u_c,v_c = np.array(Tb_nu.shape[1:])/2
# u_c,v_c = 256,256
fftTb_nu_subset = fftTb_nu[:,u_c-15:u_c+16,v_c-15:v_c+16]
real_popts_q_plus_variable_pl = []
imag_popts_q_plus_variable_pl = []
d_reals = []
d_imags = []
start=time.time()
nu=8
nv=1
# nv=31
for i in range(15,15+nu,1):
	print i,'Time since start: {}'.format(time.time()-start)
	# for j in range(nv):
	for j in range(15,15+nv,1):
		if not (i==15 and j==15):
			d_real = fftTb_nu_subset[:,i,j].real
			d_imag = fftTb_nu_subset[:,i,j].imag
			popt_r = scipy.optimize.minimize(likelihood32_little_q_plus_variable_pl_No_Fourier, np.ones(5), args=(d_real,False), method='Powell')['x']
			popt_i = scipy.optimize.minimize(likelihood32_little_q_plus_variable_pl_No_Fourier, np.ones(5), args=(d_imag,False), method='Powell')['x']
			real_popts_q_plus_variable_pl.append(popt_r)
			imag_popts_q_plus_variable_pl.append(popt_i)
			d_reals.append(d_real)
			d_imags.append(d_imag)


d_big_real = np.array(d_reals)
d_big_imag = np.array(d_imags)
# d_big = np.concatenate((d_big_real,d_big_imag))
d_big = d_big_imag

# q_plus_variable_pl_start_points = real_popts_q_plus_variable_pl
q_plus_variable_pl_start_points = imag_popts_q_plus_variable_pl
# q_plus_variable_pl_start_points = np.concatenate((real_popts_q_plus_variable_pl,imag_popts_q_plus_variable_pl))
# q_plus_variable_pl_start_points = imag_popts_q_plus_variable_pl

from pdb import set_trace as brk
n_fg_params = 5

def likelihood32_big_q_plus_variable_pl(theta, return_model=False):
	power_amps = theta[0:16]
	a_means    = theta[16:32]
	m_params   = theta[32:]
	m_param_sets = m_params.reshape(-1,37)
	a_param_sets = np.array([m_param_set[n_fg_params:] for m_param_set in m_param_sets])
	m=np.array([model_q_plus_variable_pl_plus_Fourier(x,*m_param_set) for m_param_set in m_param_sets])
	sigma=1.e-3
	chisq = np.sum((d_big-m)**2.0/sigma**2.0)

	power_amp=theta[-2]
	a_mean=theta[-1]
	aphia = np.sum([np.sum((a_param_sets[:,i])**2.0/power_amps[i]**2.0) for i in range(len(power_amps)) ]) #for the sine wave modes
	aphia = aphia + np.sum([np.sum((a_param_sets[:,i+(32/2)])**2.0/power_amps[i]**2.0) for i in range(len(power_amps)) ]) #for the cosine wave modes (which fill the second half of each a_param_set i.e. a_param_sets[:,(32/2):])
	# aphia = np.sum([np.sum((a_param_sets[:,i]-a_means[i])**2.0/power_amps[i]**2.0) for i in range(len(power_amps)) ]) #for the sine wave modes
	logdetphi = np.sum(np.log(power_amps**2.0))
	loglikelihood = chisq+aphia+logdetphi
	Print=False
	if Print: print 'loglikelihood,chisq,aphia,logdetphi', loglikelihood,chisq,aphia,logdetphi
	# print 'Time since start: {}'.format(time.time()-start)
	# brk()
	if return_model:
		return m
	else:
		return loglikelihood

# n_par = 32+1*(nu*nv-1)*35 #16 means, 16 power_amps, (nu*nv-1)*4 pl and quad params and (nu*nv-1)*32 Fourier amps
n_par = 32+len(q_plus_variable_pl_start_points)*37 #16 means, 16 power_amps, (nu*nv-1)*4 pl and quad params and (nu*nv-1)*32 Fourier amps
x032_big_q_plus_variable_pl=np.ones(n_par)
x032_big_q_plus_variable_pl[0:32]=100.0
x032_big_q_plus_variable_pl_m_params = x032_big_q_plus_variable_pl[32:]
x032_big_q_plus_variable_pl_m_params = x032_big_q_plus_variable_pl_m_params.reshape(-1,37)
for i in range(len(q_plus_variable_pl_start_points)):
	x032_big_q_plus_variable_pl_m_params[i][0:5] = q_plus_variable_pl_start_points[i]



start=time.time()
out32_big_q_plus_variable_pl = scipy.optimize.minimize(likelihood32_big_q_plus_variable_pl, x032_big_q_plus_variable_pl, method='Powell')
print 'Time since start: {}'.format(time.time()-start)



print likelihood32_big_q_plus_variable_pl(out32_big_q_plus_variable_pl['x'])
residuals_big_q_plus_variable_pl2 = d_big-likelihood32_big_q_plus_variable_pl(out32_big_q_plus_variable_pl['x'],True)

print (out32_big_q_plus_variable_pl['x'][0:16]*1.e3)**2.








print log10((out32_big['x'][0:16]*1.e0)**2.)
print log10((out32_big_qonly['x'][0:16]*1.e0)**2.)
print log10((out32_big_cubiconly['x'][0:16]*1.e0)**2.)
print log10((out32_big_cubic_pl['x'][0:16]*1.e0)**2.)


print (out32_big['x'][0:16]*1.e0)**2.
print (out32_big_qonly['x'][0:16]*1.e0)**2.
print (out32_big_cubiconly['x'][0:16]*1.e0)**2.
print (out32_big_cubic_pl['x'][0:16]*1.e0)**2.


print (out32_big['x'][0:16]*1.e3)**2.
print (out32_big_qonly['x'][0:16]*1.e3)**2.
print (out32_big_cubiconly['x'][0:16]*1.e3)**2.
print (out32_big_cubic_pl['x'][0:16]*1.e3)**2.



 
print likelihood32_big_cubic_pl(out32_big_cubic_pl['x'])
print likelihood32_big_cubiconly(out32_big_cubiconly['x'])
print likelihood32_big_qonly(out32_big_qonly['x'])
print likelihood32_big(out32_big['x'])


test = np.zeros_like(out32_big_cubic_pl['x'])
test[0:16]=out32_big['x'][0:16]
test[16:32]=out32_big_cubic_pl['x'][16:32]
testl_m_params = test[32:]
testl_m_params = testl_m_params.reshape(-1,37)
for i in range(len(cubic_pl_start_points)):
	testl_m_params[i][0:3] = out32_big['x'][32:].reshape(-1,36)[i][0:3]
	testl_m_params[i][4] = out32_big['x'][32:].reshape(-1,36)[i][3]
	testl_m_params[i][4:] = out32_big['x'][32:].reshape(-1,36)[i][3:]
	# print testl_m_params[i]

	

print likelihood32_big_cubic_pl(test)










