
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
from pdb import set_trace as brk

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
from SimData_v1d4 import calc_mean_binned_k_vals

from Linalg_v1d1 import IDFT_Array_IDFT_1D_WQ, generate_gridding_matrix_vis_ordered_to_chan_ordered_WQ
from Linalg_v1d1 import IDFT_Array_IDFT_1D_WQ_ZM, generate_gridding_matrix_vis_ordered_to_chan_ordered_ZM

from Generate_matrix_stack_v1d1 import BuildMatrices

from Utils_v1d0 import PriorC, ParseCommandLineArguments, DataUnitConversionmkandJyperpix, WriteDataToFits
from Utils_v1d0 import ExtractDataFrom21cmFASTCube, plot_signal_vs_MLsignal_residuals

from GenerateForegroundCube_v1d0 import generate_Jelic_cube, generate_data_from_loaded_EoR_cube
from GenerateForegroundCube_v1d0 import generate_test_signal_from_image_cube
from GenerateForegroundCube_v1d0 import top_hat_average_temperature_cube_to_lower_res_31x31xnf_cube

###
# Set analysis parameters
###
argv = sys.argv[1:]
PCLA = ParseCommandLineArguments()
nq = PCLA.parse(argv) #defaults to 2 (i.e. jointly fits for a second order quadratic)
# nq = 2
# nq = 0
nq = 2
npl = 0 #Overwrites quadratic term when nq=2, otherwise unused.

# nf=38
# neta=38
nf=128
neta=128
neta = neta -nq
	
nu=3
nv=3
nx=3
ny=3

sigma=20.e-1
# sigma=80.e-1
#sigma=100.e-1
sub_quad = True
# sub_quad = False

small_cube = nu<=7 and nv<=7
nuv = (nu*nv-1)
Show=False
current_file_version = 'Likelihood_v1d76_3D_ZM'
array_save_directory = 'array_storage/{}_{}_{}_{}_{}_{:.1E}/'.format(current_file_version,nu,nv,neta,nq,sigma).replace('.','d')

BM = BuildMatrices(array_save_directory, nu, nv, nx, ny, neta, nf, nq, sigma, npl=npl)
delete_existing_matrix_stack = True
BM.build_minimum_sufficient_matrix_stack(delete_existing_matrix_stack=delete_existing_matrix_stack)

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
k_vals = calc_mean_binned_k_vals(mod_k_masked, k_cube_voxels_in_bin, save_k_vals=True, k_vals_file='k_vals_nf128.txt')

# do_cylindrical_binning = True
do_cylindrical_binning = False
if do_cylindrical_binning:
	n_k_perp_bins=2
	k_cube_voxels_in_bin, modkbins_containing_voxels, k_perp_bins = generate_k_cube_model_cylindrical_binning(mod_k_masked, k_z_masked, k_y_masked, k_x_masked, n_k_perp_bins, nu,nv,nx,ny,nf,neta,nq)





#----------------------

# use_foreground_cube = True
use_foreground_cube = False

if use_foreground_cube:
	###
	# beta_experimental_mean = 2.55+0 #Used in my `Jelic' GDSE models before 5th October 2017
	beta_experimental_mean = 2.63+0   #Matches beta_150_408 in Mozden, Bowman et al. 2016
	# beta_experimental_std  = 0.1    #Used in my `Jelic' GDSE models before 5th October 2017
	beta_experimental_std  = 0.02      #A conservative over-estimate of the dbeta_150_408=0.01 (dbeta_90_190=0.02) in Mozden, Bowman et al. 2016
	gamma_mean             = -2.7     #Revise to match published values
	gamma_sigma            = 0.3      #Revise to match published values
	Tb_experimental_mean_K = 194.0    #Matches GSM mean in region A (see /users/psims/Cav/EoR/Missing_Radio_Flux/Surveys/Convert_GSM_to_HEALPIX_Map_and_Cartesian_Projection_Fits_File_v6d0_pygsm.py)
	# Tb_experimental_std_K  = 85.0   #65th percentile std at at 0.333 degree resolution in 50 deg by 50 deg maps centered on Dec=-30.0
	# Tb_experimental_std_K  = 9.0      #Coldest 12 deg.**2 region at 56 arcmin res. centered on -30. deg declination (see GSM_map_std_at_-30_dec_v1d0.ipynb)
	# Tb_experimental_std_K  = 33.0     #Median 12 deg.**2 region at 56 arcmin res. centered on -30. deg declination (see GSM_map_std_at_-30_dec_v1d0.ipynb)
	Tb_experimental_std_K  = 62.0     #70th percentile 12 deg.**2 region at 56 arcmin res. centered on -30. deg declination (see GSM_map_std_at_-30_dec_v1d0.ipynb)
	# Tb_experimental_std_K  = 62.0   #Median std at at 0.333 degree resolution in 50 deg by 50 deg maps centered on Dec=-30.0
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
	# fits_storage_dir = 'fits_storage/multi_frequency_band_test3/Jelic_nu_min_MHz_{}_TbStd_{}_beta_{}_dbeta{}/'.format(nu_min_MHz, Tb_experimental_std_K, beta_experimental_mean, beta_experimental_std).replace('.','d')
	simulation_FoV = 12.0             #Matches EoR simulation
	simulation_resolution = simulation_FoV/127. #Matches EoR sim (note: use closest odd val., so 127 rather than 128, for easier FFT normalisation)
	fits_storage_dir = 'fits_storage/multi_frequency_band_pythonPStest1/Jelic_nu_min_MHz_{}_TbStd_{}_beta_{}_dbeta{}/'.format(nu_min_MHz, Tb_experimental_std_K, beta_experimental_mean, beta_experimental_std).replace('.','d')
	###
	HF_nu_min_MHz_array = [210,220,230]
	# HF_nu_min_MHz_array = [205,215,225]
	foreground_outputs = generate_Jelic_cube(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show, beta_experimental_mean,beta_experimental_std,gamma_mean,gamma_sigma,Tb_experimental_mean_K,Tb_experimental_std_K,nu_min_MHz,channel_width_MHz, generate_additional_extrapolated_HF_foreground_cube=True, fits_storage_dir=fits_storage_dir, HF_nu_min_MHz_array=HF_nu_min_MHz_array, simulation_FoV=simulation_FoV, simulation_resolution=simulation_resolution)
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
	# Look at foreground quadratic residuals (note: EoR signal is ~10 mK)
	### 
	pylab.close('all')
	pylab.errorbar(arange(len(fg.reshape(38,8)[:,0].real)), fg.reshape(38,8)[:,0].real)
	pylab.show()

	m = np.polyval(np.polyfit(arange(len(fg.reshape(38,8)[:,0].real)),fg.reshape(38,8)[:,0].real,2), arange(len(fg.reshape(38,8)[:,0].real)))
	pylab.close('all')
	pylab.errorbar(arange(len(fg.reshape(38,8)[:,0].real)), fg.reshape(38,8)[:,0].real-m)
	pylab.show()


	# run_quad_foreground_test=True
	run_quad_foreground_test=False
	if run_quad_foreground_test:
		# Create a purely quadratic foreground model for testing - it should be fit out perfectly!
		fg_reshaped = fg.reshape(38,8)
		fg_quad_model_reshaped = np.zeros_like(fg_reshaped)
		for i_uv in range(fg_reshaped.shape[1]):
			# Derive quadratic model for spectral real
			ml_quad_r = np.polyfit(arange(len(fg_reshaped[:,i_uv].real)),fg_reshaped[:,i_uv].real,2)
			ml_quad_r[-1]=0.0 #Zero constant term to avoid numerical issues with extremely high likelihood!
			m_r = np.polyval(ml_quad_r, arange(len(fg_reshaped[:,i_uv].real)))
			# Derive quadratic model for spectral imag
			ml_quad_i = np.polyfit(arange(len(fg_reshaped[:,i_uv].imag)),fg_reshaped[:,i_uv].imag,2)
			ml_quad_i[-1]=0.0 #Zero constant term to avoid numerical issues with extremely high likelihood!
			m_i = np.polyval(ml_quad_i, arange(len(fg_reshaped[:,i_uv].imag)))
			fg_quad_model_reshaped[:,i_uv].real = m_r
			fg_quad_model_reshaped[:,i_uv].imag = m_i

		fg_quad_model = fg_quad_model_reshaped.flatten()
		m = np.polyval(np.polyfit(arange(len(fg_quad_model.reshape(38,8)[:,0].real)),fg_quad_model.reshape(38,8)[:,0].real,2), arange(len(fg_quad_model.reshape(38,8)[:,0].real)))
		pylab.close('all')
		pylab.errorbar(arange(len(fg_quad_model.reshape(38,8)[:,0].real)), fg_quad_model.reshape(38,8)[:,0].real-m)
		pylab.show()

		fg = fg_quad_model.flatten()







use_EoR_cube = True
# use_EoR_cube = False
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
	if use_foreground_cube:
		d += generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector(sigma, fg)[0]
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







