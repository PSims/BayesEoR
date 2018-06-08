#!/users/psims/anaconda2/bin/python

#--------------------------------------------
# Imports
#--------------------------------------------
from subprocess import os
import sys
# head,tail = os.path.split(os.getcwd())
head,tail = os.path.split(os.path.split(os.getcwd())[0])
sys.path.append(head)
from BayesEoR import * #Make everything available for now, this can be refined later
import BayesEoR.Params.params as p
use_MultiNest = True #Set to false for large parameter spaces
if use_MultiNest:
	from pymultinest.solve import solve
else:
	import PyPolyChord.PyPolyChord as PolyChord

#--------------------------------------------
# Set analysis parameters
#--------------------------------------------
# Model Params
args = p.BayesEoRParser()
print args
print 'p.beta', p.beta
if args.beta: p.beta = float(args.beta) #Overwrite parameter file beta with value chosen from the command line if it is included 
print 'args.beta', args.beta
print 'p.beta', p.beta

# raw_input()

nq = 2 #Overwrite PCLA selection
npl = 1 #Overwrites quadratic term when nq=2, otherwise unused.
print 'nq', nq
print 'npl', npl
sub_ML_monopole_term_model = False #Improve numerical precision. Can be used for improving numerical precision when performing evidence comparison.
sub_ML_monopole_plus_first_LW_term_model = True #Improve numerical precision. Can be used for improving numerical precision when performing evidence comparison.
sub_MLLWM = False #Improve numerical precision. DO NOT USE WHEN PERFORMING EVIDENCE COMPARISON! Can only be used for parameter estimation not evidence comparison since subtracting different MLLWM (with different evidences) from the data when comparing different LWMs will alter the relative evidences of the subtracted models. In effect subtracting a higher evidence MLLWM1 reduces the evidence for the fit to the residuals with MLLWM1 relative to fitting low evidence MLLWM2 and refitting with MLLWM2. It is only correct to compare evidences when doing no qsub or when the qsub model is fixed such as with sub_ML_monopole_term_model.
# Cube size
nf=p.nf
neta=p.neta
neta=neta -nq
nu=p.nu
nv=p.nv
nx=p.nx
ny=p.ny
# Data noise
sigma=100.e-1
# Auxiliary and derived params
small_cube = nu<=7 and nv<=7
nuv = (nu*nv-1)
Show=False
chan_selection=''
Fz_normalisation = nf**0.5
DFT2D_Fz_normalisation = (nu*nv*nf)**0.5
n_Fourier = (nu*nv-1)*nf
n_LW = (nu*nv-1)*nq
n_model = n_Fourier+n_LW
n_dat = n_Fourier
current_file_version = 'Likelihood_v1d76_3D_ZM'
array_save_directory = 'array_storage/{}_nu_{}_nv_{}_neta_{}_nq_{}_npl_{}_sigma_{:.1E}/'.format(current_file_version,nu,nv,neta,nq,npl,sigma).replace('.','d')
if npl>0:
	array_save_directory=array_save_directory.replace('_sigma', '_beta_{:.2E}_sigma'.format(p.beta))


#--------------------------------------------
# Construct matrices
#--------------------------------------------
BM = BuildMatrices(array_save_directory, nu, nv, nx, ny, neta, nf, nq, sigma, npl=npl)
overwrite_existing_matrix_stack = False #Can be set to False unless npl>0
# overwrite_existing_matrix_stack = True #Can be set to False unless npl>0
proceed_without_overwrite_confirmation = False #Allows overwrite_existing_matrix_stack to be run without having to manually accept the deletion of the old matrix stack
BM.build_minimum_sufficient_matrix_stack(overwrite_existing_matrix_stack=overwrite_existing_matrix_stack, proceed_without_overwrite_confirmation=proceed_without_overwrite_confirmation)

#--------------------------------------------
# Generate GRN data. Currently only the bin selection component of this function is being used and should probably be spun out.
#--------------------------------------------
test_sim_out = generate_test_sim_signal_with_large_spectral_scales_2_21cmFAST_Binning(nu,nv,nx,ny,nf,neta,nq)
if small_cube or True:
	s, s_im, s_LW_only, s_im_LW_only, s_fourier_only, s_im_fourier_only, bin_selector_in_k_cube_mask, high_spatial_frequency_selector_mask, k_cube_signal, k_sigma, ZM_mask, k_z_mean_mask = test_sim_out
	bin_selector_cube_ordered_list = bin_selector_in_k_cube_mask
else:
	s=test_sim_out[0]
	bin_selector_cube_ordered_list = test_sim_out[6]
	high_spatial_frequency_selector_mask = test_sim_out[7]
	k_sigma = test_sim_out[9]
	ZM_mask = test_sim_out[10]
	k_z_mean_mask = test_sim_out[11]
test_sim_out=0

#--------------------------------------------
# Define power spectral bins and coordinate cubes
#--------------------------------------------
if nq==0:
	map_bins_out = map_out_bins_for_power_spectral_coefficients_HERA_Binning(nu,nv,nx,ny,nf,neta,nq, bin_selector_cube_ordered_list,high_spatial_frequency_selector_mask, ZM_mask, k_z_mean_mask)
	bin_selector_in_model_mask_vis_ordered = map_bins_out
	bin_selector_vis_ordered_list = bin_selector_in_model_mask_vis_ordered
else:
	map_bins_out = map_out_bins_for_power_spectral_coefficients_WQ_v2_HERA_Binning(nu,nv,nx,ny,nf,neta,nq, bin_selector_cube_ordered_list,high_spatial_frequency_selector_mask, ZM_mask, k_z_mean_mask)
	bin_selector_in_model_mask_vis_ordered_WQ, LW_modes_only_boolean_array_vis_ordered = map_bins_out
	bin_selector_vis_ordered_list = bin_selector_in_model_mask_vis_ordered_WQ
map_bins_out=0

mod_k, k_x, k_y, k_z, deltakperp, deltakpara, x, y, z = generate_k_cube_in_physical_coordinates_21cmFAST_v2d0(nu,nv,nx,ny,nf,neta,p.box_size_21cmFAST_pix_sc,p.box_size_21cmFAST_Mpc_sc)
k=mod_k.copy()
k_vis_ordered = k.T.flatten()
# modk_vis_ordered_list = [k_vis_ordered[bin_selector.T.flatten()] for bin_selector in bin_selector_cube_ordered_list]

k_x_masked = generate_masked_coordinate_cubes(k_x, nu,nv,nx,ny,nf,neta,nq)
k_y_masked = generate_masked_coordinate_cubes(k_y, nu,nv,nx,ny,nf,neta,nq)
k_z_masked = generate_masked_coordinate_cubes(k_z, nu,nv,nx,ny,nf,neta,nq)
mod_k_masked = generate_masked_coordinate_cubes(mod_k, nu,nv,nx,ny,nf,neta,nq)

k_cube_voxels_in_bin, modkbins_containing_voxels = generate_k_cube_model_spherical_binning(mod_k_masked, k_z_masked, nu,nv,nx,ny,nf,neta,nq)
modk_vis_ordered_list = [mod_k_masked[k_cube_voxels_in_bin[i_bin]] for i_bin in range(len(k_cube_voxels_in_bin))]
k_vals_file_name = 'k_vals_nu_{}_nv_{}_nf_{}_nq_{}.txt'.format(nu,nv,nf,nq)
k_vals = calc_mean_binned_k_vals(mod_k_masked, k_cube_voxels_in_bin, save_k_vals=True, k_vals_file=k_vals_file_name)

do_cylindrical_binning = False
if do_cylindrical_binning:
	n_k_perp_bins=2
	k_cube_voxels_in_bin, modkbins_containing_voxels, k_perp_bins = generate_k_cube_model_cylindrical_binning(mod_k_masked, k_z_masked, k_y_masked, k_x_masked, n_k_perp_bins, nu,nv,nx,ny,nf,neta,nq)

#--------------------------------------------
# Generate mock-GDSE data
#--------------------------------------------
use_GDSE_foreground_cube = True
if use_GDSE_foreground_cube:
	###
	# GDSE foreground_outputs
	###	
	foreground_outputs = generate_Jelic_cube(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show, p.beta_experimental_mean,p.beta_experimental_std,p.gamma_mean,p.gamma_sigma,p.Tb_experimental_mean_K,p.Tb_experimental_std_K,p.nu_min_MHz,p.channel_width_MHz, generate_additional_extrapolated_HF_foreground_cube=True, fits_storage_dir=p.fits_storage_dir, HF_nu_min_MHz_array=p.HF_nu_min_MHz_array, simulation_FoV_deg=p.simulation_FoV_deg, simulation_resolution_deg=p.simulation_resolution_deg,random_seed=314211)

	fg_GDSE, s_GDSE, Tb_nu, beta_experimental_mean,beta_experimental_std,gamma_mean,gamma_sigma,Tb_experimental_mean_K,Tb_experimental_std_K,nu_min_MHz,channel_width_MHz, HF_Tb_nu = foreground_outputs
	foreground_outputs = []

	plot_figure = False
	if plot_figure:
		construct_aplpy_image_from_fits('/gpfs/data/jpober/psims/EoR/Python_Scripts/BayesEoR/git_version/BayesEoR/spec_model_tests/fits_storage/multi_frequency_band_pythonPStest1/Jelic_nu_min_MHz_159d0_TbStd_66d1866884116_beta_2d63_dbeta0d02/', 'Jelic_GDSE_cube_159MHz_mK', run_convert_from_mK_to_K=True, run_remove_unused_header_variables=True)


#--------------------------------------------
# Generate mock-free-free data
#--------------------------------------------
use_freefree_foreground_cube = True
if use_freefree_foreground_cube:
	###
	# diffuse free-free foreground_outputs
	###	
	foreground_outputs_ff = generate_Jelic_cube(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show, p.beta_experimental_mean_ff,p.beta_experimental_std_ff,p.gamma_mean_ff,p.gamma_sigma_ff,p.Tb_experimental_mean_K_ff,p.Tb_experimental_std_K_ff,p.nu_min_MHz_ff,p.channel_width_MHz_ff, generate_additional_extrapolated_HF_foreground_cube=True, fits_storage_dir=p.fits_storage_dir_ff, HF_nu_min_MHz_array=p.HF_nu_min_MHz_array_ff, simulation_FoV_deg=p.simulation_FoV_deg_ff, simulation_resolution_deg=p.simulation_resolution_deg_ff,random_seed=3142111)

	fg_ff, s_ff = foreground_outputs_ff[:2]

	plot_figure = False
	if plot_figure:
		construct_aplpy_image_from_fits('/gpfs/data/jpober/psims/EoR/Python_Scripts/BayesEoR/git_version/BayesEoR/spec_model_tests/fits_storage/free_free_emission/Free_free_nu_min_MHz_159d0_TbStd_0d698184469839_beta_2d15_dbeta1e-10/', 'Jelic_GDSE_cube_159MHz_mK', run_convert_from_mK_to_K=True, run_remove_unused_header_variables=True)


#--------------------------------------------
# Load EGS data
#--------------------------------------------
use_EGS_cube = True
if use_EGS_cube:
	print 'Using use_EGS_cube data'
	s_EGS, abc_EGS, scidata1_EGS = generate_data_from_loaded_EGS_cube(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show,p.EGS_npz_path)

#--------------------------------------------
# Load EoR data
#--------------------------------------------
use_EoR_cube = True
if use_EoR_cube:
	print 'Using use_EoR_cube data'
	s_EoR, abc, scidata1 = generate_data_from_loaded_EoR_cube_v2d0(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show,chan_selection,p.EoR_npz_path_sc)


calc_im_domain_noise = False
if calc_im_domain_noise:
	sigma_complex = sigma/2**0.5
	noise_real = np.random.normal(0,sigma_complex,nu*nv*nf)
	noise_imag = np.random.normal(0,sigma_complex,nu*nv*nf)
	noise = noise_real+1j*noise_imag

	blank_cube = np.zeros([38,512,512])

	sci_f, sci_v, sci_u = blank_cube.shape
	sci_v_centre = sci_v/2
	sci_u_centre = sci_u/2
	blank_cube_subset = blank_cube[0:nf,sci_u_centre-nu/2:sci_u_centre+nu/2+1,sci_v_centre-nv/2:sci_v_centre+nv/2+1]
	blank_cube[0:nf,sci_u_centre-nu/2:sci_u_centre+nu/2+1,sci_v_centre-nv/2:sci_v_centre+nv/2+1] = noise.reshape(blank_cube_subset.shape) * blank_cube[0].size**0.5
	axes_tuple = (1,2)
	wnim=numpy.fft.ifftshift(blank_cube+0j, axes=axes_tuple)
	wnim=numpy.fft.ifftn(wnim, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	wnim=numpy.fft.fftshift(wnim, axes=axes_tuple)

	print 'Image domain equivalent noise level:', wnim.std(), 'mK'
	print '21-cm simulation rms:', np.load('/users/psims/EoR/EoR_simulations/21cmFAST_512MPc_512pix_128pix/Fits/21cm_z10d2_mK.npz')['arr_0'].std(), 'mK'
	print '1/(S/N) level:', wnim.std() / np.load('/users/psims/EoR/EoR_simulations/21cmFAST_512MPc_512pix_128pix/Fits/21cm_z10d2_mK.npz')['arr_0'].std()



calc_uv_domain_noise_in_Jy = False
if calc_uv_domain_noise_in_Jy:
	sigma_complex = sigma/2**0.5
	noise_real = np.random.normal(0,sigma_complex,nu*nv*nf)
	noise_imag = np.random.normal(0,sigma_complex,nu*nv*nf)
	noise = noise_real+1j*noise_imag

	blank_cube = np.zeros([38,512,512])
	sci_f, sci_v, sci_u = blank_cube.shape
	sci_v_centre = sci_v/2
	sci_u_centre = sci_u/2
	blank_cube_subset = blank_cube[0:nf,sci_u_centre-nu/2:sci_u_centre+nu/2+1,sci_v_centre-nv/2:sci_v_centre+nv/2+1]
	blank_cube[0:nf,sci_u_centre-nu/2:sci_u_centre+nu/2+1,sci_v_centre-nv/2:sci_v_centre+nv/2+1] = noise.reshape(blank_cube_subset.shape) * blank_cube[0].size**0.5

	sigma_per_vis_non_standard_units = sigma
	da_sr = ((12./511)*np.pi/180.)**2.
	conversion_from_non_standard_units_to_mK_sr_units = da_sr*blank_cube[0].size**0.5
	mK_to_Jy_per_sr_conversion = (2*(p.nu_min_MHz*1.e6)**2*astropy.constants.k_B.value)/astropy.constants.c.value**2. /1.e-26

	sigma_per_vis_Jy = sigma_per_vis_non_standard_units*conversion_from_non_standard_units_to_mK_sr_units*mK_to_Jy_per_sr_conversion
	my_uv_points_to_HERA_uv_points_ratio = 960./666.
	HERA_equivalent_sigma_per_vis_Jy = sigma_per_vis_Jy/my_uv_points_to_HERA_uv_points_ratio
	print 'HERA_equivalent_sigma_per_vis_Jy:', HERA_equivalent_sigma_per_vis_Jy, 'Jy'







#--------------------------------------------
# Define data vector
#--------------------------------------------
if use_EoR_cube:
	print 'Using EoR cube'
	d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector(sigma, s_EoR)[0]
	s_Tot = s_EoR.copy()
	if use_GDSE_foreground_cube:
		print 'Using GDSE cube'
		d += generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector(0.0, fg_GDSE)[0]
		s_Tot += fg_GDSE
		s_fgs_only = fg_GDSE.copy()
	if use_freefree_foreground_cube:
		print 'Using free-free cube'
		d += generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector(0.0, s_ff)[0]
		s_Tot += s_ff
		s_fgs_only += s_ff
	if use_EGS_cube:
		print 'Using EGS cube'
		d += generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector(0.0, s_EGS)[0]
		s_Tot += s_EGS
		s_fgs_only += s_EGS
elif use_GDSE_foreground_cube:
	d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector(sigma, fg_GDSE)[0]
	s_Tot = fg_GDSE.copy()
	s_fgs_only = fg_GDSE.copy()
	print 'Using GDSE cube'
	if use_freefree_foreground_cube:
		d += generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector(0.0, s_ff)[0]
		s_Tot += s_ff.copy()
		s_fgs_only += s_ff.copy()
		print 'Using free-free cube'
	if use_EGS_cube:
		d += generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector(0.0, s_EGS)[0]
		s_Tot += s_EGS.copy()
		s_fgs_only += s_EGS.copy()
		print 'Using EGS cube'
elif use_EGS_cube:
	d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector(sigma, s_EGS)[0]
	s_Tot = s_EGS.copy()
	s_fgs_only = s_EGS.copy()
	print 'Using EGS cube'
elif use_free_free_foreground_cube:
	d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector(sigma, s_ff)[0]
	s_Tot = s_ff.copy()
	s_fgs_only = s_ff.copy()
	print 'Using free-free cube'
else:
	d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector(sigma, s_EoR)[0]
	s_Tot = s_EoR.copy()
	print 'Using EoR cube'

#--------------------------------------------
# Load base matrices used in the likelihood and define related variables
#--------------------------------------------
T_Ninv_T = BM.read_data_from_hdf5(array_save_directory+'T_Ninv_T.h5', 'T_Ninv_T')
T = BM.read_data_from_hdf5(array_save_directory+'T.h5', 'T')
block_T_Ninv_T = BM.read_data_from_hdf5(array_save_directory+'block_T_Ninv_T.h5', 'block_T_Ninv_T')
Ninv = BM.read_data_from_hdf5(array_save_directory+'Ninv.h5', 'Ninv')
Ninv_d = np.dot(Ninv,d)
dbar = np.dot(T.conjugate().T,Ninv_d)
Ninv=[]
Sigma_Diag_Indices=np.diag_indices(shape(T_Ninv_T)[0])
Npar = shape(T_Ninv_T)[0]
nDims = len(k_cube_voxels_in_bin)
x=[100.e0]*nDims
nuv = (nu*nv-1)
block_T_Ninv_T = np.array([np.hsplit(block,nuv) for block in np.vsplit(T_Ninv_T,nuv)])



masked_power_spectral_modes = np.ones(Npar)
masked_power_spectral_modes[sorted(np.hstack(k_cube_voxels_in_bin)[0])] = 0.0
masked_power_spectral_modes = masked_power_spectral_modes.astype('bool')

# if nq==0:
# 	masked_power_spectral_modes = np.logical_not(np.logical_or.reduce(bin_selector_in_model_mask_vis_ordered).reshape(-1,neta)).flatten()
# else:
# 	masked_power_spectral_modes = LW_modes_only_boolean_array_vis_ordered
from numpy import real

#--------------------------------------------
# Instantiate class and check that posterior_probability returns a finite probability (so no obvious binning errors etc.)
#--------------------------------------------
if small_cube: PSPP = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list)
PSPP_block_diag = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, block_T_Ninv_T=block_T_Ninv_T, Print=True)
self = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, block_T_Ninv_T=block_T_Ninv_T, Print=True)
start = time.time()

if small_cube:
	# print PSPP.posterior_probability([1.e0]*nDims, diagonal_sigma=False)[0]
	print PSPP_block_diag.posterior_probability([1.e0]*nDims, diagonal_sigma=False, block_T_Ninv_T=block_T_Ninv_T)[0]

if sub_MLLWM:
	pre_sub_dbar = PSPP_block_diag.dbar
	maxL_LW_fit = PSPP_block_diag.calc_SigmaI_dbar_wrapper([1.e-20]*nDims, T_Ninv_T, pre_sub_dbar, block_T_Ninv_T=block_T_Ninv_T)[0]
	maxL_LW_signal = np.dot(T,maxL_LW_fit)
	Ninv = BM.read_data_from_hdf5(array_save_directory+'Ninv.h5', 'Ninv')
	Ninv_maxL_LW_signal = np.dot(Ninv,maxL_LW_signal)
	ML_qbar = np.dot(T.conjugate().T,Ninv_maxL_LW_signal)
	q_sub_dbar = pre_sub_dbar-ML_qbar
	if small_cube: PSPP.dbar = q_sub_dbar
	PSPP_block_diag.dbar = q_sub_dbar
	if small_cube:
		print PSPP.posterior_probability([1.e0]*nDims, diagonal_sigma=False)[0]
		print PSPP_block_diag.posterior_probability([1.e0]*nDims, diagonal_sigma=False, block_T_Ninv_T=block_T_Ninv_T)[0]

if sub_ML_monopole_term_model:
	pre_sub_dbar = PSPP_block_diag.dbar
	PSPP_block_diag.inverse_LW_power_first_LW_term=1.e20 #Don't fit for the first LW term (since only fitting for the monopole)
	PSPP_block_diag.inverse_LW_power_second_LW_term=1.e20  #Don't fit for the second LW term (since only fitting for the monopole)
	maxL_LW_fit = PSPP_block_diag.calc_SigmaI_dbar_wrapper([1.e-20]*nDims, T_Ninv_T, pre_sub_dbar, block_T_Ninv_T=block_T_Ninv_T)[0]
	maxL_LW_signal = np.dot(T,maxL_LW_fit)
	Ninv = BM.read_data_from_hdf5(array_save_directory+'Ninv.h5', 'Ninv')
	Ninv_maxL_LW_signal = np.dot(Ninv,maxL_LW_signal)
	ML_qbar = np.dot(T.conjugate().T,Ninv_maxL_LW_signal)
	q_sub_dbar = pre_sub_dbar-ML_qbar
	if small_cube: PSPP.dbar = q_sub_dbar
	PSPP_block_diag.dbar = q_sub_dbar
	PSPP_block_diag.inverse_LW_power=0.0 #Remove the constraints on the LW model for subsequent parts of the analysis
	if small_cube:
		print PSPP.posterior_probability([1.e0]*nDims, diagonal_sigma=False)[0]
		print PSPP_block_diag.posterior_probability([1.e0]*nDims, diagonal_sigma=False, block_T_Ninv_T=block_T_Ninv_T)[0]

if sub_ML_monopole_plus_first_LW_term_model:
	pre_sub_dbar = PSPP_block_diag.dbar
	PSPP_block_diag.inverse_LW_power_second_LW_term=1.e20  #Don't fit for the second LW term (since only fitting for the monopole and first LW term)
	maxL_LW_fit = PSPP_block_diag.calc_SigmaI_dbar_wrapper([1.e-20]*nDims, T_Ninv_T, pre_sub_dbar, block_T_Ninv_T=block_T_Ninv_T)[0]
	maxL_LW_signal = np.dot(T,maxL_LW_fit)
	Ninv = BM.read_data_from_hdf5(array_save_directory+'Ninv.h5', 'Ninv')
	Ninv_maxL_LW_signal = np.dot(Ninv,maxL_LW_signal)
	ML_qbar = np.dot(T.conjugate().T,Ninv_maxL_LW_signal)
	q_sub_dbar = pre_sub_dbar-ML_qbar
	if small_cube: PSPP.dbar = q_sub_dbar
	PSPP_block_diag.dbar = q_sub_dbar
	PSPP_block_diag.inverse_LW_power=0.0 #Remove the constraints on the LW model for subsequent parts of the analysis
	if small_cube:
		print PSPP.posterior_probability([1.e0]*nDims, diagonal_sigma=False)[0]
		print PSPP_block_diag.posterior_probability([1.e0]*nDims, diagonal_sigma=False, block_T_Ninv_T=block_T_Ninv_T)[0]

print 'Time taken: %f'%(time.time()-start)

start = time.time()
print PSPP_block_diag.posterior_probability([1.e0]*nDims, diagonal_sigma=False, block_T_Ninv_T=block_T_Ninv_T)[0]
print 'Time taken: %f'%(time.time()-start)

start = time.time()
print PSPP_block_diag.posterior_probability([1.e0]*nDims)[0]
print 'Time taken: %f'%(time.time()-start)

#--------------------------------------------
# Plot the total, LW component, and Fourier components of the signal vs. their maximum likelihood fitted equivalents, along with the fit residuals. Save the plots to file.
#--------------------------------------------
base_dir = 'Plots'
save_dir = base_dir+'/Likelihood_v1d76_3D_ZM/'
if not os.path.isdir(save_dir):
		os.makedirs(save_dir)

Show=False
if not use_EoR_cube:
	maxL_k_cube_signal = PSPP_block_diag.calc_SigmaI_dbar_wrapper(np.array(k_sigma)**2., T_Ninv_T, dbar, block_T_Ninv_T=block_T_Ninv_T)[0]
else:
	maxL_k_cube_signal = PSPP_block_diag.calc_SigmaI_dbar_wrapper(np.array([100.0]*nDims)**2, T_Ninv_T, dbar, block_T_Ninv_T=block_T_Ninv_T)[0]

maxL_f_plus_q_signal = np.dot(T,maxL_k_cube_signal)
save_path = save_dir+'Total_signal_model_fit_and_residuals_nu_{}_nv_{}_neta_{}_nq_{}_npl_{}_sigma_{:.1E}_beta_{}.png'.format(nu,nv,neta,nq,npl,sigma,p.beta)
if use_EoR_cube:save_path=save_path.replace('.png', '_EoR_cube.png')
plot_signal_vs_MLsignal_residuals(s_Tot, maxL_f_plus_q_signal, sigma, save_path)

LW_modes_only_boolean_array_vis_ordered_v2 = np.zeros(maxL_f_plus_q_signal.size).astype('bool')
LW_modes_only_boolean_array_vis_ordered_v2[nf-1::nf] = 1
LW_modes_only_boolean_array_vis_ordered_v2[nf-2::nf] = 1
LW_modes_only_boolean_array_vis_ordered_v2[nf/2-1::nf] = 1

if small_cube and not use_EoR_cube and nq==2:
	maxL_k_cube_LW_modes = maxL_k_cube_signal.copy()
	# maxL_k_cube_LW_modes[np.logical_not(LW_modes_only_boolean_array_vis_ordered)] = 0.0
	maxL_k_cube_LW_modes[np.logical_not(LW_modes_only_boolean_array_vis_ordered_v2)] = 0.0
	q_component_of_maxL_of_f_plus_q_signal = np.dot(T,maxL_k_cube_LW_modes)
	plot_signal_vs_MLsignal_residuals(s_fgs_only, q_component_of_maxL_of_f_plus_q_signal, sigma, save_dir+'LW_component_of_signal_model_fit_and_residuals_nu_{}_nv_{}_neta_{}_nq_{}_npl_{}_sigma_{:.1E}_beta_{}.png'.format(nu,nv,neta,nq,npl,sigma,p.beta))

	maxL_k_cube_fourier_modes = maxL_k_cube_signal.copy()
	# maxL_k_cube_fourier_modes[LW_modes_only_boolean_array_vis_ordered] = 0.0
	maxL_k_cube_fourier_modes[LW_modes_only_boolean_array_vis_ordered_v2] = 0.0
	f_component_of_maxL_of_f_plus_q_signal = np.dot(T,maxL_k_cube_fourier_modes)
	plot_signal_vs_MLsignal_residuals(s_EoR, f_component_of_maxL_of_f_plus_q_signal, sigma, save_dir+'Fourier_component_of_signal_model_fit_and_residuals_nu_{}_nv_{}_neta_{}_nq_{}_npl_{}_sigma_{:.1E}_beta_{}.png'.format(nu,nv,neta,nq,npl,sigma,p.beta))

No_large_spectral_scale_model_fit = False
if No_large_spectral_scale_model_fit:
	PSPP_block_diag.inverse_LW_power=1.e10
	maxL_k_cube_signal = PSPP_block_diag.calc_SigmaI_dbar_wrapper(np.array([100.0]*nDims)**2, T_Ninv_T, dbar, block_T_Ninv_T=block_T_Ninv_T)[0]

	maxL_f_plus_q_signal = np.dot(T,maxL_k_cube_signal)
	save_path = save_dir+'Total_signal_model_fit_and_residuals_NQ.png'
	if use_EoR_cube:save_path=save_path.replace('.png', '_EoR_cube.png')
	plot_signal_vs_MLsignal_residuals(s-maxL_LW_signal, maxL_f_plus_q_signal, sigma, save_path)

	if small_cube and not use_EoR_cube:
		maxL_k_cube_fourier_modes = maxL_k_cube_signal.copy()
		maxL_k_cube_fourier_modes[LW_modes_only_boolean_array_vis_ordered] = 0.0
		f_component_of_maxL_of_f_plus_q_signal = np.dot(T,maxL_k_cube_fourier_modes)
		plot_signal_vs_MLsignal_residuals(s_fourier_only, f_component_of_maxL_of_f_plus_q_signal, sigma, save_dir+'Fourier_component_of_signal_model_fit_and_residuals_NQ.png')

#--------------------------------------------
# Sample from the posterior
#--------------------------------------------###
# PolyChord setup
log_priors_min_max = [[-5.0, 4.0] for _ in range(nDims)]
# log_priors_min_max = [[-10.1, -10.0] for _ in range(nDims)]
# log_priors_min_max[0][1] = 4.0
prior_c = PriorC(log_priors_min_max)
nDerived = 0
nDims = len(k_cube_voxels_in_bin)
# base_dir = 'chains/'
outputfiles_base_dir = 'chains/'
# outputfiles_base_dir = 'chains/beta_ev_GDSE_only/'
base_dir = outputfiles_base_dir+'clusters/'
if not os.path.isdir(base_dir):
		os.makedirs(base_dir)

log_priors = True
dimensionless_PS = True
zero_the_LW_modes = False

file_root = 'Test_mini-{}_{}_{}_{}_{}_sigma_{:.1E}-lp_F-dPS_F-v1-'.format(nu,nv,neta,nq,npl,sigma).replace('.','d')
# file_root = 'Test_mini-nu_{}_nv_{}_neta_{}_nq_{}_npl_{}_sigma_{:.1E}-lp_F-dPS_F-v1-'.format(nu,nv,neta,nq,npl,sigma).replace('.','d')
# file_root = 'Test_mini-sigma_{:.1E}-lp_F-dPS_F-v1-'.format(sigma).replace('.','d')
if chan_selection!='':file_root=chan_selection+file_root
if log_priors:
	file_root=file_root.replace('lp_F', 'lp_T')
if dimensionless_PS:
	file_root=file_root.replace('dPS_F', 'dPS_T')
if nq==0:
	file_root=file_root.replace('mini-', 'mini-NQ-')
elif zero_the_LW_modes:
	file_root=file_root.replace('mini-', 'mini-ZLWM-')
if use_EoR_cube:
	file_root=file_root.replace('Test_mini', 'EoR_mini')
if use_MultiNest:
	file_root='MN-'+file_root
if npl>0:
	file_root=file_root.replace('-v1', '-beta_{:.2E}-v1'.format(p.beta))

file_root = generate_output_file_base(file_root, version_number='1')
print 'Output file_root = ', file_root

PSPP_block_diag_Polychord = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, block_T_Ninv_T=block_T_Ninv_T, log_priors=log_priors, dimensionless_PS=dimensionless_PS, Print=True)
if zero_the_LW_modes: PSPP_block_diag_Polychord.inverse_LW_power=1.e20
if sub_MLLWM: PSPP_block_diag_Polychord.dbar = q_sub_dbar
if sub_ML_monopole_term_model: PSPP_block_diag_Polychord.dbar = q_sub_dbar
if sub_ML_monopole_plus_first_LW_term_model: PSPP_block_diag_Polychord.dbar = q_sub_dbar

start = time.time()
PSPP_block_diag_Polychord.Print=False
Nit=20
for _ in range(Nit):
	L =  PSPP_block_diag_Polychord.posterior_probability([1.e0]*nDims)[0]
PSPP_block_diag_Polychord.Print=False
print 'Average evaluation time: %f'%((time.time()-start)/float(Nit))

def likelihood(theta, calc_likelihood=PSPP_block_diag_Polychord.posterior_probability):
	return calc_likelihood(theta)

def MultiNest_likelihood(theta, calc_likelihood=PSPP_block_diag_Polychord.posterior_probability):
	return calc_likelihood(theta)[0]

if use_MultiNest:
	MN_nlive = nDims*25
	# Run MultiNest
	result = solve(LogLikelihood=MultiNest_likelihood, Prior=prior_c.prior_func, n_dims=nDims, outputfiles_basename=outputfiles_base_dir+file_root, n_live_points=MN_nlive)
else:
	precision_criterion = 0.05
	#precision_criterion = 0.001 #PolyChord default value
	nlive=nDims*10
	#nlive=nDims*25 #PolyChord default value
	# Run PolyChord
	PolyChord.mpi_notification()
	PolyChord.run_nested_sampling(PSPP_block_diag_Polychord.posterior_probability, nDims, nDerived, file_root=file_root, read_resume=False, prior=prior_c.prior_func, precision_criterion=precision_criterion, nlive=nlive)

print 'Sampling complete!'
#######################




# PowerI = PSPP_block_diag_Polychord.calc_PowerI(x)
# PhiI=PowerI
# Sigma_block_diagonals = PSPP_block_diag_Polychord.calc_Sigma_block_diagonals(T_Ninv_T, PhiI)
# dbar_blocks = np.split(dbar, PSPP_block_diag_Polychord.nuv)

# if p.useGPU:
# 	SigmaI_dbar_blocks_and_logdet_Sigma = np.array([PSPP_block_diag_Polychord.calc_SigmaI_dbar(Sigma_block_diagonals[i_block], dbar_blocks[i_block])  for i_block in range(PSPP_block_diag_Polychord.nuv)])
# 	SigmaI_dbar_blocks = SigmaI_dbar_blocks_and_logdet_Sigma[:,0]
# 	logdet_Sigma_blocks = SigmaI_dbar_blocks_and_logdet_Sigma[:,1]
# 	print np.sum(logdet_Sigma_blocks)
# else:
# 	logdet_Sigma_blocks = np.array([PSPP_block_diag_Polychord.calc_SigmaI_dbar(Sigma_block_diagonals[i_block], dbar_blocks[i_block])  for i_block in range(PSPP_block_diag_Polychord.nuv)])
# 	print np.sum([np.linalg.slogdet(Sigma_block)[1] for Sigma_block in Sigma_block_diagonals])











