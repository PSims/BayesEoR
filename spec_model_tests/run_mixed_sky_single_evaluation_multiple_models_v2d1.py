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
npl = p.npl
print 'p.beta', p.beta


print 'args.beta', args.beta
print 'p.beta', p.beta
print 'args.nq', args.nq

nq = int(args.nq)
# nq = 2 #Overwrite PCLA selection
#npl = 0 #Overwrites quadratic term when nq=2, otherwise unused.
print 'nq', nq
print 'npl', npl
sub_ML_monopole_term_model = True #Improve numerical precision. Can be used for improving numerical precision when performing evidence comparison.
sub_ML_monopole_plus_first_LW_term_model = False #Improve numerical precision. Can be used for improving numerical precision when performing evidence comparison.
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
S18a_163_sigma = 100.e-1
sigma=np.round(S18a_163_sigma * (225./163)**-2.6, 1)
# sigma=100.e-1
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
# array_save_directory = 'array_storage/{}_nu_{}_nv_{}_neta_{}_nq_{}_npl_{}_sigma_{:.1E}/'.format(current_file_version,nu,nv,neta,nq,npl,sigma).replace('.','d')
array_save_directory = 'array_storage/batch_1/{}_nu_{}_nv_{}_neta_{}_nq_{}_npl_{}_sigma_{:.1E}/'.format(current_file_version,nu,nv,neta,nq,npl,sigma).replace('.','d')
# array_save_directory = 'array_storage/batch_2/{}_nu_{}_nv_{}_neta_{}_nq_{}_npl_{}_sigma_{:.1E}/'.format(current_file_version,nu,nv,neta,nq,npl,sigma).replace('.','d')
if npl==1:
	array_save_directory_pl=array_save_directory.replace('_sigma', '_beta_{:.2E}_sigma'.format(p.beta))
if npl==2:
	array_save_directory_pl=array_save_directory.replace('_sigma', '_b1_{:.2E}_b2_{:.2E}_sigma'.format(p.beta[0], p.beta[1]))



#--------------------------------------------
# Construct matrices
#--------------------------------------------
BM = BuildMatrices(array_save_directory_pl, nu, nv, nx, ny, neta, nf, nq, sigma, npl=npl)
overwrite_existing_matrix_stack = False #Can be set to False unless npl>0
# overwrite_existing_matrix_stack = True #Can be set to False unless npl>0
proceed_without_overwrite_confirmation = False #Allows overwrite_existing_matrix_stack to be run without having to manually accept the deletion of the old matrix stack
BM.build_minimum_sufficient_matrix_stack(overwrite_existing_matrix_stack=overwrite_existing_matrix_stack, proceed_without_overwrite_confirmation=proceed_without_overwrite_confirmation)



update_matrices_to_reflect_scaled_sigma = True
if update_matrices_to_reflect_scaled_sigma:
	RMSN = RenormaliseMatricesForScaledNoise()
	updated_sigma = RMSN.calc_updated_225_sigma(S18a_163_sigma)
	RMSN.calc_matrix_renormalisation_scale_factor(sigma, updated_sigma)
	sigma = updated_sigma






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

	# Read in Haslam extrapolated to 225 MHZ HERA strip values
	jupyter_dir = '/users/psims/EoR/Python_Scripts/BayesEoR/git_version/BayesEoR/likelihood_tests/SimpleEoRtestWQ/Jupyter/'
	Cartesian_Map_Projection_Array_GSM_225_HERA_strip, mean_array, std_array, SI_vals, hour_angle = np.load(jupyter_dir+'Data/HERA_strip_stats_v1d0_data.npy')


	###
	#Calculate x-coords pixel numbers of selected field centers
	###
	field_nan_array = np.ones(len(std_array))
	#Add a 13.0/2 deg buffer region around the excluded region so the none of the field enters the excluded region
	for i in range(len(std_array)):
	    if np.isnan(std_array[i]):
	        field_nan_array[i-40:i] = np.nan
	    if np.isnan(std_array[::-1][i]):
	        field_nan_array[::-1][i-40:i] = np.nan


	excluded_values_mask = np.logical_not(np.isnan(field_nan_array[10::78]))

	hour_angle_HERA_fields_subset = hour_angle[10::78] #Define a new field every 10 degrees starting at an hour angle of ~13.75 (which leads to a good spread of SI and stds to test).
	std_array_HERA_fields_subset = std_array[10::78]
	SI_vals_HERA_fields_subset = SI_vals[10::78] 

	hour_angle_HERA_fields_subset = hour_angle_HERA_fields_subset[excluded_values_mask] #Don't consider fields in the excluded region (where the field enters within +- 20 deg of the Galactic plane)
	std_array_HERA_fields_subset = std_array_HERA_fields_subset[excluded_values_mask] 
	SI_vals_HERA_fields_subset = SI_vals_HERA_fields_subset[excluded_values_mask] 


	# use_max_std=True
	use_max_std=False
	use_min_std=True
	# use_min_std=False

	if use_max_std:
		std_field_index = std_array_HERA_fields_subset.argmax()
	if use_min_std:
		std_field_index = std_array_HERA_fields_subset.argmin()


	# p.beta_experimental_mean = 2.63
	# p.Tb_experimental_std_K = 66.1866884116033/6.
	# p.beta_experimental_mean = 2.55
	# p.Tb_experimental_std_K = 66.1866884116033/1.
	p.beta_experimental_mean = -1.0*SI_vals_HERA_fields_subset[std_field_index]
	p.Tb_experimental_std_K = std_array_HERA_fields_subset[std_field_index]

	print 'p.beta_experimental_mean', p.beta_experimental_mean
	print 'p.Tb_experimental_std_K', p.Tb_experimental_std_K

	p.nu_min_MHz = 225.0-4.0
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
use_EoR_cube = False
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





b1_b2_logL_noise_i_array = []
for noise_i in range(30):

	print 'noise_i', noise_i
	#--------------------------------------------
	# Define data vector
	#--------------------------------------------
	if use_EoR_cube:
		print 'Using EoR cube'
		d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(sigma, s_EoR, nu,nv,nx,ny,nf,neta,nq)[0]
		s_Tot = s_EoR.copy()
		if use_GDSE_foreground_cube:
			print 'Using GDSE cube'
			d += generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(0.0, fg_GDSE, nu,nv,nx,ny,nf,neta,nq)[0]
			s_Tot += fg_GDSE
			s_fgs_only = fg_GDSE.copy()
		if use_freefree_foreground_cube:
			print 'Using free-free cube'
			d += generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(0.0, s_ff, nu,nv,nx,ny,nf,neta,nq)[0]
			s_Tot += s_ff
			s_fgs_only += s_ff
		if use_EGS_cube:
			print 'Using EGS cube'
			d += generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(0.0, s_EGS, nu,nv,nx,ny,nf,neta,nq)[0]
			s_Tot += s_EGS
			s_fgs_only += s_EGS
	elif use_GDSE_foreground_cube:
		use_random_seed = False
		if use_random_seed:
			random_seed = 123456
			d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(sigma, fg_GDSE, nu,nv,nx,ny,nf,neta,nq, random_seed=random_seed)[0]
		else:
			d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(sigma, fg_GDSE, nu,nv,nx,ny,nf,neta,nq, random_seed='')[0]
		s_Tot = fg_GDSE.copy()
		s_fgs_only = fg_GDSE.copy()
		print 'Using GDSE cube'
		if use_freefree_foreground_cube:
			d += generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(0.0, s_ff, nu,nv,nx,ny,nf,neta,nq)[0]
			s_Tot += s_ff.copy()
			s_fgs_only += s_ff.copy()
			print 'Using free-free cube'
		if use_EGS_cube:
			d += generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(0.0, s_EGS, nu,nv,nx,ny,nf,neta,nq)[0]
			s_Tot += s_EGS.copy()
			s_fgs_only += s_EGS.copy()
			print 'Using EGS cube'
	elif use_EGS_cube:
		d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(sigma, s_EGS, nu,nv,nx,ny,nf,neta,nq)[0]
		s_Tot = s_EGS.copy()
		s_fgs_only = s_EGS.copy()
		print 'Using EGS cube'
	elif use_free_free_foreground_cube:
		d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(sigma, s_ff, nu,nv,nx,ny,nf,neta,nq)[0]
		s_Tot = s_ff.copy()
		s_fgs_only = s_ff.copy()
		print 'Using free-free cube'
	else:
		d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(sigma, s_EoR, nu,nv,nx,ny,nf,neta,nq)[0]
		s_Tot = s_EoR.copy()
		print 'Using EoR cube'




	###
	# Test to see how much noise is being fitted out by zeroing the signal component and fitting to the noise.
	###
	# do_noise_fit_test = True
	do_noise_fit_test = False
	if do_noise_fit_test:
		d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(sigma, fg_GDSE*0.0, nu,nv,nx,ny,nf,neta,nq)[0]
		# d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(0.0, s_EoR, nu,nv,nx,ny,nf,neta,nq)[0]


	# replace_signal_with_coloured_noise_cube=True
	replace_signal_with_coloured_noise_cube=False
	if replace_signal_with_coloured_noise_cube:
		np.random.seed(12345)
		random_im=np.random.normal(0.,100,nf*nu*nv).reshape(nf,nu,nv)
		axes_tuple = (0,1,2)
		random_k=numpy.fft.ifftshift(random_im+0j, axes=axes_tuple)
		random_k=numpy.fft.fftn(random_k, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
		random_k=numpy.fft.fftshift(random_k, axes=axes_tuple)
		# random_k = random_k/random_k.std()
		sc_z, sc_y, sc_x = np.mgrid[-nf/2:nf/2,-nv/2+1:nv/2+1,-nu/2+1:nu/2+1]
		sc_r = ((sc_x/10.)**2. + (sc_y/10.)**2. + sc_z**2.)**0.5
		scale_cube = 1. / (sc_r+sc_r.max()/1.)
		scale_cube /= scale_cube.std()
		coloured_random_k = random_k*scale_cube
		# coloured_random_k[nf/4:3*nf/4] = 0.0
		coloured_random_k[nf/2-1:nf/2+2] = 0.0
		red_noise_im=numpy.fft.ifftshift(coloured_random_k+0j, axes=axes_tuple)
		red_noise_im=numpy.fft.ifftn(red_noise_im, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
		red_noise_im=numpy.fft.fftshift(red_noise_im, axes=axes_tuple)
		# red_noise_im /= red_noise_im.std()
		pylab.close('all')
		pylab.plot(red_noise_im[:,0,1])
		pylab.show()

		# add_LLSP=True
		add_LLSP=False		
		if add_LLSP:
			np.random.seed(12345)
			random_im_1d = np.random.normal(0,1,nf*2)
			random_k_1d=numpy.fft.ifftshift(random_im_1d+0j, axes=(0,))
			random_k_1d=numpy.fft.fftn(random_k_1d, axes=(0,)) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
			random_k_1d=numpy.fft.fftshift(random_k_1d, axes=(0,))
			random_k_1d[:nf-1] = 0.0
			random_k_1d[nf+2:] = 0.0
			ls_1d=numpy.fft.ifftshift(random_k_1d+0j, axes=(0,))
			ls_1d=numpy.fft.fftn(ls_1d, axes=(0,)) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
			ls_1d=numpy.fft.fftshift(ls_1d, axes=(0,))
			pylab.plot(ls_1d[::2])
			pylab.show()
			red_noise_im_LSSP = (red_noise_im.T-ls_1d[::2]*red_noise_im[:,0,1].std()/ls_1d[::2].std()).T
			pylab.plot(red_noise_im_LSSP[:,0,1])
			pylab.show()			



		print numpy.fft.fftshift(numpy.fft.fftn(np.fft.ifftshift(red_noise_im)))[:,0,1]
		print coloured_random_k[:,0,1]

		import numpy
		axes_tuple = (1,2)
		vfft1=numpy.fft.ifftshift(red_noise_im[0:38]-red_noise_im[0].mean()+0j, axes=axes_tuple)
		vfft1=numpy.fft.fftn(vfft1, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
		vfft1=numpy.fft.fftshift(vfft1, axes=axes_tuple)
		sci_f, sci_v, sci_u = vfft1.shape
		sci_v_centre = sci_v/2
		sci_u_centre = sci_u/2
		vfft1_subset = vfft1[0:nf,sci_u_centre-nu/2:sci_u_centre+nu/2+1,sci_v_centre-nv/2:sci_v_centre+nv/2+1]
		s_before_ZM = vfft1_subset.flatten()/vfft1[0].size**0.5
		ZM_vis_ordered_mask = np.ones(nu*nv*nf)
		ZM_vis_ordered_mask[nf*((nu*nv)/2):nf*((nu*nv)/2+1)]=0
		ZM_vis_ordered_mask = ZM_vis_ordered_mask.astype('bool')
		ZM_chan_ordered_mask = ZM_vis_ordered_mask.reshape(-1, neta+nq).T.flatten()
		s_coloured_random_k = s_before_ZM[ZM_chan_ordered_mask]
		d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(sigma, s_coloured_random_k, nu,nv,nx,ny,nf,neta,nq)[0]






	#--------------------------------------------
	# Load base matrices used in the likelihood and define related variables
	#--------------------------------------------
	T_Ninv_T = BM.read_data_from_hdf5(array_save_directory_pl+'T_Ninv_T.h5', 'T_Ninv_T')
	T = BM.read_data_from_hdf5(array_save_directory_pl+'T.h5', 'T')
	block_T_Ninv_T = BM.read_data_from_hdf5(array_save_directory_pl+'block_T_Ninv_T.h5', 'block_T_Ninv_T')
	Ninv = BM.read_data_from_hdf5(array_save_directory_pl+'Ninv.h5', 'Ninv')
	# 
	if update_matrices_to_reflect_scaled_sigma:
		T_Ninv_T, block_T_Ninv_T, Ninv = RMSN.renormalise_matrices_for_scaled_noise(T_Ninv_T, block_T_Ninv_T, Ninv)
	# 
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

	from numpy import real



	#--------------------------------------------
	# Instantiate class and check that posterior_probability returns a finite probability (so no obvious binning errors etc.)
	#--------------------------------------------
	if small_cube: PSPP = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list)
	PSPP_block_diag = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, block_T_Ninv_T=block_T_Ninv_T, Print=True)
	self = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, block_T_Ninv_T=block_T_Ninv_T, Print=True)
	start = time.time()



	PSPP_block_diag.posterior_probability([1.e-10]*nDims, diagonal_sigma=False, block_T_Ninv_T=block_T_Ninv_T)[0]





	nb = 23


	logL_dict = {}
	for b1 in np.linspace(0.0,5.5,nb):
		for b2 in np.linspace(0.0,5.5,nb):
			if b1 <= b2:
				logL_dict[tuple([b1, b2])] = np.nan
			else:
							
				# p.beta = [2.0, 2.1]
				# p.beta = [-1.0, -2.0]
				# p.beta = [2.63, 2.82]
				p.beta = [b1, b2]

				# p.beta = 2.8


				nq=2
				if p.beta:
					if type(p.beta)==list:
						npl = len(p.beta)
					else:
						npl=1
						nq=1
				else:
					npl=0



				current_file_version = 'Likelihood_v1d76_3D_ZM'
				# array_save_directory = 'array_storage/{}_nu_{}_nv_{}_neta_{}_nq_{}_npl_{}_sigma_{:.1E}/'.format(current_file_version,nu,nv,neta,nq,npl,sigma).replace('.','d')
				# array_save_directory = 'array_storage/batch_1/{}_nu_{}_nv_{}_neta_{}_nq_{}_npl_{}_sigma_{:.1E}/'.format(current_file_version,nu,nv,neta,nq,npl,sigma).replace('.','d')
				# array_save_directory = 'array_storage/batch_2/{}_nu_{}_nv_{}_neta_{}_nq_{}_npl_{}_sigma_{:.1E}/'.format(current_file_version,nu,nv,neta,nq,npl,sigma).replace('.','d')
				if npl==1:
					array_save_directory_pl=array_save_directory.replace('_sigma', '_beta_{:.2E}_sigma'.format(p.beta))
				if npl==2:
					array_save_directory_pl=array_save_directory.replace('_sigma', '_b1_{:.2E}_b2_{:.2E}_sigma'.format(p.beta[0], p.beta[1]))

				#--------------------------------------------
				# Construct matrices
				#--------------------------------------------
				BM = BuildMatrices(array_save_directory_pl, nu, nv, nx, ny, neta, nf, nq, sigma, npl=npl)
				overwrite_existing_matrix_stack = False #Can be set to False unless npl>0
				# overwrite_existing_matrix_stack = True #Can be set to False unless npl>0
				proceed_without_overwrite_confirmation = False #Allows overwrite_existing_matrix_stack to be run without having to manually accept the deletion of the old matrix stack
				BM.build_minimum_sufficient_matrix_stack(overwrite_existing_matrix_stack=overwrite_existing_matrix_stack, proceed_without_overwrite_confirmation=proceed_without_overwrite_confirmation)




				#--------------------------------------------
				# Load base matrices used in the likelihood and define related variables
				#--------------------------------------------
				T_Ninv_T = BM.read_data_from_hdf5(array_save_directory_pl+'T_Ninv_T.h5', 'T_Ninv_T')
				T = BM.read_data_from_hdf5(array_save_directory_pl+'T.h5', 'T')
				block_T_Ninv_T = BM.read_data_from_hdf5(array_save_directory_pl+'block_T_Ninv_T.h5', 'block_T_Ninv_T')
				Ninv = BM.read_data_from_hdf5(array_save_directory_pl+'Ninv.h5', 'Ninv')
				# 
				if update_matrices_to_reflect_scaled_sigma:
					T_Ninv_T, block_T_Ninv_T, Ninv = RMSN.renormalise_matrices_for_scaled_noise(T_Ninv_T, block_T_Ninv_T, Ninv)
				# 
				Ninv_d = np.dot(Ninv,d)
				dbar = np.dot(T.conjugate().T,Ninv_d)
				# Ninv=[]
				Sigma_Diag_Indices=np.diag_indices(shape(T_Ninv_T)[0])
				Npar = shape(T_Ninv_T)[0]
				nDims = len(k_cube_voxels_in_bin)
				x=[100.e0]*nDims
				nuv = (nu*nv-1)
				block_T_Ninv_T = np.array([np.hsplit(block,nuv) for block in np.vsplit(T_Ninv_T,nuv)])


				#--------------------------------------------
				# Instantiate class and check that posterior_probability returns a finite probability (so no obvious binning errors etc.)
				#--------------------------------------------
				if small_cube: PSPP = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list)
				PSPP_block_diag = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, block_T_Ninv_T=block_T_Ninv_T, Print=True)
				self = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, block_T_Ninv_T=block_T_Ninv_T, Print=True)
				start = time.time()



				# PSPP_block_diag.Print_debug = True
				PSPP_block_diag.posterior_probability([1.e-10]*nDims, diagonal_sigma=False, block_T_Ninv_T=block_T_Ninv_T)[0]


				pre_sub_dbar = PSPP_block_diag.dbar
				maxL_LW_fit = PSPP_block_diag.calc_SigmaI_dbar_wrapper([1.e-20]*nDims, T_Ninv_T, pre_sub_dbar, block_T_Ninv_T=block_T_Ninv_T)[0]
				maxL_LW_signal = np.dot(T,maxL_LW_fit)
				logL = -0.5*np.dot((d-maxL_LW_signal).conjugate().T, np.dot(Ninv, (d-maxL_LW_signal)))

				print 'logL', logL
				if replace_signal_with_coloured_noise_cube:
					print '(d-maxL_LW_signal).std()/d.std()', (d-maxL_LW_signal).std()/d.std()
					print '(d-maxL_LW_signal).var()/d.var()', (d-maxL_LW_signal).var()/d.var()

				logL_dict[tuple(p.beta)] = logL



	b1_b2_logL_array = np.zeros([len(logL_dict.keys()), 3])
	for i, (key, logL) in enumerate(logL_dict.iteritems()):
		b1_b2_logL_array[i,0] = key[0]
		b1_b2_logL_array[i,1] = key[1]
		b1_b2_logL_array[i,2] = logL.real

	Show=False
	Plot=False

	b1_b2_logL_array_sorted = b1_b2_logL_array[np.lexsort(np.roll(b1_b2_logL_array[:,0:2], 1, axis=1).T)]
	b1_b2_logL_array_sorted[:,2][nb::nb] = np.nan

	if Plot:
		pylab.close('all')
		pylab.imshow(b1_b2_logL_array_sorted[:,2].reshape(nb,nb))
		pylab.colorbar()
		if Show: pylab.show()


	maxL = b1_b2_logL_array_sorted[:,2][np.logical_not(np.isnan(b1_b2_logL_array_sorted[:,2]))].max()
	plot_extent = [b1_b2_logL_array_sorted[:,0][0],b1_b2_logL_array_sorted[:,0][-1],b1_b2_logL_array_sorted[:,1][-1],b1_b2_logL_array_sorted[:,1][0]]
	pylab.close('all')


	print '\n\nnoise iteration:', noise_i
	print '\n\n'

	if do_noise_fit_test:
		perfect_signal_fit_L = -d.shape[0]/2
		b1_b2_logL_array_sorted[:,2][b1_b2_logL_array_sorted[:,2]<-1.e10]=np.nan
		plot_extent = [b1_b2_logL_array_sorted[:,0][0],b1_b2_logL_array_sorted[:,0][-1],b1_b2_logL_array_sorted[:,1][-1],b1_b2_logL_array_sorted[:,1][0]]
		pylab.close('all')
		pylab.figure(figsize=(10,10))
		pylab.imshow(b1_b2_logL_array_sorted[:,2].reshape(nb,nb)-perfect_signal_fit_L, extent=plot_extent)
		pylab.colorbar(fraction=(0.315*0.15), pad=0.01)

		levels = [-1.0, -2.0, -3.0]
		CS = pylab.contour(b1_b2_logL_array_sorted[:,1].reshape(nb,nb), b1_b2_logL_array_sorted[:,0].reshape(nb,nb), (b1_b2_logL_array_sorted[:,2].reshape(nb,nb)-maxL), levels[::-1], alpha=1.0, colors='r')

		pylab.xlabel('Power law index, $b_{1}$', fontsize=16)
		pylab.ylabel('Power law index, $b_{2}$', fontsize=16)
		pylab.tick_params(axis='both', which='major', labelsize=16)
		fig_dir = 'Plots/model_evidence_b1_b2_v1d0/'
		pylab.savefig(fig_dir+'model_evidence_b1_b2_noise_sigma_{}_noise_i_{}.png'.format(sigma, noise_i))
		if Show: pylab.show()

		noise_bias_evidence_correction = b1_b2_logL_array_sorted[:,2]-perfect_signal_fit_L
		np.save('random/noise_bias_evidence_correction_{}'.format(noise_i), noise_bias_evidence_correction)



	else:
		if Plot:
			pylab.figure(figsize=(10,10))
			pylab.imshow(b1_b2_logL_array_sorted[:,2].reshape(nb,nb)-maxL, extent=plot_extent, vmin=-5.)
			pylab.colorbar(fraction=(0.315*0.15), pad=0.01)

			levels = [-1.0, -2.0, -3.0]
			CS = pylab.contour(b1_b2_logL_array_sorted[:,1].reshape(nb,nb), b1_b2_logL_array_sorted[:,0].reshape(nb,nb), (b1_b2_logL_array_sorted[:,2].reshape(nb,nb)-maxL), levels[::-1], alpha=1.0, colors='r')

			pylab.xlabel('Power law index, $b_{1}$', fontsize=16)
			pylab.ylabel('Power law index, $b_{2}$', fontsize=16)
			pylab.tick_params(axis='both', which='major', labelsize=16)
			fig_dir = 'Plots/model_evidence_b1_b2_v1d0/'
			# pylab.savefig(fig_dir+'model_evidence_b1_b2.png')
			pylab.savefig(fig_dir+'model_evidence_b1_b2_fg_std_{}_beta_{}_noise_i_{}.png'.format(p.Tb_experimental_std_K, p.beta_experimental_mean, noise_i))
			# pylab.savefig(fig_dir+'model_evidence_b1_b2_no_EGS_no_ff.png')
			if Show: pylab.show()


		# np.save('random/b1_b2_logL/b1_b2_logL_array_sigma_{}_{}'.format(sigma, noise_i), b1_b2_logL_array_sorted[:,2].reshape(nb,nb)-maxL)
		# b1_b2_logL_noise_i_array.append(b1_b2_logL_array_sorted[:,2].reshape(nb,nb)-maxL)
		b1_b2_logL_noise_i_array.append((b1_b2_logL_array_sorted, nb, maxL))



if use_max_std:
	max_min_dir = 'max_std'
if use_min_std:
	max_min_dir = 'min_std'

if use_random_seed:
	np.save('random/b1_b2_logL/{}/b1_b2_logL_array_noise_i_total_sigma_{}_nu_{}_nv_{}'.format(max_min_dir, sigma, p.nu, p.nv), (b1_b2_logL_noise_i_array, sigma, p.Tb_experimental_std_K, p.beta_experimental_mean))
else:
	np.save('random/b1_b2_logL/{}/b1_b2_logL_array_noise_i_total_sigma_{}_nu_{}_nv_{}_multiple_noise_realisations'.format(max_min_dir, sigma, p.nu, p.nv), (b1_b2_logL_noise_i_array, sigma, p.Tb_experimental_std_K, p.beta_experimental_mean))





# Continue = True
Continue = False
if Continue:

	import numpy as np
	import pylab

	use_max_std = False
	use_min_std = True

	if use_max_std:
		max_min_dir = 'max_std'
	if use_min_std:
		max_min_dir = 'min_std'

	# b1_b2_logL_array_noise_i_total, sigma, Tb_experimental_std_K, beta_experimental_mean = np.load('random/b1_b2_logL/{}/b1_b2_logL_array_noise_i_total_sigma_{}_nu_{}_nv_{}.npy'.format(max_min_dir, 7.0, 15, 15))
	# b1_b2_logL_array_noise_i_total, sigma, Tb_experimental_std_K, beta_experimental_mean = np.load('random/b1_b2_logL/{}/b1_b2_logL_array_noise_i_total_sigma_{}_nu_{}_nv_{}.npy'.format(max_min_dir, 21.0, 15, 15))
	# b1_b2_logL_array_noise_i_total, sigma, Tb_experimental_std_K, beta_experimental_mean = np.load('random/b1_b2_logL/{}/b1_b2_logL_array_noise_i_total_sigma_{}_nu_{}_nv_{}_multiple_noise_realisations.npy'.format(max_min_dir, 7.0, 15, 15))
	b1_b2_logL_array_noise_i_total, sigma, Tb_experimental_std_K, beta_experimental_mean = np.load('random/b1_b2_logL/{}/b1_b2_logL_array_noise_i_total_sigma_{}_nu_{}_nv_{}_multiple_noise_realisations.npy'.format(max_min_dir, 21.0, 15, 15))
	# b1_b2_logL_array_noise_i_total, sigma, Tb_experimental_std_K, beta_experimental_mean = np.load('random/b1_b2_logL/{}/b1_b2_logL_array_noise_i_total_sigma_{}_nu_{}_nv_{}.npy'.format(max_min_dir, np.round(100.e-1 * (225./163)**-2.6, 1), 15, 15))
	# b1_b2_logL_array_noise_i_total, sigma, Tb_experimental_std_K, beta_experimental_mean = np.load('random/b1_b2_logL/{}/b1_b2_logL_array_noise_i_total_sigma_{}_nu_{}_nv_{}.npy'.format(max_min_dir, np.round(300.e-1 * (225./163)**-2.6, 1), 15, 15))
	# b1_b2_logL_array_noise_i_total, sigma, Tb_experimental_std_K, beta_experimental_mean = np.load('random/b1_b2_logL/{}/b1_b2_logL_array_noise_i_total_sigma_{}.npy'.format(max_min_dir, np.round(100.e-1 * (225./163)**-2.6, 1)))
	# b1_b2_logL_array_noise_i_total, sigma, Tb_experimental_std_K, beta_experimental_mean = np.load('random/b1_b2_logL/{}/b1_b2_logL_array_noise_i_total_sigma_{}.npy'.format(max_min_dir, np.round(300.e-1 * (225./163)**-2.6, 1)))
	b1_b2_logL_array_noise_i_total = np.array(b1_b2_logL_array_noise_i_total)
	b1_b2_logL_array_sorted_arrays = b1_b2_logL_array_noise_i_total[:,0]
	nb = b1_b2_logL_array_noise_i_total[:,1]
	maxL = b1_b2_logL_array_noise_i_total[:,2]
	nb = nb[0]
	b1_b2_logL_array_sorted_arrays = np.array([x for x in b1_b2_logL_array_sorted_arrays])
	b1_b2_logL_array_sorted = b1_b2_logL_array_sorted_arrays[0]
	b1_b2_logL_noise_i_array = np.array([b1_b2_logL_array_sorted_arrays[i][:,2].reshape(nb,nb) - maxL[i] for i in range(len(b1_b2_logL_array_sorted_arrays))])



	b1_b2_logL_noise_i_array = np.array(b1_b2_logL_noise_i_array)

	b1_b2_logL_noise_i_array_uncertainties = np.std(b1_b2_logL_noise_i_array, axis=0)

	pylab.figure(figsize=(10,10))
	plot_extent = [b1_b2_logL_array_sorted[:,0][0],b1_b2_logL_array_sorted[:,0][-1],b1_b2_logL_array_sorted[:,1][-1],b1_b2_logL_array_sorted[:,1][0]]
	pylab.imshow(b1_b2_logL_noise_i_array_uncertainties, extent=plot_extent, vmax=10.0)
	pylab.colorbar(fraction=(0.315*0.15), pad=0.01)

	uncertainty_contours = [1.0, 2.0, 3.0]
	CS = pylab.contour(b1_b2_logL_array_sorted[:,1].reshape(nb,nb), b1_b2_logL_array_sorted[:,0].reshape(nb,nb), (b1_b2_logL_noise_i_array_uncertainties), uncertainty_contours, alpha=1.0, colors='r', linestyles='--')

	pylab.xlabel('Power law index, $b_{1}$', fontsize=16)
	pylab.ylabel('Power law index, $b_{2}$', fontsize=16)
	pylab.tick_params(axis='both', which='major', labelsize=16)
	fig_dir = 'Plots/model_evidence_b1_b2_v1d0/{}/'.format(max_min_dir)
	pylab.tight_layout()
	pylab.savefig(fig_dir+'model_evidence_b1_b2_sample_uncertainties_sigma_{}'.format(sigma).replace('.','d')+'.png')
	pylab.show()





	b1_b2_logL_noise_i_array_means = np.mean(b1_b2_logL_noise_i_array, axis=0)
	maxL_means = b1_b2_logL_noise_i_array_means[np.logical_not(np.isnan(b1_b2_logL_noise_i_array_means))].max()


	pylab.figure(figsize=(10,10))
	pylab.imshow(b1_b2_logL_noise_i_array_means-maxL_means, extent=plot_extent, vmin=-5.0)
	pylab.colorbar(fraction=(0.315*0.15), pad=0.01)

	levels = [-1.0, -2.0, -3.0]
	CS = pylab.contour(b1_b2_logL_array_sorted[:,1].reshape(nb,nb), b1_b2_logL_array_sorted[:,0].reshape(nb,nb), (b1_b2_logL_array_sorted[:,2].reshape(nb,nb)-maxL[0]), levels[::-1], alpha=1.0, colors='r')

	pylab.xlabel('Power law index, $b_{1}$', fontsize=16)
	pylab.ylabel('Power law index, $b_{2}$', fontsize=16)
	pylab.tick_params(axis='both', which='major', labelsize=16)
	fig_dir = 'Plots/model_evidence_b1_b2_v1d0/{}/'.format(max_min_dir)
	pylab.tight_layout()
	pylab.savefig(fig_dir+'model_evidence_b1_b2_means_sigma_{}'.format(sigma).replace('.','d')+'.png')
	pylab.show()



	b1_b2_logL_noise_i_array_error_on_means =  b1_b2_logL_noise_i_array_uncertainties / b1_b2_logL_noise_i_array.shape[0]**0.5

	pylab.figure(figsize=(10,10))
	pylab.imshow(b1_b2_logL_noise_i_array_error_on_means, extent=plot_extent, vmax=5.0)
	pylab.colorbar(fraction=(0.315*0.15), pad=0.01)

	levels = [-1.0, -2.0, -3.0]
	CS = pylab.contour(b1_b2_logL_array_sorted[:,1].reshape(nb,nb), b1_b2_logL_array_sorted[:,0].reshape(nb,nb), (b1_b2_logL_array_sorted[:,2].reshape(nb,nb)-maxL[0]), levels[::-1], alpha=1.0, colors='r')

	pylab.xlabel('Power law index, $b_{1}$', fontsize=16)
	pylab.ylabel('Power law index, $b_{2}$', fontsize=16)
	pylab.tick_params(axis='both', which='major', labelsize=16)
	fig_dir = 'Plots/model_evidence_b1_b2_v1d0/{}/'.format(max_min_dir)
	pylab.tight_layout()
	pylab.savefig(fig_dir+'model_evidence_b1_b2_uncertainties_on_means_sigma_{}'.format(sigma).replace('.','d')+'.png')
	pylab.show()




	pylab.figure(figsize=(10,10))
	pylab.imshow(b1_b2_logL_noise_i_array[0], extent=plot_extent, vmin=-5.0)
	pylab.colorbar(fraction=(0.315*0.15), pad=0.01)

	levels = [-1.0, -2.0, -3.0]
	CS = pylab.contour(b1_b2_logL_array_sorted[:,1].reshape(nb,nb), b1_b2_logL_array_sorted[:,0].reshape(nb,nb), (b1_b2_logL_array_sorted[:,2].reshape(nb,nb)-maxL[0]), levels[::-1], alpha=1.0, colors='r')

	pylab.xlabel('Power law index, $b_{1}$', fontsize=16)
	pylab.ylabel('Power law index, $b_{2}$', fontsize=16)
	pylab.tick_params(axis='both', which='major', labelsize=16)
	fig_dir = 'Plots/model_evidence_b1_b2_v1d0/{}/'.format(max_min_dir)
	pylab.tight_layout()
	# pylab.savefig(fig_dir+'model_evidence_b1_b2_uncertainties_on_means_sigma_{}'.format(sigma).replace('.','d')+'.png')
	# pylab.show()



	pylab.figure(figsize=(10,10))
	pylab.imshow(b1_b2_logL_noise_i_array[0]/b1_b2_logL_noise_i_array_uncertainties, extent=plot_extent, vmin=-1.0)
	pylab.colorbar(fraction=(0.315*0.15), pad=0.01)

	levels = [-1.0, -2.0, -3.0]
	CS = pylab.contour(b1_b2_logL_array_sorted[:,1].reshape(nb,nb), b1_b2_logL_array_sorted[:,0].reshape(nb,nb), (b1_b2_logL_array_sorted[:,2].reshape(nb,nb)-maxL[0]), levels[::-1], alpha=1.0, colors='r')

	pylab.xlabel('Power law index, $b_{1}$', fontsize=16)
	pylab.ylabel('Power law index, $b_{2}$', fontsize=16)
	pylab.tick_params(axis='both', which='major', labelsize=16)
	fig_dir = 'Plots/model_evidence_b1_b2_v1d0/{}/'.format(max_min_dir)
	pylab.tight_layout()
	# pylab.savefig(fig_dir+'model_evidence_b1_b2_uncertainties_on_means_sigma_{}'.format(sigma).replace('.','d')+'.png')
	pylab.show()



	# compare_realisations_and_uncertainties=True
	compare_realisations_and_uncertainties=False
	if compare_realisations_and_uncertainties:

		for i in range(7):
			pylab.figure(figsize=(10,10))
			pylab.imshow(b1_b2_logL_noise_i_array[i], extent=plot_extent, vmin=-5.0)
			pylab.colorbar(fraction=(0.315*0.15), pad=0.01)

			levels = [-1.0, -2.0, -3.0]
			CS = pylab.contour(b1_b2_logL_array_sorted[:,1].reshape(nb,nb), b1_b2_logL_array_sorted[:,0].reshape(nb,nb), (b1_b2_logL_array_sorted[:,2].reshape(nb,nb)-maxL[0]), levels[::-1], alpha=1.0, colors='r')

			pylab.xlabel('Power law index, $b_{1}$', fontsize=16)
			pylab.ylabel('Power law index, $b_{2}$', fontsize=16)
			pylab.tick_params(axis='both', which='major', labelsize=16)
			fig_dir = 'Plots/model_evidence_b1_b2_v1d0/{}/'.format(max_min_dir)
			# pylab.savefig(fig_dir+'model_evidence_b1_b2_uncertainties_on_means_sigma_{}'.format(sigma).replace('.','d')+'.png')
		pylab.tight_layout()
		pylab.show()


		pylab.figure(figsize=(10,10))
		pylab.imshow(b1_b2_logL_noise_i_array[0]/b1_b2_logL_noise_i_array_uncertainties, extent=plot_extent, vmin=-1.0)
		pylab.colorbar(fraction=(0.315*0.15), pad=0.01)

		levels = [-1.0, -2.0, -3.0]
		CS = pylab.contour(b1_b2_logL_array_sorted[:,1].reshape(nb,nb), b1_b2_logL_array_sorted[:,0].reshape(nb,nb), (b1_b2_logL_array_sorted[:,2].reshape(nb,nb)-maxL[0]), levels[::-1], alpha=1.0, colors='r')

		pylab.xlabel('Power law index, $b_{1}$', fontsize=16)
		pylab.ylabel('Power law index, $b_{2}$', fontsize=16)
		pylab.tick_params(axis='both', which='major', labelsize=16)
		fig_dir = 'Plots/model_evidence_b1_b2_v1d0/{}/'.format(max_min_dir)
		pylab.tight_layout()
		# pylab.savefig(fig_dir+'model_evidence_b1_b2_uncertainties_on_means_sigma_{}'.format(sigma).replace('.','d')+'.png')
		pylab.show()





	pylab.figure(figsize=(10,10))
	plot_extent = [b1_b2_logL_array_sorted[:,0][0],b1_b2_logL_array_sorted[:,0][-1],b1_b2_logL_array_sorted[:,1][-1],b1_b2_logL_array_sorted[:,1][0]]
	# if sigma==10.:
	# if sigma==4.3:
	if sigma==7.0:
		# pylab.imshow(b1_b2_logL_noise_i_array[4], extent=plot_extent, vmin=-5.0)
		pylab.imshow(b1_b2_logL_noise_i_array[0], extent=plot_extent, vmin=-5.0)
	# if sigma==30.:
	# if sigma==13.0:
	if sigma==21.0:
		# pylab.imshow(b1_b2_logL_noise_i_array[4], extent=plot_extent, vmin=-5.0)
		pylab.imshow(b1_b2_logL_noise_i_array[0], extent=plot_extent, vmin=-5.0)
	pylab.colorbar(fraction=(0.315*0.15), pad=0.01)

	levels = [-1.0, -2.0, -3.0]
	CS = pylab.contour(b1_b2_logL_array_sorted[:,1].reshape(nb,nb), b1_b2_logL_array_sorted[:,0].reshape(nb,nb), (b1_b2_logL_array_sorted[:,2].reshape(nb,nb)-maxL[0]), levels[::-1], alpha=1.0, colors='r')

	pylab.xlabel('Power law index, $b_{1}$', fontsize=16)
	pylab.ylabel('Power law index, $b_{2}$', fontsize=16)
	pylab.tick_params(axis='both', which='major', labelsize=16)
	fig_dir = 'Plots/model_evidence_b1_b2_v1d0/{}/'.format(max_min_dir)
	pylab.tight_layout()
	pylab.savefig(fig_dir+'model_evidence_b1_b2_single_realisation_fg_std_{}_beta_{}_sigma_{}'.format(np.round(Tb_experimental_std_K,0), beta_experimental_mean, sigma).replace('.','d')+'.png')
	pylab.show()






