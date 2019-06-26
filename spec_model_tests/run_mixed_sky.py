#!/users/psims/anaconda2/bin/python

#--------------------------------------------
# Imports
#--------------------------------------------
from subprocess import os
import sys
head,tail = os.path.split(os.path.split(os.getcwd())[0])
sys.path.append(head)
from BayesEoR import * #Make everything available for now, this can be refined later
import BayesEoR.Params.params as p
import time

mpi_rank = 0
# run_full_analysis = False #False skips mpi and other imports that can cause crashes in ipython (note: in ipython apparently __name__ == '__main__' which is why this if statement is here instead)
run_full_analysis = True #When running an analysis this should be True.
# if __name__ == '__main__':
if run_full_analysis:
	import mpi4py
	from mpi4py import MPI
	mpi_comm = MPI.COMM_WORLD
	mpi_rank = mpi_comm.Get_rank()
	mpi_size = mpi_comm.Get_size()
	print 'mpi_comm', mpi_comm
	print 'mpi_rank', mpi_rank
	print 'mpi_size', mpi_size
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
if args.beta:
	if type(args.beta)==str:
		if args.beta.count('[') and args.beta.count(']'):
			p.beta = map(float, args.beta.replace('[','').replace(']','').split(',')) #Overwrite parameter file beta with value chosen from the command line if it is included
			npl = len(p.beta) #Overwrites quadratic term when nq=2, otherwise unused.
		else:
			p.beta = float(args.beta) #Overwrite parameter file beta with value chosen from the command line if it is included 
			npl = 1
	elif type(args.beta)==list:
		p.beta = args.beta #Overwrite parameter file beta with value chosen from the command line if it is included 
		npl = len(p.beta)
	else:
		print 'No value for betas given, using defaults.'

print 'args.beta', args.beta
print 'p.beta', p.beta
print 'args.nq', args.nq

nq = int(args.nq)
if nq>npl:
	nq=npl
# nq = 2 #Overwrite PCLA selection
# npl = 0 #Overwrites quadratic term when nq=2, otherwise unused.
# nq=npl=p.nq=p.npl = 0

print 'nq', nq
print 'npl', npl
sub_ML_monopole_term_model = False #Improve numerical precision. Can be used for improving numerical precision when performing evidence comparison.
sub_ML_monopole_plus_first_LW_term_model = False #Improve numerical precision. Can be used for improving numerical precision when performing evidence comparison.
sub_MLLWM = False #Improve numerical precision. DO NOT USE WHEN PERFORMING EVIDENCE COMPARISON! Can only be used for parameter estimation not evidence comparison since subtracting different MLLWM (with different evidences) from the data when comparing different LWMs will alter the relative evidences of the subtracted models. In effect subtracting a higher evidence MLLWM1 reduces the evidence for the fit to the residuals with MLLWM1 relative to fitting low evidence MLLWM2 and refitting with MLLWM2. It is only correct to compare evidences when doing no qsub or when the qsub model is fixed such as with sub_ML_monopole_term_model.
# Cube size
nf=p.nf
neta=p.neta
if not p.include_instrumental_effects:
	neta=neta -nq
nu=p.nu
nv=p.nv
nx=p.nx
ny=p.ny
# Data noise
sigma=50.e-1

if p.include_instrumental_effects:
		average_baseline_redundancy = p.baseline_redundancy_array.mean() #Keep average noise level consisitent with the non-instrumental case by normalizing sigma by the average baseline redundancy before scaling individual baselines by their respective redundancies
		# sigma = sigma*average_baseline_redundancy**0.5 *250.0 #Noise level in S19b
		# sigma = sigma*average_baseline_redundancy**0.5 *500.0
		sigma = sigma*average_baseline_redundancy**0.5 *1000.0
		# sigma = sigma*average_baseline_redundancy**0.5 *1500.0
		# sigma = sigma*average_baseline_redundancy**0.5 *2000.0
else:
	sigma = sigma*1.

p.sigma = sigma

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
# array_save_directory = 'array_storage/batch_1/{}_nu_{}_nv_{}_neta_{}_nq_{}_npl_{}_sigma_{:.1E}/'.format(current_file_version,nu,nv,neta,nq,npl,sigma).replace('.','d')
array_save_directory = 'array_storage/FgSpecMOptimisation/{}_nu_{}_nv_{}_neta_{}_nq_{}_npl_{}_sigma_{:.1E}/'.format(current_file_version,nu,nv,neta,nq,npl,sigma).replace('.','d')
if p.include_instrumental_effects:
	instrument_info = filter(None, p.instrument_model_directory.split('/'))[-1]
	if 	p.model_drift_scan_primary_beam:
		instrument_info = instrument_info+'_dspb'
	array_save_directory = array_save_directory[:-1]+'_instrumental/'+instrument_info+'/'
	n_vis=p.n_vis
else:
	n_vis = 0
if npl==1:
	array_save_directory=array_save_directory.replace('_sigma', '_beta_{:.2E}_sigma'.format(p.beta))
if npl==2:
	array_save_directory=array_save_directory.replace('_sigma', '_b1_{:.2E}_b2_{:.2E}_sigma'.format(p.beta[0], p.beta[1]))

if p.fit_for_monopole:
	array_save_directory = array_save_directory[:-1]+'_fit_for_monopole_eq_True/'


#--------------------------------------------
# Construct matrices
#--------------------------------------------
BM = BuildMatrices(array_save_directory, nu, nv, nx, ny, n_vis, neta, nf, nq, sigma, npl=npl)
overwrite_existing_matrix_stack = False #Can be set to False unless npl>0
# overwrite_existing_matrix_stack = True #Can be set to False unless npl>0
proceed_without_overwrite_confirmation = False #Allows overwrite_existing_matrix_stack to be run without having to manually accept the deletion of the old matrix stack
BM.build_minimum_sufficient_matrix_stack(overwrite_existing_matrix_stack=overwrite_existing_matrix_stack, proceed_without_overwrite_confirmation=proceed_without_overwrite_confirmation)


#--------------------------------------------
# Define power spectral bins and coordinate cubes
#--------------------------------------------
mod_k, k_x, k_y, k_z, deltakperp, deltakpara, x, y, z = generate_k_cube_in_physical_coordinates_21cmFAST_v2d0(nu,nv,nx,ny,nf,neta,p.box_size_21cmFAST_pix_sc,p.box_size_21cmFAST_Mpc_sc)
k=mod_k.copy()
k_vis_ordered = k.T.flatten()
k_x_masked = generate_masked_coordinate_cubes(k_x, nu,nv,nx,ny,nf,neta,nq)
k_y_masked = generate_masked_coordinate_cubes(k_y, nu,nv,nx,ny,nf,neta,nq)
k_z_masked = generate_masked_coordinate_cubes(k_z, nu,nv,nx,ny,nf,neta,nq)
mod_k_masked = generate_masked_coordinate_cubes(mod_k, nu,nv,nx,ny,nf,neta,nq)

k_cube_voxels_in_bin, modkbins_containing_voxels = generate_k_cube_model_spherical_binning_v2d1(mod_k_masked, k_z_masked, nu,nv,nx,ny,nf,neta,nq)

if p.use_uniform_prior_on_min_k_bin:
	print 'Excluding min-kz bin...'
	k_cube_voxels_in_bin = k_cube_voxels_in_bin[1:]
	modkbins_containing_voxels = modkbins_containing_voxels[1:]

modk_vis_ordered_list = [mod_k_masked[k_cube_voxels_in_bin[i_bin]] for i_bin in range(len(k_cube_voxels_in_bin))]
k_vals_file_name = 'k_vals_nu_{}_nv_{}_nf_{}_nq_{}_binning_v2d1.txt'.format(nu,nv,nf,nq)
k_vals = calc_mean_binned_k_vals(mod_k_masked, k_cube_voxels_in_bin, save_k_vals=True, k_vals_file=k_vals_file_name)

do_cylindrical_binning = False
if do_cylindrical_binning:
	n_k_perp_bins=2
	k_cube_voxels_in_bin, modkbins_containing_voxels, k_perp_bins = generate_k_cube_model_cylindrical_binning(mod_k_masked, k_z_masked, k_y_masked, k_x_masked, n_k_perp_bins, nu,nv,nx,ny,nf,neta,nq)



#--------------------------------------------
# Non-instrumental data creation
#--------------------------------------------
if not p.include_instrumental_effects:
	#--------------------------------------------
	# Generate mock-GDSE data
	#--------------------------------------------
	# p.use_GDSE_foreground_cube = False
	p.use_GDSE_foreground_cube = True
	use_GDSE_foreground_cube = p.use_GDSE_foreground_cube
	if use_GDSE_foreground_cube:
		###
		# GDSE foreground_outputs
		###	

		p.Tb_experimental_std_K = 30.0

		foreground_outputs = generate_Jelic_cube(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show, p.beta_experimental_mean,p.beta_experimental_std,p.gamma_mean,p.gamma_sigma,p.Tb_experimental_mean_K,p.Tb_experimental_std_K,p.nu_min_MHz,p.channel_width_MHz, generate_additional_extrapolated_HF_foreground_cube=True, fits_storage_dir=p.fits_storage_dir, HF_nu_min_MHz_array=p.HF_nu_min_MHz_array, simulation_FoV_deg=p.simulation_FoV_deg, simulation_resolution_deg=p.simulation_resolution_deg,random_seed=314211)

		fg_GDSE, s_GDSE, Tb_nu, beta_experimental_mean,beta_experimental_std,gamma_mean,gamma_sigma,Tb_experimental_mean_K,Tb_experimental_std_K,nu_min_MHz,channel_width_MHz, HF_Tb_nu = foreground_outputs
		foreground_outputs = []

		plot_figure = False
		if plot_figure:
			construct_aplpy_image_from_fits('/gpfs/data/jpober/psims/EoR/Python_Scripts/BayesEoR/git_version/BayesEoR/spec_model_tests/fits_storage/multi_frequency_band_pythonPStest1/Jelic_nu_min_MHz_159d0_TbStd_66d1866884116_beta_2d63_dbeta0d02/', 'Jelic_GDSE_cube_159MHz_mK', run_convert_from_mK_to_K=True, run_remove_unused_header_variables=True)

	#--------------------------------------------
	# Generate mock-free-free data
	#--------------------------------------------
	p.use_freefree_foreground_cube = False
	# p.use_freefree_foreground_cube = True
	use_freefree_foreground_cube = p.use_freefree_foreground_cube
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
	p.use_EGS_cube = False
	# p.use_EGS_cube = True
	use_EGS_cube = p.use_EGS_cube
	if use_EGS_cube:
		print 'Using use_EGS_cube data'
		s_EGS, abc_EGS, scidata1_EGS = generate_data_from_loaded_EGS_cube(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show,p.EGS_npz_path)

	#--------------------------------------------
	# Load EoR data
	#--------------------------------------------
	# p.use_EoR_cube = True
	p.use_EoR_cube = False
	use_EoR_cube = p.use_EoR_cube
	if use_EoR_cube:
		print 'Using use_EoR_cube data'
		s_EoR, abc, scidata1 = generate_data_from_loaded_EoR_cube_v2d0(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show,chan_selection,p.EoR_npz_path_sc)

		plot_figure = False
		if plot_figure:
			construct_aplpy_image_from_fits('/users/psims/EoR/EoR_simulations/21cmFAST_2048MPc_2048pix_512pix_AstroParamExploration1/Fits/output_fits/nf0d888/', '21cm_mK_z7.600_nf0.888_useTs0.0_aveTb21.24_cube_side_pix512_cube_side_Mpc2048_mK', run_convert_from_mK_to_K=False, run_remove_unused_header_variables=True)
			

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
	non_instrmental_noise_seed = 42123
	if p.use_EoR_cube:
		print 'Using EoR cube'
		d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(sigma, s_EoR, nu,nv,nx,ny,nf,neta,nq, random_seed=non_instrmental_noise_seed)[0]
		s_Tot = s_EoR.copy()
		if p.use_GDSE_foreground_cube:
			print 'Using GDSE cube'
			d += generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(0.0, fg_GDSE, nu,nv,nx,ny,nf,neta,nq)[0]
			s_Tot += fg_GDSE
			s_fgs_only = fg_GDSE.copy()
		if p.use_freefree_foreground_cube:
			print 'Using free-free cube'
			d += generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(0.0, s_ff, nu,nv,nx,ny,nf,neta,nq)[0]
			s_Tot += s_ff
			s_fgs_only += s_ff
		if p.use_EGS_cube:
			print 'Using EGS cube'
			d += generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(0.0, s_EGS, nu,nv,nx,ny,nf,neta,nq)[0]
			s_Tot += s_EGS
			s_fgs_only += s_EGS
	elif p.use_GDSE_foreground_cube:
		d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(sigma, fg_GDSE, nu,nv,nx,ny,nf,neta,nq, random_seed=non_instrmental_noise_seed)[0]
		s_Tot = fg_GDSE.copy()
		s_fgs_only = fg_GDSE.copy()
		print 'Using GDSE cube'
		if p.use_freefree_foreground_cube:
			d += generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(0.0, s_ff, nu,nv,nx,ny,nf,neta,nq)[0]
			s_Tot += s_ff.copy()
			s_fgs_only += s_ff.copy()
			print 'Using free-free cube'
		if p.use_EGS_cube:
			d += generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(0.0, s_EGS, nu,nv,nx,ny,nf,neta,nq)[0]
			s_Tot += s_EGS.copy()
			s_fgs_only += s_EGS.copy()
			print 'Using EGS cube'
	elif p.use_EGS_cube:
		d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(sigma, s_EGS, nu,nv,nx,ny,nf,neta,nq, random_seed=non_instrmental_noise_seed)[0]
		s_Tot = s_EGS.copy()
		s_fgs_only = s_EGS.copy()
		print 'Using EGS cube'
	elif p.use_freefree_foreground_cube:
		d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(sigma, s_ff, nu,nv,nx,ny,nf,neta,nq, random_seed=non_instrmental_noise_seed)[0]
		s_Tot = s_ff.copy()
		s_fgs_only = s_ff.copy()
		print 'Using free-free cube'
	else:
		d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(sigma, s_EoR, nu,nv,nx,ny,nf,neta,nq, random_seed=non_instrmental_noise_seed)[0]
		s_Tot = s_EoR.copy()
		print 'Using EoR cube'


	effective_noise = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(sigma, np.zeros(d.shape), nu,nv,nx,ny,nf,neta,nq, random_seed=non_instrmental_noise_seed)[1]
	effective_noise_std = effective_noise.std()




T_Ninv_T = BM.read_data_s2d(array_save_directory+'T_Ninv_T', 'T_Ninv_T')
Npar = shape(T_Ninv_T)[0]
fit_for_LW_power_spectrum = True
masked_power_spectral_modes = np.ones(Npar)
if not fit_for_LW_power_spectrum:
	print 'Not fitting for LW power spectrum. Zeroing relevant modes in the determinant.'
	masked_power_spectral_modes[sorted(np.hstack(k_cube_voxels_in_bin)[0])] = 0.0

masked_power_spectral_modes = masked_power_spectral_modes.astype('bool')
T = BM.read_data_s2d(array_save_directory+'T', 'T')
Finv = BM.read_data_s2d(array_save_directory+'Finv', 'Finv')

#--------------------------------------------
# Data creation with instrumental effects
#--------------------------------------------
overwrite_data_with_WN = False
if p.include_instrumental_effects:
	if overwrite_data_with_WN:
		s_WN, abc, scidata1 = generate_white_noise_signal_instrumental_im_2_vis(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z, Finv,Show,chan_selection,masked_power_spectral_modes, mod_k)
		d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_instrumental_v1(sigma, s_WN, nu,nv,nx,ny,nf,neta,nq,random_seed=2123)[0]
		effective_noise = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_instrumental_v1(sigma, s_WN, nu,nv,nx,ny,nf,neta,nq,random_seed=2123)[1]
	else:
		if p.use_EoR_cube:
			s_EoR, abc, scidata1 = generate_EoR_signal_instrumental_im_2_vis(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z, Finv,Show,chan_selection,masked_power_spectral_modes, mod_k, p.EoR_npz_path_sc)

			plot_figure = False
			if plot_figure:
				pylab.close('all')
				pylab.figure(figsize=(10,10))
				pylab.imshow(scidata1[0], cmap='gist_heat', extent=[-6.5*60,6.5*60,-6.5*60,6.5*60])
				cbar = pylab.colorbar(fraction=1./21.2, pad=0.01)
				cbar.ax.tick_params(labelsize=17) 
				pylab.xlabel('$\\Delta\\mathrm{RA(arcmin)}$', fontsize=20)
				pylab.ylabel('$\\Delta\\mathrm{Dec(arcmin)}$', fontsize=20)
				pylab.tick_params(labelsize=20)
				pylab.tight_layout()
				simulation_plots_dir = '/gpfs/data/jpober/psims/EoR/Python_Scripts/BayesEoR/git_version/BayesEoR/spec_model_tests/Plots/EoRFgSpecM_foreground_images/'
				sim_name = p.EoR_npz_path_sc.split('/')[-1].replace('.npz','').replace('.','d')+'.png'
				pylab.savefig(simulation_plots_dir+sim_name)
				pylab.show()

			EoR_noise_seed = 742123
			EoR_noise_seed = 74212
			print 'EoR_noise_seed', EoR_noise_seed
			d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_instrumental_v1(1.0*sigma, s_EoR, nu,nv,nx,ny,nf,neta,nq,random_seed=EoR_noise_seed)[0]
			effective_noise = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_instrumental_v1(1.0*sigma, s_EoR, nu,nv,nx,ny,nf,neta,nq,random_seed=EoR_noise_seed)[1]

		use_foreground_cubes = True
		# use_foreground_cubes = False
		if use_foreground_cubes:
			#--------------------------------------------
			# Generate mock GDSE data
			#--------------------------------------------
			p.use_GDSE_foreground_cube = True
			# p.use_GDSE_foreground_cube = False
			if p.use_GDSE_foreground_cube:
				print 'Using use_GDSE_foreground_cube data'
				gdsers = 314211
				save_fits = False
				foreground_outputs = generate_Jelic_cube_instrumental_im_2_vis_v2d0(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show, p.beta_experimental_mean,p.beta_experimental_std,p.gamma_mean,p.gamma_sigma,p.Tb_experimental_mean_K,p.Tb_experimental_std_K,p.nu_min_MHz,p.channel_width_MHz,Finv, generate_additional_extrapolated_HF_foreground_cube=True, fits_storage_dir=p.fits_storage_dir, HF_nu_min_MHz_array=p.HF_nu_min_MHz_array, simulation_FoV_deg=p.simulation_FoV_deg, simulation_resolution_deg=p.simulation_resolution_deg,random_seed=gdsers, save_fits=save_fits)


				fg_GDSE, s_GDSE, Tb_nu, beta_experimental_mean,beta_experimental_std,gamma_mean,gamma_sigma,Tb_experimental_mean_K,Tb_experimental_std_K,nu_min_MHz,channel_width_MHz, HF_Tb_nu, scidata1_subset = foreground_outputs
				foreground_outputs = []

				plot_figure = False
				if plot_figure:
					pylab.close('all')
					pylab.figure(figsize=(10,10))
					pylab.imshow(Tb_nu[0], cmap='gist_heat', extent=[-6.5*60,6.5*60,-6.5*60,6.5*60])
					cbar = pylab.colorbar(fraction=1./21.2, pad=0.01)
					cbar.ax.tick_params(labelsize=17) 
					pylab.xlabel('$\\Delta\\mathrm{RA(arcmin)}$', fontsize=20)
					pylab.ylabel('$\\Delta\\mathrm{Dec(arcmin)}$', fontsize=20)
					pylab.tick_params(labelsize=20)
					pylab.tight_layout()
					simulation_plots_dir = '/gpfs/data/jpober/psims/EoR/Python_Scripts/BayesEoR/git_version/BayesEoR/spec_model_tests/Plots/EoRFgSpecM_foreground_images/'
					sim_name = 'Jelic_GDSE_cube_{}_MHz_K_mean_T_{}_RMS_T_{}'.format(p.nu_min_MHz, np.round(p.Tb_experimental_mean_K,1), np.round(p.Tb_experimental_std_K,1)).replace('.','d')+'.png'
					pylab.savefig(simulation_plots_dir+sim_name)
					pylab.show()

				scale_factor = 1.0
				noise_seed = 742123

				d += generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_instrumental_v1(0.0, s_GDSE, nu,nv,nx,ny,nf,neta,nq,random_seed=noise_seed)[0]
				effective_noise = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_instrumental_v1(sigma, s_GDSE, nu,nv,nx,ny,nf,neta,nq,random_seed=noise_seed)[1]

			#--------------------------------------------
			# Generate mock EGS data
			#--------------------------------------------
			p.use_EGS_cube = True
			# p.use_EGS_cube = False
			if p.use_EGS_cube:
				print 'Using use_EGS_cube data'
				foreground_outputs = generate_data_from_loaded_EGS_cube_im_2_vis_v2d0(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show,p.EGS_npz_path,Finv)

				s_EGS, abc_EGS, scidata1_EGS = foreground_outputs
				foreground_outputs = []

				d += generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_instrumental_v1(0.0, s_EGS, nu,nv,nx,ny,nf,neta,nq,random_seed=noise_seed)[0]
				effective_noise = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_instrumental_v1(sigma, s_EGS, nu,nv,nx,ny,nf,neta,nq,random_seed=noise_seed)[1]

			#--------------------------------------------
			# Generate mock free-free data
			#--------------------------------------------
			p.use_freefree_foreground_cube = True
			# p.use_freefree_foreground_cube = False
			if p.use_freefree_foreground_cube:
				print 'Using use_freefree_foreground_cube data'
				save_fits = False
				# ffrs = 314211
				ffrs = 31421
				foreground_outputs_ff = generate_Jelic_cube_instrumental_im_2_vis_v2d0(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show, p.beta_experimental_mean_ff,p.beta_experimental_std_ff,p.gamma_mean_ff,p.gamma_sigma_ff,p.Tb_experimental_mean_K_ff,p.Tb_experimental_std_K_ff,p.nu_min_MHz_ff,p.channel_width_MHz_ff,Finv, generate_additional_extrapolated_HF_foreground_cube=True, fits_storage_dir=p.fits_storage_dir_ff, HF_nu_min_MHz_array=p.HF_nu_min_MHz_array_ff, simulation_FoV_deg=p.simulation_FoV_deg_ff, simulation_resolution_deg=p.simulation_resolution_deg_ff,random_seed=ffrs, save_fits=save_fits)

				fg_ff, s_ff, Tb_nu_ff = foreground_outputs_ff[:3]
				foreground_outputs_ff = []

				plot_figure = False
				if plot_figure:
					pylab.close('all')
					pylab.figure(figsize=(10,10))
					pylab.imshow(Tb_nu_ff[0], cmap='gist_heat', extent=[-6.5*60,6.5*60,-6.5*60,6.5*60])
					cbar = pylab.colorbar(fraction=1./21.2, pad=0.01)
					cbar.ax.tick_params(labelsize=17) 
					pylab.xlabel('$\\Delta\\mathrm{RA(arcmin)}$', fontsize=20)
					pylab.ylabel('$\\Delta\\mathrm{Dec(arcmin)}$', fontsize=20)
					pylab.tick_params(labelsize=20)
					pylab.tight_layout()
					simulation_plots_dir = '/gpfs/data/jpober/psims/EoR/Python_Scripts/BayesEoR/git_version/BayesEoR/spec_model_tests/Plots/EoRFgSpecM_foreground_images/'
					sim_name = 'Jelic_free_free_cube_{}_MHz_K_mean_T_{}_RMS_T_{}_v2d0'.format(p.nu_min_MHz_ff, np.round(p.Tb_experimental_mean_K_ff,1), np.round(p.Tb_experimental_std_K_ff,1)).replace('.','d')+'.png'
					pylab.savefig(simulation_plots_dir+sim_name)
					pylab.show()

				d += generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_instrumental_v1(0.0, s_ff, nu,nv,nx,ny,nf,neta,nq,random_seed=noise_seed)[0]
				effective_noise = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_instrumental_v1(sigma, s_ff, nu,nv,nx,ny,nf,neta,nq,random_seed=noise_seed)[1]

effective_noise_std = effective_noise.std()






# calculte_PS_of_EoR_cube_directly = True
calculte_PS_of_EoR_cube_directly = False
if calculte_PS_of_EoR_cube_directly:
	#--------------------------------------------
	# Calculate power spectrum of the EoR subset cube and output it to file
	#--------------------------------------------
	# calculate_subset_cube_power_spectrum_v1d0(nu,nv,nx,ny,nf,neta,nq,k_cube_voxels_in_bin,modkbins_containing_voxels,p.EoR_npz_path_sc,mod_k,k_z)
	calculate_subset_cube_power_spectrum_v2d0(nu,nv,nx,ny,nf,neta,nq,k_cube_voxels_in_bin,modkbins_containing_voxels,p.EoR_npz_path_sc,mod_k,k_z)

	#--------------------------------------------
	# This function calculates both the power spectrum of the EoR subset cube and the (cylindrical) power spectrum of the full EoR cube in the k-space volume accessible to the subset cube in order to demonstrate that they are consistent within sample variance.
	# NOTE: while the power spectrum of the EoR subset cube is equal to the (cylindrical) power spectrum of the full EoR cube in the k-space volume accessible to the subset cube, it is not identical to the spherically averaged power specrum of the full EoR cube. This is due to the EoR cube output by 21cmFast not being spherically symmetric! (it is approximately sperically symmetric but only to within a factor of 2 in power). As mentioned above, this is solved by comparing the power spectra in matching regions of k-space (instead of comparing the small cylindrical subset of the spherical annulus to the full spherical annulus, which would only be expected to match for a truely spherically symmetric EoR power spectrum - i.e. no redshift space distortions etc.)!
	#--------------------------------------------
	calculate_21cmFAST_EoR_cube_power_spectrum_in_subset_cube_bins_v1d0(nu,nv,nx,ny,nf,neta,nq,k_cube_voxels_in_bin,modkbins_containing_voxels,p.EoR_npz_path_sc,mod_k,k_x,k_y,k_z)


# start = time.time()
# T_Ninv_T = BM.read_data_from_hdf5(array_save_directory+'T_Ninv_T.h5', 'T_Ninv_T')
# print 'Time taken: {}'.format(time.time()-start)

# start = time.time()
# a = np.dot(T.conjugate().T,np.dot(Ninv,T))
# print 'Time taken: {}'.format(time.time()-start)

#--------------------------------------------
# Load base matrices used in the likelihood and define related variables
#--------------------------------------------
T = BM.read_data_s2d(array_save_directory+'T', 'T')
T_Ninv_T = BM.read_data_s2d(array_save_directory+'T_Ninv_T', 'T_Ninv_T')
block_T_Ninv_T = BM.read_data_s2d(array_save_directory+'block_T_Ninv_T', 'block_T_Ninv_T')
Ninv = BM.read_data_s2d(array_save_directory+'Ninv', 'Ninv')

# Ninv = BM.convert_sparse_matrix_to_dense_numpy_array(Ninv)

Ninv_d = np.dot(Ninv,d)
dbar = np.dot(T.conjugate().T,Ninv_d)
# Ninv=[]
Sigma_Diag_Indices=np.diag_indices(shape(T_Ninv_T)[0])
nDims = len(k_cube_voxels_in_bin)
d_Ninv_d = np.dot(d.conjugate(), Ninv_d)

if p.use_intrinsic_noise_fitting:
	nDims = nDims+1

if p.fit_for_spectral_model_parameters:
	nDims = nDims+p.npl

###
# nDims = nDims+3 for Gaussian prior over the three long wavelength model vectors
###
if p.use_LWM_Gaussian_prior:
	nDims = nDims+3

x=[100.e0]*nDims
if p.fit_for_monopole:
	nuv = (nu*nv)
else:
	nuv = (nu*nv-1)
block_T_Ninv_T = np.array([np.hsplit(block,nuv) for block in np.vsplit(T_Ninv_T,nuv)])
if p.include_instrumental_effects:
	block_T_Ninv_T=[]






do_null_space_test = False
if do_null_space_test:
	a = np.linalg.svd(Finv)
	U = a[0]
	U1 = U[:,0:Finv.shape[1]]

	test_im = np.random.normal(0.0,1,Finv.shape[1])
	test_vis = np.dot(Finv, np.random.normal(test_im))
	test_sigma = 1.e-5
	test_n = np.random.normal(0.0,test_sigma,Finv.shape[0])+1j*np.random.normal(0.0,test_sigma,Finv.shape[0])
	test_vis = test_vis+test_n
	test_Ninv = np.identity(Finv.shape[0])/test_sigma**2.

	U1_test_Ninv_U1 = np.dot(U1.conjugate().T, np.dot(test_Ninv,U1))
	U1_test_Ninv_U1_inv = np.linalg.inv(U1_test_Ninv_U1)
	U1_test_Ninv_d = np.dot(U1.conjugate().T, np.dot(test_Ninv,test_vis))

	test_a = np.dot(U1_test_Ninv_U1_inv, U1_test_Ninv_d)
	model_vis = np.dot(U1,test_a)

	a1 = np.dot(a[0][:, :(len(a[1]))]*a[1], a[2])
	print 'a1 == Finv?:', np.allclose(Finv, a1)
	print abs(Finv-a1).max()

	print np.sum((a[1]**2.))
	print np.sum((a[1]**2.))*0.9
	print np.sum((a[1]**2.)[0:520]) #Exclude highest weighted vectors (the highest weighted that sum to 90% of the weight)

	###

	singular_values_with_greater_than_10pow5_downweighting = np.where(a[1]<1.e-2)
	# singular_values_with_greater_than_10pow5_downweighting = np.where(a[1][520:])
	null_space_im_vectors = a[2][:,singular_values_with_greater_than_10pow5_downweighting[0]]
	ns1 = null_space_im_vectors.copy()

	ns1 = np.hstack((pl1,null_space_im_vectors))
	# ns1 = pl1.copy()

	d_im = scidata1_subset.copy()
	# if not p.fit_for_monopole:
	for i_chan in range(len(d_im)):
		d_im[i_chan] = d_im[i_chan]-d_im[i_chan].mean()
	d_im = d_im.flatten()

	sigma_im = 1.e-10
	noise = np.random.normal(0,sigma_im,d_im.size)
	d_im = d_im+noise
	N_im_inv = np.identity(d_im.size)/sigma_im**2.

	ns1_N_im_inv_ns1 = np.dot(ns1.conjugate().T, np.dot(N_im_inv, ns1))
	ns1_N_im_inv_ns1_inv = np.linalg.inv(ns1_N_im_inv_ns1)
	print np.sum(abs(np.dot(ns1_N_im_inv_ns1_inv,ns1_N_im_inv_ns1) -1.0)<1.e-10)

	ns1_N_im_inv_d_im = np.dot(ns1.conjugate().T, np.dot(N_im_inv, d_im))

	ml_theta = np.dot(ns1_N_im_inv_ns1_inv,ns1_N_im_inv_d_im)
	ml_null_im = np.dot(ns1,ml_theta)

	print (d_im-ml_null_im).max()
	print (d_im-ml_null_im).std()
	print np.log10((d_im-ml_null_im).std()/d_im.std())




# -----------------------------------------------
# -----------------------------------------------
###
# ML image model recovery testing
###
ML_image_model_recovery_testing = False
if ML_image_model_recovery_testing:

	Fprime_Fz = BM.read_data_s2d(array_save_directory+'Fprime_Fz', 'Fprime_Fz')

	###
	# Single power law fit
	###
	# pl1 = Fprime_Fz.T[19::neta+nq].T.copy()
	pl1 = Fprime_Fz.T[neta/2::neta+nq].T.copy()
	# d_im = Tb_nu[:,0:9,0:9].copy()
	d_im = scidata1_subset.copy()
	# if not p.fit_for_monopole:
	for i_chan in range(len(d_im)):
		d_im[i_chan] = d_im[i_chan]-d_im[i_chan].mean()
	d_im = d_im.flatten()

	sigma_im = 1.e-10
	noise = np.random.normal(0,sigma_im,d_im.size)
	d_im = d_im+noise
	N_im_inv = np.identity(d_im.size)/sigma_im**2.

	pl1_N_im_inv_pl1 = np.dot(pl1.conjugate().T, np.dot(N_im_inv, pl1))
	pl1_N_im_inv_pl1_inv = np.linalg.inv(pl1_N_im_inv_pl1)
	print np.sum(abs(np.dot(pl1_N_im_inv_pl1_inv,pl1_N_im_inv_pl1) -1.0)<1.e-10)

	# pl1_N_im_inv_d_im = np.dot(pl1.conjugate().T, np.dot(N_im_inv, d_im))
	pl1_N_im_inv_d_im = np.dot(pl1.conjugate().T, np.dot(N_im_inv, d_im))

	ml_theta = np.dot(pl1_N_im_inv_pl1_inv,pl1_N_im_inv_d_im)
	ml_im = np.dot(pl1,ml_theta)

	print (d_im-ml_im).max()
	print (d_im-ml_im).std()
	print np.log10((d_im-ml_im).std()/d_im.std())
	# print np.log10((d_im-ml_im-	ml_null_im).std()/d_im.std())

	###
	# Constant plus single power law fit
	###
	# cpl1 = np.array([row for i,row in enumerate(Fprime_Fz.T) if (i+((neta+nq)-38))%(neta+nq)==0 or (i+((neta+nq)-19))%(neta+nq)==0]).T
	cpl1 = np.array([row for i,row in enumerate(Fprime_Fz.T) if (i+((neta+nq)-neta+0))%(neta+nq)==0 or (i+((neta+nq)-neta/2))%(neta+nq)==0]).T

	# d_im = Tb_nu[:,0:9,0:9].copy()
	d_im = scidata1_subset.copy()
	for i_chan in range(len(d_im)):
		d_im[i_chan] = d_im[i_chan]-d_im[i_chan].mean()
	d_im = d_im.flatten()

	sigma_im = 1.e-10
	noise = np.random.normal(0,sigma_im,d_im.size)
	d_im = d_im+noise
	N_im_inv = np.identity(d_im.size)/sigma_im**2.

	cpl1_N_im_inv_cpl1 = np.dot(cpl1.conjugate().T, np.dot(N_im_inv, cpl1))
	cpl1_N_im_inv_cpl1_inv = np.linalg.inv(cpl1_N_im_inv_cpl1)
	print np.sum(abs(np.dot(cpl1_N_im_inv_cpl1_inv,cpl1_N_im_inv_cpl1) -1.0)<1.e-10)

	cpl1_N_im_inv_d_im = np.dot(cpl1.conjugate().T, np.dot(N_im_inv, d_im))

	ml_theta = np.dot(cpl1_N_im_inv_cpl1_inv,cpl1_N_im_inv_d_im)
	ml_im = np.dot(cpl1,ml_theta)

	print (d_im-ml_im).max()
	print (d_im-ml_im).std()
	print np.log10((d_im-ml_im).std()/d_im.std())

# -----------------------------------------------
# -----------------------------------------------







# run_nudft_test=True
run_nudft_test=False
if run_nudft_test:
	###
	def Produce_Coordinate_Arrays_ZM(nu, nv, nx, ny):
		#U_oversampling_Factor=nu/float(nx) #Keeps uv-plane size constant and oversampled rather than DFTing to a larger uv-plane
		#V_oversampling_Factor=nv/float(ny) #Keeps uv-plane size constant and oversampled rather than DFTing to a larger uv-plane
		#
		i_y_Vector=(np.arange(ny)-ny/2)
		#i_y_Vector=numpy.fft.fftshift(arange(ny)) #This puts the centre of x,y grid: 0,0 at the centre of the vector rather than the start
		i_y_Vector=i_y_Vector.reshape(1,ny)
		i_y_Array=np.tile(i_y_Vector,ny)
		i_y_Array_Vectorised=i_y_Array.reshape(nx*ny,1)
		i_y_AV=i_y_Array_Vectorised
		#
		i_x_Vector=(np.arange(nx)-nx/2)
		#i_x_Vector=numpy.fft.fftshift(arange(nx)) #This puts the centre of x,y grid: 0,0 at the centre of the vector rather than the start
		i_x_Vector=i_x_Vector.reshape(nx,1)
		i_x_Array=np.tile(i_x_Vector,nx)
		i_x_Array_Vectorised=i_x_Array.reshape(nx*ny,1)
		i_x_AV=i_x_Array_Vectorised
		#
		i_v_Vector=(np.arange(nu)-nu/2)
		#i_v_Vector= numpy.fft.fftshift(arange(nu)) #This puts the centre of u,v grid: 0,0 at the centre of the vector rather than the start
		i_v_Vector=i_v_Vector.reshape(1,nu)
		i_v_Array=np.tile(i_v_Vector,nv)
		i_v_Array_Vectorised=i_v_Array.reshape(1,nu*nv)
		i_v_AV=i_v_Array_Vectorised
		i_v_AV=numpy.delete(i_v_AV,[i_v_AV.size/2]) #Remove the centre uv-pix
		#
		i_u_Vector=(np.arange(nv)-nv/2)
		#i_u_Vector=numpy.fft.fftshift(arange(nv)) #This puts the centre of u,v grid: 0,0 at the centre of the vector rather than the start
		i_u_Vector=i_u_Vector.reshape(nv,1)
		i_u_Array=np.tile(i_u_Vector,nu)
		i_u_Array_Vectorised=i_u_Array.reshape(1,nv*nu)
		i_u_AV=i_u_Array_Vectorised
		i_u_AV=numpy.delete(i_u_AV,[i_u_AV.size/2]) #Remove the centre uv-pix
		#
		#
		#ExponentArray=np.exp(-2.0*np.pi*1j*( (i_x_AV*i_u_AV/float(nx)) +  (i_v_AV*i_y_AV/float(ny)) ))
		return i_x_AV, i_y_AV, i_u_AV, i_v_AV



	nu, nv, nx, ny = p.nu, p.nv, p.nx, p.ny


	###
	def DFT_Array_DFT_2D_ZM(nu, nv, nx, ny):
		#
		i_x_AV, i_y_AV, i_u_AV, i_v_AV = Produce_Coordinate_Arrays_ZM(nu, nv, nx, ny)
		#
		ExponentArray=np.exp(-2.0*np.pi*1j*( (i_x_AV*i_u_AV/float(nx)) +  (i_v_AV*i_y_AV/float(ny)) ))
		return ExponentArray


	use_HERA_uv_coverage = True
	# use_HERA_uv_coverage = False
	if use_HERA_uv_coverage:
		i_x_AV, i_y_AV, i_u_AV, i_v_AV = Produce_Coordinate_Arrays_ZM(nu, nv, nx, ny)
		Jacob_uv_dict_dir = '/gpfs/data/jpober/shared/bayeseor_files/'
		uv_array = np.load(Jacob_uv_dict_dir+'noise_hera7-hex_1.0e+01rms_38nfreqs_5ntimes_phased.npy').item()
		i_u_AV = uv_array['uvw_array'][:,0]
		i_v_AV = uv_array['uvw_array'][:,1]

		# Check uv_array is Hermitian (everything should print true)
		for uv in uv_array['uvw_array'][:,:2]:                                 
		    if not (uv in uv_array['uvw_array'][:,:2] and -uv in uv_array['uvw_array'][:,:2]):
		    	print 'ERROR, uv_array is not Hermitian'

	else:
		i_x_AV, i_y_AV, i_u_AV, i_v_AV = Produce_Coordinate_Arrays_ZM(nu, nv, nx, ny)
		# i_u_AV = i_u_AV/2.
		# i_v_AV = i_v_AV/2.
		np.random.seed(123)
		i_u_AV_pt1 = np.random.uniform(i_u_AV.min(), i_u_AV.max(), len(i_u_AV)*3)
		i_u_AV = np.hstack((i_u_AV_pt1, -1.*i_u_AV_pt1[::-1]))
		np.random.seed(456)
		i_v_AV_pt1 = np.random.uniform(i_v_AV.min(), i_v_AV.max(), len(i_v_AV)*3)
		i_v_AV = np.hstack((i_v_AV_pt1, -1.*i_v_AV_pt1[::-1]))

	pylab.close('all')
	pylab.figure()
	pylab.scatter(i_u_AV, i_v_AV)
	pylab.show()

	dft_array = np.exp(-2.0*np.pi*1j*( (i_x_AV*i_u_AV/float(nx)) +  (i_v_AV*i_y_AV/float(ny)) ))

	# dft_array = BM.read_data_from_hdf5(array_save_directory+'dft_array.h5', 'dft_array')
	Finv = block_diag(*[dft_array.T for i in range(p.nf)])

	# Finv = BM.read_data_from_hdf5(array_save_directory+'Finv.h5', 'Finv')
	Fprime_Fz = BM.read_data_s2d(array_save_directory+'Fprime_Fz', 'Fprime_Fz')
	T = np.dot(Finv, Fprime_Fz)
	# Ninv = BM.read_data_from_hdf5(array_save_directory+'Ninv.h5', 'Ninv')
	Ninv = np.identity(T.shape[0])*1./sigma**2.	+0j
	Ninv_T = np.dot(Ninv,T)
	T_Ninv_T = np.dot(T.conjugate().T,Ninv_T)

	pylab.close('all')
	pylab.figure()
	pylab.imshow((dft_array.real))
	pylab.colorbar()
	pylab.show()

	pylab.close('all')
	pylab.figure()
	pylab.imshow((abs(Finv)))
	pylab.colorbar()
	pylab.show()

	pylab.close('all')
	pylab.figure()
	pylab.imshow(np.log10(abs(T)))
	pylab.colorbar()
	pylab.show()

	theta_test = np.zeros(T.shape[1])
	# theta_test[38*0+17] = 1.0
	for i in range(8):
		theta_test[38*i+17] = 1.0

	m_test = np.dot(T,theta_test)

	pylab.figure()
	pylab.imshow(abs(m_test.reshape(38,-1)))
	pylab.colorbar()
	pylab.show()

	pylab.figure()
	pylab.errorbar(arange(38), m_test.reshape(38,-1)[:,0].real)
	pylab.errorbar(arange(38), m_test.reshape(38,-1)[:,0].imag)
	# pylab.show()

	pylab.figure()
	pylab.errorbar(arange(38), abs(m_test.reshape(38,-1)[:,0]))
	pylab.show()

	Sigma = T_Ninv_T.copy()
	Sigma[np.diag_indices(len(T_Ninv_T))] += 1.e-10
	Sigma_inv = np.linalg.inv(Sigma)
	a = np.dot(Sigma_inv, Sigma)

	pylab.figure()
	pylab.imshow(abs(a))
	pylab.colorbar()
	pylab.figure()
	pylab.imshow(np.log10(abs(a)))
	pylab.colorbar()
	pylab.show()


	if p.useGPU:
		import pycuda.autoinit
		import pycuda.driver as cuda
		for devicenum in range(cuda.Device.count()):
		    device=cuda.Device(devicenum)
		    attrs=device.get_attributes()
		    print("\n===Attributes for device %d"%devicenum)
		    for (key,value) in attrs.iteritems():
		        print("%s:%s"%(str(key),str(value)))

		import time
		import numpy as np
		import ctypes
		from numpy import ctypeslib
		from scipy import linalg
		from subprocess import os

		current_dir = os.getcwd()
		base_dir = '/'.join(current_dir.split('/')[:np.where(np.array(current_dir.split('/'))=='BayesEoR')[0][-1]])+'/BayesEoR/'
		GPU_wrap_dir = base_dir+'likelihood_tests/SimpleEoRtestWQ/GPU_wrapper/'
		wrapmzpotrf=ctypes.CDLL(GPU_wrap_dir+'wrapmzpotrf.so')
		nrhs=1
		wrapmzpotrf.cpu_interface.argtypes = [ctypes.c_int, ctypes.c_int, ctypeslib.ndpointer(np.complex128, ndim=2, flags='C'), ctypeslib.ndpointer(np.complex128, ndim=1, flags='C'), ctypes.c_int]
		print 'Computing on GPU'

		print 'performing GPU inversion check...'
		dbar=np.ones_like(d)
		Sigma = T_Ninv_T.copy()
		# powers = 1.e-10
		powers = 10.
		Sigma[np.diag_indices(len(T_Ninv_T))] += powers

		dbar_copy = dbar.copy()
		dbar_copy_copy = dbar.copy()
		# wrapmzpotrf.cpu_interface(len(Sigma), nrhs, Sigma, dbar_copy, 1) #to print debug
		wrapmzpotrf.cpu_interface(len(Sigma), nrhs, Sigma, dbar_copy, 0)

		logdet_Magma_Sigma = np.sum(np.log(np.diag(abs(Sigma))))*2 #Note: After wrapmzpotrf, Sigma is actually SigmaCho (i.e. L with SigmaLL^T)
		SigmaI_dbar = linalg.cho_solve((Sigma.conjugate().T,True), dbar_copy_copy)

		Sigma = T_Ninv_T.copy()
		Sigma[np.diag_indices(len(T_Ninv_T))] += powers
		Sigma_SigmaI_dbar = np.dot(Sigma, SigmaI_dbar)

		returned_arrays_are_equal = np.allclose(dbar, Sigma_SigmaI_dbar, rtol=sigma/1.e3)
		if returned_arrays_are_equal:
			print 'Sigma inversion worked correctly'
		else:
			print 'A problem occured with the inversion of Sigma.....'









from numpy import real

#--------------------------------------------
# Instantiate class and check that posterior_probability returns a finite probability (so no obvious binning errors etc.)
#--------------------------------------------
if small_cube: PSPP = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, Ninv, d_Ninv_d, log_priors=False, intrinsic_noise_fitting=p.use_intrinsic_noise_fitting, fit_for_spectral_model_parameters=p.fit_for_spectral_model_parameters)
PSPP_block_diag = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, Ninv, d_Ninv_d, block_T_Ninv_T=block_T_Ninv_T, Print=True, log_priors=False, intrinsic_noise_fitting=p.use_intrinsic_noise_fitting, fit_for_spectral_model_parameters=p.fit_for_spectral_model_parameters)
self = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, Ninv, d_Ninv_d, block_T_Ninv_T=block_T_Ninv_T, Print=True, log_priors=False, intrinsic_noise_fitting=p.use_intrinsic_noise_fitting, fit_for_spectral_model_parameters=p.fit_for_spectral_model_parameters)
start = time.time()

if small_cube:
	print PSPP_block_diag.posterior_probability([1.e0]*nDims, diagonal_sigma=False, block_T_Ninv_T=block_T_Ninv_T)[0]


iterative_sub_MLLWM = True #Fit iteratively to sidestep numerical limitations (complex double precision) of the inversion
# iterative_sub_MLLWM = False #Fit iteratively to sidestep numerical limitations (complex double precision) of the inversion
if sub_MLLWM:
	if not iterative_sub_MLLWM:
		pre_sub_dbar = PSPP_block_diag.dbar
		PSPP_block_diag.dimensionless_PS=True
		PSPP_block_diag.inverse_LW_power=p.inverse_LW_power #Include minimal prior over LW modes to ensure numerical stability
		if p.use_LWM_Gaussian_prior:
			fit_constraints = [9.e9,9.e9,9.e9]+[1.e-20]*(nDims-3)
		else:
			PSPP_block_diag.inverse_LW_power = 2.e-20
			fit_constraints = [1.e-20]*(nDims)
		maxL_LW_fit = PSPP_block_diag.calc_SigmaI_dbar_wrapper(fit_constraints, T_Ninv_T, pre_sub_dbar, block_T_Ninv_T=block_T_Ninv_T)[0]
		maxL_LW_signal = np.dot(T,maxL_LW_fit)
		Ninv = BM.read_data_s2d(array_save_directory+'Ninv', 'Ninv')
		Ninv_maxL_LW_signal = np.dot(Ninv,maxL_LW_signal)
		ML_qbar = np.dot(T.conjugate().T,Ninv_maxL_LW_signal)
		q_sub_dbar = pre_sub_dbar-ML_qbar
		if small_cube: PSPP.dbar = q_sub_dbar
		PSPP_block_diag.dbar = q_sub_dbar
		print '(d - s_GDSE).std()', effective_noise_std
		print '(d - maxL_LW_signal).std()', (d - maxL_LW_signal).std()
		PSPP_block_diag.inverse_LW_power = p.inverse_LW_power #Reset inverse_LW_power to ensure numerical stability with non-zero Fourier modes

		if small_cube:
			print PSPP.posterior_probability([1.e0]*nDims, diagonal_sigma=False)[0]
			print PSPP_block_diag.posterior_probability([1.e0]*nDims, diagonal_sigma=False, block_T_Ninv_T=block_T_Ninv_T)[0]
		if p.use_intrinsic_noise_fitting:
			q_sub_d = d - maxL_LW_signal
			Ninv_q_sub_d = np.dot(Ninv, q_sub_d)
			q_sub_d_Ninv_q_sub_d = np.dot(q_sub_d, Ninv_q_sub_d)
			PSPP_block_diag.d_Ninv_d = 	q_sub_d_Ninv_q_sub_d


	else:
		Ninv = BM.read_data_s2d(array_save_directory+'Ninv', 'Ninv')
		maxL_LW_fit_array = []
		maxL_LW_signal_array = []
		pre_sub_dbar = PSPP_block_diag.dbar
		PSPP_block_diag.dimensionless_PS=True
		PSPP_block_diag.inverse_LW_power=p.inverse_LW_power #Include minimal prior over LW modes to ensure numerical stability
		dbar_prime_i = dbar.copy()
		d_prime_i = d.copy()

		PSPP_block_diag.Print = True
		print 'effective_noise_std', effective_noise_std

		count=0
		# count_max = 10
		count_max = 0
		maxL_LW_signal = np.zeros(len(d))
		bias_threshold = 0.0
		# bias_threshold = -1.0

		# for i in range(7):
		while count<=count_max and d_prime_i.std() >= (1.0+bias_threshold)*effective_noise_std:
			# fit_constraints = [9.e9,9.e9,9.e9]+[1.e-20]*(nDims-3)
			# fit_constraints = [1.e9,1.e9,1.e9]+[1.e-20]*(nDims-3)
			if p.use_LWM_Gaussian_prior:
				fit_constraints = [1.e9,1.e9,1.e9]+[1.e-20]*(nDims-3)
			else:
				PSPP_block_diag.inverse_LW_power = p.inverse_LW_power
				print 'PSPP_block_diag.inverse_LW_power:', PSPP_block_diag.inverse_LW_power


				# fit_constraints = [1.e-100]*(nDims)
				# fit_constraints = [1.e-20]*(nDims)
				fit_constraints = [1.e2]*(nDims)
				fit_constraints = [1.e0]*(nDims)

			maxL_LW_fit = PSPP_block_diag.calc_SigmaI_dbar_wrapper(fit_constraints, T_Ninv_T, dbar_prime_i, block_T_Ninv_T=block_T_Ninv_T)[0]
			# maxL_LW_signal = np.dot(T,maxL_LW_fit)

			use_only_the_mean_component_of_the_foreground_model = False
			if use_only_the_mean_component_of_the_foreground_model:
				if type(p.beta)==list:
					Q_T = np.array([row for i,row in enumerate(T.T) if (i+((neta+nq)-neta/2))%(neta+nq)==0]).T
					maxL_LW_fit = np.array([row for i,row in enumerate(maxL_LW_fit) if (i+((neta+nq)-neta/2))%(neta+nq)==0]).T
				else:
					Q_T = np.array([row for i,row in enumerate(T.T) if (i+((neta+nq)-neta/2))%(neta+nq)==0]).T
					maxL_LW_fit = np.array([row for i,row in enumerate(maxL_LW_fit) if (i+((neta+nq)-neta/2))%(neta+nq)==0]).T
			else:
				if type(p.beta)==list:
					Q_T = np.array([row for i,row in enumerate(T.T) if (i+((neta+nq)-neta+0))%(neta+nq)==0 or (i+((neta+nq)-neta/2))%(neta+nq)==0 or (i+((neta+nq)-neta-1))%(neta+nq)==0]).T
					maxL_LW_fit = np.array([row for i,row in enumerate(maxL_LW_fit) if (i+((neta+nq)-neta+0))%(neta+nq)==0 or (i+((neta+nq)-neta/2))%(neta+nq)==0 or (i+((neta+nq)-neta-1))%(neta+nq)==0]).T
				else:
					Q_T = np.array([row for i,row in enumerate(T.T) if (i+((neta+nq)-neta+0))%(neta+nq)==0 or (i+((neta+nq)-neta/2))%(neta+nq)==0]).T
					maxL_LW_fit = np.array([row for i,row in enumerate(maxL_LW_fit) if (i+((neta+nq)-neta+0))%(neta+nq)==0 or (i+((neta+nq)-neta/2))%(neta+nq)==0]).T
	


			Q_T.imag=0.0
			Q_T_Ninv_Q_T = np.dot(Q_T.conjugate().T, np.dot(Ninv, Q_T))
			# Q_T_Ninv_Q_T[np.diag_indices(Q_T_Ninv_Q_T.shape[0])] += 2.e-20
			Ninv_d = np.dot(Ninv,d)
			Qdbar = np.dot(Q_T.conjugate().T,Ninv_d)
			maxL_LW_fit2 = np.linalg.solve(Q_T_Ninv_Q_T, Qdbar)
			print 'Iterative foreground pre-subtraction complete, {} orders of magnitude foreground supression achieved.\n'.format(np.log10((d-effective_noise).std()/(d-np.dot(Q_T, maxL_LW_fit)-effective_noise).std()))
			print 'Iterative foreground pre-subtraction2 complete, {} orders of magnitude foreground supression achieved.\n'.format(np.log10((d-effective_noise).std()/(d-np.dot(Q_T, maxL_LW_fit2)-effective_noise).std()))
			maxL_LW_signal2 = np.dot(Q_T, maxL_LW_fit2)

			use_joint_fit_foreground_model = True #Set to True for power law analyses
			# use_joint_fit_foreground_model = False #Set to True for power law analyses
			if not use_joint_fit_foreground_model:
				print 'Using independent foreground fitting for foreground prior'
				maxL_LW_fit = maxL_LW_fit2
				maxL_LW_signal = maxL_LW_signal2
			else:
				print 'Using foreground component of joint foreground + Fourier mode fitting for foreground prior'

			maxL_LW_signal = np.dot(Q_T,maxL_LW_fit)
			maxL_LW_fit_array.append(maxL_LW_fit)
			maxL_LW_signal_array.append(maxL_LW_signal)
			print count, (d_prime_i - maxL_LW_signal).std(), '\n'
			d_prime_i = (d_prime_i - maxL_LW_signal)
			Ninv_d_prime_i = np.dot(Ninv,d_prime_i)
			dbar_prime_i = np.dot(T.conjugate().T,Ninv_d_prime_i)
			count+=1

		maxL_LW_fit_array = np.array(maxL_LW_fit_array)
		maxL_LW_signal_array = np.array(maxL_LW_signal_array)

		total_maxL_LW_fit = np.sum(maxL_LW_fit_array, axis=0)
		# total_maxL_LW_signal = np.dot(T,total_maxL_LW_fit)
		total_maxL_LW_signal = np.dot(Q_T,total_maxL_LW_fit)
		# total_maxL_LW_signal = np.dot(Q_T,total_maxL_LW_fit) * (1.0-2.e-2)

		# ###
		# # Save total_maxL_LW_signal for inspection
		# ###
		# total_maxL_LW_signal_save_dir = 'random/total_maxL_LW_signal_storage/'+'/'.join(array_save_directory.split('/')[2:]).replace('.','d').replace('+', '')
		# if not os.path.isdir(total_maxL_LW_signal_save_dir):
		# 	os.makedirs(total_maxL_LW_signal_save_dir)
		# total_maxL_LW_signal_save_path = total_maxL_LW_signal_save_dir+'total_maxL_LW_signal'
		# # np.savez(total_maxL_LW_signal_save_path, total_maxL_LW_signal)
		# if p.beta==[-1.0,-2.0]:
		# 	print 'Loading total_maxL_LW_signal from file:', total_maxL_LW_signal_save_path+'.npz'
		# 	total_maxL_LW_signal = np.load(total_maxL_LW_signal_save_path+'.npz')['arr_0']

		Ninv_total_maxL_LW_signal = np.dot(Ninv,total_maxL_LW_signal)
		ML_qbar = np.dot(T.conjugate().T,Ninv_total_maxL_LW_signal)
		q_sub_dbar = pre_sub_dbar-ML_qbar
		if small_cube: PSPP.dbar = q_sub_dbar
		PSPP_block_diag.dbar = q_sub_dbar
		PSPP_block_diag.inverse_LW_power = p.inverse_LW_power #Reset inverse_LW_power to ensure numerical stability with non-zero Fourier modes

		print 'Iterative foreground pre-subtraction complete, {} orders of magnitude foreground supression achieved.\n'.format(np.log10((d-effective_noise).std()/(d-total_maxL_LW_signal-effective_noise).std()))

		if small_cube:
			print PSPP.posterior_probability([1.e0]*nDims, diagonal_sigma=False)[0]
			print PSPP_block_diag.posterior_probability([1.e0]*nDims, diagonal_sigma=False, block_T_Ninv_T=block_T_Ninv_T)[0]

		if p.use_intrinsic_noise_fitting:
			q_sub_d = d - total_maxL_LW_signal
			Ninv_q_sub_d = np.dot(Ninv, q_sub_d)
			q_sub_d_Ninv_q_sub_d = np.dot(q_sub_d, Ninv_q_sub_d)
			PSPP_block_diag.d_Ninv_d = 	q_sub_d_Ninv_q_sub_d








if sub_ML_monopole_term_model:
	pre_sub_dbar = PSPP_block_diag.dbar
	PSPP_block_diag.inverse_LW_power=0.0
	PSPP_block_diag.inverse_LW_power_zeroth_LW_term=p.inverse_LW_power
	PSPP_block_diag.inverse_LW_power_first_LW_term=2.e18 #Don't fit for the first LW term (since only fitting for the monopole)
	PSPP_block_diag.inverse_LW_power_second_LW_term=2.e18  #Don't fit for the second LW term (since only fitting for the monopole)
	if p.use_LWM_Gaussian_prior:
		fit_constraints = [2.e18]*1+[1.e-20]*(nDims-1)	
	else:
		fit_constraints = [1.e-20]*(nDims)
	maxL_LW_fit = PSPP_block_diag.calc_SigmaI_dbar_wrapper(fit_constraints, T_Ninv_T, pre_sub_dbar, block_T_Ninv_T=block_T_Ninv_T)[0]	
	maxL_LW_signal = np.dot(T,maxL_LW_fit)
	Ninv = BM.read_data_s2d(array_save_directory+'Ninv', 'Ninv')
	Ninv_maxL_LW_signal = np.dot(Ninv,maxL_LW_signal)
	ML_qbar = np.dot(T.conjugate().T,Ninv_maxL_LW_signal)
	q_sub_dbar = pre_sub_dbar-ML_qbar
	if small_cube: PSPP.dbar = q_sub_dbar
	PSPP_block_diag.dbar = q_sub_dbar
	# PSPP_block_diag.inverse_LW_power=0.0 #Remove the constraints on the LW model for subsequent parts of the analysis
	PSPP_block_diag.inverse_LW_power=p.inverse_LW_power #Remove the constraints on the LW model for subsequent parts of the analysis
	print 'Foreground pre-subtraction complete, {} orders of magnitude foreground supression achieved.\n'.format(np.log10((d-effective_noise).std()/(d-maxL_LW_signal-effective_noise).std()))
	if small_cube:
		print PSPP.posterior_probability([1.e0]*nDims, diagonal_sigma=False)[0]
		print PSPP_block_diag.posterior_probability([1.e0]*nDims, diagonal_sigma=False, block_T_Ninv_T=block_T_Ninv_T)[0]

if sub_ML_monopole_plus_first_LW_term_model:
	pre_sub_dbar = PSPP_block_diag.dbar
	PSPP_block_diag.inverse_LW_power=0.0
	PSPP_block_diag.inverse_LW_power_zeroth_LW_term=2.e-18 #Don't fit for the first LW term (since only fitting for the monopole)
	PSPP_block_diag.inverse_LW_power_first_LW_term=2.e-18 #Don't fit for the first LW term (since only fitting for the monopole)
	PSPP_block_diag.inverse_LW_power_second_LW_term=2.e18  #Don't fit for the second LW term (since only fitting for the monopole)
	if p.use_LWM_Gaussian_prior:
		fit_constraints = [2.e18]*2+[1.e-20]*(nDims-2)
	else:
		fit_constraints = [1.e-20]*(nDims)
	maxL_LW_fit = PSPP_block_diag.calc_SigmaI_dbar_wrapper(fit_constraints, T_Ninv_T, pre_sub_dbar, block_T_Ninv_T=block_T_Ninv_T)[0]
	maxL_LW_signal = np.dot(T,maxL_LW_fit)
	Ninv = BM.read_data_s2d(array_save_directory+'Ninv', 'Ninv')
	Ninv_maxL_LW_signal = np.dot(Ninv,maxL_LW_signal)
	ML_qbar = np.dot(T.conjugate().T,Ninv_maxL_LW_signal)
	q_sub_dbar = pre_sub_dbar-ML_qbar
	if small_cube: PSPP.dbar = q_sub_dbar
	PSPP_block_diag.dbar = q_sub_dbar
	# PSPP_block_diag.inverse_LW_power=0.0 #Remove the constraints on the LW model for subsequent parts of the analysis
	PSPP_block_diag.inverse_LW_power=p.inverse_LW_power #Remove the constraints on the LW model for subsequent parts of the analysis
	print 'Foreground pre-subtraction complete, {} orders of magnitude foreground supression achieved.\n'.format(np.log10((d-effective_noise).std()/(d-maxL_LW_signal-effective_noise).std()))
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
if not p.use_EoR_cube:
	try:
		maxL_k_cube_signal = PSPP_block_diag.calc_SigmaI_dbar_wrapper(np.array(k_sigma)**2., T_Ninv_T, dbar, block_T_Ninv_T=block_T_Ninv_T)[0]
	except:
		maxL_k_cube_signal = PSPP_block_diag.calc_SigmaI_dbar_wrapper(np.array([10.0]*nDims), T_Ninv_T, dbar, block_T_Ninv_T=block_T_Ninv_T)[0]
else:
	# maxL_k_cube_signal = PSPP_block_diag.calc_SigmaI_dbar_wrapper(np.array([10.0]*nDims)**2, T_Ninv_T, dbar, block_T_Ninv_T=block_T_Ninv_T)[0]
	maxL_k_cube_signal = PSPP_block_diag.calc_SigmaI_dbar_wrapper(np.array([10.0]*nDims), T_Ninv_T, dbar, block_T_Ninv_T=block_T_Ninv_T)[0]

maxL_f_plus_q_signal = np.dot(T,maxL_k_cube_signal)
save_path = save_dir+'Total_signal_model_fit_and_residuals_nu_{}_nv_{}_neta_{}_nq_{}_npl_{}_sigma_{:.1E}_beta_{}.png'.format(nu,nv,neta,nq,npl,sigma,p.beta)
if p.use_EoR_cube:save_path=save_path.replace('.png', '_EoR_cube.png')
if not p.include_instrumental_effects:
	plot_signal_vs_MLsignal_residuals(s_Tot, maxL_f_plus_q_signal, sigma, save_path)

LW_modes_only_boolean_array_vis_ordered_v2 = np.zeros(maxL_f_plus_q_signal.size).astype('bool')
LW_modes_only_boolean_array_vis_ordered_v2[nf-1::nf] = 1
LW_modes_only_boolean_array_vis_ordered_v2[nf-2::nf] = 1
LW_modes_only_boolean_array_vis_ordered_v2[nf/2-1::nf] = 1

if small_cube and not p.use_EoR_cube and nq==2 and not p.include_instrumental_effects:
	maxL_k_cube_LW_modes = maxL_k_cube_signal.copy()
	# maxL_k_cube_LW_modes[np.logical_not(LW_modes_only_boolean_array_vis_ordered)] = 0.0
	maxL_k_cube_LW_modes[np.logical_not(LW_modes_only_boolean_array_vis_ordered_v2)] = 0.0
	q_component_of_maxL_of_f_plus_q_signal = np.dot(T,maxL_k_cube_LW_modes)
	plot_signal_vs_MLsignal_residuals(s_fgs_only, q_component_of_maxL_of_f_plus_q_signal, sigma, save_dir+'LW_component_of_signal_model_fit_and_residuals_nu_{}_nv_{}_neta_{}_nq_{}_npl_{}_sigma_{:.1E}_beta_{}.png'.format(nu,nv,neta,nq,npl,sigma,p.beta))

	maxL_k_cube_fourier_modes = maxL_k_cube_signal.copy()
	# maxL_k_cube_fourier_modes[LW_modes_only_boolean_array_vis_ordered] = 0.0
	maxL_k_cube_fourier_modes[LW_modes_only_boolean_array_vis_ordered_v2] = 0.0
	f_component_of_maxL_of_f_plus_q_signal = np.dot(T,maxL_k_cube_fourier_modes)
	plot_signal_vs_MLsignal_residuals(s_fgs_only, f_component_of_maxL_of_f_plus_q_signal, sigma, save_dir+'Fourier_component_of_signal_model_fit_and_residuals_nu_{}_nv_{}_neta_{}_nq_{}_npl_{}_sigma_{:.1E}_beta_{}.png'.format(nu,nv,neta,nq,npl,sigma,p.beta))

No_large_spectral_scale_model_fit = False
if No_large_spectral_scale_model_fit:
	PSPP_block_diag.inverse_LW_power=1.e10
	maxL_k_cube_signal = PSPP_block_diag.calc_SigmaI_dbar_wrapper(np.array([10.0]*nDims), T_Ninv_T, dbar, block_T_Ninv_T=block_T_Ninv_T)[0]

	maxL_f_plus_q_signal = np.dot(T,maxL_k_cube_signal)
	save_path = save_dir+'Total_signal_model_fit_and_residuals_NQ.png'
	if p.use_EoR_cube:save_path=save_path.replace('.png', '_EoR_cube.png')
	plot_signal_vs_MLsignal_residuals(s-maxL_LW_signal, maxL_f_plus_q_signal, sigma, save_path)

	if small_cube and not p.use_EoR_cube and not p.include_instrumental_effects:
		maxL_k_cube_fourier_modes = maxL_k_cube_signal.copy()
		maxL_k_cube_fourier_modes[LW_modes_only_boolean_array_vis_ordered] = 0.0
		f_component_of_maxL_of_f_plus_q_signal = np.dot(T,maxL_k_cube_fourier_modes)
		plot_signal_vs_MLsignal_residuals(s_fourier_only, f_component_of_maxL_of_f_plus_q_signal, sigma, save_dir+'Fourier_component_of_signal_model_fit_and_residuals_NQ.png')

#--------------------------------------------
# Sample from the posterior
#--------------------------------------------###
# PolyChord setup
log_priors_min_max = [[-5.0, 3.0] for _ in range(nDims)]
# #Fundamental harmonic is highly correlated with the LWM so don't fit for it.
# log_priors_min_max[0] = [-20.0, -20.0] 




# log_priors_min_max[-10+1] = [-1.2, -0.4]
# log_priors_min_max[-10+2] = [-0.2, 0.3]
# log_priors_min_max[-10+3] =  [0.25, 0.75]
# log_priors_min_max[-10+4] =  [0.7, 1.25]
# log_priors_min_max[-10+5] =  [0.8, 1.25]
# log_priors_min_max[-10+6] =  [0.85, 1.3]
# log_priors_min_max[-10+7] =  [1.1, 1.5]
# log_priors_min_max[-10+8] =  [1.45, 1.8]
# log_priors_min_max[-10+9] =  [1.4, 2.1]

# log_priors_min_max[-10+1] = [-0.7, -0.65]
# log_priors_min_max[-10+2] = [-0.05, 0.05]
# log_priors_min_max[-10+3] =  [0.45, 0.50]
# log_priors_min_max[-10+4] =  [0.73, 1.80]
# log_priors_min_max[-10+5] =  [0.90, 0.95]
# log_priors_min_max[-10+6] =  [1.20, 1.25]
# log_priors_min_max[-10+7] =  [1.37, 1.42]
# log_priors_min_max[-10+8] =  [1.47, 1.52]
# log_priors_min_max[-10+9] =  [1.65, 1.70]


if p.use_LWM_Gaussian_prior:
	fg_log_priors_min = np.log10(1.e5) #Set minimum LW model priors using LW power spectrum in fit to white noise (i.e the prior min should incorporate knowledge of signal-loss in iterative pre-subtraction in order to eliminate the bias that would result from not removing this when re-fitting in the full analysis.)
	fg_log_priors_max = 6.0 #Set minimum LW model prior max using numerical stability constraint at the given signal-to-noise in the data.
	# log_priors_min_max[0] = [fg_log_priors_min, 8.0] #Set 
	log_priors_min_max[0] = [fg_log_priors_min, fg_log_priors_max] #Calibrate LW model priors using white noise fitting 
	log_priors_min_max[1] = [fg_log_priors_min, fg_log_priors_max] #Calibrate LW model priors using white noise fitting
	log_priors_min_max[2] = [fg_log_priors_min, fg_log_priors_max] #Calibrate LW model priors using white noise fitting
	if p.use_intrinsic_noise_fitting:
		log_priors_min_max[1] = log_priors_min_max[0]
		log_priors_min_max[2] = log_priors_min_max[1]
		log_priors_min_max[3] = log_priors_min_max[2]
		log_priors_min_max[0] = [1.0, 2.0] #Linear alpha_prime range
else:
	if p.fit_for_spectral_model_parameters:
		if p.use_intrinsic_noise_fitting:
				log_priors_min_max[0] = [0.30, 4.70] #Linear b1 range
				log_priors_min_max[1] = [0.0, 1.0] # b2 = b1(p0) + (5.25-b1(p0))*p1
				log_priors_min_max[2] = [1.0, 2.0] #Linear alpha_prime range
		else:
			# b1_min = (p.pl_grid_spacing/2.)+0.05 #+0.05 is to avoid rounding to below the model range
			b1_min = p.pl_min #+0.05 is to avoid rounding to below the model range
			b1_max = p.pl_max-p.pl_grid_spacing-0.05 #-0.05 is to avoid rounding to above the model range
			log_priors_min_max[0] = [b1_min, b1_max] #Linear b1 range 
			log_priors_min_max[1] = [0.0, 1.0] # b2 = b1(p0) + (b1_max-b1(p0))*p1
	else:
		if p.use_intrinsic_noise_fitting:
				log_priors_min_max[0] = [1.0, 2.0] #Linear alpha_prime range

	

print 'log_priors_min_max', log_priors_min_max

prior_c = PriorC(log_priors_min_max)
nDerived = 0
nDims = len(k_cube_voxels_in_bin)

if p.use_intrinsic_noise_fitting:
	nDims = nDims+1

if p.fit_for_spectral_model_parameters:
	nDims = nDims+2

###
# nDims = nDims+3 for Gaussian prior over the three long wavelength model vectors
###
if p.use_LWM_Gaussian_prior:
	nDims = nDims+3


outputfiles_base_dir = 'chains/'
base_dir = outputfiles_base_dir+'clusters/'
if not os.path.isdir(base_dir):
		os.makedirs(base_dir)

log_priors = True
dimensionless_PS = True
if overwrite_data_with_WN:
	dimensionless_PS = False
# dimensionless_PS = False
# zero_the_LW_modes = True
zero_the_LW_modes = False

file_root = 'Test-{}_{}_{}_{}_{}_s_{:.1E}-lp_F-dPS_F-'.format(nu,nv,neta,nq,npl,sigma).replace('.','d')
# file_root = 'Test-nu_{}_nv_{}_neta_{}_nq_{}_npl_{}_sigma_{:.1E}-lp_F-dPS_F-v1-'.format(nu,nv,neta,nq,npl,sigma).replace('.','d')
# file_root = 'Test-sigma_{:.1E}-lp_F-dPS_F-v1-'.format(sigma).replace('.','d')
if chan_selection!='':file_root=chan_selection+file_root
if npl==1:
	file_root=file_root.replace('-dPS_F', '-dPS_F-beta_{:.2E}-v1'.format(p.beta))
if npl==2:
	file_root=file_root.replace('-dPS_F', '-dPS_F_b1_{:.2F}_b2_{:.2F}-v1'.format(p.beta[0], p.beta[1]))
if log_priors:
	file_root=file_root.replace('lp_F', 'lp_T')
if dimensionless_PS:
	file_root=file_root.replace('dPS_F', 'dPS_T')
if nq==0:
	file_root=file_root.replace('mini-', 'mini-NQ-')
elif zero_the_LW_modes:
	file_root=file_root.replace('mini-', 'mini-ZLWM-')
if p.use_EoR_cube:
	file_root=file_root.replace('Test', 'EoR')
if use_MultiNest:
	file_root='MN-'+file_root

file_root = generate_output_file_base(file_root, version_number='1')

# file_root = 'MN-EoR-9_9_38_2_2_s_8d5E+04-lp_T-dPS_T_b1_2.63_b2_2.82-v5-'
# file_root = 'MN-EoR-9_9_38_2_2_s_8d5E+04-lp_T-dPS_T_b1_2.63_b2_2.82-v6-'

print 'Rank', mpi_rank
print 'Output file_root = ', file_root

PSPP_block_diag_Polychord = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, Ninv, d_Ninv_d, block_T_Ninv_T=block_T_Ninv_T, log_priors=log_priors, dimensionless_PS=dimensionless_PS, Print=True, intrinsic_noise_fitting=p.use_intrinsic_noise_fitting,fit_for_spectral_model_parameters=p.fit_for_spectral_model_parameters)
if p.include_instrumental_effects and not zero_the_LW_modes:
	PSPP_block_diag_Polychord.inverse_LW_power=p.inverse_LW_power #Include minimal prior over LW modes required for numerical stability
if zero_the_LW_modes:
	PSPP_block_diag_Polychord.inverse_LW_power=1.e20
	print 'Setting PSPP_block_diag_Polychord.inverse_LW_power to:', PSPP_block_diag_Polychord.inverse_LW_power
if sub_MLLWM: PSPP_block_diag_Polychord.dbar = q_sub_dbar
if sub_ML_monopole_term_model: PSPP_block_diag_Polychord.dbar = q_sub_dbar
if sub_ML_monopole_plus_first_LW_term_model: PSPP_block_diag_Polychord.dbar = q_sub_dbar
if p.use_intrinsic_noise_fitting and (sub_MLLWM or sub_ML_monopole_term_model or sub_ML_monopole_plus_first_LW_term_model):
	print 'Using use_intrinsic_noise_fitting'
	PSPP_block_diag_Polychord.d_Ninv_d = q_sub_d_Ninv_q_sub_d


run_limit_foreground_modes_test = False
if run_limit_foreground_modes_test:
	PSPP_block_diag_Polychord.inverse_LW_power=0.0
	PSPP_block_diag_Polychord.inverse_LW_power_zeroth_LW_term=2.e-20
	PSPP_block_diag_Polychord.inverse_LW_power_first_LW_term=2.e-20  
	PSPP_block_diag_Polychord.inverse_LW_power_second_LW_term=2.e20  



start = time.time()
PSPP_block_diag_Polychord.Print=False
Nit=20
for _ in range(Nit):
	L =  PSPP_block_diag_Polychord.posterior_probability([1.e0]*nDims)[0]
PSPP_block_diag_Polychord.Print=False
print 'Average evaluation time: %f'%((time.time()-start)/float(Nit))

print 'PSPP_block_diag_Polychord.inverse_LW_power', PSPP_block_diag_Polychord.inverse_LW_power
print 'p.Tb_experimental_std_K', p.Tb_experimental_std_K

def likelihood(theta, calc_likelihood=PSPP_block_diag_Polychord.posterior_probability):
	return calc_likelihood(theta)

def MultiNest_likelihood(theta, calc_likelihood=PSPP_block_diag_Polychord.posterior_probability):
	return calc_likelihood(theta)[0]


run_single_node_analysis = False
print 'mpi_size =', mpi_size
if mpi_size>1:
	print 'mpi_size greater than 1, running multi-node analysis\n'
else:
	print 'mpi_size = {}, analysis will only be run if run_single_node_analysis is set to True'.format(mpi_size)
	print 'run_single_node_analysis = {}\n'.format(run_single_node_analysis)

if run_single_node_analysis or mpi_size>1:
	if use_MultiNest:
		MN_nlive = nDims*25
		const_efficiency_mode = False #Use for exploring a slowly converging posterior (should be False for reliable evidence estimation).
		importance_nested_sampling = False
		# Run MultiNest
		result = solve(LogLikelihood=MultiNest_likelihood, Prior=prior_c.prior_func, n_dims=nDims, outputfiles_basename=outputfiles_base_dir+file_root, n_live_points=MN_nlive, const_efficiency_mode=const_efficiency_mode, importance_nested_sampling=importance_nested_sampling)
	else:
		precision_criterion = 0.05
		#precision_criterion = 0.001 #PolyChord default value
		nlive=nDims*10
		#nlive=nDims*25 #PolyChord default value
		# Run PolyChord
		PolyChord.mpi_notification()
		PolyChord.run_nested_sampling(PSPP_block_diag_Polychord.posterior_probability, nDims, nDerived, file_root=file_root, read_resume=False, prior=prior_c.prior_func, precision_criterion=precision_criterion, nlive=nlive)

	print 'Sampling complete!'
else:
	print 'Skipping sampling, exiting...'
#######################




# from scipy import sparse
# a = sparse.diags(np.arange(10))
# b = a * np.ones([10,10])
# print b




