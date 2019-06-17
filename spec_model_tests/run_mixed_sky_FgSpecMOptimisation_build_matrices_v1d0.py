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

mpi_rank = 0
run_full_analysis = False #False skips mpi and other imports that can cause crashes in ipython (note: in ipython apparently __name__ == '__main__' which is why this if statement is here instead)
# run_full_analysis = True #When running an analysis this should be True.
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

p.beta = [2.0,3.0]

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
		# sigma = sigma*average_baseline_redundancy**0.5 *2000.0
else:
	sigma = sigma*1.

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
array_save_directory = 'array_storage/FgSpecMOptimisation/{}_nu_{}_nv_{}_neta_{}_nq_{}_npl_{}_sigma_{:.1E}/'.format(current_file_version,nu,nv,neta,nq,npl,sigma).replace('.','d')
if p.include_instrumental_effects:
	instrument_info = filter(None, p.instrument_model_directory.split('/'))[-1]
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




T_Ninv_T = BM.read_data_from_hdf5(array_save_directory+'T_Ninv_T.h5', 'T_Ninv_T')
Npar = shape(T_Ninv_T)[0]
fit_for_LW_power_spectrum = True
masked_power_spectral_modes = np.ones(Npar)
if not fit_for_LW_power_spectrum:
	print 'Not fitting for LW power spectrum. Zeroing relevant modes in the determinant.'
	masked_power_spectral_modes[sorted(np.hstack(k_cube_voxels_in_bin)[0])] = 0.0

masked_power_spectral_modes = masked_power_spectral_modes.astype('bool')
T = BM.read_data_from_hdf5(array_save_directory+'T.h5', 'T')
Finv = BM.read_data_from_hdf5(array_save_directory+'Finv.h5', 'Finv')

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
T = BM.read_data_from_hdf5(array_save_directory+'T.h5', 'T')
T_Ninv_T = BM.read_data_from_hdf5(array_save_directory+'T_Ninv_T.h5', 'T_Ninv_T')
block_T_Ninv_T = BM.read_data_from_hdf5(array_save_directory+'block_T_Ninv_T.h5', 'block_T_Ninv_T')
Ninv = BM.read_data_from_hdf5(array_save_directory+'Ninv.h5', 'Ninv')
Ninv_d = np.dot(Ninv,d)
dbar = np.dot(T.conjugate().T,Ninv_d)
# Ninv=[]
Sigma_Diag_Indices=np.diag_indices(shape(T_Ninv_T)[0])
nDims = len(k_cube_voxels_in_bin)
d_Ninv_d = np.dot(d.conjugate(), Ninv_d)

if p.use_intrinsic_noise_fitting:
	nDims = nDims+1

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




###
# build new T
###




gridding_matrix_vis_ordered_to_chan_ordered = BM.read_data_from_hdf5(array_save_directory+'gridding_matrix_vis_ordered_to_chan_ordered.h5', 'gridding_matrix_vis_ordered_to_chan_ordered')
Fprime = BM.read_data_from_hdf5(array_save_directory+'Fprime.h5', 'Fprime')
Finv = BM.read_data_from_hdf5(array_save_directory+'Finv.h5', 'Finv')
Ninv = BM.read_data_from_hdf5(array_save_directory+'Ninv.h5', 'Ninv')

from BayesEoR.Linalg import IDFT_Array_IDFT_1D_WQ
Fz_normalisation = nf**0.5

# multi_vis_idft_array_1D_WQ=IDFT_Array_IDFT_1D_WQ(nf, neta, nq, npl=npl, nu_min_MHz=p.nu_min_MHz, channel_width_MHz=p.channel_width_MHz, beta=p.beta)*Fz_normalisation
# multi_vis_idft_array_1D_WQ = block_diag(*[idft_array_1D_WQ.T for i in range(nu*nv-1)])
# Fz = np.dot(gridding_matrix_vis_ordered_to_chan_ordered,multi_vis_idft_array_1D_WQ)
# Fprime_Fz = np.dot(Fprime,Fz)
# T = np.dot(Finv,Fprime_Fz)
# T_Ninv_T = np.dot(T.conjugate().T, np.dot(Ninv, T))
# matrix_name='T_Ninv_T_b1_{}_b2_{}'.format(np.round(p.beta[0],2),np.round(p.beta[1],2)).replace('.','d')
# BM.output_to_hdf5(idft_array_1D_WQ, array_save_directory, matrix_name+'.h5', matrix_name)


import time

# dbeta = 0.5
dbeta = 0.1
# beta_min = 0.5
# beta_max = 5.0
beta_min = 2.0
beta_max = 3.0
n_samples = int((beta_max-beta_min)/dbeta)
beta_samples = beta_min + np.arange(n_samples+1)*dbeta
start = time.time()

# beta_samples = beta_samples[:2]

for b1 in beta_samples:
	for b2 in beta_samples:
		if b1<b2:
			p.beta = [b1,b2]
			print 'p.beta', p.beta
			print 'Time taken: {}'.format(time.time()-start) 

			idft_array_1D_WQ=IDFT_Array_IDFT_1D_WQ(nf, neta, nq, npl=npl, nu_min_MHz=p.nu_min_MHz, channel_width_MHz=p.channel_width_MHz, beta=p.beta)*Fz_normalisation
			multi_vis_idft_array_1D_WQ = block_diag(*[idft_array_1D_WQ.T for i in range(nu*nv-1)])
			Fz = np.dot(gridding_matrix_vis_ordered_to_chan_ordered,multi_vis_idft_array_1D_WQ)
			Fprime_Fz = np.dot(Fprime,Fz)
			T = np.dot(Finv,Fprime_Fz)
			
			T_Ninv_T = np.dot(T.conjugate().T, np.dot(Ninv, T))
			matrix_name='T_Ninv_T_b1_{}_b2_{}'.format(np.round(p.beta[0],2),np.round(p.beta[1],2)).replace('.','d')
			BM.output_to_hdf5(T_Ninv_T, array_save_directory, matrix_name+'.h5', matrix_name)
			print 'Outputting matrix to:', array_save_directory+matrix_name+'.h5'

			dbar = np.dot(T.conjugate().T,Ninv_d)
			matrix_name='dbar_b1_{}_b2_{}'.format(np.round(p.beta[0],2),np.round(p.beta[1],2)).replace('.','d')
			BM.output_to_hdf5(dbar, array_save_directory, matrix_name+'.h5', matrix_name)
			print 'Outputting matrix to:', array_save_directory+matrix_name+'.h5'



















