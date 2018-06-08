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
argv = sys.argv[1:]
PCLA = ParseCommandLineArguments()
nq = PCLA.parse(argv) #defaults to 2 (i.e. by default we jointly fit for second order quadratics)
nq = 2 #Overwrite PCLA selection
npl = 0 #Overwrites quadratic term when nq=2, otherwise unused.
sub_MLLWM = True #Improve numerical precision
# Cube size
nf=p.nf
neta=p.neta
neta=neta -nq
nu=p.nu
nv=p.nv
nx=p.nx
ny=p.ny
# Data noise
sigma=20.e-1
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
array_save_directory = 'array_storage/{}_{}_{}_{}_{}_{:.1E}/'.format(current_file_version,nu,nv,neta,nq,sigma).replace('.','d')

#--------------------------------------------
# Construct matrices
#--------------------------------------------
BM = BuildMatrices(array_save_directory, nu, nv, nx, ny, neta, nf, nq, sigma, npl=npl)
overwrite_existing_matrix_stack = False #Can be set to False unless npl>0
# overwrite_existing_matrix_stack = True #Can be set to False unless npl>0
BM.build_minimum_sufficient_matrix_stack(overwrite_existing_matrix_stack=overwrite_existing_matrix_stack)

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
	ZM_mask = test_sim_out[	10]
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
use_foreground_cube = False

if use_foreground_cube:
	###
	beta_experimental_mean = 2.63+0   #Matches beta_150_408 in Mozden, Bowman et al. 2016
	beta_experimental_std  = 0.02      #A conservative over-estimate of the dbeta_150_408=0.01 (dbeta_90_190=0.02) in Mozden, Bowman et al. 2016
	gamma_mean             = -2.7     #Revise to match published values
	gamma_sigma            = 0.3      #Revise to match published values
	Tb_experimental_mean_K = 194.0    #Matches GSM mean in region A
	Tb_experimental_std_K  = 62.0     #70th percentile 12 deg.**2 region at 56 arcmin res. centered on -30. deg declination (see GSM_map_std_at_-30_dec_v1d0.ipynb)
	# Tb_experimental_std_K  = 62.0   #Median std at at 0.333 degree resolution in 50 deg by 50 deg maps centered on Dec=-30.0
	nu_min_MHz             = 163.0-4.0
	Tb_experimental_std_K = Tb_experimental_std_K*(nu_min_MHz/163.)**-beta_experimental_mean
	channel_width_MHz      = 0.2
	simulation_FoV_deg = 12.0             #Matches EoR simulation
	simulation_resolution_deg = simulation_FoV_deg/511. #Matches EoR sim (note: use closest odd val., so 127 rather than 128, for easier FFT normalisation)
	# fits_storage_dir = 'fits_storage/multi_frequency_band_pythonPStest1/Jelic_nu_min_MHz_{}_TbStd_{}_beta_{}_dbeta{}/'.format(nu_min_MHz, Tb_experimental_std_K, beta_experimental_mean, beta_experimental_std).replace('.','d')
	fits_storage_dir = 'fits_storage/free_free_emission/Jelic_nu_min_MHz_{}_TbStd_{}_beta_{}_dbeta{}/'.format(nu_min_MHz, Tb_experimental_std_K, beta_experimental_mean, beta_experimental_std).replace('.','d')
	# HF_nu_min_MHz_array = [210,220,230]
	HF_nu_min_MHz_array = [210]
	foreground_outputs = generate_Jelic_cube(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show, beta_experimental_mean,beta_experimental_std,gamma_mean,gamma_sigma,Tb_experimental_mean_K,Tb_experimental_std_K,nu_min_MHz,channel_width_MHz, generate_additional_extrapolated_HF_foreground_cube=True, fits_storage_dir=fits_storage_dir, HF_nu_min_MHz_array=HF_nu_min_MHz_array, simulation_FoV_deg=simulation_FoV_deg, simulation_resolution_deg=simulation_resolution_deg,random_seed=314211)
	fg, s, Tb_nu, beta_experimental_mean,beta_experimental_std,gamma_mean,gamma_sigma,Tb_experimental_mean_K,Tb_experimental_std_K,nu_min_MHz,channel_width_MHz, HF_Tb_nu = foreground_outputs
	foreground_outputs = []

	pylab.close('all')
	pylab.imshow(Tb_nu[0])
	pylab.show() 




	
foreground_outputs = generate_Jelic_cube(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show, p.beta_experimental_mean,p.beta_experimental_std,p.gamma_mean,p.gamma_sigma,p.Tb_experimental_mean_K,p.Tb_experimental_std_K,p.nu_min_MHz,p.channel_width_MHz, generate_additional_extrapolated_HF_foreground_cube=True, fits_storage_dir=p.fits_storage_dir, HF_nu_min_MHz_array=p.HF_nu_min_MHz_array, simulation_FoV_deg=p.simulation_FoV_deg, simulation_resolution_deg=p.simulation_resolution_deg,random_seed=314211)

fg, s, Tb_nu, beta_experimental_mean,beta_experimental_std,gamma_mean,gamma_sigma,Tb_experimental_mean_K,Tb_experimental_std_K,nu_min_MHz,channel_width_MHz, HF_Tb_nu = foreground_outputs
foreground_outputs = []


construct_aplpy_image_from_fits('/gpfs/data/jpober/psims/EoR/Python_Scripts/BayesEoR/git_version/BayesEoR/spec_model_tests/fits_storage/multi_frequency_band_pythonPStest1/Jelic_nu_min_MHz_159d0_TbStd_66d1866884116_beta_2d63_dbeta0d02/', 'Jelic_GDSE_cube_159MHz_mK', run_convert_from_mK_to_K=True, run_remove_unused_header_variables=True)



foreground_outputs = generate_Jelic_cube(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show, p.beta_experimental_mean_ff,p.beta_experimental_std_ff,p.gamma_mean_ff,p.gamma_sigma_ff,p.Tb_experimental_mean_K_ff,p.Tb_experimental_std_K_ff,p.nu_min_MHz_ff,p.channel_width_MHz_ff, generate_additional_extrapolated_HF_foreground_cube=True, fits_storage_dir=p.fits_storage_dir_ff, HF_nu_min_MHz_array=p.HF_nu_min_MHz_array_ff, simulation_FoV_deg=p.simulation_FoV_deg_ff, simulation_resolution_deg=p.simulation_resolution_deg_ff,random_seed=3142111)


construct_aplpy_image_from_fits('/gpfs/data/jpober/psims/EoR/Python_Scripts/BayesEoR/git_version/BayesEoR/spec_model_tests/fits_storage/free_free_emission/Free_free_nu_min_MHz_159d0_TbStd_0d698184469839_beta_2d15_dbeta1e-10/', 'Jelic_GDSE_cube_159MHz_mK', run_convert_from_mK_to_K=True, run_remove_unused_header_variables=True)




dat = numpy.load('21cm_mK_z7.600_nf0.459_useTs0.0_aveTb9.48_cube_side_pix512_cube_side_Mpc2048.npz')['arr_0']
# dat = numpy.load('21cm_mK_z10.000_nf0.782_useTs0.0_aveTb20.93_cube_side_pix512_cube_side_Mpc2048.npz')['arr_0']

nf = 38
x = arange(nf)
d = dat[:,:,0:38][0,0,:]
m = np.polyval(np.polyfit(x, d, 2), x)
m0 = np.polyval(np.polyfit(x, d, 2)[:-1], x)

pylab.errorbar(x, d)
pylab.errorbar(x, m)
pylab.show()

pylab.errorbar(x, d-d.mean())
pylab.errorbar(x, m0)
pylab.show()



axes_tuple = (0,1)
fft_dat=numpy.fft.ifftshift(dat+0j, axes=axes_tuple)
fft_dat=numpy.fft.fftn(fft_dat, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
fft_dat=numpy.fft.fftshift(fft_dat, axes=axes_tuple)

x = arange(nf)
d = fft_dat[:,:,0:nf][0,1,:]
m_r = np.polyval(np.polyfit(x, d.real, 2), x)
m_i = np.polyval(np.polyfit(x, d.imag, 2), x)

pylab.errorbar(x, d.real)
pylab.errorbar(x, m_r)
pylab.errorbar(x, d.imag)
pylab.errorbar(x, m_i)
pylab.show()


pylab.figure()
pylab.errorbar(x, d.real)
pylab.errorbar(x, m_r)
pylab.errorbar(x, d.imag)
pylab.errorbar(x, m_i)

pylab.figure()
pylab.errorbar(x, d.real - m_r)
pylab.errorbar(x, d.imag - m_i)
pylab.show()




fft_dat_uv_sub = fft_dat[256-15:256+16,256-15:256+16,:]


q_comp_subset_sums_r = []
q_comp_subset_sums_i = []
for sub_i in range(400):
	if sub_i%10==0: print sub_i
	quad_params_r = []
	quad_params_i = []
	for i in range(fft_dat_uv_sub.shape[0]):
		# if i%10==0: print i
		for j in range(fft_dat_uv_sub.shape[1]):
			d = fft_dat_uv_sub[i,j,0+sub_i:nf+sub_i]
			q_r = np.polyfit(x, d.real, 2)
			q_i = np.polyfit(x, d.imag, 2)
			m_r = np.polyval(q_r, x)
			m_i = np.polyval(q_i, x)
			quad_params_r.append(q_r)
			quad_params_i.append(q_i)
	q_comp_subset_sums_r.append(np.sum(abs(np.array(quad_params_r)), axis=0))
	q_comp_subset_sums_i.append(np.sum(abs(np.array(quad_params_i)), axis=0))


print [np.array(q_comp_subset_sums_r)[:,i].min() for i in range(3)]
print [np.array(q_comp_subset_sums_r)[:,i].max() for i in range(3)]

print [np.array(q_comp_subset_sums_r)[:,i].argmin() for i in range(3)]
print [np.array(q_comp_subset_sums_r)[:,i].argmax() for i in range(3)]






#--------------------------------------------
# Load EoR data
#--------------------------------------------
use_EoR_cube = True
if use_EoR_cube:
	print 'Using use_EoR_cube data'
	s, abc, scidata1 = generate_data_from_loaded_EoR_cube_v2d0(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show,chan_selection,p.EoR_npz_path_sc)

#--------------------------------------------
# Define data vector
#--------------------------------------------
if use_EoR_cube:
	d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector(sigma, abc)[0]
	if use_foreground_cube:
		d += generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector(sigma, fg)[0]
elif use_foreground_cube:
	d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector(sigma, fg)[0]
else:
	d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector(sigma, s)[0]

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
	maxL_LW_fit = PSPP_block_diag.calc_SigmaI_dbar_wrapper([1.e-20]*nDims, T_Ninv_T, dbar, block_T_Ninv_T=block_T_Ninv_T)[0]
	maxL_LW_signal = np.dot(T,maxL_LW_fit)
	Ninv = BM.read_data_from_hdf5(array_save_directory+'Ninv.h5', 'Ninv')
	Ninv_maxL_LW_signal = np.dot(Ninv,maxL_LW_signal)
	ML_qbar = np.dot(T.conjugate().T,Ninv_maxL_LW_signal)
	q_sub_dbar = dbar-ML_qbar
	if small_cube: PSPP.dbar = q_sub_dbar
	PSPP_block_diag.dbar = q_sub_dbar
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
save_path = save_dir+'Total_signal_model_fit_and_residuals.png'
if use_EoR_cube:save_path=save_path.replace('.png', '_EoR_cube.png')
plot_signal_vs_MLsignal_residuals(s, maxL_f_plus_q_signal, sigma, save_path)

if small_cube and not use_EoR_cube and nq==2:
	maxL_k_cube_LW_modes = maxL_k_cube_signal.copy()
	maxL_k_cube_LW_modes[np.logical_not(LW_modes_only_boolean_array_vis_ordered)] = 0.0
	q_component_of_maxL_of_f_plus_q_signal = np.dot(T,maxL_k_cube_LW_modes)
	plot_signal_vs_MLsignal_residuals(s_LW_only, q_component_of_maxL_of_f_plus_q_signal, sigma, save_dir+'LW_component_of_signal_model_fit_and_residuals.png')

	maxL_k_cube_fourier_modes = maxL_k_cube_signal.copy()
	maxL_k_cube_fourier_modes[LW_modes_only_boolean_array_vis_ordered] = 0.0
	f_component_of_maxL_of_f_plus_q_signal = np.dot(T,maxL_k_cube_fourier_modes)
	plot_signal_vs_MLsignal_residuals(s_fourier_only, f_component_of_maxL_of_f_plus_q_signal, sigma, save_dir+'Fourier_component_of_signal_model_fit_and_residuals.png')

No_large_spectral_scale_model_fit = False
if No_large_spectral_scale_model_fit:
	PSPP_block_diag.inverse_LWratic_power=1.e10
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
prior_c = PriorC(log_priors_min_max)
nDerived = 0
nDims = len(k_cube_voxels_in_bin)
base_dir = 'chains/clusters/'
if not os.path.isdir(base_dir):
		os.makedirs(base_dir)

log_priors = True
dimensionless_PS = True
zero_the_LW_modes = False
file_root = 'Test_mini-sigma_{:.1E}-lp_F-dPS_F-v1-'.format(sigma).replace('.','d')
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

file_root = generate_output_file_base(file_root, version_number='1')
print 'Output file_root = ', file_root

PSPP_block_diag_Polychord = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, block_T_Ninv_T=block_T_Ninv_T, log_priors=log_priors, dimensionless_PS=dimensionless_PS, Print=True)
if zero_the_LW_modes: PSPP_block_diag_Polychord.inverse_LW_power=1.e20
if sub_MLLWM: PSPP_block_diag_Polychord.dbar = q_sub_dbar

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
	result = solve(LogLikelihood=MultiNest_likelihood, Prior=prior_c.prior_func, n_dims=nDims, outputfiles_basename="chains/"+file_root, n_live_points=MN_nlive)
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







