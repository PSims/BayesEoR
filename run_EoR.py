#!/users/psims/anaconda2/bin/python

#--------------------------------------------
# Imports
#--------------------------------------------
from subprocess import os
import sys
head,tail = os.path.split(os.getcwd())
sys.path.append(head)
from BayesEoR import * #Make everything available for now, this can be refined later
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
# nq = 2 #Overwrite PCLA selection
sub_MLLWM = True #Improve numerical precision
# Cube size
nf=128
neta=128
neta = neta -nq
nu=3
nv=3
nx=3
ny=3
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
BM = BuildMatrices(array_save_directory, nu, nv, nx, ny, neta, nf, nq, sigma)
overwrite_existing_matrix_stack = True #Can be set to False unless npl>0
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

k = generate_k_cube_in_physical_coordinates_21cmFAST(nu,nv,nx,ny,nf,neta)[0]
k_vis_ordered = k.T.flatten()
modk_vis_ordered_list = [k_vis_ordered[bin_selector.T.flatten()] for bin_selector in bin_selector_cube_ordered_list]

mod_k, k_x, k_y, k_z, deltakperp, deltakpara, x, y, z = generate_k_cube_in_physical_coordinates_21cmFAST(nu,nv,nx,ny,nf,neta)
k_x_masked = generate_masked_coordinate_cubes(k_x, nu,nv,nx,ny,nf,neta,nq)
k_y_masked = generate_masked_coordinate_cubes(k_y, nu,nv,nx,ny,nf,neta,nq)
k_z_masked = generate_masked_coordinate_cubes(k_z, nu,nv,nx,ny,nf,neta,nq)
mod_k_masked = generate_masked_coordinate_cubes(mod_k, nu,nv,nx,ny,nf,neta,nq)

k_cube_voxels_in_bin, modkbins_containing_voxels = generate_k_cube_model_spherical_binning(mod_k_masked, k_z_masked, nu,nv,nx,ny,nf,neta,nq)
k_vals = calc_mean_binned_k_vals(mod_k_masked, k_cube_voxels_in_bin, save_k_vals=True, k_vals_file='k_vals_nf128.txt')

do_cylindrical_binning = False
if do_cylindrical_binning:
	n_k_perp_bins=2
	k_cube_voxels_in_bin, modkbins_containing_voxels, k_perp_bins = generate_k_cube_model_cylindrical_binning(mod_k_masked, k_z_masked, k_y_masked, k_x_masked, n_k_perp_bins, nu,nv,nx,ny,nf,neta,nq)

#--------------------------------------------
# Load EoR data
#--------------------------------------------
use_EoR_cube = True
if use_EoR_cube:
	print 'Using use_EoR_cube data'
	s, abc, scidata1 = generate_data_from_loaded_EoR_cube(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show,chan_selection)

#--------------------------------------------
# Define data vector
#--------------------------------------------
d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector(sigma, abc)[0]

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
x=[100.e0]*len(bin_selector_vis_ordered_list)
nuv = (nu*nv-1)
block_T_Ninv_T = np.array([np.hsplit(block,nuv) for block in np.vsplit(T_Ninv_T,nuv)])
if nq==0:
	masked_power_spectral_modes = np.logical_not(np.logical_or.reduce(bin_selector_in_model_mask_vis_ordered).reshape(-1,neta)).flatten()
else:
	masked_power_spectral_modes = LW_modes_only_boolean_array_vis_ordered
from numpy import real

#--------------------------------------------
# Instantiate class and check that posterior_probability returns a finite probability (so no obvious binning errors etc.)
#--------------------------------------------
if small_cube: PSPP = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list)
PSPP_block_diag = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, block_T_Ninv_T=block_T_Ninv_T, Print=True)
self = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, block_T_Ninv_T=block_T_Ninv_T, Print=True)
start = time.time()
if small_cube:
	print PSPP.posterior_probability([1.e0]*len(bin_selector_vis_ordered_list), diagonal_sigma=False)[0]
	print PSPP_block_diag.posterior_probability([1.e0]*len(bin_selector_vis_ordered_list), diagonal_sigma=False, block_T_Ninv_T=block_T_Ninv_T)[0]

if sub_MLLWM:
	maxL_LW_fit = PSPP_block_diag.calc_SigmaI_dbar_wrapper([1.e-20]*len(bin_selector_vis_ordered_list), T_Ninv_T, dbar, block_T_Ninv_T=block_T_Ninv_T)[0]
	maxL_LW_signal = np.dot(T,maxL_LW_fit)
	Ninv = BM.read_data_from_hdf5(array_save_directory+'Ninv.h5', 'Ninv')
	Ninv_maxL_LW_signal = np.dot(Ninv,maxL_LW_signal)
	ML_qbar = np.dot(T.conjugate().T,Ninv_maxL_LW_signal)
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


#--------------------------------------------
# Sample from the posterior
#--------------------------------------------###
# PolyChord setup
log_priors_min_max = [[-5.0, 2.0] for _ in range(len(bin_selector_vis_ordered_list))]
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
if zero_the_LW_modes or nq==0:
	file_root=file_root.replace('mini-', 'mini-NQ-')
if use_EoR_cube:
	file_root=file_root.replace('Test_mini', 'EoR_mini')
if use_MultiNest:
	file_root='MN-'+file_root

file_root = generate_output_file_base(file_root, version_number='1')
print 'Output file_root = ', file_root

PSPP_block_diag_Polychord = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, block_T_Ninv_T=block_T_Ninv_T, log_priors=log_priors, dimensionless_PS=dimensionless_PS, Print=True)
if zero_the_LW_modes: PSPP_block_diag_Polychord.inverse_LW_power=1.e10
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







