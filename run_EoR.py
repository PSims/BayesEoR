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
if args.beta: p.beta = float(args.beta) #Overwrite parameter file beta with value chosen from the command line if it is included 

nq = 2 #Overwrite PCLA selection
npl = 0 
print 'nq', nq
sub_ML_monopole_plus_first_LW_term_model = True #Improve numerical precision. Can be used for improving numerical precision when performing evidence comparison.

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
overwrite_existing_matrix_stack = False
proceed_without_overwrite_confirmation = True #Allows overwrite_existing_matrix_stack to be run without having to manually accept the deletion of the old matrix stack
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
# Load EoR data
#--------------------------------------------
use_EoR_cube = True
if use_EoR_cube:
	print 'Using use_EoR_cube data'
	s_EoR, abc, scidata1 = generate_data_from_loaded_EoR_cube_v2d0(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show,chan_selection,p.EoR_npz_path_sc)

#--------------------------------------------
# Define data vector
#--------------------------------------------
if use_EoR_cube:
	print 'Using EoR cube'
	d = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector(sigma, s_EoR)[0]
	s_Tot = s_EoR.copy()

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

from numpy import real
#--------------------------------------------
# Instantiate class and check that posterior_probability returns a finite probability (so no obvious binning errors etc.)
#--------------------------------------------
if small_cube: PSPP = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list)
PSPP_block_diag = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, block_T_Ninv_T=block_T_Ninv_T, Print=True)
self = PowerSpectrumPosteriorProbability(T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, block_T_Ninv_T=block_T_Ninv_T, Print=True)
start = time.time()

if small_cube:
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
# Sample from the posterior
#--------------------------------------------###
# PolyChord setup
log_priors_min_max = [[-5.0, 4.0] for _ in range(nDims)]
prior_c = PriorC(log_priors_min_max)
nDerived = 0
nDims = len(k_cube_voxels_in_bin)
outputfiles_base_dir = 'chains/'
base_dir = outputfiles_base_dir+'clusters/'
if not os.path.isdir(base_dir):
		os.makedirs(base_dir)

log_priors = True
dimensionless_PS = True
zero_the_LW_modes = False

file_root = 'Test_mini-{}_{}_{}_{}_{}_sigma_{:.1E}-lp_F-dPS_F-v1-'.format(nu,nv,neta,nq,npl,sigma).replace('.','d')
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











