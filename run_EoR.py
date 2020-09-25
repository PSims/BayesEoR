#--------------------------------------------
# Imports
#--------------------------------------------
# Make everything available for now, this can be refined later
from BayesEoR import *
import BayesEoR.Params.params as p

import sys
import os
import time
import ast
import numpy as np
# head, tail = os.path.split(os.path.split(os.getcwd())[0])
# sys.path.append(head)


# If False, skip mpi and other imports that can cause crashes in ipython
# When running an analysis this should be True
run_full_analysis = True
if run_full_analysis:
    import mpi4py
    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    print('\nmpi_rank: {}'.format(mpi_rank))
    print('mpi_size: {}\n'.format(mpi_size))
    use_MultiNest = True # Set to false for large parameter spaces
    if use_MultiNest:
        from pymultinest.solve import solve
    else:
        import PyPolyChord.PyPolyChord as PolyChord
else:
    mpi_rank = 0

#--------------------------------------------
# Set analysis parameters
#--------------------------------------------
# Model Params
update_params_with_command_line_arguments()
if p.beam_center is not None:
    p.beam_center = ast.literal_eval(p.beam_center)
    print('Beam center = {} (type {})'.format(p.beam_center, type(p.beam_center)))
npl = p.npl
nq = p.nq
if nq > npl:
    nq = npl

# Improve numerical precision.
# Can be used for improving numerical precision when
# performing evidence comparison.
sub_ML_monopole_term_model = False
nf = p.nf
neta = p.neta
if not p.include_instrumental_effects:
    neta = neta - nq
if not p.npix == -1:
    nu = p.npix
    nv = p.npix
    nx = p.npix
    ny = p.npix
    p.nu = p.npix
    p.nv = p.npix
    p.nx = p.npix
    p.ny = p.npix
else:
    nu = p.nu
    nv = p.nv
    nx = p.nx
    ny = p.ny

# FoV can now be passed as a command line argument
p.uv_pixel_width_wavelengths = 1.0 / np.deg2rad(p.simulation_FoV_deg)

# Temporary healpix params and imports
p.sky_model_pixel_area_sr = 4 * np.pi / (12 * p.nside**2)
# if p.simulation_FoV_deg == 12.9080728652 / 2:
# 	# Temporary fix for FoV / 2 test
# 	p.sky_model_pixel_area_sr /= 2.0**2
# Temprary fix for 3 FoV value tests
# if p.simulation_FoV_deg == 3.2270182163:
# 	p.sky_model_pixel_area_sr /= 2.0**2
# elif p.simulation_FoV_deg == 12.9080728652:
# 	p.sky_model_pixel_area_sr *= 2.0**2
# elif p.simulation_FoV_deg == 25.8161457304:
# 	p.sky_model_pixel_area_sr *= 4.0**2


if p.nside == 16:
    p.n_hpx_pix = 10
elif p.nside == 32:
    p.n_hpx_pix = 45
elif p.nside == 64:
    p.n_hpx_pix = 170
elif p.nside == 128:
    p.n_hpx_pix = 688
elif p.nside == 256:
    p.n_hpx_pix = 2746
elif p.nside == 512:
    # If scaling dA and keeping npix fixed
    p.n_hpx_pix = 10927
    # p.n_hpx_pix = 2703
    # Elif keeping dA fixed and scaling npix
    # if p.simulation_FoV_deg == 12.9080728652:
    # 	p.n_hpx_pix = 10927
    # else:
    # 	p.n_hpx_pix = 2703

# Data noise
if 'noise_data_path' not in p.__dict__.keys():
    sigma = p.sigma
else:
    effective_noise = np.load(p.noise_data_path)
    sigma = effective_noise.std()

if p.include_instrumental_effects:
    # Load uvw model in (nvis_per_chan, nchan) order
    uvw_multi_time_step_array_meters = load_uvw_instrument_sampling_m(
        p.instrument_model_directory)
    # uvw_multi_time_step_array_meters[:,:,:].reshape(-1,3)
    uvw_multi_time_step_array_meters_vectorised = np.reshape(
        uvw_multi_time_step_array_meters[:, :, :], (-1, 3))
    baseline_redundancy_array_time_vis_shaped = load_baseline_redundancy_array(
        p.instrument_model_directory)

    # n_vis sets the Number of visibilities per channel
    # (i.e. number of redundant baselines * number of time steps)
    n_vis = len(uvw_multi_time_step_array_meters_vectorised)

    # Re-weight baseline_redundancy_array (downweight to minimum
    # redundance baseline) to provide uniformly weighted data as input
    # to the analysis, so that the quick intrinsic noise fitting
    # approximation is valid, until generalised intrinsic noise fitting
    # is implemented.
    baseline_redundancy_array_time_vis_shaped = (
            baseline_redundancy_array_time_vis_shaped*0 +
            baseline_redundancy_array_time_vis_shaped.min())
    # baseline_redundancy_array_time_vis_shaped.reshape(-1,1).flatten()
    baseline_redundancy_array_vectorised = np.reshape(
        baseline_redundancy_array_time_vis_shaped, (-1,1)).flatten()

    # Keep average noise level consisitent with the non-instrumental
    # case by normalizing sigma by the average baseline redundancy
    # before scaling individual baselines by their respective
    # redundancies
    average_baseline_redundancy = np.mean(
        baseline_redundancy_array_time_vis_shaped)
    if 'noise_data_path' not in p.__dict__.keys():
        sigma = sigma * average_baseline_redundancy**0.5
    else:
        # Ensure sigma is cast as a float
        sigma = sigma*1.

    phasor_vector = np.load(
        os.path.join(p.instrument_model_directory,
                     'phasor_vector.npy'))

else:
    # Ensure sigma is cast as a float
    sigma = sigma*1.

# Check for HERA data path
if 'data_path' in p.__dict__.keys():
    use_EoR_cube = False
else:
    use_EoR_cube = True


# Auxiliary and derived params
small_cube = nu <= 7 and nv <= 7
nuv = (nu*nv - 1)
Show = False
chan_selection = ''
n_Fourier = (nu*nv - 1) * nf
n_LW = (nu*nv - 1) * nq
n_model = n_Fourier+n_LW
n_dat = n_Fourier
# current_file_version = 'Likelihood_v1d76_3D_ZM'
current_file_version = 'Likelihood_v2_3D_ZM'
array_save_directory = (
    'array_storage/batch_1/'
    + '{}_nu_{}_nv_{}_neta_{}_nq_{}_npl_{}_sigma_{:.1E}/'.format(
        current_file_version, nu, nv, neta, nq, npl, sigma).replace('.', 'd')
    )

if p.include_instrumental_effects:
    beam_info_str = ''
    if p.beam_type.lower() == 'Uniform'.lower():
        beam_info_str += '{}_beam_peak_amplitude_{}'.format(
            p.beam_type,
            str(p.beam_peak_amplitude).replace('.', 'd')
            )
    if p.beam_type.lower() == 'Gaussian'.lower():
        beam_info_str += (
            '{}_beam_peak_amplitude_{}'.format(
                p.beam_type,
                str(p.beam_peak_amplitude).replace('.', 'd'))
            )
        beam_info_str += (
            '_beam_width_{}_deg_at_{}_MHz'.format(
                str(p.FWHM_deg_at_ref_freq_MHz).replace('.', 'd'),
                str(p.PB_ref_freq_MHz).replace('.', 'd'))
            )

    instrument_model_directory_plus_beam_info = (
            p.instrument_model_directory[:-1]
            + '_{}/'.format(beam_info_str)
        )
    # instrument_info = filter(
    #     None,
    #     instrument_model_directory_plus_beam_info.split('/'))[-1]
    instrument_info =\
        instrument_model_directory_plus_beam_info.split('/')[-2]
    if p.model_drift_scan_primary_beam:
        instrument_info = instrument_info+'_dspb'
    if 'noise_data_path' in p.__dict__.keys():
        instrument_info += '_noise_vec'
    array_save_directory = (
            array_save_directory[:-1]
            + '_instrumental/'
            + instrument_info
            + '/'
        )
else:
    n_vis = 0

if npl == 1:
    array_save_directory = array_save_directory.replace(
        '_sigma',
        '_beta_{:.2E}_sigma'.format(p.beta)
        )
elif npl == 2:
    array_save_directory = array_save_directory.replace(
        '_sigma',
        '_b1_{:.2E}_b2_{:.2E}_sigma'.format(p.beta[0], p.beta[1])
        )

if p.fit_for_monopole:
    array_save_directory = (
            array_save_directory[:-1]
            + '_fit_for_monopole_eq_True/'
        )

array_save_directory = (
        array_save_directory[:-1]
        + '_nside{}/'.format(p.nside)
        # + '_nside{}_healpix_coords/'.format(p.nside)
    )
# Adding a FoV specific bit to the array save directory for FoV tests
array_save_directory = (
        array_save_directory[:-1]
        + '_fov_deg_{:.1f}/'.format(p.simulation_FoV_deg)
    )
# Uncomment for tests where npix is not identical between nsides
# array_save_directory = (
#         array_save_directory[:-1] +
#         '_fov_deg_{:.1f}_diff_npix/'.format(p.simulation_FoV_deg))

# # Temporary change for HEALPix matrix function testing
# array_save_directory = (
#         array_save_directory[:-1]
#         + '_healpix_testing/'
#     )
# # v2 includes functional removal and reformatting of Linalg and such
# # v3 includes directory rearranging
print('\nArray save directory: {}'.format(array_save_directory))

#--------------------------------------------
# Construct matrices
#--------------------------------------------
if p.include_instrumental_effects:
    if 'noise_data_path' not in p.__dict__.keys():
        BM = BuildMatrices(
            array_save_directory, nu, nv, nx, ny,
            n_vis, neta, nf, nq, sigma,
            npl=npl,
            uvw_multi_time_step_array_meters=\
                uvw_multi_time_step_array_meters,
            uvw_multi_time_step_array_meters_vectorised=\
                uvw_multi_time_step_array_meters_vectorised,
            baseline_redundancy_array_time_vis_shaped=\
                baseline_redundancy_array_time_vis_shaped,
            baseline_redundancy_array_vectorised=\
                baseline_redundancy_array_vectorised,
            phasor_vector=phasor_vector,
            beam_type=p.beam_type,
            beam_peak_amplitude=p.beam_peak_amplitude,
            beam_center=p.beam_center,
            FWHM_deg_at_ref_freq_MHz=p.FWHM_deg_at_ref_freq_MHz,
            PB_ref_freq_MHz=p.PB_ref_freq_MHz
            )
    else:
        BM = BuildMatrices(
            array_save_directory, nu, nv, nx, ny,
            n_vis, neta, nf, nq, sigma, npl=npl,
            uvw_multi_time_step_array_meters=\
                uvw_multi_time_step_array_meters,
            uvw_multi_time_step_array_meters_vectorised=\
                uvw_multi_time_step_array_meters_vectorised,
            baseline_redundancy_array_time_vis_shaped=\
                baseline_redundancy_array_time_vis_shaped,
            baseline_redundancy_array_vectorised=\
                baseline_redundancy_array_vectorised,
            phasor_vector=phasor_vector,
            beam_type=p.beam_type,
            beam_peak_amplitude=p.beam_peak_amplitude,
            beam_center=p.beam_center,
            FWHM_deg_at_ref_freq_MHz=p.FWHM_deg_at_ref_freq_MHz,
            PB_ref_freq_MHz=p.PB_ref_freq_MHz,
            effective_noise=effective_noise
            )
else:
    BM = BuildMatrices(
        array_save_directory, nu, nv, nx, ny,
        n_vis, neta, nf, nq, sigma, npl=npl)

if p.overwrite_matrices:
    print('Overwriting matrix stack')
    # Can be set to False unless npl>0
    overwrite_existing_matrix_stack = True
    proceed_without_overwrite_confirmation = True
else:
    # Can be set to False unless npl>0
    overwrite_existing_matrix_stack = False
    proceed_without_overwrite_confirmation = False

BM.build_minimum_sufficient_matrix_stack(
    overwrite_existing_matrix_stack=overwrite_existing_matrix_stack,
    proceed_without_overwrite_confirmation=
    proceed_without_overwrite_confirmation)

#--------------------------------------------
# Define power spectral bins and coordinate cubes
#--------------------------------------------
mod_k, k_x, k_y, k_z, deltakperp, deltakpara, x, y, z =\
    generate_k_cube_in_physical_coordinates_21cmFAST_v2d0(
        nu, nv, nx, ny, nf, neta,
        p.box_size_21cmFAST_pix_sc,
        p.box_size_21cmFAST_Mpc_sc
        )
k = mod_k.copy()
k_vis_ordered = k.T.flatten()
k_x_masked = generate_masked_coordinate_cubes(
    k_x, nu, nv, nx, ny, nf, neta, nq)
k_y_masked = generate_masked_coordinate_cubes(
    k_y, nu, nv, nx, ny, nf, neta, nq)
k_z_masked = generate_masked_coordinate_cubes(
    k_z, nu, nv, nx, ny, nf, neta, nq)
mod_k_masked = generate_masked_coordinate_cubes(
    mod_k, nu, nv, nx, ny, nf, neta, nq)
k_cube_voxels_in_bin, modkbins_containing_voxels = \
    generate_k_cube_model_spherical_binning_v2d1(
        mod_k_masked, k_z_masked, nu, nv, nx, ny, nf, neta, nq)
modk_vis_ordered_list = [
    mod_k_masked[k_cube_voxels_in_bin[i_bin]]
    for i_bin in range(len(k_cube_voxels_in_bin))
    ]

k_vals_file_name = (
    'k_vals_nu_{}_nv_{}_nf_{}_nq_{}_binning_v2d1_fov{:.1f}.txt'.format(
        nu, nv, nf, nq, p.simulation_FoV_deg
        )
    )
k_vals = calc_mean_binned_k_vals(
    mod_k_masked, k_cube_voxels_in_bin,
    save_k_vals=True, k_vals_file=k_vals_file_name)

do_cylindrical_binning = False
if do_cylindrical_binning:
    n_k_perp_bins = 2
    k_cube_voxels_in_bin, modkbins_containing_voxels, k_perp_bins =\
        generate_k_cube_model_cylindrical_binning(
            mod_k_masked, k_z_masked, k_y_masked, k_x_masked,
            n_k_perp_bins, nu, nv, nx, ny, nf, neta, nq)

#--------------------------------------------
# Non-instrumental data creation
#--------------------------------------------
if not p.include_instrumental_effects:
    # This needs to be updated to use astropy_healpix?
    #--------------------------------------------
    # Load EoR data
    #--------------------------------------------
    if use_EoR_cube:
        print('Using use_EoR_cube data')
        s_EoR, abc, scidata1 = generate_data_from_loaded_EoR_cube_v2d0(
            nu, nv, nx, ny, nf, neta, nq, k_x, k_y, k_z,
            Show, chan_selection, p.EoR_npz_path_sc)

    #--------------------------------------------
    # Define data vector
    #--------------------------------------------
    non_instrmental_noise_seed = 42123
    if use_EoR_cube:
        print('Using EoR cube')
        d = generate_data_and_noise_vector(
            sigma, s_EoR, nu, nv, nx, ny, nf, neta, nq,
            random_seed=non_instrmental_noise_seed)[0]
        s_Tot = s_EoR.copy()
    else:
        d = generate_data_and_noise_vector(
            sigma, s_EoR, nu,nv, nx, ny, nf, neta, nq,
            random_seed=non_instrmental_noise_seed)[0]
        s_Tot = s_EoR.copy()
        print('Using EoR cube')

    effective_noise = generate_data_and_noise_vector(
        sigma, np.zeros(d.shape), nu, nv, nx, ny, nf, neta, nq,
        random_seed=non_instrmental_noise_seed)[1]
    effective_noise_std = effective_noise.std()


#--------------------------------------------
# Load base matrices used in the likelihood and define related variables
#--------------------------------------------
T_Ninv_T = BM.read_data_s2d(
    array_save_directory +
    'T_Ninv_T',
    'T_Ninv_T')
Npar = T_Ninv_T.shape[0]
fit_for_LW_power_spectrum = True
masked_power_spectral_modes = np.ones(Npar)
masked_power_spectral_modes = masked_power_spectral_modes.astype('bool')
T = BM.read_data_s2d(array_save_directory + 'T', 'T')


#--------------------------------------------
# Data creation with instrumental effects
#--------------------------------------------
overwrite_data_with_WN = False
if p.include_instrumental_effects:
    if use_EoR_cube:
        Finv = BM.read_data_s2d(array_save_directory + 'Finv', 'Finv')
        s_EoR, abc, scidata1 = generate_EoR_signal_instrumental_im_2_vis(
            nu, nv, nx, ny, nf, neta, nq, k_x, k_y, k_z,
            Finv, Show, chan_selection, masked_power_spectral_modes,
            mod_k, p.EoR_npz_path_sc)
        # Temporary measure to save space since Finv can be large
        del Finv
    else:
        print('\nUsing data at {}'.format(p.data_path))
        s_EoR = np.load(p.data_path)

    if 'noise_data_path' not in p.__dict__.keys():
        EoR_noise_seed = 742123
        # EoR_noise_seed = 837463
        # EoR_noise_seed = 938475
        # EoR_noise_seed = 182654
        print('EoR_noise_seed =', EoR_noise_seed)
        d = generate_data_and_noise_vector_instrumental(
            1.0*sigma, s_EoR, nu, nv, nx, ny, nf, neta, nq,
            uvw_multi_time_step_array_meters_vectorised,
            baseline_redundancy_array_vectorised,
            random_seed=EoR_noise_seed)[0]
        effective_noise = generate_data_and_noise_vector_instrumental(
            1.0*sigma, s_EoR, nu, nv, nx, ny, nf, neta, nq,
            uvw_multi_time_step_array_meters_vectorised,
            baseline_redundancy_array_vectorised,
            random_seed=EoR_noise_seed)[1]
    else:
        d = s_EoR.copy()

effective_noise_std = effective_noise.std()
print('\ns_EoR.std = {:.4e}'.format(s_EoR.std()))
print('effective_noise.std = {:.4e}'.format(effective_noise_std))
print('dA = {:.4e}'.format(p.sky_model_pixel_area_sr))
print('effective SNR = {:.4e}'.format(s_EoR.std() / effective_noise_std),
      end='\n\n')


#--------------------------------------------
# Continue loading base matrices used in the
# likelihood and defining related variables
#--------------------------------------------
# block_T_Ninv_T = BM.read_data_s2d(
#     array_save_directory + 'block_T_Ninv_T',
#     'block_T_Ninv_T')
Ninv = BM.read_data_s2d(array_save_directory + 'Ninv', 'Ninv')
Ninv_d = np.dot(Ninv, d)
dbar = np.dot(T.conjugate().T, Ninv_d)
Sigma_Diag_Indices = np.diag_indices(T_Ninv_T.shape[0])
nDims = len(k_cube_voxels_in_bin)
d_Ninv_d = np.dot(d.conjugate(), Ninv_d)

print('size of T = {:.2f} GB'.format(sys.getsizeof(T) / 1.0e9))
print('size of T_Ninv_T = {:.2f} GB'.format(sys.getsizeof(T_Ninv_T) / 1.0e9))
print('size of Ninv = {:.2f} GB'.format(sys.getsizeof(Ninv) / 1.0e9))

if p.use_intrinsic_noise_fitting:
    nDims += 1

###
# nDims = nDims+3 for Gaussian prior over the
# three long wavelength model vectors
###
if p.use_LWM_Gaussian_prior:
    nDims += 3

x = [100.e0]*nDims
if p.fit_for_monopole:
    nuv = nu*nv
else:
    nuv = nu*nv-1
# block_T_Ninv_T = np.array(
#     [np.hsplit(block,nuv) for block in np.vsplit(T_Ninv_T,nuv)])
if p.include_instrumental_effects:
    block_T_Ninv_T = []

#--------------------------------------------
# Instantiate class and check that posterior_probability returns a
# finite probability (so no obvious binning errors etc.)
#--------------------------------------------
if small_cube:
    PSPP = PowerSpectrumPosteriorProbability(
        T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin,
        nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes,
        modk_vis_ordered_list, Ninv, d_Ninv_d, log_priors=False,
        intrinsic_noise_fitting=p.use_intrinsic_noise_fitting, k_vals=k_vals)
PSPP_block_diag = PowerSpectrumPosteriorProbability(
    T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin,
    nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes,
    modk_vis_ordered_list, Ninv, d_Ninv_d, block_T_Ninv_T=block_T_Ninv_T,
    Print=True, log_priors=False, k_vals=k_vals,
    intrinsic_noise_fitting=p.use_intrinsic_noise_fitting)

if small_cube:
    print(PSPP_block_diag.posterior_probability(
        [1.e0]*nDims,
        diagonal_sigma=False,
        block_T_Ninv_T=block_T_Ninv_T)[0])


if sub_ML_monopole_term_model:
    pre_sub_dbar = PSPP_block_diag.dbar
    PSPP_block_diag.inverse_LW_power = 0.0
    PSPP_block_diag.inverse_LW_power_zeroth_LW_term = p.inverse_LW_power
    # Don't fit for the first LW term (only fitting for the monopole)
    PSPP_block_diag.inverse_LW_power_first_LW_term = 2.e18
    # Don't fit for the second LW term (only fitting for the monopole)
    PSPP_block_diag.inverse_LW_power_second_LW_term = 2.e18

    if p.use_LWM_Gaussian_prior:
        fit_constraints = [2.e18] + [1.e-20] * (nDims-1)
    else:
        fit_constraints = [1.e-20] * nDims

    maxL_LW_fit = PSPP_block_diag.calc_SigmaI_dbar_wrapper(
        fit_constraints, T_Ninv_T, pre_sub_dbar,
        block_T_Ninv_T=block_T_Ninv_T)[0]
    maxL_LW_signal = np.dot(T, maxL_LW_fit)

    Ninv = BM.read_data_from_hdf5(array_save_directory + 'Ninv.h5', 'Ninv')
    Ninv_maxL_LW_signal = np.dot(Ninv, maxL_LW_signal)
    ML_qbar = np.dot(T.conjugate().T, Ninv_maxL_LW_signal)
    q_sub_dbar = pre_sub_dbar - ML_qbar
    if small_cube:
        PSPP.dbar = q_sub_dbar
    PSPP_block_diag.dbar = q_sub_dbar
    # Remove the constraints on the LW model for subsequent parts
    # of the analysis
    PSPP_block_diag.inverse_LW_power = p.inverse_LW_power
    print('Foreground pre-subtraction complete, '
          ' {} orders of magnitude foreground supression achieved.\n'.format(
                np.log10(
                    (d-effective_noise).std()
                    / (d-maxL_LW_signal-effective_noise).std()
                    )
                ))

    if small_cube:
        print(PSPP.posterior_probability(
            [1.e0]*nDims,
            diagonal_sigma=False)[0])
        print(PSPP_block_diag.posterior_probability(
            [1.e0]*nDims,
            diagonal_sigma=False,
            block_T_Ninv_T=block_T_Ninv_T)[0])


#--------------------------------------------
# Sample from the posterior
#--------------------------------------------
###
# PolyChord setup
###
# log_priors_min_max = [[-5.0, 3.0] for _ in range(nDims)]
log_priors_min_max = [[-1.0, 7.0] for _ in range(nDims)]
if p.use_LWM_Gaussian_prior:
    # Set minimum LW model priors using LW power spectrum in fit to
    # white noise (i.e the prior min should incorporate knowledge of
    # signal-removal in iterative pre-subtraction)
    fg_log_priors_min = np.log10(1.e5)
    # Set minimum LW model prior max using numerical stability
    # constraint at the given signal-to-noise in the data.
    fg_log_priors_max = 6.0
    # log_priors_min_max[0] = [fg_log_priors_min, 8.0] # Set
    # Calibrate LW model priors using white noise fitting
    log_priors_min_max[0] = [fg_log_priors_min, fg_log_priors_max]
    log_priors_min_max[1] = [fg_log_priors_min, fg_log_priors_max]
    log_priors_min_max[2] = [fg_log_priors_min, fg_log_priors_max]
    if p.use_intrinsic_noise_fitting:
        log_priors_min_max[1] = log_priors_min_max[0]
        log_priors_min_max[2] = log_priors_min_max[1]
        log_priors_min_max[3] = log_priors_min_max[2]
        log_priors_min_max[0] = [1.0, 2.0] # Linear alpha_prime range
else:
    if p.use_intrinsic_noise_fitting:
        log_priors_min_max[0] = [1.0, 2.0] # Linear alpha_prime range


print('\nlog_priors_min_max = {}'.format(log_priors_min_max))
prior_c = PriorC(log_priors_min_max)
nDerived = 0
nDims = len(k_cube_voxels_in_bin)
if p.use_intrinsic_noise_fitting:
    nDims += 1

###
# nDims = nDims+3 for Gaussian prior over the three long
# wavelength model vectors
###
if p.use_LWM_Gaussian_prior:
    nDims += 3

outputfiles_base_dir = 'chains/'
base_dir = outputfiles_base_dir+'clusters/'
if not os.path.isdir(base_dir):
    os.makedirs(base_dir)

log_priors = True
dimensionless_PS = True
if overwrite_data_with_WN:
    dimensionless_PS = False
zero_the_LW_modes = False

file_root = 'Test-{}_{}_{}_{}_{}_s_{:.1E}-lp_F-dPS_F-'.format(
    nu, nv, neta, nq, npl, sigma).replace('.', 'd')
if chan_selection != '':
    file_root = chan_selection + file_root
if npl == 1:
    file_root = file_root.replace('-dPS_F',
                                  '-dPS_F-beta_{:.2E}-v1'.format(p.beta))
if npl == 2:
    file_root = file_root.replace(
        '-dPS_F',
        '-dPS_F_b1_{:.2F}_b2_{:.2F}-v1'.format(p.beta[0], p.beta[1]))
if log_priors:
    file_root = file_root.replace('lp_F', 'lp_T')
if dimensionless_PS:
    file_root = file_root.replace('dPS_F', 'dPS_T')
if nq == 0:
    file_root = file_root.replace('mini-', 'mini-NQ-')
elif zero_the_LW_modes:
    file_root = file_root.replace('mini-', 'mini-ZLWM-')
if use_EoR_cube:
    file_root = file_root.replace('Test', 'EoR')
if use_MultiNest:
    file_root = 'MN-' + file_root

file_root = generate_output_file_base(file_root, version_number='1')
print('\nOutput file_root:', file_root)

PSPP_block_diag_Polychord = PowerSpectrumPosteriorProbability(
    T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin,
    nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes,
    modk_vis_ordered_list, Ninv, d_Ninv_d, block_T_Ninv_T=block_T_Ninv_T,
    log_priors=log_priors, dimensionless_PS=dimensionless_PS, Print=True,
    intrinsic_noise_fitting=p.use_intrinsic_noise_fitting, k_vals=k_vals
    )
if p.include_instrumental_effects and not zero_the_LW_modes:
    # Include minimal prior over LW modes required for numerical stability
    PSPP_block_diag_Polychord.inverse_LW_power = p.inverse_LW_power
if zero_the_LW_modes:
    PSPP_block_diag_Polychord.inverse_LW_power = 1.e20
    print('Setting PSPP_block_diag_Polychord.inverse_LW_power to:',
          PSPP_block_diag_Polychord.inverse_LW_power)
if sub_ML_monopole_term_model:
    SPP_block_diag_Polychord.dbar = q_sub_dbar
if p.use_intrinsic_noise_fitting and sub_ML_monopole_term_model:
    print('Using use_intrinsic_noise_fitting')
    PSPP_block_diag_Polychord.d_Ninv_d = q_sub_d_Ninv_q_sub_d

if p.useGPU:
    start = time.time()
    PSPP_block_diag_Polychord.Print = False
    Nit = 10
    # x_bad = [2.81531369, 2.4629003, -0.28626515, 5.39490566, 1.19703521,
    #          -0.77159866, 2.9684211, 4.28956387, 3.0131297]
    x_bad = [-0.48731995, 0.97696114, -0.0966692, 0.40549231, 5.35780334,
             -0.11230135, 5.95898771, 3.93845129, 6.81375599]
    print('Testing posterior calc with x={}'.format(x_bad))
    for _ in range(Nit):
        # L =  PSPP_block_diag_Polychord.posterior_probability([3.e0]*nDims)[0]
        L = PSPP_block_diag_Polychord.posterior_probability(x_bad)[0]
    print('Average evaluation time: {}'.format((time.time() - start)/float(Nit)),
          end='\n\n')


def likelihood(
        theta,
        calc_likelihood=PSPP_block_diag_Polychord.posterior_probability):
    return calc_likelihood(theta)


def MultiNest_likelihood(
        theta,
        calc_likelihood=PSPP_block_diag_Polychord.posterior_probability):
    return calc_likelihood(theta)[0]


run_single_node_analysis = False
if mpi_size > 1:
    print('mpi_size greater than 1, running multi-node analysis', end='\n\n')
else:
    print('mpi_size = {}, analysis will only be run if'\
          'run_single_node_analysis is set to True'.format(mpi_size))
    print('run_single_node_analysis = {}'.format(run_single_node_analysis),
          end='\n\n')

# import psutil
# process = psutil.Process(os.getpid())
# print('MaxRSS before log write = {} GB'.format(
#     process.memory_info()[0] / 1.0e9))
#
# if MPI.COMM_WORLD.Get_rank() == 0:
# 	write_log_file(array_save_directory, file_root)
#
# print('MaxRSS after log write = {} GB'.format(
#     process.memory_info()[0] / 1.0e9))

if run_single_node_analysis or mpi_size > 1:
    # Write log file
    if MPI.COMM_WORLD.Get_rank() == 0:
        write_log_file(array_save_directory, file_root)

    if use_MultiNest:
        MN_nlive = nDims*25
        # Run MultiNest
        result = solve(LogLikelihood=MultiNest_likelihood,
                       Prior=prior_c.prior_func,
                       n_dims=nDims,
                       outputfiles_basename=outputfiles_base_dir + file_root,
                       n_live_points=MN_nlive)
    else:
        precision_criterion = 0.05
        nlive = nDims * 10
        # Run PolyChord
        PolyChord.mpi_notification()
        PolyChord.run_nested_sampling(
            PSPP_block_diag_Polychord.posterior_probability,
            nDims,
            nDerived,
            file_root=file_root,
            read_resume=False,
            prior=prior_c.prior_func,
            precision_criterion=precision_criterion,
            nlive=nlive)

    print('Sampling complete!')
else:
    print('Skipping sampling, exiting...')
