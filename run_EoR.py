""" Driver script for running BayesEoR """
# --------------------------------------------
# Imports
# --------------------------------------------
# Make everything available for now, this can be refined later
from BayesEoR import *
import BayesEoR.Params.params as p

import sys
import os
import time
import ast
import numpy as np
from astropy import units

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

# --------------------------------------------
# Set analysis parameters
# --------------------------------------------
update_params_with_command_line_arguments()
if p.beam_center is not None:
    p.beam_center = ast.literal_eval(p.beam_center)
    print('Beam center = {} (type {})'.format(p.beam_center,
                                              type(p.beam_center)))
npl = p.npl
nq = p.nq
if nq > npl:
    nq = npl
nf = p.nf
neta = p.neta
if not p.include_instrumental_effects:
    neta = neta - nq
nu = p.nu
if p.nv is None:
    p.nv = nu
nv = p.nv
nuv = nu*nv - 1*(not p.fit_for_monopole)

if p.fov_dec_deg is None:
    p.fov_dec_deg = p.fov_ra_deg

p.delta_u_irad = 1.0 / np.deg2rad(p.fov_ra_deg)
p.delta_v_irad = 1.0 / np.deg2rad(p.fov_dec_deg)
p.delta_eta_iHz = 1.0 / (p.nf*p.channel_width_MHz*1e6)
p.sky_model_pixel_area_sr = 4 * np.pi / (12 * p.nside**2)

nu_sh = p.nu_sh
if p.nv_sh is None:
    p.nv_sh = nu_sh
nv_sh = p.nv_sh
nq_sh = p.nq_sh
npl_sh = p.npl_sh
if p.use_shg:
    nuv_sh = nu_sh*nv_sh - 1
else:
    nuv_sh = None

# Improve numerical precision when performing evidence comparison.
sub_ML_monopole_term_model = False

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
        baseline_redundancy_array_time_vis_shaped, (-1, 1)).flatten()

    # Keep average noise level consistent with the non-instrumental
    # case by normalizing sigma by the average baseline redundancy
    # before scaling individual baselines by their respective
    # redundancies
    average_baseline_redundancy = np.mean(
        baseline_redundancy_array_time_vis_shaped)
    if 'noise_data_path' not in p.__dict__.keys():
        sigma = sigma * average_baseline_redundancy**0.5
    else:
        sigma = sigma*1.

    phasor_vector = np.load(
        os.path.join(p.instrument_model_directory,
                     'phasor_vector.npy'))
    if p.unphased:
        phasor_vector = np.ones_like(phasor_vector)

else:
    sigma = sigma*1.

# Check for HERA data path
if 'data_path' in p.__dict__.keys():
    use_EoR_cube = False
else:
    use_EoR_cube = True

# Auxiliary and derived params
Show = False
chan_selection = ''

# --------------------------------------------
# Construct matrices
# --------------------------------------------
current_file_version = 'Likelihood_v2d8_3D_ZM'
array_save_directory = (
    'array_storage/batch_1/'
    + '{}_nu_{}_nv_{}_neta_{}_nq_{}_npl_{}_sigma_{:.1E}/'.format(
        current_file_version, nu, nv, neta, nq, npl, sigma).replace('.', 'd')
    )

if p.fit_for_monopole:
    array_save_directory = (
            array_save_directory[:-1]
            + '_fit_for_monopole/'
        )
# nside modifier
array_save_directory = array_save_directory[:-1] + '_nside{}/'.format(p.nside)

# FoV modifier
if p.fov_ra_deg != p.fov_dec_deg:
    fov_str = '_fov_deg_ra_{:.1f}_dec_{:.1f}'.format(p.fov_ra_deg, p.fov_dec_deg)
else:
    fov_str = '_fov_deg_{:.1f}'.format(p.fov_ra_deg)
array_save_directory = array_save_directory[:-1] + fov_str + '/'

# Append a beam center classifier
if p.beam_center is not None:
    beam_center_signs = [
        '+' if p.beam_center[i] >= 0 else '' for i in range(2)
        ]
    beam_center_str = \
        '_beam_center_RA0{}{:.2f}_DEC0{}{:.2f}'.format(
            beam_center_signs[0],
            p.beam_center[0],
            beam_center_signs[1],
            p.beam_center[1]
            )
    array_save_directory = array_save_directory[:-1] + beam_center_str + '/'

if p.unphased:
    array_save_directory = array_save_directory[:-1] + '_unphased/'

# Subharmonic grid (SHG) modifiers
if p.use_shg:
    shg_str = '_SHG'
    if nu_sh > 0:
        shg_str += '_nu_sh_{}'.format(nu_sh)
    if nv_sh > 0:
        shg_str += '_nv_sh_{}'.format(nv_sh)
    if nq_sh > 0:
        shg_str += '_nq_sh_{}'.format(nq_sh)
    if npl_sh > 0:
        shg_str += '_npl_sh_{}'.format(npl_sh)
    if p.fit_for_shg_amps:
        shg_str += '_ffsa'
    array_save_directory = array_save_directory[:-1] + shg_str + '/'

if p.include_instrumental_effects:
    beam_info_str = ''
    p.beam_type = p.beam_type.lower()
    if p.beam_type == 'uniform':
        beam_info_str += '{}_beam_peak_amplitude_{}'.format(
            p.beam_type,
            str(p.beam_peak_amplitude).replace('.', 'd')
            )
    elif p.beam_type == 'gaussian':
        beam_info_str += (
            '{}_beam_peak_amplitude_{}'.format(
                p.beam_type,
                str(p.beam_peak_amplitude).replace('.', 'd'))
            )
        if p.fwhm_deg is not None:
            beam_info_str += (
                '_beam_width_{}_deg'.format(str(p.fwhm_deg).replace('.', 'd'))
            )
        elif p.antenna_diameter is not None:
            beam_info_str += (
                '_antenna-diameter-{}m'.format(
                    str(np.round(p.antenna_diameter, decimals=2)).replace(
                        '.', 'd')
                    )
            )
        else:
            print('\nIf using a Gaussian beam, must specify either a FWHM in'
                  ' deg or an antenna diameter in meters.\nExiting...',
                  end='\n\n'
                  )
            sys.exit()
    elif p.beam_type == 'airy':
        beam_info_str += '{}_beam_antenna-diameter-{}m'.format(
            p.beam_type,
            str(np.round(p.antenna_diameter, decimals=2)).replace('.', 'd')
            )

    instrument_model_directory_plus_beam_info = (
            p.instrument_model_directory[:-1]
            + '_{}/'.format(beam_info_str)
        )
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

print('\nArray save directory: {}'.format(array_save_directory))

if p.include_instrumental_effects:
    if 'noise_data_path' not in p.__dict__.keys():
        BM = BuildMatrices(
            array_save_directory, nu, nv,
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
            fwhm_deg=p.fwhm_deg,
            antenna_diameter=p.antenna_diameter,
            delta_u_irad=p.delta_u_irad, delta_v_irad=p.delta_v_irad,
            delta_eta_iHz=p.delta_eta_iHz,
            use_shg=p.use_shg, fit_for_shg_amps=p.fit_for_shg_amps,
            nu_sh=nu_sh, nv_sh=nv_sh, nq_sh=nq_sh, npl_sh=npl_sh
            )
    else:
        BM = BuildMatrices(
            array_save_directory, nu, nv,
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
            fwhm_deg=p.fwhm_deg,
            antenna_diameter=p.antenna_diameter,
            delta_u_irad=p.delta_u_irad, delta_v_irad=p.delta_v_irad,
            delta_eta_iHz=p.delta_eta_iHz,
            use_shg=p.use_shg, fit_for_shg_amps=p.fit_for_shg_amps,
            nu_sh=nu_sh, nv_sh=nv_sh, nq_sh=nq_sh, npl_sh=npl_sh,
            effective_noise=effective_noise
            )
else:
    BM = BuildMatrices(
        array_save_directory, nu, nv,
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


# --------------------------------------------
# Define power spectral bins and coordinate cubes
# --------------------------------------------
cosmo = Cosmology()
# The box size parameters determine the side lengths of the
# cosmological volume from which the power spectrum is estimated
freqs_MHz = p.nu_min_MHz + np.arange(p.nf)*p.channel_width_MHz
bandwidth_MHz = p.channel_width_MHz * p.nf
redshift = cosmo.f2z((freqs_MHz.mean() * units.MHz).to('Hz'))
ps_box_size_ra_Mpc = (
    cosmo.dL_dth(redshift) * np.deg2rad(p.fov_ra_deg))
ps_box_size_dec_Mpc = (
    cosmo.dL_dth(redshift) * np.deg2rad(p.fov_dec_deg))
ps_box_size_para_Mpc = (
    cosmo.dL_df(redshift) * (bandwidth_MHz * 1e6))
mod_k, k_x, k_y, k_z, x, y, z =\
    generate_k_cube_in_physical_coordinates_21cmFAST_v2d0(
        nu, nv, nf, neta, ps_box_size_ra_Mpc,
        ps_box_size_dec_Mpc, ps_box_size_para_Mpc
        )
k = mod_k.copy()
k_vis_ordered = k.T.flatten() # not used for anything
k_x_masked = generate_masked_coordinate_cubes(
    k_x, nu, nv, nf, neta, nq, ps_box_size_ra_Mpc,
    ps_box_size_dec_Mpc, ps_box_size_para_Mpc
)
k_y_masked = generate_masked_coordinate_cubes(
    k_y, nu, nv, nf, neta, nq, ps_box_size_ra_Mpc,
    ps_box_size_dec_Mpc, ps_box_size_para_Mpc
)
k_z_masked = generate_masked_coordinate_cubes(
    k_z, nu, nv, nf, neta, nq, ps_box_size_ra_Mpc,
    ps_box_size_dec_Mpc, ps_box_size_para_Mpc
)
mod_k_masked = generate_masked_coordinate_cubes(
    mod_k, nu, nv, nf, neta, nq, ps_box_size_ra_Mpc,
    ps_box_size_dec_Mpc, ps_box_size_para_Mpc
)
k_cube_voxels_in_bin, modkbins_containing_voxels = \
    generate_k_cube_model_spherical_binning_v2d1(
        mod_k_masked, k_z_masked, nu, nv, nf, neta, nq,
        ps_box_size_ra_Mpc, ps_box_size_dec_Mpc, ps_box_size_para_Mpc
    )
modk_vis_ordered_list = [
    mod_k_masked[k_cube_voxels_in_bin[i_bin]]
    for i_bin in range(len(k_cube_voxels_in_bin))
    ]

k_vals_file_name = (
    'k_vals_nu_{}_nv_{}_nf_{}_nq_{}_binning_v2d1{}.txt'.format(
        nu, nv, nf, nq, fov_str
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
            n_k_perp_bins, nu, nv, nf, neta, nq,
            ps_box_size_ra_Mpc, ps_box_size_dec_Mpc, ps_box_size_para_Mpc)


# --------------------------------------------
# Load base matrices used in the likelihood and define related variables
# --------------------------------------------
T = BM.read_data_s2d(array_save_directory + 'T', 'T')
T_Ninv_T = BM.read_data_s2d(array_save_directory + 'T_Ninv_T', 'T_Ninv_T')
Npar = T_Ninv_T.shape[0]
masked_power_spectral_modes = np.ones(Npar)
masked_power_spectral_modes = masked_power_spectral_modes.astype('bool')


# --------------------------------------------
# Data creation with instrumental effects
# --------------------------------------------
if p.include_instrumental_effects:
    if use_EoR_cube:
        Finv = BM.read_data_s2d(array_save_directory + 'Finv', 'Finv')
        s_EoR, abc, scidata1 = generate_EoR_signal_instrumental_im_2_vis(
            nu, nv, nf, neta, nq, k_x, k_y, k_z,
            Finv, Show, chan_selection, masked_power_spectral_modes,
            mod_k, p.eor_sim_path)
        del Finv
    else:
        print('\nUsing data at {}'.format(p.data_path))
        s_EoR = np.load(p.data_path)

    if 'noise_data_path' not in p.__dict__.keys():
        EoR_noise_seed = p.noise_seed
        print('EoR_noise_seed =', EoR_noise_seed)
        # Assumes the instrument model contains duplicates of the
        # unphased uvw coordinates in each time entry of the
        # instrument model
        d, effective_noise, bl_conjugate_pairs_map =\
            generate_data_and_noise_vector_instrumental(
                1.0*sigma, s_EoR, nu, nv, nf, neta, nq, p.nt,
                uvw_multi_time_step_array_meters[0],
                baseline_redundancy_array_time_vis_shaped[0],
                random_seed=EoR_noise_seed)
    else:
        d = s_EoR.copy()

effective_noise_std = effective_noise.std()
print('\ns_EoR.std = {:.4e}'.format(s_EoR.std()))
print('signal is Hermitian: {}'.format(
    vector_is_hermitian(s_EoR, bl_conjugate_pairs_map, p.nt, nf,
                        uvw_multi_time_step_array_meters.shape[1])
))
print('signal + noise is Hermitian: {}'.format(
    vector_is_hermitian(d, bl_conjugate_pairs_map, p.nt, nf,
                        uvw_multi_time_step_array_meters.shape[1])
))
print('effective_noise.std = {:.4e}'.format(effective_noise_std))
print('dA = {:.4e}'.format(p.sky_model_pixel_area_sr))
print('effective SNR = {:.4e}'.format(s_EoR.std() / effective_noise_std),
      end='\n\n')


# --------------------------------------------
# Continue loading base matrices used in the
# likelihood and defining related variables
# --------------------------------------------
Ninv = BM.read_data(array_save_directory + 'Ninv', 'Ninv')
Ninv_d = Ninv * d
dbar = np.dot(T.conjugate().T, Ninv_d)
Sigma_Diag_Indices = np.diag_indices(T_Ninv_T.shape[0])
d_Ninv_d = np.dot(d.conjugate(), Ninv_d)
nDims = len(k_cube_voxels_in_bin)
if p.use_intrinsic_noise_fitting:
    nDims += 1
if p.use_LWM_Gaussian_prior:
    nDims += 3
if p.include_instrumental_effects:
    block_T_Ninv_T = []
x = [100.e0]*nDims

# --------------------------------------------
# Sample from the posterior
# --------------------------------------------
###
# PolyChord setup
###
log_priors_min_max = [[-2., 6.] for _ in range(nDims)]
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

# Sampler output
outputfiles_base_dir = 'chains/'
base_dir = outputfiles_base_dir+'clusters/'
if not os.path.isdir(base_dir):
    os.makedirs(base_dir)

log_priors = True
dimensionless_PS = True
zero_the_LW_modes = False

if p.file_root is None:
    file_root = 'Test-{}_{}_{}_{}_{}_{:.1E}-'.format( # lp_F-dPS_F-
        nu, nv, neta, nq, npl, sigma).replace('.', 'd')
    if chan_selection != '':
        file_root = chan_selection + file_root
    if npl == 1:
        file_root += '{:.2E}-'.format(p.beta)
    elif npl == 2:
        file_root += (
            '{:.2F}_{:.2F}-'.format(p.beta[0], p.beta[1]))
    if log_priors and p.n_uniform_prior_k_bins == 0:
        file_root += 'lp-'
    if dimensionless_PS:
        file_root += 'dPS-'
    if nq == 0:
        file_root = file_root.replace('mini-', 'mini-NQ-')
    elif zero_the_LW_modes:
        file_root = file_root.replace('mini-', 'mini-ZLWM-')
    if use_EoR_cube:
        file_root = file_root.replace('Test', 'EoR')
    if use_MultiNest:
        file_root = 'MN-' + file_root
    if p.use_shg:
        file_root += 'SH_{}_{}_{}_{}-'.format(
            nu_sh, nv_sh, nq_sh, npl_sh)
        if p.fit_for_shg_amps:
            file_root += 'ffsa-'
    file_root += 'v1-'
    file_root = generate_output_file_base(file_root, version_number='1')
else:
    file_root = p.file_root
print('\nOutput file_root:', file_root)

PSPP_block_diag_Polychord = PowerSpectrumPosteriorProbability(
    T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin,
    nuv, nu, nv, neta, nf, nq, masked_power_spectral_modes,
    modk_vis_ordered_list, Ninv, d_Ninv_d, block_T_Ninv_T=block_T_Ninv_T,
    log_priors=log_priors, dimensionless_PS=dimensionless_PS, Print=True,
    intrinsic_noise_fitting=p.use_intrinsic_noise_fitting, k_vals=k_vals,
    n_uniform_prior_k_bins=p.n_uniform_prior_k_bins,
    ps_box_size_ra_Mpc=ps_box_size_ra_Mpc,
    ps_box_size_dec_Mpc=ps_box_size_dec_Mpc,
    ps_box_size_para_Mpc=ps_box_size_para_Mpc,
    use_shg=p.use_shg, fit_for_shg_amps=p.fit_for_shg_amps,
    nuv_sh=nuv_sh, nu_sh=nu_sh, nv_sh=nv_sh, nq_sh=nq_sh
    )
if p.include_instrumental_effects and not zero_the_LW_modes:
    # Include minimal prior over LW modes required for numerical stability
    PSPP_block_diag_Polychord.inverse_LW_power = p.inverse_LW_power
if zero_the_LW_modes:
    PSPP_block_diag_Polychord.inverse_LW_power = 1.e20
    print('Setting PSPP_block_diag_Polychord.inverse_LW_power to:',
          PSPP_block_diag_Polychord.inverse_LW_power)

if p.useGPU:
    start = time.time()
    PSPP_block_diag_Polychord.Print = False
    Nit = 10
    for _ in range(Nit):
        L = PSPP_block_diag_Polychord.posterior_probability([1.e0]*nDims)[0]
        if not np.isfinite(L):
            print('WARNING: Infinite value returned in posterior calculation!')
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
    print('mpi_size = {}, analysis will only be run if '
          'run_single_node_analysis is set to True'.format(mpi_size))
    print('run_single_node_analysis = {}'.format(run_single_node_analysis),
          end='\n\n')

if run_single_node_analysis or mpi_size > 1:
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
