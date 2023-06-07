""" Driver script for running BayesEoR """
# --------------------------------------------
# Imports
# --------------------------------------------
import numpy as np
import os
from pathlib import Path
from pprint import pprint
from rich.panel import Panel
from rich.rule import Rule

from bayeseor.params import BayesEoRParser
from bayeseor.utils import (
    mpiprint,
    get_array_dir_name,
    generate_mock_eor_signal_instrumental,
    vector_is_hermitian
)
from bayeseor.model import (
    load_inst_model,
    generate_k_cube_in_physical_coordinates,
    mask_k_cube,
    generate_k_cube_model_spherical_binning,
    calc_mean_binned_k_vals,
    generate_data_and_noise_vector_instrumental
)
from bayeseor.matrices import BuildMatrices
from bayeseor.posterior import PriorC


args = BayesEoRParser()

if __name__ == "__main__":
    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    mpiprint(f"\nmpi_size: {mpi_size}", rank=mpi_rank, end="\n\n")
    if args.multinest:
        from pymultinest.solve import solve
    else:
        import PyPolyChord.PyPolyChord as PolyChord
else:
    # Skip mpi and other imports that can cause crashes in ipython
    mpi_rank = 0
if mpi_rank == 0:
    mpiprint(Rule("Analysis parameters", style=None), style="bold")
    pprint(args.__dict__)

# --------------------------------------------
# Set analysis parameters
# --------------------------------------------
# Check for data path
if args.data_path:
    use_EoR_cube = False
    data = np.load(args.data_path, allow_pickle=True)
    if data.dtype.kind == "O":
        # Data and noise vector saved in a single dictionary
        dict_format = True
        data = data.item()
        if "noise" in data and "sigma" not in args:
            gen_noise = False
        else:
            gen_noise = True
    else:
        # Data and noise vector saved as separate arrays
        dict_format = False
        if "noise_data_path" in args:
            gen_noise = False
        else:
            gen_noise = True
else:
    use_EoR_cube = True
    gen_noise = True

if gen_noise:
    effective_noise = None  #FIXME: change from effective_noise to noise
else:
    if dict_format:
        effective_noise = data["noise"]
    else:
        effective_noise = np.load(args.noise_data_path)
    args.sigma = effective_noise.std()

if args.include_instrumental_effects:
    # uvw_array_m must have shape (nt, nbls, 3) and stores the (u, v, w)
    # coordinates sampled by the instrument.
    # bl_red_array must have shape (nt, nbls, 1) and stores the number of
    # redundant baselines (if data are redundantly averaged) per time and per
    # baseline type.
    # If modelling phased visibilities, phasor_vec must have shape (ndata,)
    # and stores a phasor per time, frequency, and baseline that phases
    # unphased visibilities.
    uvw_array_m, bl_red_array, phasor_vec = load_inst_model(args.inst_model)
    if phasor_vec is not None and args.drift_scan:
        phasor_vec = None
    uvw_array_m_vec = np.reshape(uvw_array_m, (-1, 3))

    # n_vis sets the number of visibilities per channel,
    # i.e. number of redundant baselines * number of time steps
    n_vis = len(uvw_array_m_vec)

    # Re-weight baseline_redundancy_array (downweight to minimum
    # redundance baseline) to provide uniformly weighted data as input
    # to the analysis so that, until generalized intrinsic noise fitting is
    # implemented, the quick intrinsic noise fitting approximation is valid
    bl_red_array = bl_red_array*0 + bl_red_array.min()
    bl_red_array_vec = np.reshape(bl_red_array, (-1, 1)).flatten()

    # Keep average noise level consistent with the non-instrumental
    # case by normalizing sigma by the average baseline redundancy
    # before scaling individual baselines by their respective
    # redundancies
    avg_bl_red = np.mean(bl_red_array)
    if gen_noise:
        args.sigma *= avg_bl_red**0.5
else:
    n_vis = 0

# --------------------------------------------
# Construct matrices
# --------------------------------------------
mpiprint("\n", Panel("Building Matrices"))

args.array_dir = get_array_dir_name(args)
mpiprint(
    f"[bold]Array save directory:[/bold] {args.array_dir}", rank=mpi_rank
)
BM = BuildMatrices(
    args.array_dir,
    args.include_instrumental_effects,
    args.use_sparse_matrices,
    args.nu,
    args.nv,
    n_vis,
    args.neta,
    args.nf,
    args.nu_min_MHz,
    args.channel_width_MHz,
    args.nq,
    args.nt,
    args.integration_time_seconds,
    args.sigma,
    args.fit_for_monopole,
    nside=args.nside,
    central_jd=args.central_jd,
    telescope_latlonalt=args.telescope_latlonalt,
    drift_scan_pb=args.drift_scan,
    beam_type=args.beam_type,
    beam_peak_amplitude=args.beam_peak_amplitude,
    beam_center=args.beam_center,
    fwhm_deg=args.fwhm_deg,
    antenna_diameter=args.antenna_diameter,
    cosfreq=args.cosfreq,
    achromatic_beam=args.achromatic_beam,
    beam_ref_freq=args.beam_ref_freq,
    du_eor=args.du_eor,
    dv_eor=args.dv_eor,
    du_fg=args.du_fg,
    dv_fg=args.dv_fg,
    deta=args.deta,
    fov_ra_eor=args.fov_ra_eor,
    fov_dec_eor=args.fov_dec_eor,
    nu_fg=args.nu_fg,
    nv_fg=args.nv_fg,
    npl=args.npl,
    beta=args.beta,
    fov_ra_fg=args.fov_ra_fg,
    fov_dec_fg=args.fov_dec_fg,
    simple_za_filter=args.simple_za_filter,
    uvw_array_m=uvw_array_m,
    bl_red_array=bl_red_array,
    bl_red_array_vec=bl_red_array_vec,
    phasor_vec=phasor_vec,
    use_shg=args.use_shg,
    fit_for_shg_amps=args.fit_for_shg_amps,
    nu_sh=args.nu_sh,
    nv_sh=args.nv_sh,
    nq_sh=args.nq_sh,
    npl_sh=args.npl_sh,
    effective_noise=effective_noise,
    taper_func=args.taper_func
)

if args.overwrite_matrices:
    mpiprint(
        "\nWARNING: Overwriting matrix stack\n", rank=mpi_rank,
        style="bold red", justify="center"
    )
    clobber_matrices = True
    force_clobber = True
else:
    clobber_matrices = False
    force_clobber = False
BM.build_minimum_sufficient_matrix_stack(
    clobber_matrices=clobber_matrices, force_clobber=force_clobber
)


# --------------------------------------------
# Define power spectral bins and coordinate cubes
# --------------------------------------------
mod_k, k_x, k_y, k_z, x, y, z = generate_k_cube_in_physical_coordinates(
        args.nu, args.nv, args.neta, args.ps_box_size_ra_Mpc,
        args.ps_box_size_dec_Mpc, args.ps_box_size_para_Mpc
)
mod_k_vo = mask_k_cube(mod_k)
k_cube_voxels_in_bin, modkbins_containing_voxels = \
    generate_k_cube_model_spherical_binning(
        mod_k_vo, args.ps_box_size_para_Mpc
    )
k_vals = calc_mean_binned_k_vals(
    mod_k_vo, k_cube_voxels_in_bin, save_k_vals=False, rank=mpi_rank
)


# --------------------------------------------
# Load base matrices used in the likelihood and define related variables
# --------------------------------------------
T = BM.read_data_s2d(args.array_dir + "T", "T")
T_Ninv_T = BM.read_data_s2d(args.array_dir + "T_Ninv_T", "T_Ninv_T")
Npar = T_Ninv_T.shape[0]
masked_power_spectral_modes = np.ones(Npar)
masked_power_spectral_modes = masked_power_spectral_modes.astype(bool)


# --------------------------------------------
# Data creation with instrumental effects
# --------------------------------------------
if args.include_instrumental_effects:
    if use_EoR_cube:
        Finv = BM.read_data_s2d(args.array_dir + "Finv", "Finv")
        s_EoR, white_noise_sky = generate_mock_eor_signal_instrumental(
            Finv, args.nf, args.fov_ra_eor, args.fov_dec_eor, args.nside,
            args.telescope_latlonalt, args.central_jd, args.nt,
            args.integration_time_seconds, rank=mpi_rank
        )
        del Finv
    else:
        mpiprint("\nUsing data at {}".format(args.data_path), rank=mpi_rank)
        if dict_format:
            s_EoR = data["data"]
        else:
            s_EoR = data

    EoR_noise_seed = args.noise_seed
    if gen_noise:
        mpiprint("EoR_noise_seed =", EoR_noise_seed, rank=mpi_rank)
        # Assumes the instrument model contains duplicates of the
        # unphased uvw coordinates in each time entry of the
        # instrument model
        d, effective_noise, bl_conjugate_pairs_map =\
            generate_data_and_noise_vector_instrumental(
                1.0*args.sigma, s_EoR, args.nf, args.nt,
                uvw_array_m[0],
                bl_red_array[0],
                random_seed=EoR_noise_seed,
                rank=mpi_rank
            )
    else:
        d = s_EoR.copy()
        _, _, bl_conjugate_pairs_map =\
            generate_data_and_noise_vector_instrumental(
                1.0*args.sigma, s_EoR, args.nf, args.nt,
                uvw_array_m[0],
                bl_red_array[0],
                random_seed=EoR_noise_seed,
                rank=mpi_rank
            )

    if args.taper_func is not None:
        taper_matrix = BM.read_data(
            args.array_dir + "taper_matrix", "taper_matrix"
        )
        if args.use_sparse_matrices:
            d = taper_matrix * d
        else:
            d = np.dot(taper_matrix, d)
        del taper_matrix

effective_noise_std = effective_noise.std()
mpiprint("\ns_EoR.std = {:.4e}".format(s_EoR.std()), rank=mpi_rank)
mpiprint(
    "signal is Hermitian: {}".format(
        vector_is_hermitian(
            s_EoR, bl_conjugate_pairs_map, args.nt, args.nf,
            uvw_array_m.shape[1]
        )
    ),
    rank=mpi_rank
)
mpiprint(
    "signal + noise is Hermitian: {}".format(
        vector_is_hermitian(
            d, bl_conjugate_pairs_map, args.nt, args.nf, uvw_array_m.shape[1]
        )
    ),
    rank=mpi_rank
)
mpiprint(
    "effective_noise.std = {:.4e}".format(effective_noise_std),
    rank=mpi_rank
)
mpiprint(
    "effective SNR = {:.4e}".format(s_EoR.std() / effective_noise_std),
    end="\n\n",
    rank=mpi_rank
)


# --------------------------------------------
# Continue loading base matrices used in the
# likelihood and defining related variables
# --------------------------------------------
Ninv = BM.read_data(args.array_dir + "Ninv", "Ninv")
Ninv_d = Ninv * d
dbar = np.dot(T.conjugate().T, Ninv_d)
Sigma_Diag_Indices = np.diag_indices(T_Ninv_T.shape[0])
d_Ninv_d = np.dot(d.conjugate(), Ninv_d)
nDims = len(k_cube_voxels_in_bin)
if args.use_intrinsic_noise_fitting:
    nDims += 1
if args.use_LWM_Gaussian_prior:
    nDims += 3
if args.include_instrumental_effects:
    block_T_Ninv_T = []


# --------------------------------------------
# Sample from the posterior
# --------------------------------------------
log_priors_min_max = [  # FIXME: Add this as a parameter in params
    [-2.0, 2.0], [-1.2, 2.8], [-0.7, 3.3], [0.7, 2.7], [1.1, 3.1],
    [1.5, 3.5], [2.0, 4.0], [2.4, 4.4], [2.7, 4.7]
]
if args.use_LWM_Gaussian_prior:
    """
    WARNING: use_LWM_Gaussian_prior is currently not implemented for the
    current version of this code.  This if block has been left here for
    posterity.
    """
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
    if args.use_intrinsic_noise_fitting:
        log_priors_min_max[1] = log_priors_min_max[0]
        log_priors_min_max[2] = log_priors_min_max[1]
        log_priors_min_max[3] = log_priors_min_max[2]
        log_priors_min_max[0] = [1.0, 2.0]  # Linear alpha_prime range
else:
    if args.use_intrinsic_noise_fitting:
        log_priors_min_max[0] = [1.0, 2.0]  # Linear alpha_prime range


mpiprint("\nlog_priors_min_max = {}".format(log_priors_min_max), rank=mpi_rank)
prior_c = PriorC(log_priors_min_max)
nDerived = 0  # PolyChord parameter
dimensionless_PS = True  # FIXME: Add normalization function for P(k)

# Sampler output
output_dir = Path(args.output_dir) / "chains"
output_dir.mkdir(exist_ok=True, parents=False)

if args.file_root is None:
    file_root = f"Test-{args.nu}-{args.nv}-{args.neta}-{args.nq}-{args.npl}"
    file_root += f"{args.sigma:.1E}"
    if chan_selection != "":
        file_root = chan_selection + file_root
    if npl == 1:
        file_root += "{:.2f}-".format(p.beta)
    elif npl == 2:
        file_root += "{:.2F}_{:.2F}-".format(p.beta[0], p.beta[1])
    if log_priors:
        file_root += "lp-"
    if dimensionless_PS:
        file_root += "dPS-"
    if nq == 0:
        file_root = file_root.replace("mini-", "mini-NQ-")
    elif zero_the_LW_modes:
        file_root = file_root.replace("mini-", "mini-ZLWM-")
    if use_EoR_cube:
        file_root = file_root.replace("Test", "EoR")
    if use_MultiNest:
        file_root = "MN-" + file_root
    if p.use_shg:
        file_root += "SH_{}_{}_{}_{}-".format(nu_sh, nv_sh, nq_sh, npl_sh)
        if p.fit_for_shg_amps:
            file_root += "ffsa-"
    file_root += "v1-"
    file_root = generate_output_file_base(file_root, version_number="1")
else:
    file_root = p.file_root
mpiprint("\nOutput file_root:", file_root, rank=mpi_rank)
