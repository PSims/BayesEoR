""" Analysis settings """
import numpy as np
from astropy import constants
from astropy import units
from jsonargparse import ArgumentParser, ActionYesNo, ActionConfigFile
from jsonargparse.typing import List, Path_fr, path_type

from ..utils.cosmology import Cosmology


def BayesEoRParser():
    """
    Class used to parse command line arguments.

    """
    Path_dr = path_type("dr")  # checks for a readable directory

    parser = ArgumentParser()
    # --- Compute params ---
    parser.add_argument(
        "--gpu",
        action=ActionYesNo(yes_prefix="g", no_prefix="c"),
        default=True,
        dest="useGPU",
        help="Use GPUs (--gpu) or CPUs (--cpu).  Using GPUs is strongly "
             "encouraged as the CPU based matrix inversion methods are either "
             "slow or unreliable.  Defaults to True (use GPUs)."
    )
    parser.add_argument(
        "--single-node",
        action="store_true",
        help="If passed, run an analysis on a single node.  If the MPI size is"
             "1 and --single-node is not passed, the matrices will be built "
             "but the power spectrum analysis will not be run."
    )
    parser.add_argument(
        "--array-dir-prefix",
        type=str,
        default="./array-storage/",
        help="Directory in which matrices will be written to disk.  Must be "
             "a path to a directory which is writeable.  Defaults to "
             "'./array-storage/'."
    )
    parser.add_argument(
        "--sparse-mats",
        action=ActionYesNo(yes_prefix="sparse-", no_prefix="dense-"),
        dest="use_sparse_matrices",
        default=True,
        help="Use sparse matrices (--sparse-mats) to reduce storage "
             "requirements or use dense matrices (--dense-mats).  Defaults "
             "to True (use sparse matrices)."
    )
    parser.add_argument(
        "--overwrite-matrices",
        action="store_true",
        help="If passed, overwrite existing matrix stack."
    )
    parser.add_argument(
        "--log-priors",
        action=ActionYesNo(yes_prefix="log-", no_prefix="lin-"),
        default=True,
        help="Use log_10 (--log-priors) or linear (--lin-priors) prior "
             "values.  If using log priors (x), the prior values will first be"
             " linearized (10^x)."
    )
    parser.add_argument(
        "--uprior-bins",
        type=str,
        default="",
        help="Array indices of k-bins using a uniform prior.  Follows python "
             "slicing syntax.  Can pass a range via '1:4' (non-inclusive high "
             "end), a list of indices via '1,4,6' (no spaces between commas), "
             " a single index '3' or '-3', or 'all'.  Defaults to an empty "
             "string (all k-bins use log-uniform priors)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./chains/",
        help="Directory for sampler output.  Default behavior creates a "
             "directory 'chains' in the current working directory, i.e. "
             "outputs to './chains/'."
    )
    parser.add_argument(
        "--file-root",
        type=str,
        help="If None (default), start a new analysis.  Otherwise, resume "
             "analysis from `file_root`."
    )
    parser.add_argument(
        "--multinest",
        action=ActionYesNo(yes_prefix="multinest", no_prefix="polychord"),
        default=True,
        dest="use_Multinest",
        help="Use Multinest (--multinest) or Polychord (--polychord) as "
             "the sampler.  Defaults to using Multinest.  Using Polychord is "
             "advised for large parameter spaces."
    )
    # --- Model params ---
    # Frequency params
    parser.add_argument(
        "--nf",
        type=int,
        help="Number of frequency channels."
    )
    parser.add_argument(
        "--neta",
        type=int,
        help="Number of line-of-sight Fourier modes.  Defaults to the number "
             "of frequency channels."
    )
    parser.add_argument(
        "--freq-min",
        type=float,
        dest="nu_min_MHz",  #FIXME
        help="Minimum frequency in megahertz."
    )
    parser.add_argument(
        "--delta-freq",
        type=float,
        dest="channel_width_MHz",  #FIXME
        help="Frequency channel width in megahertz."
    )
    # Sky model params
    parser.add_argument(
        "--nside",
        type=int,
        help="HEALPix resolution parameter. Sets the resolution of the "
             "sky model.  Note, the HEALPix resolution must be chosen such "
             "that there are two HEALPix pixels per minimum fringe wavelength "
             "from the model uv-plane."
    )
    parser.add_argument(
        "--simple-za-filter",
        action="store_true",
        help="If passed, filter pixels in the sky model by zenith angle only. "
             "Otherwise, filter pixels in a rectangular region set by the FoV "
             "values along the RA and DEC axes (default)."
    )
    # EoR model params
    parser.add_argument(
        "--nu",
        type=int,
        help="Number of pixels on the u-axis of the model uv-plane for the EoR"
             " model."
    )
    parser.add_argument(
        "--nv",
        type=int,
        help="Number of pixels on the v-axis of the model uv-plane for the EoR"
             " model. Defaults to the value of '--nu'."
    )
    parser.add_argument(
        "--fov-ra-eor",
        type=float,
        help="Field of view of the Right Ascension (RA) axis of the EoR sky "
             "model in degrees."
    )
    parser.add_argument(
        "--fov-dec-eor",
        type=float,
        help="Field of view of the Declination (DEC) axis of the EoR sky model"
             " in degrees.  Defaults to the value of `--fov-ra-eor`."
    )
    # FG model params
    parser.add_argument(
        "--nu-fg",
        type=int,
        help="Number of pixels on the u-axis of the model uv-plane for the FG"
             " model.  Defaults to the value of '--nu'."
    )
    parser.add_argument(
        "--nv-fg",
        type=int,
        help="Number of pixels on the v-axis of the model uv-plane for the FG"
             " model. Defaults to the value of '--nu-fg' or '--nv' if "
             "'--nu-fg' is not defined."
    )
    parser.add_argument(
        "--fov-ra-fg",
        type=float,
        help="Field of view of the Right Ascension (RA) axis of the FG sky "
             "model in degrees.  Defaults to the value of '--fov-ra-eor'."
    )
    parser.add_argument(
        "--fov-dec-fg",
        type=float,
        help="Field of view of the Declination (DEC) axis of the FG sky model"
             " in degrees.  Defaults to the value of '--fov-ra-fg' or "
             "'--fov-dec-eor' if '--fov-ra-fg' is not defined."
    )
    parser.add_argument(
        "--fit-for-monopole",
        action="store_true",
        help="If passed, include the (u, v) = (0, 0) pixel in the "
             "model uv-plane."
    )
    parser.add_argument(
        "--nq",
        type=int,
        default=0,
        help="Number of Large Spectral Scale Model (LSSM) quadratic basis "
             "vectors.  If passing '--beta', the quadratic basis vectors are "
             "replaced by power law basis vectors according to the spectral "
             "indices in '--beta'."
    )
    parser.add_argument(
        "--beta",
        type=List[float],
        default=[2.63, 2.82],
        help="Brightness temperature power law spectral index/indices used in "
             "the LSSM.  Can be a single spectral index ('[2.63]') or multiple "
             "spectral indices can be passed ('[2.63,2.82]') to use multiple "
             "power law spectral basis vectors.  Do not put spaces after "
             "commas if using multiple spectral indices."
    )
    parser.add_argument(
        "--inverse-lw-power",
        type=float,
        default=1e-16,
        dest="inverse_LW_power",
        help="Prior on the inverse power of the LSSM coefficients.  A large "
             "value (1e16) constrains the LSSM coefficients to be zero.  A "
             "small value (1e-16) leaves the LSSM coefficients unconstrained "
             "(default)."
    )
    parser.add_argument(
        "--lw-gaussian-priors",
        action="store_true",
        dest="use_LWM_Gaussian_prior",
        help="If passed, uses a Gaussian prior on the LSSM (NOT IMPLEMENTED). "
             "Otherwise, use a uniform prior (default)."
    )
    parser.add_argument(
        "--fit-for-spectral-model-parameters",
        action="store_true",
        help="Fit for the optimal LSSM spectral indices."
    )
    parser.add_argument(
        "--pl-min",
        type=float,
        help="Minimum brightness temperature spectral index when fitting for "
             "the optimal LSSM spectral indices."
    )
    parser.add_argument(
        "--pl-max",
        type=float,
        help="Maximum brightness temperature spectral index when fitting for "
             "the optimal LSSM spectral indices."
    )
    parser.add_argument(
        "--pl-grid-spacing",
        type=float,
        help="Grid spacing for the power law spectral index axis when fitting "
             "for the optimal LSSM spectral indices."
    )
    # Noise model
    parser.add_argument(
        "--sigma",
        type=float,
        help="RMS of the visibility noise."
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=742123,
        help="Seed for numpy.random. Used to generate the noise vector. "
             "Defaults to 742123."
    )
    parser.add_argument(
        "--fit-noise",
        action="store_true",
        dest="use_intrinsic_noise_fitting",
        help="If passed, fit for the noise level."
    )
    # Instrument model
    parser.add_argument(
        "--model-instrument",
        action=ActionYesNo(yes_prefix="model-", no_prefix="no-"),
        dest="include_instrumental_effects",  #FIXME
        help="Forward model an instrument (--model-instrument) or don't "
             "include instrumental effects (--no-instrument)."
    )
    parser.add_argument(
        "--nt",
        type=int,
        help="Number of times."
    )
    parser.add_argument(
        "--dt",
        type=float,
        dest="integration_time_seconds",
        help="Time between observations in seconds."
    )
    parser.add_argument(
        "--inst-model",
        type=Path_dr,
        help="Path to a numpy compatible dictionary containing the instrument "
             "model.  A valid instrument model contains the uv-sampling with "
             "shape (Ntimes, Nbls, 3) accessible via the key 'uvw_model' and a"
             " redundancy array to account for any redundant averaging with "
             "shape (Ntimes, Nbls, 1) accessible via the key "
             "'redundancy_model'."
    )
    parser.add_argument(
        "--tele-latlonalt",
        type=List[float],
        dest="telescope_latlonalt",
        default=[-30.72152777777791, 21.428305555555557, 1073.0000000093132],
        help="Telescope location in latitude (deg), longitude (deg), and "
             "altitude (meters).  Passed as a list of floats, e.g. "
             "'--tele-latlonalt=[-30.1,125.6,80.4]'.  Do not put spaces after "
             "commas.  Defaults to the HERA telescope location."
    )
    parser.add_argument(
        "--central-jd",
        type=float,
        help="Central Julian Date of the observation."
    )
    parser.add_argument(
        "--beam-type",
        type=str,
        help="Can be 'uniform' , 'gaussian', or 'airy' (case insensitive)."
    )
    parser.add_argument(
        "--beam-peak-amplitude",
        type=float,
        default=1.0,
        help="Peak amplitude of the beam."
    )
    parser.add_argument(
        "--beam-center",
        type=List[float],
        help="Beam center offsets from the phase center in right ascension "
             "and declination in degrees.  Default behavior is the beam "
             "center aligns with the phase center.  Passed as a list of "
             "floats, e.g. '[-1.3,0.01]'.  Do not put a space after the comma."
    )
    parser.add_argument(
        "--drift-scan",
        action=ActionYesNo(yes_prefix="drift-scan", no_prefix="phased"),
        default=True,
        help="Model the instrument in drift scan mode (--drift-scan) or "
             "in phased mode (--phased).  Defaults to drift scan mode."
    )
    parser.add_argument(
        "--fwhm-deg",
        type=float,
        help="Full Width at Half Maximum (FWHM) of beam in degrees."
    )
    parser.add_argument(
        "--antenna-diameter",
        type=float,
        help="Antenna diameter in meters used for Airy beam calculations."
    )
    parser.add_argument(
        "--cosfreq",
        type=float,
        help="Cosine frequency if using a 'gausscosine' beam."
    )
    parser.add_argument(
        "--achromatic-beam",
        action="store_true",
        help="If passed, force the beam to be achromatic.  The frequency at "
             "which the beam will be calculated is set via '--beam-ref-freq'."
    )
    parser.add_argument(
        "--beam-ref-freq",
        type=float,
        help="Beam reference frequency in MHz.  Used for achromatic beams.  "
             "Defaults to the minimum frequency."
    )
    # --- Subharmonic Grid Params ---
    parser.add_argument(
        "--nu-sh",
        type=int,
        help="Number of pixels on a side for the u-axis in the SHG "
             "model uv-plane."
    )
    parser.add_argument(
        "--nv-sh",
        type=int,
        help="Number of pixels on a side for the v-axis in the SHG "
             "model uv-plane."
    )
    parser.add_argument(
        "--nq-sh",
        type=int,
        help="Number of LSSM quadratic modes for the SHG."
    )
    parser.add_argument(
        "--fit-for-shg-amps",
        action="store_true",
        help="If passed, fit for the amplitudes of the SHG pixels."
    )
    # --- Tapering params ---
    parser.add_argument(
        "--taper-func",
        type=str,
        help="Tapering function to apply to the frequency axis of the model "
             "visibilities.  Can be any valid argument to "
             "`scipy.signal.windows.get_window`."
    )
    # --- Data params ---
    parser.add_argument(
        "--data-path",
        type=Path_fr,
        help="Path to numpy readable visibility data file in mK sr."
    )
    parser.add_argument(
        "--noise-data-path",
        type=Path_fr,
        help="Path to noise file associated with data_path argument."
    )
    parser.add_argument(
        "--eor-sim-path",
        type=Path_fr,
        help="Path to 21cmFAST EoR simulation cube."
    )
    parser.add_argument(
        "--eor-random-seed",
        type=int,
        default=892736,
        help="Used to seed numpy.random if generating a mock EoR white noise "
             "signal, i.e. passing no data file via --data-path."
    )
    # --- Config file ---
    parser.add_argument(
        "--config",
        action=ActionConfigFile
    )
    args = parser.parse_args()

    if args.beam_type:
        args.beam_type = args.beam_type.lower()
    if args.taper_func:
        args.taper_func = args.taper_func.lower()
    if args.achromatic_beam and not args.beam_ref_freq:
        args.beam_ref_freq = args.nu_min_MHz

    # Update args with derived parameters
    args = calculate_derived_params(args)

    return args


def calculate_derived_params(args):
    """
    Calculate analysis parameters derived from command line arguments.

    Parameters
    ----------
    args : Namespace
        Namespace object containing command line arguments from
        `bayeseor.params.command_line_arguments.BayesEoRParser`.
    
    Returns
    -------
    args : Namespace
        Updated Namespace containing derived parameters.

    """
    cosmo = Cosmology()

    # --- Frequency ---
    args.freqs_MHz = (
        args.nu_min_MHz + np.arange(args.nf) * args.channel_width_MHz
    )
    args.bandwidth_MHz = args.channel_width_MHz * args.nf
    args.redshift = cosmo.f2z((args.freqs_MHz.mean() * units.MHz).to('Hz'))
    if not args.neta:
        args.neta = args.nf
    # Spacing along the eta axis (line-of-sight Fourier dual to frequency)
    # defined as one over the bandwidth in Hz [Hz^{-1}].
    args.deta = 1 / (args.nf * args.channel_width_MHz * 1e6)
    # Comoving line-of-sight size of the EoR volume [Mpc^{-1}]
    args.ps_box_size_para_Mpc = (
        cosmo.dL_df(args.redshift) * (args.bandwidth_MHz * 1e6)
    )

    # --- EoR Model ---
    if not args.nv:
        args.nv = args.nu
    # Number of model uv-plane pixels.  The (u, v) = (0, 0) pixel is part
    # of the FG model only, thus we calculate nuv for the EoR model as
    # `nuv = nu * nv - 1`.
    args.nuv = args.nu * args.nv - 1
    if not args.fov_dec_eor:
        args.fov_dec_eor = args.fov_ra_eor
    # Spacing along the u-axis of the model uv-plane [rad^{-1}]
    args.du_eor = 1 / np.deg2rad(args.fov_ra_eor)
    # Spacing along the v-axis of the model uv-plane [rad^{-1}]
    args.dv_eor = 1 / np.deg2rad(args.fov_dec_eor)
    # Comoving transverse size of the EoR volume [Mpc^{-1}]
    args.ps_box_size_ra_Mpc = (
        cosmo.dL_dth(args.redshift) * np.deg2rad(args.fov_ra_eor)
    )
    args.ps_box_size_dec_Mpc = (
        cosmo.dL_dth(args.redshift) * np.deg2rad(args.fov_dec_eor)
    )

    # --- FG Model ---
    if not args.nu_fg:
        args.nu_fg = args.nu
        args.nv_fg = args.nv
    elif not args.nv_fg:
        args.nv_fg = args.nu_fg
    # Number of model uv-plane pixels.  Exclude the (u, v) = (0, 0)
    # (monopole) pixel if `fit_for_monopole` is False.
    args.nuv_fg = args.nu_fg * args.nv_fg - (not args.fit_for_monopole)
    if not args.fov_ra_fg:
        args.fov_ra_fg = args.fov_ra_eor
        args.fov_dec_fg = args.fov_dec_eor
    elif not args.fov_dec_fg:
        args.fov_dec_fg = args.fov_ra_fg
    # Spacing along the u-axis of the model uv-plane [rad^{-1}]
    args.du_fg = 1 / np.deg2rad(args.fov_ra_fg)
    # Spacing along the v-axis of the model uv-plane [rad^{-1}]
    args.dv_fg = 1 / np.deg2rad(args.fov_dec_fg)
    # Number of power law spectral indices in the Large Spectral Scale Model
    if args.beta:
        args.npl = len(args.beta)  #FIXME: update if fitting for LSSM
    else:
        args.npl = 0
    if args.nq > args.npl:
        args.nq = args.npl
    if not args.include_instrumental_effects:
        # If not modelling instrumental effects, we will have fewer data points
        # and need to reduce the number of model parameters to avoid an
        # under-constrained system
        args.neta -= args.nq
    
    # --- Sub-Harmonic Grid ---
    shg_params = [args.nu_sh, args.nv_sh, args.nq_sh]
    args.use_shg = np.any([arg is not None for arg in shg_params])
    if args.use_shg:
        if not args.nv_sh:
            args.nv_sh = args.nu_sh
        if args.beta:
            args.npl_sh = len(args.beta)  #FIXME: update if fitting for LSSM
        if args.nq_sh > args.npl_sh:
            args.nq_sh = args.npl_sh
        args.nuv_sh = args.nu_sh * args.nv_sh - 1
    else:
        args.npl_sh = None
        args.nuv_sh = None
    
    # --- Instrument Model ---
    if args.achromatic_beam and not args.beam_ref_freq:
        args.beam_ref_freq = args.nu_min_MHz

    # --- Auxiliary ---
    args.speed_of_light = constants.c.to("m/s").value
    
    return args


def parse_uprior_inds(upriors_str, nkbins):
    """
    Parse a string containing array indexes.

    `upriors_str` must follow standard array slicing syntax and include no
    spaces.  Examples of valid strings:
    * '1:4': equivalent to `slice(1, 4)`
    * '1,3,4': equivalent to indexing with `[1, 3, 4]`
    * '3' or '-3'
    * 'all'

    Parameters
    ----------
    upriors_str : str
        String containing array indexes (follows array slicing syntax).
    nkbins : int
        Number of k-bins.

    Returns
    -------
    uprior_inds : array
        Boolean array that is True for any k-bins using a uniform prior.
        False entries use a log-uniform prior.

    """
    if upriors_str.lower() == 'all':
        uprior_inds = np.ones(nkbins, dtype=bool)
    else:
        uprior_inds = np.zeros(nkbins, dtype=bool)
        if ':' in upriors_str:
            bounds = list(map(int, upriors_str.split(':')))
            uprior_inds[slice(*bounds)] = True
        elif ',' in upriors_str:
            up_inds = list(map(int, upriors_str.split(',')))
            uprior_inds[up_inds] = True
        else:
            uprior_inds[int(upriors_str)] = True

    return uprior_inds
