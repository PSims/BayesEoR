import argparse
import BayesEoR.Params.params as p
from BayesEoR.Utils.Utils import mpiprint


def BayesEoRParser():
    """
    Class used to parse command line arguments.

    """
    parser = argparse.ArgumentParser()
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
             " model. Defaults to `nu`."
    )
    parser.add_argument(
        "--fov_ra_eor",
        type=float,
        help="Field of view of the Right Ascension (RA) axis of the EoR sky "
             "model in degrees."
    )
    parser.add_argument(
        "--fov_dec_eor",
        type=float,
        help="Field of view of the Declination (DEC) axis of the EoR sky model"
             " in degrees."
    )
    parser.add_argument(
        "--nside",
        type=int,
        help="HEALPix resolution parameter. Sets the resolution of the "
             "sky model."
    )
    parser.add_argument(
        "--nu_fg",
        type=int,
        help="Number of pixels on the u-axis of the model uv-plane for the FG"
             " model."
    )
    parser.add_argument(
        "--nv_fg",
        type=int,
        help="Number of pixels on the v-axis of the model uv-plane for the FG"
             " model. Defaults to `nu_fg`."
    )
    parser.add_argument(
        "--fov_ra_fg",
        type=float,
        help="Field of view of the Right Ascension (RA) axis of the FG sky "
             "model in degrees."
    )
    parser.add_argument(
        "--fov_dec_fg",
        type=float,
        help="Field of view of the Declination (DEC) axis of the FG sky model"
             " in degrees."
    )
    parser.add_argument(
        "--fit_for_monopole",
        action="store_true",
        help="If passed, include the (u, v) = (0, 0) pixel in the "
             "model uv-plane."
    )
    parser.add_argument(
        "--nq",
        type=int,
        help="Number of Large Spectral Scale Model (LSSM) basis vectors."
    )
    parser.add_argument(
        "--beta",
        help="Power law spectral index used in data model"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        help="RMS of the visibility noise."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to numpy readable visibility data file in mK sr.",
        default=None
    )
    parser.add_argument(
        "--noise_data_path",
        type=str,
        help="Path to noise file associated with data_path argument.",
        default=None
    )
    parser.add_argument(
        "--noise_seed",
        type=int,
        help="Seed for numpy.random. Used to generate the noise vector. "
             "Defaults to 742123.",
        default=742123
    )
    parser.add_argument(
        "--beam_type",
        type=str,
        help="Can be 'gaussian', 'uniform', or 'airy' (case insensitive)."
    )
    parser.add_argument(
        "--beam_peak_amplitude",
        type=float,
        help="Peak amplitude of the beam."
    )
    parser.add_argument(
        "--fwhm_deg",
        type=float,
        help="Full Width at Half Maximum (FWHM) of beam in degrees."
    )
    parser.add_argument(
        "--antenna_diameter",
        type=float,
        help="Antenna diameter in meters used for Airy beam calculations."
    )
    parser.add_argument(
        "--cosfreq",
        type=float,
        help="Cosine frequency if using a 'gausscosine' beam."
    )
    parser.add_argument(
        "--overwrite_matrices",
        action="store_true",
        default=False,
        dest="overwrite_matrices",
        help="If passed, overwrite existing matrix stack."
    )
    parser.add_argument(
        "--beam_center",
        type=str,
        help="Sets the beam pointing center in (RA, DEC) relative to "
             "the pointing center of the sky model defined by "
             "`p.telescope_latlonalt` and `p.central_jd` at zenith. "
             "Must be set via a string argument as "
             "--beam_center=\"(RA_offset,DEC_offset)\" with no spaces."
    )
    parser.add_argument(
        "--phased",
        action="store_true",
        help="If passed, the data are treated as phased."
    )
    parser.add_argument(
        "--uprior_bins",
        type=str,
        default="",
        help="Array indices of k-bins using a uniform prior.  Follows python "
             "slicing syntax.  Can pass a range via '1:4' (non-inclusive high "
             "end), a list of indices via '1,4,6' (no spaces between commas), "
             " a single index '3' or '-3', or 'all'.  Defaults to an empty "
             "string (all k-bins use log-uniform priors)."
    )
    parser.add_argument(
        "--file_root",
        type=str,
        help="Sets the file root for the sampler (Multinest/Polychord) output."
             " If passed, analysis will continue from the last checkpoint in "
             "the output file specified."
    )
    parser.add_argument(
        "--use_shg",
        action="store_true",
        help="If passed, use the SubHarmonic Grid (SHG)."
    )
    parser.add_argument(
        "--nu_sh",
        type=int,
        help="Number of pixels on a side for the u-axis in the SHG "
             "model uv-plane."
    )
    parser.add_argument(
        "--nv_sh",
        type=int,
        help="Number of pixels on a side for the v-axis in the SHG "
             "model uv-plane."
    )
    parser.add_argument(
        "--nq_sh",
        type=int,
        help="Number of LSSM quadratic modes for the SHG."
    )
    parser.add_argument(
        "--npl_sh",
        type=int,
        help="Number of power law coefficients used in the LSSM for the "
             "subharmonic grid."
    )
    parser.add_argument(
        "--fit_for_shg_amps",
        action="store_true",
        help="If passed, fit for the amplitudes of the SHG pixels."
    )
    parser.add_argument(
        "--taper_func",
        type=str,
        default=None,
        help="Tapering function to apply to the frequency axis of the model "
             "visibilities.  Can be any valid argument to "
             "`scipy.signal.windows.get_window`."
    )

    args = parser.parse_args()
    return args


def update_params_with_command_line_arguments(rank=0):
    """
    Updates variable values stored in `BayesEoR.Params.params` using command
    line arguments.  Command line arguments will overwrite any existing value
    of a variable stored in `p` where `p` is a module imported with
    `import BayesEoR.Params.params as p`.

    Parameters
    ----------
    rank : int
        MPI rank.  Only prints messages if rank == 0.

    """
    args = BayesEoRParser()
    cla_keys = [
        key for key in args.__dict__.keys()
        if not (key[0] == '_' and key[-1] == '_')]
    params_keys = [
        key for key in p.__dict__.keys()
        if not (key[0] == '_' and key[-1] == '_')]

    for key in cla_keys:
        if not args.__dict__[key] is None:
            if key == 'beta':
                if type(args.beta) == str:
                    if args.beta.count('[') and args.beta.count(']'):
                        # Overwrite parameter file beta with value
                        # chosen from the command line if it is included
                        p.beta = args.beta.replace('[', '').replace(']', '')
                        p.beta = map(float, p.beta.split(','))
                        # Overwrites quadratic term when nq=2,
                        # otherwise unused.
                        p.npl = len(p.beta)
                    else:
                        # Overwrite parameter file beta with value
                        # chosen from the command line if it is included
                        p.beta = float(args.beta)
                        p.npl = 1
                    mpiprint(
                        'Overwriting params p.{} = {} with command line '
                        'argument {} = {}'.format(
                            key, p.__dict__[key], key, args.__dict__[key]
                        ),
                        rank=rank
                    )
                else:
                    mpiprint(
                        'No value for beta(s) given, using defaults.',
                        rank=rank
                    )
            else:
                if key in params_keys:
                    mpiprint(
                        'Overwriting params p.{} = {} with command line '
                        'argument {} = {}'.format(
                            key, p.__dict__[key], key, args.__dict__[key]
                        ),
                        rank=rank
                    )
                else:
                    mpiprint(
                        'Setting params p.{} = {}'.format(
                            key, args.__dict__[key]
                        ),
                        rank=rank
                    )
                p.__dict__[key] = args.__dict__[key]
