""" Analysis settings """
import numpy as np
from copy import deepcopy
from astropy import constants
from astropy import units
from jsonargparse import ArgumentParser, ActionYesNo, ActionConfigFile
from jsonargparse.typing import List, Path_fr, path_type

from .cosmology import Cosmology


class BayesEoRParser(ArgumentParser):
    """
    Command line argument parser class for all BayesEoR analysis parameters.

    For more information on the command line syntax, please see

    .. code-block:: bash

        python run-analysis.py --help

    Attributes
    ----------

    General Parameters

    config : str, optional
        Path to a yaml configuration file containing any command line arguments
        below parsable by `jsonargparse.ArgumentParser`. Please note command
        line arguments use dashes '-' but the configuration yaml file requires
        underscores '_'.  For example, to specify the array directory prefix
        on the command line, the command line argument is `--array-dir-prefix`.
        To specify the array directory prefix in the configuration yaml file,
        use `array_dir_prefix` instead.  For an example, please see the
        provided configuration yaml file `BayesEoR/example-config.yaml`.
    verbose : bool, optional
        Verbose output. Defaults to False.
    clobber : bool, optional
        Overwrite files if they exist. This includes the matrix stack, any
        existing preprocessed data/noise vectors, and instrument model files.
        Defaults to False.

    Compute Parameters

    use_gpu : bool, optional
        Use GPUs (True) or CPUs (False). Defaults to True.
    run : bool, optional
        Run a full power spectrum analysis. To build the matrix stack only,
        run can be omitted or set to False in the configuration yaml.

    Matrix Parameters

    array_dir_prefix : str, optional
        Directory for matrix storage. Defaults to './array-storage'.
    use_sparse_matrices : bool, optional
        Use sparse matrices (True) to reduce storage requirements or dense
        matrices (False). Defaults to True.
    build_Finv_and_Fprime : bool, optional
        If True, construct Finv and Fprime independently and write 
        both matrices to disk when building the matrix stack. 
        Otherwise (default), construct the matrix product Finv_Fprime in
        place from the dense matrices comprising Finv and Fprime to minimize 
        the memory and time required to build the matrix stack.  In this
        case, only the matrix product Finv_Fprime is written to disk.
    
    Prior Parameters

    priors : list of list
        Power spectrum prior range [min, max] for each k bin.
    log_priors : bool, optional
        Assume priors on power spectrum coefficients are in log_10 units
        (True) or linear units (False). Defaults to True.
    uprior_bins : str, optional
        Array indices of k-bins using a uniform prior.  All other bins use the
        default log-uniform prior.  Follows python slicing syntax.  Can pass a
        range via '1:4' (non-inclusive high end), a list of indices via '1,4,6'
        (no spaces between commas), a single index '3' or '-3', or 'all'.
        Defaults to an empty string (all k-bins use log-uniform priors).
    inverse_LW_power : float, optional
        Prior on the inverse power of the large spectral scale model (LSSM)
        coefficients. A large value, 1e16, constrains the LSSM coefficients to
        be zero. A small value, 1e-16 (default), leaves the LSSM coefficients
        unconstrained.
    use_LWM_Gaussian_prior: bool, optional
        Use a Gaussian prior on the large spectral scale model (True,
        **NOT CURRENTLY IMPLEMENTED**). Otherwise, use a uniform prior (False,
        default)

    Sampler Parameters

    output_dir : str, optional
        Directory for sampler output. Defaults to './chains/'.
    file_root : str, optional
        If None (default), start a new analysis. Otherwise, resume analysis
        from `file_root`.
    use_Multinest : bool, optional
        Use MultiNest (True) or Polychord (False) as the sampler. Using
        Polychord is advised for large parameter spaces. Defaults to True
        (use MultiNest).
        
    Frequency Parameters

    nf : int, optional
        Number of frequency channels. Required if `data_path` points to a
        preprocessed data vector with a '.npy' suffix. Otherwise, `nf` sets
        the number of frequencies to keep starting from `freq_idx_min` or
        `freq_min`, or around `freq_center`. Defaults to None (keep all
        frequencies).
    df : float, optional
        Frequency channel width in hertz.  Required if `data_path` points
        to a preprocessed numpy-compatible file. Otherwise, if None (default),
        defaults to the frequency channel width in the input
        pyuvdata-compatible visibilities. Defaults to None.
    freq_idx_min : int, optional
        Minimum frequency channel index to keep in the data vector. Used only
        if `data_path` points to a pyuvdata-compatible visibility file.
        Defaults to None (keep all frequencies).
    freq_min : float, optional
        Minimum frequency in hertz.  If `data_path` points to a
        pyuvdata-compatible visibility file, `freq_min` sets the minimum
        frequency kept in the data vector.  All frequencies greater than or
        equal to `freq_min` will be kept, unless `nf` is specified. If None
        (default), all frequencies are kept. If `data_path` points to a
        preprocessed data vector with a '.npy' suffix, one of `freq_min` or
        `freq_center` is required.
    freq_center : float, optional
        Central frequency in Hertz. If `data_path` points to a
        pyuvdata-compatible visibility file, `nf` is also required to
        determine the number of frequencies kept around `freq_center` in the
        data vector. If None (default), all frequencies are kept. If
        `data_path` points to a preprocessed data vector with a '.npy' suffix,
        one of `freq_min` or `freq_center` is required.
    neta : int, optional
        Number of line-of-sight Fourier modes. Defaults to `nf`.
    nq : int, optional
        Number of large spectral scale model quadratic basis vectors.
        If `beta` is not None, the quadratic basis vectors are replaced by
        power law basis vectors according to the spectral indices in `beta`.
        Defaults to 0.
    beta : list of float, optional
        Brightness temperature power law spectral index/indices used in the
        large spectral scale model. Can be a single spectral index, '[2.63]',
        or multiple spectral indices can be passed, '[2.63,2.82]', to use
        multiple power law spectral basis vectors. Do not put spaces after
        commas if using multiple spectral indices. Defaults to [2.63, 2.82].
    fit_for_spectral_model_parameters : bool, optional
        Fit for the optimal large spectral scale model spectral indices.
        Defaults to False.
    pl_min : float, optional
        Minimum brightness temperature spectral index when fitting for the
        optimal large spectral scale model spectral indices. Defaults to None.
    pl_max : float, optional
        Maximum brightness temperature spectral index when fitting for the
        optimal large spectral scale model spectral indices. Defaults to None.
    pl_grid_spacing : float, optional
        Grid spacing for the power law spectral index axis when fitting for
        the optimal large spectral scale model spectral indices. Defaults to
        None.

    Time Parameters

    nt : int, optional
        Number of times. Required if `data_path` points to a preprocessed data
        vector with a '.npy' suffix. Otherwise, sets the number of times to
        keep starting from `jd_idx_min` or `jd_min`, or around `jd_center`.
        Defaults to None (keep all times).
    dt : float, optional
        Integration time in seconds. Required if `data_path` points to a
        preprocessed data vector with a '.npy' suffix. Otherwise, if None
        (default), defaults to the integration time in the input
        pyuvdata-compatible visibilities. Defaults to None.
    jd_idx_min : int, optional
        Minimum time index to keep in the data vector. Used only if `data_path`
        points to a pyuvdata-compatible visibility file. Defaults to None (keep
        all times). Defaults to None (keep all times).
    jd_min : float, optional
        Minimum time as a Julian date. If `data_path` points to a
        pyuvdata-compatible visibility file, `jd_min` sets the minimum time
        kept in the data vector.  All times greater than or equal to `jd_min`
        will be kept, unless `nt` is specified. If None (default), all times
        are kept. If `data_path` points to a preprocessed data vector with a
        '.npy' suffix, one of `jd_min` or `jd_center` is required.
    jd_center : float, optional
        Central time as a Julian date. If `data_path` points to a
        pyuvdata-compatible visibility file, `nt` is also required to
        determine the number of times kept around `jd_center` in the data
        vector. If None (default), all times are kept. If `data_path` points
        to a preprocessed data vector with a '.npy' suffix, one of `jd_min` or
        `jd_center` is required.

    Model Image Parameters

    nside : int
        HEALPix resolution parameter.  Sets the resolution of the sky model.
        Note, the HEALPix resolution must be chosen such that there are two
        HEALPix pixels per minimum fringe wavelength from the model uv-plane
        to satisfy the Nyquist-Shannon sampling theorem.
    fov_ra_eor : float
        Field of view of the right ascension axis of the EoR sky model in
        degrees.
    fov_dec_eor : float, optional
        Field of view of the declination axis of the EoR sky model in
        degrees. Defaults to `fov_ra_eor`.
    fov_ra_fg : float, optional
        Field of view of the right ascension axis of the foreground sky model
        in degrees.  Defaults to `fov_ra_eor`.
    fov_dec_fg : float, optional
        Field of view of the declination axis of the foreground sky model in
        degrees.  Defaults to `fov_ra_fg` or `fov_dec_eor` if `fov_ra_fg` is
        not defined.
    simple_za_filter : bool, optional
        Filter pixels in the sky model by zenith angle only (True, default).
        Otherwise, filter pixels in a rectangular region set by the field of
        view values along the RA and Dec axes (False). It is suggested to
        set `simple_za_filter` to True (see issue #11).

    Model uv-Plane Parameters

    nu : int
        Number of pixels on the u-axis of the model uv-plane for the EoR model.
    nv : int, optional
        Number of pixels on the v-axis of the model uv-plane for the EoR model.
        Defaults to `nu`.
    nu_fg : int, optional
        Number of pixels on the u-axis of the model uv-plane for the foreground
        model.  Defaults to `nu`.
    nv_fg : int, optional
        Number of pixels on the v-axis of the model uv-plane for the foreground
        model. Defaults to `nu_fg` or `nv` if `nu_fg` is not defined.
    fit_for_monopole : bool, optional
        Include (True) or exclude (False) the ``(u, v) == (0, 0)`` pixel in
        the model uv-plane. Defaults to False.

    Noise Model Parameters

    sigma : float, optional
        Standard deviation of the visibility noise in mK sr. Required if
        `calc_noise` is False and `data_path` points to a pyuvdata-compatible
        visibility file or `noise_data_path` is None and `data_path` points to
        a preprocessed numpy-compatible visibility vector. Defaults to None.
    noise_seed : float, optional
        Seed for `numpy.random`. Used to generate the noise vector if adding
        noise to the input visibility data. Defaults to 742123.
    use_intrinsic_noise_fitting : bool, optional
        Fit for the noise level. Defaults to False.

    Instrument Model Parameters

    model_instrument : bool, optional
        Forward model an instrument (True) or exclude instrumental effects
        (False). Defaults to True.
    inst_model : str, optional
        Path to a directory containing the instrument model. This directory
        must at least contain two numpy-compatible files: `uvw_model.npy`
        containing the sampled (u, v, w) coordinates with shape
        (nt, nbls, 3) where nbls is the number of baselines, and
        `redundancy_model.npy` containing the number of baselines in a each
        sampled (u, v, w) with shape (nt, nbls, 1). Please see
        `bayeseor.model.instrument` for more details. Used only if
        `include_instrumental_effects` is True. Defaults to None.
    telescope_latlonalt : list of float, optional
        Telescope location in latitude (deg), longitude (deg), and altitude
        (meters). Passed as a list of floats, e.g. '[30.1,125.6,80.4]'. Do not
        put spaces after commas. Required if `include_instrumental_effects`
        is True. Defaults to None.
    drift_scan : bool, optional
        Model the instrument in drift scan mode (True) or in phased mode
        (False). Used only if `include_instrumental_effects` is True. Defaults
        to True.
    beam_type : str
        Path to a pyuvdata-compatible beam file or one of 'uniform',
        'gaussian', 'airy', 'gausscosine', or 'taperairy'. Used only if
        `include_instrumental_effects` is True. Defaults to None.
    fwhm_deg : float, optional
        Full width at half maximum of beam in degrees. Used only if
        `include_instrumental_effects` is True and `beam_type` is 'airy',
        'gaussian', or 'gausscosine'. Defaults to None.
    antenna_diameter : float, optional
        Antenna (aperture) diameter in meters. Used only if
        `include_instrumental_effects` is True and `beam_type` is 'airy',
        'gaussian', or 'gausscosine'. Defaults to None.
    cosfreq : float, optional
        Cosine frequency if using a 'gausscosine' beam. Used only if
        `include_instrumental_effects` is True. Defaults to None.
    achromatic_beam : bool, optional
        Force the beam to be achromatic. The frequency at which the beam will
        be calculated is set via `beam_ref_freq`. Used only if
        `include_instrumental_effects` is True. Defaults to False.
    beam_ref_freq : bool, optional
        Beam reference frequency in hertz. Used only if
        `include_instrumental_effects` is True. Defaults to `freq_min`.
    beam_peak_amplitude : float, optional
        Peak amplitude of the beam. Used only if `include_instrumental_effects`
        is True. Defaults to 1.0.
    beam_center : list of float, optional
        Beam center offsets from the phase center in RA and Dec in degrees.
        Default behavior is the beam center aligns with the phase center.
        Passed as a list of floats, e.g. '[-1.3,0.01]'. Do not include a space
        after the comma. Defaults to None.

    Subharmonic Grid Parameters

    nu_sh : int, optional
        Number of pixels on a side for the u-axis in the subharmonic grid
        model uv-plane. Defaults to None.
    nv_sh : int, optional
        Number of pixels on a side for the v-axis in the subharmonic grid
        model uv-plane. Defaults to `nu_sh`.
    fit_for_shg_amps : bool, optional
        Fit for the amplitudes of the subharmonic grid pixels. Defaults to
        False.
    
    Tapering Parameters

    taper_func : str, optional
        Taper function applied to the frequency axis of the visibilities.
        Can be any valid argument to `scipy.signal.windows.get_window`.
        Defaults to None.
    
    Input Data Parameters

    data_path : str
        Path to either a pyuvdata-compatible visibility file or a preprocessed
        numpy-compatible visibility vector in units of mK sr.
    noise_data_path : str, optional
        Path to a preprocessed numpy-compatible noise visibility vector in
        units of mK sr.  Required if `calc_noise` is False and `sigma` is None.
    ant_str : str, optional
        Antenna downselect string. If `data_path` points to a
        pyuvdata-compatible visibility file, `ant_str` determines what
        baselines to keep in the data vector. Please see
        `pyuvdata.UVData.select` for more details. Defaults to 'cross'
        (cross-correlation baselines only).
    bl_cutoff : float, optional
        Baseline length cutoff in meters. If `data_path` points to a
        pyuvdata-compatible visibility file, `bl_cutoff` determines the longest
        baselines kept in the data vector. Defaults to None (keep all
        baselines).
    form_pI : bool, optional
        Form pseudo-Stokes I visibilities. Otherwise, use the polarization
        specified by `pol`. Used only if `data_path` points to a
        pyuvdata-compatible visibility file. Defaults to True.
    pI_norm : float, optional
        Normalization, ``N``, used in forming pseudo-Stokes I from XX and YY
        via ``pI = N * (XX + YY)``. Used only if `data_path` points to a
        pyuvdata-compatible visibility file and `form_pI` is True. Defaults to
        1.0.
    pol : str, optional
        Case-insensitive polarization string. Can be one of 'xx', 'yy', or 'pI'
        for XX, YY, or pseudo-Stokes I polarization, respectively. Used only if
        `data_path` points to a pyuvdata-compatible visibility file and
        `form_pI` is False. Defaults to 'xx'.
    redundant_avg : bool, optional
        Redundantly average the data. Used only if `data_path` points to a
        pyuvdata-compatible visibility file. Defaults to False.
    uniform_redundancy : bool, optional
        Force the redundancy model to be uniform. Used only if `data_path`
        points to a pyuvdata-compatible visibility file. Defaults to False.
    phase_time : float, optional
        The time to which the visibilities will be phased as a Julian date.
        Used only if `drift_scan` is False.  If `drift_scan` is False and
        `phase_time` is None, `phase_time` will be automatically set to the
        central time in the data. Used only if `data_path` points to a
        pyuvdata-compatible visibility file. Defaults to None.
    calc_noise : bool, optional
        Calculate a noise estimate from the visibilities via differencing
        adjacent times per baseline and frequency. Used only if `data_path`
        points to a pyuvdata-compatible visibility file. Defaults to False.
    save_vis : bool, optional
        Write visibility vector to disk in `out_dir`. If `calc_noise` is True,
        also save the noise vector. Used only if `data_path` points to a
        pyuvdata-compatible visibility file. Defaults to False.
    save_model : bool, optional
        Write instrument model (antenna pairs, (u, v, w) sampling, and
        redundancy model) to disk in `out_dir`. If `phase` is True, also save
        the phasor vector. Used only if `data_path` points to a
        pyuvdata-compatible visibility file. Defaults to False.

    """
    def __init__(self, *args, parse_as_dict=False, **kwargs):
        super().__init__(*args, parse_as_dict=parse_as_dict, **kwargs)
        # General Parameters
        self.add_argument(
            "--config",
            action=ActionConfigFile,
            help="Path to a yaml configuration file containing any command line arguments"
                 "below parsable by `jsonargparse.ArgumentParser`. Please note command"
                 "line arguments use dashes '-' but the configuration yaml file requires"
                 "underscores '_'.  For example, to specify the array directory prefix"
                 "on the command line, the command line argument is `--array-dir-prefix`."
                 "To specify the array directory prefix in the configuration yaml file,"
                 "use `array_dir_prefix` instead.  For an example, please see the"
                 "provided configuration yaml file `BayesEoR/example-config.yaml`."
        )
        self.add_argument(
            "-v", "--verbose",
            action="store_true",
            dest="verbose",
            help="Verbose output."
        )
        self.add_argument(
            "--clobber",
            action="store_true",
            help="Overwrite files if they exist. This includes the matrix stack, any "
                 "existing preprocessed data/noise vectors, and instrument model files."
        )
        # Compute Parameters
        self.add_argument(
            "--gpu",
            action=ActionYesNo(yes_prefix="g", no_prefix="c"),
            default=True,
            dest="use_gpu",
            help="Use GPUs (--gpu) or CPUs (--cpu)."
        )
        self.add_argument(
            "--run",
            action="store_true",
            help="Run a full power spectrum analysis. To build the matrix "
                 "stack only, --run can be omitted or set to False in the "
                 "configuration yaml."
        )
        # Matrix Parameters
        self.add_argument(
            "--array-dir-prefix",
            type=str,
            default="./matrices/",
            help="Directory for matrix storage. Defaults to './matrices/'."
        )
        self.add_argument(
            "--sparse-mats",
            action=ActionYesNo(yes_prefix="sparse-", no_prefix="dense-"),
            dest="use_sparse_matrices",  # FIXME: remove need for dest
            default=True,
            help="Use sparse matrices (--sparse-mats) to reduce storage "
                 "requirements or dense matrices (--dense-mats)."
        )
        self.add_argument(
            "--build-Finv-and-Fprime",
            action="store_true",
            help="If passed, construct Finv and Fprime independently and write "
                 "both matrices to disk when building the matrix stack. "
                 "Otherwise (default), construct the matrix product Finv_Fprime "
                 "in place from the dense matrices comprising Finv and Fprime to "
                 "minimize the memory and time required to build the matrix "
                 "stack. In this case, only the matrix product Finv_Fprime is "
                 "written to disk."
        )
        # Prior Parameters
        self.add_argument(
            "--priors",
            type=List[list],
            help="Power spectrum prior range [min, max] for each k bin."
        )
        self.add_argument(
            "--log-priors",
            action=ActionYesNo(yes_prefix="log-", no_prefix="lin-"),
            default=True,
            help="Assume priors on power spectrum coefficients are in log_10 units "
                 "('--log-priors', True) or linear units ('--lin-priors', False)."
        )
        self.add_argument(
            "--uprior-bins",
            type=str,
            default="",
            help="Array indices of k-bins using a uniform prior.  Follows python "
                 "slicing syntax.  Can pass a range via '1:4' (non-inclusive high "
                 "end), a list of indices via '1,4,6' (no spaces between commas), "
                 " a single index '3' or '-3', or 'all'.  Defaults to an empty "
                 "string (all k-bins use log-uniform priors)."
        )
        self.add_argument(
            "--inverse-lw-power",
            type=float,
            default=1e-16,
            dest="inverse_LW_power",  # FIXME: remove need for dest
            help="Prior on the inverse power of the large spectral scale model (LSSM) "
                 "coefficients. A large value, 1e16, constrains the LSSM coefficients to "
                 "be zero. A small value, 1e-16 (default), leaves the LSSM coefficients "
                 "unconstrained."
        )
        self.add_argument(
            "--lw-gaussian-priors",
            action="store_true",
            dest="use_LWM_Gaussian_prior",  # FIXME: remove need for dest
            help="If passed, Use a Gaussian prior on the large spectral scale model "
                 "(**NOT CURRENTLY IMPLEMENTED**). Otherwise, use a uniform prior (default)."
        )
        self.add_argument(
            "--output-dir",
            type=str,
            default="./chains/",
            help="Directory for sampler output.  Defaults to './chains/'."
        )
        self.add_argument(
            "--file-root",
            type=str,
            help="If None (default), start a new analysis. Otherwise, resume "
                "analysis from `file_root`."
        )
        self.add_argument(
            "--multinest",
            action=ActionYesNo(yes_prefix="multinest", no_prefix="polychord"),
            default=True,
            dest="use_Multinest",  # FIXME: remove need for dest
            help="Use Multinest (--multinest) or Polychord (--polychord) as "
                 "the sampler. Defaults to using Multinest. Using Polychord is "
                 "advised for large parameter spaces."
        )
        # Frequency Parameters
        self.add_argument(
            "--nf",
            type=int,
            help="Number of frequency channels. Required if `data_path` points to a "
                 "preprocessed data vector with a '.npy' suffix. Otherwise, `nf` sets "
                 "the number of frequencies to keep starting from `freq_idx_min` or "
                 "`freq_min`, or around `freq_center`. Defaults to None (keep all "
                 "frequencies)."
        )
        self.add_argument(
            "--df",
            type=float,
            help="Frequency channel width in hertz.  Required if `data_path` points "
                 "to a preprocessed numpy-compatible file. Otherwise, if None (default), "
                 "defaults to the frequency channel width in the input "
                 "pyuvdata-compatible visibilities. Defaults to None."
        )
        self.add_argument(
            "--freq-idx-min",
            type=int,
            help="Minimum frequency channel index to keep in the data vector. Used only "
                 "if `data_path` points to a pyuvdata-compatible visibility file. "
                 "Defaults to None (keep all frequencies)."
        )
        self.add_argument(
            "--freq-min",
            type=float,
            help="Minimum frequency in hertz.  If `data_path` points to a "
                 "pyuvdata-compatible visibility file, `freq_min` sets the minimum "
                 "frequency kept in the data vector.  All frequencies greater than or "
                 "equal to `freq_min` will be kept, unless `nf` is specified. If None "
                 "(default), all frequencies are kept. If `data_path` points to a "
                 "preprocessed data vector with a '.npy' suffix, one of `freq_min` or "
                 "`freq_center` is required."
        )
        self.add_argument(
            "--freq-center",
            type=float,
            help="Central frequency in Hertz. If `data_path` points to a "
                 "pyuvdata-compatible visibility file, `nf` is also required to "
                 "determine the number of frequencies kept around `freq_center` in the "
                 "data vector. If None (default), all frequencies are kept. If "
                 "`data_path` points to a preprocessed data vector with a '.npy' suffix, "
                 "one of `freq_min` or `freq_center` is required."
        )
        self.add_argument(
            "--neta",
            type=int,
            help="Number of line-of-sight Fourier modes. Defaults to `nf`."
        )
        self.add_argument(
            "--nq",
            type=int,
            default=0,
            help="Number of large spectral scale model quadratic basis vectors. "
                 "If `beta` is not None, the quadratic basis vectors are replaced by "
                 "power law basis vectors according to the spectral indices in `beta`. "
                 "Defaults to 0."
        )
        self.add_argument(
            "--beta",
            type=List[float],
            default=[2.63, 2.82],
            help="Brightness temperature power law spectral index/indices used in the "
                 "large spectral scale model. Can be a single spectral index, '[2.63]', "
                 "or multiple spectral indices can be passed, '[2.63,2.82]', to use "
                 "multiple power law spectral basis vectors. Do not put spaces after "
                 "commas if using multiple spectral indices. Defaults to [2.63, 2.82]."
        )
        self.add_argument(
            "--fit-for-spectral-model-parameters",
            action="store_true",
            help="Fit for the optimal large spectral scale model spectral indices."
        )
        self.add_argument(
            "--pl-min",
            type=float,
            help="Minimum brightness temperature spectral index when fitting for "
                 "the optimal large spectral scale model spectral indices."
        )
        self.add_argument(
            "--pl-max",
            type=float,
            help="Maximum brightness temperature spectral index when fitting for "
                "the optimal large spectral scale model spectral indices."
        )
        self.add_argument(
            "--pl-grid-spacing",
            type=float,
            help="Grid spacing for the power law spectral index axis when fitting "
                "for the optimal large spectral scale model spectral indices."
        )
        # Time Parameters
        self.add_argument(
            "--nt",
            type=int,
            help="Number of times. Required if `data_path` points to a preprocessed data "
                 "vector with a '.npy' suffix. Otherwise, sets the number of times to "
                 "keep starting from `jd_idx_min` or `jd_min`, or around `jd_center`. "
                 "Defaults to None (keep all times)."
        )
        self.add_argument(
            "--dt",
            type=float,
            help="Integration time in seconds. Required if `data_path` points to a "
                 "preprocessed data vector with a '.npy' suffix. Otherwise, if None "
                 "(default), defaults to the integration time in the input "
                 "pyuvdata-compatible visibilities. Defaults to None."
        )
        self.add_argument(
            "--jd-idx-min",
            type=int,
            help="Minimum time index to keep in the data vector. Used only if `data_path` "
                 "points to a pyuvdata-compatible visibility file. Defaults to None (keep "
                 "all times). Defaults to None (keep all times)."
        )
        self.add_argument(
            "--jd-min",
            type=float,
            help="Minimum time as a Julian date. If `data_path` points to a "
                 "pyuvdata-compatible visibility file, `jd_min` sets the minimum time "
                 "kept in the data vector.  All times greater than or equal to `jd_min` "
                 "will be kept, unless `nt` is specified. If None (default), all times "
                 "are kept. If `data_path` points to a preprocessed data vector with a "
                 "'.npy' suffix, one of `jd_min` or `jd_center` is required."
        )
        self.add_argument(
            "--jd-center",
            type=float,
            help="Central time as a Julian date. If `data_path` points to a "
                 "pyuvdata-compatible visibility file, `nt` is also required to "
                 "determine the number of times kept around `jd_center` in the data "
                 "vector. If None (default), all times are kept. If `data_path` points "
                 "to a preprocessed data vector with a '.npy' suffix, one of `jd_min` or "
                 "`jd_center` is required."
        )
        # Model Image Parameters
        self.add_argument(
            "--nside",
            type=int,
            help="HEALPix resolution parameter. Sets the resolution of the "
                 "image domain model.  Note, the HEALPix resolution must be "
                 "chosen such that there are two HEALPix pixels per minimum "
                 "fringe wavelength from the model uv-plane to satisfy the "
                 "Nyquist-Shannon sampling theorem."
        )
        self.add_argument(
            "--fov-ra-eor",
            type=float,
            help="Field of view of the right ascension axis of the EoR sky "
                 "model in degrees."
        )
        self.add_argument(
            "--fov-dec-eor",
            type=float,
            help="Field of view of the declination axis of the EoR sky model in "
                 "degrees. Defaults to `fov_ra_eor`."
        )
        self.add_argument(
            "--fov-ra-fg",
            type=float,
            help="Field of view of the right ascension axis of the foreground sky model "
                 "in degrees.  Defaults to `fov_ra_eor`."
        )
        self.add_argument(
            "--fov-dec-fg",
            type=float,
            help="Field of view of the declination axis of the foreground sky model in "
                 "degrees.  Defaults to `fov_ra_fg` or `fov_dec_eor` if `fov_ra_fg` is "
                 "not defined."
        )
        self.add_argument(
            # FIXME: see BayesEoR issue #11
            # There is a bug in the rectangular pixel selection logic due to the
            # wrapping of Right Ascension.  We are thus forcing the analysis to use
            # the zenith angle filter (a circular sky model FoV) as this selection
            # method avoids this bug.
            "--simple-za-filter",
            action="store_true",
            default=True,
            help="Filter pixels in the sky model by zenith angle only (True, default). "
                 "Otherwise, filter pixels in a rectangular region set by the field of "
                 "view values along the RA and Dec axes (False). It is suggested to "
                 "set `simple_za_filter` to True (see issue #11)."
        )
        # Model uv-Plane Parameters
        self.add_argument(
            "--nu",
            type=int,
            help="Number of pixels on the u-axis of the model uv-plane for the EoR model."
        )
        self.add_argument(
            "--nv",
            type=int,
            help="Number of pixels on the v-axis of the model uv-plane for the EoR model. "
                 "Defaults to `nu`."
        )
        self.add_argument(
            "--nu-fg",
            type=int,
            help="Number of pixels on the u-axis of the model uv-plane for the foreground "
                 "model.  Defaults to `nu`."
        )
        self.add_argument(
            "--nv-fg",
            type=int,
            help="Number of pixels on the v-axis of the model uv-plane for the foreground "
                 "model. Defaults to `nu_fg` or `nv` if `nu_fg` is not defined."
        )
        self.add_argument(
            "--fit-for-monopole",
            action="store_true",
            help="Include (True) or exclude (False) the ``(u, v) == (0, 0)`` pixel in "
                 "the model uv-plane. Defaults to False."
        )
        # Noise Model Parameters
        self.add_argument(
            "--sigma",
            type=float,
            help="Standard deviation of the visibility noise in mK sr. Required if "
                 "`calc_noise` is False and `data_path` points to a pyuvdata-compatible "
                 "visibility file or `noise_data_path` is None and `data_path` points to "
                 "a preprocessed numpy-compatible visibility vector. Defaults to None."
        )
        self.add_argument(
            "--noise-seed",
            type=int,
            default=742123,
            help="Seed for `numpy.random`. Used to generate the noise vector if adding "
                 "noise to the input visibility data. Defaults to 742123."
        )
        self.add_argument(
            "--fit-noise",
            action="store_true",
            dest="use_intrinsic_noise_fitting",  # FIXME: remove need for dest
            help="Fit for the noise level."
        )
        # Instrument Model Parameters
        self.add_argument(
            "--model-instrument",
            action=ActionYesNo(yes_prefix="model-", no_prefix="no-"),
            dest="include_instrumental_effects",  # FIXME: remove need for dest
            default=True,
            help="Forward model an instrument (--model-instrument) or exclude "
                 "instrumental effects (--no-instrument)."
        )
        self.add_argument(
            "--inst-model",
            type=path_type("dr"),
            help="Path to a directory containing the instrument model. This directory "
                 "must at least contain two numpy-compatible files: `uvw_model.npy` "
                 "containing the sampled (u, v, w) coordinates with shape "
                 "(nt, nbls, 3) where nbls is the number of baselines, and "
                 "`redundancy_model.npy` containing the number of baselines in a each "
                 "sampled (u, v, w) with shape (nt, nbls, 1). Please see "
                 "`bayeseor.model.instrument` for more details. Used only if "
                 "`include_instrumental_effects` is True. Defaults to None."
        )
        self.add_argument(
            "--tele-latlonalt",
            type=List[float],
            dest="telescope_latlonalt",  # FIXME: remove need for dest
            help="Telescope location in latitude (deg), longitude (deg), and altitude "
                 "(meters). Passed as a list of floats, e.g. '[30.1,125.6,80.4]'. Do not "
                 "put spaces after commas. Required if `include_instrumental_effects` "
                 "is True. Defaults to None."
        )
        self.add_argument(
            "--drift-scan",
            action=ActionYesNo(yes_prefix="drift-scan", no_prefix="phased"),
            default=True,
            help="Model the instrument in drift scan mode (--drift-scan) or "
                 "in phased mode (--phased).  Used only if `include_instrumental_effects` "
                 "is True. Defaults to drift scan mode."
        )
        self.add_argument(
            "--beam-type",
            type=str,
            help="Path to a pyuvdata-compatible beam file or one of 'uniform', "
                 "'gaussian', 'airy', 'gausscosine', or 'taperairy'. Used only if "
                 "`include_instrumental_effects` is True."
        )        
        self.add_argument(
            "--fwhm-deg",
            type=float,
            help="Full width at half maximum of beam in degrees. Used only if "
                 "`include_instrumental_effects` is True and `beam_type` is 'airy', "
                 "'gaussian', or 'gausscosine'."
        )
        self.add_argument(
            "--antenna-diameter",
            type=float,
            help="Antenna (aperture) diameter in meters. Used only if "
                 "`include_instrumental_effects` is True and `beam_type` is 'airy', "
                 "'gaussian', or 'gausscosine'."
        )
        self.add_argument(
            "--cosfreq",
            type=float,
            help="Cosine frequency if using a 'gausscosine' beam. Used only if "
                 "`include_instrumental_effects` is True."
        )
        self.add_argument(
            "--achromatic-beam",
            action="store_true",
            help="Force the beam to be achromatic. The frequency at which the beam will "
                 "be calculated is set via `beam_ref_freq`. Used only if "
                 "`include_instrumental_effects` is True."
        )
        self.add_argument(
            "--beam-ref-freq",
            type=float,
            help="Beam reference frequency in hertz. Used only if "
                 "`include_instrumental_effects` is True. Defaults to `freq_min`."
        )
        self.add_argument(
            "--beam-peak-amplitude",
            type=float,
            default=1.0,
            help="Peak amplitude of the beam. Used only if `include_instrumental_effects` "
                 "is True."
        )
        self.add_argument(
            "--beam-center",
            type=List[float],
            help="Beam center offsets from the phase center in RA and Dec in degrees. "
                 "Default behavior is the beam center aligns with the phase center. "
                 "Passed as a list of floats, e.g. '[-1.3,0.01]'. Do not include a space "
                 "after the comma."
        )

        # Subharmonic Grid Parameters
        self.add_argument(
            "--nu-sh",
            type=int,
            help="Number of pixels on a side for the u-axis in the subharmonic grid "
                 "model uv-plane."
        )
        self.add_argument(
            "--nv-sh",
            type=int,
            help="Number of pixels on a side for the v-axis in the subharmonic grid "
                 "model uv-plane. Defaults to `nu_sh`."
        )
        self.add_argument(
            "--fit-for-shg-amps",
            action="store_true",
            help="Fit for the amplitudes of the subharmonic grid pixels."
        )
        # Tapering Parameters
        self.add_argument(
            "--taper-func",
            type=str,
            help="Tapering function to apply to the frequency axis of the model "
                 "visibilities.  Can be any valid argument to "
                 "`scipy.signal.windows.get_window`."
        )
        # Input Data Parameters
        self.add_argument(
            "--data-path",
            type=Path_fr,
            help="Path to either a pyuvdata-compatible visibility file or a preprocessed "
                 "numpy-compatible visibility vector in units of mK sr."
        )
        self.add_argument(
            "--noise-data-path",
            type=Path_fr,
            help="Path to a preprocessed numpy-compatible noise visibility vector in "
                 "units of mK sr.  Required if `calc_noise` is False and `sigma` is None."
        )
        self.add_argument(
            "--ant-str",
            type=str,
            default="cross",
            help="Antenna downselect string. If `data_path` points to a "
                 "pyuvdata-compatible visibility file, `ant_str` determines what "
                 "baselines to keep in the data vector. Please see "
                 "`pyuvdata.UVData.select` for more details. Defaults to 'cross' "
                 "(cross-correlation baselines only)."
        )
        self.add_argument(
            "--bl-cutoff",
            type=float,
            help="Baseline length cutoff in meters. If `data_path` points to a "
                 "pyuvdata-compatible visibility file, `bl_cutoff` determines the longest "
                 "baselines kept in the data vector. Defaults to None (keep all "
                 "baselines)."
        )
        self.add_argument(
            "--form-pI",
            action="store_true",
            default=True,
            help="Form pseudo-Stokes I visibilities. Otherwise, use the polarization "
                 "specified by `pol`. Used only if `data_path` points to a "
                 "pyuvdata-compatible visibility file."
        )
        self.add_argument(
            "--pI-norm",
            type=float,
            default=1.0,
            help="Normalization, ``N``, used in forming pseudo-Stokes I from XX and YY "
                 "via ``pI = N * (XX + YY)``. Used only if `data_path` points to a "
                 "pyuvdata-compatible visibility file and `form_pI` is True."
        )
        self.add_argument(
            "--pol",
            type=str,
            default="xx",
            help="Case-insensitive polarization string. Can be one of 'xx', 'yy', or 'pI' "
                 "for XX, YY, or pseudo-Stokes I polarization, respectively. Used only if "
                 "`data_path` points to a pyuvdata-compatible visibility file and "
                 "`form_pI` is False."
        )
        self.add_argument(
            "--redundant-avg",
            action="store_true",
            help="Redundantly average the data. Used only if `data_path` points to a "
                 "pyuvdata-compatible visibility file."
        )
        self.add_argument(
            "--uniform-redundancy",
            action="store_true",
            help="Force the redundancy model to be uniform. Used only if `data_path` "
                 "points to a pyuvdata-compatible visibility file."
        )
        self.add_argument(
            "--phase-time",
            type=float,
            help="The time to which the visibilities will be phased as a Julian date. "
                 "Used only if `drift_scan` is False.  If `drift_scan` is False and "
                 "`phase_time` is None, `phase_time` will be automatically set to the "
                 "central time in the data. Used only if `data_path` points to a "
                 "pyuvdata-compatible visibility file."
        )
        self.add_argument(
            "--calc-noise",
            action="store_true",
            help="Calculate a noise estimate from the visibilities via differencing "
                 "adjacent times per baseline and frequency. Used only if `data_path` "
                 "points to a pyuvdata-compatible visibility file."
        )
        self.add_argument(
            "--save-vis",
            action="store_true",
            help="Write visibility vector to disk in `out_dir`. If `calc_noise` is True, "
                 "also save the noise vector. Used only if `data_path` points to a "
                 "pyuvdata-compatible visibility file."
        )
        self.add_argument(
            "--save-model",
            action="store_true",
            help="Write instrument model (antenna pairs, (u, v, w) sampling, and "
                 "redundancy model) to disk in `out_dir`. If `phase` is True, also save "
                 "the phasor vector. Used only if `data_path` points to a "
                 "pyuvdata-compatible visibility file."
        )

    def parse_args(self, args_str=None):
        """
        Parse arguments from `sys.argv` or `args`.

        Parameters
        ----------
        args_str : list of str, optional
            Command line arguments as a list of strings, e.g.
            '["--config", "example_config.yaml"]'.  If None (default),
            pulls from `sys.argv`.

        Returns
        -------
        args : Namespace
            Namespace of parsed arguments.

        """
        args = super().parse_args(args_str)

        if args.beam_type:
            args.beam_type = args.beam_type.lower()
        if args.taper_func:
            args.taper_func = args.taper_func.lower()

        return args
