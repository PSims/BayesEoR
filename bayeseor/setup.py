""" Convenience functions for setting up a BayesEoR analysis. """

import numpy as np
from collections.abc import Sequence
from astropy.time import Time
from astropy import units
from astropy.units import Quantity
from pathlib import Path
from pyuvdata import __version__ as pyuvdata_version
from rich.panel import Panel
from scipy import sparse
import warnings

from .cosmology import Cosmology
from .matrices.build import BuildMatrices
from .model.instrument import load_inst_model
from .model.noise import generate_gaussian_noise
from .model.k_cube import (
    generate_k_cube_in_physical_coordinates,
    mask_k_cube,
    generate_k_cube_model_spherical_binning,
    calc_mean_binned_k_vals
)
from .posterior import PowerSpectrumPosteriorProbability
from .vis import preprocess_uvdata
from .utils import (
    mpiprint, save_numpy_dict, load_numpy_dict, parse_uprior_inds
)


def run_setup(
    *,
    nf : int | None = None,
    neta : int | None = None,
    nq : int = 0,
    beta : list[float] | None = None,
    nt : int | None = None,
    nu : int,
    nv : int | None = None,
    nu_fg : int | None = None,
    nv_fg : int | None = None,
    fit_for_monopole : bool = False,
    nu_sh : int | None = None,
    nv_sh : int | None = None,
    fit_for_shg_amps : bool = False,
    nside : int,
    fov_ra_eor : float,
    fov_dec_eor : float | None = None,
    fov_ra_fg : float | None = None,
    fov_dec_fg : float | None = None,
    simple_za_filter : bool = True,
    taper_func : str | None = None,
    drift_scan : bool = True,
    include_instrumental_effects : bool = True,
    beam_type : str,
    beam_center : list[float, float] | None = None,
    achromatic_beam : bool = False,
    beam_peak_amplitude : float = 1.0,
    fwhm_deg : float | None = None,
    antenna_diameter : float | None = None,
    cosfreq : float | None = None,
    beam_ref_freq : float | None = None,
    data_path : Path | str,
    ant_str : str = "cross",
    bl_cutoff : Quantity | float | None = None,
    freq_idx_min : int | None = None,
    freq_min : Quantity | float | None = None,
    freq_center : Quantity | float | None = None,
    df : Quantity | float | None = None,
    jd_idx_min : int | None = None,
    jd_min : Time | float | None = None,
    jd_center : Time | float | None = None,
    dt : Quantity | float | None = None,
    form_pI : bool = True,
    pI_norm : float = 1.0,
    pol : str = "xx",
    redundant_avg : bool = False,
    uniform_redundancy : bool = False,
    phase : bool = False,
    phase_time : Time | float | None = None,
    calc_noise : bool = False,
    save_vis : bool = False,
    save_model : bool = False,
    save_dir : Path | str | None = None,
    clobber : bool = False,
    sigma : float | None = None,
    noise_seed : int = 742123,
    noise_data_path : Path | str | None = None,
    inst_model : Path | str | None = None,
    save_k_vals : bool = True,
    telescope_name : str = "",
    telescope_latlonalt : Sequence[float] | None = None,
    array_dir_prefix : Path | str = "./matrices/",
    use_sparse_matrices : bool = True,
    build_Finv_and_Fprime : bool = True,
    output_dir : Path | str = "./",
    mkdir : bool = True,
    file_root : str | None = None,
    priors : Sequence[float],
    log_priors : bool = False,
    uprior_bins : str = "",
    uprior_inds : np.ndarray | None = None,
    dimensionless_PS : bool = True,
    inverse_LW_power : float = 1e-16,
    use_gpu : bool = True,
    use_intrinsic_noise_fitting : bool = False,
    use_LWM_Gaussian_prior : bool = False,
    use_EoR_cube : bool = False,
    use_Multinest : bool = True,
    return_vis : bool = False,
    return_uvd : bool = False,
    return_ks : bool = False,
    return_bm : bool = False,
    verbose : bool = False,
    rank : bool = 0,
    **kwargs
):
    """
    Run setup steps.

    Parameters
    ----------
    nf : int, optional
        Number of frequency channels. Required if `data_path` points to a
        preprocessed data vector with a '.npy' suffix. Otherwise, sets the
        number of frequencies to keep starting from `freq_idx_min`, the
        channel corresponding to `freq_min`, or around `freq_center`.
        Defaults to None (keep all frequencies).
    neta : int, optional
        Number of Line of Sight (LoS, frequency axis) Fourier modes. Defaults
        to `nf`.
    nq : int, optional
        Number of large spectral scale model basis vectors. If `beta` is None,
        the basis vectors are quadratic in frequency. If `beta` is not None,
        the basis vectors are power laws with brightness temperature spectral
        indices from `beta`.
    beta : list of float, optional
        Brightness temperature power law spectral index/indices used in the
        large spectral scale model. Can be a single spectral index, e.g.
        [2.63], or multiple spectral indices, e.g. [2.63, 2.82]. Defaults to
        [2.63, 2.82].
    nt : int, optional
        Number of times. Required if `data_path` points to a preprocessed data
        vector with a '.npy' suffix. Otherwise, sets the number of times to
        keep starting from `jd_idx_min`, the time corresponding to `jd_min`
        or around `jd_center`. Defaults to None (keep all times).
    nu : int
        Number of pixels on a side for the u-axis in the EoR model uv-plane.
    nv : int, optional
        Number of pixels on a side for the v-axis in the EoR model uv-plane.
        Defaults to `nu`.
    nu_fg : int, optional
        Number of pixels on a side for the u-axis in the foreground model
        uv-plane. Defaults to `nu`.
    nv_fg : int, optional
        Number of pixels on a side for the v-axis in the foreground model
        uv-plane. Defaults to `nv` if `nu_fg` is None or `nu_fg`.
    fit_for_monopole : bool, optional
        Fit for the (u, v) = (0, 0) pixel in the model uv-plane.
    nu_sh : int, optional
        Number of pixels on a side for the u-axis in the subharmonic grid
        model uv-plane. Defaults to None.
    nv_sh : int, optional
        Number of pixels on a side for the v-axis in the subharmonic grid
        model uv-plane. Defaults to `nu_sh`.
    fit_for_shg_amps : bool, optional
        Fit for the amplitudes of the subharmonic grid pixels. Defaults to
        False.
    nside : int
        HEALPix nside parameter.
    fov_ra_eor : float
        Field of view of the Right Ascension axis of the EoR sky model in
        degrees.
    fov_dec_eor : float, optional
        Field of view of the Declination axis of the EoR sky model in degrees.
        Defaults to `fov_ra_eor`.
    fov_ra_fg : float, optional
        Field of view of the Right Ascension axis of the foreground sky model
        in degrees. Defaults to `fov_ra_eor`.
    fov_dec_fg : float, optional
        Field of view of the Declination axis of the foreground sky model in
        degrees. Defaults to `fov_dec_eor` if `fov_ra_fg` is None or
        `fov_ra_fg`.
    simple_za_filter : bool, optional
        Filter pixels in the sky model by zenith angle only. Defaults to True.
    taper_func : str, optional
        Taper function applied to the frequency axis of the visibilities.
        Can be any valid argument to `scipy.signal.windows.get_window`.
        Defaults to None.
    drift_scan : bool, optional
        Model drift scan (True) or phased (False) visibilities. Defaults to
        True.
    include_instrumental_effects : bool, optional
        Forward model an instrument. Defaults to True.
    beam_type : str
        Path to a pyuvdata-compatible beam file or one of 'uniform',
        'gaussian', 'airy', 'gausscosine', or 'taperairy'. Used only if
        `include_instrumental_effects` is True. Defaults to None.
    beam_center : list of float, optional
        Beam center offsets from the phase center in right ascension and
        declination in degrees. Used only if `include_instrumental_effects` is
        True. Defaults to None.
    achromatic_beam : bool, optional
        Force the beam to be achromatic. The frequency at which the beam will
        be calculated is set via `beam_ref_freq`. Used only if
        `include_instrumental_effects` is True. Defaults to False.
    beam_peak_amplitude : float, optional
        Peak amplitude of the beam. Used only if `include_instrumental_effects`
        is True. Defaults to 1.0.
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
    beam_ref_freq : float, optional
        Beam reference frequency in hertz. Used only if
        `include_instrumental_effects` is True. Defaults to `freq_min`.
    data_path : pathlib.Path or str
        Path to either a pyuvdata-compatible visibility file or a preprocessed
        numpy-compatible visibility vector in units of mK sr.
    ant_str : str, optional
        Antenna downselect string. If `data_path` points to a
        pyuvdata-compatible visibility file, `ant_str` determines what
        baselines to keep in the data vector. Please see
        `pyuvdata.UVData.select` for more details. Defaults to 'cross'
        (cross-correlation baselines only).
    bl_cutoff : astropy.Quantity or float, optional
        Baseline length cutoff in meters. If `data_path` points to a
        pyuvdata-compatible visibility file, `bl_cutoff` determines the longest
        baselines kept in the data vector. Defaults to None (keep all
        baselines).
    freq_idx_min : int, optional
        Minimum frequency channel index to keep in the data vector. Used only
        if `data_path` points to a pyuvdata-compatible visibility file.
        Defaults to None (keep all frequencies).
    freq_min : astropy.Quantity or float, optional
        Minimum frequency in hertz if not a Quantity. If `data_path` points to
        a pyuvdata-compatible visibility file, `freq_min` sets the minimum
        frequency kept in the data vector.  All frequencies greater than or
        equal to `freq_min` will be kept, unless `nf` is specified. If None
        (default), all frequencies are kept. If `data_path` points to a
        preprocessed data vector with a '.npy' suffix, one of `freq_min` or
        `freq_center` is required. Defaults to None (keep all frequencies).
    freq_center : astropy.Quantity or float, optional
        Central frequency in hertz if not a Quantity. If `data_path` points to
        a pyuvdata-compatible visibility file, `nf` is also required to
        determine the number of frequencies kept around `freq_center` in the
        data vector. If None (default), all frequencies are kept. If
        `data_path` points to a preprocessed data vector with a '.npy' suffix,
        one of `freq_min` or `freq_center` is required. Defaults to None (keep
        all frequencies).
    df : astropy.Quantity or float, optional
        Frequency channel width in hertz if not a Quantity. Required if
        `data_path` points to a preprocessed data vector with a '.npy' suffix.
        Overwritten by the frequency channel width in the UVData object if
        `data_path` points to a pyuvdata-compatible file containing
        visibilities. Defaults to None.
    jd_idx_min : int, optional
        Minimum time index to keep in the data vector. Used only if `data_path`
        points to a pyuvdata-compatible visibility file. Defaults to None (keep
        all times). Defaults to None (keep all times).
    jd_min : astropy.Time or float, optional
        Minimum time to keep in the data vector as a Julian date if not a Time.
        One of `jd_min` or `jd_center` is required if `data_path` points to a
        preprocessed data vector with a '.npy' suffix. Defaults to None (keep
        all times).
    jd_center : astropy.Time or float, optional
        Central time, as a Julian date if not a Time, around which `nt`
        times will be kept in the data vector. `nt` must also be passed,
        otherwise an error is raised. Defaults to None (keep all times).
    dt : astropy.Quantity or float, optional
        Integration time in seconds if not a Quantity. Required if `data_path`
        points to a preprocessed data vector with a '.npy' suffix. Overwritten
        by the integration time in the UVData object if `data_path` points to
        a pyuvdata-compatible file. Defaults to None.
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
    phase : bool, optional
        Create a "phasor vector" which can be used to phase each visibility
        in the data vector as a function of baseline, time, and frequency via
        element-wise multiplication. Defaults to False.
    phase_time : astropy.Time or float, optional
        The time to which the visibilities will be phased as a Julian date if
        not a Time. If `phase` is True and `phase_time` is None, `phase_time`
        will be automatically set to the central time in the data. Used only
        if `data_path` points to a pyuvdata-compatible visibility file.
        Defaults to None.
    calc_noise : bool, optional
        Calculate a noise estimate from the visibilities via differencing
        adjacent times per baseline and frequency. Used only if `data_path`
        points to a pyuvdata-compatible visibility file. Defaults to False.
    save_vis : bool, optional
        Write visibility vector to disk in `save_dir`. If `calc_noise` is True,
        also save the noise vector. Used only if `data_path` points to a
        pyuvdata-compatible visibility file. Defaults to False.
    save_model : bool, optional
        Write instrument model (antenna pairs, (u, v, w) sampling, and
        redundancy model) to disk in `save_dir`. If `phase` is True, also save
        the phasor vector. Used only if `data_path` points to a
        pyuvdata-compatible visibility file. Defaults to False.
    save_dir : pathlib.Path or str, optional
        Output directory if `save_vis` or `save_model` is True. Defaults to
        ``Path(output_dir) / file_root``.
    clobber : bool, optional
        Clobber files on disk if they exist.  Defaults to False.
    sigma : float, optional
        Standard deviation of the visibility noise in mK sr. Required if
        `calc_noise` is False and `data_path` points to a pyuvdata-compatible
        visibility file or `noise_data_path` is None and `data_path` points to
        a preprocessed numpy-compatible visibility vector. Defaults to None.
    noise_seed : int, optional
        Used to seed `np.random` when generating the noise vector. Defaults to
        742123.
    noise_data_path : pathlib.Path or str, optional
        Path to a preprocessed numpy-compatible noise visibility vector in
        units of mK sr. Defaults to None.
    inst_model : pathlib.Path or str, optional
        Path to directory containing instrument model files (`uvw_array.npy`,
        `redundancy_model.npy`, and optionally `phasor_vector.npy`). Required
        if `data_path` points to a preprocessed data vector with a '.npy'
        suffix. Used only if `include_instrumental_effects` is True. Defaults
        to None.
    save_k_vals : bool, optional
        Save k bin files (means, edges, and number of voxels in each bin).
        Defaults to True.
    telescope_name : str, optional
        Telescope identifier string. Defaults to ''.
    telescope_latlonalt : sequence of float, optional
        Telescope location tuple as (latitude in degrees, longitude in degrees,
        altitude in meters). Required if `include_instrumental_effects` is
        True. Defaults to None.
    array_dir_prefix : pathlib.Path or str, optional
        Array directory prefix. Defaults to './matrices/'.
    use_sparse_matrices : bool, optional
        Use sparse arrays. Defaults to True.
    build_Finv_and_Fprime : bool
        If True, construct the matrix product Finv_Fprime in place from the
        dense matrices comprising Finv and Fprime to minimize the memory and
        time required to build the matrix stack. In this case, only the matrix
        product Finv_Fprime is written to disk. Otherwise, construct Finv and
        Fprime independently and save both matrices to disk.
    output_dir : pathlib.Path or str, optional
        Parent directory for sampler output. Defaults to './'.
    mkdir : bool, optional
        Make ``Path(output_dir) / file_root`` if it doesn't exist. Defaults to
        True.
    file_root : str, optional
        Sampler output directory name. If None (default), start a new analysis.
        Otherwise, resume analysis from `file_root`.
    priors : sequence of float
        Prior [min, max] for each k bin as a a sequence, e.g. [[min1, max1],
        [min2, max2], ...].
    log_priors : bool, optional
        Assume priors on power spectrum coefficients are in log_10 units.
        Defaults to False.
    uprior_bins : str, optional
        Array indicees of k bins using a uniform prior. Follows python slicing
        syntax. Can pass a range via '1:4' (non-inclusive high end), a list of
        indices via '1,4,6' (no spaces between commas), a single index '3' or
        '-3', or 'all'. Defaults to "" (all k bins use log-uniform priors).
    uprior_inds : numpy.ndarray, optional
        Boolean 1D array that is True for any k bins using a uniform prior.
        False entries use a log-uniform prior. If both `uprior_str` and
        `uprior_inds` are passed, `uprior_inds` will take precedence. Defaults
        to None (all k bins use a log-uniform prior).
    dimensionless_PS : bool, optional
        Fit for the dimensionless power spectrum, :math:`\\Delta^2(k)` (True),
        or the power spectrum, :math:`P(k)` (False). Defaults to True.
    inverse_LW_power : float, optional
        Prior on the inverse power of the large spectral scale model
        coefficients. Defaults to 1e-16.
    use_gpu : bool, optional
        Use GPUs (True) or CPUs (False). Defaults to True.
    use_intrinsic_noise_fitting : bool, optional
        Fit for the noise level. This option is currently not implemented but
        will be reimplemented in the future. Defaults to False.
    use_LWM_Gaussian_prior : bool, optional
        Use a Gaussian prior (True) or a uniform prior (False) on the large
        spectral scale model. This option is currently not implemented but
        will be reimplemented in the future. Defaults to False.
    use_EoR_cube : bool, optional
        Use internally simulated data generated from a EoR cube. This
        functionality is not currently supported but will be implemented in
        the future. Defaults to False.
    use_Multinest : bool, optional
        Use MultiNest sampler (True) or PolyChord (False). Support for
        PolyChord will be added in the future. Defaults to True.
    return_vis : bool, optional
        Return a dictionary, `vis_dict`, with value (key) pairs of: the
        visibility vector ('vis'), the noise ('noise'), noisy visibilities if
        `noise_data_path` is None and `sigma` is not None ('vis_noisy'), a
        dictionary containing the array index mapping of conjugated baselines
        in the visibility vector ('bl_conj_pairs_map'), the instrumentally
        sampled (u, v, w) ('uvws'), the number of redundant baselines averaged
        in each (u, v, w) ('redundancy'), the frequency channel width ('df'),
        the integration time ('dt'), and the phasor vector ('phasor') if
        `phase` is True. Defaults to False.
    return_uvd : bool, optional
        Return the UVData object as part of `vis_dict` if `return_vis` is True.
        Defaults to False.
    return_ks : bool, optional
        Return the mean of each k bin and a list of sublists containing the
        flattened indices of all voxels included in each k bin. Defaults to
        False.
    return_bm : bool, optional
        Return the matrix building class instance used to construct the matrix
        stack. Defaults to False.
    verbose : bool, optional
        Verbose output. Defaults to False.
    rank : int, optional
        MPI rank. Defaults to 0.
    **kwargs : :class:`.params.BayesEoRParser` attributes
        Catch-all for auxiliary BayesEoRParser attributes so the function may
        be called using the arguments parsed by a BayesEoRParser instance via
        e.g. `run_setup(**args)`.

    Returns
    -------
    pspp : :class:`.posterior.PowerSpectrumPosteriorProbability`
        Posterior probability class instance.
    sampler_dir : pathlib.Path
        Path to sampler output directory, ``Path(output_dir) / file_root``.
    vis_dict : dict
        Dictionary with the following key: value pairs

        - vis_noisy: noisy visibility vector with shape (nf*nbls*nt,)
        - noise: noise vector with shape (nf*nbls*nt,)
        - bl_conj_pairs_map: array index mapping of conjugate baseline pairs
        - uvws: instrumentally sampled (u, v, w) coordinates with shape
          (nt, nbls, 3)
        - redundancy: number of redundantly averaged baselines per (u, v, w)
          with shape (nt, nbls, 1)
        - freqs: frequency channels in Hz
        - df: frequency channel width in Hz
        - jds: Julian dates
        - dt: integration time in seconds
        - vis: optional noise free visibility vector with shape (nf*nbls*nt,)
          if `noise_data_path` is None or `calc_noise` is False
        - antpairs: optional baseline antenna pairs for each (u, v, w) if
          `inst_model` contains a file 'antpairs.npy' or if `data_path` points
          to a pyuvdata-compatible file
        - phasor: optional phasor vector if `phased` is True with shape
          (nf*nbls*nt,)
        - tele_name: optional telescope name if `data_path` points to a
          pyuvdata-compatible file with a valid telescope name attribute
        - uvd: optional UVData object if `data_path` points to a
          pyuvdata-compatible file and `return_uvd` is True
    k_vals : numpy.ndarray
        Mean of each k bin. Returned only if `return_ks` is True.
    k_cube_voxels_in_bin : list
        List of sublists containing the flattened 3D k-space cube index of all
        |k| that fall within a given k bin. Returned only if `return_ks` is
        True.
    bm : :class:`.matrices.build.BuildMatrices`
        Matrix building class instance. Returned only if `return_bm` is True.

    """
    if use_intrinsic_noise_fitting:
        # FIXME
        raise NotImplementedError(
            "use_intrinsic_noise_fitting is not currently implemented. It "
            "will be reimplemented in the future. For now, please set "
            "use_intrinsic_noise_fitting to False."
        )
    if use_LWM_Gaussian_prior:
        # FIXME
        raise NotImplementedError(
            "use_LWM_Gaussian_prior is not currently implemented. It will be "
            "reimplemented in the future. For now, please set "
            "use_LWM_Gaussian_prior to False."
        )
    if sigma is None and (noise_data_path is None or not calc_noise):
        raise ValueError(
            "sigma cannot be None if noise_data_path is None or calc_noise "
            "is False. The input visibilities in either case are assumed to "
            "be noise free and require sigma to generate noise."
        )
    if output_dir is None:
        raise ValueError("output_dir cannot be None")
    if include_instrumental_effects and telescope_latlonalt is None:
        raise ValueError(
            "telescope_latlonalt cannot be None if "
            "include_instrumental_effects is true"
        )

    # print_rank will only trigger print if verbose is True and rank == 0
    print_rank = 1 - (verbose and rank == 0)

    # We must first process the input data to assign nf and nt
    # as they are not required parameters if data_path points to
    # a pyuvdata-compatible file.  If nf and nt are none, they
    # are assigned values based on the number of frequencies and
    # times in the pyuvdata-compatible file.
    mpiprint("\n", Panel("Data and Noise"), rank=print_rank)
    if save_vis or save_model:
        # Catalog relevant command line arguments for posterity so
        # the data vector can be recreated exactly before parameters
        # like nf or nt get potentially overwritten after processing
        # the data vector.
        vis_args = dict(
            fp=data_path.as_posix(),
            ant_str=ant_str,
            bl_cutoff=bl_cutoff,
            freq_min=freq_min,
            freq_idx_min=freq_idx_min,
            freq_center=freq_center,
            Nfreqs=nf,
            jd_min=jd_min,
            jd_idx_min=jd_idx_min,
            jd_center=jd_center,
            Ntimes=nt,
            form_pI=form_pI,
            pI_norm=pI_norm,
            pol=pol,
            redundant_avg=redundant_avg,
            uniform_redundancy=uniform_redundancy,
            phase=phase,
            phase_time=phase_time,
            calc_noise=calc_noise
        )
        extra = dict(pyuvdata_version=pyuvdata_version)
    vis_dict = get_vis_data(
        data_path=data_path,
        ant_str=ant_str,
        bl_cutoff=bl_cutoff,
        freq_idx_min=freq_idx_min,
        freq_min=freq_min,
        freq_center=freq_center,
        nf=nf,
        df=df,
        jd_idx_min=jd_idx_min,
        jd_min=jd_min,
        jd_center=jd_center,
        nt=nt,
        dt=dt,
        form_pI=form_pI,
        pI_norm=pI_norm,
        pol=pol,
        redundant_avg=redundant_avg,
        uniform_redundancy=uniform_redundancy,
        phase=phase,
        phase_time=phase_time,
        calc_noise=calc_noise,
        sigma=sigma,
        noise_seed=noise_seed,
        noise_data_path=noise_data_path,
        inst_model=inst_model,
        return_uvd=return_uvd,
        verbose=verbose,
        rank=rank,
    )
    vis_noisy = vis_dict["vis_noisy"]
    noise = vis_dict["noise"]
    uvws = vis_dict["uvws"]
    redundancy = vis_dict["redundancy"]
    nf = len(vis_dict["freqs"])
    nu_min_MHz = (vis_dict["freqs"][0] * units.Hz).to("MHz").value
    channel_width_MHz = (vis_dict["df"] * units.Hz).to("MHz").value
    nt = len(vis_dict["jds"])
    jd_center = vis_dict["jds"][nt//2]
    dt = vis_dict["dt"]
    if "phasor" in vis_dict:
        phasor = vis_dict["phasor"]
    else:
        phasor = None

    # Assign optional kwargs if None
    # Model k cube params
    if neta is None:
        neta = nf
    if nv is None:
        nv = nu
    if nu_fg is None:
        nu_fg = nu
        nv_fg = nv
    elif nv_fg is None:
        nv_fg = nu_fg
    # Model image params
    if fov_dec_eor is None:
        fov_dec_eor = fov_ra_eor
    if fov_ra_fg is None:
        fov_ra_fg = fov_ra_eor
        fov_dec_fg = fov_dec_eor
    elif fov_dec_fg is None:
        fov_dec_fg = fov_ra_fg
    # Foreground model params
    if beta is not None:
        npl = len(beta)  # FIXME: update if fitting for LSSM
    else:
        npl = 0
    if nq > npl:
        nq = npl
    # Subharmonic grid params
    use_shg = np.any([kwarg is not None for kwarg in [nu_sh, nv_sh]])
    if use_shg:
        if nu_sh is None:
            raise ValueError("nu_sh is required if using the subharmonic grid")
        if nv_sh is None:
            nv_sh = nu_sh

    # Derived params
    cosmo = Cosmology()
    redshift = cosmo.f2z(vis_dict["freqs"].mean() * units.Hz)
    bandwidth = (vis_dict["df"] * units.Hz) * nf
    # EoR model
    # Spacing along the eta axis (line-of-sight Fourier dual to frequency)
    # defined as one over the bandwidth in Hz [1/Hz].
    deta = 1 / bandwidth.to("Hz").value
    # Spacing along the u-axis of the EoR model uv-plane [1/rad]
    du_eor = 1 / np.deg2rad(fov_ra_eor)
    # Spacing along the v-axis of the EoR model uv-plane [1/rad]
    dv_eor = 1 / np.deg2rad(fov_dec_eor)
    # Comoving line-of-sight size of the EoR volume [Mpc]
    ps_box_size_para_Mpc = cosmo.dL_df(redshift) * bandwidth.to("Hz").value
    # Comoving transverse size of the EoR volume along RA [Mpc]
    ps_box_size_ra_Mpc = cosmo.dL_dth(redshift) * np.deg2rad(fov_ra_eor)
    # Comoving transverse size of the EoR volume along Dec [Mpc]
    ps_box_size_dec_Mpc = cosmo.dL_dth(redshift) * np.deg2rad(fov_dec_eor)
    # Foreground model
    # Spacing along the u-axis of the model uv-plane [1/rad]
    du_fg = 1 / np.deg2rad(fov_ra_fg)
    # Spacing along the v-axis of the model uv-plane [1/rad]
    dv_fg = 1 / np.deg2rad(fov_dec_fg)
    # Beam model
    if achromatic_beam:
        if beam_ref_freq is None:
            beam_ref_freq = nu_min_MHz
        else:
            # Hz -> MHz for compatibility with BuildMatrices
            beam_ref_freq = (beam_ref_freq * units.Hz).to("MHz").value

    # Output directory generation
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    if file_root is None:
        # We store the Slurm job ID in the file_root directory when
        # building the matrix stack, if running in a Slurm environment, so
        # we need to first make the file_root directory a valid, writable
        # directory if it doesn't already exist.
        file_root = generate_file_root(
            nu=nu,
            nv=nv,
            neta=neta,
            nq=nq,
            npl=npl,
            sigma=sigma,
            fit_for_monopole=fit_for_monopole,
            beta=beta,
            log_priors=log_priors,
            dimensionless_PS=dimensionless_PS,
            inverse_LW_power=inverse_LW_power,
            use_EoR_cube=use_EoR_cube,
            use_Multinest=use_Multinest,
            use_shg=use_shg,
            nu_sh=nu_sh,
            nv_sh=nv_sh,
            nq_sh=nq,
            npl_sh=npl,
            fit_for_shg_amps=fit_for_shg_amps,
            output_dir=output_dir,
        )

    sampler_dir = output_dir / file_root
    if mkdir:
        sampler_dir.mkdir(exist_ok=True, parents=True)

    if save_vis or save_model:
        if save_dir is None:
            save_dir = sampler_dir
        elif not isinstance(save_dir, Path):
            save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
    if save_vis:
        vis_path = save_dir / "vis_noisy.npy"
        mpiprint(f"\nSaving data vector(s) to disk:", rank=print_rank)
        mpiprint(f"\tVisibility vector: {vis_path}", rank=print_rank)
        save_numpy_dict(
            vis_path,
            vis_dict["vis_noisy"],
            vis_args,
            extra=extra,
            clobber=clobber
        )
        noise_path = save_dir / "noise.npy"
        mpiprint(f"\tNoise vector: {noise_path}", rank=print_rank)
        save_numpy_dict(
            noise_path, noise, vis_args, extra=extra, clobber=clobber
        )
    if save_model:
        mpiprint(f"\nSaving instrument model to disk:", rank=print_rank)
        # If data_path points to a numpy-compatible file, then an instrument
        # model is required as input and there's no reason to save the
        # instrument model to disk. If data_path points to a
        # pyuvdata-compatible file, we will always have antpairs, uvws, and
        # a redundancy model. The phasor is the only optional quantity to save.
        ants_path = save_dir / "antpairs.npy"
        mpiprint(f"\tAntpairs: {ants_path}", rank=print_rank)
        save_numpy_dict(
            ants_path,
            vis_dict["antpairs"],
            vis_args,
            extra=extra,
            clobber=clobber
        )
        uvw_path = save_dir / "uvw_model.npy"
        mpiprint(f"\t(u, v, w) model: {uvw_path}", rank=print_rank)
        save_numpy_dict(uvw_path, uvws, vis_args, extra=extra, clobber=clobber)
        red_path = save_dir / "redundancy_model.npy"
        mpiprint(f"\tRedundancy model: {red_path}", rank=print_rank)
        save_numpy_dict(
            red_path, redundancy, vis_args, extra=extra, clobber=clobber
        )
        if phase:
            phasor_path = save_dir / "phasor_vector.npy"
            mpiprint(f"\tPhasor vector: {phasor_path}", rank=print_rank)
            save_numpy_dict(
                phasor_path, phasor, vis_args, extra=extra, clobber=clobber
            )

    if verbose and rank == 0:
        mpiprint("\n", Panel("Output Directory"))
        mpiprint(f"\n{sampler_dir.absolute().as_posix()}")
    elif rank == 0:
        mpiprint(
            "\n[bold]Output directory:[/bold] "
            f"{sampler_dir.absolute().as_posix()}"
        )

    mpiprint("\n", Panel("Model k Cube"), rank=print_rank)
    k_vals, k_cube_voxels_in_bin = build_k_cube(
        nu=nu,
        nv=nv,
        neta=neta,
        ps_box_size_ra_Mpc=ps_box_size_ra_Mpc,
        ps_box_size_dec_Mpc=ps_box_size_dec_Mpc,
        ps_box_size_para_Mpc=ps_box_size_para_Mpc,
        save_k_vals=save_k_vals,
        output_dir=sampler_dir,
        clobber=clobber,
        verbose=verbose,
        rank=rank
    )
    mpiprint(f"\nk bins: {len(k_vals)}", rank=print_rank)
    mpiprint(
        f"k bin centers: {np.round(k_vals, decimals=3)} 1/Mpc", rank=print_rank
    )
    vox_per_bin = [len(kinds[0]) for kinds in k_cube_voxels_in_bin]
    mpiprint(f"Voxels per bin: {vox_per_bin}", rank=print_rank)

    mpiprint("\n", Panel("Matrices"), rank=print_rank)
    bm = build_matrices(
        nu=nu,
        du_eor=du_eor,
        nv=nv,
        dv_eor=dv_eor,
        nu_fg=nu_fg,
        du_fg=du_fg,
        nv_fg=nv_fg,
        dv_fg=dv_fg,
        nf=nf,
        neta=neta,
        deta=deta,
        fit_for_monopole=fit_for_monopole,
        use_shg=use_shg,
        nu_sh=nu_sh,
        nv_sh=nv_sh,
        nq_sh=nq,
        npl_sh=npl,
        fit_for_shg_amps=fit_for_shg_amps,
        nu_min_MHz=nu_min_MHz,
        channel_width_MHz=channel_width_MHz,
        nq=nq,
        npl=npl,
        beta=beta,
        sigma=sigma,
        nside=nside,
        fov_ra_eor=fov_ra_eor,
        fov_dec_eor=fov_dec_eor,
        fov_ra_fg=fov_ra_fg,
        fov_dec_fg=fov_dec_fg,
        simple_za_filter=simple_za_filter,
        telescope_name=telescope_name,
        telescope_latlonalt=telescope_latlonalt,
        uvws=uvws,
        redundancy=redundancy,
        phasor=phasor,
        nt=nt,
        jd_center=jd_center,
        dt=dt,
        beam_type=beam_type,
        beam_center=beam_center,
        achromatic_beam=achromatic_beam,
        beam_peak_amplitude=beam_peak_amplitude,
        fwhm_deg=fwhm_deg,
        antenna_diameter=antenna_diameter,
        cosfreq=cosfreq,
        beam_ref_freq=beam_ref_freq,
        drift_scan=drift_scan,
        taper_func=taper_func,
        include_instrumental_effects=include_instrumental_effects,
        noise_data_path=noise_data_path,
        array_dir_prefix=array_dir_prefix,
        mkdir=True,
        use_sparse_matrices=use_sparse_matrices,
        build_Finv_and_Fprime=build_Finv_and_Fprime,
        noise=None,  # FIXME: see BayesEoR issue #55
        clobber=clobber,
        verbose=verbose,
        rank=rank
    )

    mpiprint("\n", Panel("Posterior"), rank=print_rank)
    # Temporarily suppress output from bm.dot_product
    bm_verbose = bm.verbose
    bm.verbose = False
    Ninv = bm.read_data("Ninv")
    T = bm.read_data("T")
    Ninv_d = bm.dot_product(Ninv, vis_noisy)
    dbar = bm.dot_product(T.conj().T, Ninv_d)
    d_Ninv_d = np.dot(vis_noisy.conj(), Ninv_d)
    T_Ninv_T = bm.read_data("T_Ninv_T")
    if include_instrumental_effects:
        block_T_Ninv_T = []
    else:
        block_T_Ninv_T = bm.read_data("block_T_Ninv_T")
    n_dims = k_vals.size
    bm.verbose = bm_verbose
    
    # This code left for when use_intrinsic_noise_fitting and
    # use_LWM_Gaussian_prior are reimplemented
    if use_intrinsic_noise_fitting:
        n_dims += 1
    if use_LWM_Gaussian_prior:
        n_dims += 3
    if use_LWM_Gaussian_prior:
        # Set minimum LW model priors using LW power spectrum in fit to
        # white noise (i.e the prior min should incorporate knowledge of
        # signal-removal in iterative pre-subtraction)
        fg_log_priors_min = np.log10(1.e5)
        # Set minimum LW model prior max using numerical stability
        # constraint at the given signal-to-noise in the data.
        fg_log_priors_max = 6.0
        # priors[0] = [fg_log_priors_min, 8.0] # Set
        # Calibrate LW model priors using white noise fitting
        priors[0] = [fg_log_priors_min, fg_log_priors_max]
        priors[1] = [fg_log_priors_min, fg_log_priors_max]
        priors[2] = [fg_log_priors_min, fg_log_priors_max]
        if use_intrinsic_noise_fitting:
            priors[1] = priors[0]
            priors[2] = priors[1]
            priors[3] = priors[2]
            priors[0] = [1.0, 2.0]  # Linear alpha_prime range
    elif use_intrinsic_noise_fitting:
        priors[0] = [1.0, 2.0]  # Linear alpha_prime range

    cosmo = Cosmology()
    redshift = cosmo.f2z(bm.freqs_hertz.mean())

    if uprior_inds is not None:
        assert uprior_inds.size == k_vals.size, (
            "uprior_inds must have size equal to the number of k bins, "
            "k_vals.size"
        )
        assert uprior_inds.dtype == bool, "uprior_inds must have dtype bool"
    elif uprior_inds != "":
        uprior_inds = parse_uprior_inds(uprior_bins, k_vals.size)
    else:
        uprior_inds = None

    posterior_msg = (
        "\nInstantiating posterior class" + (verbose and rank == 0)*":"
    )
    mpiprint(posterior_msg, style="bold", rank=rank, end="\n\n")
    pspp = build_posterior(
        k_vals=k_vals,
        k_cube_voxels_in_bin=k_cube_voxels_in_bin,
        nu=nu,
        nv=nv,
        neta=neta,
        nf=nf,
        nq=nq,
        redshift=redshift,
        Ninv=Ninv,
        dbar=dbar,
        d_Ninv_d=d_Ninv_d,
        T_Ninv_T=T_Ninv_T,
        ps_box_size_ra_Mpc=ps_box_size_ra_Mpc,
        ps_box_size_dec_Mpc=ps_box_size_dec_Mpc,
        ps_box_size_para_Mpc=ps_box_size_para_Mpc,
        include_instrumental_effects=include_instrumental_effects,
        log_priors=log_priors,
        inverse_LW_power=inverse_LW_power,
        dimensionless_PS=dimensionless_PS,
        block_T_Ninv_T=block_T_Ninv_T,
        use_shg=use_shg,
        use_gpu=use_gpu,
        priors=priors,
        uprior_inds=uprior_inds,
        use_intrinsic_noise_fitting=use_intrinsic_noise_fitting,
        use_LWM_Gaussian_prior=use_LWM_Gaussian_prior,
        verbose=verbose,
        rank=rank
    )

    return_vals = (pspp, sampler_dir)
    if return_vis:
        return_vals += (vis_dict,)
    if return_ks:
        return_vals += (k_vals, k_cube_voxels_in_bin)
    if return_bm:
        return_vals += (bm,)
    return return_vals

def generate_file_root(
    *,
    nu : int,
    nv : int,
    neta : int,
    nq : int,
    sigma : float,
    fit_for_monopole : bool = False,
    beta : list[float] | None = None,
    log_priors : bool = False,
    dimensionless_PS : bool = True,
    use_Multinest : bool = True,
    use_shg : bool = False,
    nu_sh : int | None = None,
    nv_sh : int | None = None,
    nq_sh : int | None = None,
    fit_for_shg_amps : bool = False,
    output_dir : Path | str = Path("./"),
    **kwargs
):
    """
    Generate the directory name for the sampler outputs.

    Parameters
    ----------
    nu : int
        Number of pixels on a side for the u-axis in the EoR model uv-plane.
    nv : int
        Number of pixels on a side for the v-axis in the EoR model uv-plane.
    neta : int
        Number of Line of Sight (LoS, frequency axis) Fourier modes.
    nq : int
        Number of large spectral scale model quadratic basis vectors.
    sigma : float
        Standard deviation of the visibility noise in mK sr.
    fit_for_monopole : bool, optional
        Fit for the (u, v) = (0, 0) pixel in the model uv-plane.
    beta : list of float, optional
        Brightness temperature power law spectral index/indices used in the
        large spectral scale model. Defaults to None.
    log_priors : bool, optional
        Assume priors on power spectrum coefficients are in log_10 units.
        Defaults to False.
    dimensionless_PS : bool, optional
        Fit for the dimensionless power spectrum, :math:`\\Delta^2(k)` (True),
        or the power spectrum, :math:`P(k)` (False). Defaults to True.
    use_Multinest : bool, optional
        Use MultiNest sampler (True) or Polychord (False). Defaults to True.
    use_shg : bool, optional
        Use the subharmonic grid. Defaults to False.
    nu_sh : int, optional
        Number of pixels on a side for the u-axis in the subharmonic grid
        model uv-plane. Used only if `use_shg` is True. Defaults to None.
    nv_sh : int, optional
        Number of pixels on a side for the v-axis in the subharmonic grid
        model uv-plane. Used only if `use_shg` is True. Defaults to None.
    nq_sh : int, optional
        Number of large spectral scale model quadratic basis vectors for the
        subharmonic grid. Used only if `use_shg` is True. Defaults to None.
    fit_for_shg_amps : bool, optional
        Fit for the amplitudes of the subharmonic grid pixels. Defaults to
        False.
    output_dir : pathlib.Path or str, optional
        Parent directory for sampler output. Defaults to ``Path('./')``.
    **kwargs : :class:`.params.BayesEoRParser` attributes
        Catch-all for auxiliary BayesEoRParser attributes so the function may
        be called using the arguments parsed by a BayesEoRParser instance via
        e.g. `generate_file_root(**args)`.

    Returns
    -------
    file_root : str
        Sampler output directory name.

    """
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    file_root = f"{nu}-{nv}-{neta}-{nq}"
    if fit_for_monopole:
        file_root += "-ffm"
    if beta is not None:
        beta_str = ""
        for b in beta:
            beta_str += f"-{b:.2f}"
        file_root += beta_str
    file_root += f"-{sigma:.1E}"
    if log_priors:
        file_root += "-lp"
    if dimensionless_PS:
        file_root += "-dPS"
    if use_Multinest:
        file_root = "MN-" + file_root
    else:
        file_root = "PC-" + file_root
    if use_shg:
        file_root += (
            f"-SHG-{nu_sh}-{nv_sh}-{nq_sh}"
        )
        if fit_for_shg_amps:
            file_root += "-ffsa"
    file_root += "-v1"
    
    # Add code from generate_output_file_base here
    # These file suffixes are used to check if a given output_dir / file_root
    # directory contains an analysis which has already started.
    suffixes = ["phys_live.txt", ".resume", "resume.dat"]

    def check_for_files(directory, suffixes):
        Nfiles = 0
        for suffix in suffixes:
            Nfiles += len(list(directory.glob(f"*{suffix}")))
        return Nfiles > 0

    while check_for_files(output_dir / file_root, suffixes):
        current_version = int(file_root.split("-v")[-1])
        next_version = current_version + 1
        file_root = file_root.replace(
            f"v{current_version}", f"v{next_version}"
        )

    file_root += "/"

    return file_root

def get_vis_data(
    *,
    data_path : Path | str,
    ant_str : str = "cross",
    bl_cutoff : Quantity | float | None = None,
    freq_idx_min : int | None = None,
    freq_min : Quantity | float | None = None,
    freq_center : Quantity | float | None = None,
    nf : int | None = None,
    df : Quantity | float | None = None,
    jd_idx_min : int | None = None,
    jd_min : Time | float | None = None,
    jd_center : Time | float | None = None,
    nt : int | None = None,
    dt : Quantity | float | None = None,
    form_pI : bool = True,
    pI_norm : float = 1.0,
    pol : str = "xx",
    redundant_avg : bool = False,
    uniform_redundancy : bool = False,
    phase : bool = False,
    phase_time : Time | float | None = None,
    calc_noise : bool = False,
    sigma : float | None = None,
    noise_seed : int | None = 742123,
    save_vis : bool = False,
    save_model : bool = False,
    save_dir : Path | str = Path("./"),
    clobber : bool = False,
    noise_data_path : Path | str | None = None,
    inst_model : Path | str | None = None,
    return_uvd : bool = False,
    verbose : bool = False,
    rank : int = 0,
    **kwargs
):
    """
    Load or generate a one-dimensional visibility vector.

    Parameters
    ----------
    data_path : pathlib.Path or str
        Path to either a pyuvdata-compatible visibility file or a preprocessed
        numpy-compatible visibility vector in units of mK sr. Defaults to None.
    ant_str : str, optional
        Antenna downselect string. If `data_path` points to a
        pyuvdata-compatible visibility file, `ant_str` determines what
        baselines to keep in the data vector. Please see
        `pyuvdata.UVData.select` for more details. Defaults to 'cross'
        (cross-correlation baselines only).
    bl_cutoff : astropy.Quantity or float, optional
        Baseline length cutoff in meters. If `data_path` points to a
        pyuvdata-compatible visibility file, `bl_cutoff` determines the longest
        baselines kept in the data vector. Defaults to None (keep all
        baselines).
    freq_idx_min : int, optional
        Minimum frequency channel index to keep in the data vector. Defaults
        to None (keep all frequencies).
    freq_min : astropy.Quantity or float, optional
        Minimum frequency to keep in the data vector in hertz if not a
        Quantity. All frequencies greater than or equal to `freq_min` will be
        kept, unless `nf` is specified. Required if `data_path` points to a
        preprocessed data vector with a '.npy' suffix. Defaults to None (keep
        all frequencies).
    freq_center : astropy.Quantity or float, optional
        Central frequency, in hertz if not a Quantity, around which `nf`
        frequencies will be kept in the data vector. `nf` must also be
        passed, otherwise an error is raised. Defaults to None (keep all
        frequencies).
    nf : int, optional
        Number of frequency channels. Required if `data_path` points to a
        preprocessed data vector with a '.npy' suffix. Otherwise, sets the
        number of frequencies to keep starting from `freq_idx_min`, the
        channel corresponding to `freq_min`, or around `freq_center`.
        Defaults to None (keep all frequencies).
    df : astropy.Quantity or float, optional
        Frequency channel width in hertz if not a Quantity. Required if
        `data_path` points to a preprocessed data vector with a '.npy' suffix.
        Overwritten by the frequency channel width in the UVData object if
        `data_path` points to a pyuvdata-compatible file containing
        visibilities. Defaults to None.
    jd_idx_min : int, optional
        Minimum time index to keep in the data vector. Defaults to None (keep
        all times).
    jd_min : astropy.Time or float, optional
        Minimum time to keep in the data vector as a Julian date if not a Time.
        One of `jd_min` or `jd_center` is required if `data_path` points to a
        preprocessed data vector with a '.npy' suffix. Defaults to None (keep
        all times).
    jd_center : astropy.Time or float, optional
        Central time, as a Julian date if not a Time, around which `nt`
        times will be kept in the data vector. `nt` must also be passed,
        otherwise an error is raised. Defaults to None (keep all times).
    nt : int, optional
        Number of times. Required if `data_path` points to a preprocessed data
        vector with a '.npy' suffix. Otherwise, sets the number of times to
        keep starting from `jd_idx_min`, the time corresponding to `jd_min`
        or around `jd_center`. Defaults to None (keep all times).
    dt : astropy.Quantity or float, optional
        Integration time in seconds if not a Quantity. Required if `data_path`
        points to a preprocessed data vector with a '.npy' suffix. Overwritten
        by the integration time in the UVData object if `data_path` points to
        a pyuvdata-compatible file. Defaults to None.
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
    phase : bool, optional
        Create a "phasor vector" which can be used to phase each visibility
        in the data vector as a function of baseline, time, and frequency via
        element-wise multiplication. Defaults to False.
    phase_time : astropy.Time or float, optional
        The time to which the visibilities will be phased as a Julian date if
        not a Time. If `phase` is True and `phase_time` is None, `phase_time`
        will be automatically set to the central time in the data. Used only
        if `data_path` points to a pyuvdata-compatible visibility file.
        Defaults to None.
    calc_noise : bool, optional
        Calculate a noise estimate from the visibilities via differencing
        adjacent times per baseline and frequency. Used only if `data_path`
        points to a pyuvdata-compatible visibility file. Defaults to False.
    sigma : float, optional
        Standard deviation of the visibility noise in mK sr. Required if
        `calc_noise` is False and `data_path` points to a pyuvdata-compatible
        visibility file or `noise_data_path` is None and `data_path` points to
        a preprocessed numpy-compatible visibility vector. Defaults to None.
    noise_seed : int, optional
        Used to seed `np.random` when generating the noise vector. Defaults to
        742123.
    save_vis : bool, optional
        Write visibility vector to disk in `save_dir`. If `calc_noise` is True,
        also save the noise vector. Used only if `data_path` points to a
        pyuvdata-compatible visibility file. Defaults to False.
    save_model : bool, optional
        Write instrument model (antenna pairs, (u, v, w) sampling, and
        redundancy model) to disk in `save_dir`. If `phase` is True, also save
        the phasor vector. Used only if `data_path` points to a
        pyuvdata-compatible visibility file. Defaults to False.
    save_dir : pathlib.Path or str, optional
        Output directory if `save_vis` or `save_model` is True. Defaults to
        ``Path('./')``.
    clobber : bool, optional
        Clobber files on disk if they exist. Defaults to False.
    noise_data_path : pathlib.Path or str, optional
        Path to a preprocessed numpy-compatible noise visibility vector in
        units of mK sr. Defaults to None.
    inst_model : pathlib.Path or str, optional
        Path to directory containing instrument model files (`uvw_array.npy`,
        `redundancy_model.npy`, and optionally `phasor_vector.npy`). Required
        if `data_path` points to a preprocessed data vector with a '.npy'
        suffix. Used only if `include_instrumental_effects` is True. Defaults
        to None.
    return_uvd : bool, optional
        Return the UVData object if `data_path` points to a pyuvdata-compatible
        file. Defaults to False.
    verbose : bool, optional
        Verbose output. Defaults to False.
    rank : int, optional
        MPI rank. Defaults to 0.
    **kwargs : :class:`.params.BayesEoRParser` attributes
        Catch-all for auxiliary BayesEoRParser attributes so the function may
        be called using the arguments parsed by a BayesEoRParser instance via
        e.g. `get_vis_data(**args)`.

    Returns
    -------
    vis_dict : dict
        Dictionary with the following key: value pairs

        - vis_noisy: noisy visibility vector with shape (nf*nbls*nt,)
        - noise: noise vector with shape (nf*nbls*nt,)
        - bl_conj_pairs_map: array index mapping of conjugate baseline pairs
        - uvws: instrumentally sampled (u, v, w) coordinates with shape
          (nt, nbls, 3)
        - redundancy: number of redundantly averaged baselines per (u, v, w)
          with shape (nt, nbls, 1)
        - freqs: frequency channels in Hz
        - df: frequency channel width in Hz
        - jds: Julian dates
        - dt: integration time in seconds
        - vis: optional noise free visibility vector with shape (nf*nbls*nt,)
          if `noise_data_path` is None or `calc_noise` is False
        - antpairs: optional baseline antenna pairs for each (u, v, w) if
          `inst_model` contains a file 'antpairs.npy' or if `data_path` points
          to a pyuvdata-compatible file
        - phasor: optional phasor vector if `phased` is True with shape
          (nf*nbls*nt,)
        - tele_name: optional telescope name if `data_path` points to a
          pyuvdata-compatible file with a valid telescope name attribute
        - uvd: optional UVData object if `data_path` points to a
          pyuvdata-compatible file and `return_uvd` is True

    """
    # If `noise_data_path` is None or `calc_noise` is False,
    # the input visbilities are assumed to be noise free and we
    # need to simulate noise to form a noisy visibility vector.
    simulate_noise = noise_data_path is None or not calc_noise
    if simulate_noise and sigma is None:
        raise ValueError(
            "sigma cannot be None if noise_data_path is None or calc_noise "
            "is False. The input visibilities in either case are assumed to "
            "be noise free and require sigma to generate noise."
        )

    # print_rank will only trigger print if verbose is True and rank == 0
    print_rank = 1 - (verbose and rank == 0)

    if data_path is not None:
        if not isinstance(data_path, Path):
            data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"{data_path} does not exist")
        if data_path.suffix == ".npy":
            required_freq = np.all([
                nf is not None,
                df is not None,
                freq_min is not None or freq_center is not None
            ])
            if not required_freq:
                raise ValueError(
                    "nf, df, and one of (freq_min, freq_center) are all "
                    "required kwargs when loading a preprocessed data vector "
                    "(data_path has a .npy suffix)"
                )
            required_time = np.all([
                nt is not None,
                dt is not None,
                jd_min is not None or jd_center is not None
            ])
            if not required_time:
                raise ValueError(
                    "nt, dt, and one of (jd_min, jd_center) are all "
                    "required kwargs when loading a preprocessed data vector "
                    "(data_path has a .npy suffix)"
                )
            if inst_model is None:
                raise ValueError(
                    "inst_model is required when loading a preprocessed data "
                    "vector (data_path has a .npy suffix)"
                )

            if not isinstance(inst_model, Path):
                inst_model = Path(inst_model)
            required_files = ["uvw_model.npy", "redundancy_model.npy"]
            required_files_exist = np.all(
                [(inst_model / f).exists() for f in required_files]
            )
            if not required_files_exist:
                raise ValueError(
                    "inst_model must point to a directory with "
                    "uvw_model.npy and redundancy_model.npy"
                )

            mpiprint(
                "\nLoading numpy-compatible data:", style="bold", rank=rank
            )
            mpiprint(f"\nReading data from: {data_path}", rank=rank)
            vis = load_numpy_dict(data_path)

            mpiprint(f"Reading instrument model from: {inst_model}", rank=rank)
            uvws, redundancy, antpairs, phasor = load_inst_model(inst_model)

            if noise_data_path is not None:
                if not isinstance(noise_data_path, Path):
                    noise_data_path = Path(noise_data_path)
                if not noise_data_path.exists():
                    raise FileNotFoundError(
                        f"{noise_data_path} does not exist"
                    )
                noise = load_numpy_dict(noise_data_path)
            else:
                noise = None
            
            tele_name = None
            uvd = None
            if freq_min is not None:
                freqs = freq_min + np.arange(nf)*df
            else:
                freqs = freq_center + np.arange(-(nf//2), nf//2 + nf%2)*df
            if jd_min is not None:
                jds = (
                    Time(jd_min, format="jd")
                    + np.arange(nt)*(dt*units.s).to("d")
                )
            else:
                jds = (
                    Time(jd_center, format="jd")
                    + np.arange(-(nt//2), nt//2 + nt%2)*(dt*units.s).to("d")
                )
            jds = jds.jd

        elif data_path.suffix in [".uvh5", ".uvfits", ".ms"]:
            mpiprint(
                "\nPreprocessing pyuvdata-compatibile data:",
                style="bold",
                rank=rank
            )
            vis, antpairs, uvws, redundancy, phasor, noise, uvd = \
                preprocess_uvdata(
                    data_path,
                    ant_str=ant_str,
                    bl_cutoff=bl_cutoff,
                    freq_idx_min=freq_idx_min,
                    freq_min=freq_min,
                    freq_center=freq_center,
                    Nfreqs=nf,
                    jd_idx_min=jd_idx_min,
                    jd_min=jd_min,
                    jd_center=jd_center,
                    Ntimes=nt,
                    form_pI=form_pI,
                    pI_norm=pI_norm,
                    pol=pol,
                    redundant_avg=redundant_avg,
                    uniform_redundancy=uniform_redundancy,
                    phase=phase,
                    phase_time=phase_time,
                    calc_noise=calc_noise,
                    return_uvd=True,
                    save_vis=save_vis,
                    save_model=save_model,
                    save_dir=save_dir,
                    clobber=clobber,
                    verbose=verbose,
                    rank=rank
                )
            # Check if the frequency array has the Nspws axis for
            # backwards compatibility with old versions of pyuvdata
            trim_nspws_ax = len(uvd.freq_array.shape) > 1
            freqs = uvd.freq_array
            if trim_nspws_ax:
                freqs = freqs[0]
            df = freqs[1] - freqs[0]
            nf = freqs.size

            jds = Time(np.unique(uvd.time_array), format="jd")
            dt = (jds[1] - jds[0]).to("s").value
            jds = jds.jd
            nt = jds.size

            try:
                # Old versions of pyuvdata use telescope_name atribute which
                # has been replaced by telescope.name in newer versions
                tele_name = uvd.telescope_name
            except:
                tele_name = uvd.telescope.name

        if simulate_noise:
            mpiprint("\nGenerating visibility noise:", style="bold", rank=rank)
            mpiprint(f"\nNoise std. dev. = {sigma:.2e} mK sr", rank=rank)
            vis_noisy, noise, bl_conj_pairs_map = generate_gaussian_noise(
                sigma,
                vis,
                nf,
                nt,
                uvws[0],
                redundancy[0],
                random_seed=noise_seed,
                rank=print_rank
            )
        else:
            # The input visibilities are noisy
            vis_noisy = vis

        vis_dict = {
            "vis_noisy": vis_noisy,
            "noise": noise,
            "bl_conj_pairs_map": bl_conj_pairs_map,
            "uvws": uvws,
            "redundancy": redundancy,
            "freqs": freqs,
            "df": df,
            "jds": jds,
            "dt": dt,
        }
        if simulate_noise:
            vis_dict["vis"] = vis
        if antpairs is not None:
            vis_dict["antpairs"] = antpairs
        if phase:
            vis_dict["phasor"] = phasor
        if tele_name is not None:
            vis_dict["tele_name"] = tele_name
        if uvd is not None and return_uvd:
            vis_dict["uvd"] = uvd
    else:
        # FIXME
        raise NotImplementedError(
            "data_path must not be None. Internal simulated data will be "
            "implemented in the future."
        )

    return vis_dict

def build_k_cube(
    *,
    nu : int,
    nv : int,
    neta : int,
    ps_box_size_ra_Mpc : float,
    ps_box_size_dec_Mpc : float,
    ps_box_size_para_Mpc : float,
    save_k_vals : bool = False,
    output_dir : Path | str = "./",
    clobber : bool = False,
    verbose : bool = False,
    rank : int = 0,
    **kwargs
):
    """
    Build the model k cube.

    Parameters
    ----------
    nu : int
        Number of pixels on a side for the u-axis in the EoR model uv-plane.
    nv : int
        Number of pixels on a side for the v-axis in the EoR model uv-plane.
    neta : int
        Number of Line of Sight (LoS, frequency axis) Fourier modes.
    ps_box_size_ra_Mpc : float
        Right Ascension (RA) axis extent of the cosmological volume in Mpc from
        which the power spectrum is estimated.
    ps_box_size_dec_Mpc : float
        Declination (DEC) axis extent of the cosmological volume in Mpc from
        which the power spectrum is estimated.
    ps_box_size_para_Mpc : float
        LoS extent of the cosmological volume in Mpc from which the power
        spectrum is estimated.
    save_k_vals : bool, optional
        Save k bin files (means, edges, and number of voxels in each bin).
        Defaults to False.
    output_dir : pathlib.Path or str, optional
        Output directory if `save_k_vals` is True.
    clobber : bool, optional
        Clobber files on disk if they exist.  Defaults to False.
    verbose : bool, optional
        Verbose output. Defaults to False.
    rank : int, optional
        MPI rank. Defaults to 0.
    **kwargs : :class:`.params.BayesEoRParser` attributes
        Catch-all for auxiliary BayesEoRParser attributes so the function may
        be called using the arguments parsed by a BayesEoRParser instance via
        e.g. `build_k_cube(**args)`.

    Returns
    -------
    k_vals : numpy.ndarray
        Mean of each k bin.
    k_cube_voxels_in_bin : list
        List containing sublists for each k bin.  Each sublist contains the
        flattened 3D k-space cube index of all |k| that fall within a given
        k bin.

    """
    required_kwargs = [
        nu,
        nv,
        neta,
        ps_box_size_ra_Mpc,
        ps_box_size_dec_Mpc,
        ps_box_size_para_Mpc
    ]
    if not np.all([arg is not None for arg in required_kwargs]):
        raise ValueError(
            "nu, nv, neta, ps_box_size_ra_Mpc, ps_box_size_dec_Mpc, "
            "and ps_box_size_para_Mpc must all not be None"
        )

    # print_rank will only trigger print if verbose is True and rank == 0
    print_rank = 1 - (verbose and rank == 0)
    mod_k, _, _, _, _, _, _ = generate_k_cube_in_physical_coordinates(
        nu,
        nv,
        neta,
        ps_box_size_ra_Mpc,
        ps_box_size_dec_Mpc,
        ps_box_size_para_Mpc
    )
    mod_k_vo = mask_k_cube(mod_k)
    k_cube_voxels_in_bin, modkbins_containing_voxels = \
        generate_k_cube_model_spherical_binning(
            mod_k_vo, ps_box_size_para_Mpc
        )
    k_vals = calc_mean_binned_k_vals(
        mod_k_vo,
        k_cube_voxels_in_bin,
        save_k_vals=(save_k_vals and rank == 0),
        k_vals_dir=output_dir,
        clobber=clobber
    )

    return k_vals, k_cube_voxels_in_bin

def generate_array_dir(
    *,
    nu : int,
    nv : int,
    nu_fg : int,
    nv_fg : int,
    neta : int,
    fit_for_monopole : bool = False,
    use_shg : bool = False,
    nu_sh : int | None = None,
    nv_sh : int | None = None,
    nq_sh : int | None = None,
    fit_for_shg_amps : bool = False,
    nu_min_MHz : float,
    channel_width_MHz=None,
    nq : int = 0,
    beta : list[float] | None = None,
    sigma : float,
    nside : int,
    fov_ra_eor : float,
    fov_dec_eor : float,
    fov_ra_fg : float,
    fov_dec_fg : float,
    simple_za_filter : bool = True,
    include_instrumental_effects : bool = True,
    telescope_name : str = "",
    nbls : int | None = None,
    nt : int | None = None,
    dt : float | None = None,
    drift_scan : bool = True,
    beam_type : str | None = None,
    beam_center : list[float] | None = None,
    achromatic_beam : bool = False,
    beam_peak_amplitude : float | None = 1.0,
    fwhm_deg : float | None = None,
    antenna_diameter : float | None = None,
    cosfreq : float | None = None,
    beam_ref_freq : float | None = None,
    noise_data_path : Path | str | None = None,
    taper_func : str | None = None,
    array_dir_prefix : Path | str = Path("./matrices/"),
    **kwargs
):
    """
    Generate the directory name for BayesEoR matrices based on analysis params.

    Parameters
    ----------
    nu : int
        Number of pixels on a side for the u-axis in the EoR model uv-plane.
    nv : int
        Number of pixels on a side for the v-axis in the EoR model uv-plane.
    nu_fg : int
        Number of pixels on a side for the u-axis in the foreground model
        uv-plane.
    nv_fg : int
        Number of pixels on a side for the v-axis in the foreground model
        uv-plane.
    neta : int
        Number of Line of Sight (LoS, frequency axis) Fourier modes.
    fit_for_monopole : bool, optional
        Fit for the (u, v) = (0, 0) pixel in the model uv-plane. Defaults to
        False.
    use_shg : bool, optional
        Use the subharmonic grid. Defaults to False.
    nu_sh : int, optional
        Number of pixels on a side for the u-axis in the subharmonic grid
        model uv-plane. Used only if `use_shg` is True. Defaults to None.
    nv_sh : int, optional
        Number of pixels on a side for the v-axis in the subharmonic grid
        model uv-plane. Used only if `use_shg` is True. Defaults to None.
    nq_sh : int, optional
        Number of large spectral scale model quadratic basis vectors for the
        subharmonic grid. Used only if `use_shg` is True. Defaults to None.
    fit_for_shg_amps : bool, optional
        Fit for the amplitudes of the subharmonic grid pixels. Defaults to
        False.
    nu_min_MHz : float
        Minimum frequency in megahertz.
    channel_width_MHz : float
        Frequency channel width in megahertz.
    nq : int, optional
        Number of large spectral scale model quadratic basis vectors. Defaults
        to 0.
    beta : list of float, optional
        Brightness temperature power law spectral index/indices used in the
        large spectral scale model. Defaults to None.
    sigma : float
        Standard deviation of the visibility noise in mK sr.
    nside : int
        HEALPix nside parameter.
    fov_ra_eor : float
        Field of view of the Right Ascension axis of the EoR sky model in
        degrees.
    fov_dec_eor : float
        Field of view of the Declination axis of the EoR sky model in degrees.
    fov_ra_fg : float
        Field of view of the Right Ascension axis of the foreground sky model
        in degrees.
    fov_dec_fg : float
        Field of view of the Declination axis of the foreground sky model in
        degrees.
    simple_za_filter : bool, optional
        Filter pixels in the sky model by zenith angle only. Defaults to True.
    include_instrumental_effects : bool, optional
        Forward model an instrument. Defaults to True.
    telescope_name : str, optional
        Telescope identifier string. Used only if
        `include_instrumental_effects` is True. Defaults to ''.
    nbls : int, optional
        Number of baselines being modelled. Used only if
        `include_instrumental_effects` is True. Defaults to None.
    nt : float, optional
        Number of times. Used only if `include_instrumental_effects` is True.
        Defaults to None.
    dt : float, optional
        Integration time in seconds. Used only if
        `include_instrumental_effects` is True. Defaults to None.
    drift_scan : bool, optional
        Model drift scan (True) or phased (False) visibilities. Used only if
        `include_instrumental_effects` is True. Defaults to True.
    beam_type : str, optional
        Path to a pyuvdata-compatible beam file or one of 'uniform',
        'gaussian', 'airy', 'gausscosine', or 'taperairy'. Used only if
        `include_instrumental_effects` is True. Defaults to None.
    beam_center : list of float, optional
        Beam center offsets from the phase center in right ascension and
        declination in degrees. Used only if `include_instrumental_effects` is
        True. Defaults to None.
    achromatic_beam : bool, optional
        Force the beam to be achromatic. The frequency at which the beam will
        be calculated is set via `beam_ref_freq`. Used only if
        `include_instrumental_effects` is True. Defaults to False.
    beam_peak_amplitude : float, optional
        Peak amplitude of the beam. Used only if `include_instrumental_effects`
        is True. Defaults to 1.0.
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
    beam_ref_freq : float, optional
        Beam reference frequency in megahertz. Used only if
        `include_instrumental_effects` is True. Defaults to None.
    noise_data_path : pathlib.Path or str, optional
        Path to a preprocessed numpy-compatible noise visibility vector in
        units of mK sr. Defaults to None.
    taper_func : str, optional
        Taper function applied to the frequency axis of the visibilities.
        Can be any valid argument to `scipy.signal.windows.get_window`.
        Defaults to None.
    array_dir_prefix : pathlib.Path or str, optional
        Array directory prefix. Defaults to ``Path('./matrices/')``.
    **kwargs : :class:`.params.BayesEoRParser` attributes
        Catch-all for auxiliary BayesEoRParser attributes so the function may
        be called using the arguments parsed by a BayesEoRParser instance via
        e.g. `generate_array_dir(**args)`.

    Returns
    -------
    matrices_path : str
        Path containing the uniquely identifying info for each analysis, i.e.
        model parameters and the instrument model.

    """
    if array_dir_prefix is not None:
        if not isinstance(array_dir_prefix, Path):
            array_dir_prefix = Path(array_dir_prefix)
        matrices_path = array_dir_prefix
    else:
        matrices_path = Path("./")

    model_str = f"nu{nu}-nv{nv}-neta{neta}-nq{nq}"
    if not (nu == nu_fg and nv == nv_fg):
        model_str += f"-nufg{nu_fg}-nvfg{nv_fg}"
    if fit_for_monopole:
        model_str += "-ffm"
    if use_shg:
        shg_str = "shg"
        if nu_sh > 0:
            shg_str += f"-nush{nu_sh}"
        if nv_sh > 0 and nu_sh != nv_sh:
            shg_str += f"-nvsh{nv_sh}"
        if nq_sh > 0:
            shg_str += f"-nqsh{nq_sh}"
        if fit_for_shg_amps:
            shg_str += "-ffsa"
        model_str += f"-{shg_str}"
    matrices_path /= model_str

    freq_str = f"fmin{nu_min_MHz:.2f}MHz-df{channel_width_MHz*1e3:.2f}kHz"
    if nq > 0 and beta is not None:
        beta_str = "beta-"
        beta_str += "-".join([f"{beta[i]:.2f}" for i in range(len(beta))])
        freq_str += f"-{beta_str}"
    matrices_path /= freq_str

    fovs_match = (
        fov_ra_eor == fov_ra_fg
        and fov_dec_eor == fov_dec_fg
    )
    img_str = "fov"
    if not fovs_match:
        img_str += "-eor"
    if not fov_ra_eor == fov_dec_eor and not simple_za_filter:
        img_str += f"-ra{fov_ra_eor:.1f}d-dec{fov_dec_eor:.1f}d"
    else:
        img_str += f"{fov_ra_eor:.1f}d"
    if not fovs_match:
        img_str += "-fg"
        if fov_ra_fg != fov_dec_fg and not simple_za_filter:
            img_str += f"-ra{fov_ra_fg:.1f}d-dec{fov_dec_fg:.1f}d"
        else:
            img_str += f"{fov_ra_fg:.1f}d"
    if not simple_za_filter:
        img_str += "-rect"
    img_str += f"-nside{nside}"
    matrices_path /= img_str
    
    if include_instrumental_effects:
        inst_str = ""
        if telescope_name != "":
            inst_str += f"{telescope_name}-"
        if nbls is not None:
            inst_str += f"nbls{nbls}-"
        inst_str += f"nt{nt}-dt{dt:.2f}s"
        if not drift_scan:
            inst_str += "-phased"
        matrices_path /= inst_str
    
        beam_str = ""
        if not "." in beam_type:
            beam_str = f"{beam_type}"
            if achromatic_beam:
                beam_str = "achromatic-" + beam_str
            if (not beam_peak_amplitude == 1
                and beam_type in ["uniform", "gaussian", "gausscosine"]):
                beam_str += f"-peak{beam_peak_amplitude}"
            
            if beam_type in ["gaussian", "gausscosine"]:
                if fwhm_deg is not None:
                    beam_str += f"-fwhm{fwhm_deg:.4f}d"
                elif antenna_diameter is not None:
                    beam_str += (
                        f"-diam{antenna_diameter}m"
                    )
                if beam_type == "gausscosine":
                    beam_str += f"-cosfreq{cosfreq:.2f}wls"
            elif beam_type in ["airy", "taperairy"]:
                beam_str += f"-diam{antenna_diameter}m"
                if beam_type == "taperairy":
                    beam_str += f"-fwhm{fwhm_deg}d"
            if achromatic_beam:
                beam_str += f"-fref{beam_ref_freq:.2f}MHz"
        else:
            beam_str = Path(beam_type).stem

        if beam_center is not None:
            beam_center_signs = [
                "+" if beam_center[i] >= 0 else "" for i in range(2)
            ]
            beam_center_str = "bmctr-RA0{}{:.2f}-DEC0{}{:.2f}".format(
                    beam_center_signs[0],
                    beam_center[0],
                    beam_center_signs[1],
                    beam_center[1]
            )
            beam_str += f"-{beam_center_str}"

        matrices_path /= f"{beam_str}"

    noise_str = f"sigma{sigma:.2e}"
    if noise_data_path is not None:
        noise_str += "-noisevec"
    matrices_path /= noise_str
    
    if taper_func:
        matrices_path /= f"{taper_func}"

    return str(matrices_path) + "/"

def build_matrices(
    *,
    nu : int,
    du_eor : float,
    nv : int,
    dv_eor : float,
    nu_fg : int,
    du_fg : float,
    nv_fg : int,
    dv_fg : float,
    nf : int,
    neta : int,
    deta : float,
    fit_for_monopole : bool = False,
    use_shg : bool = False,
    nu_sh : int | None = None,
    nv_sh : int | None = None,
    nq_sh : int | None = None,
    npl_sh : int | None = None,
    fit_for_shg_amps : bool = False,
    nu_min_MHz : float,
    channel_width_MHz : float,
    nq : int = 0,
    npl : int = 0,
    beta : list[float] | None = None,
    sigma : float,
    nside : int,
    fov_ra_eor : float,
    fov_dec_eor : float,
    fov_ra_fg : float,
    fov_dec_fg : float,
    simple_za_filter : bool = True,
    include_instrumental_effects : bool = True,
    telescope_latlonalt : Sequence[float] | None = None,
    nt : int,
    jd_center : float,
    dt : float,
    beam_type : str,
    beam_center : list[float] | None = None,
    achromatic_beam : bool = False,
    beam_peak_amplitude : float = 1.0,
    fwhm_deg : float | None = None,
    antenna_diameter : float | None = None,
    cosfreq : float | None = None,
    beam_ref_freq : float | None = None,
    drift_scan : bool = True,
    telescope_name : str = "",
    uvws : np.ndarray,
    redundancy : np.ndarray,
    noise : np.ndarray,
    phasor : np.ndarray | None = None,
    taper_func : str | None = None,
    noise_data_path : Path | str | None = None,
    use_sparse_matrices : bool = True,
    build_Finv_and_Fprime : bool = True,
    array_dir_prefix : Path | str = Path("./matrices/"),
    mkdir : bool = True,
    build_stack : bool = True,
    clobber : bool = False,
    verbose : bool = False,
    rank : int = 0
):
    """
    Create a directory for and build the BayesEoR matrix stack.

    Parameters
    ----------
    nu : int
        Number of pixels on a side for the u-axis in the EoR model uv-plane.
    du_eor : float
        Fourier mode spacing along the u axis in inverse radians of the
        EoR model uv-plane.
    nv : int
        Number of pixels on a side for the v-axis in the EoR model uv-plane.
    dv_eor : float
        Fourier mode spacing along the v axis in inverse radians of the
        EoR model uv-plane.
    nu_fg : int
        Number of pixels on a side for the u-axis in the foreground model
        uv-plane.
    du_fg : float
        Fourier mode spacing along the u axis in inverse radians of the
        FG model uv-plane.
    nv_fg : int
        Number of pixels on a side for the v-axis in the foreground model
        uv-plane.
    dv_fg : float
        Fourier mode spacing along the v axis in inverse radians of the
        FG model uv-plane.
    nf : int
        Number of frequency channels.
    neta : int
        Number of Line of Sight (LoS, frequency axis) Fourier modes.
    deta : float
        Fourier mode spacing along the eta (line of sight, frequency) axis in
        inverse Hz.
    fit_for_monopole : bool, optional
        Fit for the (u, v) = (0, 0) pixel in the model uv-plane.
    use_shg : bool, optional
        Use the subharmonic grid. Defaults to False.
    nu_sh : int, optional
        Number of pixels on a side for the u-axis in the subharmonic grid
        model uv-plane. Used only if `use_shg` is True. Defaults to None.
    nv_sh : int, optional
        Number of pixels on a side for the v-axis in the subharmonic grid
        model uv-plane. Used only if `use_shg` is True. Defaults to None.
    nq_sh : int, optional
        Number of large spectral scale model quadratic basis vectors for the
        subharmonic grid. Used only if `use_shg` is True. Defaults to None.
    npl_sh : int, optional
        Number of large spectral scale model power law basis vectors for the
        subharmonic grid. Defaults to None.
    fit_for_shg_amps : bool, optional
        Fit for the amplitudes of the subharmonic grid pixels. Defaults to
        False.
    nu_min_MHz : float
        Minimum frequency in megahertz.
    channel_width_MHz : float
        Frequency channel width in megahertz.
    nq : int, optional
        Number of large spectral scale model quadratic basis vectors. Defaults
        to 0.
    npl : int, optional
        Number of large spectral scale model power law basis vectors. Defaults
        to 0.
    beta : list of float, optional
        Brightness temperature power law spectral index/indices used in the
        large spectral scale model. Defaults to None.
    sigma : float
        Standard deviation of the visibility noise in mK sr.
    nside : int
        HEALPix nside parameter.
    fov_ra_eor : float
        Field of view of the Right Ascension axis of the EoR sky model in
        degrees.
    fov_dec_eor : float
        Field of view of the Declination axis of the EoR sky model in degrees.
    fov_ra_fg : float
        Field of view of the Right Ascension axis of the foreground sky model
        in degrees.
    fov_dec_fg : float
        Field of view of the Declination axis of the foreground sky model in
        degrees.
    simple_za_filter : bool, optional
        Filter pixels in the sky model by zenith angle only. Defaults to True.
    include_instrumental_effects : bool, optional
        Forward model an instrument. Defaults to True.
    telescope_latlonalt : sequence of floats, optional
        Telescope location tuple as (latitude in degrees, longitude in degrees,
        altitude in meters). Required if `include_instrumental_effects` is
        True. Defaults to None.
    nt : float
        Number of times.
    jd_center : float
        Central time as a Julian date.
    dt : float
        Integration time in seconds.
    beam_type : str
        Path to a pyuvdata-compatible beam file or one of 'uniform',
        'gaussian', 'airy', 'gausscosine', or 'taperairy'. Used only if
        `include_instrumental_effects` is True. Defaults to None.
    beam_center : list of float, optional
        Beam center offsets from the phase center in right ascension and
        declination in degrees. Used only if `include_instrumental_effects`
        is True.
    achromatic_beam : bool, optional
        Force the beam to be achromatic. The frequency at which the beam will
        be calculated is set via `beam_ref_freq`. Used only if
        `include_instrumental_effects` is True. Defaults to False.
    beam_peak_amplitude : float
        Peak amplitude of the beam. Used only if `include_instrumental_effects`
        is True. Defaults to 1.0.
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
    beam_ref_freq : float, optional
        Beam reference frequency in megahertz. Used only if
        `include_instrumental_effects` is True. Defaults to None.
    drift_scan : bool, optional
        Model drift scan (True, default) or phased (False) visibilities.
    telescope_name : str, optional
        Telescope identifier string. Defaults to ''.
    uvws : numpy.ndarray
        Array containing the (u(t), v(t), w(t)) coordinates of the instrument
        model with shape (nt, nbls, 3).
    redundancy : numpy.ndarray
        Array containing the number of redundant baselines at each
        (u(t), v(t), w(t)) in the instrument model with shape (nt, nbls, 1).
    noise : numpy.ndarray
        Noise vector with shape (nt*nf*nbls,).
    phasor : numpy.ndarray, optional
        Array with shape (nt*nf*nbls,) that contains the phasor term used to
        phase visibilities after performing the nuDFT from HEALPix (l, m, f) to
        instrumentally sampled, unphased (u, v, f).  Defaults to None, i.e.
        modelling unphased visibilities.
    taper_func : str, optional
        Taper function applied to the frequency axis of the visibilities.
        Can be any valid argument to `scipy.signal.windows.get_window`.
        Defaults to None.
    noise_data_path : pathlib.Path or str, optional
        Path to a preprocessed numpy-compatible noise visibility vector in
        units of mK sr. Defaults to None.
    use_sparse_matrices : bool, optional
        Use sparse arrays. Defaults to True.
    build_Finv_and_Fprime : bool, optional
        Construct Finv and Fprime independently and save both matrices to disk.
        Otherwise, construct the matrix product Finv_Fprime in place from the
        dense matrices comprising Finv and Fprime to minimize the memory and
        time required to build the matrix stack.  In this case, only the
        matrix product Finv_Fprime is written to disk. Defaults to True.
    array_dir_prefix : pathlib.Path or str, optional
        Array directory prefix. Defaults to ``Path('./matrices/')``.
    mkdir : bool, optional
        Make array directory (including parents). Defaults to True.
    build_stack : bool, optional
        Build the matrix stack (True) or instantiate the BuildMatrices class
        only (False). Defaults to True.
    clobber : bool, optional
        Clobber files on disk if they exist.  Defaults to False.
    verbose : bool, optional
        Verbose output. Defaults to False.
    rank : int, optional
        MPI rank. Defaults to 0.

    Returns
    -------
    bm : :class:`.matrices.build.BuildMatrices`
        BuildMatrices class instance used to construct the matrix stack.

    """
    if noise is not None:
        warnings.warn(
            "There is a known issue when `noise` is not None (please see "
            "BayesEoR issue #55 for more details.  For now, `noise` will "
            "be forced to None."
        )
        noise = None
    if include_instrumental_effects and telescope_latlonalt is None:
        raise ValueError(
            "telescope_latlonalt cannot be None if "
            "include_instrumental_effects is true"
        )

    # print_rank will only trigger print if verbose is True and rank == 0
    print_rank = 1 - (verbose and rank == 0)

    if uvws is not None:
        nbls = len(uvws[0])
        uvws_vec = uvws.copy().reshape(-1, 3)
        n_vis = len(uvws_vec)
    else:
        nbls = None
        uvws_vec = None
        n_vis = None
    
    if redundancy is not None:
        redundancy_vec = redundancy.copy().reshape(-1, 1).flatten()

    array_dir = generate_array_dir(
        nu=nu,
        nv=nv,
        nu_fg=nu_fg,
        nv_fg=nv_fg,
        neta=neta,
        fit_for_monopole=fit_for_monopole,
        use_shg=use_shg,
        nu_sh=nu_sh,
        nv_sh=nv_sh,
        nq_sh=nq_sh,
        npl_sh=npl_sh,
        fit_for_shg_amps=fit_for_shg_amps,
        nu_min_MHz=nu_min_MHz,
        channel_width_MHz=channel_width_MHz,
        nq=nq,
        npl=npl,
        beta=beta,
        sigma=sigma,
        nside=nside,
        fov_ra_eor=fov_ra_eor,
        fov_dec_eor=fov_dec_eor,
        fov_ra_fg=fov_ra_fg,
        fov_dec_fg=fov_dec_fg,
        simple_za_filter=simple_za_filter,
        telescope_name=telescope_name,
        nbls=nbls,
        nt=nt,
        dt=dt,
        beam_type=beam_type,
        beam_center=beam_center,
        drift_scan=drift_scan,
        taper_func=taper_func,
        include_instrumental_effects=include_instrumental_effects,
        achromatic_beam=achromatic_beam,
        beam_peak_amplitude=beam_peak_amplitude,
        fwhm_deg=fwhm_deg,
        antenna_diameter=antenna_diameter,
        cosfreq=cosfreq,
        beam_ref_freq=beam_ref_freq,
        noise_data_path=noise_data_path,
        array_dir_prefix=array_dir_prefix,
        mkdir=mkdir
    )
    mpiprint(
        f"\n[bold]Matrix stack directory:[/bold] {array_dir}",
        end="\n\n",
        rank=rank
    )

    if not Path(array_dir).exists() and build_stack and rank > 0:
        # We currently do not support MPI when building the matrix stack.
        # This process must be run on a single process to avoid parallel
        # processes trying to write the the same file simultaneously.
        # This check should hopefully catch cases where the matrix stack
        # is being built my >= 1 parallel processes and only proceed
        # with matrix construction on the root process with rank == 0.
        raise RuntimeError(
            f"The matrix stack cannot be built using MPI. Matrix construction "
            f"will only proceed on rank 0. Error raised on rank {rank}."
        )

    bm = BuildMatrices(
        nu=nu,
        du_eor=du_eor,
        nv=nv,
        dv_eor=dv_eor,
        nu_fg=nu_fg,
        du_fg=du_fg,
        nv_fg=nv_fg,
        dv_fg=dv_fg,
        nf=nf,
        neta=neta,
        deta=deta,
        fit_for_monopole=fit_for_monopole,
        use_shg=use_shg,
        nu_sh=nu_sh,
        nv_sh=nv_sh,
        nq_sh=nq_sh,
        npl_sh=npl_sh,
        fit_for_shg_amps=fit_for_shg_amps,
        f_min=nu_min_MHz,
        df=channel_width_MHz,
        nq=nq,
        npl=npl,
        beta=beta,
        sigma=sigma,
        nside=nside,
        fov_ra_eor=fov_ra_eor,
        fov_dec_eor=fov_dec_eor,
        fov_ra_fg=fov_ra_fg,
        fov_dec_fg=fov_dec_fg,
        simple_za_filter=simple_za_filter,
        include_instrumental_effects=include_instrumental_effects,
        telescope_latlonalt=telescope_latlonalt,
        nt=nt,
        jd_center=jd_center,
        dt=dt,
        beam_type=beam_type,
        beam_center=beam_center,
        achromatic_beam=achromatic_beam,
        beam_peak_amplitude=beam_peak_amplitude,
        fwhm_deg=fwhm_deg,
        antenna_diameter=antenna_diameter,
        cosfreq=cosfreq,
        beam_ref_freq=beam_ref_freq,
        drift_scan=drift_scan,
        uvw_array_m=uvws,
        bl_red_array=redundancy,
        phasor=phasor,
        effective_noise=None,
        taper_func=taper_func,
        array_save_directory=array_dir,
        use_sparse_matrices=use_sparse_matrices,
        Finv_Fprime=np.logical_not(build_Finv_and_Fprime),
        verbose=(verbose and rank == 0)
    )

    if build_stack:
        if clobber:
            mpiprint(
                "\nWARNING: Overwriting matrix stack\n", rank=rank,
                style="bold red", justify="center"
            )
        bm.build_minimum_sufficient_matrix_stack(
            clobber_matrices=clobber, force_clobber=clobber
        )

    return bm

def build_posterior(
    *,
    k_vals : np.ndarray,
    k_cube_voxels_in_bin : list,
    nu : int,
    nv : int,
    neta : int,
    nf : int,
    nq : int,
    redshift : float,
    Ninv : np.ndarray | sparse.spmatrix,
    dbar : np.ndarray,
    d_Ninv_d : float,
    T_Ninv_T : np.ndarray,
    ps_box_size_ra_Mpc : float,
    ps_box_size_dec_Mpc : float,
    ps_box_size_para_Mpc : float,
    include_instrumental_effects : bool = True,
    priors : Sequence[float],
    log_priors : bool = False,
    uprior_inds : np.ndarray | None = None,
    dimensionless_PS : bool = True,
    inverse_LW_power : float = 1e-16,
    block_T_Ninv_T : list | None = None,
    use_shg : bool = False,
    use_gpu : bool = True,
    use_intrinsic_noise_fitting : bool = False,
    use_LWM_Gaussian_prior : bool = False,
    verbose : bool = False,
    rank : int = 0
):
    """
    Instantiate the power spectrum posterior class.

    Parameters
    ----------
    k_vals : numpy.ndarray
        Mean of each k bin.
    k_cube_voxels_in_bin : list
        List containing sublists for each k bin.  Each sublist contains the
        flattened 3D k-space cube index of all |k| that fall within a given
        k bin.
    nu : int
        Number of pixels on a side for the u-axis in the EoR model uv-plane.
    nv : int
        Number of pixels on a side for the v-axis in the EoR model uv-plane.
    neta : int
        Number of Line of Sight (LoS, frequency axis) Fourier modes.
    nf : int
        Number of frequency channels.
    nq : int
        Number of large spectral scale model quadratic basis vectors.
    redshift : float
        Cosmological redshift.
    Ninv : numpy.ndarray or scipy.sparse
        Inverse noise covariance matrix.
    dbar : numpy.ndarray
        Inverse-noise-weighted projection of the data onto the model computed
        as :math:`\\bar{d} = T^\\dagger N^{-1}d` where :math:`d` are the noisy
        visibilities.
    d_Ninv_d : numpy.ndarray
        Matrix-vector product :math:`d^\\dagger N^{-1} d` where :math:`d` are
        the noisy visibilities.
    T_Ninv_T : numpy.ndarray
        Matrix product :math:`T^\\dagger N^{-1} T`.
    ps_box_size_ra_Mpc : float
        Right Ascension (RA) axis extent of the cosmological volume in Mpc from
        which the power spectrum is estimated.
    ps_box_size_dec_Mpc : float
        Declination (DEC) axis extent of the cosmological volume in Mpc from
        which the power spectrum is estimated.
    ps_box_size_para_Mpc : float
        LoS extent of the cosmological volume in Mpc from which the power
        spectrum is estimated.
    include_instrumental_effects : bool, optional
        Forward model an instrument. Defaults to True.
    priors : sequence of float
        Prior [min, max] for each k bin as a a sequence, e.g. [[min1, max1],
        [min2, max2], ...].
    log_priors : bool, optional
        Assume priors on power spectrum coefficients are in log_10 units.
        Defaults to False.
    uprior_inds : numpy.ndarray, optional
        Boolean 1D array that is True for any k bins using a uniform prior.
        False entries use a log-uniform prior. Defaults to None (all k bins
        use a log-uniform prior).
    dimensionless_PS : bool, optional
        Fit for the dimensionless power spectrum, :math:`\\Delta^2(k)` (True),
        or the power spectrum, :math:`P(k)` (False). Defaults to True.
    inverse_LW_power : float, optional
        Prior on the inverse power of the large spectral scale model
        coefficients. Defaults to 1e-16.
    block_T_Ninv_T : list, optional
        Block-diagonal representation of T_Ninv_T with blocks stored as arrays
        within a list. Used only if `include_instrumental_effects` is False.
        Defaults to None.
    use_shg : bool, optional
        Use the subharmonic grid. Defaults to False.
    use_gpu : bool, optional
        Use GPUs (True) or CPUs (False). Defaults to True.
    use_intrinsic_noise_fitting : bool, optional
        Fit for the noise level. Defaults to False.
    use_LWM_Gaussian_prior : bool, optional
        Use a Gaussian prior (True) or a uniform prior (False) on the large
        spectral scale model. This option is currently not implemented but
        will be reimplemented in the future. Defaults to False.
    verbose : bool, optional
        Verbose output. Defaults to False.
    rank : int, optional
        MPI rank. Defaults to 0.
    
    Returns
    -------
    pspp : :class:`.posterior.PowerSpectrumPosteriorProbability`
        Posterior probability class instance.

    """
    if use_LWM_Gaussian_prior:
        raise NotImplementedError(
            "use_LWM_Gaussian_prior is not currently implemented. It will be "
            "reimplemented in the future. For now, please set "
            "use_LWM_Gaussian_prior to False."
        )
    if not dimensionless_PS:
        raise NotImplementedError(
            "Modelling the power spectrum P(k) is not currently implemented. "
            "It will be implemented in the future. For now, please set "
            "dimensionless_PS to True."
        )
    if block_T_Ninv_T is None:
        block_T_Ninv_T = []

    # print_rank will only trigger print if verbose is True and rank == 0
    print_rank = 1 - (verbose and rank == 0)

    # The EoR model uv-plane excludes the (u, v) = (0, 0) pixel, so the number
    # of EoR model uv-plane pixels is nu*nv - 1
    nuv = nu*nv - 1

    if use_LWM_Gaussian_prior:
        # use_LWM_Gaussian_prior not implemented
        # Code copied for posterity
        mpiprint(
            "WARNING: use_LWM_Gaussian_prior is not currently implemented."
            " Results might be inaccurate.",
            style="bold red",
            justify="center",
            rank=rank
        )
        # Set minimum LW model priors using LW power spectrum in fit to
        # white noise (i.e the prior min should incorporate knowledge of
        # signal-removal in iterative pre-subtraction)
        fg_log_priors_min = np.log10(1.e5)
        # Set minimum LW model prior max using numerical stability
        # constraint at the given signal-to-noise in the data.
        fg_log_priors_max = 6.0
        # Calibrate LW model priors using white noise fitting
        priors[0] = [fg_log_priors_min, fg_log_priors_max]
        priors[1] = [fg_log_priors_min, fg_log_priors_max]
        priors[2] = [fg_log_priors_min, fg_log_priors_max]
        if use_intrinsic_noise_fitting:
            priors[1] = priors[0]
            priors[2] = priors[1]
            priors[3] = priors[2]
            priors[0] = [1.0, 2.0]  # Linear alpha_prime range
    else:
        if use_intrinsic_noise_fitting:
            priors[0] = [1.0, 2.0]  # Linear alpha_prime range
    ps_unit = "mK^2"
    if not dimensionless_PS:
        ps_unit += " Mpc^3"
    mpiprint(f"priors = {priors} {ps_unit}", rank=print_rank)

    pspp = PowerSpectrumPosteriorProbability(   
        T_Ninv_T,
        dbar,
        k_vals,
        k_cube_voxels_in_bin,
        nuv,
        neta,
        nf,
        nq,
        Ninv,
        d_Ninv_d,
        redshift,
        ps_box_size_ra_Mpc,
        ps_box_size_dec_Mpc,
        ps_box_size_para_Mpc,
        include_instrumental_effects=include_instrumental_effects,
        log_priors=log_priors,
        uprior_inds=uprior_inds,
        inverse_LW_power=inverse_LW_power,
        dimensionless_PS=dimensionless_PS,
        block_T_Ninv_T=block_T_Ninv_T,
        intrinsic_noise_fitting=use_intrinsic_noise_fitting,
        use_shg=use_shg,
        rank=rank,
        use_gpu=use_gpu,
        verbose=verbose
    )

    return pspp
