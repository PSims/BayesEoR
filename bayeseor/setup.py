""" Convenience functions for setting up a BayesEoR analysis. """

import numpy as np
from collections.abc import Sequence
from astropy.time import Time
from astropy.units import Quantity
from pathlib import Path
from rich.panel import Panel
from scipy import sparse
import warnings

from .matrices.build import BuildMatrices
from .model.instrument import load_inst_model
from .model.noise import generate_data_and_noise_vector_instrumental
from .model.k_cube import (
    generate_k_cube_in_physical_coordinates,
    mask_k_cube,
    generate_k_cube_model_spherical_binning,
    calc_mean_binned_k_vals
)
from .posterior import PowerSpectrumPosteriorProbability
from .vis import preprocess_uvdata
from .utils import mpiprint, Cosmology, load_numpy_dict, parse_uprior_inds

def run_setup(
    *,
    nf : int,
    neta : int,
    deta : float,
    nq : int = 0,
    beta : list[float] | None = None,
    nt : int,
    nu : int,
    nv : int,
    nu_fg : int,
    nv_fg : int,
    fit_for_monopole : bool = False,
    du_eor : float,
    dv_eor : float,
    du_fg : float,
    dv_fg : float,
    use_shg : bool = False,
    nu_sh : int | None = None,
    nv_sh : int | None = None,
    nq_sh : int | None = None,
    npl_sh : int | None = None,
    fit_for_shg_amps : bool = False,
    ps_box_size_ra_Mpc : float,
    ps_box_size_dec_Mpc : float,
    ps_box_size_para_Mpc : float,
    nside : int,
    fov_ra_eor : float,
    fov_dec_eor : float,
    fov_ra_fg : float,
    fov_dec_fg : float,
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
    nu_min_MHz : Quantity | float | None = None,
    freq_center : Quantity | float | None = None,
    channel_width_MHz : Quantity | float | None = None,
    jd_idx_min : int | None = None,
    jd_min : Time | float | None = None,
    central_jd : Time | float | None = None,
    integration_time_seconds : Quantity | float | None = None,
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
    telescope_latlonalt : Sequence[float] = (0, 0, 0),
    array_dir_prefix : Path | str = "./matrices/",
    use_sparse_matrices : bool = True,
    build_Finv_and_Fprime : bool = True,
    output_dir : Path | str = "./",
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
    nf : int
        Number of frequency channels. If `setup_data_vec` is True, sets the
        number of frequencies to keep starting from `freq_idx_min`, the
        channel corresponding to `nu_min_MHz`, or around `freq_center`. Defaults
        to None (keep all frequencies).
    neta : int
        Number of Line of Sight (LoS, frequency axis) Fourier modes.
    deta : float
        Fourier mode spacing along the eta (line of sight, frequency) axis in
        inverse Hz.
    nq : int
        Number of large spectral scale model basis vectors. If `beta` is None,
        the basis vectors are quadratic in frequency. If `beta` is not None,
        the basis vectors are power laws with brightness temperature spectral
        indices from `beta`.
    beta : list of float, optional
        Brightness temperature power law spectral index/indices used in the
        large spectral scale model. Can be a single spectral index, e.g.
        [2.63], or multiple spectral indices, e.g. [2.63, 2.82]. Defaults to
        [2.63, 2.82].
    nt : int
        Number of times. If `setup_data_vec` is True, sets the number of times
        to keep starting from `jd_idx_min`, the time corresponding to `jd_min`
        or around `central_jd`. Defaults to None (keep all times).
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
    fit_for_monopole : bool, optional
        Fit for the (u, v) = (0, 0) pixel in the model uv-plane.
    du_eor : float
        Fourier mode spacing along the u axis in inverse radians of the
        EoR model uv-plane.
    dv_eor : float
        Fourier mode spacing along the v axis in inverse radians of the
        EoR model uv-plane.
    du_fg : float
        Fourier mode spacing along the u axis in inverse radians of the
        FG model uv-plane.
    dv_fg : float
        Fourier mode spacing along the v axis in inverse radians of the
        FG model uv-plane.
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
        subharmonic grid. Used only if `use_shg` is True. Defaults to None.
    fit_for_shg_amps : bool, optional
        Fit for the amplitudes of the subharmonic grid pixels. Defaults to
        False.
    ps_box_size_ra_Mpc : float
        Right Ascension (RA) axis extent of the cosmological volume in Mpc from
        which the power spectrum is estimated.
    ps_box_size_dec_Mpc : float
        Declination (DEC) axis extent of the cosmological volume in Mpc from
        which the power spectrum is estimated.
    ps_box_size_para_Mpc : float
        LoS extent of the cosmological volume in Mpc from which the power
        spectrum is estimated.
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
        'gaussian', 'airy', 'gausscosine', or 'taperairy'.
    beam_center : list of float, optional
        Beam center offsets from the phase center in right ascension and
        declination in degrees. Defaults to None.
    achromatic_beam : bool, optional
        Force the beam to be achromatic. Defaults to False.
    beam_peak_amplitude : float, optional
        Peak amplitude of the beam. Defaults to 1.0.
    fwhm_deg : float, optional
        Full width at half maximum of beam in degrees. Used only if `beam_type`
        is 'airy', 'gaussian', or 'gausscosine'. Defaults to None.
    antenna_diameter : float, optional
        Antenna (aperture) diameter in meters. Used only if `beam_type` is
        'airy', 'gaussian', or 'gausscosine'. Defaults to None.
    cosfreq : float, optional
        Cosine frequency if using a 'gausscosine' beam. Defaults to None.
    beam_ref_freq : float, optional
        Beam reference frequency in MHz. Defaults to None.
    data_path : pathlib.Path or str
        Path to either a pyuvdata-compatible file containing visibilities or
        a numpy-compatible file containing a preprocessed visibility vector.
    ant_str : str, optional
        Antenna downselect string. This determines what baselines to keep in
        the data vector. Please see `pyuvdata.UVData.select` for more details.
        Defaults to 'cross' (cross-correlation baselines only).
    bl_cutoff : :class:`astropy.Quantity` or float, optional
        Baseline length cutoff in meters if not a Quantity. Defaults to None
        (keep all baselines).
    freq_idx_min : int, optional
        Minimum frequency channel index to keep in the data vector. Defaults
        to None (keep all frequencies).
    nu_min_MHz : :class:`astropy.Quantity` or float, optional
        Minimum frequency to keep in the data vector in Hertz if not a
        Quantity. All frequencies greater than or equal to `nu_min_MHz` will be
        kept, unless `nf` is specified. Defaults to None (keep all
        frequencies).
    freq_center : :class:`astropy.Quantity` or float, optional
        Central frequency, in Hertz if not a Quantity, around which `nf`
        frequencies will be kept in the data vector. `nf` must also be
        passed, otherwise an error is raised. Defaults to None (keep all
        frequencies).
    channel_width_MHz : :class:`astropy.Quantity` or float, optional
        Frequency channel width in Hertz if not a Quantity. Defaults to None.
        Overwritten by the frequency channel width in the UVData object if
        `data_path` points to a pyuvdata-compatible file containing
        visibilities and `setup_data_vec` is True.
    jd_idx_min : int, optional
        Minimum time index to keep in the data vector. Defaults to None (keep
        all times).
    jd_min : :class:`astropy.Time` or float, optional
        Minimum time to keep in the data vector as a Julian date if not a Time.
        Defaults to None (keep all times).
    central_jd : :class:`astropy.Time` or float, optional
        Central time, as a Julian date if not a Time, around which `nt`
        times will be kept in the data vector. `nt` must also be passed,
        otherwise an error is raised. Defaults to None (keep all times).
    integration_time_seconds : :class:`astropy.Quantity` or float, optional
        Integration time in seconds of not a Quantity. Defaults to None.
        Overwritten by the integration time in the UVData object if `data_path`
        points to a pyuvdata-compatible file containing visibilities and
        `setup_data_vec` is True.
    form_pI : bool, optional
        Form pseudo-Stokes I visibilities. Otherwise, use the polarization
        specified by `pol`. Defaults to True.
    pI_norm : float, optional
        Normalization, ``N``, used in forming pseudo-Stokes I from XX and YY
        via ``pI = N * (XX + YY)``. Defaults to 1.0.
    pol : str, optional
        Case-insensitive polarization string. Used only if `form_pI` is False.
        Defaults to 'xx'.
    redundant_avg : bool, optional
        Redundantly average the data.  Defaults to False.
    uniform_redundancy : bool, optional
        Force the redundancy model to be uniform. Defaults to False.
    phase : bool, optional
        Create a "phasor vector" which can be used to phase each visibility
        in the data vector as a function of baseline, time, and frequency via
        element-wise multiplication. Defaults to False.
    phase_time : :class:`astropy.Time` or float, optional
        The time to which the visibilities will be phased as a Julian date if
        not a Time. If `phase` is True and `phase_time` is None, `phase_time`
        will be automatically set to the central time in the data. Defaults to
        None.
    calc_noise : bool, optional
        Calculate a noise estimate from the visibilities via differencing
        adjacent times per baseline and frequency. Defaults to False.
    save_vis : bool, optional
        Write visibility vector to disk in `save_dir`. If `calc_noise` is True,
        also save the noise vector. Defaults to False.
    save_model : bool, optional
        Write instrument model (antenna pairs, (u, v, w) sampling, and
        redundancy model) to disk in `save_dir`. If `phase` is True, also save
        the phasor vector. Defaults to False.
    save_dir : pathlib.Path or str, optional
        Output directory if `save_vis` or `save_model` is True. Defaults to
        ``Path(output_dir) / file_root``.
    clobber : bool, optional
        Clobber files on disk if they exist.  Defaults to False.
    sigma : float, optional
        Standard deviation of the visibility noise. Only used if
        `setup_data_vec` is True and `noise_data_path` is None. Defaults to
        None.
    noise_seed : int, optional
        Used to seed `np.random` when generating the noise vector. Defaults to
        742123.
    noise_data_path : pathlib.Path or str, optional
        Path to a numpy-compatible file containing a preprocessed noise vector.
        Defaults to None.
    inst_model : pathlib.Path or str, optional
        Path to directory containing instrument model files (`uvw_array.npy`,
        `redundancy_model.npy`, and optionally `phasor_vector.npy`). Defaults
        to None.
    save_k_vals : bool, optional
        Save k bin files (means, edges, and number of voxels in each bin).
        Defaults to True.
    telescope_name : str, optional
        Telescope identifier string. Defaults to ''.
    telescope_latlonalt : sequence of float, optional
        The latitude, longitude, and altitude of the telescope in degrees,
        degrees, and meters, respectively.
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
        Fit for the noise level. Defaults to False.
    use_LWM_Gaussian_prior : bool, optional
        Use a Gaussian prior (True) or a uniform prior (False) on the large
        spectral scale model. This option is currently not implemented bu
        will be reimplemented in the future. Defaults to False.
    use_EoR_cube : bool, optional
        Use internally simulated data generated from a EoR cube. This
        functionality is not currently supported but will be implemented in
        the future. Defaults to False.
    use_Multinest : bool, optional
        Use MultiNest sampler (True) or PolyChord (False). Support for
        PolyChord will be added in the future. Defaults to True.
    return_vis : bool, optional
        Return a dictionary with value (key) pairs of: the visibility vector
        ('vis'), the noise ('noise'), noisy visibilities if `noise_data_path`
        is None and `sigma` is not None ('vis_noisy'), a dictionary containing
        the array index mapping of conjugated baselines in the visibility 
        vector ('bl_conj_pairs_map'), the instrumentally sampled (u, v, w)
        ('uvws'), the number of redundant baselines averaged in each (u, v, w)
        ('redundancy'), the frequency channel width ('df'), the integration
        time ('dt'), and the phasor vector ('phasor') if `phase` is True.
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
    pspp : :class:`bayeseor.posterior.PowerSpectrumPosteriorProbability`
        Posterior probability class instance.
    vis_dict : dict
        Dictionary with value (key) pairs of: the visibility vector ('vis'),
        the noise ('noise'), noisy visibilities if `noise_data_path` is None
        and `sigma` is not None ('vis_noisy'), a dictionary containing the
        array index mapping of conjugated baselines in the visibility vector
        ('bl_conj_pairs_map'), the instrumentally sampled (u, v, w) ('uvws'),
        the number of redundant baselines averaged in each (u, v, w)
        ('redundancy'), the frequency channel width ('df'), the integration
        time ('dt'), and the phasor vector ('phasor') if `phase` is True.
        Returned only if `return_vis` is True.
    k_vals : numpy.ndarray
        Mean of each k bin. Returned only if `return_ks` is True.
    k_cube_voxels_in_bin : list
        List of sublists containing the flattened 3D k-space cube index of all
        |k| that fall within a given k bin. Returned only if `return_ks` is
        True.
    bm : :class:`bayeseor.matrices.BuildMatrices`
        Matrix building class instance. Returned only if `return_bm` is True.

    """
    # print_rank will only trigger print if verbose is True and rank == 0
    print_rank = 1 - (verbose and rank == 0)

    if use_LWM_Gaussian_prior:
        raise NotImplementedError(
            "use_LWM_Gaussian_prior is not currently implemented. It will be "
            "reimplemented in the future. For now, please set "
            "use_LWM_Gaussian_prior to False."
        )

    if sigma is None and noise_data_path is None:
        raise ValueError(
            "sigma cannot be None if setup_data_vec is True "
            "and noise_data_path is None"
        )

    if beta is not None:
        npl = len(beta)
    else:
        npl = 0
    
    # Setup output directory for sampler output
    if output_dir is None:
        raise ValueError("output_dir cannot be None")
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
            beta=beta,
            log_priors=log_priors,
            dimensionless_PS=dimensionless_PS,
            inverse_LW_power=inverse_LW_power,
            use_EoR_cube=use_EoR_cube,
            use_Multinest=use_Multinest,
            use_shg=use_shg,
            nu_sh=nu_sh,
            nv_sh=nv_sh,
            nq_sh=nq_sh,
            npl_sh=npl_sh,
            fit_for_shg_amps=fit_for_shg_amps,
            output_dir=output_dir,
        )

    output_dir /= file_root
    output_dir.mkdir(exist_ok=True, parents=True)

    if save_vis or save_model and save_dir is None:
        save_dir = output_dir

    mpiprint("\n", Panel("Data and Noise"), rank=print_rank)
    vis_dict = get_vis_data(
        data_path=data_path,
        ant_str=ant_str,
        bl_cutoff=bl_cutoff,
        freq_idx_min=freq_idx_min,
        nu_min_MHz=nu_min_MHz,
        freq_center=freq_center,
        nf=nf,
        channel_width_MHz=channel_width_MHz,
        jd_idx_min=jd_idx_min,
        jd_min=jd_min,
        central_jd=central_jd,
        nt=nt,
        integration_time_seconds=integration_time_seconds,
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
        save_vis=save_vis,
        save_model=save_model,
        save_dir=save_dir,
        clobber=clobber,
        noise_data_path=noise_data_path,
        inst_model=inst_model,
        verbose=verbose,
        rank=rank,
    )
    if "vis_noisy" in vis_dict:
        vis = vis_dict["vis"]
        vis_noisy = vis_dict["vis_noisy"]
    else:
        # Input visibilities contain noise
        vis = None
        vis_noisy = vis_dict["vis"]
    noise = vis_dict["noise"]
    bl_conj_pairs_map = vis_dict["bl_conj_pairs_map"]
    uvws = vis_dict["uvws"]
    redundancy = vis_dict["redundancy"]
    channel_width_MHz = vis_dict["df"]
    integration_time_seconds = vis_dict["dt"]
    if "phasor" in vis_dict:
        phasor = vis_dict["phasor"]
    else:
        phasor = None

    mpiprint("\n", Panel("Model k cube"), rank=print_rank)
    k_vals, k_cube_voxels_in_bin = build_k_cube(
        nu=nu,
        nv=nv,
        neta=neta,
        ps_box_size_ra_Mpc=ps_box_size_ra_Mpc,
        ps_box_size_dec_Mpc=ps_box_size_dec_Mpc,
        ps_box_size_para_Mpc=ps_box_size_para_Mpc,
        save_k_vals=save_k_vals,
        output_dir=output_dir,
        clobber=clobber,
        verbose=verbose,
        rank=rank
    )

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
        telescope_latlonalt=telescope_latlonalt,
        uvws=uvws,
        redundancy=redundancy,
        phasor=phasor,
        nt=nt,
        central_jd=central_jd,
        integration_time_seconds=integration_time_seconds,
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
        noise=noise,
        clobber=clobber,
        verbose=verbose,
        rank=rank
    )

    mpiprint("\n", Panel("Posterior"), rank=print_rank)
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

    mpiprint("\nInstantiating posterior class:", style="bold", rank=print_rank)
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

    return_vals = (pspp,)
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
        Standard deviation of the visibility noise.
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
            f"v{version_number}", f"v{next_version}"
        )
        version_number = next_version

    file_root += "/"

    return file_root

def get_vis_data(
    *,
    data_path : Path | str,
    ant_str : str = "cross",
    bl_cutoff : Quantity | float | None = None,
    freq_idx_min : int | None = None,
    nu_min_MHz : Quantity | float | None = None,
    freq_center : Quantity | float | None = None,
    nf : int | None = None,
    channel_width_MHz : Quantity | float | None = None,
    jd_idx_min : int | None = None,
    jd_min : Time | float | None = None,
    central_jd : Time | float | None = None,
    nt : int | None = None,
    integration_time_seconds : Quantity | float | None = None,
    form_pI : bool = True,
    pI_norm : float = 1.0,
    pol : str = "xx",
    redundant_avg : bool = False,
    uniform_redundancy : bool = False,
    phase : bool = False,
    phase_time : Time | float | None = None,
    calc_noise : bool = False,
    sigma : float,
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
        Path to either a pyuvdata-compatible file containing visibilities or
        a numpy-compatible file containing a preprocessed visibility vector.
        Defaults to None.
    ant_str : str, optional
        Antenna downselect string. This determines what baselines to keep in
        the data vector. Please see `pyuvdata.UVData.select` for more details.
        Defaults to 'cross' (cross-correlation baselines only).
    bl_cutoff : :class:`astropy.Quantity` or float, optional
        Baseline length cutoff in meters if not a Quantity. Defaults to None
        (keep all baselines).
    freq_idx_min : int, optional
        Minimum frequency channel index to keep in the data vector. Defaults
        to None (keep all frequencies).
    nu_min_MHz : :class:`astropy.Quantity` or float, optional
        Minimum frequency to keep in the data vector in Hertz if not a
        Quantity. All frequencies greater than or equal to `nu_min_MHz` will be
        kept, unless `nf` is specified. Defaults to None (keep all
        frequencies).
    freq_center : :class:`astropy.Quantity` or float, optional
        Central frequency, in Hertz if not a Quantity, around which `nf`
        frequencies will be kept in the data vector. `nf` must also be
        passed, otherwise an error is raised. Defaults to None (keep all
        frequencies).
    nf : int, optional
        Number of frequency channels. If `setup_data_vec` is True, sets the
        number of frequencies to keep starting from `freq_idx_min`, the
        channel corresponding to `nu_min_MHz`, or around `freq_center`. Defaults
        to None (keep all frequencies).
    channel_width_MHz : :class:`astropy.Quantity` or float, optional
        Frequency channel width in Hertz if not a Quantity. Defaults to None.
        Overwritten by the frequency channel width in the UVData object if
        `data_path` points to a pyuvdata-compatible file containing
        visibilities and `setup_data_vec` is True.
    jd_idx_min : int, optional
        Minimum time index to keep in the data vector. Defaults to None (keep
        all times).
    jd_min : :class:`astropy.Time` or float, optional
        Minimum time to keep in the data vector as a Julian date if not a Time.
        Defaults to None (keep all times).
    central_jd : :class:`astropy.Time` or float, optional
        Central time, as a Julian date if not a Time, around which `nt`
        times will be kept in the data vector. `nt` must also be passed,
        otherwise an error is raised. Defaults to None (keep all times).
    nt : int, optional
        Number of times. If `setup_data_vec` is True, sets the number of times
        to keep starting from `jd_idx_min`, the time corresponding to `jd_min`
        or around `central_jd`. Defaults to None (keep all times).
    integration_time_seconds : :class:`astropy.Quantity` or float, optional
        Integration time in seconds of not a Quantity. Defaults to None.
        Overwritten by the integration time in the UVData object if `data_path`
        points to a pyuvdata-compatible file containing visibilities and
        `setup_data_vec` is True.
    form_pI : bool, optional
        Form pseudo-Stokes I visibilities. Otherwise, use the polarization
        specified by `pol`. Defaults to True.
    pI_norm : float, optional
        Normalization, ``N``, used in forming pseudo-Stokes I from XX and YY
        visibilities via ``pI = N * (XX + YY)``. Defaults to 1.0.
    pol : str, optional
        Case-insensitive polarization string. Used only if `form_pI` is False.
        Defaults to 'xx'.
    redundant_avg : bool, optional
        Redundantly average the data.  Defaults to False.
    uniform_redundancy : bool, optional
        Force the redundancy model to be uniform. Defaults to False.
    phase : bool, optional
        Create a "phasor vector" which can be used to phase each visibility
        in the data vector as a function of baseline, time, and frequency via
        element-wise multiplication. Defaults to False.
    phase_time : :class:`astropy.Time` or float, optional
        The time to which the visibilities will be phased as a Julian date if
        not a Time. If `phase` is True and `phase_time` is None, `phase_time`
        will be automatically set to the central time in the data. Defaults to
        None.
    calc_noise : bool, optional
        Calculate a noise estimate from the visibilities via differencing
        adjacent times per baseline and frequency. Defaults to False.
    sigma : float
        Standard deviation of the visibility noise.
    noise_seed : int, optional
        Used to seed `np.random` when generating the noise vector. Defaults to
        742123.
    save_vis : bool, optional
        Write visibility vector to disk in `save_dir`. If `calc_noise` is True,
        also save the noise vector. Defaults to False.
    save_model : bool, optional
        Write instrument model (antenna pairs, (u, v, w) sampling, and
        redundancy model) to disk in `save_dir`. If `phase` is True, also save
        the phasor vector. Defaults to False.
    save_dir : pathlib.Path or str, optional
        Output directory if `save_vis` or `save_model` is True. Defaults to
        ``Path('./')``.
    clobber : bool, optional
        Clobber files on disk if they exist. Defaults to False.
    noise_data_path : pathlib.Path or str, optional
        Path to a numpy-compatible file containing a preprocessed noise vector.
        Defaults to None.
    inst_model : pathlib.Path or str, optional
        Path to directory containing instrument model files (`uvw_array.npy`,
        `redundancy_model.npy`, and optionally `phasor_vector.npy`). Defaults
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
        - vis: visibility vector with shape (nf*nbls*nt,)
        - noise: noise vector with shape (nf*nbls*nt,)
        - vis_noisy: visibility + noise vector
        - bl_conj_pairs_map: array index mapping of conjugate baseline pairs
        - uvws: instrumentally sampled (u, v, w) coordinates with shape
          (nt, nbls, 3)
        - redundancy: number of redundantly averaged baselines per (u, v, w)
          with shape (nt, nbls, 1)
        - df: frequency channel width in Hz
        - dt: integration time in seconds
        - phasor: optional phasor vector if `phased` is True with shape
          (nf*nbls*nt,)
        - tele_name: optional telescope name if `data_path` points to a
          pyuvdata-compatible file with a valid telescope name attribute
        - uvd: optional UVData object if `data_path` points to a
          pyuvdata-compatible file with a valid telescope name attribute

    """
    # print_rank will only trigger print if verbose is True and rank == 0
    print_rank = 1 - (verbose and rank == 0)

    if data_path is not None:
        if not isinstance(data_path, Path):
            data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"{data_path} does not exist")
        mpiprint(
            "\nUsing data at {}".format(data_path), rank=print_rank
        )
        if data_path.suffix == ".npy":
            vis = load_numpy_dict(data_path)

            if channel_width_MHz is None:
                raise ValueError(
                    "channel_width_MHz must be specified if loading a "
                    "preprocessed data vector (data_path has .npy suffix)"
                )
            if integration_time_seconds is None:
                raise ValueError(
                    "integration_time_seconds must be specified if "
                    "loading a preprocessed data vector (data_path has "
                    ".npy suffix)"
                )

            if inst_model is None:
                raise ValueError(
                    "inst_model cannot be None if loading a "
                    "preprocessed data vector (data_path has .npy suffix)"
                )
            else:
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
                uvws, redundancy, phasor = load_inst_model(inst_model)

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

        elif data_path.suffix in [".uvh5", ".uvfits", ".ms"]:
            vis, _, uvws, redundancy, phasor, noise, uvd = \
                preprocess_uvdata(
                    data_path,
                    ant_str=ant_str,
                    bl_cutoff=bl_cutoff,
                    freq_idx_min=freq_idx_min,
                    freq_min=nu_min_MHz*1e6,
                    freq_center=freq_center,
                    Nfreqs=nf,
                    jd_idx_min=jd_idx_min,
                    jd_min=jd_min,
                    jd_center=central_jd,
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
            try:
                # The future_array_shapes attribute is a legacy attribute
                # that has been removed as of pyuvdata 3.2, but this check
                # should remain for backwards compatibility.
                future_array_shapes = uvd.__getattribute__(
                    "_future_array_shapes"
                )
                future_array_shapes = future_array_shapes.value
            except:
                future_array_shapes = False
            freqs = uvd.freq_array
            if not future_array_shapes:
                freqs = freqs[0]
            channel_width_MHz = freqs[1] - freqs[0]  # Hz

            jds = Time(np.unique(uvd.time_array), format="jd")
            integration_time_seconds = (jds[1] - jds[0]).to("s").value

            tele_name = uvd.telescope_name

        if noise is not None:
            vis_noisy = vis + noise
        elif sigma is not None:
            mpiprint(
                f"Generating noise vector with std. dev. = {sigma:.2e}",
                rank=print_rank
            )
            vis_noisy, noise, bl_conj_pairs_map = \
                generate_data_and_noise_vector_instrumental(
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
            # Input visibilities contain noise
            vis_noisy = None

        vis_dict = {
            "vis": vis,
            "noise": noise,
            "bl_conj_pairs_map": bl_conj_pairs_map,
            "uvws": uvws,
            "redundancy": redundancy,
            "df": channel_width_MHz,
            "dt": integration_time_seconds,
        }
        if vis_noisy is not None:
            vis_dict["vis_noisy"] = vis_noisy
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
        save_k_vals=save_k_vals,
        k_vals_dir=output_dir,
        clobber=clobber,
        rank=print_rank
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
    integration_time_seconds : float | None = None,
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
        Minimum frequency in MHz.
    channel_width_MHz : float
        Frequency channel width in MHz.
    nq : int, optional
        Number of large spectral scale model quadratic basis vectors. Defaults
        to 0.
    beta : list of float, optional
        Brightness temperature power law spectral index/indices used in the
        large spectral scale model. Defaults to None.
    sigma : float
        Standard deviation of the visibility noise.
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
    integration_time_seconds : float, optional
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
        Force the beam to be achromatic. Used only if
        `include_instrumental_effects` is True. Defaults to False.
    beam_peak_amplitude : float, optional
        Peak amplitude of the beam. Used only if `include_instrumental_effects`
        is True. Defaults to 1.0.
    fwhm_deg : float, optional
        Full width at half maximum of beam in degrees. Used only if `beam_type`
        is 'airy', 'gaussian', or 'gausscosine'. Defaults to None.
    antenna_diameter : float, optional
        Antenna (aperture) diameter in meters. Used only if `beam_type` is
        'airy', 'gaussian', or 'gausscosine'. Defaults to None.
    cosfreq : float, optional
        Cosine frequency if using a 'gausscosine' beam. Defaults to None.
    beam_ref_freq : float, optional
        Beam reference frequency in MHz. Used only if
        `include_instrumental_effects` is True. Defaults to None.
    noise_data_path : pathlib.Path or str, optional
        Path to a numpy-compatible file containing a preprocessed noise vector.
        Defaults to None.
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
        inst_str += f"nt{nt}-dt{integration_time_seconds:.2f}s"
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
    telescope_latlonalt : Sequence[float] = (0, 0, 0),
    nt : int,
    central_jd : float,
    integration_time_seconds : float,
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
        Minimum frequency in MHz.
    channel_width_MHz : float
        Frequency channel width in MHz.
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
        Standard deviation of the visibility noise.
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
        altitude in meters). Defaults to (0, 0, 0).
    nt : float
        Number of times.
    central_jd : float
        Central time as a Julian date.
    integration_time_seconds : float
        Integration time in seconds.
    beam_type : str
        Path to a pyuvdata-compatible beam file or one of 'uniform',
        'gaussian', 'airy', 'gausscosine', or 'taperairy'.
    beam_center : list of float, optional
        Beam center offsets from the phase center in right ascension and
        declination in degrees.
    achromatic_beam : bool, optional
        Force the beam to be achromatic. Defaults to False.
    beam_peak_amplitude : float
        Peak amplitude of the beam. Defaults to 1.0.
    fwhm_deg : float, optional
        Full width at half maximum of beam in degrees. Used only if `beam_type`
        is 'airy', 'gaussian', or 'gausscosine'. Defaults to None.
    antenna_diameter : float, optional
        Antenna (aperture) diameter in meters. Used only if `beam_type` is
        'airy', 'gaussian', or 'gausscosine'. Defaults to None.
    cosfreq : float, optional
        Cosine frequency if using a 'gausscosine' beam. Defaults to None.
    beam_ref_freq : float, optional
        Beam reference frequency in MHz. Defaults to None.
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
        Path to a numpy-compatible file containing a preprocessed noise vector.
        Defaults to None.
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
    bm : :class:`bayeseor.matrices.build.BuildMatrices`
        BuildMatrices class instance used to construct the matrix stack.

    """
    if noise is not None:
        warnings.warn(
            "There is a known issue when `noise` is not None (please see "
            "BayesEoR issue #55 for more details.  For now, `noise` will "
            "be forced to None."
        )
        noise = None

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
        integration_time_seconds=integration_time_seconds,
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
        f"[bold]Array save directory:[/bold] {array_dir}", rank=print_rank
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
        central_jd=central_jd,
        dt=integration_time_seconds,
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
        bl_red_array_vec=redundancy.reshape(-1, 1).flatten(),  # TODO: remove
        n_vis=n_vis,  # TODO: remove
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
                "\nWARNING: Overwriting matrix stack\n", rank=print_rank,
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
    pspp : :class:`bayeseor.posterior.PowerSpectrumPosteriorProbability`
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

    mpiprint("\n", Panel("Posterior"), rank=print_rank)
    if use_LWM_Gaussian_prior:
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
    else:
        if use_intrinsic_noise_fitting:
            priors[0] = [1.0, 2.0]  # Linear alpha_prime range
    mpiprint("priors = {}".format(priors), rank=print_rank)
    
    mpiprint(
        "\nInstantiating posterior class:", style="bold", rank=print_rank
    )
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
