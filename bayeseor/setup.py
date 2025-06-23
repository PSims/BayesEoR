""" Convenience functions for setting up a BayesEoR analysis. """

import numpy as np
from astropy.time import Time
from pathlib import Path

from .vis import load_numpy_dict, preprocess_uvdata
from .model.noise import generate_data_and_noise_vector_instrumental
from .model.k_cube import (
    generate_k_cube_in_physical_coordinates,
    mask_k_cube,
    generate_k_cube_model_spherical_binning,
    calc_mean_binned_k_vals
)

def run_setup(
    nf,
    neta,
    nq,
    nt,
    nu,
    nv,
    nu_fg,
    nv_fg,
    ps_box_size_ra_Mpc,
    ps_box_size_dec_Mpc,
    ps_box_size_para_Mpc,
    beta=[2.63, 2.82],
    full=False,
    preproc_data=False,
    data_path=None,
    ant_str="cross",
    bl_cutoff=None,
    freq_idx_min=None,
    freq_min=None,
    freq_center=None,
    df=None,
    jd_idx_min=None,
    jd_min=None,
    jd_center=None,
    form_pI=True,
    pI_norm=1.0,
    pol="xx",
    redundant_avg=True,
    uniform_redundancy=False,
    phase=False,
    phase_time=False,
    calc_noise=False,
    save_vis=False,
    save_model=False,
    save_dir="./",
    clobber=False,
    sigma=None,
    random_seed=None,
    noise_path=None,
    inst_model_dir=None,
    build_k_cube=False,
    save_k_vals=False,
    build_matrices=False,
    make_output_dir=False,
    output_dir="./",
    file_root=None,
    log_priors=False,
    dimensionless_PS=True,
    inverse_LW_power=1e-16,
    use_EoR_cube=False,
    use_Multinest=True,
    use_shg=False,
    nu_sh=None,
    nv_sh=None,
    nq_sh=None,
    npl_sh=None,
    fit_for_shg_amps=False,
    verbose=False,
    rank=0
):
    """
    Run setup steps.

    Parameters
    ----------
    nf : int
        Number of frequency channels. If `preproc_data` is True, sets the
        number of frequencies to keep starting from `freq_idx_min`, the
        channel corresponding to `freq_min`, or around `freq_center`. Defaults
        to None (keep all frequencies).
    neta : int
        Number of Line of Sight (LoS, frequency axis) Fourier modes.
    nq : int
        Number of large spectral scale model basis vectors. If `beta` is None,
        the basis vectors are quadratic in frequency. If `beta` is not None,
        the basis vectors are power laws with brightness temperature spectral
        indices from `beta`.
    nt : int
        Number of times. If `preproc_data` is True, sets the number of times
        to keep starting from `jd_idx_min`, the time corresponding to `jd_min`
        or around `jd_center`. Defaults to None (keep all times).
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
    ps_box_size_ra_Mpc : float
        Right Ascension (RA) axis extent of the cosmological volume in Mpc from
        which the power spectrum is estimated.
    ps_box_size_dec_Mpc : float
        Declination (DEC) axis extent of the cosmological volume in Mpc from
        which the power spectrum is estimated.
    ps_box_size_para_Mpc : float
        LoS extent of the cosmological volume in Mpc from which the power
        spectrum is estimated.
    beta : list of float, optional
        Brightness temperature power law spectral index/indices used in the
        large spectral scale model. Can be a single spectral index, e.g.
        [2.63], or multiple spectral indices, e.g. [2.63, 2.82]. Defaults to
        [2.63, 2.82].
    full : bool, optional
        Run the full setup.  Sets `preproc_data`, `build_k_cube`,
        `build_matrices`, and `make_output_dir` to True.  Defaults to False.
    preproc_data : bool, optional
        Load and/or preprocess visibility data. The instrument model will be
        loaded or created as part of this process. Noise visibilities will also
        be generated if noise is to be added to the data. Defaults to False.
    data_path : :class:`pathlib.Path` or str, optional
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
    freq_min : :class:`astropy.Quantity` or float, optional
        Minimum frequency to keep in the data vector in Hertz if not a
        Quantity. All frequencies greater than or equal to `freq_min` will be
        kept, unless `nf` is specified. Defaults to None (keep all
        frequencies).
    freq_center : :class:`astropy.Quantity` or float, optional
        Central frequency, in Hertz if not a Quantity, around which `nf`
        frequencies will be kept in the data vector. `nf` must also be
        passed, otherwise an error is raised. Defaults to None (keep all
        frequencies).
    df : :class:`astropy.Quantity` or float, optional
        Frequency channel width in Hertz if not a Quantity. Defaults to None.
        Overwritten by the frequency channel width in the UVData object if
        `data_path` points to a pyuvdata-compatible file containing
        visibilities and `preproc_data` is True.
    jd_idx_min : int, optional
        Minimum time index to keep in the data vector. Defaults to None (keep
        all times).
    jd_min : :class:`astropy.Time` or float, optional
        Minimum time to keep in the data vector as a Julian date if not a Time.
        Defaults to None (keep all times).
    jd_center : :class:`astropy.Time` or float, optional
        Central time, as a Julian date if not a Time, around which `nt`
        times will be kept in the data vector. `nt` must also be passed,
        otherwise an error is raised. Defaults to None (keep all times).
    dt : :class:`astropy.Quantity` or float, optional
        Integration time in seconds of not a Quantity. Defaults to None.
        Overwritten by the integration time in the UVData object if `data_path`
        points to a pyuvdata-compatible file containing visibilities and
        `preproc_data` is True.
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
        Redundantly average the data.  Defaults to True.
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
    save_dir : :class:`pathlib.Path` or str, optional
        Output directory if `save_vis` or `save_model` is True. Defaults to
        './'.
    clobber : bool, optional
        Clobber files on disk if they exist.  Defaults to False.
    sigma : float, optional
        Standard deviation of the visibility noise. Only used if `preproc_data`
        is True and `noise_path` is None. Defaults to None.
    random_seed : int, optional
        Used to seed `np.random` when generating the noise vector. Defaults to
        None.
    noise_path : :class:`pathlib.Path` or str, optional
        Path to a numpy-compatible file containing a preprocessed noise vector.
        Defaults to None.
    inst_model_dir : :class:`pathlib.Path` or str, optional
        Path to directory containing instrument model files (`uvw_array.npy`,
        `redundancy_model.npy`, and optionally `phasor_vector.npy`). Defaults
        to False.
    build_k_cube : bool, optional
        Build the model k cube. Defaults to False.
    save_k_vals : bool, optional
        Save k bin files (means, edges, and number of voxels in each bin).
        Defaults to False.
    build_matrices : bool, optional
        Build and/or load the matrix stack. Defaults to False.
    make_output_dir : bool, optional
        Make the output directory for sampler output.
    output_dir : :class:`pathlib.Path` or str, optional
        Parent directory for sampler output. Defaults to './'.
    file_root : str, optional
        Sampler output directory name. If None (default), start a new analysis.
        Otherwise, resume analysis from `file_root`.
    log_priors : bool, optional
        Assume priors on power spectrum coefficients are in log_10 units.
        Defaults to False.
    dimensionless_PS : bool, optional
        Fit for the dimensionless power spectrum, :math:`\\Delta^2(k)` (True,
        default), or the power spectrum, :math:`P(k)` (False).
    inverse_LW_power : float, optional
        Prior on the inverse power of the large spectral scale model
        coefficients.
    use_EoR_cube : bool, optional
        Use internally simulated data generated from a EoR cube. This
        functionality is not currently supported but will be implemented in
        the future.
    use_Multinest : bool, optional
        Use MultiNest sampler (True, default) or Polychord (False).
    use_shg : bool, optional
        Use the subharmonic grid. Defaults to False.
    nu_sh : int, optional
        Number of pixels on a side for the u-axis in the subharmonic grid
        model uv-plane. Defaults to None.
    nv_sh : int, optional
        Number of pixels on a side for the v-axis in the subharmonic grid
        model uv-plane. Defaults to None.
    nq_sh : int, optional
        Number of large spectral scale model quadratic basis vectors for the
        subharmonic grid. Defaults to None.
    npl_sh : int, optional
        Number of large spectral scale model power law basis vectors for the
        subharmonic grid. Defaults to None.
    fit_for_shg_amps : bool, optional
        Fit for the amplitudes of the subharmonic grid pixels. Defaults to
        False.
    verbose : bool, optional
        Print statements useful for debugging. Defaults to False.
    rank : int, optional
        MPI rank. Defaults to 0.

    """
    
    """
    Required steps:
    1.  [vis.read_data] Read visibility data (load directly via pyuvdata)
    2.  [..model.instrument] Load instrument model
    3.  [..utils.?] Create output directory
    4.  [..matrices.build_matrices] Build matrices
    5.  [..model.k_cube] Create model k cube
    #.  [driver?] Load T and T_Ninv_T (used where? why loaded here?)
    6.  [vis.mock_data_*] Create mock data (requires matrix stack for Finv)
    7.  [..model.noise] Add synthetic noise to data (optional)
    #.  [?] Apply taper matrix (leave this in?  it's not documented as working)
    8.  [vis.?] Hermitian data checks
    9.  [driver?] Calculate dbar and d_Ninv_d (why here? requires matrix stack)
    10. [..posterior] Set up priors
    11. [..posterior] Create posterior class (change name, too long)
    12. [driver?] Estimate posterior calculation time
    13. [DELETE] Write MAP dict
    14. [..run?] Define likelihood function for MultiNest/PolyChord
    15. [..run] Run analysis
    """
    if full:
        preproc_data = True
        build_k_cube = True
        build_matrices = True
        make_output_dir = True
    
    if build_k_cube and save_k_vals and not make_output_dir:
        make_output_dir = True

    if sigma is None and preproc_data and noise_path is None:
        raise ValueError(
            "sigma cannot be None if preproc_data is True "
            "and noise_path is None"
        )

    if beta is not None:
        npl = len(beta)
    else:
        npl = 0
    
    if preproc_data:
        if data_path is not None:
            if not isinstance(data_path, Path):
                data_path = Path(data_path)
            if not data_path.exists():
                raise FileNotFoundError(f"{data_path} does not exist")
            if data_path.suffix == ".npy":
                vis = load_numpy_dict(data_path)

                if inst_model_dir is None:
                    raise ValueError(
                        "inst_model_dir cannot be None if loading a "
                        "preprocessed data vector (data_path has .npy suffix)"
                    )
                else:
                    if not isinstance(inst_model_dir, Path):
                        inst_model_dir = Path(inst_model_dir)
                    required_files = ["uvw_model.npy", "redundancy_model.npy"]
                    dir_files = inst_model_dir.glob("*.npy")
                    if not np.all([f in dir_files for f in required_files]):
                        raise ValueError(
                            "inst_model_dir must point to a directory with "
                            "uvw_model.npy and redundancy_model.npy"
                        )
                    uvws = load_numpy_dict(
                        inst_model_dir / "uvw_model.npy"
                    )
                    redundancy = load_numpy_dict(
                        inst_model_dir / "redundancy_model.npy"
                    )
                    if phase:
                        phasor = load_numpy_dict(
                            inst_model_dir / "phasor_vector.npy"
                        )

                if noise_path is not None:
                    if not isinstance(noise_path, Path):
                        noise_path = Path(noise_path)
                    if not noise_path.exists():
                        raise FileNotFoundError(f"{noise_path} does not exist")
                    noise = load_numpy_dict(noise_path)
                    vis_noisy = vis + noise

            elif data_path.suffix in [".uvh5", ".uvfits", ".ms"]:
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
                try:
                    # The future_array_shapes attribute is a legacy attribute that has
                    # been removed as of pyuvdata 3.2, but this check should remain for
                    # backwards compatibility.
                    future_array_shapes = uvd.__getattribute__(
                        "_future_array_shapes"
                    )
                    future_array_shapes = future_array_shapes.value
                except:
                    future_array_shapes = False
                freqs = uvd.freq_array
                if not future_array_shapes:
                    freqs = freqs[0]
                df = freqs[1] - freqs[0]  # Hz

                jds = Time(np.unique(uvd.time_array), format="jd")
                dt = (jds[1] - jds[0]).to("s").value

                if noise is not None:
                    vis_noisy = vis + noise
                else:
                    vis_noisy, noise, bl_conj_pairs_map = \
                        generate_data_and_noise_vector_instrumental(
                            sigma,
                            vis,
                            nf,
                            nt,
                            uvws,
                            redundancy,
                            random_seed=random_seed,
                            rank=rank
                        )
        else:
            # FIXME
            raise NotImplementedError(
                "data_path must not be None. Internal simulated data will be "
                "added in the future."
            )

    if make_output_dir:
        if output_dir is None:
            raise ValueError(
                "If make_output_dir is True, output_dir cannot be None"
            )
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    if file_root is None:
        file_root = generate_file_root(
            output_dir,
            nu,
            nv,
            neta,
            nq,
            npl,
            sigma,
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
            fit_for_shg_amps=fit_for_shg_amps
        )

    if build_k_cube:
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
            rank=rank
        )


def generate_file_root(
    output_dir,
    nu,
    nv,
    neta,
    nq,
    npl,
    sigma,
    beta=None,
    log_priors=False,
    dimensionless_PS=True,
    inverse_LW_power=1e-16,
    use_EoR_cube=False,
    use_Multinest=True,
    use_shg=False,
    nu_sh=None,
    nv_sh=None,
    nq_sh=None,
    npl_sh=None,
    fit_for_shg_amps=False
):
    """
    Generate the directory name for the sampler outputs.

    Parameters
    ----------
    output_dir : :class:`pathlib.Path` or str
        Parent directory for sampler output.
    nu : int
        Number of pixels on a side for the u-axis in the EoR model uv-plane.
    nv : int
        Number of pixels on a side for the v-axis in the EoR model uv-plane.
    neta : int
        Number of Line of Sight (LoS, frequency axis) Fourier modes.
    nq : int
        Number of large spectral scale model quadratic basis vectors.
    npl : int
        Number of large spectral scale model power law basis vectors.
    sigma : float
        Standard deviation of the visibility noise.
    beta : list of float, optional
        Brightness temperature power law spectral index/indices used in the
        large spectral scale model. Defaults to None.
    log_priors : bool, optional
        Assume priors on power spectrum coefficients are in log_10 units.
        Defaults to False.
    dimensionless_PS : bool, optional
        Fit for the dimensionless power spectrum, :math:`\\Delta^2(k)` (True,
        default), or the power spectrum, :math:`P(k)` (False).
    inverse_LW_power : float, optional
        Prior on the inverse power of the large spectral scale model
        coefficients.
    use_EoR_cube : bool, optional
        Use internally simulated data generated from a EoR cube. This
        functionality is not currently supported but will be implemented in
        the future.
    use_Multinest : bool, optional
        Use MultiNest sampler (True, default) or Polychord (False).
    use_shg : bool, optional
        Use the subharmonic grid. Defaults to False.
    nu_sh : int, optional
        Number of pixels on a side for the u-axis in the subharmonic grid
        model uv-plane. Defaults to None.
    nv_sh : int, optional
        Number of pixels on a side for the v-axis in the subharmonic grid
        model uv-plane. Defaults to None.
    nq_sh : int, optional
        Number of large spectral scale model quadratic basis vectors for the
        subharmonic grid. Defaults to None.
    npl_sh : int, optional
        Number of large spectral scale model power law basis vectors for the
        subharmonic grid. Defaults to None.
    fit_for_shg_amps : bool, optional
        Fit for the amplitudes of the subharmonic grid pixels. Defaults to
        False.

    Returns
    -------
    file_root : str
        Sampler output directory name.

    """
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    file_root = f"Test-{nu}-{nv}-{neta}-{nq}-{npl}"
    file_root += f"-{sigma:.1E}"
    if beta:
        beta_str = ""
        for b in beta:
            beta_str += f"-{b:.2f}"
        file_root += beta_str
    if log_priors:
        file_root += "-lp"
    if dimensionless_PS:
        file_root += "-dPS"
    if nq == 0:
        file_root = file_root.replace("mini-", "mini-NQ-")
    elif inverse_LW_power >= 1e16:
        file_root = file_root.replace("mini-", "mini-ZLWM-")
    if use_EoR_cube:
        file_root = file_root.replace("Test", "EoR")
    if use_Multinest:
        file_root = "MN-" + file_root
    else:
        file_root = "PC-" + file_root
    if use_shg:
        file_root += (
            f"-SH-{nu_sh}-{nv_sh}-{nq_sh}-{npl_sh}"
        )
        if fit_for_shg_amps:
            file_root += "ffsa-"
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

def get_array_dir_name(
    nu=None,
    nv=None,
    nu_fg=None,
    nv_fg=None,
    neta=None,
    fit_for_monopole=False,
    use_shg=False,
    nu_sh=None,
    nv_sh=None,
    nq_sh=None,
    npl_sh=None,
    fit_for_shg_amps=False,
    freq_min=None,
    df=None,
    nq=None,
    npl=None,
    beta=None,
    sigma=None,
    nside=None,
    fov_ra_eor=None,
    fov_dec_eor=None,
    fov_ra_fg=None,
    fov_dec_fg=None,
    simple_za_filter=False,
    beam_center=None,
    drift_scan=True,
    taper_func=None,
    include_instrumental_effects=True,
    beam_type=None,
    achromatic_beam=False,
    beam_peak_amplitude=None,
    fwhm_deg=None,
    antenna_diameter=None,
    cosfreq=None,
    beam_ref_freq=None,
    inst_model=None,
    noise_data_path=None,
    prefix=Path("./matrices/")
):
    """
    Generate the output path for BayesEoR matrices based on analysis params.

    This function constructs two strings which form two subdirectories:
      1. The `analysis_dir` string contains all the unique analysis/model
         parameters, e.g. the field(s) of view, nu, nv, neta, etc.
      2. The `inst_dir` string contains all of the instrument model specific
         parameters, e.g. the instrument model filename, beam type, beam
         center, integration time, etc.
    The final array save directory is produced via
    ```
    matrices_path = Path(prefix) / analysis_dir / inst_dir
    ```

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
    fit_for_monopole : bool
        Fit for the (u, v) = (0, 0) pixel in the model uv-plane.
    use_shg : bool
        Use the subharmonic grid.
    nu_sh : int
        Number of pixels on a side for the u-axis in the subharmonic grid
        model uv-plane. Defaults to None.
    nv_sh : int
        Number of pixels on a side for the v-axis in the subharmonic grid
        model uv-plane. Defaults to None.
    nq_sh : int
        Number of large spectral scale model quadratic basis vectors for the
        subharmonic grid. Defaults to None.
    npl_sh : int
        Number of large spectral scale model power law basis vectors for the
        subharmonic grid. Defaults to None.
    fit_for_shg_amps : bool
        Fit for the amplitudes of the subharmonic grid pixels. Defaults to
        False.
    freq_min : float
        Minimum frequency in MHz.
    df : float
        Frequency channel width in MHz.
    nq : int
        Number of large spectral scale model quadratic basis vectors.
    npl : int
        Number of large spectral scale model power law basis vectors.
    beta : list of float
        Brightness temperature power law spectral index/indices used in the
        large spectral scale model.
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
    simple_za_filter : bool
        Filter pixels in teh sky model by zenith angle only.
    beam_center : list of float
        Beam center offsets from the phase center in right ascension and
        declination in degrees.
    drift_scan : bool
        Model drift scan (True) or phased (False) visibilities.
    taper_func : str
        Taper function applied to the frequency axis of the visibilities.
    include_instrumental_effects : bool
        Forward model an instrument.
    beam_type : str
        Path to a pyuvdata-compatible beam file or one of 'uniform',
        'gaussian', 'airy', 'gausscosine', or 'taperairy'.
    achromatic_beam : bool
        Force the beam to be achromatic.
    beam_peak_amplitude : float
        Peak amplitude of the beam.
    fwhm_deg : float
        Full width at half maximum of beam in degrees.
    antenna_diameter : float
        Antenna (aperture) diameter in meters.
    cosfreq : float
        Cosine frequency if using a 'gausscosine' beam.
    beam_ref_freq : float
        Beam reference frequency in MHz.
    inst_model : :class:`pathlib.Path` or str
        To be replaced by telescope_name (see # FIXME).
    noise_data_path : :class:`pathlib.Path` or str
        Path to a numpy-compatible file containing a preprocessed noise vector.
    prefix : :class:`pathlib.Path` or str, optional
        Array directory prefix.  Defaults to './matrices/'.

    Returns
    -------
    matrices_path : Path
        Path containing the uniquely identifying info for each analysis, i.e.
        model parameters and the instrument model.

    """
    if prefix is not None:
        if not isinstance(prefix, Path):
            prefix = Path(prefix)
        matrices_path = prefix
    else:
        matrices_path = Path("./")

    # Root matrix dir
    analysis_dir = (
        f"nu-{nu}-nv-{nv}-neta-{neta}"
        + f"-fmin-{freq_min:.2f}MHz"
        + f"-df-{df*1e3:.2f}kHz"
        + f"-sigma-{sigma:.2E}-nside-{nside}"
    )

    fovs_match = (
        fov_ra_eor == fov_ra_fg
        and fov_dec_eor == fov_dec_fg
    )
    fov_str = "-fov-deg"
    if not fovs_match:
        fov_str += "-eor"
    if not fov_ra_eor == fov_dec_eor and not simple_za_filter:
        fov_str += f"-ra-{fov_ra_eor:.1f}-dec-{fov_dec_eor:.1f}"
    else:
        fov_str += f"-{fov_ra_eor:.1f}"
    if not fovs_match:
        fov_str += "-fg"
        if fov_ra_fg != fov_dec_fg and not simple_za_filter:
            fov_str += f"-ra-{fov_ra_fg:.1f}-dec-{fov_dec_fg:.1f}"
        else:
            fov_str += f"-{fov_ra_fg:.1f}"
    if simple_za_filter:
        fov_str += "-za-filter"
    analysis_dir += fov_str

    nu_nv_match = (
        nu == nu_fg and nv == nv_fg
    )
    if not nu_nv_match:
        analysis_dir += f"-nufg-{nu_fg}-nvfg-{nv_fg}"
    
    analysis_dir += f"-nq-{nq}"
    if nq > 0:
        if npl == 1:
            analysis_dir += f"-beta-{beta[0]:.2f}"
        else:
            for i in range(npl):
                analysis_dir += f"-b{i+1}-{beta[i]:.2f}"
    if fit_for_monopole:
        analysis_dir += "-ffm"

    if use_shg:
        shg_str = "-shg"
        if nu_sh > 0:
            shg_str += f"-nush-{nu_sh}"
        if nv_sh > 0:
            shg_str += f"-nvsh-{nv_sh}"
        if nq_sh > 0:
            shg_str += f"-nqsh-{nq_sh}"
        if npl_sh > 0:
            shg_str += f"-nplsh-{npl_sh}"
        if fit_for_shg_amps:
            shg_str += "-ffsa"
        analysis_dir += shg_str
    
    if beam_center is not None:
        beam_center_signs = [
            "+" if beam_center[i] >= 0 else "" for i in range(2)
        ]
        beam_center_str = "-beam-center-RA0{}{:.2f}-DEC0{}{:.2f}".format(
                beam_center_signs[0],
                beam_center[0],
                beam_center_signs[1],
                beam_center[1]
        )
        analysis_dir += beam_center_str
    
    if not drift_scan:
        analysis_dir += "-phased"
    
    if taper_func:
        analysis_dir += f"-{taper_func}"
    
    matrices_path /= analysis_dir
    
    if include_instrumental_effects:
        beam_info_str = ""
        if not "." in beam_type:
            beam_info_str = f"{beam_type}-beam"
            if achromatic_beam:
                beam_info_str = "achromatic-" + beam_info_str
            if (not beam_peak_amplitude == 1
                and beam_type in ["uniform", "gaussian", "gausscosine"]):
                beam_info_str += f"-peak-amp-{beam_peak_amplitude}"
            
            if beam_type in ["gaussian", "gausscosine"]:
                if fwhm_deg is not None:
                    beam_info_str += f"-fwhm-{fwhm_deg:.4f}deg"
                elif antenna_diameter is not None:
                    beam_info_str += (
                        f"-antenna-diameter-{antenna_diameter}m"
                    )
                if beam_type == "gausscosine":
                    beam_info_str += f"-cosfreq-{cosfreq:.2f}wls"
            elif beam_type in ["airy", "taperairy"]:
                beam_info_str += f"-antenna-diameter-{antenna_diameter}m"
                if beam_type == "taperairy":
                    beam_info_str += f"-fwhm-{fwhm_deg}deg"
            if achromatic_beam:
                beam_info_str += f"-ref-freq-{beam_ref_freq:.2f}MHz"
        else:
            beam_info_str = Path(beam_type).stem

        # FIXME: inst_model isn't useful anymore, could potentially add a new
        # kwarg like e.g. telescope_name.
        inst_dir = "-".join((Path(inst_model).name, beam_info_str))
        if drift_scan:
            inst_dir += "-dspb"
        if noise_data_path is not None:
            inst_dir += "-noise-vec"

        matrices_path /= inst_dir
    
    matrices_path.mkdir(exist_ok=True, parents=True)

    return str(matrices_path) + "/", fov_str
