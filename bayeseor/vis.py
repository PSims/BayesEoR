import numpy as np
from astropy import units
from astropy.units import Quantity
from astropy.time import Time
from pathlib import Path
from pyuvdata import UVData
from pyuvdata.utils import polstr2num
import warnings

from .model.healpix import Healpix
from .utils import mpiprint


def preprocess_uvdata(
    fp,
    ant_str="cross",
    bl_cutoff=None,
    freq_idx_min=None,
    freq_min=None,
    freq_center=None,
    Nfreqs=None,
    jd_idx_min=None,
    jd_min=None,
    jd_center=None,
    Ntimes=None,
    form_pI=True,
    pI_norm=1.0,
    pol="xx",
    redundant_avg=True,
    uniform_redundancy=False,
    phase=False,
    phase_time=False,
    calc_noise=False,
    return_uvd=False,
    save_vis=False,
    save_model=False,
    save_dir="./",
    clobber=False,
    verbose=False,
    rank=0
):
    """
    Read visibility data from disk and form a one-dimensional data vector.

    This function loads a `pyuvdata`-compatible file containing visibilities
    for a set of baselines (Nbls), times (Ntimes), and frequencies (Nfreqs)
    and forms a one-dimensional data vector with shape
    (2*Nbls * Ntimes * Nfreqs,) and an accompanying instrument model of
    sampled (u, v, w) coordinates and a redundancy model (when averaging
    redundant baselines). The factor of 2 in the data vector shape comes from
    the fact that the input visibility vector must be Hermitian, so we copy
    all baselines at (u, v) to (-u, -v) and conjugate the data accordingly.
    This is captured in the instrument model as the instrument model also
    contains all (u, v) and conjugated (-u, -v).

    Optionally, if `phase` and/or `calc_noise` is True, a phasor vector and/or
    noise vector is also produced. The phasor vector is calculated as the 
    phase required to rotate 1+0j to the corresponding location of the phased
    (u, v, w) coordinate for a given baseline per frequency and time.  The
    noise is estimated by differencing visibilities at even and odd time steps
    per baseline and frequency.

    Parameters
    ----------
    fp : Path or str
        Path to pyuvdata-compatible file containing visibilities.
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
        kept, unless `Nfreqs` is specified. Defaults to None (keep all
        frequencies).
    freq_center : :class:`astropy.Quantity` or float, optional
        Central frequency, in Hertz if not a Quantity, around which `Nfreqs`
        frequencies will be kept in the data vector. `Nfreqs` must also be
        passed, otherwise an error is raised. Defaults to None (keep all
        frequencies).
    Nfreqs : int, optional
        Number of frequencies to keep starting from `freq_idx_min`, the
        channel corresponding to `freq_min`, or around `freq_center`. Defaults
        to None (keep all frequencies).
    jd_idx_min : int, optional
        Minimum time index to keep in the data vector. Defaults to None (keep
        all times).
    jd_min : :class:`astropy.Time` or float, optional
        Minimum time to keep in the data vector as a Julian date if not a Time.
        Defaults to None (keep all times).
    jd_center : :class:`astropy.Time` or float, optional
        Central time, as a Julian date if not a Time, around which `Ntimes`
        times will be kept in the data vector. `Ntimes` must also be passed,
        otherwise an error is raised. Defaults to None (keep all times).
    Ntimes : int, optional
        Number of times to keep starting from `jd_idx_min`, the time
        corresponding to `jd_min` or around `jd_center`. Defaults to None (keep
        all times).
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
    return_uvd : bool, optional
        Return the preprocessed UVData object. Defaults to False.
    save_vis : bool, optional
        Write visibility vector to disk in `save_dir`. If `calc_noise` is True,
        also save the noise vector. Defaults to False.
    save_model : bool, optional
        Write instrument model (antenna pairs, (u, v, w) sampling, and
        redundancy model) to disk in `save_dir`. If `phase` is True, also save
        the phasor vector. Defaults to False.
    save_dir : Path or str, optional
        Output directory if `save_vis` or `save_model` is True. Defaults to
        './'.
    clobber : bool, optional
        Clobber files on disk if they exist.  Defaults to False.
    verbose : bool, optional
        Print statements useful for debugging. Defaults to False.
    rank : int, optional
        MPI rank. Defaults to 0.

    Returns
    -------
    vis : :class:`numpy.ndarray`
        Visibility vector with shape (2*Nbls*Nfreqs*Ntimes,).
    antpairs : list of tuple
        List of antenna pair tuples corresponding to the (u, v, w) coordinates
        in `uvws`, in identical order, with length (2*Nbls).
    uvws : :class:`numpy.ndarray`
        Sampled (u, v, w) coordinates with shape (Ntimes, 2*Nbls, 3). The
        ordering of the Nbls axis matches the ordering of the baselines in
        `vis_vec`.
    redundancy : :class:`numpy.ndarray`
        Redundancy model containing the number of baselines averaged within
        a redundant baseline group.
    phasor : :class:`numpy.ndarray`
        Phasor vector which can be multiplied element-wise into `vis` to form
        phased visibilities. Returned only if `phase` is True.
    noise : :class:`numpy.ndarray`
        Estimated noise vector with shape (2*Nbls*Nfreqs*Ntimes,). Returned
        only if `calc_noise` is True.
    uvd : :class:`pyuvdata.UVData`
        Preprocessed UVData object.  Returned only if `return_uvd` is True.

    """
    if not isinstance(fp, Path):
        fp = Path(fp)
    if not fp.exists():
        raise FileNotFoundError(f"No such file or directory: '{fp}'")
    
    if save_vis or save_model:
        if save_dir is None:
            raise ValueError(
                "save_dir cannot be none if save_vis or save_model is True"
            )
        else:
            if not isinstance(save_dir, Path):
                save_dir = Path(save_dir)
        if save_vis:
            vis_path = save_dir / "vis_vector.npy"
            if calc_noise:
                noise_path = save_dir / "noise_vector.npy"
        if save_model:
            uvw_path = save_dir / "uvw_model.npy"
            ants_path = save_dir / "antpairs.npy"
            red_path = save_dir / "redundancy_model.npy"
            if phase:
                phasor_path = save_dir / "phasor_vector.npy"

    if redundant_avg and not uniform_redundancy:
        # If the data are noisy and being redundantly averaged, the noise
        # estimate derived from the redundantly-averaged visibilities will
        # automatically account for the number of baselines in each
        # redundant group. To avoid double counting, we exclude the number
        # of baselines in each redundant group from the redundancy model.
        uniform_redundancy = True

    # print_rank will only trigger print if verbose is True and rank == 0
    print_rank = 1 - (verbose and rank==0)

    # Preprocess metadata
    uvd = UVData()
    mpiprint(f"\nReading data from: {fp}", rank=print_rank)
    uvd.read(fp, read_data=False)
    uvd.select(ant_str=ant_str)

    if uvd.vis_units not in ["Jy", "K str"]:
        raise ValueError(
            "This code requires calibrated input visibilities in units of "
            "Janskys or kelvin steradians. The input UVData object has "
            f"incompatible units of {uvd.vis_units}."
        )

    try:
        # The future_array_shapes attribute is a legacy attribute that has
        # been removed as of pyuvdata 3.2, but this check should remain for
        # backwards compatibility.
        future_array_shapes = uvd.__getattribute__("_future_array_shapes").value
    except:
        future_array_shapes = False

    if bl_cutoff is not None:
        if not isinstance(bl_cutoff, Quantity):
            bl_cutoff = Quantity(bl_cutoff, unit="m")
        mpiprint(
            f"\nBaseline downselect: keeping |b| <= {bl_cutoff} m",
            rank=print_rank
        )
        bl_lengths = Quantity(
            np.sqrt(np.sum(uvd.uvw_array[:, :2]**2, axis=1)),
            unit="m"
        )
        blt_inds = np.where(bl_lengths <= bl_cutoff)[0]
        mpiprint(f"\tBaselines before length select: {uvd.Nbls}", rank=print_rank)
        uvd.select(blt_inds=blt_inds)
        mpiprint(f"\tBaselines after length select:  {uvd.Nbls}", rank=print_rank)
    bls = uvd.get_antpairs()
    
    # Frequency downselect
    if np.any([param is not None for param in [freq_min, freq_idx_min, freq_center]]):
        freqs = Quantity(uvd.freq_array, unit="Hz")
        if not future_array_shapes:
            freqs = freqs[0]
        if freq_center is not None:
            if Nfreqs is None:
                raise ValueError("Must pass Nfreqs with freq_center")
            if not isinstance(freq_center, Quantity):
                freq_center = Quantity(freq_center, unit="Hz")
            if freq_center.to("Hz") < freqs[0].to("Hz"):
                raise ValueError(
                    f"freq_center ({freq_center.to('MHz')}) < minimum frequency "
                    f"in data ({freqs[0].to('MHz')})"
                )
            if freq_center.to("Hz") > freqs[-1].to("Hz"):
                raise ValueError(
                    f"freq_center ({freq_center.to('MHz')}) > maximum frequency "
                    f"in data ({freqs[-1].to('MHz')})"
                )
            freq_idx_center = np.argmin(
                np.abs(freqs.to("Hz") - freq_center.to("Hz"))
            )
            if freq_idx_center - (Nfreqs//2) < 0:
                raise ValueError(
                    f"Invalid combination of freq_center ({freq_center}) "
                    f"and Nfreqs ({Nfreqs}).  There are fewer than Nfreqs//2 "
                    "frequencies less than or equal to freq_center."
                )
            if freq_idx_center + Nfreqs//2 > freqs.size - 1:
                warnings.warn(
                    "There are less than Nfreqs//2 frequencies "
                    "greater than or equal to freq_center.  This combination "
                    f"of freq_center ({freq_center}) and Nfreqs ({Nfreqs}) "
                    "will result in an asymmetric set of frequencies with "
                    "a central value < freq_center."
                )
            mpiprint(
                f"\nFrequency downselect: keeping {Nfreqs} frequencies "
                + f"centered on {freq_center}",
                rank=print_rank
            )
            freq_idx_min = freq_idx_center - (Nfreqs//2)
            freq_min = freqs[freq_idx_min]
        elif freq_min is not None:
            if not isinstance(freq_min, Quantity):
                freq_min = Quantity(freq_min, unit="Hz")
            if freq_min.to("Hz") < freqs[0].to("Hz"):
                warnings.warn(
                    f"freq_min ({freq_min.to('MHz')}) < minimum frequency "
                    f"in data ({freqs[0].to('MHz')}).  All frequencies will "
                    "kept in the data vector if Nfreqs is not specified."
                )
            if freq_min.to("Hz") > freqs[-1].to("Hz"):
                raise ValueError(
                    f"freq_min ({freq_min.to('MHz')}) > maximum frequency "
                    f"in data ({freqs[-1].to('MHz')})"
                )
            freq_idx_min = np.where(freqs.to("Hz") >= freq_min.to("Hz"))[0][0]
        else:
            if freq_idx_min < 0:
                raise ValueError("freq_idx_min must be positive")
            if freq_idx_min >= freqs.size:
                raise ValueError(
                    "freq_idx_min cannot exceed the number of "
                    "frequencies in the data"
                )
            freq_min = freqs[freq_idx_min]
        if Nfreqs is None:
            Nfreqs = freqs.size - freq_idx_min
        else:
            if Nfreqs <= 0:
                raise ValueError("Nfreqs must be positive")
        if freq_center is None:
            mpiprint(
                f"\nFrequency downselect: keeping {Nfreqs} frequencies "
                + f">= {freq_min.to('MHz'):.2f}",
                rank=print_rank
            )
        if Nfreqs > freqs[freq_idx_min : freq_idx_min+Nfreqs].size:
            warnings.warn(
                "this combination of freq_min or freq_idx_min and "
                "Nfreqs will result in fewer than Nfreqs frequencies being "
                "kept in the data vector."
            )
        freqs = freqs[freq_idx_min : freq_idx_min+Nfreqs]
        mpiprint(f"\tNfreqs before frequency select: {uvd.Nfreqs}", rank=print_rank)
        mpiprint(f"\tNfreqs after frequency select:  {freqs.size}", rank=print_rank)
        mpiprint(
            f"\tMinimum frequency in data vector: {freqs[0].to('MHz'):.2f}",
            rank=print_rank
        )
        mpiprint(
            "\tCentral frequency in data vector: "
            + f"{freqs[freqs.size//2].to('MHz'):.2f}",
            rank=print_rank
        )
        mpiprint(
            f"\tMaximum frequency in data vector: {freqs[-1].to('MHz'):.2f}",
            rank=print_rank
        )
        freqs = freqs.to("Hz").value
    else:
        freqs = None
    
    # Time downselect
    if np.any([param is not None for param in [jd_min, jd_idx_min, jd_center]]):
        jds = Time(np.unique(uvd.time_array), format="jd")
        if jd_center is not None:
            if Ntimes is None:
                raise ValueError("Must pass Ntimes with jd_center")
            if not isinstance(jd_center, Time):
                jd_center = Time(jd_center, format="jd")
            if jd_center.jd < jds[0].jd:
                raise ValueError(
                    f"jd_center ({jd_center.jd}) < minimum time "
                    f"in data ({jds[0].jd})"
                )
            if jd_center.jd > jds[-1].jd:
                raise ValueError(
                    f"jd_center ({jd_center.jd}) > maximum time "
                    f"in data ({jds[-1].jd})"
                )
            jd_idx_center = np.argmin(np.abs(jds.jd - jd_center.jd))
            if jd_idx_center - (Ntimes//2) < 0:
                raise ValueError(
                    f"Invalid combination of jd_center ({jd_center}) and "
                    f"Ntimes ({Ntimes}).  There are fewer than Ntimes//2 times "
                    "less than or equal to jd_center."
                )
            if jd_idx_center + Ntimes//2 > jds.size - 1:
                warnings.warn(
                    "There are less than Ntimes//2 times greater "
                    "than or equal to jd_center.  This combination of "
                    f"jd_center ({jd_center}) and Ntimes ({Ntimes}) will "
                    "result in an asymmetric set of times with a central "
                    "value < jd_center."
                )
            mpiprint(
                f"\nTime downselect: keeping {Ntimes} times "
                + f"centered on {jd_center}",
                rank=print_rank
            )
            jd_idx_min = jd_idx_center - (Ntimes//2)
            jd_min = jds[jd_idx_min]
        elif jd_min is not None:
            if not isinstance(jd_min, Time):
                jd_min = Time(jd_min, format="jd")
            if jd_min.jd < jds[0].jd:
                warnings.warn(
                    f"jd_min ({jd_min.jd}) < minimum time in data ({jds[0].jd}). "
                    "All times will be kept in the data vector if Ntimes is "
                    "not specified."
                )
            if jd_min.jd > jds[-1].jd:
                raise ValueError(
                    f"jd_min ({jd_min.jd}) > maximum time in data ({jds[-1].jd})"
                )
            jd_idx_min = np.where(jds.jd >= jd_min.jd)[0][0]
        else:
            if jd_idx_min < 0:
                raise ValueError("jd_idx_min must be positive")
            if jd_idx_min >= jds.size:
                raise ValueError(
                    "jd_idx_min cannot exceed the number of times in the data"
                )
            jd_min = jds[jd_idx_min]
        if Ntimes is None:
            Ntimes = jds.size - jd_idx_min
        else:
            if Ntimes <= 0:
                raise ValueError("Ntimes must be positive")
        if jd_center is None:
            mpiprint(
                f"\nTime downselect: keeping {Ntimes} times >= {jd_center}",
                rank=print_rank
            )
        if Ntimes > jds[jd_idx_min : jd_idx_min+Ntimes].size:
            warnings.warn(
                "this combination of jd_min or jd_idx_min and Ntimes "
                "will result in fewer than Ntimes times being kept in the "
                "data vector."
            )
        jds = jds[jd_idx_min : jd_idx_min+Ntimes]
        mpiprint(f"\tNtimes before time select: {uvd.Ntimes}", rank=print_rank)
        mpiprint(f"\tNtimes after time select:  {jds.size}", rank=print_rank)
        mpiprint(f"\tMinimum time in data vector: {jds[0].jd}", rank=print_rank)
        mpiprint(
            f"\tCentral time in data vector: {jds[jds.size//2].jd}", rank=print_rank
        )
        mpiprint(f"\tMaximum time in data vector: {jds[-1].jd}", rank=print_rank)
        jds = jds.jd
    else:
        jds = None

    uvd = UVData()
    uvd.read(fp, times=jds, frequencies=freqs, bls=bls)

    if np.sum(uvd.flag_array) > 0:
        # Remove any fully flagged baselines
        antpairs = uvd.get_antpairs()
        good_antpairs = []
        for antpair in antpairs:
            flags = uvd.get_flags(antpair)  # + ('xx',) ?
            if np.sum(flags) < flags.size:
                good_antpairs.append(antpair)
        mpiprint("\nRemoving fully flagged baselines")
        mpiprint("\tBaselines before flagging selection:", uvd.Nbls)
        uvd.select(bls=good_antpairs)
        mpiprint("\tBaselines after flagging selection: ", uvd.Nbls)
    
    if np.any(~uvd._check_for_cat_type("unprojected")):
        # Ensure that data are unphased to begin.  If phasing, they will be
        # rephased later to allow for the phasor vector to be created.
        # This if statement avoids an unecessary warning being printed if the
        # data are already unphased.
        uvd.unproject_phase()
    
    if form_pI:
        mpiprint("\nForming pseudo-Stokes I visibilities", rank=print_rank)
        uvd = form_pI_vis(uvd, norm=pI_norm)
        pol = "pI"
    else:
        if not polstr2num(pol) in uvd.polarization_array:
            raise ValueError(
                f"Polarization {pol} not present in uvd.polarization_array"
            )
        uvd.select(polarizations=[pol])
    
    if uvd.vis_units == "K str":
        uvd.data_array *= 1e3
    else:
        if future_array_shapes:
            uvd.data_array[..., 0] = jy_to_ksr(
                uvd.data_array[..., 0], uvd.freq_array, mK=True
            )
        else:
            uvd.data_array[:, 0, :, 0] = jy_to_ksr(
                uvd.data_array[:, 0, :, 0], uvd.freq_array[0], mK=True
            )
    # WARNING: In BayesEoR, we operate with visibilities in units of 'mK sr'.
    # UVData objects have an attribute vis_units which can be one of 'Jy', 
    # 'K str', or 'uncalib'.  In principle, we could manually set vis_units
    # to 'mK str', but this causes errors with some pyuvdata functions, so
    # we don't bother modifying vis_units here. This means that the vis_units
    # attribute will remain unchanged and its value will not reflect the
    # value of the visibilities in uvd.data_array or in the data vector which
    # will have units of 'mK sr'.
    
    if redundant_avg:
        mpiprint("\nRedundantly averaging visibilities")
        if not uniform_redundancy:
            red_grps = uvd.get_redundancies()[0]
        uvd.compress_by_redundancy(method="average")
    # WARNING: this if condition for the baseline-time axis ordering will 
    # fail if we ever analyze data with baseline-dependent averaging.
    if uvd.blt_order != ("time", "baseline"):
        uvd.reorder_blts(order="time")
    antpairs = uvd.get_antpairs()
    Nbls = uvd.Nbls
    Nbls_vec = 2*Nbls
    
    # FIXME: there is no need for the time axis to be part of the
    # instrument model, it should be removed.  It is a vestigial axis
    # from an old attempt to model phased visibilities using phased
    # (u, v, w) coordinates instead of the phasor vector.
    uvws = np.zeros((uvd.Ntimes, Nbls_vec, 3), dtype=float)
    redundancy = np.ones((uvd.Ntimes, Nbls_vec, 1), dtype=float)
    for i_bl, antpair in enumerate(antpairs):
        uvw = uvd.uvw_array[uvd.antpair2ind(antpair)[0]]
        uvws[:, i_bl] = uvw
        uvws[:, Nbls+i_bl] = -1*uvw
        if redundant_avg and not uniform_redundancy:
            bl = uvd.antnums_to_baseline(*antpair)
            for bls in red_grps:
                if bl in bls:
                    redundancy[:, i_bl] = len(bls)
                    redundancy[:, Nbls+i_bl] = len(bls)
    # Add conjugated antenna pairs to antpairs for a one-to-one mapping
    # with the (u, v, w) coordinates in uvws
    antpairs += [antpair[::-1] for antpair in antpairs]
    
    vis, phasor, noise = uvd_to_vector(
        uvd,
        pol=pol,
        phase=phase,
        phase_time=phase_time,
        calc_noise=calc_noise,
        verbose=verbose,
        rank=rank
    )
    
    if save_vis or save_model:
        args = dict(
            fp=fp.as_posix(),
            ant_str=ant_str,
            bl_cutoff=bl_cutoff,
            freq_min=freq_min,
            freq_idx_min=freq_idx_min,
            freq_center=freq_center,
            Nfreqs=Nfreqs,
            jd_min=jd_min,
            jd_idx_min=jd_idx_min,
            jd_center=jd_center,
            Ntimes=Ntimes,
            form_pI=form_pI,
            pI_norm=pI_norm,
            pol=pol,
            redundant_avg=redundant_avg,
            uniform_redundancy=uniform_redundancy,
            phase=phase,
            phase_time=phase_time,
            calc_noise=calc_noise
        )
    if save_vis:
        mpiprint(f"\nSaving data vector(s) to disk:", rank=print_rank)
        mpiprint(f"\tVisibility vector: {vis_path}", rank=print_rank)
        save_numpy_dict(vis_path, vis, args, clobber=clobber)
        if calc_noise:
            mpiprint(f"\tNoise vector: {noise_path}", rank=print_rank)
            save_numpy_dict(noise_path, noise, args, clobber=clobber)
    if save_model:
        mpiprint(f"\nSaving instrument model to disk:", rank=print_rank)
        mpiprint(f"\tAntpairs: {ants_path}", rank=print_rank)
        save_numpy_dict(ants_path, antpairs, args, clobber=clobber)
        mpiprint(f"\t(u, v, w) model: {uvw_path}", rank=print_rank)
        save_numpy_dict(uvw_path, uvws, args, clobber=clobber)
        mpiprint(f"\tRedundancy model: {red_path}", rank=print_rank)
        save_numpy_dict(red_path, redundancy, args, clobber=clobber)
        if phase:
            mpiprint(f"\tPhasor vector: {phasor_path}", rank=print_rank)
            save_numpy_dict(phasor_path, phasor, args, clobber=clobber)
    
    return_vals = (vis, antpairs, uvws, redundancy, phasor, noise)
    if return_uvd:
        return_vals += (uvd,)
    else:
        return_vals += (None,)
    return return_vals

def uvd_to_vector(
    uvd,
    pol="xx",
    phase=False,
    phase_time=False,
    calc_noise=False,
    verbose=False,
    rank=0
):
    """
    Form a one-dimensional data vector from a :class:`pyuvdata.UVData` object.

    Parameters
    ----------
    uvd : :class:`pyuvdata.UVData`
        UVData object containing unphased visibilities for a single
        polarization specified by `pol`.
    pol : str, optional
        Case-insensitive polarization string. Defaults to 'xx'.
    save_vis : bool, optional
        Write visibility vector to disk in `save_dir`. Defaults to False.
    save_model : bool, optional
        Write instrument model to disk in `save_dir`. Defaults to False.
    save_dir : Path or str, optional
        Output directory for visibility vector which must be specified if
        `save_vis` or `save_model` is True. Defaults to None.
    uniform_redundancy : bool, optional
        Force the redundancy model to be uniform.
    phase : bool, optional
        Create a "phasor vector" which is created identically to the data
        vector which can be used to phase each visibility as a function of
        baseline, time, and frequency using element-wise multiplication.
    phase_time : :class:`astropy.Time` or float, optional
        The time to which the visibilities will be phased as a Julian date if
        not a Time. Defaults to None (phase visibilities to the central time
        if `phase` is True).
    calc_noise : bool, optional
        Calculate a noise estimate from the visibilities via differencing
        adjacent times. Defaults to False.
    verbose : bool, optional
        Print statements useful for debugging. Defaults to False.
    rank : int, optional
        MPI rank. Defaults to 0.
    
    Returns
    -------
    vis_vec : :class:`numpy.ndarray`
        Visibility vector with shape (Nbls*Nfreqs*Ntimes,).
    phasor_vec : :class:`numpy.ndarray`
        Phasor vector which can be multiplied element-wise into `vis` to form
        phased visibilities. Returned only if `phase` is True.
    noise_vec : :class:`numpy.ndarray`
        Estimated noise vector with shape (Nbls*Nfreqs*Ntimes,). Returned only
        if `calc_noise` is True.

    """
    if uvd.Npols > 1:
        raise ValueError(
            "Input UVData object must have only one polarization"
        )
    
    # print_rank will only trigger print if verbose is True and rank == 0
    print_rank = 1 - (verbose and rank==0)

    try:
        # The future_array_shapes attribute is a legacy attribute that has
        # been removed as of pyuvdata 3.2, but this check should remain for
        # backwards compatibility.
        future_array_shapes = uvd.__getattribute__("_future_array_shapes").value
    except:
        future_array_shapes = False

    antpairs = uvd.get_antpairs()

    if phase:
        uvd_phasor = uvd.copy()
        # We form the phasor vector by phasing 1+0j for each time
        # and frequency so that the resulting data vector can be
        # phased by a simple element-wise multiplication.  We have
        # found that this produces better results for modelling phased
        # visibilities than dealing with time-dependent (u, v, w)
        # coordinates.
        uvd_phasor.data_array = np.ones_like(uvd_phasor.data_array)

        jds = Time(np.unique(uvd.time_array), format="jd")
        if phase_time is not None:
            if not isinstance(phase_time, Time):
                phase_time = Time(phase_time, format="jd")
            if phase_time.jd < jds[0].jd:
                warnings.warn(
                    f"phase_time ({phase_time}) < minimum time in data ({jds[0]})"
                )
            if phase_time.jd > jds[-1].jd:
                warnings.warn(
                    f"phase_time ({phase_time}) > maximum time in data ({jds[-1]})"
                )
        else:
            warnings.warn(
                "phase is True but no phase_time specified.  Phasing data to "
                "the central time step."
            )
            phase_time = jds[jds.size//2]
        mpiprint(f"Phasing data to time {phase_time}", rank=print_rank)
        uvd_phasor.phase_to_time(phase_time)

    if calc_noise:
        mpiprint(f"\nEstimating noise", rank=print_rank)
        uvd_noise = uvd.copy()
        for antpair in antpairs:
            vis = uvd_noise.get_data(antpair + (pol,), force_copy=True)
            even_times = vis[::2]
            odd_times = np.zeros_like(even_times)
            odd_times[:-1] = vis[1::2]
            if uvd.Ntimes%2 == 1:
                odd_times[-1] = vis[-2]
            eo_diff = even_times - odd_times
            noise = np.zeros_like(vis)
            noise[::2] = eo_diff
            if uvd.Ntimes%2 == 1:
                noise[1::2] = eo_diff[:-1]
            else:
                noise[1::2] = eo_diff
            blt_inds = uvd_noise.antpair2ind(*antpair)
            if future_array_shapes:
                uvd_noise.data_array[blt_inds, :, 0] = noise
            else:
                uvd_noise.data_array[blt_inds, 0, :, 0] = noise

    # The data vector in BayesEoR needs to be Hermitian, so we include
    # both (u, v) and (-u, -v) in the data vector and instrument model.
    # Thus, we create a vector with shape (2*Nbls*Ntimes*Nfreqs,).
    vis_vec = np.zeros(2*uvd.Nblts*uvd.Nfreqs, dtype=complex)
    if phase:
        phasor_vec = np.zeros_like(vis_vec)
    if calc_noise:
        noise_vec = np.zeros_like(vis_vec)

    # Data are ordered by time, frequency, then baseline.  The baseline
    # axis evolves most rapidly, then frequency, then time.  For example,
    # The first Nbls entries in the data vector are the visibilities for
    # all baselines at the 0th frequency and 0th time, the second Nbls
    # entries in the data vector are the visibilities for all baselines
    # at the 1st frequency and 0th time, etc.  The first Nbls*Nfreqs
    # entries are thus the visibilities for all baselines and frequencies
    # at the 0th time.
    Nbls = uvd.Nbls
    Nbls_vec = 2*Nbls
    mpiprint(f"\nForming data vector(s)", rank=print_rank)
    for i_bl, antpair in enumerate(antpairs):
        antpairpol = antpair + (pol,)
        vis = uvd.get_data(antpairpol)
        vis_vec[i_bl :: Nbls_vec] = vis.flatten()
        vis_vec[Nbls+i_bl :: Nbls_vec] = vis.flatten().conj()
        if phase:
            phasor = uvd_phasor.get_data(antpairpol)
            phasor_vec[i_bl :: Nbls_vec] = phasor.flatten()
            phasor_vec[Nbls+i_bl :: Nbls_vec] = phasor.flatten().conj()
        if calc_noise:
            noise = uvd_noise.get_data(antpairpol)
            noise_vec[i_bl :: Nbls_vec] = noise.flatten()
            noise_vec[Nbls+i_bl :: Nbls_vec] = noise.flatten().conj()

    return_vals = (vis_vec,)
    if phase:
        return_vals += (phasor_vec,)
    else:
        return_vals += (None,)
    if calc_noise:
        return_vals += (noise_vec,)
    else:
        return_vals += (None,)
    return return_vals

def form_pI_vis(uvd, norm=1.0):
    """
    Form pseudo-Stokes I visibilities by summing XX and YY visibilities.

    Parameters
    ----------
    uvd : :class:`pyuvdata.UVData`
        UVData object containing XX and YY polarization visibilities.
    norm : float, optional
        Normalization, ``N``, used in forming pseudo-Stokes I from XX and YY
        via ``pI = N * (XX + YY)``. Defaults to 1.0.

    Returns
    -------
    uvd : :class:`pyuvdata.UVData`
        UVData object containing pI visibilities.

    """
    assert isinstance(uvd, UVData), "uvd must be a pyuvdata.UVData object."

    if polstr2num("pI") not in uvd.polarization_array:
        xx_pol_num = polstr2num("xx")
        yy_pol_num = polstr2num("yy")
        xpol_ind = np.where(uvd.polarization_array == xx_pol_num)[0]
        ypol_ind = np.where(uvd.polarization_array == yy_pol_num)[0]
        uvd.data_array[..., xpol_ind] += uvd.data_array[..., ypol_ind]
        uvd.data_array *= norm
        uvd.select(polarizations=["xx"])
        uvd.polarization_array[0] = polstr2num("pI")

    return uvd

def jy_to_ksr(data, freqs, mK=False):
    """
    Convert visibilities from units of Janskys to kelvin steradians.

    Parameters
    ----------
    data : :class:`numpy.ndarray`
        Two-dimensional array of visibilities with frequency on the second
        axis, i.e. `data` has shape (Ntimes, Nfreqs) or (Nbls*Ntimes, Nfreqs).
    freqs : :class:`astropy.Quantity` or :class:`numpy.ndarray`
        Frequencies along the second axis of `data` in Hertz if not a Quantity.
    mK : bool, optional
        Return data in milikelvin units, i.e. mK sr. Defaults to False (data
        returned in K sr).

    Returns
    -------
    data : :class:`numpy.ndarray`
        Visibilities in units of K sr (or mK sr if `mK` is True).

    """
    if not isinstance(freqs, Quantity):
        freqs = Quantity(freqs, units.Hz)

    equiv = units.brightness_temperature(freqs, beam_area=1*units.sr)
    if mK:
        temp_unit = units.mK
    else:
        temp_unit = units.K
    conv_factor = (1*units.Jy).to(temp_unit, equivalencies=equiv)
    conv_factor *= units.sr / units.Jy

    return data * conv_factor[np.newaxis, :].value

def mock_data_from_eor_cube(
    nu,
    nv,
    nf,
    neta,
    nq,
    chan_selection,
    eor_npz_path=None,
    rank=0
):
    """
    Genenerate a signal vector of visibilities from a 21cmFAST simulated cube in mK.

    Parameters
    ----------
    nu : int
        Number of pixels on a side for the u-axis in the model uv-plane.
    nv : int
        Number of pixels on a side for the v-axis in the model uv-plane.
    nf : int
        Number of frequency channels.
    neta : int
        Number of Line of Sight (LoS, frequency axis) Fourier modes.
    nq : int
        Number of quadratic modes in the Large Spectral Scale Model (LSSM).
    chan_selection : str
        Frequency channel indices used to downselect the LoS axis of the 21cmFAST cube.
    eor_npz_path : str
        Path to a numpy compatible 21cmFAST cube file.
    rank : int
        MPI rank.

    Returns
    -------
    s : :class:`numpy.ndarray` of complex floats
        Signal vector of visibilities generated from the 21cmFAST cube.
    eor_cube : :class:`numpy.ndarray` of floats
        Full 21cmFAST cube.

    Notes
    -----
    * This is a legacy function and needs to be updated for modern functionality.
      This code was originally written when the image domain model was comprised
      of a uniform, rectilinear grid in (l, m) as opposed to the current
      implementation which uses HEALPix pixels for (l, m).

    """
    mpiprint("Using use_EoR_cube data", rank=rank)
    eor_cube = np.load(eor_npz_path)["arr_0"]

    axes_tuple = (1, 2)
    if chan_selection == "0_38_":
        vfft1 = np.fft.ifftshift(eor_cube[0:38]-eor_cube[0].mean() + 0j,
                                 axes=axes_tuple)
    elif chan_selection == "38_76_":
        vfft1 = np.fft.ifftshift(eor_cube[38:76]-eor_cube[0].mean() + 0j,
                                 axes=axes_tuple)
    elif chan_selection == "76_114_":
        vfft1 = np.fft.ifftshift(eor_cube[76:114]-eor_cube[0].mean() + 0j,
                                 axes=axes_tuple)
    else:
        vfft1 = np.fft.ifftshift(eor_cube[0:nf]-eor_cube[0].mean() + 0j,
                                 axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    vfft1 = np.fft.fftn(vfft1, axes=axes_tuple)
    vfft1 = np.fft.fftshift(vfft1, axes=axes_tuple)

    sci_f, sci_v, sci_u = vfft1.shape
    sci_v_centre = sci_v//2
    sci_u_centre = sci_u//2
    vfft1_subset = vfft1[0:nf,
                         sci_u_centre - nu//2:sci_u_centre + nu//2 + 1,
                         sci_v_centre - nv//2:sci_v_centre + nv//2 + 1]
    # s_before_ZM = vfft1_subset.flatten() / vfft1[0].size**0.5
    s_before_ZM = vfft1_subset.flatten()
    ZM_vis_ordered_mask = np.ones(nu*nv*nf)
    ZM_vis_ordered_mask[nf*((nu*nv)//2):nf*((nu*nv)//2 + 1)] = 0
    ZM_vis_ordered_mask = ZM_vis_ordered_mask.astype(bool)
    ZM_chan_ordered_mask = ZM_vis_ordered_mask.reshape(-1, neta+nq).T.flatten()
    s = s_before_ZM[ZM_chan_ordered_mask]

    return s, eor_cube


def generate_mock_eor_signal_instrumental(
        Finv, nf, fov_ra_deg, fov_dec_deg, nside, telescope_latlonalt,
        central_jd, nt, int_time, wn_rms=6.4751478, random_seed=123456,
        beam_type="uniform", rank=0):
    """
    Generate a mock dataset using numpy generated white noise for the sky
    signal. Instrumental effects are included via the calculation of
    visibilities using `Finv`.

    Parameters
    ----------
    Finv : :class:`numpy.ndarray` of complex floats
        2D non-uniform DFT matrix describing the transformation from
        (l, m, n, f) to instrumentally sampled, phased (u, v, w, f) space.
    nf : int
        Number of frequency channels.
    fov_ra_deg : float
        Field of view in degrees of the RA axis of the sky model.
    fov_dec_deg : float
        Field of view in degrees of the DEC axis of the sky model.
    nside : int
        HEALPix nside parameter.
    telescope_latlonalt : tuple of floats
        The latitude, longitude, and altitude of the telescope in degrees,
        degrees, and meters, respectively.
    central_jd : float
        Central time step of the observation in JD2000 format.
    nt : int
        Number of times.
    int_time : float
        Integration time in seconds.
    wn_rms : float
        RMS of the white noise sky model in milikelvin. Defaults to
        6.4751478 milikelvin.
    random_seed : int
        Used to seed `np.random` when generating the sky realization.
    rank : int
        MPI rank.

    Returns
    -------
    s : :class:`numpy.ndarray` of complex floats
        Signal vector of visibilities generated from the white noise sky
        realization.
    white_noise_sky : :class:`numpy.ndarray` of floats
        White noise sky realization.

    """
    mpiprint("Generating white noise sky signal...", rank=rank)
    hpx = Healpix(
        fov_ra_deg=fov_ra_deg,
        fov_dec_deg=fov_dec_deg,
        nside=nside,
        telescope_latlonalt=telescope_latlonalt,
        central_jd=central_jd,
        nt=nt,
        int_time=int_time,
        beam_type=beam_type
    )
    # RMS scaled to hold the power spectrum amplitude constant
    wn_rms *= nside / 256
    mpiprint(f"Seeding numpy.random with {random_seed}", rank=rank)
    np.random.seed(random_seed)
    white_noise_sky = np.random.normal(0.0, wn_rms, (nf, hpx.npix_fov))
    white_noise_sky -= white_noise_sky.mean(axis=1)[:, None]
    s = np.dot(Finv, white_noise_sky.flatten())

    return s, white_noise_sky
