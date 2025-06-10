import numpy as np
from astropy import units
from astropy.units import Quantity
from astropy.time import Time
from pathlib import Path
from pyuvdata import UVData
from pyuvdata.utils import polstr2num
import warnings

from ..model.healpix import Healpix
from ..utils import mpiprint


def preprocess_data(
    fp,
    ant_str="cross",
    save_vis=False,
    save_model=False,
    out_dir=Path("./"),
    uniform_redundancy=False,
    single_bls=False,
    bl_cutoff=None,
    freq_min=None,
    freq_idx_min=None,
    freq_center=None,
    Nfreqs=None,
    jd_min=None,
    jd_idx_min=None,
    jd_center=None,
    Ntimes=None,
    phase=False,
    phase_time=False,
    form_pI=True,
    pI_norm=1.0,
    pol="xx",
    calc_noise=False,
    blgroup_noise=False,
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

    Parameters
    ----------
    fp : Path or str
        Path to pyuvdata-compatible file containing visibilities.
    ant_str : str, optional
        Antenna downselect string. This determines what baselines to keep in
        the data vector. Please see `pyuvdata.UVData.select` for more details.
        Defaults to 'cross' (cross-correlation baselines only).
    save_vis : bool, optional
        Write visibility vector to disk in `out_dir`.
    save_model : bool, optional
        Write instrument model to disk in `out_dir`.
    out_dir : Path or str, optional
        Output directory for visibility vector if `save_vec` is True.
    uniform_redundancy : bool, optional
        Force the redundancy model to be uniform.
    single_bls : bool, optional
        Create data vectors for each individual baseline.
    bl_cutoff : float, optional
        Baseline length cutoff in meters. Defaults to None (keep all
        baselines).
    freq_min : :class:`astropy.Quantity` or float, optional
        Minimum frequency to keep in the data vector in Hertz if not a
        Quantity. All frequencies greater than or equal to `freq_min` will be
        kept, unless `Nfreqs` is specified. Defaults to None (keep all
        frequencies).
    freq_idx_min : int, optional
        Minimum frequency channel index to keep in the data vector. Defaults
        to None (keep all frequencies).
    freq_center : :class:`astropy.Quantity` or float, optional
        Central frequency around which `Nfreqs` frequencies will be kept in the
        data vector in Hertz of not a Quantity. `Nfreqs` must also be passed,
        otherwise an error is raised. Note, if `Nfreqs` is even, the resulting
        set of frequencies kept will be asymmetric around `freq_center` with 
        one additional frequency channel on the low end, but the central 
        frequency channel, ``freqs[Nfreqs//2]``, will still match
        `freq_center`.  Defaults to None (keep all frequencies).
    Nfreqs : int, optional
        Number of frequencies to keep starting from `freq_idx_min` or the
        channel corresponding to `freq_min`. Defaults to None (keep all
        frequencies).
    jd_min : :class:`astropy.Time` or float, optional
        Minimum time to keep in the data vector as a Julian date if not a Time.
        Defaults to None (keep all times).
    jd_idx_min : int, optional
        Minimum time index to keep in the data vector. Defaults to None (keep
        all times).
    jd_center : :class:`astropy.Time` or float, optional
        Central Julian date around which `Ntimes` times will be kept in the
        data vector as a Julian date if not a Time. `Ntimes` must also be
        passed, otherwise an error is raised.  Note, if `Ntimes` is even, the
        resulting set of times kept will be asymmetric around `jd_center` with
        one additional time being kept on the low end, but the central time,
        ``jds[Ntimes//2]``, will still match `jd_center`. Defaults to None
        (keep all times).
    Ntimes : int, optional
        Number of times to keep starting from `jd_idx_min` or the time
        corresponding to `jd_min`. Defaults to None (keep all times).
    phase : bool, optional
        Create a "phasor vector" which is created identically to the data
        vector which can be used to phase each visibility as a function of
        baseline, time, and frequency using element-wise multiplication.
    phase_time : :class:`astropy.Time` or float, optional
        The time to which the visibilities will be phased as a Julian date if
        not a Time. Defaults to None (phase visibilities to the central time
        if `phase` is True).
    form_pI : bool, optional
        Form pseudo-Stokes I visibilities. Defaults to True. Otherwise, use
        the polarization specified by `pol`.
    pI_norm : float, optional
        Normalization, ``N``, used in forming pseudo-Stokes I from XX and YY
        via ``pI = N * (XX + YY)``. Defaults to 1.0.
    pol : str, optional
        Case-insensitive polarization string. Defaults to 'xx'.
    calc_noise : bool, optional
        Calculate a noise estimate from the visibilities via differencing
        adjacent times. By default, this time-differenced noise estimate is
        calculated for each baseline independently. All baselines within a
        redundant baseline group can be used simultaneously to form the noise
        estimate if `blgroup_noise` is True.
    blgroup_noise : bool, optional
        Use all baselines within a redundant baseline group to form the noise
        estimate and assign this noise estimate to all baselines in the group.
    verbose : bool, optional
        Print statements useful for debugging. Defaults to False.
    rank : int, optional
        MPI rank. Defaults to 0.

    Returns
    -------
    vis_vec : :class:`numpy.ndarray`
        Visibility vector with shape (Nbls*Nfreqs*Ntimes,).
    uvw_array : :class:`numpy.ndarray`
        Sampled (u, v, w) with shape (Ntimes, Nbls, 3). The ordering of the
        Nbls axis matches the ordering of the baselines in `vis_vec`.
    red_array : :class:`numpy.ndarray`
        Redundancy model containing the number of baselines averaged within
        a redundant baseline group. This redundancy is uniform (all 1s) if
        `uniform_redundancy` is True.
    noise : :class:`numpy.ndarray`, optional
        Estimated noise vector with shape (Nbls*Nfreqs*Ntimes,). Returned only
        if `calc_noise` is True.

    """
    if not isinstance(fp, Path):
        fp = Path(fp)
    if not fp.exists():
        raise FileNotFoundError(f"No such file or directory: '{fp}'")

    # Modify MPI rank so we only print if verbose is True and rank == 0
    rank = 1 - (verbose and rank==0)

    # Preprocess metadata
    uvd = UVData()
    mpiprint(f"\nReading data from: {fp}", rank=rank)
    uvd.read(fp, ant_str=ant_str, read_data=False)

    if bl_cutoff:
        mpiprint(
            f"\nBaseline downselect: keeping |b| <= {bl_cutoff} m",
            rank=rank
        )
        bl_lengths = np.sqrt(np.sum(uvd.uvw_array[:, :2]**2, axis=1))
        blt_inds = np.where(bl_lengths <= bl_cutoff)[0]
        mpiprint(f"\tBaselines before length select: {uvd.Nbls}", rank=rank)
        uvd.select(blt_inds=blt_inds)
        mpiprint(f"\tBaselines after length select:  {uvd.Nbls}", rank=rank)
        bls_to_read = uvd.get_antpairs()
    
    # Frequency downselect
    if freq_min or freq_idx_min or freq_center:
        freqs = Quantity(uvd.freq_array, unit="Hz")
        if not uvd.future_array_shapes:
            freqs = freqs[0]
        if freq_center:
            if Nfreqs is None:
                raise ValueError("Must pass Nfreqs with freq_center")
            if not isinstance(freq_center, Quantity):
                freq_center = Quantity(freq_center, unit="Hz")
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
                    "WARNING: There are more than Nfreqs//2 frequencies "
                    "greater than or equal to freq_center.  This combination "
                    f"of freq_center ({freq_center}) and Nfreqs ({Nfreqs}) "
                    "will result in an asymmetric set of frequencies with "
                    "a central value < freq_center."
                )
            freq_idx_min = freq_idx_center - (Nfreqs//2)
            freq_min = freqs[freq_idx_min]
        elif freq_min:
            if not isinstance(freq_min, Quantity):
                freq_min = Quantity(freq_min, unit="Hz")
            freq_idx_min = np.where(freqs.to("Hz") >= freq_min.to("Hz"))[0][0]
        else:
            freq_min = freqs[freq_idx_min]
        if Nfreqs is None:
            Nfreqs = freqs.size - freq_idx_min
        if not freq_center:
            mpiprint(
                f"\nFrequency downselect: keeping {Nfreqs} frequencies "
                + f">= {freq_min.to('MHz'):.2f}",
                rank=rank
            )
        if freq_idx_min + Nfreqs >= freqs.size:
            warnings.warn(
                "WARNING: this combination of freq_min or freq_idx_min and "
                "Nfreqs will result in fewer than Nfreqs frequencies being "
                "kept in the data vector."
            )
        freqs = freqs[freq_idx_min : freq_idx_min+Nfreqs]
        mpiprint(f"\tNfreqs before frequency select: {uvd.Nfreqs}", rank=rank)
        mpiprint(f"\tNfreqs after frequency select:  {freqs.size}", rank=rank)
        mpiprint(
            f"\tMinimum frequency in data vector: {freqs[0].to('MHz'):.2f}",
            rank=rank
        )
        mpiprint(
            "\tCentral frequency in data vector: "
            + f"{freqs[freqs.size//2].to('MHz'):.2f}",
            rank=rank
        )
        mpiprint(
            f"\tMaximum frequency in data vector: {freqs[-1].to('MHz'):.2f}",
            rank=rank
        )
    
    # Time downselect
    if jd_min or jd_idx_min or jd_center:
        jds = Time(np.unique(uvd.time_array), format="jd")
        if jd_center:
            if Ntimes is None:
                raise ValueError("Must pass Ntimes with jd_center")
            if not isinstance(jd_center, Time):
                jd_center = Time(jd_center, format="jd")
            jd_idx_center = np.argmin(np.abs(jds.jd - jd_center.jd))
            if jd_idx_center - (Ntimes//2) < 0:
                raise ValueError(
                    f"Invalid combination of jd_center ({jd_center}) and "
                    f"Ntimes ({Ntimes}).  There are fewer than Ntimes//2 times "
                    "less than or equal to jd_center."
                )
            if jd_idx_center + Ntimes//2 > jds.size - 1:
                warnings.warn(
                    "WARNING: There are more than Ntimes//2 times greater "
                    "than or equal to jd_center.  This combination of "
                    f"jd_center ({jd_center}) and Ntimes ({Ntimes}) will "
                    "result in an asymmetric set of times with a central "
                    "value < jd_center."
                )
            mpiprint(
                f"\nTime downselect: keeping {Ntimes} times "
                + f"centered on {jd_center}",
                rank=rank
            )
        elif jd_min:
            if not isinstance(jd_min, Time):
                jd_min = Time(jd_min, format="jd")
            jd_idx_min = np.where(jds.jd >= jd_min.jd)[0][0]
        else:
            jd_min = jds[jd_idx_min]
        if Ntimes is None:
            Ntimes = jds.size - jd_idx_min
        if not jd_center:
            mpiprint(
                f"\nTime downselect: keeping {Ntimes} times >= {jd_center}",
                rank=rank
            )
        if jd_idx_min + Ntimes >= jds.size:
            warnings.warn(
                "WARNING: this combination of jd_min or jd_idx_min and Ntimes "
                "will result in fewer than Ntimes times being kept in the "
                "data vector."
            )
        jds = jds[jd_idx_min : jd_idx_min+Ntimes]
        mpiprint(f"\tNtimes before time select: {uvd.Ntimes}", rank=rank)
        mpiprint(f"\tNtimes after time select:  {jds.size}", rank=rank)
        mpiprint(f"\tMinimum time in data vector: {jds[0].jd}", rank=rank)
        mpiprint(
            f"\tCentral time in data vector: {jds[jds.size//2].jd}", rank=rank
        )
        mpiprint(f"\tMaximum time in data vector: {jds[-1].jd}", rank=rank)
        
    # Preprocess data...

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
    Convert visibilities from units of Janskys to Kelvin steradians.

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
