import numpy as np
from astropy import units
from astropy.units import Quantity
from pathlib import Path
from pyuvdata import UVData

from ...model.healpix import Healpix
from ...utils import mpiprint


def preprocess_data(
    fp,
    ant_str=None,
    save_vec=False,
    save_model=False,
    out_dir=Path("./"),
    uniform_redundancy=False,
    single_bls=False,
    bl_cutoff=None,
    freq_min=None,
):
    """
    Read visibility data from disk and form a one-dimensional data vector.

    This function loads a `pyuvdata`-compatible file containing visibilities
    for a set of baselines (Nbls), times (Ntimes), and frequencies (Nfreqs)
    and forms a one-dimensional data vector with shape
    (2*Nbls * Ntimes * Nfreqs,) and an accompanying instrument model of
    sampled (u, v, w) coordinates and a redundancy model (when averaging
    redundant baselines).  The factor of 2 in the data vector shape comes from
    the fact that the input visibility vector must be Hermitian, so we copy
    all baselines at (u, v) to (-u, -v) and conjugate the data accordingly.
    This is captured in the instrument model as the instrument model also
    contains all (u, v) and conjugated (-u, -v).

    Parameters
    ----------
    fp : Path or str
        Path to pyuvdata-compatible file containing visibilities.
    ant_str : str, optional
        Antenna downselect string.  This determines what baselines to keep in
        the data vector.  Please see `pyuvdata.UVData.select` for more details.
    save_vec : bool, optional
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
        Baseline length cutoff in meters.  Defaults to None (keep all
        baselines).
    freq_min : :class:`astropy.Quantity` or float, optional
        Minimum frequency to keep in the data vector in Hertz if not a
        Quantity.  Defaults to None (keep all frequencies).
    freq_idx_min : int, optional
        Minimum frequency channel index to keep in the data vector.  Defaults
        to None (keep all frequencies).
    Nfreqs : int, optional
        Number of frequencies to keep starting from `freq_idx_min` or the
        channel corresponding to `freq_min`.  Defaults to None (keep all
        frequencies).
    time_min : :class:`astropy.Time` or float, optional
        Minimum time to keep in the data vector expressed as a Julian Date.
        Defaults to None (keep all times).
    time_idx_min : int, optional
        Minimum time index to keep in the data vector.  Defaults to None (keep
        all times).
    Ntimes : int, optional
        Number of times to keep starting from `time_idx_min` or the time
        corresponding to `time_min`.  Defaults to None (keep all times).
    phase : bool, optional
        Create a "phasor vector" which is created identically to the data
        vector which can be used to phase each visibility as a function of
        baseline, time, and frequency using element-wise multiplication.
    phase_time : :class:`astropy.Time` or float, optional
        The time to which the visibilities will be phased as a Julian Date.
        Defaults to None (phase visibilities to the central time if `phase`
        is True).
    form_pI : bool, optional
        Form pseudo-Stokes I visibilities.  Defaults to True.  Otherwise, use
        the polarization specified by `pol`.
    pI_norm : float, optional
        Normalization, ``N``, used in forming pseudo-Stokes I from XX and YY
        via ``pI = N * (XX + YY)``.  Defaults to 1.0.
    pol : str, optional
        Case-insensitive polarization string.  Defaults to 'xx'.
    calc_noise : bool, optional
        Calculate a noise estimate from the visibilities via differencing
        adjacent times.  By default, this time-differenced noise estimate is
        calculated for each baseline independently.  All baselines within a
        redundant baseline group can be used simultaneously to form the noise
        estimate if `blgroup_noise` is True.
    blgroup_noise : bool, optional
        Use all baselines within a redundant baseline group to form the noise
        estimate and assign this noise estimate to all baselines in the group.

    Returns
    -------
    vis_vec : :class:`numpy.ndarray`
        Visibility vector with shape (Nbls*Nfreqs*Ntimes,).
    uvw_array : :class:`numpy.ndarray`
        Sampled (u, v, w) with shape (Ntimes, Nbls, 3).  The ordering of the
        Nbls axis matches the ordering of the baselines in `vis_vec`.
    red_array : :class:`numpy.ndarray`
        Redundancy model containing the number of baselines averaged within
        a redundant baseline group.  This redundancy is uniform (all 1s) if
        `uniform_redundancy` is True.
    noise : :class:`numpy.ndarray`, optional
        Estimated noise vector with shape (Nbls*Nfreqs*Ntimes,).  Returned only
        if `calc_noise` is True.

    """
    uvd = UVData()
    uvd.read(fp)
    # Preprocess data...

def form_pseudo_stokes_vis(uvd, convention=1.0):
    """
    Form pseudo-Stokes I visibilities by summing XX + YY visibilities.
    """

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
        Return data in milikelvin units, i.e. mK sr.  Defaults to False (data
        returned in K sr).

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

def form_data_vector(
    uvd,
    save=False,
    out_dir=Path("./")
):
    """
    Form a one-dimensional visibility data vector from a UVData object.

    

    Parameters
    ----------
    uvd : UVData
        UVData object containing visibilities.
    

    Returns
    -------
    vis_vec : :class:`numpy.ndarray`
        One-dimensional vector of visibilities.
    uvw_array : :class:`numpy.ndarray`
        Sampled (u, v, w) coordinates with shape (Ntimes, 2*Nbls, 3).

    """


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
    signal.  Instrumental effects are included via the calculation of
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
