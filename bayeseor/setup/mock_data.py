import numpy as np

from ..model.healpix import Healpix
from .utils import mpiprint


def generate_data_from_loaded_eor_cube(
        nu, nv, nf, neta, nq, chan_selection, eor_npz_path=None, rank=0):
    """
    Genenerate a signal vector of visibilities from a 21cmFAST simulated cube
    in mK.

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
        Frequency channel indices used to downselect the LoS axis of the
        21cmFAST cube.
    eor_npz_path : str
        Path to a numpy compatible 21cmFAST cube file.
    rank : int
        MPI rank.

    Returns
    -------
    s : np.ndarray of complex floats
        Signal vector of visibilities generated from the 21cmFAST cube.
    eor_cube : np.ndarray of floats
        Full 21cmFAST cube.

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
    Finv : np.ndarray of complex floats
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
    s : np.ndarray of complex floats
        Signal vector of visibilities generated from the white noise sky
        realization.
    white_noise_sky : np.ndarray of floats
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
