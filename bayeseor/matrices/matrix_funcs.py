from numpy import *  # don't know if this will be an issue
import os  # can be removed after astropy_healpix fixes
import numpy as np
from scipy import sparse

"""
    Useful DFT link:
    https://www.cs.cf.ac.uk/Dave/Multimedia/node228.html
"""


# FT array coordinate functions
def sampled_uv_vectors(nu, nv, du=1.0, dv=1.0, exclude_mean=True):
    """
    Creates vectorized arrays of 2D grid coordinates for the rectilinear
    model uv-plane.

    Parameters
    ----------
    nu : int
        Number of pixels on a side for the u-axis in the model uv-plane.
    nv : int
        Number of pixels on a side for the v-axis in the model uv-plane.
    du : float, optional
        Spacing between adjacent u, i.e. :math:`\\Delta u`.  Defaults to 1.
    dv : float, optional
        Spacing between adjacent v, i.e. :math:`\\Delta v`.  Defaults to 1.
    exclude_mean : bool
        If True, remove the (u, v) = (0, 0) pixel from the model
        uv-plane coordinate arrays. Defaults to True.

    Returns
    -------
    us_vec : np.ndarray of float
        Flattened vector (C ordering) of u coordinates in the model uv plane.
    vs_vec : np.ndarray of float
        Flattened vector (C ordering) of v coordinates in the model uv plane.

    """
    us, vs = np.meshgrid(np.arange(nu) - nu//2, np.arange(nv) - nv//2)
    us_vec = du * us.reshape(1, nu*nv)
    vs_vec = dv * vs.reshape(1, nu*nv)

    if exclude_mean:
        us_vec = np.delete(us_vec, (nu*nv)//2, axis=1)
        vs_vec = np.delete(vs_vec, (nu*nv)//2, axis=1)

    return us_vec, vs_vec


def build_nudft_array(coords_in, coords_out, sign=-1):
    """
    Construct a non-uniform discrete Fourier transform (NUDFT) matrix.
    
    Parameters
    ----------
    coords_in : np.ndarray of float
        Input coordinates with shape (Nin,) or (Nax, Nin) where Nin is the
        number of input coordinates and Nax is the number of axes.  Nax must
        be the same for `coords_in` and `coords_out`.  `coords_in` and
        `coords_out` should have reciprocal units.
    coords_out : np.ndarray of float
        Output coordinates with shape (Nout,) or (Nax, Nout) where Nout is the
        number of output coordinates and Nax is the number of axes.  Nax must
        be the same for `coords_in` and `coords_out`.  `coords_in` and
        `coords_out` should have reciprocal units.
    sign : {-1, +1}, optional
        Sign of the :math:`2\\pi i` term in the argument of the exponent.
        Defaults to -1 (negative).

    Returns
    -------
    nudft : np.ndarray of complex
        Complex, NUDFT matrix.

    Examples
    --------
    In the examples below, we construct uniform discrete Fourier transform
    (DFT) matrices so we can compare the results with `numpy.fft`.  The input
    and output coordinates need not be uniformly spaced in practice.

    In this example, we construct a one-dimensional NUDFT matrix relating
    frequency and delay:

    >>> freqs = np.linspace(100, 150, 21)
    >>> freqs -= freqs[freqs.size//2]  # for consistency with numpy
    >>> df = freqs[1] -  freqs[0]
    >>> delays = np.fft.fftshift(np.fft.fftfreq(freqs.size, d=df))
    >>> coords_in = freqs
    >>> print(f"{coords_in.shape = }")
    coords_in.shape = (21,)
    >>> coords_out = delays
    >>> print(f"{coords_out.shape = }")
    coords_out.shape = (21,)
    >>> nudft = build_nudft_array(coords_in, coords_out)
    >>> print(f"{nudft.shape = }")
    nudft.shape = (21, 21)

    Subtracting the central frequency from `freqs` in this example is only
    required for comparing the result of applying `nudft` to a one-dimensional
    signal with the resulting FFTd quantity from `numpy.fft.fft`:    

    >>> np.random.seed(912387)
    >>> signal = 1j*np.random.normal(0, 1, freqs.size)
    >>> signal += np.random.normal(0, 1, freqs.size)
    >>> nudft_signal = nudft @ signal
    >>> fft_signal = np.fft.ifftshift(signal)
    >>> fft_signal = np.fft.fft(fft_signal)
    >>> fft_signal = np.fft.fftshift(fft_signal)
    >>> print(np.abs(nudft_signal - fft_signal).mean())
    3.9389409635500696e-15

    In this example, we construct a two-dimensional NUDFT matrix relating
    (l, m) to (u, v):

    >>> Nl = 15
    >>> Nm = 18
    >>> dl = 1.3
    >>> dm = 1.3
    >>> ls = dl * np.arange(-(Nl//2), Nl//2 + Nl%2)
    >>> ms = dl * np.arange(-(Nm//2), Nm//2 + Nm%2)
    >>> ls_mg, ms_mg = np.meshgrid(ls, ms)
    >>> ls_vec = ls_mg.flatten()
    >>> ms_vec = ms_mg.flatten()
    >>> us = np.fft.fftshift(np.fft.fftfreq(ls.size, d=dl))
    >>> vs = np.fft.fftshift(np.fft.fftfreq(ms.size, d=dm))
    >>> us_mg, vs_mg = np.meshgrid(us, vs)
    >>> us_vec = us_mg.flatten()
    >>> vs_vec = vs_mg.flatten()
    >>> coords_in = np.vstack((ls_vec, ms_vec))
    >>> print(f"{coords_in.shape = }")
    coords_in.shape = (2, 270)
    >>> coords_out = np.vstack((us_vec, vs_vec))
    >>> print(f"{coords_out.shape = }")
    coords_out.shape = (2, 270)
    >>> nudft = build_nudft_array(coords_in, coords_out)
    >>> print(f"{nudft.shape = }")
    nudft.shape = (270, 270)

    The for loop in this code is only used if `Nax`, the number of coordinate
    axes, is greater than 1.  In the example above, because `coords_in` and
    `coords_out` contain are two-dimensional arrays, with the zeroth axis
    indexing the `Nax` axis, the for loop accounts for the fact that both
    coordinates must appear in the argument of the exponent for the two-
    dimensional Fourier transform to be applied correctly to both coordinate
    axes.

    Comparing the results of applying `nudft` to a flattened two-dimensional
    input signal yields:

    >>> signal = 1j*np.random.normal(0, 1, ls_mg.shape)
    >>> signal += np.random.normal(0, 1, ls_mg.shape)
    >>> nudft_signal = (nudft @ signal.flatten()).reshape(us_mg.shape)
    >>> axes = (0, 1)
    >>> fft_signal = np.fft.ifftshift(signal, axes=axes)
    >>> fft_signal = np.fft.fftn(fft_signal, axes=axes)
    >>> fft_signal = np.fft.fftshift(fft_signal, axes=axes)
    >>> print(np.abs(nudft_signal - fft_signal).mean())
    2.7322899219671044e-14

    """
    if len(coords_in.shape) == 1:
        coords_in = coords_in.reshape(1, -1)
    if len(coords_out.shape) == 1:
        coords_out = coords_out.reshape(1, -1)

    if coords_in.shape[0] != coords_out.shape[0]:
        raise ValueError(
            f"coords_in and coords_out must have the same number of "
            f"coordinate axes but have {coords_in.shape[0]} and "
            f"{coords_out.shape[0]}, respectively."
        )

    nudft = np.exp(sign*2*np.pi*1j*np.outer(coords_out[0], coords_in[0]))
    if coords_in.shape[0] > 1:
        for i in range(1, coords_in.shape[0]):
            nudft *= np.exp(
                sign*2*np.pi*1j*np.outer(coords_out[i], coords_in[i])
            )

    return nudft


def Produce_Coordinate_Arrays_ZM_SH(nu, nv):
    """
    Creates vectorized arrays of 2D subharmonic grid (SHG) coordinates for the
    model uv-plane.

    Parameters
    ----------
    nu : int
        Number of pixels on a side for the u-axis in the SH model uv-plane.
    nv : int
        Number of pixels on a side for the v-axis in the SH model uv-plane.

    """
    us, vs = np.meshgrid(np.arange(nu) - nu//2, np.arange(nv) - nv//2)
    us_vec = us.reshape(1, nu*nv)
    us_vec = np.delete(us_vec, (nu*nv)//2, axis=1)
    vs_vec = vs.reshape(1, nu*nv)
    vs_vec = np.delete(vs_vec, (nu*nv)//2, axis=1)

    return us_vec, vs_vec


# Finv functions
def nuDFT_Array_DFT_2D_v2d0(
        sampled_lmn_coords_radians, sampled_uvw_coords_wavelengths):
    """
    Non-uniform DFT from floating point (l, m, n) to instrumentally
    sampled (u, v, w) from the instrument model.

    Used in the construction of a single frequency's block in Finv.

    Parameters
    ----------
    sampled_lmn_coords_radians : np.ndarray of floats
        Array with shape (npix, 3) containing the (l, m, n) coordinates
        of the image space HEALPix model in units of radians.
    sampled_uvw_coords_wavelengths : np.ndarray of floats
        Array with shape (nbls, 3) containing the (u, v, w) coordinates
        at a single frequency in units of wavelengths (inverse radians).

    Returns
    -------
    ExponentArray : np.ndarray of complex floats
        Non-uniform DFT array with shape (nbls, npix).

    """

    # Use HEALPix sampled (l, m) coords
    i_l_AV = sampled_lmn_coords_radians[:, 0].reshape(-1, 1)
    i_m_AV = sampled_lmn_coords_radians[:, 1].reshape(-1, 1)
    i_n_AV = sampled_lmn_coords_radians[:, 2].reshape(-1, 1)

    # Use instrumental uv coords loaded in params
    i_u_AV = sampled_uvw_coords_wavelengths[:, 0].reshape(1, -1)
    i_v_AV = sampled_uvw_coords_wavelengths[:, 1].reshape(1, -1)
    i_w_AV = sampled_uvw_coords_wavelengths[:, 2].reshape(1, -1)

    # This formulation expects (l, m) and (u, v)
    # in radians and wavelengths, respectively
    ExponentArray = np.exp(
        +2.0*np.pi*1j*(
                (i_l_AV*i_u_AV)
                + (i_m_AV*i_v_AV)
                + (i_n_AV*i_w_AV)
            )
        )
    # The sign of the argument of the exponent was chosen to be positive
    # to agree with the sign convention used in both healvis and
    # pyuvsim for the visibility equation

    return ExponentArray.T


# Fprime functions
def nuidft_matrix_2d(nu, nv, du, dv, l_vec, m_vec, exclude_mean=True):
    """
    Generates a non-uniform inverse DFT matrix for (u, v) --> HEALPix (l, m).
    
    Includes the (u, v) = (0, 0) pixel if `fit_for_monopole = True`.

    Parameters
    ----------
    nu : int
        Number of pixels on a side for the u-axis in the model uv-plane.
    nv : int
        Number of pixels on a side for the v-axis in the model uv-plane.
    du : float
        Spacing between sampled modes along the u-axis of the model uv-plane.
    dv : float
        Spacing between sampled modes along the v-axis of the model uv-plane.
    l_vec : array-like
        Vector of sampled direction cosines (HEALPix pixel centers) along the
        right ascension axis.
    m_vec : array-like
        Vector of sampled direction cosines (HEALPix pixel centers) along the
        declination axis.
    exclude_mean : boolean
        If True, exclude the (u, v) = (0, 0) pixel.

    Returns
    -------
    nudft_array_2d : np.ndarray of complex floats
        Non-uniform 2D DFT matrix with shape (npix, nuv).

    Notes
    -----
    * Used in the construction of `Fprime`

    """
    u_vec, v_vec = sampled_uv_vectors(
        nu, nv, exclude_mean=exclude_mean, du=du, dv=dv
    )

    l_vec = l_vec.reshape(-1, 1)
    m_vec = m_vec.reshape(-1, 1)

    # Sign change for consistency, Finv chosen to
    # have + to match healvis
    nudft_array_2d = np.exp(-2.0*np.pi*1j*(l_vec*u_vec + m_vec*v_vec))
    nudft_array_2d /= nu * nv

    return nudft_array_2d


def IDFT_Array_IDFT_2D_ZM_SH(
        nu_sh, nv_sh, sampled_lm_coords_radians, spacing='linear'):
    """
    Generates a non-uniform (might want to update the function name)
    inverse DFT matrix that goes from subharmonic grid (SHG) model (u, v) to
    HEALPix (l, m) pixel centers.  Includes the (u, v) = (0, 0) pixel
    if `exclude_mean` = False.

    Used in the construction of `Fprime` if using the SHG.

    Parameters
    ----------
    nu_sh : int
        Number of pixels on a side for the u-axis in the SH model uv-plane.
    nv_sh : int
        Number of pixels on a side for the v-axis in the SH model uv-plane.
    sampled_lm_coords_radians : array_like, shape (nhpx, 2)
        Array containing the (l, m) coordinates of the image space HEALPix
        model in units of radians.
    exclude_mean : boolean
        If True, exclude the (u, v) = (0, 0) pixel from the SHG.
    spacing : {'linear', 'log'}
        Controls the spacing of the SHG modes.  If linear, the spacing between
        SHG modes is the coarse grid spacing `delta_u_irad` divided by the
        number of SHG modes `nu_sh`.  If log, the SHG modes are determined via
        `np.logspace`.

    Returns
    -------
    ExponentArray : np.ndarray, shape (npix, nuv_sh)
        Non-uniform, complex 2D DFT matrix.

    """
    # Replace x and y coordinate arrays with sampled_lm_coords_radians
    x_vec = sampled_lm_coords_radians[:, 0].reshape(-1, 1)
    y_vec = sampled_lm_coords_radians[:, 1].reshape(-1, 1)

    # The uv coordinates need to be rescaled to units of
    # wavelengths by multiplying by the uv pixel area
    # This calculation of the SHG pixel width is only true when using a
    # linear spacing and will need to be reworked if using a log spacing.
    # This also needs to be updated to account for a FoV which differs
    # along the l and m axes.
    if spacing == 'linear':
        u_vec, v_vec = Produce_Coordinate_Arrays_ZM_SH(nu_sh, nv_sh)
        du_sh = p.delta_u_irad / nu_sh
        dv_sh = p.delta_v_irad / nv_sh
        u_vec = u_vec.astype('float') * du_sh
        v_vec = v_vec.astype('float') * dv_sh
    else:
        max_u_sh = p.delta_u_irad
        max_v_sh = p.delta_v_irad
        min_u_sh = 1 / 2  # horizon to horizon period in l-units
        min_v_sh = 1 / 2  # horizon to horizon period in m-units
        u_sh = np.logspace(
            np.log10(min_u_sh), np.log10(max_u_sh),
            nu_sh//2, endpoint=False
        )
        u_sh = np.concatenate((-u_sh[::-1], np.zeros(1), u_sh))
        v_sh = np.logspace(
            np.log10(min_v_sh), np.log10(max_v_sh),
            nv_sh//2, endpoint=False
        )
        v_sh = np.concatenate((-v_sh[::-1], np.zeros(1), v_sh))

        u_sh, v_sh = np.meshgrid(u_sh, v_sh)
        u_vec = u_sh.reshape(1, -1)
        u_vec = np.delete(u_vec, u_vec.size//2, axis=1)
        v_vec = v_sh.reshape(1, -1)
        v_vec = np.delete(v_vec, v_vec.size//2, axis=1)

    # Sign change for consistency, Finv chosen to
    # have + to match healvis
    ExponentArray = np.exp(-2.0*np.pi*1j*(x_vec*u_vec + y_vec*v_vec))

    return ExponentArray


# Fz functions
def build_lssm_basis_vectors(
        nf, nq=2, npl=0, f_min=159.0, df=0.2, beta=2.63):
    """
    Construct the Large Spectral Scale Model (LSSM) basis vectors.

    Uses polynomial (if `npl = 0`) and/or power law (if `npl > 0`) basis
    vectors.  It must be the case that `len(beta) == npl` if `npl > 0`.
    If `npl == nq`, all basis vectors will be power laws.

    Parameters
    ----------
    nf : int
        Number of frequency channels.
    nq : int
        Number of quadratic modes in the Large Spectral Scale Model (LSSM).
    npl : int
        Number of polynomial modes in `beta`.
    f_min : float
        Minimum frequency channel in MHz.
    df : float
        Frequency channel width in MHz.
    beta : array-like of floats
        Polynomial powers when replacing the default quadratic
        modes with terms of the form `(nu / f_min)**-beta[i]`.

    Returns
    -------
    basis_vectors : np.ndarray of floats
        Array containing the LSSM basis vectors with shape (nq, nf).

    """
    basis_vectors = np.zeros([nq, nf]) + 0j
    freq_array = f_min + np.arange(nf) * df
    if nq == 1:
        x = np.arange(nf) - nf/2.
        basis_vectors[0] = x
        if npl == 1:
            m_pl = np.array(
                [(freq_array[i_f] / f_min)**-beta
                 for i_f in range(len(freq_array))]
                )
            basis_vectors[0] = m_pl
            print('\nLinear LW mode replaced with power-law model')
            print('f_min = ', f_min)
            print('df = ', df)
            print('beta = ', beta, '\n')

    if nq == 2:
        x = np.arange(nf) - nf/2.
        basis_vectors[0] = x
        basis_vectors[1] = x**2
        if npl == 1:
            m_pl = np.array(
                [(freq_array[i_f] / f_min)**-beta
                 for i_f in range(len(freq_array))]
                )
            basis_vectors[1] = m_pl
            print('\nQuadratic LW mode replaced with power-law model')
            print('f_min = ', f_min)
            print('df = ', df)
            print('beta = ', beta, '\n')
        if npl == 2:
            m_pl1 = np.array(
                [(freq_array[i_f] / f_min)**-beta[0]
                 for i_f in range(len(freq_array))]
                )
            basis_vectors[0] = m_pl1
            print('\nLinear LW mode replaced with power-law model')
            print('beta1 = ', beta[0], '\n')
            m_pl2 = np.array(
                [(freq_array[i_f] / f_min)**-beta[1]
                 for i_f in range(len(freq_array))]
                )
            basis_vectors[1] = m_pl2
            print('\nQuadratic LW mode replaced with power-law model')
            print('f_min = ', f_min)
            print('df = ', df)
            print('beta2 = ', beta[1], '\n')

    if nq == 3:
        x = np.arange(nf) - nf/2.
        basis_vectors[0] = x
        basis_vectors[1] = x**2
        basis_vectors[1] = x**3
        if npl == 1:
            m_pl = np.array(
                [(freq_array[i_f] / f_min)**-beta
                 for i_f in range(len(freq_array))]
                )
            basis_vectors[1] = m_pl
            print('\nQuadratic LW mode replaced with power-law model')
            print('f_min = ', f_min)
            print('df = ', df)
            print('beta = ', beta, '\n')

        if npl == 2:
            m_pl1 = np.array(
                [(freq_array[i_f] / f_min)**-beta[0]
                 for i_f in range(len(freq_array))]
                )
            basis_vectors[0] = m_pl1
            print('\nLinear LW mode replaced with power-law model')
            print('beta1 = ', beta[0], '\n')
            m_pl2 = np.array(
                [(freq_array[i_f] / f_min)**-beta[1]
                 for i_f in range(len(freq_array))]
                )
            basis_vectors[1] = m_pl2
            print('\nQuadratic LW mode replaced with power-law model')
            print('f_min = ', f_min)
            print('df = ', df)
            print('beta2 = ', beta[1], '\n')

        if npl == 3:
            m_pl1 = np.array(
                [(freq_array[i_f] / f_min)**-beta[0]
                 for i_f in range(len(freq_array))]
                )
            basis_vectors[0] = m_pl1
            print('\nLinear LW mode replaced with power-law model')
            print('beta1 = ', beta[0], '\n')
            m_pl2 = np.array(
                [(freq_array[i_f] / f_min)**-beta[1]
                 for i_f in range(len(freq_array))]
                )
            # Potential bug if passing len(beta) == 3
            # Should this call beta[2] instead of beta[1]?
            basis_vectors[1] = m_pl2
            print('\nQuadratic LW mode replaced with power-law model')
            print('beta2 = ', beta[1], '\n')
            m_pl3 = np.array(
                [(freq_array[i_f] / f_min)**-beta[1]
                 for i_f in range(len(freq_array))]
                )
            basis_vectors[2] = m_pl3
            print('\nCubic LW mode replaced with power-law model')
            print('f_min = ', f_min)
            print('df = ', df)
            print('beta3 = ', beta[2], '\n')

    if nq == 4:
        basis_vectors[0] = np.arange(nf)
        basis_vectors[1] = np.arange(nf)**2.0
        basis_vectors[2] = 1j*np.arange(nf)
        basis_vectors[3] = 1j*np.arange(nf)**2

    return basis_vectors.T


def idft_matrix_1d(
        nf, neta, nq=0, npl=None, f_min=None, df=None,
        beta=None, include_eta0=True):
    """
    Generate a 1D DFT matrix transforming eta -> frequency.

    Used in the construction of `Fz`.

    Parameters
    ----------
    nf : int
        Number of frequency channels.
    neta : int
        Number of Line of Sight (LoS, frequency axis) Fourier modes.
    nq : int
        Number of quadratic Large Spectral Scale Model (LSSM) basis vectors.
    npl : int
        Number of power law LSSM basis vectors.  Overrides nq, i.e. replaces
        npl <= nq quadratic basis vectors with power law basis vectors.
    f_min : float
        Minimum frequency in megahertz.
    df : float
        Frequency channel width in megahertz.
    beta : float or array-like
        Spectral index(indices) for the `npl` power law LSSM basis vectors.
    include_eta0 : bool
        If True, include eta=0 in the DFT, otherwise exclude it.  Defaults
        to True.

    Returns
    -------
    idft_array : np.ndarray of complex floats
        1D DFT matrix with shape `(nf, neta - (not include_eta0))`.

    """
    i_f = (np.arange(nf)-nf//2).reshape(-1, 1)
    i_eta = (np.arange(neta)-neta//2)
    if not include_eta0:
        i_eta = i_eta[i_eta != 0.0].reshape(1, -1)

    # Sign change for consistency, Finv chosen
    # to have + sign to match healvis
    idft_array = np.exp(-2.0*np.pi*1j*(i_eta*i_f / nf))

    if nq > 0:
        lssm_basis_vectors = build_lssm_basis_vectors(
            nf, nq=nq, npl=npl, f_min=f_min, df=df, beta=beta
        )
        idft_array = np.hstack(
            (idft_array, lssm_basis_vectors)
        )

    return idft_array


def idft_array_idft_1d_sh(
        nf, neta, nq_sh, npl_sh, fit_for_shg_amps=False,
        f_min=None, df=None, beta=None):
    """
    Generate a 1D DFT matrix for the FT along the
    LoS axis from eta -> frequency.  Analagous to
    IDFT_Array_IDFT_1D with the exception that this
    function includes the subharmonic grid (SHG) terms used
    for modeling power on scales larger than the image
    size.

    Used in the construction of `Fz` if using the subharmonic grid.

    Parameters
    ----------
    nf : int
        Number of frequency channels.
    neta : int
        Number of Line of Sight (LoS, frequency axis) Fourier modes.
    nq_sh : int
        Number of quadratic modes.
    npl_sh : int
        Number of power law modes.
    fit_for_shg_amps : boolean
        If true, include pixels in DFT matrix.  Otherwise,
        only model large spectral scale structure.
    f_min : float, optional
        Minimum frequency channel bin center in MHz. Required if `nq_sh` > 0.
    df : float, optional
        Frequency channel width in MHz. Required if `nq_sh` > 0.
    beta : list or tuple of floats, optional
        Power law spectral indices. Required if `nq_sh` > 0.

    Returns
    -------
    idft_array_sh : np.ndarray of complex floats
        Matrix containing the 1D DFT matrix and/or the LSSM for the SHG pixels
        if `fit_for_shg_amps` = True and/or `nq_sh` > 0.

    """
    if not fit_for_shg_amps:
        neta = 1
    i_f = (np.arange(nf) - nf//2).reshape(-1, 1)
    i_eta = (np.arange(neta) - neta//2).reshape(1, -1)

    # Sign change for consistency, Finv chosen
    # to have + sign to match healvis
    idft_array_sh = np.exp(-2.0*np.pi*1j*(i_eta*i_f / nf))
    idft_array_sh /= nf

    if nq_sh > 0:
        # Construct large spectral scale model (LSSM) for the SHG modes
        lssm_sh = build_lssm_basis_vectors(
            nf, nq=nq_sh, npl=npl_sh, f_min=f_min, df=df, beta=beta
        )
        idft_array_sh = np.hstack([idft_array_sh, lssm_sh.T])

    return idft_array_sh


# Gridding matrix functions
def generate_gridding_matrix_vo2co(
        nu, nv, nf, exclude_mean=True, use_sparse=True):
    """
    Generates a matrix which transforms a visibility ordered vector to a
    channel ordered vector.

    Parameters
    ----------
    nu : int
        Number of pixels on a side for the u-axis in the model uv-plane.
    nv : int
        Number of pixels on a side for the v-axis in the model uv-plane.
    nf : int
        Number of frequency channels.
    exclude_mean : bool
        If True, excludes the (u, v) = (0, 0) pixel from the model
        uv-plane coordinate arrays. Defaults to True.
    use_sparse : bool
        If True, use scipy.sparse matrices.  Otherwise, use numpy.ndarray.

    Returns
    -------
    gridding_matrix : np.ndarray
        Array containing the mapping from visibility ordering to channel
        ordering.

    """
    nuv = nu*nv - exclude_mean
    vis_grab_order = np.arange(nuv)
    vals_per_chan = vis_grab_order.size

    matrix_dims = (vals_per_chan*nf, vals_per_chan*nf)
    if use_sparse:
        gridding_matrix = sparse.lil_matrix(matrix_dims)
    else:
        gridding_matrix = np.zeros(matrix_dims)
    for i in range(nf):
        for j, vis_grab_val in enumerate(vis_grab_order):
            row_number = (i*vals_per_chan) + j
            # Pixel to grab from vis-ordered vector
            # and place as next chan-ordered value
            grid_pix = i + vis_grab_val*nf
            gridding_matrix[row_number, grid_pix] = 1

    return gridding_matrix
