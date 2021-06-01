from numpy import * # don't know if this will be an issue
import os # can be removed after astropy_healpix fixes
import numpy as np

from BayesEoR.Linalg.healpix import Healpix
import BayesEoR.Params.params as p

"""
    Useful DFT link:
    https://www.cs.cf.ac.uk/Dave/Multimedia/node228.html
"""


# FT array coordinate functions
def Produce_Coordinate_Arrays_ZM(nu, nv, nx=None, ny=None, exclude_mean=True):
    """
    Creates vectorized arrays of 2D grid coordinates for the rectilinear
    model uv-plane and, if nx and ny are passed, and the rectilinear
    lm-space sky model.

    Parameters
    ----------
    nu : int
        Number of pixels on a side for the u-axis in the model uv-plane.
    nv : int
        Number of pixels on a side for the v-axis in the model uv-plane.
    nx : int
        Number of pixels on a side for the l-axis in the lm-space
        sky model. Defaults to None. (Deprecated).
    ny : int
        Number of pixels on a side for the m-axis in the lm-space
        sky model. Defaults to None. (Deprecated).
    exclude_mean : bool
        If True, remove the (u, v) = (0, 0) pixel from the model
        uv-plane coordinate arrays. Defaults to True.
    """
    us, vs = np.meshgrid(np.arange(nu) - nu//2, np.arange(nv) - nv//2)
    us_vec = us.flatten().reshape(1, nu*nv)
    vs_vec = vs.flatten().reshape(1, nu*nv)

    if exclude_mean:
        us_vec = np.delete(us_vec, (nu*nv)//2, axis=1)
        vs_vec = np.delete(vs_vec, (nu*nv)//2, axis=1)

    return us_vec, vs_vec


def Produce_Coordinate_Arrays_ZM_SH(
        nu, nv, exclude_mean=True, spacing='linear'):
    """
    Creates vectorized arrays of 2D subharmonic grid (SHG) coordinates for the
    model uv-plane.

    Parameters
    ----------
    nu : int
        Number of pixels on a side for the u-axis in the SH model uv-plane.
    nv : int
        Number of pixels on a side for the v-axis in the SH model uv-plane.
    exclude_mean : bool
        If True, remove the (u, v) = (0, 0) pixel from the SH model
        uv-plane coordinate arrays. Defaults to True.
    spacing : str
        Needs development. Can be 'linear' or 'log' spacing of the SHG pixels.
    """
    # Assuming uniform spacing for now
    us, vs = np.meshgrid(np.arange(nu) - nu//2, np.arange(nv) - nv//2)
    us_vec = us.flatten().reshape(1, nu*nv)
    vs_vec = vs.flatten().reshape(1, nu*nv)

    if exclude_mean:
        us_vec = np.delete(us_vec, (nu*nv)//2, axis=1)
        vs_vec = np.delete(vs_vec, (nu*nv)//2, axis=1)

    return us_vec, vs_vec


# Finv functions
def DFT_Array_DFT_2D_ZM(
        nu, nv, nx, ny,
        X_oversampling_Factor=1.0, Y_oversampling_Factor=1.0,
        U_oversampling_Factor=1.0, V_oversampling_Factor=1.0):
    """
    Constructs a uniform DFT matrix which goes from rectilinear
    (l, m, f) sky model coordinates to rectilinear (u, v, f)
    coordinates.

    Used to construct `Finv` if `include_instrumental_effects = False`.
    """
    exclude_mean = True
    if p.fit_for_monopole:
        exclude_mean = False
    i_x_AV, i_y_AV, i_u_AV, i_v_AV =\
        Produce_Coordinate_Arrays_ZM(
            nu, nv, nx=nx, ny=ny, exclude_mean=exclude_mean)

    if U_oversampling_Factor != 1.0:
        i_x_AV, i_y_AV, i_u_AV, i_v_AV =\
            Calc_Coords_Large_Im_to_High_Res_uv(
                i_x_AV, i_y_AV, i_u_AV, i_v_AV,
                U_oversampling_Factor, V_oversampling_Factor)
    if X_oversampling_Factor != 1.0:
        i_x_AV, i_y_AV, i_u_AV, i_v_AV =\
            Calc_Coords_High_Res_Im_to_Large_uv(
                i_x_AV, i_y_AV, i_u_AV, i_v_AV,
                U_oversampling_Factor, V_oversampling_Factor)

    # Updated for python 3: float division is default
    ExponentArray = np.exp(
        -2.0*np.pi*1j*(
                (i_x_AV*i_u_AV / nx)
                + (i_v_AV*i_y_AV / ny)
            )
        )
    return ExponentArray


def nuDFT_Array_DFT_2D_v2d0(
        sampled_lmn_coords_radians,
        sampled_uvw_coords_wavelengths):
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
def IDFT_Array_IDFT_2D_ZM(
        nu, nv,
        sampled_lm_coords_radians,
        exclude_mean=True,
        U_oversampling_Factor=1.0, V_oversampling_Factor=1.0):
    """
    Generates a non-uniform (might want to update the function name)
    inverse DFT matrix that goes from rectilinear model (u, v) to
    HEALPix (l, m) pixel centers.  Includes the (u, v) = (0, 0) pixel
    if `fit_for_monopole = True`.

    Used in the construction of `Fprime`.

    Parameters
    ----------
    nu : int
        Number of pixels on a side for the u-axis in the model uv-plane.
    nv : int
        Number of pixels on a side for the v-axis in the model uv-plane.
    sampled_lm_coords_radians : np.ndarray of floats
        Array with shape (npix, 2) containing the (l, m) coordinates
        of the image space HEALPix model in units of radians.
    exclude_mean : boolean
        If True, exclude the (u, v) = (0, 0) pixel.
    U_oversampling_Factor : float
        Factor by which the subharmonic grid is oversampled relative
        to the coarse grid along the u-axis.
    V_oversampling_Factor : float
        Factor by which the subharmonic grid is oversampled relative
        to the coarse grid.

    Returns
    -------
    ExponentArray : np.ndarray of complex floats
        Non-uniform 2D DFT matrix with shape (npix, nuv).
    """
    i_u_AV, i_v_AV =\
        Produce_Coordinate_Arrays_ZM(nu, nv, exclude_mean=exclude_mean)

    # Replace x and y coordinate arrays with sampled_lm_coords_radians
    i_x_AV = sampled_lm_coords_radians[:, 0].reshape(-1, 1)
    i_y_AV = sampled_lm_coords_radians[:, 1].reshape(-1, 1)

    if U_oversampling_Factor != 1.0:
        i_x_AV, i_y_AV, i_u_AV, i_v_AV =\
            Calc_Coords_Large_Im_to_High_Res_uv(
                i_x_AV, i_y_AV, i_u_AV, i_v_AV,
                U_oversampling_Factor, V_oversampling_Factor)

    # The uv coordinates need to be rescaled to units of
    # wavelengths by multiplying by the uv pixel width
    i_u_AV = i_u_AV.astype('float') * p.delta_u_irad
    i_v_AV = i_v_AV.astype('float') * p.delta_v_irad
    # Sign change for consistency, Finv chosen to
    # have + to match healvis
    ExponentArray = np.exp(-2.0*np.pi*1j*(i_x_AV*i_u_AV + i_v_AV*i_y_AV))

    ExponentArray /= (
            nu*U_oversampling_Factor
            * nv*V_oversampling_Factor
        )
    return ExponentArray.T


def IDFT_Array_IDFT_2D_ZM_SH(
        nu_sh, nv_sh, sampled_lm_coords_radians, exclude_mean=True):
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

    Returns
    -------
    ExponentArray : np.ndarray, shape (npix, nuv_sh)
        Non-uniform, complex 2D DFT matrix.
    """
    u_vec, v_vec =\
        Produce_Coordinate_Arrays_ZM_SH(nu_sh, nv_sh, exclude_mean=exclude_mean)

    # Replace x and y coordinate arrays with sampled_lm_coords_radians
    x_vec = sampled_lm_coords_radians[:, 0].reshape(-1, 1)
    y_vec = sampled_lm_coords_radians[:, 1].reshape(-1, 1)

    # The uv coordinates need to be rescaled to units of
    # wavelengths by multiplying by the uv pixel area
    # This calculation of the SHG pixel width is only true when using a
    # linear spacing and will need to be reworked if using a log spacing.
    # This also needs to be updated to account for a FoV which differs
    # along the l and m axes.
    du_sh = p.delta_u_irad / nu_sh
    dv_sh = p.delta_v_irad / nv_sh
    u_vec = u_vec.astype('float') * du_sh
    v_vec = v_vec.astype('float') * dv_sh

    # Sign change for consistency, Finv chosen to
    # have + to match healvis
    ExponentArray = np.exp(-2.0*np.pi*1j*(x_vec*u_vec + y_vec*v_vec))

    return ExponentArray


# Fz functions
def quadratic_array_linear_plus_quad_modes_only_v2(nf, nq=2, **kwargs):
    """
    Construct the quadratic (long wavelength) arrays or replace the
    quadratic arrays with polynomial coefficients given by `npl`'
    and `beta`.

    Parameters
    ----------
    nf : int
        Number of frequency channels.
    nq : int
        Number of quadratic modes. Permissible values of `nq` are:
          - 1: include the linear mode
          - 2: include the linear and quadratic mode
          - 3: include the linear, quadratic, and cubic mode
          - 4: include complex linear and quadratic modes
    npl : int
        Number of polynomial modes in `beta`.
    nu_min_MHz : float
        Minimum frequency channel in MHz.
    channel_width_MHz : float
        Frequency channel width in MHz.
    beta : array-like of floats
        Polynomial powers when replacing the default quadratic
        modes with terms of the form `(nu / nu_min)**-beta[i]`.

    Returns
    -------
    quadratic_array : np.ndarray of floats
        Array containing the long wavelength modes, either quadratic
        or polynomial if `npl > 0`, with shape (nq, nf).
    """
    # ===== Defaults =====
    default_npl = 0
    default_nu_min_MHz = (163.0-4.0)
    default_channel_width_MHz = 0.2
    default_beta = 2.63

    # ===== Inputs =====
    npl = kwargs.pop(
        'npl', default_npl)
    nu_min_MHz = kwargs.pop(
        'nu_min_MHz', default_nu_min_MHz)
    channel_width_MHz = kwargs.pop(
        'channel_width_MHz', default_channel_width_MHz)
    beta = kwargs.pop(
        'beta', default_beta)

    quadratic_array = np.zeros([nq, nf]) + 0j
    nu_array_MHz = (
            nu_min_MHz + np.arange(float(nf)) * channel_width_MHz)
    if nq == 1:
        x = np.arange(nf) - nf/2.
        quadratic_array[0] = x
        if npl == 1:
            m_pl = np.array(
                [(nu_array_MHz[i_nu] / nu_min_MHz)**-beta
                 for i_nu in range(len(nu_array_MHz))]
                )
            quadratic_array[0] = m_pl
            print('\nLinear LW mode replaced with power-law model')
            print('nu_min_MHz = ', nu_min_MHz)
            print('channel_width_MHz = ', channel_width_MHz)
            print('beta = ', beta, '\n')

    if nq == 2:
        x = np.arange(nf) - nf/2.
        quadratic_array[0] = x
        quadratic_array[1] = x**2
        if npl == 1:
            m_pl = np.array(
                [(nu_array_MHz[i_nu] / nu_min_MHz)**-beta
                 for i_nu in range(len(nu_array_MHz))]
                )
            quadratic_array[1] = m_pl
            print('\nQuadratic LW mode replaced with power-law model')
            print('nu_min_MHz = ', nu_min_MHz)
            print('channel_width_MHz = ', channel_width_MHz)
            print('beta = ', beta, '\n')
        if npl == 2:
            m_pl1 = np.array(
                [(nu_array_MHz[i_nu] / nu_min_MHz)**-beta[0]
                 for i_nu in range(len(nu_array_MHz))]
                )
            quadratic_array[0] = m_pl1
            print('\nLinear LW mode replaced with power-law model')
            print('beta1 = ', beta[0], '\n')
            m_pl2 = np.array(
                [(nu_array_MHz[i_nu] / nu_min_MHz)**-beta[1]
                 for i_nu in range(len(nu_array_MHz))]
                )
            quadratic_array[1] = m_pl2
            print('\nQuadratic LW mode replaced with power-law model')
            print('nu_min_MHz = ', nu_min_MHz)
            print('channel_width_MHz = ', channel_width_MHz)
            print('beta2 = ', beta[1], '\n')

    if nq == 3:
        x = np.arange(nf) - nf/2.
        quadratic_array[0] = x
        quadratic_array[1] = x**2
        quadratic_array[1] = x**3
        if npl == 1:
            m_pl = np.array(
                [(nu_array_MHz[i_nu] / nu_min_MHz)**-beta
                 for i_nu in range(len(nu_array_MHz))]
                )
            quadratic_array[1] = m_pl
            print('\nQuadratic LW mode replaced with power-law model')
            print('nu_min_MHz = ', nu_min_MHz)
            print('channel_width_MHz = ', channel_width_MHz)
            print('beta = ', beta, '\n')

        if npl == 2:
            m_pl1 = np.array(
                [(nu_array_MHz[i_nu] / nu_min_MHz)**-beta[0]
                 for i_nu in range(len(nu_array_MHz))]
                )
            quadratic_array[0] = m_pl1
            print('\nLinear LW mode replaced with power-law model')
            print('beta1 = ', beta[0], '\n')
            m_pl2 = np.array(
                [(nu_array_MHz[i_nu] / nu_min_MHz)**-beta[1]
                 for i_nu in range(len(nu_array_MHz))]
                )
            quadratic_array[1] = m_pl2
            print('\nQuadratic LW mode replaced with power-law model')
            print('nu_min_MHz = ', nu_min_MHz)
            print('channel_width_MHz = ', channel_width_MHz)
            print('beta2 = ', beta[1], '\n')

        if npl == 3:
            m_pl1 = np.array(
                [(nu_array_MHz[i_nu] / nu_min_MHz)**-beta[0]
                 for i_nu in range(len(nu_array_MHz))]
                )
            quadratic_array[0] = m_pl1
            print('\nLinear LW mode replaced with power-law model')
            print('beta1 = ', beta[0], '\n')
            m_pl2 = np.array(
                [(nu_array_MHz[i_nu] / nu_min_MHz)**-beta[1]
                 for i_nu in range(len(nu_array_MHz))]
                )
            # Potential bug if passing len(beta) == 3
            # Should this call beta[2] instead of beta[1]?
            quadratic_array[1] = m_pl2
            print('\nQuadratic LW mode replaced with power-law model')
            print('beta2 = ', beta[1], '\n')
            m_pl3 = np.array(
                [(nu_array_MHz[i_nu] / nu_min_MHz)**-beta[1]
                 for i_nu in range(len(nu_array_MHz))]
                )
            quadratic_array[2] = m_pl3
            print('\nCubic LW mode replaced with power-law model')
            print('nu_min_MHz = ', nu_min_MHz)
            print('channel_width_MHz = ', channel_width_MHz)
            print('beta3 = ', beta[2], '\n')

    if nq == 4:
        quadratic_array[0] = np.arange(nf)
        quadratic_array[1] = np.arange(nf)**2.0
        quadratic_array[2] = 1j*np.arange(nf)
        quadratic_array[3] = 1j*np.arange(nf)**2

    return quadratic_array


def IDFT_Array_IDFT_1D(nf, neta):
    """
    Generate a uniform 1D DFT matrix for the FT along the
    LoS axis from eta -> frequency.  This matrix will be
    consistent with np.fft.fft(*).conjugate() due to the
    choice of sign convention.  The normalization was
    initially chosen to compare with np.fft.fft.

    Used in the construction of `Fz` if `nq = 0`.

    Parameters
    ----------
    nf : int
        Number of frequency channels.
    neta : int
        Number of LoS Fourier modes.

    Returns
    -------
    ExponentArray : np.ndarray of complex floats
        Uniform 1D DFT matrix with shape (nf, neta).
    """
    # Updated for python 3: floor division
    i_f = (np.arange(nf)-nf//2).reshape(-1, 1)
    i_eta = (np.arange(neta)-neta//2).reshape(1, -1)

    # Sign change for consistency, Finv chosen
    # to have + sign to match healvis
    # Updated for python 3: float division is default
    ExponentArray = np.exp(-2.0*np.pi*1j*(i_eta*i_f / nf))

    return ExponentArray / float(nf)


def IDFT_Array_IDFT_1D_WQ(nf, neta, nq, **kwargs):
    """
    Generate a 1D DFT matrix for the FT along the
    LoS axis from eta -> frequency.  Analagous to
    IDFT_Array_IDFT_1D with the exception that this
    function includes the quadratic mode terms used
    for modeling power on spectral scales larger
    than the bandwidth.

    Used in the construction of `Fz` if `nq > 0`.

    Parameters
    ----------
    nf : int
        Number of frequency channels.
    neta : int
        Number of LoS Fourier modes.
    nq : int
        Number of quadratic modes.

    Returns
    -------
    ExponentArray : np.ndarray of complex floats
        1D DFT matrix with shape (nf, neta + nq).
    """

    # ===== Defaults =====
    default_npl = 0
    default_nu_min_MHz = (163.0-4.0)
    default_channel_width_MHz = 0.2
    default_beta = 2.63

    # ===== Inputs =====
    npl = kwargs.pop(
        'npl', default_npl)
    nu_min_MHz = kwargs.pop(
        'nu_min_MHz', default_nu_min_MHz)
    channel_width_MHz = kwargs.pop(
        'channel_width_MHz', default_channel_width_MHz)
    beta = kwargs.pop(
        'beta', default_beta)

    # Updated for python 3: floor division
    i_f = (np.arange(nf) - nf//2).reshape(-1, 1)
    # Updated for python 3: floor division
    i_eta = (np.arange(neta) - neta//2).reshape(1, -1)

    # Sign change for consistency, Finv chosen
    # to have + sign to match healvis
    # Updated for python 3: float division is default
    ExponentArray = np.exp(-2.0*np.pi*1j*(i_eta*i_f / nf))
    # Updated for python 3: float division is default
    ExponentArray /= nf

    quadratic_array = quadratic_array_linear_plus_quad_modes_only_v2(
        nf, nq, npl=npl, nu_min_MHz=nu_min_MHz,
        channel_width_MHz=channel_width_MHz, beta=beta)

    Exponent_plus_quadratic_array = np.hstack(
        (ExponentArray, quadratic_array.T)
        )
    return Exponent_plus_quadratic_array.T


def idft_array_idft_1d_sh(
        nf, neta, nq_sh, npl_sh,
        fit_for_shg_amps=False,
        nu_min_MHz=None,
        channel_width_MHz=None,
        beta=None
        ):
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
        Number of LoS Fourier modes.
    nq_sh : int
        Number of quadratic modes.
    npl_sh : int
        Number of power law modes.
    fit_for_shg_amps : boolean
        If true, include pixels in DFT matrix.  Otherwise,
        only model large spectral scale structure.
    nu_min_MHz : float, optional
        Minimum frequency channel bin center in MHz. Required if `nq_sh` > 0.
    channel_width_MHz : float, optional
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
        lssm_sh = quadratic_array_linear_plus_quad_modes_only_v2(
            nf, nq_sh, npl=npl_sh, nu_min_MHz=nu_min_MHz,
            channel_width_MHz=channel_width_MHz, beta=beta)
        idft_array_sh = np.hstack([idft_array_sh, lssm_sh.T])

    return idft_array_sh


# Gridding matrix functions
def calc_vis_selection_numbers(nu, nv):
    required_chan_order = np.arange(nu*nv).reshape(nv, nu)
    visibility_spectrum_order = required_chan_order.T
    # Updated for python 3: floor division
    r = np.sqrt(
        (np.arange(nu) - nu//2).reshape(-1, 1)**2.
        + (np.arange(nv) - nv//2).reshape(1, -1)**2.
        )
    # No values should be masked if the mean is included
    non_excluded_values_mask = r >= 0.0
    visibility_spectrum_order_ZM =\
        visibility_spectrum_order[non_excluded_values_mask]
    grab_order_for_vis_spectrum_ordered_to_chan_ordered_ZM =\
        visibility_spectrum_order_ZM.argsort()
    return grab_order_for_vis_spectrum_ordered_to_chan_ordered_ZM


def calc_vis_selection_numbers_ZM(nu, nv):
    required_chan_order = np.arange(nu*nv).reshape(nu, nv)
    visibility_spectrum_order = required_chan_order.T
    # Updated for python 3: floor division
    r = np.sqrt(
        (np.arange(nu) - nu // 2).reshape(-1, 1) ** 2.
        + (np.arange(nv) - nv // 2).reshape(1, -1) ** 2.
        )
    # True for everything other than the central pixel (note r == r.T)
    non_excluded_values_mask = r > 0.5
    visibility_spectrum_order_ZM =\
        visibility_spectrum_order[non_excluded_values_mask]
    grab_order_for_vis_spectrum_ordered_to_chan_ordered_ZM =\
        visibility_spectrum_order_ZM.argsort()
    return grab_order_for_vis_spectrum_ordered_to_chan_ordered_ZM


def calc_vis_selection_numbers_SH(
        nu, nv, U_oversampling_Factor=1.0, V_oversampling_Factor=1.0):
    required_chan_order = np.arange(nu*nv).reshape(nu, nv)
    visibility_spectrum_order = required_chan_order.T
    # Updated for python 3: floor division
    r = np.sqrt(
        (np.arange(nu) - nu // 2).reshape(-1, 1) ** 2.
        + (np.arange(nv) - nv // 2).reshape(1, -1) ** 2.
        )
    # True for everything other than the central 9 pix
    non_excluded_values_mask = r > 1.5
    visibility_spectrum_order_ZM_coarse_grid =\
        visibility_spectrum_order[non_excluded_values_mask]
    grab_order_for_vis_spectrum_ordered_to_chan_ordered_ZM_coarse_grid =\
        visibility_spectrum_order_ZM_coarse_grid.argsort()
    grab_order_for_vis_spectrum_ordered_to_chan_ordered_ZM_SH_grid =\
        calc_vis_selection_numbers_ZM(
            3*U_oversampling_Factor, 3*V_oversampling_Factor)
    return grab_order_for_vis_spectrum_ordered_to_chan_ordered_ZM_coarse_grid,\
           grab_order_for_vis_spectrum_ordered_to_chan_ordered_ZM_SH_grid


def generate_gridding_matrix_vis_ordered_to_chan_ordered(
        nu, nv, nf, exclude_mean=True):
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

    Returns
    -------
    gridding_matrix_vis_ordered_to_chan_ordered : np.ndarray
        Array containing the mapping from visibility ordering to channel
        ordering.
    """
    nuv = nu*nv - exclude_mean
    vis_grab_order = np.arange(nuv)
    vals_per_chan = vis_grab_order.size

    gridding_matrix_vis_ordered_to_chan_ordered = np.zeros(
        [vals_per_chan*nf, vals_per_chan*nf]
        )
    for i in range(nf):
        for j, vis_grab_val in enumerate(vis_grab_order):
            row_number = (i*vals_per_chan) + j
            # Pixel to grab from vis-ordered vector
            # and place as next chan-ordered value
            grid_pix = i + vis_grab_val*nf
            gridding_matrix_vis_ordered_to_chan_ordered[
                    row_number, grid_pix
                ] = 1

    return gridding_matrix_vis_ordered_to_chan_ordered


def generate_gridding_matrix_vis_ordered_to_chan_ordered_ZM(nu, nv, nf):
    if p.fit_for_monopole:
        vis_grab_order = calc_vis_selection_numbers(nu, nv)
    else:
        vis_grab_order = calc_vis_selection_numbers_ZM(nu, nv)
    vals_per_chan = vis_grab_order.size

    gridding_matrix_vis_ordered_to_chan_ordered = np.zeros(
        [vals_per_chan*(nf-1), vals_per_chan*(nf-1)]
        )
    for i in range(nf-1):
        for j, vis_grab_val in enumerate(vis_grab_order):
            row_number = (i*vals_per_chan) + j
            # Pixel to grab from vis-ordered vector
            # and place as next chan-ordered value
            grid_pix = i + vis_grab_val*(nf-1)
            # print(i, j, vis_grab_val, row_number, grid_pix)
            gridding_matrix_vis_ordered_to_chan_ordered[
                    row_number, grid_pix
                ] = 1
    return gridding_matrix_vis_ordered_to_chan_ordered


def generate_gridding_matrix_vis_ordered_to_chan_ordered_WQ(nu,nv,nf):
    """
        Re-order matrix from vis-ordered to chan-ordered and place
        Fourier modes at the top and quadratic modes at the bottom.
    """
    if p.fit_for_monopole:
        vis_grab_order = calc_vis_selection_numbers(nu, nv)
    else:
        vis_grab_order = calc_vis_selection_numbers_ZM(nu, nv)
    vals_per_chan = vis_grab_order.size
    Fourier_vals_per_chan = vis_grab_order.size
    quadratic_vals_per_chan = 2

    gridding_matrix_vis_ordered_to_chan_ordered = np.zeros(
        [vals_per_chan*(nf+2), vals_per_chan*(nf+2)]
        )
    for i in range(nf):
        for j, vis_grab_val in enumerate(vis_grab_order):
            row_number = (i*Fourier_vals_per_chan) + j
            # pixel to grab from vis-ordered vector
            # and place as next chan-ordered value
            grid_pix = i + vis_grab_val*(nf+2)
            # print(i, j, vis_grab_val, row_number, grid_pix)
            gridding_matrix_vis_ordered_to_chan_ordered[
                    row_number, grid_pix
                ] = 1
    for j, vis_grab_val in enumerate(vis_grab_order):
        for i in range(2):
            # Place quadratic modes after all of the
            # Fourier modes in the resulting vector
            n_fourier_modes = nf*Fourier_vals_per_chan
            row_number = n_fourier_modes + j*quadratic_vals_per_chan + i
            # Pixel to grab from vis-ordered vector
            # and place as next chan-ordered value
            grid_pix = nf + i + vis_grab_val*(nf+2)
            # print(i, j, vis_grab_val, row_number, grid_pix)
            gridding_matrix_vis_ordered_to_chan_ordered[
                    row_number, grid_pix
                ] = 1
    return gridding_matrix_vis_ordered_to_chan_ordered


def Calc_Coords_High_Res_Im_to_Large_uv(
        i_x_AV, i_y_AV, i_u_AV, i_v_AV,
        X_oversampling_Factor=1.0, Y_oversampling_Factor=1.0):

    # Updated for python 3: float division is default
    # Y_oversampling_Factor = float(Y_oversampling_Factor)
    # X_oversampling_Factor = float(X_oversampling_Factor)

    # Keeps xy-plane size constant and oversampled
    # rather than DFTing from a larger xy-plane
    i_y_AV = i_y_AV / Y_oversampling_Factor
    i_x_AV = i_x_AV / X_oversampling_Factor

    return i_x_AV, i_y_AV, i_u_AV, i_v_AV


def Calc_Coords_Large_Im_to_High_Res_uv(
        i_x_AV, i_y_AV, i_u_AV, i_v_AV,
        U_oversampling_Factor=1.0, V_oversampling_Factor=1.0):

    # Keeps uv-plane size constant and oversampled
    # rather than DFTing to a larger uv-plane
    i_v_AV = i_v_AV / V_oversampling_Factor
    i_u_AV = i_u_AV / U_oversampling_Factor

    return i_x_AV, i_y_AV, i_u_AV, i_v_AV


def Restore_Centre_Pixel(Array, MeanVal=0.0):
    # Updated for python 3: floor division
    Restored_Array = np.insert(Array, [Array.size//2], [MeanVal])
    return Restored_Array


def Delete_Centre_Pix(Array):
    # Updated for python 3: floor division
    Array = np.delete(Array, [Array.size//2])
    return Array


def Calc_Indices_Centre_3x3_Grid(GridSize):
    GridLength = int(GridSize**0.5)

    LenX = LenY = GridLength

    GridIndex = np.arange(LenX*LenY).reshape(LenX, LenY)
    Mask = zeros(LenX*LenY).reshape(LenX, LenY)
    # Updated for python 3: floor division
    Mask[
    len(Mask)//2 - 1 : len(Mask)//2 + 2,
    len(Mask[0])//2 - 1 : len(Mask[0])//2 + 2
    ] = 1
    MaskOuterPoints = Mask.astype('bool')

    return GridIndex, MaskOuterPoints


def Delete_Centre_3x3_Grid(Array):
    GridSize = Array.size
    GridIndex, MaskOuterPoints = Calc_Indices_Centre_3x3_Grid(GridSize)
    OuterArray = np.delete(Array, GridIndex[MaskOuterPoints])
    return OuterArray


def N_is_Odd(N):
    return N%2


def Calc_Indices_Centre_NxN_Grid(GridSize, N):
    GridLength = int(GridSize**0.5)
    LenX = LenY = GridLength

    GridIndex = np.arange(LenX*LenY).reshape(LenX, LenY)
    Mask = zeros(LenX*LenY).reshape(LenX, LenY)
    if N_is_Odd(N):
        # Updated for python 3: floor division
        Mask[
        len(Mask)//2 - (N//2) : len(Mask)//2 + (N//2 + 1),
        len(Mask[0])//2 - (N//2) : len(Mask[0])//2 + (N//2 + 1)
        ] = 1
    else:
        # Updated for python 3: floor division
        Mask[
        len(Mask) // 2 - (N // 2): len(Mask) // 2 + (N // 2),
        len(Mask[0]) // 2 - (N // 2): len(Mask[0]) // 2 + (N // 2)
        ] = 1
    MaskOuterPoints=Mask.astype('bool')

    return GridIndex, MaskOuterPoints


def Obtain_Centre_NxN_Grid(Array, N):
    GridSize = Array.size
    GridIndex, MaskOuterPoints = Calc_Indices_Centre_NxN_Grid(GridSize, N)
    Centre_NxN_Grid = Array.flatten()[GridIndex[MaskOuterPoints]]
    return Centre_NxN_Grid


def Restore_Centre_3x3_Grid(Array, MeanVal=0.0):
    LenRestoredArray = Array.size + 9

    GridSize = LenRestoredArray
    GridIndex, MaskOuterPoints = Calc_Indices_Centre_3x3_Grid(GridSize)

    CurrentPointsIndex = GridIndex[np.where(np.logical_not(MaskOuterPoints))]
    RestoredPointsIndex = GridIndex[np.where((MaskOuterPoints))]

    ConcatIndices = np.concatenate((CurrentPointsIndex, RestoredPointsIndex))
    SortedIndices = ConcatIndices.argsort()

    Restored_Array_Unsorted = np.append(Array, [MeanVal]*9)
    Restored_Array = Restored_Array_Unsorted[SortedIndices]

    return Restored_Array


def Restore_Centre_NxN_Grid(Array1, Array2, N):
    LenRestoredArray = Array1.size + N*N

    GridSize = LenRestoredArray
    GridIndex, MaskOuterPoints = Calc_Indices_Centre_NxN_Grid(GridSize, N)

    CurrentPointsIndex = GridIndex[np.where(np.logical_not(MaskOuterPoints))]
    RestoredPointsIndex = GridIndex[np.where((MaskOuterPoints))]

    ConcatIndices = np.concatenate((CurrentPointsIndex, RestoredPointsIndex))
    SortedIndices = ConcatIndices.argsort()

    Restored_Array_Unsorted = np.append(Array1, Array2)
    Restored_Array = Restored_Array_Unsorted[SortedIndices]

    return Restored_Array


def Generate_Combined_Coarse_plus_Subharmic_uv_grids(
        nu, nv, nx, ny,
        X_oversampling_Factor, Y_oversampling_Factor,
        U_oversampling_Factor, V_oversampling_Factor,
        ReturnSeparateCoarseandSHarrays=False):
    # U_oversampling_Factor and V_oversampling_Factor are the
    # factors by which the subharmonic grid is oversampled
    # relative to the coarse grid.

    nu_SH = 3 * int(U_oversampling_Factor)
    nv_SH = 3 * int(V_oversampling_Factor)
    n_SH = (nu_SH*nv_SH) - 1

    i_x_AV_C, i_y_AV_C, i_u_AV_C, i_v_AV_C =\
        Produce_Coordinate_Arrays_ZM_Coarse(nu, nv, nx, ny)
    i_x_AV_SH, i_y_AV_SH, i_u_AV_SH, i_v_AV_SH =\
        Produce_Coordinate_Arrays_ZM_SH(nu_SH, nv_SH, nx, ny)

    if U_oversampling_Factor != 1.0:
        i_x_AV_SH, i_y_AV_SH, i_u_AV_SH, i_v_AV_SH =\
            Calc_Coords_Large_Im_to_High_Res_uv(
                i_x_AV_SH, i_y_AV_SH, i_u_AV_SH, i_v_AV_SH,
                U_oversampling_Factor, V_oversampling_Factor)
    if X_oversampling_Factor != 1.0:
        i_x_AV_C, i_y_AV_C, i_u_AV_C, i_v_AV_C =\
            Calc_Coords_High_Res_Im_to_Large_uv(
                i_x_AV_C, i_y_AV_C, i_u_AV_C, i_v_AV_C,
                X_oversampling_Factor, Y_oversampling_Factor)

    # Combine Coarse and subharmic uv-grids.
    i_u_AV = np.concatenate((i_u_AV_C, i_u_AV_SH))
    i_v_AV = np.concatenate((i_v_AV_C, i_v_AV_SH))

    i_x_AV = i_x_AV_C
    i_y_AV = i_y_AV_C

    if not ReturnSeparateCoarseandSHarrays:
        return i_u_AV, i_v_AV, i_x_AV, i_y_AV
    else:
        return i_u_AV_C, i_u_AV_SH, i_v_AV_C, i_v_AV_SH, i_x_AV, i_y_AV
