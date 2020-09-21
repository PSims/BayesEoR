import numpy as np
from subprocess import os
from astropy_healpix import HEALPix

import BayesEoR.Params.params as p

use_foreground_cube = True
# use_foreground_cube = False


def generate_data_from_loaded_EoR_cube_v2d0(
        nu, nv, nx, ny, nf, neta, nq, k_x, k_y, k_z, Show, chan_selection,
        EoR_npz_path=('/users/psims/EoR/EoR_simulations/'
                      '21cmFAST_512MPc_512pix_128pix/Fits/21cm_z10d2_mK.npz')):

    print('Using use_EoR_cube data')
    # Replace Gaussian signal with EoR cube
    scidata1 = np.load(EoR_npz_path)['arr_0']

    base_dir = 'Plots'
    save_dir = base_dir+'/Likelihood_v1d75_3D_ZM/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    import numpy
    axes_tuple = (1, 2)
    if chan_selection == '0_38_':
        vfft1 = np.fft.ifftshift(scidata1[0:38]-scidata1[0].mean() + 0j,
                                    axes=axes_tuple)
    elif chan_selection == '38_76_':
        vfft1 = np.fft.ifftshift(scidata1[38:76]-scidata1[0].mean() + 0j,
                                    axes=axes_tuple)
    elif chan_selection == '76_114_':
        vfft1 = np.fft.ifftshift(scidata1[76:114]-scidata1[0].mean() + 0j,
                                    axes=axes_tuple)
    else:
        vfft1 = np.fft.ifftshift(scidata1[0:nf]-scidata1[0].mean() + 0j,
                                    axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    vfft1 = np.fft.fftn(vfft1, axes=axes_tuple)
    vfft1 = np.fft.fftshift(vfft1, axes=axes_tuple)

    sci_f, sci_v, sci_u = vfft1.shape
    # Updated for python 3: floor division
    sci_v_centre = sci_v//2
    # Updated for python 3: floor division
    sci_u_centre = sci_u//2
    # Updated for python 3: floor division
    vfft1_subset = vfft1[0 : nf,
                         sci_u_centre - nu//2 : sci_u_centre + nu//2 + 1,
                         sci_v_centre - nv//2 : sci_v_centre + nv//2 + 1]
    # s_before_ZM = vfft1_subset.flatten() / vfft1[0].size**0.5
    s_before_ZM = vfft1_subset.flatten()
    ZM_vis_ordered_mask = np.ones(nu*nv*nf)
    # Updated for python 3: floor division
    ZM_vis_ordered_mask[nf*((nu*nv)//2) : nf*((nu*nv)//2 + 1)] = 0
    ZM_vis_ordered_mask = ZM_vis_ordered_mask.astype('bool')
    ZM_chan_ordered_mask = ZM_vis_ordered_mask.reshape(-1, neta+nq).T.flatten()
    s = s_before_ZM[ZM_chan_ordered_mask]
    abc = s

    return s, abc, scidata1


def generate_EoR_signal_instrumental_im_2_vis(
        nu, nv, nx, ny, nf, neta, nq, k_x, k_y, k_z,
        Finv, Show, chan_selection, masked_power_spectral_modes,
        mod_k, EoR_npz_path):

    # print('Using use_EoR_cube data')
    ###
    # Replace Gaussian signal with EoR cube
    ###
    # scidata1 = np.load(EoR_npz_path)['arr_0']
    #
    # #Overwrite EoR cube with white noise
    # # np.random.seed(21287254)
    # # np.random.seed(123)
    # # scidata1 = np.random.normal(0,scidata1.std()*1.,[nf, nu, nv])*0.5
    # scidata1 = np.random.normal(0, scidata1.std() * 1.0, (nf, nu, nv))
    # print('EoR cube (white noise) stddev = {} mK'.format(scidata1.std()))
    #
    #
    # # if not p.fit_for_monopole:
    # # 	d_im = scidata1.copy()
    # # 	for i_chan in range(len(d_im)):
    # # 		d_im[i_chan] = d_im[i_chan]-d_im[i_chan].mean()
    # # scidata1 = d_im.copy()
    #
    # axes_tuple = (0, 1, 2)
    # scidata1_kcube = np.fft.ifftshift(
    # 	scidata1[0:nf]-scidata1[0:nf].mean() + 0j, axes=axes_tuple)
    # # FFT (python pre-normalises correctly! -- see
    # # parsevals theorem for discrete fourier transform.)
    # scidata1_kcube = np.fft.fftn(scidata1_kcube, axes=axes_tuple)
    # scidata1_kcube = np.fft.fftshift(scidata1_kcube, axes=axes_tuple)
    #
    # sci_f, sci_v, sci_u = scidata1_kcube.shape
    # # Updated for python 3: floor division
    # sci_v_centre = sci_v//2
    # # Updated for python 3: floor division
    # sci_u_centre = sci_u//2
    # scidata1_kcube_subset = scidata1_kcube[
    # 						0: nf,
    # 						sci_u_centre - nu // 2: sci_u_centre + nu // 2 + 1,
    # 						sci_v_centre - nv // 2: sci_v_centre + nv // 2 + 1]
    #
    # # if not p.use_intrinsic_noise_fitting:
    # # 	# Zero modes that are not currently fit for until intrinsic
    # #     # noise fitting (which models these terms) has been implemented
    # #
    # # 	print('No intrinsic noise fitting model. '
    # #           'Low-pass filtering the EoR cube')
    # # 	Hermitian_small_spacial_scale_mask = np.zeros(
    # #         scidata1_kcube_subset.shape)
    # # 	Hermitian_small_spacial_scale_mask[0] = 1 #Nyquist mode
    # # 	Hermitian_small_spacial_scale_mask[1] = 1 #2nd highest freq
    # # 	# Hermitian_small_spacial_scale_mask[2] = 1 #3nd highest freq
    # # 	# Hermitian_small_spacial_scale_mask[-2] = 1 #3nd highest freq
    # # 	Hermitian_small_spacial_scale_mask[-1] = 1 #2nd highest freq
    # # 	scidata1_kcube_subset[
    # #             Hermitian_small_spacial_scale_mask.astype('bool')
    # #         ] = 0.0
    # # else:
    # # 	print('Intrinsic noise fitting model included. '
    # #           'Using full EoR cube (no low-pass filter)')
    #
    # axes_tuple = (0, 1, 2)
    # scidata1_subset = np.fft.ifftshift(
    # 	scidata1_kcube_subset + 0j, axes=axes_tuple)
    # # FFT (python pre-normalises correctly! -- see
    # # parsevals theorem for discrete fourier transform.)
    # scidata1_subset = np.fft.ifftn(scidata1_subset, axes=axes_tuple)
    # scidata1_subset = np.fft.fftshift(scidata1_subset, axes=axes_tuple)

    print('Generating white noise signal...')
    # This function needs to be updated to use astropy.healpix info
    # to generate a sky model
    # Make a cube in the subset space using a stddev
    # scaled to match the healvis sky realizations
    hp = HEALPix(p.nside)
    np.random.seed(12346)
    rms_mK = 12.951727094335597 # rms of healvis sky model for nside=512
    # Scale rms based on pixel area
    if not p.nside == 512:
        scaling = p.nside / 512
        rms_mK *= scaling
    white_noise_sky = np.random.normal(0.0, rms_mK, (nf, hp.npix))

    for i_f in range(nf):
        white_noise_sky[i_f] -= white_noise_sky[i_f].mean()
    scidata1 = white_noise_sky.copy()
    s = np.dot(Finv, white_noise_sky.flatten())
    abc = s

    return s, abc, scidata1
