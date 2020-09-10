
###
# Imports
###
import numpy as np
import numpy
from numpy import arange, shape, log10, pi
import scipy
import pylab
from scipy.linalg import block_diag
from subprocess import os
import sys
from scipy import stats
from pdb import set_trace as brk

from BayesEoR.SimData import\
    GenerateForegroundCube,\
    update_Tb_experimental_std_K_to_correct_for_normalisation_resolution
from BayesEoR.Utils import\
    DataUnitConversionmKAndJyPerPix, WriteDataToFits
from BayesEoR.Utils import ExtractDataFrom21cmFASTCube

import BayesEoR.Params.params as p

use_foreground_cube = True
# use_foreground_cube = False

def generate_Jelic_cube(
        nu, nv, nx, ny, nf, neta, nq, k_x, k_y, k_z, Show,
        beta_experimental_mean, beta_experimental_std, gamma_mean, gamma_sigma,
        Tb_experimental_mean_K, Tb_experimental_std_K,
        nu_min_MHz, channel_width_MHz, **kwargs):

    # ===== Defaults =====
    default_generate_additional_extrapolated_HF_foreground_cube = False
    default_HF_nu_min_MHz = 225
    default_fits_storage_dir = 'fits_storage/Jelic/'
    default_HF_nu_min_MHz_array = [205,215,225]
    default_simulation_FoV_deg = 12.0
    default_simulation_resolution_deg = 12.0/127
    default_random_seed = 3142
    # Size of EoR cube foreground simulation should match
    # (used when calculating fits header variables)
    default_cube_side_Mpc = 2048.0
    # Redshift of EoR cube foreground simulation should match
    # (used when calculating fits header variables)
    default_redshift = 7.6

    # ===== Inputs =====
    generate_additional_extrapolated_HF_foreground_cube = kwargs.pop(
        'generate_additional_extrapolated_HF_foreground_cube',
        default_generate_additional_extrapolated_HF_foreground_cube)
    HF_nu_min_MHz = kwargs.pop(
        'HF_nu_min_MHz',
        default_HF_nu_min_MHz)
    fits_storage_dir = kwargs.pop(
        'fits_storage_dir',
        default_fits_storage_dir)
    HF_nu_min_MHz_array = kwargs.pop(
        'HF_nu_min_MHz_array',
        default_fits_storage_dir)
    simulation_FoV_deg = kwargs.pop(
        'simulation_FoV_deg',
        default_simulation_FoV_deg)
    simulation_resolution_deg = kwargs.pop(
        'simulation_resolution_deg',
        default_simulation_resolution_deg)
    random_seed = kwargs.pop(
        'random_seed',
        default_random_seed)
    cube_side_Mpc = kwargs.pop(
        'random_seed',
        default_cube_side_Mpc)
    redshift = kwargs.pop(
        'random_seed',
        default_redshift)

    n_sim_pix = int(simulation_FoV_deg/simulation_resolution_deg + 0.5)

    low_res_to_high_res_std_conversion_factor =\
        update_Tb_experimental_std_K_to_correct_for_normalisation_resolution(
            Tb_experimental_std_K,
            simulation_FoV_deg,
            simulation_resolution_deg)
    Tb_experimental_std_K = (Tb_experimental_std_K
                             * low_res_to_high_res_std_conversion_factor)

    GFC = GenerateForegroundCube(
        nu, nv, neta, nq, beta_experimental_mean, beta_experimental_std,
        gamma_mean, gamma_sigma, Tb_experimental_mean_K, Tb_experimental_std_K,
        nu_min_MHz, channel_width_MHz, random_seed=random_seed)

    Tb_nu, A, beta, Tb, nu_array_MHz =\
        GFC.generate_normalised_Tb_A_and_beta_fields(
            n_sim_pix, n_sim_pix, n_sim_pix, n_sim_pix, nf, neta, nq)
    # Tb_nu, A, beta, Tb, nu_array_MHz =\
    # 	GFC.generate_normalised_Tb_A_and_beta_fields(
    # 		513, 513, 513, 513, nf, neta, nq)
    # Tb_nu, A, beta, Tb, nu_array_MHz =\
    # 	GFC.generate_normalised_Tb_A_and_beta_fields(
    # 		nu, nv, nx, ny, nf, neta, nq)
    Tb_nu2 = np.array(
        [Tb_nu[0] * (nu_array_MHz[i]/nu_array_MHz[0])**-beta_experimental_mean
         for i in range(len(nu_array_MHz))]
        )

    if generate_additional_extrapolated_HF_foreground_cube:
        # HF_nu_min_MHz = 225
        # HF_nu_min_MHz_array = [205, 215, 225]
        for HF_nu_min_MHz_i in range(len(HF_nu_min_MHz_array)):
            HF_nu_min_MHz = HF_nu_min_MHz_array[HF_nu_min_MHz_i]
            HF_Tb_nu = generate_additional_HF_Jelic_cube(
                A, HF_nu_min_MHz, beta, fits_storage_dir,
                nu, nv, nx, ny, nf, neta, nq,
                k_x, k_y, k_z, Show,
                beta_experimental_mean, beta_experimental_std,
                gamma_mean, gamma_sigma,
                Tb_experimental_mean_K, Tb_experimental_std_K,
                nu_min_MHz, channel_width_MHz,
                cube_side_Mpc=cube_side_Mpc, redshift=redshift)
    else:
        HF_Tb_nu = []

    beta = A = Tb = []

    ED = ExtractDataFrom21cmFASTCube(plot_data=False)
    bandwidth_MHz, central_frequency_MHz, output_21cmFast_box_width_deg =\
        ED.calculate_box_size_in_degrees_and_MHz(cube_side_Mpc, redshift)

    DUC = DataUnitConversionmKAndJyPerPix()
    Data_mK = Tb_nu * 1.e3

    Channel_Frequencies_Array_Hz = nu_array_MHz * 1.e6
    Pixel_Height_rad = (output_21cmFast_box_width_deg * (np.pi/180.)
                        / Tb_nu.shape[1])
    DUC.convert_from_mK_to_Jy_per_pix(
        Data_mK, Channel_Frequencies_Array_Hz, Pixel_Height_rad)
    Data_Jy_per_Pixel = ED.convert_from_mK_to_Jy_per_pix(
        Data_mK, Channel_Frequencies_Array_Hz, Pixel_Height_rad)

    output_fits_file_name =\
        'Jelic_GDSE_cube_{:d}MHz.fits'.format(int(nu_min_MHz))
    output_fits_path1 = fits_storage_dir + output_fits_file_name
    output_fits_path2 = (
            fits_storage_dir
            + '/ZNPS{:d}/'.format(int(nu_min_MHz))
            + output_fits_file_name)
    print(output_fits_path1, '\n'+ output_fits_path2)
    WD2F = WriteDataToFits()
    WD2F.write_data(Data_Jy_per_Pixel, output_fits_path1,
                    Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)
    WD2F.write_data(Data_Jy_per_Pixel, output_fits_path2,
                    Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)

    output_fits_file_name =\
        'Jelic_GDSE_cube_{:d}MHz_mK.fits'.format(int(nu_min_MHz))
    output_fits_path1 = fits_storage_dir + output_fits_file_name
    output_fits_path2 = (
            fits_storage_dir
            + '/ZNPS{:d}/'.format(int(nu_min_MHz))
            + output_fits_file_name)
    print(output_fits_path1, '\n'+ output_fits_path2)
    WD2F = WriteDataToFits()
    WD2F.write_data(Data_mK, output_fits_path1,
                    Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)
    WD2F.write_data(Data_mK, output_fits_path2,
                    Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)

    import pylab
    pylab.figure()
    pylab.imshow(Tb_nu[0], cmap=pylab.cm.jet)
    pylab.colorbar(fraction=(0.315*0.15), pad=0.01)
    pylab.figure()
    pylab.imshow(Tb_nu[-1], cmap=pylab.cm.jet)
    pylab.colorbar(fraction=(0.315*0.15), pad=0.01)
    if Show: pylab.show()

    import pylab
    pylab.errorbar(log10(nu_array_MHz), log10(Tb_nu[:, 0, 0]))
    pylab.errorbar(log10(nu_array_MHz), log10(Tb_nu[:, 1, 1]))
    pylab.errorbar(log10(nu_array_MHz), log10(Tb_nu[:, 2, 2]))
    if Show: pylab.show()

    pylab.close('all')
    # Show=False

    # Inspect quadratic residuals
    quad_coeffs = np.polyfit(nu_array_MHz, Tb_nu[:, 0, 0], 2)
    quad_fit = (quad_coeffs[0]*nu_array_MHz**2.
                + quad_coeffs[1]*nu_array_MHz**1.
                + quad_coeffs[2]*nu_array_MHz**0.)
    residuals = Tb_nu[:, 0, 0] - quad_fit

    base_dir = 'Plots'
    save_dir = base_dir+'/Likelihood_v1d75_3D_ZM/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    import numpy
    axes_tuple = (0,)
    res_fft = numpy.fft.ifftshift(residuals + 0j, axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    res_fft = numpy.fft.fftn(res_fft, axes=axes_tuple)
    res_fft = numpy.fft.fftshift(res_fft, axes=axes_tuple)

    fig, ax = pylab.subplots(nrows=1, ncols=1, figsize=(20, 20))
    ax.errorbar(
        log10(nu_array_MHz)[np.fft.fftshift(np.fft.fftfreq(nf)) != 0],
        log10(abs(res_fft))[np.fft.fftshift(np.fft.fftfreq(nf)) != 0],
        color='red', fmt='-')
    # ax.errorbar(
    # 	log10(nu_array_MHz)[np.fft.fftshift(np.fft.fftfreq(38)) != 0],
    # 	log10(abs(res_fft))[np.fft.fftshift(np.fft.fftfreq(38)) != 0],
    # 	color='red', fmt='-')
    ax.set_ylabel('log(Amplitude)', fontsize=20)
    ax.set_xlabel('log-fftfreq', fontsize=20)
    ax.tick_params(labelsize=20)
    fig.savefig(save_dir+'foreground_quadsub_residuals_ffted.png')
    if Show:fig.show()

    print('Using use_foreground_cube data')
    # Replace Gaussian signal with foreground cube
    # Tb_nu_mK = Tb_nu * 1.e2
    Tb_nu_mK = Tb_nu * 1.e3
    scidata1 = Tb_nu_mK
    # scidata1 = random_quad[0:nf, 0:nu, 0:nv]

    base_dir = 'Plots'
    save_dir = base_dir+'/Likelihood_v1d75_3D_ZM/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    import numpy
    axes_tuple = (1, 2)
    vfft1 = numpy.fft.ifftshift(scidata1[0:nf]-scidata1[0].mean() + 0j,
                                axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    vfft1 = numpy.fft.fftn(vfft1, axes=axes_tuple)
    vfft1 = numpy.fft.fftshift(vfft1, axes=axes_tuple)

    import pylab
    pylab.figure()
    pylab.imshow(abs(vfft1[0]), cmap=pylab.cm.jet)
    pylab.colorbar(fraction=(0.315*0.15), pad=0.01)
    if Show: pylab.show()

    pylab.close('all')

    sci_f, sci_v, sci_u = vfft1.shape
    # Updated for python 3: floor division
    sci_v_centre = sci_v//2
    # Updated for python 3: floor division
    sci_u_centre = sci_u//2
    # Updated for python 3: floor division
    vfft1_subset = vfft1[0 : nf,
                         sci_u_centre - nu//2 : sci_u_centre + nu//2 + 1,
                         sci_v_centre - nv//2 : sci_v_centre + nv//2 + 1]
    s_before_ZM = vfft1_subset.flatten() / vfft1[0].size**0.5
    ZM_vis_ordered_mask = np.ones(nu*nv*nf)
    # Updated for python 3: floor division
    ZM_vis_ordered_mask[nf*((nu*nv)//2)  : nf*((nu*nv)//2 + 1)] = 0
    ZM_vis_ordered_mask = ZM_vis_ordered_mask.astype('bool')
    ZM_chan_ordered_mask = ZM_vis_ordered_mask.reshape(-1, neta+nq).T.flatten()
    s = s_before_ZM[ZM_chan_ordered_mask]

    fg = s

    # Inspect quadratic residuals in uv
    dat1 = fg.reshape(-1, nu*nv - 1)[:, 0].real[::-1]
    quad_coeffs = np.polyfit(nu_array_MHz, dat1, 2)
    quad_fit = (quad_coeffs[0]*nu_array_MHz**2.
                + quad_coeffs[1]*nu_array_MHz**1.
                + quad_coeffs[2]*nu_array_MHz**0.)
    residuals = dat1 - quad_fit

    dat2 = fg.reshape(-1, nu*nv - 1)[:, 0].imag[::-1]
    quad_coeffs2 = np.polyfit(nu_array_MHz, dat2, 2)
    quad_fit2 = (quad_coeffs2[0]*nu_array_MHz**2.
                 + quad_coeffs2[1]*nu_array_MHz**1.
                 + quad_coeffs2[2]*nu_array_MHz**0.)
    residuals2 = dat2 - quad_fit2

    base_dir = 'Plots'
    save_dir = base_dir+'/Likelihood_v1d75_3D_ZM/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig, ax = pylab.subplots(nrows=2, ncols=1, figsize=(20, 20))
    ax[0].errorbar(nu_array_MHz, residuals,  color='red')
    ax[0].errorbar(nu_array_MHz, residuals2,  color='blue', fmt='-')
    ax[0].legend(['Quad residuals real', 'Quad residuals imag'], fontsize=20)
    ax[0].set_ylabel('Amplitude, arbitrary units', fontsize=20)
    ax[0].set_xlabel('Frequency, MHz', fontsize=20)
    ax[1].errorbar(log10(nu_array_MHz),
                   log10(fg.reshape(-1,nu*nv-1)[:,0].real),
                   color='red', fmt='-')
    ax[1].errorbar(log10(nu_array_MHz),
                   log10(fg.reshape(-1,nu*nv-1)[:,0].imag),
                   color='blue', fmt='-')
    ax[1].legend(['real','imag'], fontsize=20)
    ax[1].set_ylabel('log(Amplitude), arbitrary units', fontsize=20)
    ax[1].set_xlabel('log(Frequency), MHz', fontsize=20)
    for axi in ax.ravel(): axi.tick_params(labelsize=20)
    # fig.savefig(save_dir+'foreground_quadsub_residuals.png')
    if Show:fig.show()

    import numpy
    axes_tuple = (0,)
    res_fft = numpy.fft.ifftshift(residuals + 0j, axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    res_fft = numpy.fft.fftn(res_fft, axes=axes_tuple)
    res_fft = numpy.fft.fftshift(res_fft, axes=axes_tuple)

    k_cubed_log_res_fft = (
            log10(abs(res_fft*k_z[:, 0, 0]**3.))
        )[np.fft.fftshift(np.fft.fftfreq(nf)) != 0]

    fig, ax = pylab.subplots(nrows=1, ncols=1, figsize=(20, 20))
    ax.errorbar(log10(nu_array_MHz)[np.fft.fftshift(np.fft.fftfreq(nf)) != 0],
                k_cubed_log_res_fft, color='red', fmt='-')
    ax.set_ylabel('log(Amplitude)', fontsize=20)
    ax.set_xlabel('log-fftfreq', fontsize=20)
    ax.tick_params(labelsize=20)
    # fig.savefig(save_dir+'foreground_quadsub_residuals_ffted.png')
    if Show:
        fig.show()

    a1 = log10(k_z[:,0,0]**3.)[20:] + 4.16
    a2 = (log10(abs(res_fft)))[20:] + 1.33
    print(a1+a2-0.8)

    axes_tuple = (0,)
    res_fft = numpy.fft.ifftshift(residuals + 0j, axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    res_fft = numpy.fft.fftn(res_fft, axes=axes_tuple)
    res_fft = numpy.fft.fftshift(res_fft, axes=axes_tuple)

    k_cubed_log_res_fft = (
            log10(abs(res_fft))
        )[np.fft.fftshift(np.fft.fftfreq(nf)) != 0]
    # k_cubed_log_res_fft = (
    # 		log10(abs(res_fft*k_z[:, 0, 0]**3.))
    # 	)[np.fft.fftshift(np.fft.fftfreq(38)) != 0]

    fig, ax = pylab.subplots(nrows=1, ncols=1, figsize=(20, 20))
    ax.errorbar(log10(nu_array_MHz)[np.fft.fftshift(np.fft.fftfreq(nf)) != 0],
                k_cubed_log_res_fft, color='red', fmt='-')
    # ax.errorbar(log10(nu_array_MHz)[np.fft.fftshift(np.fft.fftfreq(38)) != 0],
    # 			k_cubed_log_res_fft, color='red', fmt='-')
    ax.set_ylabel('log(Amplitude)', fontsize=20)
    ax.set_xlabel('log-fftfreq', fontsize=20)
    ax.tick_params(labelsize=20)
    # fig.savefig(save_dir+'foreground_quadsub_residuals_ffted.png')
    if Show:
        fig.show()

    return fg, s, Tb_nu, beta_experimental_mean, beta_experimental_std,\
           gamma_mean, gamma_sigma,\
           Tb_experimental_mean_K, Tb_experimental_std_K,\
           nu_min_MHz, channel_width_MHz, HF_Tb_nu


def generate_Jelic_cube_instrumental_im_2_vis(
        nu, nv, nx, ny, nf, neta, nq,
        k_x, k_y, k_z, Show,
        beta_experimental_mean, beta_experimental_std,
        gamma_mean, gamma_sigma,
        Tb_experimental_mean_K, Tb_experimental_std_K,
        nu_min_MHz, channel_width_MHz, Finv, **kwargs):

    # ===== Defaults =====
    default_generate_additional_extrapolated_HF_foreground_cube = False
    default_HF_nu_min_MHz = 225
    default_fits_storage_dir = 'fits_storage/Jelic/'
    default_HF_nu_min_MHz_array = [205, 215, 225]
    default_simulation_FoV_deg = 12.0
    default_simulation_resolution_deg = 12.0/127
    default_random_seed = 3142
    # Size of EoR cube foreground simulation should match
    # (used when calculating fits header variables)
    default_cube_side_Mpc = 2048.0
    # Redshift of EoR cube foreground simulation should match
    # (used when calculating fits header variables)
    default_redshift = 7.6

    # ===== Inputs =====
    generate_additional_extrapolated_HF_foreground_cube = kwargs.pop(
        'generate_additional_extrapolated_HF_foreground_cube',
        default_generate_additional_extrapolated_HF_foreground_cube)
    HF_nu_min_MHz = kwargs.pop(
        'HF_nu_min_MHz',
        default_HF_nu_min_MHz)
    fits_storage_dir = kwargs.pop(
        'fits_storage_dir',
        default_fits_storage_dir)
    HF_nu_min_MHz_array = kwargs.pop(
        'HF_nu_min_MHz_array',
        default_fits_storage_dir)
    simulation_FoV_deg = kwargs.pop(
        'simulation_FoV_deg',
        default_simulation_FoV_deg)
    simulation_resolution_deg = kwargs.pop(
        'simulation_resolution_deg',
        default_simulation_resolution_deg)
    random_seed = kwargs.pop(
        'random_seed',
        default_random_seed)
    cube_side_Mpc = kwargs.pop(
        'random_seed',
        default_cube_side_Mpc)
    redshift = kwargs.pop(
        'random_seed',
        default_redshift)

    n_sim_pix = int(simulation_FoV_deg/simulation_resolution_deg + 0.5)

    low_res_to_high_res_std_conversion_factor =\
        update_Tb_experimental_std_K_to_correct_for_normalisation_resolution(
            Tb_experimental_std_K,
            simulation_FoV_deg,
            simulation_resolution_deg)
    Tb_experimental_std_K = (Tb_experimental_std_K
                             * low_res_to_high_res_std_conversion_factor)

    GFC = GenerateForegroundCube(
        nu, nv, neta, nq, beta_experimental_mean, beta_experimental_std,
        gamma_mean, gamma_sigma, Tb_experimental_mean_K, Tb_experimental_std_K,
        nu_min_MHz, channel_width_MHz, random_seed=random_seed)

    Tb_nu, A, beta, Tb, nu_array_MHz =\
        GFC.generate_normalised_Tb_A_and_beta_fields(
            n_sim_pix, n_sim_pix, n_sim_pix, n_sim_pix, nf, neta, nq)

    HF_Tb_nu = []
    beta = A = Tb = []

    ED = ExtractDataFrom21cmFASTCube(plot_data=False)
    bandwidth_MHz, central_frequency_MHz, output_21cmFast_box_width_deg =\
        ED.calculate_box_size_in_degrees_and_MHz(cube_side_Mpc, redshift)

    DUC = DataUnitConversionmKAndJyPerPix()
    Data_mK = Tb_nu * 1.e3

    Channel_Frequencies_Array_Hz = nu_array_MHz * 1.e6
    Pixel_Height_rad = (output_21cmFast_box_width_deg * (np.pi/180.)
                        / Tb_nu.shape[1])
    DUC.convert_from_mK_to_Jy_per_pix(
        Data_mK, Channel_Frequencies_Array_Hz, Pixel_Height_rad)
    Data_Jy_per_Pixel = ED.convert_from_mK_to_Jy_per_pix(
        Data_mK, Channel_Frequencies_Array_Hz, Pixel_Height_rad)

    output_fits_file_name =\
        'Jelic_GDSE_cube_{:d}MHz.fits'.format(int(nu_min_MHz))
    output_fits_path1 = fits_storage_dir + output_fits_file_name
    output_fits_path2 = (fits_storage_dir
                         + '/ZNPS{:d}/'.format(int(nu_min_MHz))
                         + output_fits_file_name)
    print(output_fits_path1, '\n'+ output_fits_path2)
    write_to_file = True
    write_to_file = False
    WD2F = WriteDataToFits()
    if write_to_file:
        WD2F.write_data(
            Data_Jy_per_Pixel, output_fits_path1,
            Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)
        WD2F.write_data(
            Data_Jy_per_Pixel, output_fits_path2,
            Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)

    output_fits_file_name =\
        'Jelic_GDSE_cube_{:d}MHz_mK.fits'.format(int(nu_min_MHz))
    output_fits_path1 = fits_storage_dir + output_fits_file_name
    output_fits_path2 = (fits_storage_dir
                         + '/ZNPS{:d}/'.format(int(nu_min_MHz))
                         + output_fits_file_name)
    print(output_fits_path1, '\n'+ output_fits_path2)
    WD2F = WriteDataToFits()
    if write_to_file:
        WD2F.write_data(
            Data_mK, output_fits_path1,
            Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)
        WD2F.write_data(
            Data_mK, output_fits_path2,
            Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)

    base_dir = 'Plots'
    save_dir = base_dir+'/Likelihood_v1d75_3D_ZM/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    print('Using use_foreground_cube data')
    # Replace Gaussian signal with foreground cube
    # Tb_nu_mK = Tb_nu * 1.e2
    Tb_nu_mK = Tb_nu * 1.e3
    scidata1 = Tb_nu_mK
    # scidata1 = random_quad[0:nf, 0:nu, 0:nv]

    base_dir = 'Plots'
    save_dir = base_dir+'/Likelihood_v1d75_3D_ZM/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    import numpy
    axes_tuple = (1, 2)
    vfft1 = numpy.fft.ifftshift(scidata1[0:nf] + 0j, axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    vfft1 = numpy.fft.fftn(vfft1, axes=axes_tuple)
    vfft1 = numpy.fft.fftshift(vfft1, axes=axes_tuple)

    pylab.close('all')

    sci_f, sci_v, sci_u = vfft1.shape
    # Updated for python 3: floor division
    sci_v_centre = sci_v//2
    # Updated for python 3: floor division
    sci_u_centre = sci_u//2
    # Updated for python 3: floor division
    vfft1_subset = vfft1[0 : nf,
                         sci_u_centre - nu//2 : sci_u_centre + nu//2 + 1,
                         sci_v_centre - nv//2 : sci_v_centre + nv//2 + 1]

    axes_tuple = (1, 2)
    scidata1_subset = numpy.fft.ifftshift(vfft1_subset + 0j, axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    scidata1_subset = numpy.fft.ifftn(scidata1_subset, axes=axes_tuple)
    scidata1_subset = numpy.fft.fftshift(scidata1_subset, axes=axes_tuple)

    s = np.dot(Finv,scidata1_subset.reshape(-1, 1)).flatten()
    abc = s
    fg = s

    return fg, s, Tb_nu, beta_experimental_mean, beta_experimental_std,\
           gamma_mean, gamma_sigma,\
           Tb_experimental_mean_K, Tb_experimental_std_K,\
           nu_min_MHz, channel_width_MHz, HF_Tb_nu


def generate_Jelic_cube_instrumental_im_2_vis_v2d0(
        nu, nv, nx, ny, nf, neta, nq, k_x, k_y, k_z, Show,
        beta_experimental_mean, beta_experimental_std,
        gamma_mean, gamma_sigma,
        Tb_experimental_mean_K, Tb_experimental_std_K,
        nu_min_MHz, channel_width_MHz, Finv, **kwargs):

    # ===== Defaults =====
    default_generate_additional_extrapolated_HF_foreground_cube = False
    default_HF_nu_min_MHz = 225
    default_fits_storage_dir = 'fits_storage/Jelic/'
    default_HF_nu_min_MHz_array = [205,215,225]
    default_simulation_FoV_deg = 12.0
    default_simulation_resolution_deg = 12.0/127
    default_random_seed = 3142
    # Size of EoR cube foreground simulation should match
    # (used when calculating fits header variables)
    default_cube_side_Mpc = 2048.0
    # Redshift of EoR cube foreground simulation should match
    # (used when calculating fits header variables)
    default_redshift = 7.6
    default_save_fits = False

    # ===== Inputs =====
    generate_additional_extrapolated_HF_foreground_cube = kwargs.pop(
        'generate_additional_extrapolated_HF_foreground_cube',
        default_generate_additional_extrapolated_HF_foreground_cube)
    HF_nu_min_MHz = kwargs.pop(
        'HF_nu_min_MHz',
        default_HF_nu_min_MHz)
    fits_storage_dir = kwargs.pop(
        'fits_storage_dir',
        default_fits_storage_dir)
    HF_nu_min_MHz_array = kwargs.pop(
        'HF_nu_min_MHz_array',
        default_fits_storage_dir)
    simulation_FoV_deg = kwargs.pop(
        'simulation_FoV_deg',
        default_simulation_FoV_deg)
    simulation_resolution_deg = kwargs.pop(
        'simulation_resolution_deg',
        default_simulation_resolution_deg)
    random_seed = kwargs.pop(
        'random_seed',
        default_random_seed)
    cube_side_Mpc = kwargs.pop(
        'random_seed',
        default_cube_side_Mpc)
    redshift = kwargs.pop(
        'random_seed',
        default_redshift)
    save_fits = kwargs.pop(
        'save_fits',
        default_save_fits)

    n_sim_pix = int(simulation_FoV_deg/simulation_resolution_deg + 0.5)

    low_res_to_high_res_std_conversion_factor =\
        update_Tb_experimental_std_K_to_correct_for_normalisation_resolution(
            Tb_experimental_std_K,
            simulation_FoV_deg,
            simulation_resolution_deg)
    Tb_experimental_std_K = (Tb_experimental_std_K
                             * low_res_to_high_res_std_conversion_factor)

    GFC = GenerateForegroundCube(
        nu, nv, neta, nq, beta_experimental_mean, beta_experimental_std,
        gamma_mean, gamma_sigma, Tb_experimental_mean_K, Tb_experimental_std_K,
        nu_min_MHz, channel_width_MHz, random_seed=random_seed)

    Tb_nu, A, beta, Tb, nu_array_MHz =\
        GFC.generate_normalised_Tb_A_and_beta_fields(
            n_sim_pix, n_sim_pix, n_sim_pix, n_sim_pix, nf, neta, nq)

    HF_Tb_nu = []
    beta = A = Tb = []

    ED = ExtractDataFrom21cmFASTCube(plot_data=False)
    bandwidth_MHz, central_frequency_MHz, output_21cmFast_box_width_deg =\
        ED.calculate_box_size_in_degrees_and_MHz(cube_side_Mpc, redshift)

    DUC = DataUnitConversionmKAndJyPerPix()
    Data_mK = Tb_nu * 1.e3

    Channel_Frequencies_Array_Hz = nu_array_MHz * 1.e6
    Pixel_Height_rad = (output_21cmFast_box_width_deg * (np.pi/180.)
                        / Tb_nu.shape[1])
    DUC.convert_from_mK_to_Jy_per_pix(
        Data_mK, Channel_Frequencies_Array_Hz, Pixel_Height_rad)
    Data_Jy_per_Pixel = ED.convert_from_mK_to_Jy_per_pix(
        Data_mK, Channel_Frequencies_Array_Hz, Pixel_Height_rad)

    output_fits_file_name =\
        'Jelic_GDSE_cube_{:d}MHz.fits'.format(int(nu_min_MHz))
    output_fits_path1 = fits_storage_dir + output_fits_file_name
    output_fits_path2 = (fits_storage_dir
                         + '/ZNPS{:d}/'.format(int(nu_min_MHz))
                         + output_fits_file_name)
    print(output_fits_path1, '\n'+ output_fits_path2)
    write_to_file = save_fits
    # write_to_file = False
    WD2F = WriteDataToFits()
    if write_to_file:
        WD2F.write_data(
            Data_Jy_per_Pixel, output_fits_path1,
            Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)
        WD2F.write_data(
            Data_Jy_per_Pixel, output_fits_path2,
            Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)

    output_fits_file_name =\
        'Jelic_GDSE_cube_{:d}MHz_mK.fits'.format(int(nu_min_MHz))
    output_fits_path1 = fits_storage_dir + output_fits_file_name
    output_fits_path2 = (fits_storage_dir
                         + '/ZNPS{:d}/'.format(int(nu_min_MHz))
                         + output_fits_file_name)
    print(output_fits_path1, '\n'+ output_fits_path2)
    WD2F = WriteDataToFits()
    if write_to_file:
        WD2F.write_data(
            Data_mK, output_fits_path1,
            Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)
        WD2F.write_data(
            Data_mK, output_fits_path2,
            Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)

    base_dir = 'Plots'
    save_dir = base_dir+'/Likelihood_v1d75_3D_ZM/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    print('Using use_foreground_cube data')
    # Replace Gaussian signal with foreground cube
    # Tb_nu_mK = Tb_nu * 1.e2
    Tb_nu_mK = Tb_nu * 1.e3
    scidata1 = Tb_nu_mK
    # scidata1 = random_quad[0:nf, 0:nu, 0:nv]

    base_dir = 'Plots'
    save_dir = base_dir+'/Likelihood_v1d75_3D_ZM/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    import numpy
    axes_tuple = (1, 2)
    vfft1 = numpy.fft.ifftshift(scidata1[0:nf] + 0j, axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    vfft1 = numpy.fft.fftn(vfft1, axes=axes_tuple)
    vfft1 = numpy.fft.fftshift(vfft1, axes=axes_tuple)

    pylab.close('all')

    sci_f, sci_v, sci_u = vfft1.shape
    # Updated for python 3: floor division
    sci_v_centre = sci_v//2
    # Updated for python 3: floor division
    sci_u_centre = sci_u//2
    # Updated for python 3: floor division
    vfft1_subset = vfft1[0 : nf,
                         sci_u_centre - nu//2 : sci_u_centre + nu//2 + 1,
                         sci_v_centre - nv//2 : sci_v_centre + nv//2 + 1]

    axes_tuple = (1,2)
    scidata1_subset = numpy.fft.ifftshift(vfft1_subset + 0j, axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    scidata1_subset = numpy.fft.ifftn(scidata1_subset, axes=axes_tuple)
    scidata1_subset = numpy.fft.fftshift(scidata1_subset, axes=axes_tuple)

    # s = np.dot(Finv,scidata1_subset.reshape(-1,1)).flatten()

    if not p.fit_for_monopole:
        # Remove channel means for interferometric image (not required
        # for a gridded model but necessary when using a large FWHM PB
        # model or small model FoV due to the non-closing nudft (i.e.
        # for fringes with a non-integer number of wavelengths across
        # the model image))
        d_im = scidata1_subset.copy()
        for i_chan in range(len(d_im)):
            d_im[i_chan] = d_im[i_chan] - d_im[i_chan].mean()
        # s2 = np.dot(Finv, d_im.reshape(-1, 1)).flatten()
        scidata1_subset = d_im
        s = np.dot(Finv, d_im.reshape(-1, 1)).flatten()
    else:
        s = np.dot(Finv, scidata1_subset.reshape(-1, 1)).flatten()

    abc = s
    fg = s

    return fg, s, Tb_nu, beta_experimental_mean, beta_experimental_std,\
           gamma_mean, gamma_sigma,\
           Tb_experimental_mean_K, Tb_experimental_std_K,\
           nu_min_MHz, channel_width_MHz, HF_Tb_nu, scidata1_subset


def generate_additional_HF_Jelic_cube(
        A, HF_nu_min_MHz, beta, fits_storage_dir,
        nu, nv, nx, ny, nf, neta, nq,
        k_x, k_y, k_z, Show,
        beta_experimental_mean, beta_experimental_std,
        gamma_mean, gamma_sigma,
        Tb_experimental_mean_K, Tb_experimental_std_K,
        nu_min_MHz, channel_width_MHz, **kwargs):

    # HF_nu_min_MHz = 225
    HF_nu_array_MHz = HF_nu_min_MHz+np.arange(nf) * channel_width_MHz
    HF_A_nu = np.array(
        [A * (HF_nu_array_MHz[i_nu]/nu_min_MHz)**-beta
         for i_nu in range(len(HF_nu_array_MHz))]
        )
    HF_Tb_nu = np.sum(HF_A_nu, axis=1)

    ED = ExtractDataFrom21cmFASTCube(plot_data=False)
    cube_side_Mpc = 3072.0
    redshift = 7.6
    # cube_side_Mpc = 2048.0
    # redshift = 10.26
    bandwidth_MHz, central_frequency_MHz, output_21cmFast_box_width_deg =\
        ED.calculate_box_size_in_degrees_and_MHz(cube_side_Mpc, redshift)

    DUC = DataUnitConversionmKAndJyPerPix()
    Data_mK = HF_Tb_nu*1.e3

    Channel_Frequencies_Array_Hz = HF_nu_array_MHz * 1.e6
    Pixel_Height_rad = (output_21cmFast_box_width_deg * (np.pi/180.)
                        / HF_Tb_nu.shape[1])
    DUC.convert_from_mK_to_Jy_per_pix(
        Data_mK, Channel_Frequencies_Array_Hz, Pixel_Height_rad)
    Data_Jy_per_Pixel = ED.convert_from_mK_to_Jy_per_pix(
        Data_mK, Channel_Frequencies_Array_Hz, Pixel_Height_rad)

    output_fits_file_name =\
        'Jelic_GDSE_cube_{:d}MHz_mK.fits'.format(int(HF_nu_min_MHz))
    output_fits_path1 = fits_storage_dir + output_fits_file_name
    output_fits_path2 = (fits_storage_dir
                         + '/ZNPS{:d}/'.format(int(HF_nu_min_MHz))
                         + output_fits_file_name)
    print(output_fits_path1, '\n'+ output_fits_path2)
    # write_to_file = True
    write_to_file = False
    WD2F = WriteDataToFits()
    if write_to_file:
        WD2F.write_data(
            Data_Jy_per_Pixel, output_fits_path1,
            Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)
        WD2F.write_data(
            Data_Jy_per_Pixel, output_fits_path2,
            Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)

    output_fits_file_name =\
        'Jelic_GDSE_cube_{:d}MHz_mK.fits'.format(int(HF_nu_min_MHz))
    output_fits_path1 = fits_storage_dir + output_fits_file_name
    output_fits_path2 = (fits_storage_dir
                         + '/ZNPS{:d}/'.format(int(HF_nu_min_MHz))
                         + output_fits_file_name)
    print(output_fits_path1, '\n'+ output_fits_path2)
    WD2F = WriteDataToFits()
    if write_to_file:
        WD2F.write_data(
            Data_mK, output_fits_path1,
            Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)
        WD2F.write_data(
            Data_mK, output_fits_path2,
            Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)

    return HF_Tb_nu


def top_hat_average_temperature_cube_to_lower_res_31x31xnf_cube(Tb_nu):
    bl_size = Tb_nu[0].shape[0] / 31
    dat_subset = 31*bl_size
    averaged_cube = []
    for i_freq in range(len(Tb_nu)):
        Tb_nu_bls = np.array(
            [Tb_nu[i_freq][:dat_subset, :dat_subset][
                i*bl_size:(i+1)*bl_size,j*bl_size:(j+1)*bl_size]
             for i in range(31) for j in range(31)])
        Tb_nu_bls_means = np.array(
            [x.mean() for x in Tb_nu_bls]).reshape(31, 31)
        averaged_cube.append(Tb_nu_bls_means)
    averaged_cube = np.array(averaged_cube)
    return averaged_cube


def generate_data_from_loaded_EoR_cube(
        nu, nv, nx, ny, nf, neta, nq, k_x, k_y, k_z, Show, chan_selection):

    print('Using use_EoR_cube data')
    # Replace Gaussian signal with EoR cube
    scidata1 = np.load(
        '/users/psims/EoR/EoR_simulations/21cmFAST_512MPc_512pix_128pix/'\
        'Fits/21cm_z10d2_mK.npz')['arr_0']

    base_dir = 'Plots'
    save_dir = base_dir+'/Likelihood_v1d75_3D_ZM/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # scidata1 =\
    # 	top_hat_average_temperature_cube_to_lower_res_31x31xnf_cube(scidata1)
    scidata1 = scidata1[0:nf, :124, :124]

    import numpy
    axes_tuple = (1,2)
    if chan_selection == '0_38_':
        vfft1 = numpy.fft.ifftshift(scidata1[0:38]-scidata1[0].mean() + 0j,
                                    axes=axes_tuple)
    elif chan_selection == '38_76_':
        vfft1 = numpy.fft.ifftshift(scidata1[38:76]-scidata1[0].mean() + 0j,
                                    axes=axes_tuple)
    elif chan_selection == '76_114_':
        vfft1 = numpy.fft.ifftshift(scidata1[76:114]-scidata1[0].mean() + 0j,
                                    axes=axes_tuple)
    else:
        vfft1 = numpy.fft.ifftshift(scidata1[0:nf]-scidata1[0].mean() + 0j,
                                    axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    vfft1 = numpy.fft.fftn(vfft1, axes=axes_tuple)
    vfft1 = numpy.fft.fftshift(vfft1, axes=axes_tuple)

    sci_f, sci_v, sci_u = vfft1.shape
    # Updated for python 3: floor division
    sci_v_centre = sci_v//2
    # Updated for python 3: floor division
    sci_u_centre = sci_u//2
    # Updated for python 3: floor division
    vfft1_subset = vfft1[0 : nf,
                         sci_u_centre - nu//2 : sci_u_centre + nu//2 + 1,
                         sci_v_centre - nv//2 : sci_v_centre + nv//2 + 1]
    s_before_ZM = vfft1_subset.flatten() / vfft1[0].size**0.5
    ZM_vis_ordered_mask = np.ones(nu*nv*nf)
    # Updated for python 3: floor division
    ZM_vis_ordered_mask[nf*((nu*nv)//2) : nf*((nu*nv)//2 + 1)] = 0
    ZM_vis_ordered_mask = ZM_vis_ordered_mask.astype('bool')
    ZM_chan_ordered_mask = ZM_vis_ordered_mask.reshape(-1, neta+nq).T.flatten()
    s = s_before_ZM[ZM_chan_ordered_mask]
    abc = s

    return s, abc, scidata1


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
        vfft1 = numpy.fft.ifftshift(scidata1[0:38]-scidata1[0].mean() + 0j,
                                    axes=axes_tuple)
    elif chan_selection == '38_76_':
        vfft1 = numpy.fft.ifftshift(scidata1[38:76]-scidata1[0].mean() + 0j,
                                    axes=axes_tuple)
    elif chan_selection == '76_114_':
        vfft1 = numpy.fft.ifftshift(scidata1[76:114]-scidata1[0].mean() + 0j,
                                    axes=axes_tuple)
    else:
        vfft1 = numpy.fft.ifftshift(scidata1[0:nf]-scidata1[0].mean() + 0j,
                                    axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    vfft1 = numpy.fft.fftn(vfft1, axes=axes_tuple)
    vfft1 = numpy.fft.fftshift(vfft1, axes=axes_tuple)

    sci_f, sci_v, sci_u = vfft1.shape
    # Updated for python 3: floor division
    sci_v_centre = sci_v//2
    # Updated for python 3: floor division
    sci_u_centre = sci_u//2
    # Updated for python 3: floor division
    vfft1_subset = vfft1[0 : nf,
                         sci_u_centre - nu//2 : sci_u_centre + nu//2 + 1,
                         sci_v_centre - nv//2 : sci_v_centre + nv//2 + 1]
    s_before_ZM = vfft1_subset.flatten() / vfft1[0].size**0.5
    # s_before_ZM = vfft1_subset.flatten()
    ZM_vis_ordered_mask = np.ones(nu*nv*nf)
    # Updated for python 3: floor division
    ZM_vis_ordered_mask[nf*((nu*nv)//2) : nf*((nu*nv)//2 + 1)] = 0
    ZM_vis_ordered_mask = ZM_vis_ordered_mask.astype('bool')
    ZM_chan_ordered_mask = ZM_vis_ordered_mask.reshape(-1, neta+nq).T.flatten()
    s = s_before_ZM[ZM_chan_ordered_mask]
    abc = s

    return s, abc, scidata1


def generate_white_noise_signal_instrumental_k_2_vis(
        nu, nv, nx, ny, nf, neta, nq, k_x, k_y, k_z,
        T, Show, chan_selection, masked_power_spectral_modes):

    print('Using use_WN_cube data')
    EoR_npz_path = p.EoR_npz_path_sc

    # Replace Gaussian signal with EoR cube
    scidata1 = np.load(EoR_npz_path)['arr_0']

    # Overwrite EoR cube with white noise
    np.random.seed(123)
    scidata1 = np.random.normal(0, scidata1.std()*1., scidata1.shape) * 0.5

    base_dir = 'Plots'
    save_dir = base_dir+'/Likelihood_v1d75_3D_ZM/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    axes_tuple = (0, 1, 2)
    scidata1_kcube = numpy.fft.ifftshift(
        scidata1[0:38]-scidata1[0:38].mean() + 0j, axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    scidata1_kcube = numpy.fft.fftn(scidata1_kcube, axes=axes_tuple)
    scidata1_kcube = numpy.fft.fftshift(scidata1_kcube, axes=axes_tuple)

    sci_f, sci_v, sci_u = scidata1_kcube.shape
    # Updated for python 3: floor division
    sci_v_centre = sci_v//2
    # Updated for python 3: floor division
    sci_u_centre = sci_u//2
    # Updated for python 3: floor division
    scidata1_kcube_subset = scidata1_kcube[
                            0 : nf,
                            sci_u_centre - nu//2 : sci_u_centre + nu//2 + 1,
                            sci_v_centre - nv//2 : sci_v_centre + nv//2 + 1]
    scidata1_kcube_subset_before_ZM = (
            scidata1_kcube_subset.flatten() / scidata1_kcube.size**0.5)
    ZM_vis_ordered_mask = np.ones(nu*nv*nf)
    # Updated for python 3: floor division
    ZM_vis_ordered_mask[nf*((nu*nv)//2) : nf*((nu*nv)//2 + 1)] = 0
    ZM_vis_ordered_mask = ZM_vis_ordered_mask.astype('bool')
    ZM_chan_ordered_mask = ZM_vis_ordered_mask.reshape(-1, neta+nq).T.flatten()
    scidata1_kcube_subset_ZM =\
        scidata1_kcube_subset_before_ZM[ZM_chan_ordered_mask]

    # Zero modes that correspond to / are replaced with foreground
    # parameters (in the parameter vector that T is desiged to
    # operate on) before applying T!
    # Note: to apply masked_power_spectral_modes (which is vis_ordered)
    # correctly scidata1_kcube_subset_ZM should also be vis_ordered
    # however here it's vis_ordered. The only reason this isn't breaking
    # things is because it's white noise which means the ordering makes
    # no difference here...
    scidata1_kcube_subset_ZM[masked_power_spectral_modes] = 0.0
    s = np.dot(T, scidata1_kcube_subset_ZM.reshape(-1, 1)).flatten()
    abc = s

    return s, abc, scidata1


def generate_white_noise_signal_instrumental_im_2_vis(
        nu, nv, nx, ny, nf, neta, nq, k_x, k_y, k_z,
        Finv, Show, chan_selection, masked_power_spectral_modes, mod_k):

    print('Using use_WN_cube data')
    EoR_npz_path = p.EoR_npz_path_sc

    # Replace Gaussian signal with EoR cube
    scidata1 = np.load(EoR_npz_path)['arr_0']

    # Overwrite EoR cube with white noise
    # np.random.seed(21287254)
    # np.random.seed(4123)
    # np.random.seed(54123)
    # np.random.seed(154123)
    np.random.seed(123)
    scidata1 = np.random.normal(0, scidata1.std()*1., [nf, nu, nv]) * 0.5

    axes_tuple = (0, 1, 2)
    scidata1_kcube = numpy.fft.ifftshift(
        scidata1[0:38]-scidata1[0:38].mean() + 0j, axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    scidata1_kcube = numpy.fft.fftn(scidata1_kcube, axes=axes_tuple)
    scidata1_kcube = numpy.fft.fftshift(scidata1_kcube, axes=axes_tuple)

    sci_f, sci_v, sci_u = scidata1_kcube.shape
    # Updated for python 3: floor division
    sci_v_centre = sci_v//2
    # Updated for python 3: floor division
    sci_u_centre = sci_u//2
    # Updated for python 3: floor division
    scidata1_kcube_subset = scidata1_kcube[
                            0 : nf,
                            sci_u_centre - nu//2 : sci_u_centre + nu//2 + 1,
                            sci_v_centre - nv//2 : sci_v_centre + nv//2 + 1]

    red_noise_power_law_coefficient = 3.0
    print('Using red_noise_power_law_coefficient:',
          red_noise_power_law_coefficient)
    red_noise_scaling_cube = 1./(mod_k**(red_noise_power_law_coefficient/2.0))
    red_noise_scaling_cube[np.isinf(red_noise_scaling_cube)] = 1.0
    scidata1_kcube_subset_scaled = scidata1_kcube_subset*red_noise_scaling_cube

    ZM_vis_ordered_mask = np.ones(nu*nv*nf)
    # Updated for python 3: floor division
    ZM_vis_ordered_mask[nf*((nu*nv)//2) : nf*((nu*nv)//2 + 1)] = 0
    ZM_vis_ordered_mask = ZM_vis_ordered_mask.astype('bool')
    ZM_chan_ordered_mask = ZM_vis_ordered_mask.reshape(-1, neta+nq).T.flatten()

    total_masked_power_spectral_modes = np.ones_like(ZM_chan_ordered_mask)
    unmasked_power_spectral_modes_chan_ordered = np.logical_not(
        masked_power_spectral_modes
        ).reshape(-1,neta+nq).T
    # Make mask symmetric for real model image
    # Updated for python 3: floor division
    centre_chan = (nf//2)
    if centre_chan%2 != 0:
        centre_chan =centre_chan - 1
    for i, chan in enumerate(unmasked_power_spectral_modes_chan_ordered):
        if np.sum(chan) == 0.0 and i != centre_chan:
            unmasked_power_spectral_modes_chan_ordered[-1-i] = chan
    total_masked_power_spectral_modes[ZM_chan_ordered_mask] =\
        unmasked_power_spectral_modes_chan_ordered.flatten()
    total_masked_power_spectral_modes = np.logical_and(
        total_masked_power_spectral_modes, ZM_chan_ordered_mask)

    # Zero modes that are not part of the data model
    # scidata1_kcube_subset_scaled[
    # 	total_masked_power_spectral_modes.reshape(
    # 		scidata1_kcube_subset_scaled.shape)
    # ] = 0.0

    axes_tuple = (0, 1, 2)
    scidata1_subset_scaled = numpy.fft.ifftshift(
        scidata1_kcube_subset_scaled + 0j, axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    scidata1_subset_scaled = numpy.fft.ifftn(
        scidata1_subset_scaled, axes=axes_tuple)
    scidata1_subset_scaled = numpy.fft.fftshift(
        scidata1_subset_scaled, axes=axes_tuple)

    # Apply correct normalisation in the image domain for the relavent invariance testing
    # Correct invariance - RMS of a (white...) noise cube should be constant as a function of resolution
    preset_cube_rms = 100.e0
    scidata1_subset_scaled = (scidata1_subset_scaled
                              * preset_cube_rms/scidata1_subset_scaled.std())
    s = np.dot(Finv, scidata1_subset_scaled.reshape(-1, 1)).flatten()
    abc = s

    return s, abc, scidata1


def generate_EoR_signal_instrumental_im_2_vis(
        nu, nv, nx, ny, nf, neta, nq, k_x, k_y, k_z,
        Finv, Show, chan_selection, masked_power_spectral_modes,
        mod_k, EoR_npz_path):

    print('Using use_EoR_cube data')
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
    # scidata1_kcube = numpy.fft.ifftshift(
    # 	scidata1[0:nf]-scidata1[0:nf].mean() + 0j, axes=axes_tuple)
    # # FFT (python pre-normalises correctly! -- see
    # # parsevals theorem for discrete fourier transform.)
    # scidata1_kcube = numpy.fft.fftn(scidata1_kcube, axes=axes_tuple)
    # scidata1_kcube = numpy.fft.fftshift(scidata1_kcube, axes=axes_tuple)
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
    # # 	Hermitian_small_spacial_scale_mask = np.zeros(scidata1_kcube_subset.shape)
    # # 	Hermitian_small_spacial_scale_mask[0] = 1 #Nyquist mode
    # # 	Hermitian_small_spacial_scale_mask[1] = 1 #2nd highest freq
    # # 	# Hermitian_small_spacial_scale_mask[2] = 1 #3nd highest freq
    # # 	# Hermitian_small_spacial_scale_mask[-2] = 1 #3nd highest freq
    # # 	Hermitian_small_spacial_scale_mask[-1] = 1 #2nd highest freq
    # # 	scidata1_kcube_subset[Hermitian_small_spacial_scale_mask.astype('bool')] = 0.0
    # # 	# scidata1_kcube_subset[Hermitian_small_spacial_scale_mask.astype('bool')] *= 1.e1
    # # else:
    # # 	print 'Intrinsic noise fitting model included. Using full EoR cube (no low-pass filter)'
    #
    # axes_tuple = (0, 1, 2)
    # scidata1_subset = numpy.fft.ifftshift(
    # 	scidata1_kcube_subset + 0j, axes=axes_tuple)
    # # FFT (python pre-normalises correctly! -- see
    # # parsevals theorem for discrete fourier transform.)
    # scidata1_subset = numpy.fft.ifftn(scidata1_subset, axes=axes_tuple)
    # scidata1_subset = numpy.fft.fftshift(scidata1_subset, axes=axes_tuple)

    # This function needs to be updated to use astropy.healpix info
    # to generate a sky model
    # Make a cube in the subset space using a stddev
    # scaled to match the healvis sky realizations
    np.random.seed(12346)
    rms_mK = 1.0341928997136198 # scaled to match healvis sims for nx = 9
    # rms_mK = 0.8043383678108019 # scaled to match healvis sims for nx = 7
    scidata1_subset = np.random.normal(0, rms_mK, (nf, ny, nx))

    # Temporary workaround for nside=16 healpix coordinates in Finv & Fprime
    # np.random.seed(18237)
    # rms_mK = 0.40474845072666593
    # n_hpx_pix = 10
    # scidata1_subset = np.random.normal(0, rms_mK, (nf, n_hpx_pix))

    for i_f in range(nf):
        scidata1_subset[i_f] -= scidata1_subset[i_f].mean()
    scidata1 = scidata1_subset.copy()
    s = np.dot(Finv, scidata1_subset.reshape(-1, 1)).flatten()
    # Temporary workaround for nside=16 healpix coordinates in Finv & Fprime
    # s = np.dot(Finv, scidata1_subset.flatten())
    abc = s

    return s, abc, scidata1


def calculate_subset_cube_power_spectrum_v1d0(
        nu, nv, nx, ny, nf, neta, nq, k_cube_voxels_in_bin,
        modkbins_containing_voxels, EoR_npz_path, mod_k, k_z):

    # Replace Gaussian signal with EoR cube
    scidata1 = np.load(EoR_npz_path)['arr_0']

    axes_tuple = (0, 1, 2)
    scidata1_kcube = numpy.fft.ifftshift(
        scidata1[0:nf]-scidata1[0:nf].mean() + 0j, axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    scidata1_kcube = numpy.fft.fftn(scidata1_kcube, axes=axes_tuple)
    scidata1_kcube = numpy.fft.fftshift(scidata1_kcube, axes=axes_tuple)

    sci_f, sci_v, sci_u = scidata1_kcube.shape
    # Updated for python 3: floor division
    sci_v_centre = sci_v//2
    # Updated for python 3: floor division
    sci_u_centre = sci_u//2
    # Updated for python 3: floor division
    scidata1_kcube_subset = scidata1_kcube[
                            0: nf,
                            sci_u_centre - nu // 2: sci_u_centre + nu // 2 + 1,
                            sci_v_centre - nv // 2: sci_v_centre + nv // 2 + 1]

    # if not p.use_intrinsic_noise_fitting:
    # 	# Zero modes that are not currently fit for until intrinsic
    # 	# noise fitting (which models these terms) has been implemented
    # 	print('No intrinsic noise fitting model.'
    # 		  ' Low-pass filtering the EoR cube')
    # 	Hermitian_small_spacial_scale_mask =\
    # 		np.zeros(scidata1_kcube_subset.shape)
    # 	Hermitian_small_spacial_scale_mask[0] = 1 # Nyquist mode
    # 	Hermitian_small_spacial_scale_mask[1] = 1 # 2nd highest freq
    # 	# Hermitian_small_spacial_scale_mask[2] = 1 # 3nd highest freq
    # 	# Hermitian_small_spacial_scale_mask[-2] = 1 # 3nd highest freq
    # 	Hermitian_small_spacial_scale_mask[-1] = 1 # 2nd highest freq
    # 	scidata1_kcube_subset[
    # 		Hermitian_small_spacial_scale_mask.astype('bool')
    # 	] = 0.0
    # 	# scidata1_kcube_subset[
    # 	# 	Hermitian_small_spacial_scale_mask.astype('bool')
    # 	# ] *= 1.e1
    # else:
    # 	print('Intrinsic noise fitting model included.'
    # 		  ' Using full EoR cube (no low-pass filter)')

    for i in range(len(scidata1_kcube_subset)):
        print(i, scidata1_kcube_subset[i].std())

    # scidata1_kcube_subset

    ZM_vis_ordered_mask = np.ones(nu*nv*nf)
    # Updated for python 3: floor division
    ZM_vis_ordered_mask[nf*((nu*nv)//2) : nf*((nu*nv)//2 + 1)] = 0
    ZM_vis_ordered_mask = ZM_vis_ordered_mask.astype('bool')
    ZM_chan_ordered_mask = ZM_vis_ordered_mask.reshape(-1, nf).T.flatten()

    # scidata1_kcube_subset.flatten()[ZM_chan_ordered_mask]
    # mod_k.flatten()[ZM_chan_ordered_mask]

    chan_ordered_bin_pixels_list = [np.where(
        np.logical_and(
            k_z.flatten()[ZM_chan_ordered_mask] != 0.0,
            np.logical_and(
                mod_k.flatten()[ZM_chan_ordered_mask]
                > modkbins_containing_voxels[i][0][0],
                mod_k.flatten()[ZM_chan_ordered_mask]
                <= modkbins_containing_voxels[i][0][1])
            )
        )
        for i in range(len(modkbins_containing_voxels))]

    excluded_zeroed_voxels =\
        np.where(scidata1_kcube_subset.flatten()[ZM_chan_ordered_mask] == 0.0)

    excluded_zeroed_voxel_locations_in_chan_ordered_bin_pixels_list =\
        [
            [
                np.where(chan_ordered_bin_pixels_list[i_bin][0]
                         == excluded_zeroed_voxels[0][i_ex])[0]
                for i_ex in range(len(excluded_zeroed_voxels[0]))
                if np.where(chan_ordered_bin_pixels_list[i_bin][0]
                            == excluded_zeroed_voxels[0][i_ex])[0]
                ]
            for i_bin in range(len(chan_ordered_bin_pixels_list))
            ]

    chan_ordered_bin_pixels_list_without_zeroed_excluded_pixels =\
        np.array(chan_ordered_bin_pixels_list).copy()
    chan_ordered_bin_pixels_list_without_zeroed_excluded_pixels = [np.delete(
        chan_ordered_bin_pixels_list[i_bin][0],
        excluded_zeroed_voxel_locations_in_chan_ordered_bin_pixels_list[i_bin])
        for i_bin in range(len(chan_ordered_bin_pixels_list))
        ]

    mean_ks = []
    variances = []
    powers = []
    dimensionless_powers = []
    for i in range(len(chan_ordered_bin_pixels_list)):
        ks_in_bin = mod_k.flatten()
        ks_in_bin = ks_in_bin[ZM_chan_ordered_mask]
        ks_in_bin = ks_in_bin[
            chan_ordered_bin_pixels_list_without_zeroed_excluded_pixels[i]]
        amplitudes_in_bin = scidata1_kcube_subset.flatten()
        amplitudes_in_bin = amplitudes_in_bin[ZM_chan_ordered_mask]
        amplitudes_in_bin = amplitudes_in_bin[
            chan_ordered_bin_pixels_list_without_zeroed_excluded_pixels[i]]
        mean_k = ks_in_bin.mean()
        variance = amplitudes_in_bin.var()
        power = (abs(amplitudes_in_bin)**2.).mean()
        dimensionless_power =\
            (ks_in_bin**3. * abs(amplitudes_in_bin)**2.).mean()
        print(i, mean_k, variance, power, power/variance)
        mean_ks.append(mean_k)
        variances.append(variance)
        powers.append(power)
        dimensionless_powers.append(dimensionless_power)

    subset_ps_output_dir =\
        '/users/psims/EoR/Python_Scripts/BayesEoR/git_version/BayesEoR/'\
        'spec_model_tests/random/subset_ps/{}/'.format(
            EoR_npz_path.split('/')[-2].replace('.npz','').replace('.','d'))
    subset_ps_output_file =\
        EoR_npz_path.split('/')[-1].replace('.npz','').replace('.','d')
    subset_ps_output_path = subset_ps_output_dir+subset_ps_output_file

    print('Saving unnormalised dimensionless power spectrum to: \n',
          subset_ps_output_path)
    if not os.path.isdir(subset_ps_output_dir):
            os.makedirs(subset_ps_output_dir)
    np.savetxt(
        subset_ps_output_path, np.vstack((mean_ks, dimensionless_powers)).T)

    return np.vstack((mean_ks, dimensionless_powers)).T


def calculate_subset_cube_power_spectrum_v2d0(
        nu, nv, nx, ny, nf, neta, nq, k_cube_voxels_in_bin,
        modkbins_containing_voxels, EoR_npz_path, mod_k, k_z):

    # Replace Gaussian signal with EoR cube
    scidata1 = np.load(EoR_npz_path)['arr_0']

    axes_tuple = (0, 1, 2)
    scidata1_kcube = numpy.fft.ifftshift(
        scidata1[0:nf]-scidata1[0:nf].mean() + 0j, axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    scidata1_kcube = numpy.fft.fftn(scidata1_kcube, axes=axes_tuple)
    scidata1_kcube = numpy.fft.fftshift(scidata1_kcube, axes=axes_tuple)

    sci_f, sci_v, sci_u = scidata1_kcube.shape
    # Updated for python 3: floor division
    sci_v_centre = sci_v//2
    # Updated for python 3: floor division
    sci_u_centre = sci_u//2
    # Updated for python 3: floor division
    scidata1_kcube_subset = scidata1_kcube[
                            0: nf,
                            sci_u_centre - nu // 2: sci_u_centre + nu // 2 + 1,
                            sci_v_centre - nv // 2: sci_v_centre + nv // 2 + 1]

    for i in range(len(scidata1_kcube_subset)):
        print(i, scidata1_kcube_subset[i].std())

    # scidata1_kcube_subset

    ZM_vis_ordered_mask = np.ones(nu*nv*nf)
    # Updated for python 3: floor division
    ZM_vis_ordered_mask[nf*((nu*nv)//2) : nf*((nu*nv)//2 + 1)] = 0
    ZM_vis_ordered_mask = ZM_vis_ordered_mask.astype('bool')
    ZM_chan_ordered_mask = ZM_vis_ordered_mask.reshape(-1, nf).T.flatten()

    # scidata1_kcube_subset.flatten()[ZM_chan_ordered_mask]
    # mod_k.flatten()[ZM_chan_ordered_mask]

    chan_ordered_bin_pixels_list = [np.where(
        np.logical_and(
            k_z.flatten()[ZM_chan_ordered_mask] != 0.0,
            np.logical_and(
                mod_k.flatten()[ZM_chan_ordered_mask]
                > modkbins_containing_voxels[i][0][0],
                mod_k.flatten()[ZM_chan_ordered_mask]
                <= modkbins_containing_voxels[i][0][1])
            )
        )
        for i in range(len(modkbins_containing_voxels))]

    excluded_zeroed_voxels =\
        np.where(scidata1_kcube_subset.flatten()[ZM_chan_ordered_mask] == 0.0)

    excluded_zeroed_voxel_locations_in_chan_ordered_bin_pixels_list =\
        [
            [
                np.where(chan_ordered_bin_pixels_list[i_bin][0]
                         == excluded_zeroed_voxels[0][i_ex])[0]
                for i_ex in range(len(excluded_zeroed_voxels[0]))
                if np.where(chan_ordered_bin_pixels_list[i_bin][0]
                            == excluded_zeroed_voxels[0][i_ex])[0]
                ]
            for i_bin in range(len(chan_ordered_bin_pixels_list))
            ]

    chan_ordered_bin_pixels_list_without_zeroed_excluded_pixels =\
        np.array(chan_ordered_bin_pixels_list).copy()
    chan_ordered_bin_pixels_list_without_zeroed_excluded_pixels = [np.delete(
        chan_ordered_bin_pixels_list[i_bin][0],
        excluded_zeroed_voxel_locations_in_chan_ordered_bin_pixels_list[i_bin])
        for i_bin in range(len(chan_ordered_bin_pixels_list))
        ]

    mean_ks = []
    variances = []
    powers = []
    dimensionless_powers = []
    for i in range(len(chan_ordered_bin_pixels_list)):
        ks_in_bin = mod_k.flatten()
        ks_in_bin = ks_in_bin[ZM_chan_ordered_mask]
        ks_in_bin = ks_in_bin[
            chan_ordered_bin_pixels_list_without_zeroed_excluded_pixels[i]]
        amplitudes_in_bin = scidata1_kcube_subset.flatten()
        amplitudes_in_bin = amplitudes_in_bin[ZM_chan_ordered_mask]
        amplitudes_in_bin = amplitudes_in_bin[
            chan_ordered_bin_pixels_list_without_zeroed_excluded_pixels[i]]
        mean_k = ks_in_bin.mean()
        variance = amplitudes_in_bin.var()
        power = (abs(amplitudes_in_bin)**2.).mean()
        dimensionless_power =\
            (ks_in_bin**3. * abs(amplitudes_in_bin)**2.).mean()
        print(i, mean_k, variance, power, power/variance)
        mean_ks.append(mean_k)
        variances.append(variance)
        powers.append(power)
        dimensionless_powers.append(dimensionless_power)

    # Correctly normalize the dimensionless power spectrum.
    # Since I am doing a numpy fft (which follows the same convention as
    # fftw used in 21cmFast) the normalisation should be realtively
    # straightforward / similar to normalisation_21cmFast. Because I
    # start by taking a 38/512 subset in frequency, the ffted values
    # will be (38/512)**0.5 times smaller than the equivalent values in
    # the fft of the full cube so there will be an additional
    # (512/38.)**0.5 factor in amplitude relative to
    # normalisation_21cmFast
    subset_amplitude_normalisation = (512/38.)**0.5
    subset_power_spectrum_normalisation = subset_amplitude_normalisation**2.

    ###
    #     *((float *)deldel_T + HII_R_FFT_INDEX(i,j,k)) = (delta_T[HII_R_INDEX(i,j,k)]/ave - 1)*VOLUME/(HII_TOT_NUM_PIXELS+0.0);
    # if (DIMENSIONAL_T_POWER_SPEC){
    #   *((float *)deldel_T + HII_R_FFT_INDEX(i,j,k)) *= ave;
    # }

    # Note: DIMENSIONAL_T_POWER_SPEC in 21cmFast just means calculating
    # the standard mK^2 power spectrum i.e. the `dimensionless power
    # spectrum' of 21 cm cosmology (which isn't really a dimensionless
    # power spectrum i.e. 21cmFast uses standard cosmological power
    # spectrum parlance instead of 21 cm cosmology naming conventions).

    # p_box[ct] += pow(k_mag,3)*pow(cabs(deldel_T[HII_C_INDEX(n_x, n_y, n_z)]), 2)/(2.0*PI*PI*VOLUME);
    ###

    VOLUME = p.box_size_21cmFAST_Mpc_sc**3. # Full cube volume in Mpc^3
    HII_TOT_NUM_PIXELS = p.box_size_21cmFAST_pix_sc**3. # Full cube Npix
    amplitude_normalisation_21cmFast = (VOLUME/HII_TOT_NUM_PIXELS)
    explicit_21cmFast_power_spectrum_normalisation = 1./(2.0*np.pi**2.*VOLUME)
    # amplitude_normalisation_21cmFast**2.
    # * explicit_power_spectrum_normalisation =
    # (VOLUME/HII_TOT_NUM_PIXELS)**2. / (2.0*np.pi**2.*VOLUME)
    full_21cmFast_power_spectrum_normalisation =\
        VOLUME / (2.0*np.pi**2.*HII_TOT_NUM_PIXELS**2.)
    full_power_spectrum_normalisation = (
            subset_power_spectrum_normalisation
            * full_21cmFast_power_spectrum_normalisation)

    dimensionless_powers_normalised = (
            full_power_spectrum_normalisation
            * np.array(dimensionless_powers))

    subset_ps_output_dir =\
        '/users/psims/EoR/Python_Scripts/BayesEoR/git_version/BayesEoR/'\
        'spec_model_tests/random/subset_ps/{}/'.format(
            EoR_npz_path.split('/')[-2].replace('.npz','').replace('.','d'))
    subset_ps_output_file =\
        EoR_npz_path.split('/')[-1].replace('.npz','').replace('.','d')
    subset_ps_output_path = subset_ps_output_dir+subset_ps_output_file

    print('Saving unnormalised dimensionless power spectrum to: \n', subset_ps_output_path)
    if not os.path.isdir(subset_ps_output_dir):
            os.makedirs(subset_ps_output_dir)
    np.savetxt(subset_ps_output_path,
               np.vstack((mean_ks, dimensionless_powers_normalised)).T)

    return np.vstack((mean_ks, dimensionless_powers_normalised)).T


def calculate_21cmFAST_EoR_cube_power_spectrum_in_subset_cube_bins_v1d0(
        nu, nv, nx, ny, nf, neta, nq, k_cube_voxels_in_bin,
        modkbins_containing_voxels, EoR_npz_path, mod_k, k_x, k_y, k_z):

    # Replace Gaussian signal with EoR cube
    scidata1 = np.load(EoR_npz_path)['arr_0']

    axes_tuple = (0, 1, 2)
    EoR_full_kcube = numpy.fft.ifftshift(
        scidata1-scidata1.mean() + 0j, axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    EoR_full_kcube = numpy.fft.fftn(EoR_full_kcube, axes=axes_tuple)
    EoR_full_kcube = numpy.fft.fftshift(EoR_full_kcube, axes=axes_tuple)

    chan_offset = 0
    # chan_offset = 100
    # chan_offset = 400
    # chan_offset = 250
    # chan_offset = 150
    # chan_offset = 350
    axes_tuple = (0, 1, 2)
    scidata1_kcube = numpy.fft.ifftshift(
        scidata1[chan_offset+0 : chan_offset+nf]-scidata1[0:nf].mean() + 0j,
        axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    scidata1_kcube = numpy.fft.fftn(scidata1_kcube, axes=axes_tuple)
    scidata1_kcube = numpy.fft.fftshift(scidata1_kcube, axes=axes_tuple)

    sci_f, sci_v, sci_u = scidata1_kcube.shape
    # Updated for python 3: floor division
    sci_v_centre = sci_v//2
    # Updated for python 3: floor division
    sci_u_centre = sci_u//2
    # Updated for python 3: floor division
    scidata1_kcube_subset = scidata1_kcube[
                            0: nf,
                            sci_u_centre - nu // 2: sci_u_centre + nu // 2 + 1,
                            sci_v_centre - nv // 2: sci_v_centre + nv // 2 + 1]

    for i in range(len(scidata1_kcube_subset)):
        print(i, scidata1_kcube_subset[i].std())

    ZM_vis_ordered_mask = np.ones(nu*nv*nf)
    # Updated for python 3: floor division
    ZM_vis_ordered_mask[nf*((nu*nv)//2) : nf*((nu*nv)//2 + 1)] = 0
    ZM_vis_ordered_mask = ZM_vis_ordered_mask.astype('bool')
    ZM_chan_ordered_mask = ZM_vis_ordered_mask.reshape(-1, nf).T.flatten()

    chan_ordered_bin_pixels_list = [np.where(
        np.logical_and(
            k_z.flatten()[ZM_chan_ordered_mask] != 0.0,
            np.logical_and(
                mod_k.flatten()[ZM_chan_ordered_mask]
                > modkbins_containing_voxels[i][0][0],
                mod_k.flatten()[ZM_chan_ordered_mask]
                <= modkbins_containing_voxels[i][0][1])
            )
        )
        for i in range(len(modkbins_containing_voxels))]

    excluded_zeroed_voxels =\
        np.where(scidata1_kcube_subset.flatten()[ZM_chan_ordered_mask] == 0.0)

    excluded_zeroed_voxel_locations_in_chan_ordered_bin_pixels_list =\
        [
            [
                np.where(chan_ordered_bin_pixels_list[i_bin][0]
                         == excluded_zeroed_voxels[0][i_ex])[0]
                for i_ex in range(len(excluded_zeroed_voxels[0]))
                if np.where(chan_ordered_bin_pixels_list[i_bin][0]
                            == excluded_zeroed_voxels[0][i_ex])[0]
                ]
            for i_bin in range(len(chan_ordered_bin_pixels_list))
            ]

    chan_ordered_bin_pixels_list_without_zeroed_excluded_pixels = \
        np.array(chan_ordered_bin_pixels_list).copy()
    chan_ordered_bin_pixels_list_without_zeroed_excluded_pixels = [np.delete(
        chan_ordered_bin_pixels_list[i_bin][0],
        excluded_zeroed_voxel_locations_in_chan_ordered_bin_pixels_list[i_bin])
        for i_bin in range(len(chan_ordered_bin_pixels_list))
        ]

    delta_k = 2.0*np.pi / p.EoR_analysis_cube_x_Mpc
    # Updated for python 3: floor division
    full_cube_k_z, full_cube_k_y, full_cube_k_x =\
        np.mgrid[
            -p.EoR_analysis_cube_x_pix//2 : p.EoR_analysis_cube_x_pix//2,
            -p.EoR_analysis_cube_x_pix//2 : p.EoR_analysis_cube_x_pix//2,
            -p.EoR_analysis_cube_x_pix//2 : p.EoR_analysis_cube_x_pix//2
        ] * delta_k
    full_cube_mod_k = (
        (full_cube_k_x**2.0 + full_cube_k_y**2.0 + full_cube_k_z**2.0)**0.5)
    u_equals_v_equals_zero_selector = np.logical_and(
        full_cube_k_x == 0.0, full_cube_k_y == 0.0)
    u_equals_v_equals_zero_mask = np.logical_not(
        u_equals_v_equals_zero_selector)

    k_x_min = k_x.min()
    k_x_max = k_x.max()
    k_y_min = k_y.min()
    k_y_max = k_y.max()
    k_xy_subset_selector = np.logical_and(
        np.logical_and(
            full_cube_k_x >= k_x_min,
            full_cube_k_x <= k_x_max),
        np.logical_and(
            full_cube_k_y >= k_y_min,
            full_cube_k_y <= k_y_max)
        )

    bin_limits_array = []
    subset_cube_mean_ks = []
    subset_cube_variances = []
    subset_cube_powers = []
    subset_cube_dimensionless_powers = []
    full_cube_mean_ks = []
    full_cube_variances = []
    full_cube_powers = []
    full_cube_dimensionless_powers = []

    for i in range(len(chan_ordered_bin_pixels_list)):
        subset_cube_ks_in_bin = mod_k.flatten()
        subset_cube_ks_in_bin = subset_cube_ks_in_bin[ZM_chan_ordered_mask]
        subset_cube_ks_in_bin =\
            subset_cube_ks_in_bin[
                chan_ordered_bin_pixels_list_without_zeroed_excluded_pixels[i]
            ]

        k_min = subset_cube_ks_in_bin.min()
        k_max = subset_cube_ks_in_bin.max()
        bin_limits = [k_min, k_max]
        bin_limits_array.append(bin_limits)

        subset_cube_amplitudes_in_bin = scidata1_kcube_subset.flatten()
        subset_cube_amplitudes_in_bin =\
            subset_cube_amplitudes_in_bin[
                ZM_chan_ordered_mask
            ]
        subset_cube_amplitudes_in_bin =\
            subset_cube_amplitudes_in_bin[
                chan_ordered_bin_pixels_list_without_zeroed_excluded_pixels[i]
            ]
        subset_cube_mean_k = subset_cube_ks_in_bin.mean()
        subset_cube_variance = subset_cube_amplitudes_in_bin.var()
        subset_cube_power = (abs(subset_cube_amplitudes_in_bin)**2.).mean()
        subset_cube_dimensionless_power = (
                subset_cube_ks_in_bin**3.
                * abs(subset_cube_amplitudes_in_bin)**2.
        ).mean()
        print(i, subset_cube_mean_k, subset_cube_variance,
              subset_cube_power, subset_cube_power/subset_cube_variance)

        subset_cube_mean_ks.append(subset_cube_mean_k)
        subset_cube_variances.append(subset_cube_variance)
        subset_cube_powers.append(subset_cube_power)
        subset_cube_dimensionless_powers.append(
            subset_cube_dimensionless_power)


        # full_cube_voxels_in_bin_indices = np.where(
        # 	np.logical_and(
        # 		full_cube_mod_k >= k_min,
        # 		full_cube_mod_k <= k_max
        # 		)
        # 	)
        spherical_annulus_selector = np.logical_and(
            u_equals_v_equals_zero_mask,np.logical_and(
                full_cube_mod_k >= k_min, full_cube_mod_k <= k_max
                )
            )
        spherical_annulus_selector = np.logical_and(
            k_xy_subset_selector, spherical_annulus_selector)

        full_cube_voxels_in_bin_indices = np.where(spherical_annulus_selector)
        full_cube_ks_in_bin = full_cube_mod_k[full_cube_voxels_in_bin_indices]
        full_cube_amplitudes_in_bin =\
            EoR_full_kcube[full_cube_voxels_in_bin_indices]
        full_cube_mean_k = full_cube_ks_in_bin.mean()
        full_cube_variance = full_cube_amplitudes_in_bin.var()
        full_cube_power = (abs(full_cube_amplitudes_in_bin)**2.).mean()
        full_cube_dimensionless_power = (
                full_cube_ks_in_bin**3.
                * abs(full_cube_amplitudes_in_bin)**2.
        ).mean()
        print(i, full_cube_mean_k, full_cube_variance,
              full_cube_power, full_cube_power/full_cube_variance)

        full_cube_mean_ks.append(full_cube_mean_k)
        full_cube_variances.append(full_cube_variance)
        full_cube_powers.append(full_cube_power)
        full_cube_dimensionless_powers.append(full_cube_dimensionless_power)

    subset_cube_powers = np.array(subset_cube_powers)
    full_cube_powers = np.array(full_cube_powers)

    subset_cube_dimensionless_powers =\
        np.array(subset_cube_dimensionless_powers)
    full_cube_dimensionless_powers = np.array(full_cube_dimensionless_powers)

    # print(subset_cube_powers/full_cube_powers)
    print(subset_cube_powers*(512./38)/full_cube_powers)
    print('Ratio of subset_dimensionless_power spectrum to the full EoR cube '
          'dimesionless power spectrum calculated in the same region of '
          'k-space as accessible to the subset cube (rather than over the '
          'full spherical annulus). All coefficients should be consistent '
          'with 1 to within sample variance:',
          subset_cube_dimensionless_powers * (512./38)
          / full_cube_dimensionless_powers)

    # Correctly normalize the dimensionless power spectrum.
    # Since I am doing a numpy fft (which follows the same convention as
    # fftw used in 21cmFast) the normalisation should be realtively
    # straightforward / similar to normalisation_21cmFast. Because I
    # start by taking a 38/512 subset in frequency, the ffted values
    # will be (38/512)**0.5 times smaller than the equivalent values in
    # the fft of the full cube so there will be an additional
    # (512/38.)**0.5 factor in amplitude relative to
    # normalisation_21cmFast
    subset_amplitude_normalisation = (512/38.)**0.5
    subset_power_spectrum_normalisation = subset_amplitude_normalisation**2.

    ###
    #     *((float *)deldel_T + HII_R_FFT_INDEX(i,j,k)) = (delta_T[HII_R_INDEX(i,j,k)]/ave - 1)*VOLUME/(HII_TOT_NUM_PIXELS+0.0);
    # if (DIMENSIONAL_T_POWER_SPEC){
    #   *((float *)deldel_T + HII_R_FFT_INDEX(i,j,k)) *= ave;
    # }

    # Note: DIMENSIONAL_T_POWER_SPEC in 21cmFast just means calculating
    # the standard mK^2 power spectrum i.e. the `dimensionless power
    # spectrum' of 21 cm cosmology (which isn't really a dimensionless
    # power spectrum i.e. 21cmFast uses standard cosmological power
    # spectrum parlance instead of 21 cm cosmology naming conventions).

    # p_box[ct] += pow(k_mag,3)*pow(cabs(deldel_T[HII_C_INDEX(n_x, n_y, n_z)]), 2)/(2.0*PI*PI*VOLUME);
    ###

    VOLUME = p.box_size_21cmFAST_Mpc_sc**3. #Full cube volume in Mpc^3
    HII_TOT_NUM_PIXELS = p.box_size_21cmFAST_pix_sc**3. #Full cube Npix
    amplitude_normalisation_21cmFast = (VOLUME/HII_TOT_NUM_PIXELS)
    explicit_21cmFast_power_spectrum_normalisation = 1./(2.0*np.pi**2.*VOLUME)
    # amplitude_normalisation_21cmFast**2.
    # * explicit_power_spectrum_normalisation =
    # (VOLUME/HII_TOT_NUM_PIXELS)**2. / (2.0*np.pi**2.*VOLUME)
    full_21cmFast_power_spectrum_normalisation = (
            VOLUME / (2.0*np.pi**2.*HII_TOT_NUM_PIXELS**2.))
    full_cube_dimensionless_powers_normalised = (
            full_21cmFast_power_spectrum_normalisation
            * full_cube_dimensionless_powers)

    subset_full_power_spectrum_normalisation = (
            subset_power_spectrum_normalisation
            * full_21cmFast_power_spectrum_normalisation)
    subset_cube_dimensionless_powers_normalised = (
            subset_full_power_spectrum_normalisation
            * subset_cube_dimensionless_powers)

    # subset_ps_output_dir =\
    # 	'/users/psims/EoR/Python_Scripts/BayesEoR/git_version/BayesEoR/'\
    # 	'spec_model_tests/random/subset_ps/{}/'.format(
    # 		EoR_npz_path.split('/')[-2].replace('.npz','').replace('.','d'))
    # subset_ps_output_file =\
    # 	EoR_npz_path.split('/')[-1].replace('.npz','').replace('.','d')
    # subset_ps_output_path = subset_ps_output_dir+subset_ps_output_file

    # print('Saving unnormalised dimensionless power spectrum to: \n',
    # 	  subset_ps_output_path)
    # if not os.path.isdir(subset_ps_output_dir):
    # 		os.makedirs(subset_ps_output_dir)
    # np.savetxt(subset_ps_output_path,
    # 		   np.vstack((mean_ks, dimensionless_powers_normalised)).T)

    return np.vstack(
                (subset_cube_mean_ks,
                 subset_cube_dimensionless_powers_normalised)).T,\
           np.vstack(
                (full_cube_mean_ks,
                 full_cube_dimensionless_powers_normalised)).T


def generate_data_from_loaded_EGS_cube(
        nu, nv, nx, ny, nf, neta, nq, k_x, k_y, k_z, Show,
        EGS_npz_path=('/users/psims/Cav/EoR/Missing_Radio_Flux/Surveys/'
                     'Flux_Variance_Maps/S_Cubed/'
                     'S_163_10nJy_Image_Cube_v34_18_deg_NV_15JyCN_With_'
                     'Synchrotron_Self_Absorption/Fits/Flux_Density_Upper'
                     '_Lim_15.0__Flux_Density_Lower_Lim_0.0/mk_cube/151_'
                     'Flux_values_10NanoJansky_limit_data_result_18_Degree'
                     '_Cube_RA_Dec_Degrees_and__10_pow_LogFlux_Columns_and_'
                     'Source_Redshifts_and_Source_SI_and_Source_AGN_Type_'
                     'Comb__mk.npz')):

    print('Using EGS foreground data')
    scidata1 = np.squeeze(np.load(EGS_npz_path)['arr_0'])
    # Take 12 deg. subset to match EoR cube
    scidata1 = scidata1[:, 10:10+512, 10:10+512]

    base_dir = 'Plots'
    save_dir = base_dir+'/Likelihood_v1d75_3D_ZM/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # scidata1 =\
    # 	top_hat_average_temperature_cube_to_lower_res_31x31xnf_cube(scidata1)
    scidata1 = scidata1[0:nf, :, :]

    # scidata1 = np.array([x-x.mean() for x in scidata1])

    import numpy
    axes_tuple = (1, 2)
    vfft1 = numpy.fft.ifftshift(
        scidata1[0:nf]-scidata1[0].mean() + 0j, axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    vfft1 = numpy.fft.fftn(vfft1, axes=axes_tuple)
    vfft1 = numpy.fft.fftshift(vfft1, axes=axes_tuple)

    sci_f, sci_v, sci_u = vfft1.shape
    # Updated for python 3: floor division
    sci_v_centre = sci_v//2
    # Updated for python 3: floor division
    sci_u_centre = sci_u//2
    # Updated for python 3: floor division
    vfft1_subset = vfft1[0 : nf,
                         sci_u_centre - nu//2 : sci_u_centre + nu//2 + 1,
                         sci_v_centre - nv//2 : sci_v_centre + nv//2 + 1]
    s_before_ZM = vfft1_subset.flatten() / vfft1[0].size**0.5
    # s_before_ZM = vfft1_subset.flatten()
    ZM_vis_ordered_mask = np.ones(nu*nv*nf)
    # Updated for python 3: floor division
    ZM_vis_ordered_mask[nf*((nu*nv)//2) : nf*((nu*nv)//2 + 1)] = 0
    ZM_vis_ordered_mask = ZM_vis_ordered_mask.astype('bool')
    ZM_chan_ordered_mask = ZM_vis_ordered_mask.reshape(-1, neta+nq).T.flatten()
    s = s_before_ZM[ZM_chan_ordered_mask]
    abc = s

    return s, abc, scidata1


def generate_data_from_loaded_EGS_cube_im_2_vis(
        nu, nv, nx, ny, nf, neta, nq, k_x, k_y, k_z, Show, EGS_npz_path, Finv):

    print('Using EGS foreground data')
    scidata1 = np.squeeze(np.load(EGS_npz_path)['arr_0'])
    # Take 12 deg. subset to match EoR cube
    scidata1 = scidata1[:, 10:10+512, 10:10+512]

    base_dir = 'Plots'
    save_dir = base_dir+'/Likelihood_v1d75_3D_ZM/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # scidata1 =\
    # 	top_hat_average_temperature_cube_to_lower_res_31x31xnf_cube(scidata1)
    scidata1 = scidata1[0:nf, :, :]

    import numpy
    axes_tuple = (1, 2)
    vfft1 = numpy.fft.ifftshift(
        scidata1[0:nf]-scidata1[0].mean() + 0j, axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    vfft1 = numpy.fft.fftn(vfft1, axes=axes_tuple)
    vfft1 = numpy.fft.fftshift(vfft1, axes=axes_tuple)

    sci_f, sci_v, sci_u = vfft1.shape
    # Updated for python 3: floor division
    sci_v_centre = sci_v//2
    # Updated for python 3: floor division
    sci_u_centre = sci_u//2
    # Updated for python 3: floor division
    vfft1_subset = vfft1[0 : nf,
                         sci_u_centre - nu//2 : sci_u_centre + nu//2 + 1,
                         sci_v_centre - nv//2 : sci_v_centre + nv//2 + 1]

    axes_tuple = (1, 2)
    scidata1_subset = numpy.fft.ifftshift(vfft1_subset + 0j, axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    scidata1_subset = numpy.fft.ifftn(scidata1_subset, axes=axes_tuple)
    scidata1_subset = numpy.fft.fftshift(scidata1_subset, axes=axes_tuple)

    s = np.dot(Finv,scidata1_subset.reshape(-1,1)).flatten()
    abc = s

    return s, abc, scidata1


def generate_data_from_loaded_EGS_cube_im_2_vis_v2d0(
        nu, nv, nx, ny, nf, neta, nq, k_x, k_y, k_z, Show, EGS_npz_path, Finv):

    print('Using EGS foreground data')
    scidata1 = np.squeeze(np.load(EGS_npz_path)['arr_0'])
    # Take 12 deg. subset to match EoR cube
    scidata1 = scidata1[:,10:10+512,10:10+512]
    # scidata1 = np.swapaxes(scidata1,1,2)

    base_dir = 'Plots'
    save_dir = base_dir+'/Likelihood_v1d75_3D_ZM/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # scidata1 =\
    # 	top_hat_average_temperature_cube_to_lower_res_31x31xnf_cube(scidata1)
    scidata1 = scidata1[0:nf, :, :]

    import numpy
    axes_tuple = (1, 2)
    vfft1 = numpy.fft.ifftshift(
        scidata1[0:nf]-scidata1[0].mean() + 0j, axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    vfft1 = numpy.fft.fftn(vfft1, axes=axes_tuple)
    vfft1 = numpy.fft.fftshift(vfft1, axes=axes_tuple)

    sci_f, sci_v, sci_u = vfft1.shape
    # Updated for python 3: floor division
    sci_v_centre = sci_v//2
    # Updated for python 3: floor division
    sci_u_centre = sci_u//2
    # Updated for python 3: floor division
    vfft1_subset = vfft1[0 : nf,
                         sci_u_centre - nu//2 : sci_u_centre + nu//2 + 1,
                         sci_v_centre - nv//2 : sci_v_centre + nv//2 + 1]

    axes_tuple = (1, 2)
    scidata1_subset = numpy.fft.ifftshift(vfft1_subset + 0j, axes=axes_tuple)
    # FFT (python pre-normalises correctly! -- see
    # parsevals theorem for discrete fourier transform.)
    scidata1_subset = numpy.fft.ifftn(scidata1_subset, axes=axes_tuple)
    scidata1_subset = numpy.fft.fftshift(scidata1_subset, axes=axes_tuple)

    # s = np.dot(Finv, scidata1_subset.reshape(-1, 1)).flatten()

    if not p.fit_for_monopole:
        # Remove channel means for interferometric image (not required
        # for a gridded model but necessary when using a large FWHM PB
        # model or small model FoV due to the non-closing nudft (i.e.
        # for fringes with a non-integer number of wavelengths across
        # the model image))
        d_im = scidata1_subset.copy()
        for i_chan in range(len(d_im)):
            d_im[i_chan] = d_im[i_chan]-d_im[i_chan].mean()
        # s2 = np.dot(Finv, d_im.reshape(-1, 1)).flatten()
        s = np.dot(Finv, d_im.reshape(-1, 1)).flatten()
    else:
        s = np.dot(Finv, scidata1_subset.reshape(-1, 1)).flatten()
    abc = s

    return s, abc, scidata1


def generate_test_signal_from_image_cube(
        nu, nv, nx, ny, nf, neta, nq, image_cube_mK, output_fits_file_name):

    beta_experimental_mean = 2.55+0 # Revise to match published values
    beta_experimental_std = 0.1 # Revise to match published values
    gamma_mean = -2.7 # Revise to match published values
    gamma_sigma = 0.3 # R evise to match published values
    Tb_experimental_mean_K = 194.0 # Matches GSM mean in region A
    # (see /users/psims/Cav/EoR/Missing_Radio_Flux/Surveys/Convert_GSM_
    #  to_HEALPIX_Map_and_Cartesian_Projection_Fits_File_v6d0_pygsm.py)
    Tb_experimental_std_K = 23.0 # Matches GSM in region A at 0.333
    # degree resolution (i.e. for a 50 degree map 150 pixels across).
    # Note: std. is a function of resultion so the foreground map should
    # be made at the same resolution for this std normalisation to be
    # accurate.
    # Tb_experimental_mean_K = 240.0 # Revise to match published values
    # Tb_experimental_std_K  = 4.0 # Revise to match published values
    nu_min_MHz = 120.0
    channel_width_MHz = 0.2

    GFC = GenerateForegroundCube(
        nu, nv, neta, nq, beta_experimental_mean, beta_experimental_std,
        gamma_mean, gamma_sigma, Tb_experimental_mean_K, Tb_experimental_std_K,
        nu_min_MHz, channel_width_MHz)

    Tb_nu, A, beta, Tb, nu_array_MHz =\
        GFC.generate_normalised_Tb_A_and_beta_fields(
            513, 513, 513, 513, nf, neta, nq)

    ED = ExtractDataFrom21cmFASTCube(plot_data=False)
    cube_side_Mpc = 2048.0
    redshift = 10.26
    bandwidth_MHz, central_frequency_MHz, output_21cmFast_box_width_deg =\
        ED.calculate_box_size_in_degrees_and_MHz(cube_side_Mpc, redshift)

    DUC = DataUnitConversionmKAndJyPerPix()

    Data_mK = image_cube_mK
    # Data_mK = Data_mK + random_quad
    Channel_Frequencies_Array_Hz = nu_array_MHz * 1.e6
    Pixel_Height_rad = (output_21cmFast_box_width_deg * (np.pi/180.)
                        / Tb_nu.shape[1])
    DUC.convert_from_mK_to_Jy_per_pix(
        Data_mK, Channel_Frequencies_Array_Hz, Pixel_Height_rad)
    Data_Jy_per_Pixel = ED.convert_from_mK_to_Jy_per_pix(
        Data_mK, Channel_Frequencies_Array_Hz, Pixel_Height_rad)

    # output_fits_file_name = 'Quad_only.fits'
    # output_fits_file_name = 'White_noise_cube_with_quad.fits'
    # output_fits_file_name = 'Jelic_GDSE_cube.fits'
    output_fits_path = 'fits_storage/'+output_fits_file_name
    print(output_fits_path)
    WD2F = WriteDataToFits()
    WD2F.write_data(
        Data_Jy_per_Pixel, output_fits_path,
        Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)


def Generate_Poisson_Distribution_with_Exact_Input_Mean(
        Mean, N_Data_Points, **kwargs):
    """
        Generate values from a Poisson distribution and get a resulting
        distribution with exactly the input mean.
    """
    Mean=float(Mean)
    P = stats.poisson.rvs(Mean, size=N_Data_Points)
    Target_Sum=int(N_Data_Points*Mean)

    Print=True
    for key in kwargs:
        if key == ('Target_Sum'):
            Target_Sum = kwargs[key]
        if key == ('Print'):
            Print = kwargs[key]

    if Print:
        print('Target_Sum', Target_Sum)
    while sum(P) > Target_Sum:
        P = stats.poisson.rvs(Mean, size=N_Data_Points)

    Sum_Current_P_Dist = sum(P)
    if Print:
        print('Sum_Current_P_Dist', Sum_Current_P_Dist)
    while Sum_Current_P_Dist != Target_Sum:
        # Updated for python 3: float division now default
        Updated_Mean = (Target_Sum-Sum_Current_P_Dist) / N_Data_Points

        Sum_Current_P_Dist = sum(P)
        Potential_Supplementary_P = stats.poisson.rvs(
            abs(Updated_Mean), size=N_Data_Points)
        while ((Sum_Current_P_Dist + sum(Potential_Supplementary_P))
               > Target_Sum):
            Potential_Supplementary_P = \
                stats.poisson.rvs(abs(Updated_Mean), size=N_Data_Points)

        P += Potential_Supplementary_P
        Sum_Current_P_Dist = sum(P)
        if Print:
            print('Sum_Current_P_Dist', Sum_Current_P_Dist)

    return P
