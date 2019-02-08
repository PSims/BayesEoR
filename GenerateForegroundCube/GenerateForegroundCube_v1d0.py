
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
# sys.path.append(os.path.expanduser('~/EoR/Python_Scripts/BayesEoR/SpatialPS/PolySpatialPS/'))

from BayesEoR.SimData import GenerateForegroundCube, update_Tb_experimental_std_K_to_correct_for_normalisation_resolution
from BayesEoR.Utils import PriorC, ParseCommandLineArguments, DataUnitConversionmkandJyperpix, WriteDataToFits
from BayesEoR.Utils import ExtractDataFrom21cmFASTCube

import BayesEoR.Params.params as p

#----------------------

use_foreground_cube = True
# use_foreground_cube = False



def generate_Jelic_cube(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show, beta_experimental_mean,beta_experimental_std,gamma_mean,gamma_sigma,Tb_experimental_mean_K,Tb_experimental_std_K,nu_min_MHz,channel_width_MHz, **kwargs):

	##===== Defaults =======
	default_generate_additional_extrapolated_HF_foreground_cube = False
	default_HF_nu_min_MHz = 225
	default_fits_storage_dir = 'fits_storage/Jelic/'
	default_HF_nu_min_MHz_array = [205,215,225]
	default_simulation_FoV_deg = 12.0
	default_simulation_resolution_deg = 12.0/127
	default_random_seed = 3142
	default_cube_side_Mpc = 2048.0 #Size of EoR cube foreground simulation should match (used when calculating fits header variables)
	default_redshift = 7.6 #Redshift of EoR cube foreground simulation should match (used when calculating fits header variables)

	
	##===== Inputs =======
	generate_additional_extrapolated_HF_foreground_cube=kwargs.pop('generate_additional_extrapolated_HF_foreground_cube',default_generate_additional_extrapolated_HF_foreground_cube)
	HF_nu_min_MHz=kwargs.pop('HF_nu_min_MHz',default_HF_nu_min_MHz)
	fits_storage_dir=kwargs.pop('fits_storage_dir',default_fits_storage_dir)
	HF_nu_min_MHz_array=kwargs.pop('HF_nu_min_MHz_array',default_fits_storage_dir)
	simulation_FoV_deg=kwargs.pop('simulation_FoV_deg',default_simulation_FoV_deg)
	simulation_resolution_deg=kwargs.pop('simulation_resolution_deg',default_simulation_resolution_deg)
	random_seed=kwargs.pop('random_seed',default_random_seed)
	cube_side_Mpc=kwargs.pop('random_seed',default_cube_side_Mpc)
	redshift=kwargs.pop('random_seed',default_redshift)

	n_sim_pix = int(simulation_FoV_deg/simulation_resolution_deg + 0.5)

	low_res_to_high_res_std_conversion_factor = update_Tb_experimental_std_K_to_correct_for_normalisation_resolution(Tb_experimental_std_K, simulation_FoV_deg, simulation_resolution_deg)
	Tb_experimental_std_K = Tb_experimental_std_K*low_res_to_high_res_std_conversion_factor

	GFC = GenerateForegroundCube(nu,nv,neta,nq, beta_experimental_mean, beta_experimental_std, gamma_mean, gamma_sigma, Tb_experimental_mean_K, Tb_experimental_std_K, nu_min_MHz, channel_width_MHz,random_seed=random_seed)

	Tb_nu, A, beta, Tb, nu_array_MHz = GFC.generate_normalised_Tb_A_and_beta_fields(n_sim_pix,n_sim_pix,n_sim_pix,n_sim_pix,nf,neta,nq)
	# Tb_nu, A, beta, Tb, nu_array_MHz = GFC.generate_normalised_Tb_A_and_beta_fields(513,513,513,513,nf,neta,nq)
	# Tb_nu, A, beta, Tb, nu_array_MHz = GFC.generate_normalised_Tb_A_and_beta_fields(nu,nv,nx,ny,nf,neta,nq)
	# Tb_nu2 = np.array([Tb_nu[0]*(nu_array_MHz[i]/nu_array_MHz[0])**-beta_experimental_mean for i in range(len(nu_array_MHz))])


	if generate_additional_extrapolated_HF_foreground_cube:
		# HF_nu_min_MHz = 225
		# HF_nu_min_MHz_array = [205,215,225]
		for HF_nu_min_MHz_i in range(len(HF_nu_min_MHz_array)):
			HF_nu_min_MHz = HF_nu_min_MHz_array[HF_nu_min_MHz_i]
			HF_Tb_nu = generate_additional_HF_Jelic_cube(A,HF_nu_min_MHz,beta,fits_storage_dir,nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show, beta_experimental_mean,beta_experimental_std,gamma_mean,gamma_sigma,Tb_experimental_mean_K,Tb_experimental_std_K,nu_min_MHz,channel_width_MHz,cube_side_Mpc=cube_side_Mpc,redshift=redshift)

	else:
		HF_Tb_nu = []


	beta=A=Tb=[]

	###

	ED = ExtractDataFrom21cmFASTCube(plot_data=False)
	bandwidth_MHz, central_frequency_MHz, output_21cmFast_box_width_deg = ED.calculate_box_size_in_degrees_and_MHz(cube_side_Mpc, redshift)
	
	DUC = DataUnitConversionmkandJyperpix()
	Data_mK = Tb_nu*1.e3

	
	Channel_Frequencies_Array_Hz = nu_array_MHz*1.e6
	Pixel_Height_rad = output_21cmFast_box_width_deg*(np.pi/180.)/Tb_nu.shape[1]
	DUC.convert_from_mK_to_Jy_per_pix(Data_mK, Channel_Frequencies_Array_Hz, Pixel_Height_rad)
	Data_Jy_per_Pixel = ED.convert_from_mK_to_Jy_per_pix(Data_mK, Channel_Frequencies_Array_Hz, Pixel_Height_rad)
	
	output_fits_file_name = 'Jelic_GDSE_cube_{:d}MHz.fits'.format(int(nu_min_MHz))
	output_fits_path1 = fits_storage_dir+output_fits_file_name
	output_fits_path2 = fits_storage_dir+'/ZNPS{:d}/'.format(int(nu_min_MHz))+output_fits_file_name
	print output_fits_path1, '\n'+ output_fits_path2
	WD2F = WriteDataToFits()
	WD2F.write_data(Data_Jy_per_Pixel, output_fits_path1, Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)
	WD2F.write_data(Data_Jy_per_Pixel, output_fits_path2, Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)

	output_fits_file_name = 'Jelic_GDSE_cube_{:d}MHz_mK.fits'.format(int(nu_min_MHz))
	output_fits_path1 = fits_storage_dir+output_fits_file_name
	output_fits_path2 = fits_storage_dir+'/ZNPS{:d}/'.format(int(nu_min_MHz))+output_fits_file_name
	print output_fits_path1, '\n'+ output_fits_path2
	WD2F = WriteDataToFits()
	WD2F.write_data(Data_mK, output_fits_path1, Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)
	WD2F.write_data(Data_mK, output_fits_path2, Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)

	###

	

	import pylab
	pylab.figure()
	pylab.imshow(Tb_nu[0], cmap=pylab.cm.jet)
	pylab.colorbar(fraction=(0.315*0.15), pad=0.01)
	pylab.figure()
	pylab.imshow(Tb_nu[-1], cmap=pylab.cm.jet)
	pylab.colorbar(fraction=(0.315*0.15), pad=0.01)
	if Show: pylab.show()

	import pylab
	pylab.errorbar(log10(nu_array_MHz), log10(Tb_nu[:,0,0]))
	pylab.errorbar(log10(nu_array_MHz), log10(Tb_nu[:,1,1]))
	pylab.errorbar(log10(nu_array_MHz), log10(Tb_nu[:,2,2]))
	if Show: pylab.show()

	pylab.close('all')
	# Show=False

	###
	# Inspect quadratic residuals
	###
	quad_coeffs = np.polyfit(nu_array_MHz, Tb_nu[:,0,0], 2)
	quad_fit = quad_coeffs[0]*nu_array_MHz**2. + quad_coeffs[1]*nu_array_MHz**1. + quad_coeffs[2]*nu_array_MHz**0. 
	residuals = Tb_nu[:,0,0]-quad_fit

	# quad_coeffs2 = np.polyfit(nu_array_MHz, Tb_nu2[:,0,0], 2)
	# quad_fit2 = quad_coeffs2[0]*nu_array_MHz**2. + quad_coeffs2[1]*nu_array_MHz**1. + quad_coeffs2[2]*nu_array_MHz**0. 
	# residuals2 = Tb_nu2[:,0,0]-quad_fit2

	base_dir = 'Plots'
	save_dir = base_dir+'/Likelihood_v1d75_3D_ZM/'
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)


	import numpy
	axes_tuple = (0,)
	res_fft=numpy.fft.ifftshift(residuals+0j, axes=axes_tuple)
	res_fft=numpy.fft.fftn(res_fft, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	res_fft=numpy.fft.fftshift(res_fft, axes=axes_tuple)

	fig,ax = pylab.subplots(nrows=1, ncols=1, figsize=(20,20))
	ax.errorbar(log10(nu_array_MHz)[np.fft.fftshift(np.fft.fftfreq(nf))!=0], log10(abs(res_fft))[np.fft.fftshift(np.fft.fftfreq(nf))!=0],  color='red', fmt='-')
	# ax.errorbar(log10(nu_array_MHz)[np.fft.fftshift(np.fft.fftfreq(38))!=0], log10(abs(res_fft))[np.fft.fftshift(np.fft.fftfreq(38))!=0],  color='red', fmt='-')
	ax.set_ylabel('log(Amplitude)', fontsize=20)
	ax.set_xlabel('log-fftfreq', fontsize=20)
	ax.tick_params(labelsize=20)
	fig.savefig(save_dir+'foreground_quadsub_residuals_ffted.png')
	if Show:fig.show()




	print 'Using use_foreground_cube data'
	#----------------------
	###
	# Replace Gaussian signal with foreground cube
	###
	# Tb_nu_mK = Tb_nu*1.e2
	Tb_nu_mK = Tb_nu*1.e3
	scidata1 = Tb_nu_mK
	# scidata1 = random_quad[0:nf,0:nu,0:nv]

	
	base_dir = 'Plots'
	save_dir = base_dir+'/Likelihood_v1d75_3D_ZM/'
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)

	import numpy
	axes_tuple = (1,2)
	vfft1=numpy.fft.ifftshift(scidata1[0:nf]-scidata1[0].mean()+0j, axes=axes_tuple)
	vfft1=numpy.fft.fftn(vfft1, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	vfft1=numpy.fft.fftshift(vfft1, axes=axes_tuple)

	import pylab
	pylab.figure()
	pylab.imshow(abs(vfft1[0]), cmap=pylab.cm.jet)
	pylab.colorbar(fraction=(0.315*0.15), pad=0.01)
	if Show: pylab.show()

	pylab.close('all')
	
	sci_f, sci_v, sci_u = vfft1.shape
	sci_v_centre = sci_v/2
	sci_u_centre = sci_u/2
	vfft1_subset = vfft1[0:nf,sci_u_centre-nu/2:sci_u_centre+nu/2+1,sci_v_centre-nv/2:sci_v_centre+nv/2+1]
	s_before_ZM = vfft1_subset.flatten()/vfft1[0].size**0.5
	ZM_vis_ordered_mask = np.ones(nu*nv*nf)
	ZM_vis_ordered_mask[nf*((nu*nv)/2):nf*((nu*nv)/2+1)]=0
	ZM_vis_ordered_mask = ZM_vis_ordered_mask.astype('bool')
	ZM_chan_ordered_mask = ZM_vis_ordered_mask.reshape(-1, neta+nq).T.flatten()
	s = s_before_ZM[ZM_chan_ordered_mask]

	fg = s

#----------------------


	###
	# Inspect quadratic residuals in uv
	###
	dat1 = fg.reshape(-1,nu*nv-1)[:,0].real[::-1]
	quad_coeffs = np.polyfit(nu_array_MHz, dat1, 2)
	quad_fit = quad_coeffs[0]*nu_array_MHz**2. + quad_coeffs[1]*nu_array_MHz**1. + quad_coeffs[2]*nu_array_MHz**0. 
	residuals = dat1-quad_fit

	dat2 = fg.reshape(-1,nu*nv-1)[:,0].imag[::-1]
	quad_coeffs2 = np.polyfit(nu_array_MHz, dat2, 2)
	quad_fit2 = quad_coeffs2[0]*nu_array_MHz**2. + quad_coeffs2[1]*nu_array_MHz**1. + quad_coeffs2[2]*nu_array_MHz**0. 
	residuals2 = dat2-quad_fit2

	base_dir = 'Plots'
	save_dir = base_dir+'/Likelihood_v1d75_3D_ZM/'
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)

	fig,ax = pylab.subplots(nrows=2, ncols=1, figsize=(20,20))
	ax[0].errorbar((nu_array_MHz), (residuals),  color='red')
	ax[0].errorbar((nu_array_MHz), (residuals2),  color='blue', fmt='-')
	ax[0].legend(['Quad residuals real', 'Quad residuals imag'], fontsize=20)
	ax[0].set_ylabel('Amplitude, arbitrary units', fontsize=20)
	ax[0].set_xlabel('Frequency, MHz', fontsize=20)
	ax[1].errorbar(log10(nu_array_MHz), log10(fg.reshape(-1,nu*nv-1)[:,0].real),  color='red', fmt='-')
	ax[1].errorbar(log10(nu_array_MHz), log10(fg.reshape(-1,nu*nv-1)[:,0].imag),  color='blue', fmt='-')
	ax[1].legend(['real','imag'], fontsize=20)
	ax[1].set_ylabel('log(Amplitude), arbitrary units', fontsize=20)
	ax[1].set_xlabel('log(Frequency), MHz', fontsize=20)
	for axi in ax.ravel(): axi.tick_params(labelsize=20)
	# fig.savefig(save_dir+'foreground_quadsub_residuals.png')
	if Show:fig.show()


	import numpy
	axes_tuple = (0,)
	res_fft=numpy.fft.ifftshift(residuals+0j, axes=axes_tuple)
	res_fft=numpy.fft.fftn(res_fft, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	res_fft=numpy.fft.fftshift(res_fft, axes=axes_tuple)


	
	k_cubed_log_res_fft = (log10(abs(res_fft*k_z[:,0,0]**3.)))[np.fft.fftshift(np.fft.fftfreq(nf))!=0]

	fig,ax = pylab.subplots(nrows=1, ncols=1, figsize=(20,20))
	ax.errorbar(log10(nu_array_MHz)[np.fft.fftshift(np.fft.fftfreq(nf))!=0], k_cubed_log_res_fft,  color='red', fmt='-')
	ax.set_ylabel('log(Amplitude)', fontsize=20)
	ax.set_xlabel('log-fftfreq', fontsize=20)
	ax.tick_params(labelsize=20)
	# fig.savefig(save_dir+'foreground_quadsub_residuals_ffted.png')
	if Show:fig.show()


	a1 = log10(k_z[:,0,0]**3.)[20:]+4.16
	a2 = (log10(abs(res_fft)))[20:]+1.33
	print a1+a2-0.8
	

	import numpy
	axes_tuple = (0,)
	res_fft=numpy.fft.ifftshift(residuals+0j, axes=axes_tuple)
	res_fft=numpy.fft.fftn(res_fft, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	res_fft=numpy.fft.fftshift(res_fft, axes=axes_tuple)
	
	k_cubed_log_res_fft = (log10(abs(res_fft)))[np.fft.fftshift(np.fft.fftfreq(nf))!=0]
	# k_cubed_log_res_fft = (log10(abs(res_fft*k_z[:,0,0]**3.)))[np.fft.fftshift(np.fft.fftfreq(38))!=0]
	fig,ax = pylab.subplots(nrows=1, ncols=1, figsize=(20,20))
	ax.errorbar(log10(nu_array_MHz)[np.fft.fftshift(np.fft.fftfreq(nf))!=0], k_cubed_log_res_fft,  color='red', fmt='-')
	# ax.errorbar(log10(nu_array_MHz)[np.fft.fftshift(np.fft.fftfreq(38))!=0], k_cubed_log_res_fft,  color='red', fmt='-')
	ax.set_ylabel('log(Amplitude)', fontsize=20)
	ax.set_xlabel('log-fftfreq', fontsize=20)
	ax.tick_params(labelsize=20)
	# fig.savefig(save_dir+'foreground_quadsub_residuals_ffted.png')
	if Show:fig.show()

	return fg, s, Tb_nu, beta_experimental_mean,beta_experimental_std,gamma_mean,gamma_sigma,Tb_experimental_mean_K,Tb_experimental_std_K,nu_min_MHz,channel_width_MHz, HF_Tb_nu


## ======================================================================================================
## ======================================================================================================

def generate_additional_HF_Jelic_cube(A,HF_nu_min_MHz,beta,fits_storage_dir,nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show, beta_experimental_mean,beta_experimental_std,gamma_mean,gamma_sigma,Tb_experimental_mean_K,Tb_experimental_std_K,nu_min_MHz,channel_width_MHz, **kwargs):

	# HF_nu_min_MHz = 225
	HF_nu_array_MHz = HF_nu_min_MHz+np.arange(nf)*channel_width_MHz
	HF_A_nu = np.array([A*(HF_nu_array_MHz[i_nu]/nu_min_MHz)**-beta for i_nu in range(len(HF_nu_array_MHz))])
	HF_Tb_nu = np.sum(HF_A_nu, axis=1)

	ED = ExtractDataFrom21cmFASTCube(plot_data=False)
	cube_side_Mpc = 3072.0
	redshift = 7.6
	# cube_side_Mpc = 2048.0
	# redshift = 10.26
	bandwidth_MHz, central_frequency_MHz, output_21cmFast_box_width_deg = ED.calculate_box_size_in_degrees_and_MHz(cube_side_Mpc, redshift)
	
	DUC = DataUnitConversionmkandJyperpix()
	Data_mK = HF_Tb_nu*1.e3
	
	Channel_Frequencies_Array_Hz = HF_nu_array_MHz*1.e6
	Pixel_Height_rad = output_21cmFast_box_width_deg*(np.pi/180.)/HF_Tb_nu.shape[1]
	DUC.convert_from_mK_to_Jy_per_pix(Data_mK, Channel_Frequencies_Array_Hz, Pixel_Height_rad)
	Data_Jy_per_Pixel = ED.convert_from_mK_to_Jy_per_pix(Data_mK, Channel_Frequencies_Array_Hz, Pixel_Height_rad)
	
	output_fits_file_name = 'Jelic_GDSE_cube_{:d}MHz_mK.fits'.format(int(HF_nu_min_MHz))
	output_fits_path1 = fits_storage_dir+output_fits_file_name
	output_fits_path2 = fits_storage_dir+'/ZNPS{:d}/'.format(int(HF_nu_min_MHz))+output_fits_file_name
	print output_fits_path1, '\n'+ output_fits_path2
	WD2F = WriteDataToFits()
	WD2F.write_data(Data_Jy_per_Pixel, output_fits_path1, Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)
	WD2F.write_data(Data_Jy_per_Pixel, output_fits_path2, Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)
	
	output_fits_file_name = 'Jelic_GDSE_cube_{:d}MHz_mK.fits'.format(int(HF_nu_min_MHz))
	output_fits_path1 = fits_storage_dir+output_fits_file_name
	output_fits_path2 = fits_storage_dir+'/ZNPS{:d}/'.format(int(HF_nu_min_MHz))+output_fits_file_name
	print output_fits_path1, '\n'+ output_fits_path2
	WD2F = WriteDataToFits()
	WD2F.write_data(Data_mK, output_fits_path1, Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)
	WD2F.write_data(Data_mK, output_fits_path2, Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)

	return HF_Tb_nu



## ======================================================================================================
## ======================================================================================================

def top_hat_average_temperature_cube_to_lower_res_31x31xnf_cube(Tb_nu):
	bl_size = Tb_nu[0].shape[0]/31
	dat_subset = 31*bl_size
	averaged_cube = []
	for i_freq in range(len(Tb_nu)):
		# Tb_nu_bls = np.array([Tb_nu[0][:496,:496][i*bl_size:(i+1)*bl_size,j*bl_size:(j+1)*bl_size] for i in range(31) for j in range(31)])
		Tb_nu_bls = np.array([Tb_nu[i_freq][:dat_subset,:dat_subset][i*bl_size:(i+1)*bl_size,j*bl_size:(j+1)*bl_size] for i in range(31) for j in range(31)])
		Tb_nu_bls_means = np.array([x.mean() for x in Tb_nu_bls]).reshape(31,31)
		averaged_cube.append(Tb_nu_bls_means)

	averaged_cube = np.array(averaged_cube)
	return averaged_cube



## ======================================================================================================
## ======================================================================================================

def generate_data_from_loaded_EoR_cube(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show,chan_selection):

	print 'Using use_EoR_cube data'
	#----------------------
	###
	# Replace Gaussian signal with EoR cube
	###
	scidata1 = np.load('/users/psims/EoR/EoR_simulations/21cmFAST_512MPc_512pix_128pix/Fits/21cm_z10d2_mK.npz')['arr_0']

	base_dir = 'Plots'
	save_dir = base_dir+'/Likelihood_v1d75_3D_ZM/'
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)

	# scidata1 = top_hat_average_temperature_cube_to_lower_res_31x31xnf_cube(scidata1)
	scidata1 = scidata1[0:nf,:124,:124]

	import numpy
	axes_tuple = (1,2)
	if chan_selection=='0_38_':
		vfft1=numpy.fft.ifftshift(scidata1[0:38]-scidata1[0].mean()+0j, axes=axes_tuple)
	elif chan_selection=='38_76_':
		vfft1=numpy.fft.ifftshift(scidata1[38:76]-scidata1[0].mean()+0j, axes=axes_tuple)
	elif chan_selection=='76_114_':
		vfft1=numpy.fft.ifftshift(scidata1[76:114]-scidata1[0].mean()+0j, axes=axes_tuple)
	else:
		vfft1=numpy.fft.ifftshift(scidata1[0:nf]-scidata1[0].mean()+0j, axes=axes_tuple)
	vfft1=numpy.fft.fftn(vfft1, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	vfft1=numpy.fft.fftshift(vfft1, axes=axes_tuple)


	sci_f, sci_v, sci_u = vfft1.shape
	sci_v_centre = sci_v/2
	sci_u_centre = sci_u/2
	vfft1_subset = vfft1[0:nf,sci_u_centre-nu/2:sci_u_centre+nu/2+1,sci_v_centre-nv/2:sci_v_centre+nv/2+1]
	s_before_ZM = vfft1_subset.flatten()/vfft1[0].size**0.5
	ZM_vis_ordered_mask = np.ones(nu*nv*nf)
	ZM_vis_ordered_mask[nf*((nu*nv)/2):nf*((nu*nv)/2+1)]=0
	ZM_vis_ordered_mask = ZM_vis_ordered_mask.astype('bool')
	ZM_chan_ordered_mask = ZM_vis_ordered_mask.reshape(-1, neta+nq).T.flatten()
	s = s_before_ZM[ZM_chan_ordered_mask]

	abc = s

	return s, abc, scidata1



## ======================================================================================================
## ======================================================================================================


def generate_data_from_loaded_EoR_cube_v2d0(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show,chan_selection,EoR_npz_path='/users/psims/EoR/EoR_simulations/21cmFAST_512MPc_512pix_128pix/Fits/21cm_z10d2_mK.npz'):

	print 'Using use_EoR_cube data'
	#----------------------
	###
	# Replace Gaussian signal with EoR cube
	###
	scidata1 = np.load(EoR_npz_path)['arr_0']

	base_dir = 'Plots'
	save_dir = base_dir+'/Likelihood_v1d75_3D_ZM/'
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)

	# scidata1 = top_hat_average_temperature_cube_to_lower_res_31x31xnf_cube(scidata1)
	# scidata1 = scidata1[0:nf,:124,:124]

	np.random.seed(12345)
	scidata1 = np.random.normal(0,scidata1.std()*3.,scidata1.shape)

	import numpy
	axes_tuple = (1,2)
	if chan_selection=='0_38_':
		vfft1=numpy.fft.ifftshift(scidata1[0:38]-scidata1[0].mean()+0j, axes=axes_tuple)
	elif chan_selection=='38_76_':
		vfft1=numpy.fft.ifftshift(scidata1[38:76]-scidata1[0].mean()+0j, axes=axes_tuple)
	elif chan_selection=='76_114_':
		vfft1=numpy.fft.ifftshift(scidata1[76:114]-scidata1[0].mean()+0j, axes=axes_tuple)
	else:
		vfft1=numpy.fft.ifftshift(scidata1[0:nf]-scidata1[0].mean()+0j, axes=axes_tuple)
	vfft1=numpy.fft.fftn(vfft1, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	vfft1=numpy.fft.fftshift(vfft1, axes=axes_tuple)


	sci_f, sci_v, sci_u = vfft1.shape
	sci_v_centre = sci_v/2
	sci_u_centre = sci_u/2
	vfft1_subset = vfft1[0:nf,sci_u_centre-nu/2:sci_u_centre+nu/2+1,sci_v_centre-nv/2:sci_v_centre+nv/2+1]
	s_before_ZM = vfft1_subset.flatten()/vfft1[0].size**0.5
	# s_before_ZM = vfft1_subset.flatten()
	ZM_vis_ordered_mask = np.ones(nu*nv*nf)
	ZM_vis_ordered_mask[nf*((nu*nv)/2):nf*((nu*nv)/2+1)]=0
	ZM_vis_ordered_mask = ZM_vis_ordered_mask.astype('bool')
	ZM_chan_ordered_mask = ZM_vis_ordered_mask.reshape(-1, neta+nq).T.flatten()
	s = s_before_ZM[ZM_chan_ordered_mask]

	abc = s

	return s, abc, scidata1



## ======================================================================================================
## ======================================================================================================

def generate_white_noise_signal(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,T,Show,chan_selection,masked_power_spectral_modes):

	print 'Using use_WN_cube data'


	EoR_npz_path = p.EoR_npz_path


	#----------------------
	###
	# Replace Gaussian signal with EoR cube
	###
	scidata1 = np.load(EoR_npz_path)['arr_0']

	#Overwrite EoR cube with white noise
	# np.random.seed(21287254)
	np.random.seed(123)
	scidata1 = np.random.normal(0,scidata1.std()*1.,scidata1.shape)*0.5

	base_dir = 'Plots'
	save_dir = base_dir+'/Likelihood_v1d75_3D_ZM/'
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)


	axes_tuple = (0,1,2)
	scidata1_kcube=numpy.fft.ifftshift(scidata1[0:38]-scidata1[0:38].mean()+0j, axes=axes_tuple)
	scidata1_kcube=numpy.fft.fftn(scidata1_kcube, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	scidata1_kcube=numpy.fft.fftshift(scidata1_kcube, axes=axes_tuple)


	sci_f, sci_v, sci_u = scidata1_kcube.shape
	sci_v_centre = sci_v/2
	sci_u_centre = sci_u/2
	scidata1_kcube_subset = scidata1_kcube[0:nf,sci_u_centre-nu/2:sci_u_centre+nu/2+1,sci_v_centre-nv/2:sci_v_centre+nv/2+1]
	scidata1_kcube_subset_before_ZM = scidata1_kcube_subset.flatten()/scidata1_kcube.size**0.5
	ZM_vis_ordered_mask = np.ones(nu*nv*nf)
	ZM_vis_ordered_mask[nf*((nu*nv)/2):nf*((nu*nv)/2+1)]=0
	ZM_vis_ordered_mask = ZM_vis_ordered_mask.astype('bool')
	ZM_chan_ordered_mask = ZM_vis_ordered_mask.reshape(-1, neta+nq).T.flatten()
	scidata1_kcube_subset_ZM = scidata1_kcube_subset_before_ZM[ZM_chan_ordered_mask]

	###
	# Zero all modes that correspond to / are replaced with foreground parameters (in the parameter vector that T is desiged to operate on) before applying T!
	###
	scidata1_kcube_subset_ZM[masked_power_spectral_modes] = 0.0

	# T = BM.read_data_from_hdf5(array_save_directory+'T.h5', 'T')
	s = np.dot(T,scidata1_kcube_subset_ZM.reshape(-1,1)).flatten()
	abc = s

	return s, abc, scidata1



## ======================================================================================================
## ======================================================================================================


def generate_data_from_loaded_EGS_cube(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show,EGS_npz_path='/users/psims/Cav/EoR/Missing_Radio_Flux/Surveys/Flux_Variance_Maps/S_Cubed/S_163_10nJy_Image_Cube_v34_18_deg_NV_15JyCN_With_Synchrotron_Self_Absorption/Fits/Flux_Density_Upper_Lim_15.0__Flux_Density_Lower_Lim_0.0/mk_cube/151_Flux_values_10NanoJansky_limit_data_result_18_Degree_Cube_RA_Dec_Degrees_and__10_pow_LogFlux_Columns_and_Source_Redshifts_and_Source_SI_and_Source_AGN_Type_Comb__mk.npz'):

	print 'Using EGS foreground data'
	#----------------------

	scidata1 = np.squeeze(np.load(EGS_npz_path)['arr_0'])[:,10:10+512,10:10+512] #Take 12 deg. subset to match EoR cube

	base_dir = 'Plots'
	save_dir = base_dir+'/Likelihood_v1d75_3D_ZM/'
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)

	# scidata1 = top_hat_average_temperature_cube_to_lower_res_31x31xnf_cube(scidata1)
	scidata1 = scidata1[0:nf,:,:]

	import numpy
	axes_tuple = (1,2)
	vfft1=numpy.fft.ifftshift(scidata1[0:nf]-scidata1[0].mean()+0j, axes=axes_tuple)
	vfft1=numpy.fft.fftn(vfft1, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	vfft1=numpy.fft.fftshift(vfft1, axes=axes_tuple)


	sci_f, sci_v, sci_u = vfft1.shape
	sci_v_centre = sci_v/2
	sci_u_centre = sci_u/2
	vfft1_subset = vfft1[0:nf,sci_u_centre-nu/2:sci_u_centre+nu/2+1,sci_v_centre-nv/2:sci_v_centre+nv/2+1]
	s_before_ZM = vfft1_subset.flatten()/vfft1[0].size**0.5
	# s_before_ZM = vfft1_subset.flatten()
	ZM_vis_ordered_mask = np.ones(nu*nv*nf)
	ZM_vis_ordered_mask[nf*((nu*nv)/2):nf*((nu*nv)/2+1)]=0
	ZM_vis_ordered_mask = ZM_vis_ordered_mask.astype('bool')
	ZM_chan_ordered_mask = ZM_vis_ordered_mask.reshape(-1, neta+nq).T.flatten()
	s = s_before_ZM[ZM_chan_ordered_mask]

	abc = s

	return s, abc, scidata1



## ======================================================================================================
## ======================================================================================================


# def generate_test_signal_from_image_cube(nu,nv,nx,ny,nf,neta,nq,image_cube):

# 	import numpy
# 	axes_tuple = (1,2)
# 	vfft1=numpy.fft.ifftshift(image_cube[0:nf]-image_cube[0].mean()+0j, axes=axes_tuple)
# 	vfft1=numpy.fft.fftn(vfft1, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
# 	vfft1=numpy.fft.fftshift(vfft1, axes=axes_tuple)
	
# 	sci_f, sci_v, sci_u = vfft1.shape
# 	sci_v_centre = sci_v/2
# 	sci_u_centre = sci_u/2
# 	vfft1_subset = vfft1[0:nf,sci_u_centre-nu/2:sci_u_centre+nu/2+1,sci_v_centre-nv/2:sci_v_centre+nv/2+1]
# 	s_before_ZM = vfft1_subset.flatten()/vfft1[0].size**0.5
# 	ZM_vis_ordered_mask = np.ones(nu*nv*nf)
# 	ZM_vis_ordered_mask[nf*((nu*nv)/2):nf*((nu*nv)/2+1)]=0
# 	ZM_vis_ordered_mask = ZM_vis_ordered_mask.astype('bool')
# 	ZM_chan_ordered_mask = ZM_vis_ordered_mask.reshape(-1, neta+nq).T.flatten()
# 	s = s_before_ZM[ZM_chan_ordered_mask]

# 	return s


## ======================================================================================================
## ======================================================================================================


def generate_test_signal_from_image_cube(nu,nv,nx,ny,nf,neta,nq,image_cube_mK,output_fits_file_name):

	beta_experimental_mean = 2.55+0  #Revise to match published values
	beta_experimental_std  = 0.1   #Revise to match published values
	gamma_mean                     = -2.7  #Revise to match published values
	gamma_sigma                    = 0.3   #Revise to match published values
	Tb_experimental_mean_K = 194.0 #Matches GSM mean in region A (see /users/psims/Cav/EoR/Missing_Radio_Flux/Surveys/Convert_GSM_to_HEALPIX_Map_and_Cartesian_Projection_Fits_File_v6d0_pygsm.py)
	Tb_experimental_std_K  = 23.0   #Matches GSM in region A at 0.333 degree resolution (i.e. for a 50 degree map 150 pixels across). Note: std. is a function of resultion so the foreground map should be made at the same resolution for this std normalisation to be accurate
	# Tb_experimental_mean_K = 240.0 #Revise to match published values
	# Tb_experimental_std_K  = 4.0   #Revise to match published values
	nu_min_MHz                     = 120.0
	channel_width_MHz              = 0.2

	GFC = GenerateForegroundCube(nu,nv,neta,nq, beta_experimental_mean, beta_experimental_std, gamma_mean, gamma_sigma, Tb_experimental_mean_K, Tb_experimental_std_K, nu_min_MHz, channel_width_MHz)

	Tb_nu, A, beta, Tb, nu_array_MHz = GFC.generate_normalised_Tb_A_and_beta_fields(513,513,513,513,nf,neta,nq)
	
	ED = ExtractDataFrom21cmFASTCube(plot_data=False)
	cube_side_Mpc = 2048.0
	redshift = 10.26
	bandwidth_MHz, central_frequency_MHz, output_21cmFast_box_width_deg = ED.calculate_box_size_in_degrees_and_MHz(cube_side_Mpc, redshift)
	
	DUC = DataUnitConversionmkandJyperpix()

	Data_mK = image_cube_mK
	# Data_mK = Data_mK+random_quad
	Channel_Frequencies_Array_Hz = nu_array_MHz*1.e6
	Pixel_Height_rad = output_21cmFast_box_width_deg*(np.pi/180.)/Tb_nu.shape[1]
	DUC.convert_from_mK_to_Jy_per_pix(Data_mK, Channel_Frequencies_Array_Hz, Pixel_Height_rad)
	Data_Jy_per_Pixel = ED.convert_from_mK_to_Jy_per_pix(Data_mK, Channel_Frequencies_Array_Hz, Pixel_Height_rad)
	
	# output_fits_file_name = 'Quad_only.fits'
	# output_fits_file_name = 'White_noise_cube_with_quad.fits'
	# output_fits_file_name = 'Jelic_GDSE_cube.fits'
	output_fits_path = 'fits_storage/'+output_fits_file_name
	print output_fits_path
	WD2F = WriteDataToFits()
	WD2F.write_data(Data_Jy_per_Pixel, output_fits_path, Box_Side_cMpc=cube_side_Mpc, simulation_redshift=redshift)



## ======================================================================================================
## ======================================================================================================


def Generate_Poisson_Distribution_with_Exact_Input_Mean(Mean, N_Data_Points, **kwargs):
	"""
	Generate values from a Poisson distribution and get a resulting distribution with exactly the input mean.
	"""
	#
	Mean=float(Mean)
	#
	P = stats.poisson.rvs(Mean, size=N_Data_Points)
	Target_Sum=int(N_Data_Points*Mean)
	
	###
	Print=True
	#
	for key in kwargs:
		if key==('Target_Sum'):Target_Sum=kwargs[key]
		if key==('Print'):Print=kwargs[key]

	
	if Print: print 'Target_Sum', Target_Sum
	while sum(P)>Target_Sum:
		P = stats.poisson.rvs(Mean, size=N_Data_Points)
	
	Sum_Current_P_Dist=sum(P)
	if Print: print 'Sum_Current_P_Dist', Sum_Current_P_Dist
	while Sum_Current_P_Dist!=Target_Sum:
		Updated_Mean=(Target_Sum-Sum_Current_P_Dist)/float(N_Data_Points)
		
		Sum_Current_P_Dist=sum(P)
		Potential_Supplementary_P=stats.poisson.rvs(abs(Updated_Mean), size=N_Data_Points)
		while (Sum_Current_P_Dist+sum(Potential_Supplementary_P))>Target_Sum:
			Potential_Supplementary_P=stats.poisson.rvs(abs(Updated_Mean), size=N_Data_Points)
		
		P+=Potential_Supplementary_P
		Sum_Current_P_Dist=sum(P)
		if Print: print 'Sum_Current_P_Dist', Sum_Current_P_Dist
	
	return P
	
	

# Poisson_16deg_1div32_res = Generate_Poisson_Distribution_with_Exact_Input_Mean(10.0, 512*512).reshape(512,512)
# print Poisson_16deg_1div32_res.std()

# axes_tuple=(0,1,)
# vfft1=numpy.fft.ifftshift(Poisson_16deg_1div32_res-Poisson_16deg_1div32_res.mean(), axes=axes_tuple)
# vfft1=numpy.fft.ifftn(vfft1, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
# vfft1=numpy.fft.fftshift(vfft1, axes=axes_tuple)

# pylab.figure()
# pylab.imshow(Poisson_16deg_1div32_res)
# pylab.colorbar()

# pylab.figure()
# pylab.imshow(log10(abs(vfft1)))
# pylab.colorbar()
# pylab.show()


# from scipy import signal
# import numpy as np
# x, y = np.meshgrid(np.arange(512)-256, np.arange(512)-256)
# r = (x**2.+y**2.)**0.5
# sigma, mu = 2.0, 0.0
# gaussian = np.exp(-( (r-mu)**2 / ( 2.0 * sigma**2 ) ) )
# gaussian = gaussian/np.sum(gaussian)

# pylab.figure()
# pylab.imshow(gaussian)
# pylab.colorbar()
# pylab.show()

# sigma3 = 3*int(sigma)
# # smoothed_Poisson_16deg_1div32_res = signal.convolve2d(Poisson_16deg_1div32_res, gaussian, boundary='symm', mode='same')
# smoothed_Poisson_16deg_1div32_res2 = signal.fftconvolve(Poisson_16deg_1div32_res, gaussian, mode='same')
# print smoothed_Poisson_16deg_1div32_res2[sigma3:-sigma3,sigma3:-sigma3].mean()
# print smoothed_Poisson_16deg_1div32_res2[sigma3:-sigma3,sigma3:-sigma3].std()


# print Poisson_16deg_1div32_res.std()

# pylab.figure()
# pylab.imshow(smoothed_Poisson_16deg_1div32_res2[sigma3:-sigma3,sigma3:-sigma3])
# pylab.colorbar()
# pylab.show()


	
## ======================================================================================================
## ======================================================================================================





