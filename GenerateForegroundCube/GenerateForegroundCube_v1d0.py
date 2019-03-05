
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







def generate_Jelic_cube_instrumental_im_2_vis(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show, beta_experimental_mean,beta_experimental_std,gamma_mean,gamma_sigma,Tb_experimental_mean_K,Tb_experimental_std_K,nu_min_MHz,channel_width_MHz,Finv, **kwargs):

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



	base_dir = 'Plots'
	save_dir = base_dir+'/Likelihood_v1d75_3D_ZM/'
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)




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
	vfft1=numpy.fft.ifftshift(scidata1[0:nf]+0j, axes=axes_tuple)
	vfft1=numpy.fft.fftn(vfft1, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	vfft1=numpy.fft.fftshift(vfft1, axes=axes_tuple)

	pylab.close('all')
	
	sci_f, sci_v, sci_u = vfft1.shape
	sci_v_centre = sci_v/2
	sci_u_centre = sci_u/2
	vfft1_subset = vfft1[0:nf,sci_u_centre-nu/2:sci_u_centre+nu/2+1,sci_v_centre-nv/2:sci_v_centre+nv/2+1]


	axes_tuple = (1,2)
	scidata1_subset=numpy.fft.ifftshift(vfft1_subset+0j, axes=axes_tuple)
	scidata1_subset=numpy.fft.ifftn(scidata1_subset, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	scidata1_subset=numpy.fft.fftshift(scidata1_subset, axes=axes_tuple)


	s = np.dot(Finv,scidata1_subset.reshape(-1,1)).flatten()
	abc = s

	fg = s


	return fg, s, Tb_nu, beta_experimental_mean,beta_experimental_std,gamma_mean,gamma_sigma,Tb_experimental_mean_K,Tb_experimental_std_K,nu_min_MHz,channel_width_MHz, HF_Tb_nu


## ======================================================================================================
## ======================================================================================================






# kwargs = {}

# beta_experimental_mean,beta_experimental_std,gamma_mean,gamma_sigma,Tb_experimental_mean_K,Tb_experimental_std_K,nu_min_MHz,channel_width_MHz = p.beta_experimental_mean,1.e-10,p.gamma_mean,p.gamma_sigma,p.Tb_experimental_mean_K,p.Tb_experimental_std_K,p.nu_min_MHz,p.channel_width_MHz

# generate_additional_extrapolated_HF_foreground_cube=True
# fits_storage_dir=p.fits_storage_dir
# HF_nu_min_MHz_array=p.HF_nu_min_MHz_array
# simulation_FoV_deg=p.simulation_FoV_deg
# simulation_resolution_deg=p.simulation_resolution_deg
# random_seed=314211

# from BayesEoR.SimData import GenerateForegroundCube, update_Tb_experimental_std_K_to_correct_for_normalisation_resolution
# from BayesEoR.Utils import PriorC, ParseCommandLineArguments, DataUnitConversionmkandJyperpix, WriteDataToFits
# from BayesEoR.Utils import ExtractDataFrom21cmFASTCube






# y1=np.log10(Tb_nu[:,17,110])

# x1 = np.log10((nu_min_MHz+np.arange(nf)*channel_width_MHz)/nu_min_MHz*1.e6)

# np.polyfit(x1,y1,1)

# for i in range(vfft1_subset.shape[1]):
# 	for j in range(vfft1_subset.shape[2]):
# 		y1=np.log10(abs(vfft1_subset[:,i,j].real))
# 		print i,j,np.polyfit(x1,y1,1)


# x1 = np.log10((nu_min_MHz+np.arange(nf)*channel_width_MHz)/nu_min_MHz*1.e6)
# y1=np.log10(abs(scidata1_subset[:,1,3].real))
# np.polyfit(x1,y1,1)

# x1 = np.log10(np.arange(len(y1))+1)

# for i in range(scidata1_subset.shape[1]):
# 	for j in range(scidata1_subset.shape[2]):
# 		y1=np.log10(abs(scidata1_subset[:,i,j].real))
# 		print i,j,np.polyfit(x1,y1,1)





# idft_array_1D_WQ = BM.read_data_from_hdf5(array_save_directory+'idft_array_1D_WQ.h5', 'idft_array_1D_WQ')

# y1 = np.log10(idft_array_1D_WQ[-2].real)
# np.polyfit(x1,y1,1)



# scidata1_subset[:,8,8].real = idft_array_1D_WQ[-2].real * 1.e9

# # m = T.theta
# # theta_hat = (T.conjugate().T . Ninv . T)**-1 T . Ninv . d

# scidata1_subset[:,8,8].real/idft_array_1D_WQ[-2].real


# idft_array_1D_WQ = BM.read_data_from_hdf5(array_save_directory+'idft_array_1D_WQ.h5', 'idft_array_1D_WQ')*1.e5
# idft_array_1D_WQ = idft_array_1D_WQ[-2:]

# d_test = scidata1_subset[:,8,8]
# Ninv_test = np.identity(len(scidata1_subset[:,8,8].real))
# theta_hat = np.dot(np.linalg.inv(np.dot(idft_array_1D_WQ.conjugate(), np.dot(Ninv_test, idft_array_1D_WQ.T))), np.dot(idft_array_1D_WQ, np.dot(Ninv_test, d_test.reshape(-1,1))))


# print scidata1_subset[:,8,8]/idft_array_1D_WQ[-2].real
# print theta_hat






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

def generate_white_noise_signal_instrumental_k_2_vis(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,T,Show,chan_selection,masked_power_spectral_modes):

	print 'Using use_WN_cube data'


	EoR_npz_path = p.EoR_npz_path_sc


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
	# Zero modes that correspond to / are replaced with foreground parameters (in the parameter vector that T is desiged to operate on) before applying T!
	###
	# Note: to apply masked_power_spectral_modes (which is vis_ordered) correctly scidata1_kcube_subset_ZM should also be vis_ordered however here it's vis_ordered. The only reason this isn't breaking things is because it's white noise which means the ordering makes no difference here.....
	scidata1_kcube_subset_ZM[masked_power_spectral_modes] = 0.0

	# T = BM.read_data_from_hdf5(array_save_directory+'T.h5', 'T')
	s = np.dot(T,scidata1_kcube_subset_ZM.reshape(-1,1)).flatten()
	abc = s

	return s, abc, scidata1




## ======================================================================================================
## ======================================================================================================

def generate_white_noise_signal_instrumental_im_2_vis(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Finv,Show,chan_selection,masked_power_spectral_modes,mod_k):

	print 'Using use_WN_cube data'


	EoR_npz_path = p.EoR_npz_path_sc


	#----------------------
	###
	# Replace Gaussian signal with EoR cube
	###
	scidata1 = np.load(EoR_npz_path)['arr_0']

	#Overwrite EoR cube with white noise
	# np.random.seed(21287254)
	# np.random.seed(4123)
	# np.random.seed(54123)
	# np.random.seed(154123)
	np.random.seed(123)
	scidata1 = np.random.normal(0,scidata1.std()*1.,[nf,nu,nv])*0.5




	axes_tuple = (0,1,2)
	scidata1_kcube=numpy.fft.ifftshift(scidata1[0:38]-scidata1[0:38].mean()+0j, axes=axes_tuple)
	scidata1_kcube=numpy.fft.fftn(scidata1_kcube, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	scidata1_kcube=numpy.fft.fftshift(scidata1_kcube, axes=axes_tuple)

	sci_f, sci_v, sci_u = scidata1_kcube.shape
	sci_v_centre = sci_v/2
	sci_u_centre = sci_u/2
	scidata1_kcube_subset = scidata1_kcube[0:nf,sci_u_centre-nu/2:sci_u_centre+nu/2+1,sci_v_centre-nv/2:sci_v_centre+nv/2+1]

	red_noise_power_law_coefficient = 3.0
	print 'Using red_noise_power_law_coefficient:', red_noise_power_law_coefficient
	red_noise_scaling_cube = 1./(mod_k**(red_noise_power_law_coefficient/2.0))
	red_noise_scaling_cube[np.isinf(red_noise_scaling_cube)] = 1.0
	scidata1_kcube_subset_scaled = scidata1_kcube_subset*red_noise_scaling_cube

	ZM_vis_ordered_mask = np.ones(nu*nv*nf)
	ZM_vis_ordered_mask[nf*((nu*nv)/2):nf*((nu*nv)/2+1)]=0
	ZM_vis_ordered_mask = ZM_vis_ordered_mask.astype('bool')
	ZM_chan_ordered_mask = ZM_vis_ordered_mask.reshape(-1, neta+nq).T.flatten()

	total_masked_power_spectral_modes = np.ones_like(ZM_chan_ordered_mask)
	unmasked_power_spectral_modes_chan_ordered = np.logical_not(masked_power_spectral_modes).reshape(-1,neta+nq).T
	#Make mask symmetric for real model image
	centre_chan = (nf/2)
	if centre_chan%2 != 0:
		centre_chan=centre_chan-1
	for i, chan in enumerate(unmasked_power_spectral_modes_chan_ordered):
		if np.sum(chan)==0.0 and i!=centre_chan:
			unmasked_power_spectral_modes_chan_ordered[-1-i] = chan
	total_masked_power_spectral_modes[ZM_chan_ordered_mask] = unmasked_power_spectral_modes_chan_ordered.flatten()
	total_masked_power_spectral_modes = np.logical_and(total_masked_power_spectral_modes, ZM_chan_ordered_mask)

	###
	# Zero modes that are not part of the data model
	###
	# scidata1_kcube_subset_scaled[total_masked_power_spectral_modes.reshape(scidata1_kcube_subset_scaled.shape)] = 0.0

	axes_tuple = (0,1,2)
	scidata1_subset_scaled=numpy.fft.ifftshift(scidata1_kcube_subset_scaled+0j, axes=axes_tuple)
	scidata1_subset_scaled=numpy.fft.ifftn(scidata1_subset_scaled, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	scidata1_subset_scaled=numpy.fft.fftshift(scidata1_subset_scaled, axes=axes_tuple)

	###
	# Apply correct normalisation in the image domain for the relavent invariance testing
	# Correct invariance - RMS of a (white...) noise cube should be constant as a function of resolution
	###

	preset_cube_rms = 100.e0
	scidata1_subset_scaled = scidata1_subset_scaled*preset_cube_rms/scidata1_subset_scaled.std()

	s = np.dot(Finv,scidata1_subset_scaled.reshape(-1,1)).flatten()
	abc = s

	return s, abc, scidata1





# for i in range(len(scidata1_kcube_subset_scaled)):
# 	print i, scidata1_kcube_subset_scaled[i].std()

# scidata1_kcube_subset_scaled

# ZM_vis_ordered_mask = np.ones(nu*nv*nf)
# ZM_vis_ordered_mask[nf*((nu*nv)/2):nf*((nu*nv)/2+1)]=0
# ZM_vis_ordered_mask = ZM_vis_ordered_mask.astype('bool')
# ZM_chan_ordered_mask = ZM_vis_ordered_mask.reshape(-1, neta+nq).T.flatten()


# scidata1_kcube_subset_scaled.flatten()[ZM_chan_ordered_mask]

# mod_k.flatten()[ZM_chan_ordered_mask]

# k_cube_voxels_in_bin

# modkbins_containing_voxels

# chan_ordered_bin_pixels_list = [np.where(np.logical_and(mod_k.flatten()[ZM_chan_ordered_mask]>modkbins_containing_voxels[i][0][0], mod_k.flatten()[ZM_chan_ordered_mask]<=modkbins_containing_voxels[i][0][1])) for i in range(len(modkbins_containing_voxels))]


# mean_ks = []
# variances = []
# for i in range(len(chan_ordered_bin_pixels_list)):
# 	mean_k = mod_k.flatten()[ZM_chan_ordered_mask][chan_ordered_bin_pixels_list[i]].mean()
# 	variance = scidata1_kcube_subset_scaled.flatten()[ZM_chan_ordered_mask][chan_ordered_bin_pixels_list[i]].var()
# 	print i, mean_k, variance
# 	mean_ks.append(mean_k)
# 	variances.append(variance)

# print np.polyfit(np.log10(mean_ks)[:], np.log10(variances)[:], 1)
# # print np.polyfit(np.log10(mean_ks)[:-1], np.log10(variances)[:-1], 1)



## ======================================================================================================
## ======================================================================================================

def generate_EoR_signal_instrumental_im_2_vis(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Finv,Show,chan_selection,masked_power_spectral_modes,mod_k,EoR_npz_path):

	print 'Using use_EoR_cube data'


	#----------------------
	###
	# Replace Gaussian signal with EoR cube
	###
	scidata1 = np.load(EoR_npz_path)['arr_0']

	#Overwrite EoR cube with white noise
	# np.random.seed(21287254)
	# np.random.seed(123)
	# scidata1 = np.random.normal(0,scidata1.std()*1.,[nf,nu,nv])*0.5




	axes_tuple = (0,1,2)
	scidata1_kcube=numpy.fft.ifftshift(scidata1[0:nf]-scidata1[0:nf].mean()+0j, axes=axes_tuple)
	scidata1_kcube=numpy.fft.fftn(scidata1_kcube, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	scidata1_kcube=numpy.fft.fftshift(scidata1_kcube, axes=axes_tuple)

	sci_f, sci_v, sci_u = scidata1_kcube.shape
	sci_v_centre = sci_v/2
	sci_u_centre = sci_u/2
	scidata1_kcube_subset = scidata1_kcube[0:nf,sci_u_centre-nu/2:sci_u_centre+nu/2+1,sci_v_centre-nv/2:sci_v_centre+nv/2+1]

	###
	#Zero modes that are not currently fit for until intrinsic noise fitting (which models these terms) has been implemented
	###
	Hermitian_small_spacial_scale_mask = np.zeros(scidata1_kcube_subset.shape)
	Hermitian_small_spacial_scale_mask[0] = 1 #Nyquist mode
	Hermitian_small_spacial_scale_mask[1] = 1 #2nd highest freq
	# Hermitian_small_spacial_scale_mask[2] = 1 #3nd highest freq
	# Hermitian_small_spacial_scale_mask[-2] = 1 #3nd highest freq
	Hermitian_small_spacial_scale_mask[-1] = 1 #2nd highest freq

	scidata1_kcube_subset[Hermitian_small_spacial_scale_mask.astype('bool')] = 0.0

	axes_tuple = (0,1,2)
	scidata1_subset=numpy.fft.ifftshift(scidata1_kcube_subset+0j, axes=axes_tuple)
	scidata1_subset=numpy.fft.ifftn(scidata1_subset, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	scidata1_subset=numpy.fft.fftshift(scidata1_subset, axes=axes_tuple)

	# scidata1_subset = scidata1_subset/scidata1_kcube.size**0.5

	s = np.dot(Finv,scidata1_subset.reshape(-1,1)).flatten()
	abc = s


	return s, abc, scidata1





## ======================================================================================================
## ======================================================================================================

def calculate_subset_cube_power_spectrum_v1d0(nu,nv,nx,ny,nf,neta,nq,scidata1_kcube_subset,k_cube_voxels_in_bin,modkbins_containing_voxels):

	scidata1_kcube_subset

	k_cube_voxels_in_bin
	modkbins_containing_voxels


	# EoR_npz_path = '/users/psims/EoR/EoR_simulations/21cmFAST_2048MPc_2048pix_512pix_AstroParamExploration1/Fits/npzs/Zeta10.0_Tvir1.0e+05_mfp22.2_Taue0.041_zre-1.000_delz-1.000_512_2048Mpc/21cm_mK_z7.600_nf0.883_useTs0.0_aveTb21.06_cube_side_pix512_cube_side_Mpc2048.npz'

	for i in range(len(scidata1_kcube_subset)):
		print i, scidata1_kcube_subset[i].std()

	scidata1_kcube_subset

	ZM_vis_ordered_mask = np.ones(nu*nv*nf)
	ZM_vis_ordered_mask[nf*((nu*nv)/2):nf*((nu*nv)/2+1)]=0
	ZM_vis_ordered_mask = ZM_vis_ordered_mask.astype('bool')
	ZM_chan_ordered_mask = ZM_vis_ordered_mask.reshape(-1, neta+nq).T.flatten()


	scidata1_kcube_subset.flatten()[ZM_chan_ordered_mask]

	mod_k.flatten()[ZM_chan_ordered_mask]

	chan_ordered_bin_pixels_list = [np.where(np.logical_and(mod_k.flatten()[ZM_chan_ordered_mask]>modkbins_containing_voxels[i][0][0], mod_k.flatten()[ZM_chan_ordered_mask]<=modkbins_containing_voxels[i][0][1])) for i in range(len(modkbins_containing_voxels))]

	excluded_zeroed_voxels = np.where(scidata1_kcube_subset.flatten()[ZM_chan_ordered_mask]==0.0)

	excluded_zeroed_voxel_locations_in_chan_ordered_bin_pixels_list = [[np.where(chan_ordered_bin_pixels_list[i_bin][0]==excluded_zeroed_voxels[0][i_ex])[0] for i_ex in range(len(excluded_zeroed_voxels[0])) if np.where(chan_ordered_bin_pixels_list[i_bin][0]==excluded_zeroed_voxels[0][i_ex])[0]] for i_bin in range(len(chan_ordered_bin_pixels_list)) ]


	chan_ordered_bin_pixels_list_without_zeroed_excluded_pixels = np.array(chan_ordered_bin_pixels_list).copy()

	chan_ordered_bin_pixels_list_without_zeroed_excluded_pixels = [np.delete(chan_ordered_bin_pixels_list[i_bin][0], excluded_zeroed_voxel_locations_in_chan_ordered_bin_pixels_list[i_bin]) for i_bin in range(len(chan_ordered_bin_pixels_list))]

	mean_ks = []
	variances = []
	powers = []
	dimensionless_powers = []
	for i in range(len(chan_ordered_bin_pixels_list)):
		ks_in_bin = mod_k.flatten()[ZM_chan_ordered_mask][chan_ordered_bin_pixels_list_without_zeroed_excluded_pixels[i]]
		amplitudes_in_bin = scidata1_kcube_subset.flatten()[ZM_chan_ordered_mask][chan_ordered_bin_pixels_list_without_zeroed_excluded_pixels[i]]
		mean_k = ks_in_bin.mean()
		variance = amplitudes_in_bin.var()
		power = (abs(amplitudes_in_bin)**2.).mean()
		dimensionless_power = (ks_in_bin**3. * abs(amplitudes_in_bin)**2.).mean()
		print i, mean_k, variance, power, power/variance
		mean_ks.append(mean_k)
		variances.append(variance)
		powers.append(power)
		dimensionless_powers.append(dimensionless_power)


	subset_ps_output_dir = '/users/psims/EoR/Python_Scripts/BayesEoR/git_version/BayesEoR/spec_model_tests/random/subset_ps/{}/'.format(EoR_npz_path.split('/')[-2].replace('.npz','').replace('.','d'))
	subset_ps_output_file = EoR_npz_path.split('/')[-1].replace('.npz','').replace('.','d')
	subset_ps_output_path = subset_ps_output_dir+subset_ps_output_file

	print 'Saving unnormalised dimensionless power spectrum to: \n', subset_ps_output_path
	if not os.path.isdir(subset_ps_output_dir):
			os.makedirs(subset_ps_output_dir)
	np.savetxt(subset_ps_output_path, np.vstack((mean_ks, dimensionless_powers)).T)













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





