import numpy as np
import numpy
from numpy import arange, shape, log10, pi
import scipy
from subprocess import os
import sys
import pylab
from pdb import set_trace
from pdb import set_trace as brk
from numpy import * 
import pylab
import pylab as P
import scipy
from scipy import optimize
from scipy.optimize import curve_fit
from scipy import misc
from scipy import ndimage
import BayesEoR.Params.params as p

from BayesEoR.Linalg import IDFT_Array_IDFT_2D_ZM_SH, makeGaussian, Produce_Full_Coordinate_Arrays, Produce_Coordinate_Arrays_ZM
from BayesEoR.Linalg import Produce_Coordinate_Arrays_ZM_Coarse, Produce_Coordinate_Arrays_ZM_SH, Calc_Coords_High_Res_Im_to_Large_uv
from BayesEoR.Linalg import Calc_Coords_Large_Im_to_High_Res_uv, Restore_Centre_Pixel, Calc_Indices_Centre_3x3_Grid
from BayesEoR.Linalg import Delete_Centre_3x3_Grid, Delete_Centre_Pix, N_is_Odd, Calc_Indices_Centre_NxN_Grid, Obtain_Centre_NxN_Grid
from BayesEoR.Linalg import Restore_Centre_3x3_Grid, Restore_Centre_NxN_Grid, Generate_Combined_Coarse_plus_Subharmic_uv_grids
from BayesEoR.Linalg import IDFT_Array_IDFT_2D_ZM_SH, DFT_Array_DFT_2D, IDFT_Array_IDFT_2D, DFT_Array_DFT_2D_ZM, IDFT_Array_IDFT_2D_ZM
from BayesEoR.Linalg import Construct_Hermitian, Construct_Hermitian_Gridding_Matrix, Construct_Hermitian_Gridding_Matrix_CosSin
from BayesEoR.Linalg import Construct_Hermitian_Gridding_Matrix_CosSin_SH_v4, IDFT_Array_IDFT_1D, IDFT_Array_IDFT_1D
from BayesEoR.Linalg import generate_gridding_matrix_vis_ordered_to_chan_ordered


## ======================================================================================================
## ======================================================================================================

def generate_test_sim_signal(nu,nv,nx,ny,nf,neta):
	# ---------------------------------------------
	###
	# Generate test sim data
	###
	Fz_normalisation = nf**0.5
	DFT2D_Fz_normalisation = (nu*nv*nf)**0.5
	from scipy.linalg import block_diag
	dft_array = DFT_Array_DFT_2D_ZM(nu,nv,nx,ny)
	multi_chan_dft_array_noZMchan = block_diag(*[dft_array.T for i in range(nf)])
	k_z, k_y, k_x = np.mgrid[-(nf/2):(nf/2),-(nu/2):(nu/2)+1,-(nv/2):(nv/2)+1]
	mod_k = (k_z**2. + k_y**2. + k_x**2.)**0.5
	k_cube_power_scaling_cube = np.zeros([nf,nu,nv])
	k_sigma_1 = 2.0
	k_sigma_2 = 10.0
	bin_limit = 2.0
	bin_1_selector_in_model_mask = np.logical_and(bin_limit>mod_k, mod_k>0.0)
	bin_2_selector_in_model_mask = mod_k>=bin_limit

	#----------------
	###
	# Do not include high spatial frequency structure in the data except when performing wn-fitting in the analysis
	###
	Nyquist_k_z_mode = k_z[0,0,0]
	Second_highest_frequency_k_z_mode = k_z[-1,0,0]
	high_spatial_frequency_selector_mask = np.logical_or(k_z==Nyquist_k_z_mode, k_z==Second_highest_frequency_k_z_mode)
	high_spatial_frequency_mask = np.logical_not(high_spatial_frequency_selector_mask)

	bin_1_selector_in_model_mask = np.logical_and(bin_1_selector_in_model_mask, high_spatial_frequency_mask)
	bin_2_selector_in_model_mask = np.logical_and(bin_2_selector_in_model_mask, high_spatial_frequency_mask)
	#----------------

	k_cube_power_scaling_cube[bin_1_selector_in_model_mask] = k_sigma_1 #It would be more accurate to call this the k_cube_std_scaling_cube 
	k_cube_power_scaling_cube[bin_2_selector_in_model_mask] = k_sigma_2 #It would be more accurate to call this the k_cube_std_scaling_cube

	np.random.seed(123)
	random_im=np.random.normal(0.,1,nf*nu*nv).reshape(nf,nu,nv)
	axes_tuple = (0,1,2)
	random_k=numpy.fft.ifftshift(random_im+0j, axes=axes_tuple)
	random_k=numpy.fft.fftn(random_k, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	random_k=numpy.fft.fftshift(random_k, axes=axes_tuple)
	random_k = random_k/random_k.std()

	k_perp_3D = (k_x**2.+k_y**2)**0.5
	ZM_mask = k_perp_3D>0.0
	bin_1_selector_in_model_mask_and_ZM = np.logical_and.reduce((bin_1_selector_in_model_mask, ZM_mask))
	bin_2_selector_in_model_mask_and_ZM = np.logical_and.reduce((bin_2_selector_in_model_mask, ZM_mask))
	k_cube_signal = k_cube_power_scaling_cube*random_k	

	axes_tuple = (0,1,2)
	im_power_scaling=numpy.fft.ifftshift(random_k+0j, axes=axes_tuple)
	im_power_scaling=numpy.fft.ifftn(im_power_scaling, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	im_power_scaling=numpy.fft.fftshift(im_power_scaling, axes=axes_tuple)

	s_im=numpy.fft.ifftshift(k_cube_signal+0j, axes=axes_tuple)
	s_im=numpy.fft.ifftn(s_im, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	s_im=numpy.fft.fftshift(s_im, axes=axes_tuple)

	s_im = s_im/(1./DFT2D_Fz_normalisation)
	s = np.dot(multi_chan_dft_array_noZMchan/(nu*nv)**0.5, s_im.flatten())

	return s, s_im, k_cube_signal, k_cube_power_scaling_cube, random_k, bin_1_selector_in_model_mask_and_ZM, bin_2_selector_in_model_mask_and_ZM, bin_1_selector_in_model_mask, bin_2_selector_in_model_mask, k_sigma_1, k_sigma_2, bin_limit, high_spatial_frequency_selector_mask, high_spatial_frequency_mask


## ======================================================================================================
## ======================================================================================================

def generate_k_cube_in_physical_coordinates(nu,nv,nx,ny,nf,neta):
	# ---------------------------------------------
	###
	# Generate k_cube pixel coordinates
	###
	z, y, x = np.mgrid[-(nf/2):(nf/2),-(nu/2):(nu/2)+1,-(nv/2):(nv/2)+1]
	uvcellsize = 2.5 #A distance of one pixel in the uv-plane corresponds to 2.5 lambda
	DCov=9745.3730 #comoving distance at redshift 10.26
	deltakperp = 2.*np.pi*uvcellsize/DCov # k_x = 2*pi*u/D_M, k_y = 2*pi*v/D_M
	deltakpara=0.04134*38.0/nf
	# modkscaleterm=1.5 #Value used in BEoRfgs
	k_z = z*deltakpara
	k_y = y*deltakperp
	k_x = x*deltakperp
	mod_k_physical = (k_z**2. + k_y**2. + k_x**2.)**0.5
	return mod_k_physical


## ======================================================================================================
## ======================================================================================================

def generate_k_cube_in_physical_coordinates_21cmFAST(nu,nv,nx,ny,nf,neta):
	# ---------------------------------------------
	###
	# Generate k_cube pixel coordinates
	###
	z, y, x = np.mgrid[-(nf/2):(nf/2),-(nu/2):(nu/2)+1,-(nv/2):(nv/2)+1]

	###
	# Define k-perp and k-para pixel scaling, following 21cmFAST
	###
	box_size_x_21cmFAST_pix = 512.
	box_size_y_21cmFAST_pix = 512.
	box_size_z_21cmFAST_pix = 512.
	box_size_x_21cmFAST_Mpc = 2048.
	box_size_y_21cmFAST_Mpc = 2048.
	box_size_z_21cmFAST_Mpc = 2048.
	box_size_xy_MyCube_Mpc = 2048. #I 2D DFT the whole (128, 128) pix = (512,512) Mpc channels. Taking a subset of the pixels in the uv-plane, subsequently, filters high resolution (high k-perp) values but doesn't alter deltakperp which is the smallest k-perp value (i.e. of one pixel) in the plane.
	box_size_z_MyCube_Mpc = 2048.*nf/box_size_z_21cmFAST_pix #I take a subset of the channels in the xyf cube i.e. before Fourier transforming. I then 1D DFT the subset. Therefore deltakpara, the smallest k-parallel value accessible, is now much larger because I have effectively filtered out the large scales (== low k) when taking the subset.
	deltakperp = 2.*np.pi/box_size_xy_MyCube_Mpc
	deltakpara=2.*pi/box_size_z_MyCube_Mpc
	# EoRVolume = 770937185.063917
	# modkscaleterm=1.5 #Value used in BEoRfgs
	k_z = z*deltakpara
	k_y = y*deltakperp
	k_x = x*deltakperp
	mod_k_physical = (k_z**2. + k_y**2. + k_x**2.)**0.5

	return mod_k_physical, k_x, k_y, k_z, deltakperp, deltakpara, x, y, z


## ======================================================================================================
## ======================================================================================================

def generate_k_cube_in_physical_coordinates_21cmFAST_v2d0(nu,nv,nx,ny,nf,neta,box_size_21cmFAST_pix,box_size_21cmFAST_Mpc):
	# ---------------------------------------------
	###
	# Generate k_cube pixel coordinates
	###
	z, y, x = np.mgrid[-(nf/2):(nf/2),-(nu/2):(nu/2)+1,-(nv/2):(nv/2)+1]

	###
	# Define k-perp and k-para pixel scaling, following 21cmFAST
	###
	# box_size_21cmFAST_pix = 512.
	# box_size_21cmFAST_Mpc = 2048.
	box_size_21cmFAST_pix = float(box_size_21cmFAST_pix)
	box_size_21cmFAST_Mpc = float(box_size_21cmFAST_Mpc)
	box_size_xy_MyCube_Mpc = box_size_21cmFAST_Mpc #I 2D DFT the whole (128, 128) pix = (512,512) Mpc channels. Taking a subset of the pixels in the uv-plane filters high resolution (high k-perp) values but doesn't alter deltakperp.
	box_size_z_MyCube_Mpc = box_size_21cmFAST_Mpc*nf/box_size_21cmFAST_pix #I take a subset of the channels in the xyf cube i.e. before Fourier transforming. I then 1D DFT the subset. Therefore deltakpara, the smallest k-parallel value accessible, is now much larger because I have effectively filtered out the large scales (== low k) when taking the subset.
	deltakperp = 2.*np.pi/box_size_xy_MyCube_Mpc
	deltakpara=2.*pi/box_size_z_MyCube_Mpc
	# EoRVolume = 770937185.063917
	# modkscaleterm=1.5 #Value used in BEoRfgs
	k_z = z*deltakpara
	k_y = y*deltakperp
	k_x = x*deltakperp
	mod_k_physical = (k_z**2. + k_y**2. + k_x**2.)**0.5

	return mod_k_physical, k_x, k_y, k_z, deltakperp, deltakpara, x, y, z


## ======================================================================================================
## ======================================================================================================


def construct_GRN_unitary_hermitian_k_cube(nu,nv,neta,nq):
	n_kz = neta+nq
	n_ky = nv
	n_kx = nu

	n_kz_even =  n_kz%2==0
	if n_kz_even:
		n_kz_odd = n_kz-1
	else:
		n_kz_odd = n_kz

	np.random.seed(2391)
	complex_GRN_real = np.random.normal(0,1,(n_kx*n_ky*n_kz_odd)/2)
	np.random.seed(1234)
	complex_GRN_imag = np.random.normal(0,1,(n_kx*n_ky*n_kz_odd)/2)
	complex_GRN = complex_GRN_real+1j*complex_GRN_imag
	complex_GRN = complex_GRN/complex_GRN.std()

	image_cube_mean = 1.0
	GRN_k_cube_vector = np.hstack((complex_GRN, image_cube_mean, complex_GRN[::-1].conjugate()))
	GRN_k_cube = GRN_k_cube_vector.reshape(n_kz_odd,n_ky,n_kx)

	if n_kz_even:
		#Add in Nyquist channel but give it zero power (since it is currently not part of the data model).
		GRN_k_cube_even = np.zeros([n_kz_odd+1,n_ky,n_kx])+0j
		GRN_k_cube_even[1:n_kz_odd+1,0:n_ky,0:n_kx]=GRN_k_cube
		GRN_k_cube = GRN_k_cube_even

	return GRN_k_cube


## ======================================================================================================
## ======================================================================================================

def construct_GRN_scaled_hermitian_k_cube(nu,nv,neta,nq,bin_selector_in_k_cube_mask,k_sigma):
	n_kz = neta+nq
	n_ky = nv
	n_kx = nu

	n_kz_even =  n_kz%2==0
	if n_kz_even:
		n_kz_odd = n_kz-1
	else:
		n_kz_odd = n_kz

	np.random.seed(2391)
	complex_GRN_real = np.random.normal(0,1,(n_kx*n_ky*n_kz_odd)/2)
	np.random.seed(1234)
	complex_GRN_imag = np.random.normal(0,1,(n_kx*n_ky*n_kz_odd)/2)
	complex_GRN = complex_GRN_real+1j*complex_GRN_imag
	complex_GRN = complex_GRN/complex_GRN.std()

	###
	# Impart input power spectrum onto the unitary cube
	###
	for i_bin in range(len(bin_selector_in_k_cube_mask)):
		if n_kz_even:
			#Ignore unmoddelled Nyquist bin in mask by starting at mask channel 1
			n_kxy =(n_kx*n_ky)
			current_vals = complex_GRN[bin_selector_in_k_cube_mask[i_bin].flatten()[n_kxy:n_kxy+complex_GRN.size]]
			current_std = np.hstack((current_vals, current_vals.conjugate())).std()
			complex_GRN[bin_selector_in_k_cube_mask[i_bin].flatten()[n_kxy:n_kxy+complex_GRN.size]] *= k_sigma[i_bin]/current_std
		else:
			current_vals = complex_GRN[bin_selector_in_k_cube_mask[i_bin].flatten()[:n_kxy/2+(n_kxy*n_kz_odd)/2]]
			current_std = np.hstack((current_vals, current_vals.conjugate())).std()
			complex_GRN[bin_selector_in_k_cube_mask[i_bin].flatten()[:n_kxy/2+(n_kxy*n_kz_odd)/2]] *= k_sigma[i_bin]/current_std

	image_cube_mean = 1.0
	GRN_k_cube_vector = np.hstack((complex_GRN, image_cube_mean, complex_GRN[::-1].conjugate()))
	GRN_k_cube = GRN_k_cube_vector.reshape(n_kz_odd,n_ky,n_kx)

	if n_kz_even:
		#Add in Nyquist channel but give it zero power (since it is currently not part of the data model).
		GRN_k_cube_even = np.zeros([n_kz_odd+1,n_ky,n_kx])+0j
		GRN_k_cube_even[1:n_kz_odd+1,0:n_ky,0:n_kx]=GRN_k_cube
		GRN_k_cube = GRN_k_cube_even

	return GRN_k_cube


## ======================================================================================================
## ======================================================================================================

def generate_test_sim_signal_with_large_spectral_scales_2_HERA_Binning(nu,nv,nx,ny,nf,neta,nq):
	# ---------------------------------------------
	###
	# Generate test sim data
	###
	Fz_normalisation = nf**0.5
	DFT2D_Fz_normalisation = (nu*nv*nf)**0.5

	from scipy.linalg import block_diag
	dft_array = DFT_Array_DFT_2D_ZM(nu,nv,nx,ny)
	multi_chan_dft_array_noZMchan = block_diag(*[dft_array.T for i in range(nf)])
	###
	# Generate k_cube pixel coordinates
	###
	z, y, x = np.mgrid[-(nf/2):(nf/2),-(nu/2):(nu/2)+1,-(nv/2):(nv/2)+1]

	###
	# Define xy-pixel to uv scaling
	###
	uvcellsize = 2.5 #A distance of one pixel in the uv-plane corresponds to 2.5 lambda

	###
	# Define uv-pixel to k_xy and z-pixel to k_z scaling
	###
	DCov=9745.3730 #comoving distance at redshift 10.26
	deltakperp = 2.*np.pi*uvcellsize/DCov # k_x = 2*pi*u/D_M, k_y = 2*pi*v/D_M
	deltakpara=0.04134*38.0/nf

	EoRVolume = 770937185.063917
	modkscaleterm=1.5 #Value used in BEoRfgs

	k_z_phys = z*deltakpara
	k_y_phys = y*deltakperp
	k_x_phys = x*deltakperp
	mod_k_physical = (k_z_phys**2. + k_y_phys**2. + k_x_phys**2.)**0.5

	k_z, k_y, k_x = k_z_phys, k_y_phys, k_x_phys

	mod_k = (k_z**2. + k_y**2. + k_x**2.)**0.5
	k_cube_power_scaling_cube = np.zeros([nf,nu,nv])

	k_sigma_1 = 10.**(4.0/2.)
	k_sigma_2 = 10.**(3.6/2.)
	k_sigma_3 = 10.**(3.1/2.)
	k_sigma_4 = 10.**(2.7/2.)
	k_sigma_5 = 10.**(2.3/2.)
	k_sigma_6 = 10.**(1.9/2.)
	k_sigma_7 = 10.**(1.4/2.)
	k_sigma_8 = 10.**(1.0/2.)

	k_sigma = [k_sigma_1, k_sigma_2, k_sigma_3, k_sigma_4, k_sigma_5, k_sigma_6, k_sigma_7, k_sigma_8]

	modkscaleterm=1.5 #Value used in BEoRfgs
	binsize=deltakperp*2

	numKbins = 50
	modkbins = np.zeros([numKbins,2])
	modkbins[0,0]=0
	modkbins[0,1]=binsize

	for m1 in range(1,numKbins,1):
		binsize=binsize*modkscaleterm
		modkbins[m1,0]=modkbins[m1-1,1]
		modkbins[m1,1]=modkscaleterm*modkbins[m1,0]

	total_elements = 0
	bin_selector_in_k_cube_mask=[]
	n_bins = 0
	for i_bin in range(numKbins):
		#NOTE: By requiring k_z>0 the constant term in the 1D FFT is now effectively a quadratic mode!
		# If it is to be included explicitly with the quadratic modes, then k_z==0 should be added to the quadratic selector mask
		n_elements = np.sum(np.logical_and.reduce((mod_k>modkbins[i_bin,0], mod_k<=modkbins[i_bin,1], k_z>0)))
		if n_elements>0:
			n_bins+=1
	 		print i_bin, modkbins[i_bin,0], modkbins[i_bin,1], n_elements 
	 		total_elements+=n_elements
	 		bin_selector_in_k_cube_mask.append(np.logical_and(mod_k>modkbins[i_bin,0], mod_k<=modkbins[i_bin,1], k_z>0))

	print total_elements, mod_k.size


	#----------------
	###
	# Do not include high spatial frequency structure in the power spectral data since  these terms aren't included in the data model
	###
	Nyquist_k_z_mode = k_z[0,0,0]
	Second_highest_frequency_k_z_mode = k_z[-1,0,0]
	# Mean_k_z_mode = 0.0
	#NOTE: the k_z=0 term should not necessarily be masked out since it is still required as a quadratic component (and is not currently explicitly added in there) even if it is not used for calculating the power spectrum.
	high_spatial_frequency_selector_mask = np.logical_or.reduce((k_z==Nyquist_k_z_mode, k_z==Second_highest_frequency_k_z_mode))
	high_spatial_frequency_mask = np.logical_not(high_spatial_frequency_selector_mask)

	for i_bin in range(len(bin_selector_in_k_cube_mask)):
		bin_selector_in_k_cube_mask[i_bin] = np.logical_and(bin_selector_in_k_cube_mask[i_bin], high_spatial_frequency_mask)
	#----------------
	Mean_k_z_mode = 0.0
	k_z_mean_mask = k_z!=Mean_k_z_mode

	k_perp_3D = (k_x**2.+k_y**2)**0.5
	ZM_mask = k_perp_3D>0.0
	#----------------
	###
	# Construct zero-mean dataset
	###
	for i_bin in range(len(bin_selector_in_k_cube_mask)):
		bin_selector_in_k_cube_mask[i_bin] = np.logical_and(bin_selector_in_k_cube_mask[i_bin], ZM_mask)
	
	#----------------
	k_cube_signal = construct_GRN_scaled_hermitian_k_cube(nu,nv,neta,nq,bin_selector_in_k_cube_mask,k_sigma)
	

	axes_tuple = (0,1,2)
	s_im=numpy.fft.ifftshift(k_cube_signal+0j, axes=axes_tuple)
	s_im=numpy.fft.ifftn(s_im, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	s_im=numpy.fft.fftshift(s_im, axes=axes_tuple)
	#----------------
	###
	# Add in large spectral-scale power
	###
	a0=0.0
	a1=1.0
	a2=2.0
	quad_large_spectral_scale_model = np.zeros(s_im.shape)
	np.random.seed(123)
	q1_amplitudes = np.random.normal(10,1,nu*nv).reshape(nu,nv)
	np.random.seed(321)
	q2_amplitudes = np.random.normal(10,1,nu*nv).reshape(nu,nv)
	for i in range(nu):
		for j in range(nv):
			quad_large_spectral_scale_model[:,i,j] = a0+a1*q1_amplitudes[i,j]*k_z[:,i,j]+a2*q2_amplitudes[i,j]*(k_z[:,i,j]+0.0)**2
			# quad_large_spectral_scale_model[:,i,j] = a1+i*k_z[:,i,j]+j*(k_z[:,i,j])**2
			# quad_large_spectral_scale_model[:,i,j] = a1+i*k_z[:,i,j]+j*(k_z[:,i,j]+10)**2

	s_im_quad_only = quad_large_spectral_scale_model
	s_im_fourier_only = s_im.copy()
	s_im = s_im_fourier_only + s_im_quad_only
	#----------------

	s_im_quad_only = s_im_quad_only/(1./DFT2D_Fz_normalisation)
	s_im_fourier_only = s_im_fourier_only/(1./DFT2D_Fz_normalisation)
	s_im = s_im/(1./DFT2D_Fz_normalisation)

	s_quad_only = np.dot(multi_chan_dft_array_noZMchan/(nu*nv)**0.5, s_im_quad_only.flatten())
	s_fourier_only = np.dot(multi_chan_dft_array_noZMchan/(nu*nv)**0.5, s_im_fourier_only.flatten())
	s = np.dot(multi_chan_dft_array_noZMchan/(nu*nv)**0.5, s_im.flatten())


	from scipy.linalg import block_diag
	idft_array_1D=IDFT_Array_IDFT_1D(nf, neta+nq)*Fz_normalisation #Note: the nf**0.5 normalisation factor results in a symmetric transform and is necessary for correctly estimating the power spectrum. However, it is not consistent with Python's asymmetric DFTs, therefore this factor needs to be removed when comparing to np.fft.fftn cross-checks!
	multi_vis_idft_array_1D = block_diag(*[idft_array_1D for i in range(nu*nv-1)])

	k2 = np.dot(multi_vis_idft_array_1D.conjugate().T, s_fourier_only.reshape(neta+nq,-1).T.flatten())

	return s, s_im, s_quad_only, s_im_quad_only, s_fourier_only, s_im_fourier_only, bin_selector_in_k_cube_mask,high_spatial_frequency_selector_mask, k_cube_signal, k_sigma, ZM_mask, k_z_mean_mask


## ======================================================================================================
## ======================================================================================================

def generate_test_sim_signal_with_large_spectral_scales_2_21cmFAST_Binning(nu,nv,nx,ny,nf,neta,nq):
	# ---------------------------------------------
	###
	# Generate test sim data
	###
	Fz_normalisation = nf**0.5
	DFT2D_Fz_normalisation = (nu*nv*nf)**0.5

	from scipy.linalg import block_diag
	dft_array = DFT_Array_DFT_2D_ZM(nu,nv,nx,ny)
	multi_chan_dft_array_noZMchan = block_diag(*[dft_array.T for i in range(nf)])
	###
	# Generate k_cube physical coordinates to match the 21cmFAST input simulation
	###
	mod_k, k_x, k_y, k_z, deltakperp, deltakpara, x, y, z = generate_k_cube_in_physical_coordinates_21cmFAST_v2d0(nu,nv,nx,ny,nf,neta,p.box_size_21cmFAST_pix_sc,p.box_size_21cmFAST_Mpc_sc)
	k_cube_power_scaling_cube = np.zeros([nf,nu,nv])

	k_sigma_1 = 10.**(4.0/2.)
	k_sigma_2 = 10.**(3.6/2.)
	k_sigma_3 = 10.**(3.1/2.)
	k_sigma_4 = 10.**(2.7/2.)
	k_sigma_5 = 10.**(2.3/2.)
	k_sigma_6 = 10.**(1.9/2.)
	k_sigma_7 = 10.**(1.4/2.)
	k_sigma_8 = 10.**(1.0/2.)
	k_sigma_9 = 10.**(1.0/2.)
	k_sigma_10 = 10.**(1.0/2.)

	k_sigma = [k_sigma_1, k_sigma_2, k_sigma_3, k_sigma_4, k_sigma_5, k_sigma_6, k_sigma_7, k_sigma_8, k_sigma_9, k_sigma_10]


	modkscaleterm=1.5 #Value used in BEoRfgs and in 21cmFAST binning
	# binsize=deltakperp*1 #Value used in 21cmFAST
	# binsize=deltakperp*4 #Value used in BEoRfgs
	binsize=deltakperp*2 #Value used in BEoRfgs

	numKbins = 50
	modkbins = np.zeros([numKbins,2])
	modkbins[0,0]=0
	modkbins[0,1]=binsize

	for m1 in range(1,numKbins,1):
		binsize=binsize*modkscaleterm
		modkbins[m1,0]=modkbins[m1-1,1]
		modkbins[m1,1]=modkscaleterm*modkbins[m1,0]
 		# print m1, modkbins[m1,0], modkbins[m1,1]

	total_elements = 0
	bin_selector_in_k_cube_mask=[]
	n_bins = 0
	for i_bin in range(numKbins):
		#NOTE: By requiring k_z>0 the constant term in the 1D FFT is now effectively a quadratic mode.
		# If it is to be included explicitly with the quadratic modes, then k_z==0 should be added to the quadratic selector mask
		n_elements = np.sum(np.logical_and.reduce((mod_k>modkbins[i_bin,0], mod_k<=modkbins[i_bin,1], k_z>0)))
		if n_elements>0:
			n_bins+=1
	 		print i_bin, modkbins[i_bin,0], modkbins[i_bin,1], n_elements 
	 		total_elements+=n_elements
	 		bin_selector_in_k_cube_mask.append(np.logical_and(mod_k>modkbins[i_bin,0], mod_k<=modkbins[i_bin,1], k_z>0))

	print total_elements, mod_k.size

	print 'deltakperp, deltakpara', deltakperp, deltakpara
	# raw_input()


	#----------------
	###
	# Do not include high spatial frequency structure in the power spectral data if nq>0, since these terms aren't included in the data model
	###
	Nyquist_k_z_mode = k_z[0,0,0]
	Second_highest_frequency_k_z_mode = k_z[-1,0,0]
	#NOTE: the k_z=0 term should not necessarily be masked out since it is still required as a quadratic component (and is not currently explicitly added in there) even if it is not used for calculating the power spectrum.
	# NOTE 2: bin_selector_in_k_cube_mask is actually only used for working out which parts of the k-cube NOT to use (should change to a name more descriptive of its later usage) when calculating LogDetPhi in the likelihood and hence the k_z=0 term (which should not be in that determinant!) should be masked out here! ...probably.
	Mean_k_z_mode = 0.0
	k_z_mean_mask = k_z!=Mean_k_z_mode

	if nq==1:
		high_spatial_frequency_selector_mask = k_z==Nyquist_k_z_mode
	elif nq==2:
		high_spatial_frequency_selector_mask = np.logical_or.reduce((k_z==Nyquist_k_z_mode, k_z==Second_highest_frequency_k_z_mode))
	else:
		high_spatial_frequency_selector_mask = np.zeros(k_z.shape).astype('bool')
	high_spatial_frequency_mask = np.logical_not(high_spatial_frequency_selector_mask)

	for i_bin in range(len(bin_selector_in_k_cube_mask)):
		if nq>0:
			# bin_selector_in_k_cube_mask[i_bin] = np.logical_and(bin_selector_in_k_cube_mask[i_bin], high_spatial_frequency_mask)
			bin_selector_in_k_cube_mask[i_bin] = np.logical_and(bin_selector_in_k_cube_mask[i_bin], high_spatial_frequency_mask, k_z_mean_mask)
		else:
			bin_selector_in_k_cube_mask[i_bin] = np.logical_and(bin_selector_in_k_cube_mask[i_bin], k_z_mean_mask)
	#----------------

	k_perp_3D = (k_x**2.+k_y**2)**0.5
	ZM_mask = k_perp_3D>0.0
	#----------------
	###
	# Construct zero-mean dataset
	###
	for i_bin in range(len(bin_selector_in_k_cube_mask)):
		bin_selector_in_k_cube_mask[i_bin] = np.logical_and(bin_selector_in_k_cube_mask[i_bin], ZM_mask)
	#----------------

	k_cube_signal = construct_GRN_scaled_hermitian_k_cube(nu,nv,neta,nq,bin_selector_in_k_cube_mask,k_sigma)
	
	axes_tuple = (0,1,2)
	s_im=numpy.fft.ifftshift(k_cube_signal+0j, axes=axes_tuple)
	s_im=numpy.fft.ifftn(s_im, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	s_im=numpy.fft.fftshift(s_im, axes=axes_tuple)

	#----------------
	###
	# Add in large spectral-scale power
	###
	a0=0.0
	a1=1.0
	a2=2.0
	quad_large_spectral_scale_model = np.zeros(s_im.shape)
	np.random.seed(123)
	q1_amplitudes = np.random.normal(10,1,nu*nv).reshape(nu,nv)
	np.random.seed(321)
	q2_amplitudes = np.random.normal(10,1,nu*nv).reshape(nu,nv)
	for i in range(nu):
		for j in range(nv):
			quad_large_spectral_scale_model[:,i,j] = a0+a1*q1_amplitudes[i,j]*k_z[:,i,j]+a2*q2_amplitudes[i,j]*(k_z[:,i,j]+0.0)**2
			# quad_large_spectral_scale_model[:,i,j] = a1+i*k_z[:,i,j]+j*(k_z[:,i,j])**2
			# quad_large_spectral_scale_model[:,i,j] = a1+i*k_z[:,i,j]+j*(k_z[:,i,j]+10)**2

	s_im_quad_only = quad_large_spectral_scale_model
	s_im_fourier_only = s_im.copy()
	s_im = s_im_fourier_only + s_im_quad_only
	#----------------

	s_im_quad_only = s_im_quad_only/(1./DFT2D_Fz_normalisation)
	s_im_fourier_only = s_im_fourier_only/(1./DFT2D_Fz_normalisation)
	s_im = s_im/(1./DFT2D_Fz_normalisation)

	s_quad_only = np.dot(multi_chan_dft_array_noZMchan/(nu*nv)**0.5, s_im_quad_only.flatten())
	s_fourier_only = np.dot(multi_chan_dft_array_noZMchan/(nu*nv)**0.5, s_im_fourier_only.flatten())
	s = np.dot(multi_chan_dft_array_noZMchan/(nu*nv)**0.5, s_im.flatten())

	from scipy.linalg import block_diag
	idft_array_1D=IDFT_Array_IDFT_1D(nf, neta+nq)*Fz_normalisation #Note: the nf**0.5 normalisation factor results in a symmetric transform and is necessary for correctly estimating the power spectrum. However, it is not consistent with Python's asymmetric DFTs, therefore this factor needs to be removed when comparing to np.fft.fftn cross-checks!
	multi_vis_idft_array_1D = block_diag(*[idft_array_1D for i in range(nu*nv-1)])

	k2 = np.dot(multi_vis_idft_array_1D.conjugate().T, s_fourier_only.reshape(neta+nq,-1).T.flatten())

	return s, s_im, s_quad_only, s_im_quad_only, s_fourier_only, s_im_fourier_only, bin_selector_in_k_cube_mask,high_spatial_frequency_selector_mask, k_cube_signal, k_sigma, ZM_mask, k_z_mean_mask


## ======================================================================================================
## ======================================================================================================

def map_out_bins_for_power_spectral_coefficients_HERA_Binning(nu,nv,nx,ny,nf,neta,nq, bin_selector_in_k_cube_mask, high_spatial_frequency_selector_mask, ZM_mask, k_z_mean_mask):
	# ---------------------------------------------
	###
	# Map out bins for power spectral coefficients
	###
	k_z, k_y, k_x = np.mgrid[-(nf/2):(nf/2),-(nu/2):(nu/2)+1,-(nv/2):(nv/2)+1]
	mod_k = (k_z**2. + k_y**2. + k_x**2.)**0.5
	bin_selector_in_k_cube_mask_vis_ordered = [bin_selector_in_k_cube_mask[i_bin].T.flatten() for i_bin in range(len(bin_selector_in_k_cube_mask))]
	k_perp_3D = (k_x**2.+k_y**2)**0.5
	ZM_mask = k_perp_3D>0.0
	ZM_2D_mask_vis_ordered = ZM_mask.T.flatten()
	high_spatial_frequency_mask_vis_ordered = np.logical_not(high_spatial_frequency_selector_mask.T.flatten())
	# k_z_mean_mask_vis_ordered = k_z_mean_mask.T.flatten()

	if nq>0:
		ZM_2D_and_high_spatial_frequencies_mask_vis_ordered = np.logical_and(high_spatial_frequency_mask_vis_ordered, ZM_2D_mask_vis_ordered)
		# ZM_2D_and_high_spatial_frequencies_mask_vis_ordered = np.logical_and.reduce(high_spatial_frequency_mask_vis_ordered, ZM_2D_mask_vis_ordered, k_z_mean_mask_vis_ordered)
	else:
		ZM_2D_and_high_spatial_frequencies_mask_vis_ordered = ZM_2D_mask_vis_ordered
		# ZM_2D_and_high_spatial_frequencies_mask_vis_ordered = np.logical_and(ZM_2D_mask_vis_ordered, k_z_mean_mask_vis_ordered)
	
	bin_selector_in_model_mask_vis_ordered = [bin_selector_in_k_cube_mask_vis_ordered[i_bin][ZM_2D_and_high_spatial_frequencies_mask_vis_ordered] for i_bin in range(len(bin_selector_in_k_cube_mask_vis_ordered))]

	return bin_selector_in_model_mask_vis_ordered


## ======================================================================================================
## ======================================================================================================

def map_out_bins_for_power_spectral_coefficients_WQ_v2_HERA_Binning(nu,nv,nx,ny,nf,neta,nq, bin_selector_in_k_cube_mask,high_spatial_frequency_selector_mask, ZM_mask, k_z_mean_mask):
	# ---------------------------------------------
	###
	# Map out bins for power spectral coefficients
	###

	bin_selector_in_model_mask_vis_ordered = map_out_bins_for_power_spectral_coefficients_HERA_Binning(nu,nv,nx,ny,nf,neta,nq, bin_selector_in_k_cube_mask, high_spatial_frequency_selector_mask, ZM_mask, k_z_mean_mask)

	bin_selector_in_model_mask_vis_ordered_reshaped = [bin_selector_in_model_mask_vis_ordered[i_bin].reshape(-1,neta) for i_bin in range(len(bin_selector_in_model_mask_vis_ordered))]

	WQ_boolean_array = np.zeros([bin_selector_in_model_mask_vis_ordered_reshaped[0].shape[0], nq]).astype('bool')
	bin_selector_in_model_mask_vis_ordered_reshaped_WQ = [np.hstack((bin_selector_in_model_mask_vis_ordered_reshaped[i_bin], WQ_boolean_array)) for i_bin in range(len(bin_selector_in_model_mask_vis_ordered_reshaped))]

	bin_selector_in_model_mask_vis_ordered_WQ = [bin_selector_in_model_mask_vis_ordered_reshaped_WQ[i_bin].flatten() for i_bin in range(len(bin_selector_in_model_mask_vis_ordered_reshaped_WQ))]

	k_z_mean_mask_vis_ordered = np.logical_not(k_z_mean_mask).T[np.logical_and(ZM_mask.T, np.logical_not(high_spatial_frequency_selector_mask).T)]
	k_z_mean_mask_vis_ordered_reshaped = k_z_mean_mask_vis_ordered.reshape(bin_selector_in_model_mask_vis_ordered_reshaped[0].shape)

	Quad_modes_only_boolean_array_vis_ordered = np.hstack((k_z_mean_mask_vis_ordered_reshaped.astype('bool'), np.logical_not(WQ_boolean_array))).flatten()

	return bin_selector_in_model_mask_vis_ordered_WQ, Quad_modes_only_boolean_array_vis_ordered


## ======================================================================================================
## ======================================================================================================

def map_out_bins_for_power_spectral_coefficients(nu,nv,nx,ny,nf,neta,nq, bin_1_selector_in_k_cube_mask, bin_2_selector_in_k_cube_mask, bin_3_selector_in_k_cube_mask, bin_4_selector_in_k_cube_mask, bin_5_selector_in_k_cube_mask, bin_6_selector_in_k_cube_mask, high_spatial_frequency_selector_mask):
	# ---------------------------------------------
	###
	# Map out bins for power spectral coefficients
	###

	k_z, k_y, k_x = np.mgrid[-(nf/2):(nf/2),-(nu/2):(nu/2)+1,-(nv/2):(nv/2)+1]
	mod_k = (k_z**2. + k_y**2. + k_x**2.)**0.5

	bin_1_selector_in_k_cube_mask_vis_ordered = bin_1_selector_in_k_cube_mask.T.flatten()
	bin_2_selector_in_k_cube_mask_vis_ordered = bin_2_selector_in_k_cube_mask.T.flatten()
	bin_3_selector_in_k_cube_mask_vis_ordered = bin_3_selector_in_k_cube_mask.T.flatten()
	bin_4_selector_in_k_cube_mask_vis_ordered = bin_4_selector_in_k_cube_mask.T.flatten()
	bin_5_selector_in_k_cube_mask_vis_ordered = bin_5_selector_in_k_cube_mask.T.flatten()
	bin_6_selector_in_k_cube_mask_vis_ordered = bin_6_selector_in_k_cube_mask.T.flatten()

	k_perp_3D = (k_x**2.+k_y**2)**0.5
	ZM_mask = k_perp_3D>0.0

	ZM_2D_mask_vis_ordered = ZM_mask.T.flatten()
	high_spatial_frequency_mask_vis_ordered = np.logical_not(high_spatial_frequency_selector_mask.T.flatten())

	ZM_2D_and_high_spatial_frequencies_mask_vis_ordered = np.logical_and(high_spatial_frequency_mask_vis_ordered, ZM_2D_mask_vis_ordered)

	bin_1_selector_in_model_mask_vis_ordered = bin_1_selector_in_k_cube_mask_vis_ordered[ZM_2D_and_high_spatial_frequencies_mask_vis_ordered]
	bin_2_selector_in_model_mask_vis_ordered = bin_2_selector_in_k_cube_mask_vis_ordered[ZM_2D_and_high_spatial_frequencies_mask_vis_ordered]
	bin_3_selector_in_model_mask_vis_ordered = bin_3_selector_in_k_cube_mask_vis_ordered[ZM_2D_and_high_spatial_frequencies_mask_vis_ordered]
	bin_4_selector_in_model_mask_vis_ordered = bin_4_selector_in_k_cube_mask_vis_ordered[ZM_2D_and_high_spatial_frequencies_mask_vis_ordered]
	bin_5_selector_in_model_mask_vis_ordered = bin_5_selector_in_k_cube_mask_vis_ordered[ZM_2D_and_high_spatial_frequencies_mask_vis_ordered]
	bin_6_selector_in_model_mask_vis_ordered = bin_6_selector_in_k_cube_mask_vis_ordered[ZM_2D_and_high_spatial_frequencies_mask_vis_ordered]

	return bin_1_selector_in_model_mask_vis_ordered, bin_2_selector_in_model_mask_vis_ordered, bin_3_selector_in_model_mask_vis_ordered, bin_4_selector_in_model_mask_vis_ordered, bin_5_selector_in_model_mask_vis_ordered, bin_6_selector_in_model_mask_vis_ordered,high_spatial_frequency_selector_mask


## ======================================================================================================
## ======================================================================================================

def map_out_bins_for_power_spectral_coefficients_WQ_v2(nu,nv,nx,ny,nf,neta,nq, bin_1_selector_in_k_cube_mask, bin_2_selector_in_k_cube_mask , bin_3_selector_in_k_cube_mask, bin_4_selector_in_k_cube_mask, bin_5_selector_in_k_cube_mask, bin_6_selector_in_k_cube_mask,high_spatial_frequency_selector_mask):
	# ---------------------------------------------
	###
	# Map out bins for power spectral coefficients
	###

	bin_1_selector_in_model_mask_vis_ordered, bin_2_selector_in_model_mask_vis_ordered, bin_3_selector_in_model_mask_vis_ordered, bin_4_selector_in_model_mask_vis_ordered, bin_5_selector_in_model_mask_vis_ordered, bin_6_selector_in_model_mask_vis_ordered,high_spatial_frequency_selector_mask = map_out_bins_for_power_spectral_coefficients(nu,nv,nx,ny,nf,neta,nq, bin_1_selector_in_k_cube_mask, bin_2_selector_in_k_cube_mask, bin_3_selector_in_k_cube_mask, bin_4_selector_in_k_cube_mask, bin_5_selector_in_k_cube_mask, bin_6_selector_in_k_cube_mask, high_spatial_frequency_selector_mask)

	bin_1_selector_in_model_mask_vis_ordered_reshaped = bin_1_selector_in_model_mask_vis_ordered.reshape(-1,neta)
	bin_2_selector_in_model_mask_vis_ordered_reshaped = bin_2_selector_in_model_mask_vis_ordered.reshape(-1,neta)
	bin_3_selector_in_model_mask_vis_ordered_reshaped = bin_3_selector_in_model_mask_vis_ordered.reshape(-1,neta)
	bin_4_selector_in_model_mask_vis_ordered_reshaped = bin_4_selector_in_model_mask_vis_ordered.reshape(-1,neta)
	bin_5_selector_in_model_mask_vis_ordered_reshaped = bin_5_selector_in_model_mask_vis_ordered.reshape(-1,neta)
	bin_6_selector_in_model_mask_vis_ordered_reshaped = bin_6_selector_in_model_mask_vis_ordered.reshape(-1,neta)

	WQ_boolean_array = np.zeros([bin_1_selector_in_model_mask_vis_ordered_reshaped.shape[0], nq]).astype('bool')
	bin_1_selector_in_model_mask_vis_ordered_reshaped_WQ = np.hstack((bin_1_selector_in_model_mask_vis_ordered_reshaped, WQ_boolean_array))
	bin_2_selector_in_model_mask_vis_ordered_reshaped_WQ = np.hstack((bin_2_selector_in_model_mask_vis_ordered_reshaped, WQ_boolean_array))
	bin_3_selector_in_model_mask_vis_ordered_reshaped_WQ = np.hstack((bin_3_selector_in_model_mask_vis_ordered_reshaped, WQ_boolean_array))
	bin_4_selector_in_model_mask_vis_ordered_reshaped_WQ = np.hstack((bin_4_selector_in_model_mask_vis_ordered_reshaped, WQ_boolean_array))
	bin_5_selector_in_model_mask_vis_ordered_reshaped_WQ = np.hstack((bin_5_selector_in_model_mask_vis_ordered_reshaped, WQ_boolean_array))
	bin_6_selector_in_model_mask_vis_ordered_reshaped_WQ = np.hstack((bin_6_selector_in_model_mask_vis_ordered_reshaped, WQ_boolean_array))

	bin_1_selector_in_model_mask_vis_ordered_WQ = bin_1_selector_in_model_mask_vis_ordered_reshaped_WQ.flatten()
	bin_2_selector_in_model_mask_vis_ordered_WQ = bin_2_selector_in_model_mask_vis_ordered_reshaped_WQ.flatten()
	bin_3_selector_in_model_mask_vis_ordered_WQ = bin_3_selector_in_model_mask_vis_ordered_reshaped_WQ.flatten()
	bin_4_selector_in_model_mask_vis_ordered_WQ = bin_4_selector_in_model_mask_vis_ordered_reshaped_WQ.flatten()
	bin_5_selector_in_model_mask_vis_ordered_WQ = bin_5_selector_in_model_mask_vis_ordered_reshaped_WQ.flatten()
	bin_6_selector_in_model_mask_vis_ordered_WQ = bin_6_selector_in_model_mask_vis_ordered_reshaped_WQ.flatten()

	Quad_modes_only_boolean_array_vis_ordered = np.hstack((np.zeros_like(bin_1_selector_in_model_mask_vis_ordered_reshaped).astype('bool'), np.logical_not(WQ_boolean_array))).flatten()

	return bin_1_selector_in_model_mask_vis_ordered_WQ, bin_2_selector_in_model_mask_vis_ordered_WQ, bin_3_selector_in_model_mask_vis_ordered_WQ, bin_4_selector_in_model_mask_vis_ordered_WQ, bin_5_selector_in_model_mask_vis_ordered_WQ, bin_6_selector_in_model_mask_vis_ordered_WQ, Quad_modes_only_boolean_array_vis_ordered


## ======================================================================================================
## ======================================================================================================

def generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector(sigma, s):
	sigma_squared_array = np.ones(s.size)*sigma**2 + 0j*np.ones(s.size)*sigma**2
	N = np.diag(sigma_squared_array)
	Ninv = np.diag(1./sigma_squared_array)
	if sigma>0: logNDet=np.sum(np.log(sigma_squared_array))
	sigma_complex = sigma/2**0.5
	noise_real = np.random.normal(0,sigma_complex,s.size)
	noise_imag = np.random.normal(0,sigma_complex,s.size)
	noise = noise_real+1j*noise_imag
	d=s+noise

	return d, noise, N, Ninv




def generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2(sigma, s, nu,nv,nx,ny,nf,neta,nq, **kwargs):

	##===== Defaults =======
	default_random_seed = ''
	
	##===== Inputs =======
	random_seed=kwargs.pop('random_seed',default_random_seed)

	if random_seed:
		print 'Using the following random_seed for dataset noise:', random_seed
		np.random.seed(random_seed)

	real_noise_cube = np.random.normal(0,sigma,[nf,ny,nx])

	import numpy
	axes_tuple = (1,2)
	vfft1=numpy.fft.ifftshift(real_noise_cube+0j, axes=axes_tuple)
	vfft1=numpy.fft.fftn(vfft1, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	vfft1=numpy.fft.fftshift(vfft1, axes=axes_tuple)

	sci_f, sci_v, sci_u = vfft1.shape
	sci_v_centre = sci_v/2
	sci_u_centre = sci_u/2
	vfft1_subset = vfft1[0:nf,sci_u_centre-nu/2:sci_u_centre+nu/2+1,sci_v_centre-nv/2:sci_v_centre+nv/2+1]
	noise_before_ZM = vfft1_subset.flatten()
	ZM_vis_ordered_mask = np.ones(nu*nv*nf)
	ZM_vis_ordered_mask[nf*((nu*nv)/2):nf*((nu*nv)/2+1)]=0
	ZM_vis_ordered_mask = ZM_vis_ordered_mask.astype('bool')
	ZM_chan_ordered_mask = ZM_vis_ordered_mask.reshape(-1, neta+nq).T.flatten()
	noise = noise_before_ZM[ZM_chan_ordered_mask]
	if sigma == 0:
		noise = noise*0.0
	else:
		noise = noise * sigma/noise.std()

	sigma_squared_array = np.ones(s.size)*sigma**2 + 0j*np.ones(s.size)*sigma**2
	N = np.diag(sigma_squared_array)
	Ninv = np.diag(1./sigma_squared_array)
	if sigma>0: logNDet=np.sum(np.log(sigma_squared_array))

	d=s+noise

	return d, noise, N, Ninv










Deal_with_Primary_Beam_Effects = False
###
# NOTE: if Deal_with_Primary_Beam_Effects is set to True the code below should be turned into a function to output a frequency dependent primary beam model matrix to be integrated into the likelihood.
###
if Deal_with_Primary_Beam_Effects:
	#######################
	###
	# Imaging test
	###
	Show=False
	Im = np.dot(Fprime_Fz,k_cube_signal.T.flatten()[ZM_2D_mask_vis_ordered]).reshape(-1,5,5)
	pylab.imshow(Im[0].real)
	pylab.colorbar()
	if Show:pylab.show()

	min_frequency_MHz = 120.0
	chan_width_MHz = 20.0 #Just for testing purposes - for easy visualisation of the PB changing width with frequency
	# chan_width_MHz = 0.2
	im_size_degs = 10.0
	pix_size_degs = float(im_size_degs)/nx

	###
	# Generate a frequency dependant (channel ordered) PB cube which defines the diagonal of the PB matrix when flattened
	###
	frequency, theta, phi = np.mgrid[0:nf,-(ny/2):(ny/2)+1,-(nx/2):(nx/2)+1]
	frequency_MHz = min_frequency_MHz+(frequency*chan_width_MHz)
	angular_radius = (theta**2. + phi**2.)**0.5
	angular_radius_degs = angular_radius*pix_size_degs
	PB_FWHM_degs = 8.
	PB_sigma_degs = PB_FWHM_degs/(2.*(2*np.log(2))**0.5)
	frequency_dependent_PB_sigma_degs = PB_sigma_degs * (frequency_MHz/frequency_MHz[0]) #linear scaling with frequency
	PB = np.exp(-0.5*(angular_radius_degs**2./frequency_dependent_PB_sigma_degs**2.))
	PB_matrix = np.diag(PB.flatten())


## ======================================================================================================
## ======================================================================================================

	###
	# Build up the PB matrix one channel at a time - this method is probably clearer when it comes to generalising to use custom primary beam models as a function of frequency. In that case, the Generate_PB fuction just needs to be switched out with an appropriate function for collecting the relevant model for the PB at that particular frequency.
	###
	def Generate_PB(angular_radius_degs_2D, frequency_dependent_PB_sigma_deg):
		PB_2D = np.exp(-0.5*(angular_radius_degs_2D**2./frequency_dependent_PB_sigma_deg**2.))
		return PB_2D

	PB_matrix = scipy.linalg.block_diag(*[np.diag(Generate_PB(angular_radius_degs[0], frequency_dependent_PB_sigma_degs[i][0]).flatten()) for i in range(len(frequency_dependent_PB_sigma_degs))])

	Show=False
	pylab.figure()
	pylab.imshow(PB[0].real)
	pylab.colorbar()	
	pylab.figure()
	pylab.imshow(PB[-1].real)
	pylab.colorbar()
	if Show:pylab.show()
	#######################

## ======================================================================================================
## ======================================================================================================

def update_Tb_experimental_std_K_to_correct_for_normalisation_resolution(Tb_experimental_std_K, simulation_FoV_deg, default_simulation_resolution_deg):

	# coding: utf-8

	# Generate initial unnormalised high-res GDSE sim:
	# - Generate high resolution white noise image
	# - fft to the uv-domain
	# - scale according to input spatial power law
	# 
	# Create simulation subsets and products required to normalise the high-res. sim. relative to the low resolution GSM model.
	# Constructing two additional products - the low-res centre of the high-res. uv-plane (containing only the scales sampled by the GSM) and a high-res uv-plane with all of the high-res information (the information not present in the GSM) zeroed.
	# ffting these products to the uv-domain should produce images with consistent standard deviations (confirming this will ensures that the GSM scaling factor can be correctly applied to the high-res uv-plane:
	# - Define a separate copy of the low-res centre of the high-res. uv-plane (containing only the scales sampled by the GSM)
	# - Define a separate copy of the high-res uv-plane with all of the high-res information (the information not present in the GSM) zeroed.
	# 
	# Derive normalisation constant:
	# - fft copy of the low-res centre of the high-res. uv-plane to the image. Call this GDSE_lr
	# - fft high-res uv-plane with all of the high-res information (the information not present in the GSM) zeroed to the image. Call this GDSE_hr_low_pass_filtered.
	# - GDSE_lr is at the same resolution as the GSM so can be normalised directly. GDSE_hr_low_pass_filtered is a Fourier interpolated version of GDSE_lr onto a higher resolution grid but with no aditional high-res power, it should therefore have the same standard deviation as GDSE_lr and the same normalisation factor should therefore be applicable. Confirm that this is the case!
	# 

	# Import libs

	# import numpy
	# import numpy as np
	# from numpy import * 
	# import pylab
	# import pylab as P
	# import scipy
	# from scipy import optimize
	# from scipy.optimize import curve_fit
	# from scipy import misc
	# from scipy import ndimage
	# import numpy as np

	Show=False

	# Define constants: GSM normalisation, FoV, resolution etc.

	# Tb_experimental_std_K = 62.0
	Tb_experimental_std_K = 62.0 #~70th percentile std according to GSM_map_std_at_-30_dec_v1d0.ipynb
	# FoV_deg = 12.34 #12.0 degrees is the required FoV for the foregrounds (matching the 12.0407656283 deg. of the 128*128 pix EoR sim)
	# FoV_deg = 12.134 #12.0 degrees is the required FoV for the foregrounds (matching the 12.0407656283 deg. of the 128*128 pix EoR sim)
	FoV_deg = simulation_FoV_deg
	FoV_rad = FoV_deg*(np.pi/180.)
	###
	#low-res. image constants
	###
	# lres_deg = 1./3 #1/3 deg Haslam map
	lres_deg = (56./60) #56 arcmin resolution Remazeilles Haslam map
	lres_rad = lres_deg*(np.pi/180.)
	N_lr = int(FoV_deg/lres_deg) #FoV_deg/start_res = 36
	if N_lr%2==0:
		N_lr = N_lr+1 #Simplify fft normalisation by using an odd n_side
	print N_lr
	d_Omega_lr = lres_rad**2.
	duv_lr = 1./d_Omega_lr

	###
	#high-res. image constants
	###
	# N_hr = 127
	# hres_deg = FoV_deg/N_hr
	hres_deg = default_simulation_resolution_deg
	N_hr = int(FoV_deg/default_simulation_resolution_deg)
	if N_hr%2==0:
		N_hr = N_hr+1 #Simplify fft normalisation by using an odd n_side
	hres_rad = hres_deg*(np.pi/180.)
	print N_hr
	print hres_deg #EoR sim. is 0.09406848147109376 deg.
	d_Omega_hr = hres_rad**2.
	duv_hr = 1./d_Omega_hr

	###
	#Spatial power spectrum
	###
	spatial_power_spectrum_index = -3.0
	spatial_amplitude_spectrum_index = spatial_power_spectrum_index/2.

	# Generate a GRN realisation map and fft to the uv-domain

	np.random.seed(1274+3142)
	wn_im_hr=np.random.normal(0,10,N_hr**2).reshape(N_hr,N_hr)
	axes_tuple=(0,1)
	fft_wn_im_hr=numpy.fft.ifftshift(wn_im_hr, axes=axes_tuple)
	fft_wn_im_hr=numpy.fft.fftn(fft_wn_im_hr, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	fft_wn_im_hr=numpy.fft.fftshift(fft_wn_im_hr, axes=axes_tuple)
	fft_wn_im_hr = d_Omega_hr*fft_wn_im_hr #Apply physical normalisation for FFT as per Beardsley doc.

	# Generate scaling field to impart desired spatial power spectrum

	u_axis_hr = (arange(N_hr)-(N_hr/2)) * (1./FoV_rad) #uv-plane in inverse degrees
	v_axis_hr = (arange(N_hr)-(N_hr/2)) * (1./FoV_rad) #uv-plane in inverse degrees
	u_hr,v_hr = np.meshgrid(u_axis_hr,v_axis_hr)
	u_vec_hr = u_hr.flatten()
	v_vec_hr = v_hr.flatten()
	uv_vec_hr = (u_vec_hr**2.+v_vec_hr**2.)**0.5

	uv_hr = (u_hr**2.+v_hr**2.)**0.5
	uv_spatial_power_law_scaling_hr = uv_hr**spatial_power_spectrum_index #i.e. uv_hr**-3.0
	uv_amplitude_scaling_hr = uv_spatial_power_law_scaling_hr**0.5
	uv_amplitude_scaling_hr[np.where(np.isinf(uv_amplitude_scaling_hr))] = 0.0 #Set the mean to zero
	uv_amplitude_scaling_hr = uv_amplitude_scaling_hr/uv_amplitude_scaling_hr.mean() #Unitary tranform

	# Scale fft'd white noise image in the uv-plane to impart GDSE spatial power spectrum

	scaled_fft_wn_im_hr = fft_wn_im_hr*uv_amplitude_scaling_hr

	P.imshow(np.log10(abs(scaled_fft_wn_im_hr)))
	P.colorbar()
	if Show: P.show()

	# Note: scaled_fft_wn_im_hr is the uv-plane representation of my unnormalised GDSE simulation.
	# Next step: perform normalisation.
	# To do this, create simulation subsets and products required to normalise the high-res. sim. relative to the low resolution GSM model. Constructing two additional products - the low-res centre of the high-res. uv-plane (containing only the scales sampled by the GSM) and a high-res uv-plane with all of the high-res information (the information not present in the GSM) zeroed. ffting these products to the uv-domain should produce images with consistent standard deviations (confirming this will ensures that the GSM scaling factor can be correctly applied to the high-res uv-plane.

	# Define a separate copy of the low-res centre of the high-res. uv-plane (containing only the scales sampled by the GSM)

	#The width of the outer part of the high-res. uv-plane containing information not sampled by the GSM
	padding_width = (N_hr-N_lr)/2
	print padding_width, N_hr, N_lr

	# Low-res. uv-plane to be fft'd to the image for direct comparison to the GSM standard deviation
	scaled_fft_wn_im_lr = scaled_fft_wn_im_hr[padding_width:-padding_width,padding_width:-padding_width].copy()

	P.imshow(np.log10(abs(scaled_fft_wn_im_lr)))
	P.colorbar()
	if Show: P.show()

	# Define a separate copy of the high-res uv-plane with all of the high-res information (the information not present in the GSM) zeroed.

	zero_padded_scaled_fft_wn_im_lr = np.zeros([N_hr,N_hr])+0.0j
	zero_padded_scaled_fft_wn_im_lr[padding_width:-padding_width,padding_width:-padding_width] = scaled_fft_wn_im_lr

	P.imshow(np.log10(abs(zero_padded_scaled_fft_wn_im_lr)))
	P.colorbar()
	if Show: P.show()

	# Derive normalisation constant.
	# fft copy of the low-res centre of the high-res. uv-plane to the image. Call this GDSE_lr
	# fft high-res uv-plane with all of the high-res information (the information not present in the GSM) zeroed to the image. Call this GDSE_hr_low_pass_filtered.

	scaled_wn_im_lr=numpy.fft.ifftshift(scaled_fft_wn_im_lr, axes=axes_tuple)
	scaled_wn_im_lr=numpy.fft.ifftn(scaled_wn_im_lr, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	scaled_wn_im_lr=numpy.fft.fftshift(scaled_wn_im_lr, axes=axes_tuple)
	scaled_wn_im_lr=duv_lr*scaled_wn_im_lr

	zero_padded_scaled_wn_im_lr=numpy.fft.ifftshift(zero_padded_scaled_fft_wn_im_lr, axes=axes_tuple)
	zero_padded_scaled_wn_im_lr=numpy.fft.ifftn(zero_padded_scaled_wn_im_lr, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	zero_padded_scaled_wn_im_lr=numpy.fft.fftshift(zero_padded_scaled_wn_im_lr, axes=axes_tuple)
	zero_padded_scaled_wn_im_lr=duv_hr*zero_padded_scaled_wn_im_lr

	print 'scaled_wn_im_lr.shape, scaled_wn_im_lr.std():', scaled_wn_im_lr.shape, scaled_wn_im_lr.std()
	print 'zero_padded_scaled_wn_im_lr.shape, zero_padded_scaled_wn_im_lr.std():', zero_padded_scaled_wn_im_lr.shape, zero_padded_scaled_wn_im_lr.std()

	P.figure()
	P.imshow(scaled_wn_im_lr.real)
	P.colorbar()

	P.figure()
	P.imshow(zero_padded_scaled_wn_im_lr.real)
	P.colorbar()
	if Show: P.show()

	# GDSE_lr is at the same resolution as the GSM so can be normalised directly. GDSE_hr_low_pass_filtered is a Fourier interpolated version of GDSE_lr onto a higher resolution grid but with no aditional high-res power, it should therefore have the same standard deviation as GDSE_lr and the same normalisation factor should therefore be applicable. Confirm that this is the case!

	gsm_normalisation_factor = (Tb_experimental_std_K/scaled_wn_im_lr.std())
	normalised_scaled_wn_im_lr = scaled_wn_im_lr* gsm_normalisation_factor
	normalised_zero_padded_scaled_wn_im_lr = zero_padded_scaled_wn_im_lr* gsm_normalisation_factor

	print duv_hr/duv_lr
	print 'normalised_scaled_wn_im_lr.shape, normalised_scaled_wn_im_lr.std():', normalised_scaled_wn_im_lr.shape, normalised_scaled_wn_im_lr.std()
	print 'normalised_zero_padded_scaled_wn_im_lr.shape, normalised_zero_padded_scaled_wn_im_lr.std():',normalised_zero_padded_scaled_wn_im_lr.shape, normalised_zero_padded_scaled_wn_im_lr.std()

	P.figure()
	P.imshow(normalised_scaled_wn_im_lr.real)
	P.colorbar()

	P.figure()
	P.imshow(normalised_zero_padded_scaled_wn_im_lr.real)
	P.colorbar()
	if Show: P.show()

	normalised_scaled_fft_wn_im_hr = scaled_fft_wn_im_hr*gsm_normalisation_factor

	P.imshow(np.log10(abs(normalised_scaled_fft_wn_im_hr)))
	P.colorbar()
	if Show: P.show()

	normalised_scaled_wn_im_hr=numpy.fft.ifftshift(normalised_scaled_fft_wn_im_hr, axes=axes_tuple)
	normalised_scaled_wn_im_hr=numpy.fft.ifftn(normalised_scaled_wn_im_hr, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
	normalised_scaled_wn_im_hr=numpy.fft.fftshift(normalised_scaled_wn_im_hr, axes=axes_tuple)
	normalised_scaled_wn_im_hr=duv_hr*normalised_scaled_wn_im_hr

	print 'normalised_scaled_wn_im_lr.shape, normalised_scaled_wn_im_lr.std():', normalised_scaled_wn_im_lr.shape, normalised_scaled_wn_im_lr.std()
	print 'normalised_zero_padded_scaled_wn_im_lr.shape, normalised_zero_padded_scaled_wn_im_lr.std():',normalised_zero_padded_scaled_wn_im_lr.shape, normalised_zero_padded_scaled_wn_im_lr.std()
	print 'normalised_scaled_wn_im_hr.shape, normalised_scaled_wn_im_hr.std():', normalised_scaled_wn_im_hr.shape, normalised_scaled_wn_im_hr.std()
	low_res_to_high_res_std_conversion_factor = normalised_scaled_wn_im_hr.std()/normalised_scaled_wn_im_lr.std()
	print 'low_res_to_high_res_std_conversion_factor', low_res_to_high_res_std_conversion_factor

	P.figure()
	P.imshow(normalised_scaled_wn_im_lr.real)
	P.colorbar()

	P.figure()
	P.imshow(normalised_zero_padded_scaled_wn_im_lr.real)
	P.colorbar()

	P.figure()
	P.imshow(normalised_scaled_wn_im_hr.real)
	P.colorbar()
	if Show: P.show()

	P.close('all')

	return low_res_to_high_res_std_conversion_factor

	
## ======================================================================================================
## ======================================================================================================

class GenerateForegroundCube(object):

	def __init__(self, nu,nv,neta,nq, beta_experimental_mean, beta_experimental_std, gamma_mean, gamma_sigma, Tb_experimental_mean_K, Tb_experimental_std_K, nu_min_MHz, channel_width_MHz, **kwargs):

		##===== Defaults =======
		default_random_seed = 3142
		default_k_parallel_scaling = False
		
		##===== Inputs =======
		self.random_seed=kwargs.pop('random_seed',default_random_seed)
		self.k_parallel_scaling=kwargs.pop('k_parallel_scaling',default_k_parallel_scaling)

		self.gamma_mean = gamma_mean
		self.gamma_sigma = gamma_sigma
		self.Tb_experimental_mean_K = Tb_experimental_mean_K 
		self.Tb_experimental_std_K = Tb_experimental_std_K
		self.beta_experimental_mean = beta_experimental_mean
		self.beta_experimental_std = beta_experimental_std
		self.nu_min_MHz = nu_min_MHz
		self.channel_width_MHz = channel_width_MHz

	def generate_GRN_for_A_and_beta_fields(self, nu,nv,nx,ny,nf,neta,nq, gamma_mean, gamma_sigma):
		n_kz = neta+nq
		n_ky = nv
		n_kx = nu

		n_kz_even =  n_kz%2==0
		if n_kz_even:
			n_kz_odd = n_kz-1
		else:
			n_kz_odd = n_kz

		np.random.seed(2391+self.random_seed)
		complex_GRN_real = np.random.normal(0,1,(n_kx*n_ky*n_kz_odd)/2)
		np.random.seed(1234+self.random_seed)
		complex_GRN_imag = np.random.normal(0,1,(n_kx*n_ky*n_kz_odd)/2)
		complex_GRN = complex_GRN_real+1j*complex_GRN_imag
		complex_GRN = complex_GRN/complex_GRN.std()

		mod_k, k_x, k_y, k_z, deltakperp, deltakpara, x, y, z = generate_k_cube_in_physical_coordinates_21cmFAST_v2d0(nu,nv,nx,ny,nf,neta,p.box_size_21cmFAST_pix_sc,p.box_size_21cmFAST_Mpc_sc)

		if n_kz_even:
			#Ignore unmoddelled Nyquist bin in mask by starting at mask channel 1
			flattened_mod_k = mod_k[1:].flatten()
			flattened_k_x = k_x[1:].flatten()
			flattened_k_y = k_y[1:].flatten()
			flattened_k_z = k_z[1:].flatten()
		else:
			flattened_mod_k = mod_k.flatten()
			flattened_k_x = k_x.flatten()
			flattened_k_y = k_y.flatten()
			flattened_k_z = k_z.flatten()
		flattened_mod_k_half_cube = flattened_mod_k[:flattened_mod_k.size/2]
		flattened_k_x_half_cube = flattened_k_x[:flattened_k_x.size/2]
		flattened_k_y_half_cube = flattened_k_y[:flattened_k_y.size/2]
		flattened_k_z_half_cube = flattened_k_z[:flattened_k_z.size/2]
		flattened_k_parallel_half_cube = (flattened_k_x_half_cube**2.+flattened_k_y_half_cube**2.)**0.5
		k_parallel0 = flattened_k_parallel_half_cube[-1]
		k_parallel_argmin = flattened_k_parallel_half_cube.argmin()
		flattened_k_parallel_half_cube[k_parallel_argmin::nx**2] = flattened_k_parallel_half_cube[k_parallel_argmin+1] #Remove k_parallel=0 values to avoid infs

		###
		# Impart input power spectrum onto the unitary half-cube
		###
		np.random.seed(1274+self.random_seed)
		gamma_grn = np.random.normal(gamma_mean, gamma_sigma, complex_GRN.size)
		if self.k_parallel_scaling:
			p_k = abs(flattened_k_parallel_half_cube/k_parallel0)**gamma_grn  * flattened_mod_k_half_cube**gamma_grn #Injected spatial power spectrum
			# set_trace()
			print p_k
		else:
			p_k = flattened_mod_k_half_cube**gamma_grn #Injected spatial power spectrum
		a_k = p_k**0.5 #Spatial amplitude spectrum, corresponding to p_k, with which to multiply complex_GRN to produce a random realisation
		foreground_k_space_amplitudes_half_cube = complex_GRN*a_k

		###
		# Generate Hermitian cube from half-cube + a chosen mean at (kx,ky,kz)=(0,0,0)
		###
		image_cube_mean = 1.0
		foreground_k_space_amplitudes_cube_vector = np.hstack((foreground_k_space_amplitudes_half_cube, image_cube_mean, foreground_k_space_amplitudes_half_cube[::-1].conjugate()))
		foreground_k_space_amplitudes_cube = foreground_k_space_amplitudes_cube_vector.reshape(n_kz_odd,n_ky,n_kx)

		if n_kz_even:
			#Add in Nyquist channel but give it zero power (since it is currently not part of the data model).
			foreground_k_space_amplitudes_cube_even = np.zeros([n_kz_odd+1,n_ky,n_kx])+0j
			foreground_k_space_amplitudes_cube_even[1:n_kz_odd+1,0:n_ky,0:n_kx]=foreground_k_space_amplitudes_cube
			foreground_k_space_amplitudes_cube = foreground_k_space_amplitudes_cube_even

		return foreground_k_space_amplitudes_cube

	def calculate_unnormalised_A_and_beta_fields(self, nu,nv,nx,ny,nf,neta,nq, gamma_mean, gamma_sigma):
		foreground_k_space_amplitudes_cube = self.generate_GRN_for_A_and_beta_fields(nu,nv,nx,ny,nf,neta,nq, gamma_mean, gamma_sigma)

		axes_tuple=(0,1,2)
		unnormalised_A=numpy.fft.ifftshift(foreground_k_space_amplitudes_cube+0j, axes=axes_tuple)
		unnormalised_A=numpy.fft.fftn(unnormalised_A, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
		unnormalised_A=numpy.fft.fftshift(unnormalised_A, axes=axes_tuple)
		unnormalised_beta = unnormalised_A.copy()
		return unnormalised_A, unnormalised_beta.real

	def generate_normalised_Tb_A_and_beta_fields(self, nu,nv,nx,ny,nf,neta,nq):
		Tb_experimental_mean_K = self.Tb_experimental_mean_K
		Tb_experimental_std_K = self.Tb_experimental_std_K
		beta_experimental_mean = self.beta_experimental_mean
		beta_experimental_std = self.beta_experimental_std
		nu_min_MHz = self.nu_min_MHz
		channel_width_MHz = self.channel_width_MHz
		gamma_mean = self.gamma_mean
		gamma_sigma = self.gamma_sigma

		unnormalised_A, unnormalised_beta = self.calculate_unnormalised_A_and_beta_fields(nu,nv,nx,ny,nf,neta,nq, gamma_mean, gamma_sigma)

		unnormalised_Tb = np.sum(unnormalised_A.real, axis=0)
		Tb_std_normalisation = (Tb_experimental_std_K/unnormalised_Tb.std())
		# Tb_mean_normalisation = (Tb_experimental_mean_K-unnormalised_Tb.mean())
		# Tb_std_normalisation = (Tb_experimental_std_K/unnormalised_Tb.std())
		# Tb = Tb_mean_normalisation + unnormalised_Tb*Tb_std_normalisation

		Tb = unnormalised_Tb*Tb_std_normalisation
		Tb_mean_normalisation = (Tb_experimental_mean_K-Tb.mean())
		Tb = Tb + Tb_mean_normalisation

		beta = unnormalised_beta*(beta_experimental_std/unnormalised_beta.std())
		beta = beta+(beta_experimental_mean-beta.mean())

		nu_array_MHz = nu_min_MHz+np.arange(nf)*channel_width_MHz
		# nu_array_MHz = nu_min_MHz+np.arange(38)*channel_width_MHz
		A_mean_normalisation = Tb_mean_normalisation/float(unnormalised_A.shape[0]) #Since this value is going to get summed over unnormalised_A.shape[0] times when summing to Tb
		A = A_mean_normalisation + unnormalised_A.real*Tb_std_normalisation
		A_nu = np.array([A*(nu_array_MHz[i_nu]/nu_min_MHz)**-beta for i_nu in range(len(nu_array_MHz))])
		Tb_nu = np.sum(A_nu, axis=1)
		print 'Tb_nu[0] - Tb', Tb_nu[0] - Tb
		# brk()
		return Tb_nu, A, beta, Tb, nu_array_MHz


## ======================================================================================================
## ======================================================================================================

def generate_masked_coordinate_cubes(cube_to_mask, nu,nv,nx,ny,nf,neta,nq):
	###
	# Generate k_cube physical coordinates to match the 21cmFAST input simulation
	###
	mod_k, k_x, k_y, k_z, deltakperp, deltakpara, x, y, z = generate_k_cube_in_physical_coordinates_21cmFAST_v2d0(nu,nv,nx,ny,nf,neta,p.box_size_21cmFAST_pix_sc,p.box_size_21cmFAST_Mpc_sc)

	#----------------
	###
	# Do not include high spatial frequency structure in the power spectral data since  these terms aren't included in the data model
	###
	Nyquist_k_z_mode = k_z[0,0,0]
	Second_highest_frequency_k_z_mode = k_z[-1,0,0]
	# Mean_k_z_mode = 0.0
	#NOTE: the k_z=0 term should not necessarily be masked out since it is still required as a quadratic component (and is not currently explicitly added in there) even if it is not used for calculating the power spectrum.
	if nq==1:
		high_spatial_frequency_selector_mask = k_z==Nyquist_k_z_mode
	else:
		high_spatial_frequency_selector_mask = np.logical_or.reduce((k_z==Nyquist_k_z_mode, k_z==Second_highest_frequency_k_z_mode))
	high_spatial_frequency_mask = np.logical_not(high_spatial_frequency_selector_mask)

	#----------------

	Mean_k_z_mode = 0.0
	k_z_mean_mask = k_z!=Mean_k_z_mode

	k_perp_3D = (k_x**2.+k_y**2)**0.5
	ZM_mask = k_perp_3D>0.0
	ZM_selector_mask = np.logical_not(ZM_mask)


	ZM_2D_mask_vis_ordered = ZM_mask.T.flatten()
	high_spatial_frequency_mask_vis_ordered = np.logical_not(high_spatial_frequency_selector_mask.T.flatten())

	if nq>0:
		ZM_2D_and_high_spatial_frequencies_mask_vis_ordered = np.logical_and(high_spatial_frequency_mask_vis_ordered, ZM_2D_mask_vis_ordered)
	else:
		ZM_2D_and_high_spatial_frequencies_mask_vis_ordered = ZM_2D_mask_vis_ordered

	model_cube_to_mask_vis_ordered = cube_to_mask.T.flatten()[ZM_2D_and_high_spatial_frequencies_mask_vis_ordered]

	if nq>0:
		model_cube_to_mask_vis_ordered_reshaped = model_cube_to_mask_vis_ordered.reshape(-1,neta)

		WQ_boolean_array = np.zeros([model_cube_to_mask_vis_ordered_reshaped.shape[0], nq]).astype('bool')
		WQ_inf_array = np.zeros([model_cube_to_mask_vis_ordered_reshaped.shape[0], nq])+np.inf
		model_cube_to_mask_vis_ordered_reshaped_WQ = np.hstack((model_cube_to_mask_vis_ordered_reshaped, WQ_inf_array))

		model_cube_to_mask_vis_ordered_WQ = model_cube_to_mask_vis_ordered_reshaped_WQ.flatten()
		print model_cube_to_mask_vis_ordered_WQ.reshape(8,-1).T

		k_z_mean_mask_vis_ordered = np.logical_not(k_z_mean_mask).T[np.logical_and(ZM_mask.T, np.logical_not(high_spatial_frequency_selector_mask).T)]
		k_z_mean_mask_vis_ordered_reshaped = k_z_mean_mask_vis_ordered.reshape(model_cube_to_mask_vis_ordered_reshaped.shape)

		Quad_modes_only_boolean_array_vis_ordered = np.hstack((k_z_mean_mask_vis_ordered_reshaped.astype('bool'), np.logical_not(WQ_boolean_array))).flatten()
		print Quad_modes_only_boolean_array_vis_ordered.reshape(8,-1).T

		return model_cube_to_mask_vis_ordered_WQ
		# return model_cube_to_mask_vis_ordered_WQ, Quad_modes_only_boolean_array_vis_ordered
	else:
		return model_cube_to_mask_vis_ordered


## ======================================================================================================
## ======================================================================================================

def generate_k_cube_model_spherical_binning(mod_k_masked, k_z_masked, nu,nv,nx,ny,nf,neta,nq):

	###
	# Generate k_cube physical coordinates to match the 21cmFAST input simulation
	###
	mod_k, k_x, k_y, k_z, deltakperp, deltakpara, x, y, z = generate_k_cube_in_physical_coordinates_21cmFAST_v2d0(nu,nv,nx,ny,nf,neta,p.box_size_21cmFAST_pix_sc,p.box_size_21cmFAST_Mpc_sc)
	
	modkscaleterm=1.5 #Value used in BEoRfgs and in 21cmFAST binning
	# binsize=deltakperp*1 #Value used in 21cmFAST
	# binsize=deltakperp*4 #Value used in BEoRfgs
	binsize=deltakperp*2 #Value used in BEoRfgs

	numKbins = 50
	modkbins = np.zeros([numKbins,2])
	modkbins[0,0]=0
	modkbins[0,1]=binsize

	for m1 in range(1,numKbins,1):
		binsize=binsize*modkscaleterm
		modkbins[m1,0]=modkbins[m1-1,1]
		modkbins[m1,1]=modkscaleterm*modkbins[m1,0]
			# print m1, modkbins[m1,0], modkbins[m1,1]

	total_elements = 0
	n_bins = 0
	#
	modkbins_containing_voxels=[]
	#
	for i_bin in range(numKbins):
		#NOTE: By requiring k_z>0 the constant term in the 1D FFT is now effectively a quadratic mode!
		# If it is to be included explicitly with the quadratic modes, then k_z==0 should be added to the quadratic selector mask
		n_elements = np.sum(np.logical_and.reduce((mod_k_masked>modkbins[i_bin,0], mod_k_masked<=modkbins[i_bin,1], k_z_masked!=0)))
		if n_elements>0:
			n_bins+=1
	 		print i_bin, modkbins[i_bin,0], modkbins[i_bin,1], n_elements 
	 		total_elements+=n_elements
	 		modkbins_containing_voxels.append((modkbins[i_bin], n_elements))

	print total_elements, mod_k_masked.size

	k_cube_voxels_in_bin=[]
	count=0
	for i_bin in range(len(modkbins_containing_voxels)):
		relevant_voxels = np.where(np.logical_and.reduce((mod_k_masked>modkbins_containing_voxels[i_bin][0][0], mod_k_masked<=modkbins_containing_voxels[i_bin][0][1], k_z_masked!=0)))
		print relevant_voxels
		print (len(relevant_voxels[0]))
		count += len(relevant_voxels[0])
		k_cube_voxels_in_bin.append(relevant_voxels)

	print count #should be mod_k_masked.shape[0]-3*nuv
	return k_cube_voxels_in_bin, modkbins_containing_voxels



## ======================================================================================================
## ======================================================================================================

def create_directory(Directory,**kwargs):
	
	if not os.path.exists(Directory):
		print 'Directory not found: \n\n'+Directory+"\n"
		print 'Creating required directory structure..'
		os.makedirs(Directory)
	
	return 0


## ======================================================================================================
## ======================================================================================================

def calc_mean_binned_k_vals(mod_k_masked, k_cube_voxels_in_bin, **kwargs):
	##===== Defaults =======
	default_save_k_vals = False
	default_k_vals_file = 'k_vals.txt'
	default_k_vals_dir = 'k_vals'
	
	##===== Inputs =======
	save_k_vals=kwargs.pop('save_k_vals',default_save_k_vals)
	k_vals_file=kwargs.pop('k_vals_file',default_k_vals_file)
	k_vals_dir=kwargs.pop('k_vals_dir',default_k_vals_dir)

	k_vals = []
	print '---Calculating k-vals---'
	for i_bin in range(len(k_cube_voxels_in_bin)):
		mean_mod_k = mod_k_masked[k_cube_voxels_in_bin[i_bin]].mean()
		k_vals.append(mean_mod_k)
		print i_bin, mean_mod_k

	if save_k_vals:
		create_directory(k_vals_dir)
		np.savetxt(k_vals_dir+'/'+k_vals_file, k_vals)
		
	return k_vals


## ======================================================================================================
## ======================================================================================================

def generate_k_cube_model_cylindrical_binning(mod_k_masked, k_z_masked, k_y_masked, k_x_masked, n_k_perp_bins, nu,nv,nx,ny,nf,neta,nq):
	###
	# Generate k_cube physical coordinates to match the 21cmFAST input simulation
	###
	mod_k, k_x, k_y, k_z, deltakperp, deltakpara, x, y, z = generate_k_cube_in_physical_coordinates_21cmFAST_v2d0(nu,nv,nx,ny,nf,neta,p.box_size_21cmFAST_pix_sc,p.box_size_21cmFAST_Mpc_sc)

	###
	# define mod_k binning
	###
	modkscaleterm=1.5 #Value used in BEoRfgs and in 21cmFAST binning
	# binsize=deltakperp*1 #Value used in 21cmFAST
	# binsize=deltakperp*4 #Value used in BEoRfgs
	binsize=deltakperp*2 #Value used in BEoRfgs

	numKbins = 50
	modkbins = np.zeros([numKbins,2])
	modkbins[0,0]=0
	modkbins[0,1]=binsize

	for m1 in range(1,numKbins,1):
		binsize=binsize*modkscaleterm
		modkbins[m1,0]=modkbins[m1-1,1]
		modkbins[m1,1]=modkscaleterm*modkbins[m1,0]
			# print m1, modkbins[m1,0], modkbins[m1,1]

	total_elements = 0
	n_bins = 0
	modkbins_containing_voxels=[]
	#
	for i_bin in range(numKbins):
		#NOTE: By requiring k_z>0 the constant term in the 1D FFT is now effectively a quadratic mode!
		# If it is to be included explicitly with the quadratic modes,, then k_z==0 should be added to the quadratic selector mask
		n_elements = np.sum(np.logical_and.reduce((mod_k_masked>modkbins[i_bin,0], mod_k_masked<=modkbins[i_bin,1], k_z_masked>0)))
		if n_elements>0:
			n_bins+=1
	 		print i_bin, modkbins[i_bin,0], modkbins[i_bin,1], n_elements 
	 		total_elements+=n_elements
	 		modkbins_containing_voxels.append((modkbins[i_bin], n_elements))

	print total_elements, mod_k_masked.size

	###
	# define k_perp binning
	###
	k_perp_3D = (k_x_masked**2.+k_y_masked**2)**0.5
	k_perp_min = k_perp_3D[np.isfinite(k_perp_3D)].min()
	k_perp_max = k_perp_3D[np.isfinite(k_perp_3D)].max()

	def output_k_perp_bins(k_perp_min, k_perp_max, n_k_perp_bins):
		k_perp_bins = np.vstack((np.linspace(k_perp_min*0.99, k_perp_max*1.01, n_k_perp_bins)[:-1],np.linspace(k_perp_min*0.99, k_perp_max*1.01, n_k_perp_bins)[1:])).T
		return k_perp_bins

	def return_k_perp_bins_with_voxels(k_perp_bins):
		total_elements = 0
		n_bins = 0
		k_perp_bins_containing_voxels=[]
		for i_bin in range(len(k_perp_bins)):
			#NOTE: By requiring k_z>0 the constant term in the 1D FFT is now effectively a quadratic mode!
			# If it is to be included explicitly with the quadratic modes,, then k_z==0 should be added to the quadratic selector mask
			k_perp_constraint = np.logical_and.reduce((k_perp_3D>k_perp_bins[i_bin][0], k_perp_3D<=k_perp_bins[i_bin][1]))
			n_elements = np.sum(k_perp_constraint)
			if n_elements>0:
				n_bins+=1
		 		print i_bin, k_perp_bins[i_bin,0], k_perp_bins[i_bin,1], n_elements 
		 		total_elements+=n_elements
		 		k_perp_bins_containing_voxels.append(k_perp_bins[i_bin])
		print total_elements, mod_k_masked.size #Note: total_elements should be mod_k_masked.size-2*nuv since it doesn't include the linear and quadratic mode channels
		return k_perp_bins_containing_voxels


	n_k_perp_bins_array = [n_k_perp_bins for _ in range(len(modkbins_containing_voxels))]

	k_perp_bins = [return_k_perp_bins_with_voxels(output_k_perp_bins(k_perp_min, k_perp_max, n_k_perp_bins_val)) for n_k_perp_bins_val in n_k_perp_bins_array] #Currently using equal bin widths for all k_z but can make the number of bins / the bin width a function of k_z if that turns out to be useful (i.e. use larger bins at larger k_z where there is less power).
	# k_perp_bins = [np.linspace(k_perp_min*0.99, k_perp_max*1.01, n_k_perp_bins) for _ in range(len(modkbins_containing_voxels))] #Currently using equal bin widths for all k_z but can make the number of bins / the bin width a function of k_z if that turns out to be useful (i.e. use larger bins at larger k_z where there is less power).

	k_cube_voxels_in_bin=[]
	count=0
	for i_mod_k_bin in range(len(modkbins_containing_voxels)):
		for j_k_perp_bin in range(len(k_perp_bins[0])):
			#Since calculating the cylindrical power spectrum - bin in k_z rather than mod_k. However maintain the mod_k bin limits for the k_z binning for approximate bin size consistency with previous results when calculating the cylindrical power spectrum with HERA 37.
			k_z_constraint = np.logical_and.reduce((abs(k_z_masked)>modkbins_containing_voxels[i_mod_k_bin][0][0], abs(k_z_masked)<=modkbins_containing_voxels[i_mod_k_bin][0][1], k_z_masked!=0))
			k_perp_constraint = np.logical_and.reduce((k_perp_3D>k_perp_bins[i_mod_k_bin][j_k_perp_bin][0], k_perp_3D<=k_perp_bins[i_mod_k_bin][j_k_perp_bin][1]))
			relevant_voxels = np.where(np.logical_and(k_z_constraint, k_perp_constraint))
			print relevant_voxels
			print (len(relevant_voxels[0]))
			count += len(relevant_voxels[0])
			k_cube_voxels_in_bin.append(relevant_voxels)

	print count #should be mod_k_masked.shape[0]-3*nuv
	return k_cube_voxels_in_bin, modkbins_containing_voxels, k_perp_bins


## ======================================================================================================
## ======================================================================================================





