


import numpy as np
import numpy
from numpy import arange, shape
import scipy
from subprocess import os
import sys
sys.path.append(os.path.expanduser('~/EoR/PolyChord1d9/PolyChord_WorkingInitSetup_Altered/'))
sys.path.append(os.path.expanduser('~/EoR/Python_Scripts/BayesEoR/SpatialPS/PolySpatialPS/'))
import PyPolyChord.PyPolyChord as PolyChord
import pylab

from Linalg_v1d0 import IDFT_Array_IDFT_2D_ZM_SH, makeGaussian, Produce_Full_Coordinate_Arrays, Produce_Coordinate_Arrays_ZM
from Linalg_v1d0 import Produce_Coordinate_Arrays_ZM_Coarse, Produce_Coordinate_Arrays_ZM_SH, Calc_Coords_High_Res_Im_to_Large_uv
from Linalg_v1d0 import Calc_Coords_Large_Im_to_High_Res_uv, Restore_Centre_Pixel, Calc_Indices_Centre_3x3_Grid
from Linalg_v1d0 import Delete_Centre_3x3_Grid, Delete_Centre_Pix, N_is_Odd, Calc_Indices_Centre_NxN_Grid, Obtain_Centre_NxN_Grid
from Linalg_v1d0 import Restore_Centre_3x3_Grid, Restore_Centre_NxN_Grid, Generate_Combined_Coarse_plus_Subharmic_uv_grids
from Linalg_v1d0 import IDFT_Array_IDFT_2D_ZM_SH, DFT_Array_DFT_2D, IDFT_Array_IDFT_2D, DFT_Array_DFT_2D_ZM, IDFT_Array_IDFT_2D_ZM
from Linalg_v1d0 import Construct_Hermitian, Construct_Hermitian_Gridding_Matrix, Construct_Hermitian_Gridding_Matrix_CosSin
from Linalg_v1d0 import Construct_Hermitian_Gridding_Matrix_CosSin_SH_v4, IDFT_Array_IDFT_1D, IDFT_Array_IDFT_1D
from Linalg_v1d0 import generate_gridding_matrix_vis_ordered_to_chan_ordered

from SimData_v1d0 import generate_test_sim_signal, map_out_bins_for_power_spectral_coefficients
from SimData_v1d0 import generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector
from SimData_v1d0 import generate_test_sim_signal_with_large_spectral_scales_1

nf=18
neta=18

nu=3
nv=3
nx=3
ny=3

Fz_normalisation = nf**0.5
DFT2D_Fz_normalisation = (nu*nv*nf)**0.5
n_Fourier = (nu*nv-1)*nf
n_quad = (nu*nv-1)*4
n_model = n_Fourier+n_quad
n_dat = n_Fourier


###
# Generate base matrices used in the likelihood
###
from scipy.linalg import block_diag
idft_array_1D=IDFT_Array_IDFT_1D(nf, neta)*Fz_normalisation #Note: the nf**0.5 normalisation factor results in a symmetric transform and is necessary for correctly estimating the power spectrum. However, it is not consistent with Python's asymmetric DFTs, therefore this factor needs to be removed when comparing to np.fft.fftn cross-checks!
idft_array = IDFT_Array_IDFT_2D_ZM(nu,nv,nx,ny)
dft_array = DFT_Array_DFT_2D_ZM(nu,nv,nx,ny)
multi_vis_idft_array_1D = block_diag(*[idft_array_1D for i in range(nu*nv-1)])
multi_chan_idft_array_noZMchan = block_diag(*[idft_array.T for i in range(nf)])
multi_chan_dft_array_noZMchan = block_diag(*[dft_array.T for i in range(nf)])
gridding_matrix_vis_ordered_to_chan_ordered = generate_gridding_matrix_vis_ordered_to_chan_ordered(nu,nv,nf)
gridding_matrix_chan_ordered_to_vis_ordered = gridding_matrix_vis_ordered_to_chan_ordered.T #NOTE: taking the transpose reverses the gridding. Important: This is what happens in dbar where Fz.conjugate().T is multiplied by d and the gridding_matrix_vis_ordered_to_chan_ordered.conjugate().T part of Fz changes d from being chan-ordered initially to vis-ordered after application (Note - .conjugate does nothing to the gridding matrix component of Fz, which is real, it only changes the 1D IDFT component into an 1D DFT).
Fz = np.dot(gridding_matrix_vis_ordered_to_chan_ordered, multi_vis_idft_array_1D)
Fprime = multi_chan_idft_array_noZMchan
Finv = multi_chan_dft_array_noZMchan
Fprime_Fz = np.dot(Fprime,Fz)

###
# Generate data = s+n (with known s and n), to estimate the power spectrum of
###
sigma=20.e-2
# s, s_im, k_cube_signal, k_cube_power_scaling_cube, random_k, bin_1_mask_and_ZM, bin_2_mask_and_ZM, bin_1_mask, bin_2_mask, k_sigma_1, k_sigma_2, bin_limit = generate_test_sim_signal_with_large_spectral_scales_1(nu,nv,nx,ny,nf,neta)
s, s_im, k_cube_signal, k_cube_power_scaling_cube, random_k, bin_1_mask_and_ZM, bin_2_mask_and_ZM, bin_1_mask, bin_2_mask, k_sigma_1, k_sigma_2, bin_limit = generate_test_sim_signal(nu,nv,nx,ny,nf,neta)
bin_1_mask_vis_ordered, bin_2_mask_vis_ordered, k_perp_3D, ZM_mask, bin_1_mask_correct_order_zm, bin_2_mask_correct_order_zm, ZM_2D_mask_vis_ordered = map_out_bins_for_power_spectral_coefficients(nu,nv,nx,ny,nf,neta, bin_1_mask, bin_2_mask)
d, noise, N, Ninv = generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector(sigma, s, bin_1_mask_vis_ordered, bin_2_mask_vis_ordered)

###
# Generate specific matrices used in the likelihood
###
T = np.dot(Finv,Fprime_Fz)
Ninv_d = np.dot(Ninv,d)
dbar = np.dot(T.conjugate().T,Ninv_d)

Ninv_T = np.dot(Ninv,T)
T_Ninv_T = np.dot(T.conjugate().T,Ninv_T)

Show=False
pylab.figure()
pylab.imshow(abs(T_Ninv_T))
pylab.colorbar()
if Show:pylab.show()






###
# Define some helper variables and the likelihood (actually the posterior probability distribution)
###
Sigma_Diag_Indices=np.diag_indices(shape(T_Ninv_T)[0])
Npar = shape(T_Ninv_T)[0]
x=[1.0]

from numpy import real
def log_likelihood(x, fit_single_elems=False, T_Ninv_T=T_Ninv_T, dbar=dbar):
	phi = [0.0]
	# phi = [0.0]*len(x)
	try:
		PowerI=np.ones(Npar)/k_cube_signal[bin_1_mask].var()*5.
		if fit_single_elems:
			PowerI[i_bin] = 1./x[0]**2.
		else:
			# PowerI=np.ones(Npar)/x[0]**2.
			PowerI[bin_1_mask_vis_ordered] = 1./x[0]**2.
			PowerI[bin_2_mask_vis_ordered] = 1./x[1]**2.
		PhiI=PowerI
		Sigma=T_Ninv_T.copy()
		Sigma[Sigma_Diag_Indices]+=PhiI

		# Sigmacho = scipy.linalg.cholesky(Sigma, lower=True).astype(np.complex256)
		# SigmaI_dbar = scipy.linalg.cho_solve((Sigmacho,True), dbar)

		SigmaI = scipy.linalg.inv(Sigma)
		SigmaI_dbar = np.dot(SigmaI, dbar)
		dbarSigmaIdbar=np.dot(dbar.conjugate().T,SigmaI_dbar)

		logSigmaDet=np.linalg.slogdet(Sigma)[1]
		# logSigmaDet=2.*np.sum(np.log(np.diag(Sigmacho)))
		logPhiDet=-1*np.sum(np.log(PhiI)).real #Only possible because Phi is diagonal (otherwise would need to calc np.linalg.slogdet(Phi)). -1 factor is to get logPhiDet from logPhiIDet. Note: taking the real part of this calculation matches the solution given by np.linalg.slogdet(Phi))

		dbarSigmaIdbar=np.dot(dbar.conjugate().T,SigmaI_dbar)

		MargLogL =  -0.5*logSigmaDet -0.5*logPhiDet + 0.5*dbarSigmaIdbar
		vals = map(real, (-0.5*logSigmaDet, -0.5*logPhiDet, 0.5*dbarSigmaIdbar, MargLogL))
		# print vals[0], vals[1], vals[2], vals[3]
		MargLogL =  MargLogL.real
		return (MargLogL.squeeze())*1.0, phi
	except:
		return -np.inf, phi



###
# Calculate the power spectrum of a single bin manually
###
correct_answer = k_cube_signal[bin_1_mask_and_ZM].std()

from numpy import linspace
param_array=[]
loglike_array=[]
for i in linspace(0,2.e1,2000):
	loglike,phi=log_likelihood([i, 1.])
	param_array.append(i)
	loglike_array.append(loglike)

maxLx=[]
argmax =  np.array(loglike_array)[np.logical_not(np.isinf(loglike_array))].argmax()
maxLx.append(np.array(param_array)[np.logical_not(np.isinf(loglike_array))][argmax])
print 'maxLx[-1], np.dot(T.conjugate().T,s).std(), sigma:'
print maxLx[-1], np.dot(T.conjugate().T,s).std(), sigma
print maxLx[-1], correct_answer
print np.dot(T.conjugate().T, s)[bin_1_mask_vis_ordered].std()


###
# Plot (unnormalised) posterior probability array
###
import pylab
fig,ax=pylab.subplots()
ax.errorbar(param_array, loglike_array)
if Show:fig.show()




s1, s_im1, k_cube_signal, k_cube_power_scaling_cube, random_k, bin_1_mask_and_ZM, bin_2_mask_and_ZM, bin_1_mask, bin_2_mask, k_sigma_1, k_sigma_2, bin_limit = generate_test_sim_signal(nu,nv,nx,ny,nf,neta)


s2, s_im2, k_cube_signal, k_cube_power_scaling_cube, random_k, bin_1_mask_and_ZM, bin_2_mask_and_ZM, bin_1_mask, bin_2_mask, k_sigma_1, k_sigma_2, bin_limit = generate_test_sim_signal_with_large_spectral_scales_1(nu,nv,nx,ny,nf,neta)

v1 = np.dot(multi_chan_dft_array_noZMchan/(nu*nv)**0.5, s_im1.flatten())

v2 = np.dot(multi_chan_dft_array_noZMchan/(nu*nv)**0.5, s_im2.flatten())



axes_tuple = (0,1,2)
vfft1=numpy.fft.ifftshift(s_im1+0j, axes=axes_tuple)
vfft1=numpy.fft.fftn(vfft1, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
vfft1=numpy.fft.fftshift(vfft1, axes=axes_tuple)

vfft2=numpy.fft.ifftshift(s_im2+0j, axes=axes_tuple)
vfft2=numpy.fft.fftn(vfft2, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
vfft2=numpy.fft.fftshift(vfft2, axes=axes_tuple)








def calc_SigmaI_dbar(x, fit_single_elems=False, T_Ninv_T=T_Ninv_T, dbar=dbar, **kwargs):
	##===== Defaults =======
	diagonal_sigma = True

	##===== Inputs =======
	if 'diagonal_sigma' in kwargs:
		diagonal_sigma=kwargs['diagonal_sigma']
	
	PowerI=np.ones(Npar)/k_cube_signal[bin_1_mask].var()*5.
	if fit_single_elems:
		PowerI[i_bin] = 1./x[0]**2.
	else:
		# PowerI=np.ones(Npar)/x[0]**2.
		PowerI[:n_Fourier][bin_1_mask_vis_ordered] = 1./x[0]**2.
		PowerI[:n_Fourier][bin_2_mask_vis_ordered] = 1./x[1]**2.
	PowerI[n_Fourier:] = 0.0 #Set inverse power in quadratic modes to zero (infinite width Gaussian -> unconstrained / uniform probability distribution quad amplitudes?)
	PhiI=PowerI
	Sigma=T_Ninv_T.copy()
	Sigma[Sigma_Diag_Indices]+=PhiI

	# Sigmacho = scipy.linalg.cholesky(Sigma, lower=True).astype(np.complex256)
	# SigmaI_dbar = scipy.linalg.cho_solve((Sigmacho,True), dbar)

	if diagonal_sigma:
		# print 'in diagonal_sigma'
		SigmaI = np.diag(1./Sigma.diagonal())
	else:
		SigmaI = scipy.linalg.inv(Sigma)
	SigmaI_dbar = np.dot(SigmaI, dbar)
	return SigmaI_dbar


# maxL_k_cube_signal = calc_SigmaI_dbar([0.1,0.01], diagonal_sigma=True)
maxL_k_cube_signal = calc_SigmaI_dbar([2.,10.], diagonal_sigma=True)
# maxL_k_cube_signal = calc_SigmaI_dbar([np.dot(T.conjugate().T, s)[bin_1_mask_vis_ordered].std(), np.dot(T.conjugate().T, s)[bin_2_mask_vis_ordered].std()], diagonal_sigma=True)
# maxL_k_cube_signal = calc_SigmaI_dbar([maxLx[-1], maxLx[-1]], diagonal_sigma=True)

print maxL_k_cube_signal[0:6]
print k_cube_signal.T.flatten()[0:6]


pylab.close('all')
a=(maxL_k_cube_signal - k_cube_signal.T.flatten()[ZM_mask.T.flatten()])
pylab.errorbar(arange(len(a)), abs(a))
pylab.show()





# ###
# # Calculate the power spectrum of using PolyChord
# ###

# #######################
# ###
# # PolyChord setup
# Bin1 = [0.0, 100.0]
# Bin2 = [0.0, 100.0]

# # priors_min_max=[Bin1,]
# priors_min_max=[Bin1,Bin2]

# class PriorC(object):
# 	def __init__(self, priors_min_max):
# 		self.priors_min_max=priors_min_max

# 	def prior_func(self, cube):
# 		pmm = self.priors_min_max
# 		theta = []
#         	for i_p in range(len(cube)):
# 			theta_i = pmm[i_p][0]+((pmm[i_p][1]-pmm[i_p][0])*cube[i_p])
#         		theta.append(theta_i)
#         	return theta


# prior_c = PriorC(priors_min_max)
# nDerived = 0
# nDims = 2

# base_dir = 'chains'
# if not os.path.isdir(base_dir):
#     	os.makedirs(base_dir+'/clusters/')

# file_root = 'test_v1d0-'

# PolyChord.mpi_notification()
# PolyChord.run_nested_sampling(log_likelihood, nDims, nDerived, file_root=file_root, read_resume=False, prior=prior_c.prior_func)


# print np.dot(T.conjugate().T, s)[bin_1_mask_vis_ordered].std()
# print np.dot(T.conjugate().T, s)[bin_2_mask_vis_ordered].std()
# #######################



















