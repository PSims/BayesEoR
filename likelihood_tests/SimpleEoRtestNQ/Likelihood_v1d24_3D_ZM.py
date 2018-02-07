import numpy as np
import numpy
from numpy import arange, shape
import scipy
from subprocess import os
import sys
sys.path.append(os.path.expanduser('~/EoR/PolyChord1d9/PolyChord_WorkingInitSetup_Altered/'))
sys.path.append(os.path.expanduser('~/EoR/Python_Scripts/BayesEoR/SpatialPS/PolySpatialPS/'))
import PyPolyChord.PyPolyChord as PolyChord

from Linalg_v1d0 import IDFT_Array_IDFT_2D_ZM_SH, makeGaussian, Produce_Full_Coordinate_Arrays, Produce_Coordinate_Arrays_ZM
from Linalg_v1d0 import Produce_Coordinate_Arrays_ZM_Coarse, Produce_Coordinate_Arrays_ZM_SH, Calc_Coords_High_Res_Im_to_Large_uv
from Linalg_v1d0 import Calc_Coords_Large_Im_to_High_Res_uv, Restore_Centre_Pixel, Calc_Indices_Centre_3x3_Grid
from Linalg_v1d0 import Delete_Centre_3x3_Grid, Delete_Centre_Pix, N_is_Odd, Calc_Indices_Centre_NxN_Grid, Obtain_Centre_NxN_Grid
from Linalg_v1d0 import Restore_Centre_3x3_Grid, Restore_Centre_NxN_Grid, Generate_Combined_Coarse_plus_Subharmic_uv_grids
from Linalg_v1d0 import IDFT_Array_IDFT_2D_ZM_SH, DFT_Array_DFT_2D, IDFT_Array_IDFT_2D, DFT_Array_DFT_2D_ZM, IDFT_Array_IDFT_2D_ZM
from Linalg_v1d0 import Construct_Hermitian, Construct_Hermitian_Gridding_Matrix, Construct_Hermitian_Gridding_Matrix_CosSin
from Linalg_v1d0 import Construct_Hermitian_Gridding_Matrix_CosSin_SH_v4, IDFT_Array_IDFT_1D, IDFT_Array_IDFT_1D


nf=6
neta=6

nu=5
nv=5
nx=5
ny=5

Fz_normalisation = nf**0.5
DFT2D_Fz_normalisation = (nu*nv*nf)**0.5

#-----------------------------------------------------
#-----------zero-mean-frequency 1D DFT transforms-----
#----------------------------------------------------- 

###
# Note: for the zero-mean-frequency 1D DFT transform, clearly the 2D DFT and IDF multi-channel transforms also have to be updated to transform one fewer channel
###

###
# Test 1D
###
idft_array_1D=IDFT_Array_IDFT_1D(nf, neta)*Fz_normalisation #Note: the nf**0.5 normalisation factor results in a symmetric transform and is necessary for correctly estimating the power spectrum. However, it is not consistent with Python's asymmetric DFTs, therefore this factor needs to be removed when comparing to np.fft.fftn cross-checks!
random_eta = np.random.normal(0,1,nf*nu*nv).reshape(nf,nu,nv)+1j*np.random.normal(0,1,nf*nu*nv).reshape(nf,nu,nv)

axes_tuple = (0,)
ifft_y_1d=numpy.fft.ifftshift(random_eta+0j, axes=axes_tuple)
ifft_y_1d=numpy.fft.ifftn(ifft_y_1d, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
ifft_y_1d=numpy.fft.fftshift(ifft_y_1d, axes=axes_tuple)

from scipy.linalg import block_diag
multi_vis_idft_array_1D = block_diag(*[idft_array_1D for i in range(nu*nv)])
ifft_y_1d_2d1 = np.dot(multi_vis_idft_array_1D, random_eta.T.reshape(-1,1)).reshape(nu,nv,nf).T

print ifft_y_1d[-1] - ifft_y_1d_2d1[-1]/Fz_normalisation
print ifft_y_1d[1] - ifft_y_1d_2d1[1]/Fz_normalisation





###
# Test 2D
###

axes_tuple = (1,2)
ifft_y=numpy.fft.ifftshift(random_eta+0j, axes=axes_tuple)
ifft_y=numpy.fft.ifftn(ifft_y, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
ifft_y=numpy.fft.fftshift(ifft_y, axes=axes_tuple)

idft_array = IDFT_Array_IDFT_2D(nu,nv,nx,ny)
multi_chan_idft_array = block_diag(*[idft_array for i in range(nf)])
ifft_y2 = np.dot(multi_chan_idft_array, random_eta.reshape(-1,1)).reshape(ifft_y.shape)
print abs(ifft_y-ifft_y2).max()


axes_tuple = (1,2)
fft_y=numpy.fft.ifftshift(ifft_y+0j, axes=axes_tuple)
fft_y=numpy.fft.fftn(fft_y, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
fft_y=numpy.fft.fftshift(fft_y, axes=axes_tuple)

print abs(random_eta-fft_y).max()

dft_array = DFT_Array_DFT_2D(nu,nv,nx,ny)
multi_chan_dft_array = block_diag(*[dft_array for i in range(nf)])
fft_y2 = np.dot(multi_chan_dft_array, ifft_y.reshape(-1,1)).reshape(fft_y.shape)
print abs(fft_y-fft_y2).max()








from scipy.linalg import block_diag

idft_array_1D=IDFT_Array_IDFT_1D(nf, neta)*Fz_normalisation #Note: the nf**0.5 normalisation factor results in a symmetric transform and is necessary for correctly estimating the power spectrum. However, it is not consistent with Python's asymmetric DFTs, therefore this factor needs to be removed when comparing to np.fft.fftn cross-checks!
idft_array = IDFT_Array_IDFT_2D_ZM(nu,nv,nx,ny)
dft_array = DFT_Array_DFT_2D_ZM(nu,nv,nx,ny)

multi_vis_idft_array_1D = block_diag(*[idft_array_1D for i in range(nu*nv-1)])
multi_chan_idft_array_noZMchan = block_diag(*[idft_array.T for i in range(nf)])
multi_chan_dft_array_noZMchan = block_diag(*[dft_array.T for i in range(nf)])


#------------------------------------------
#------Construct gridding matrix-----------
#------------------------------------------
###
# Note: the gridding logic works as follows:
# - In visibility order there are 37 complex numbers per vis (in the ZM case) follwing the 1D DFT (eta -> frequency)
# - Following gridding there will be (nu*nv-1) coarse grid points per channel
# - If d1 is the data in channel order with the numbers:
# [[0, 1, 2],
#  [3, 4, 5],
#  [6, 7, 8]]
# in d1[0], however, 4 is removed by the ZM; then the order of the first values in the visibility spectra in d1.T.flatten() will be: 0,3,6,1,7,2,5,8. So, the gridder needs to grab values in the order: 0*37+i_chan, 3*37+i_chan, 6*37+i_chan, 1*37+i_chan, etc. up to 8.

# could pre-build a list with e.g. 0,3,6,1,7,2,5,8 in it (generalised to the relevant nu and nv size) for selection from, rather than generating on the fly!?

def calc_vis_selection_numbers(nu,nv):
	required_chan_order = arange(nu*nv).reshape(nu,nv)
	visibility_spectrum_order = required_chan_order.T
	grab_order_for_visibility_spectrum_ordered_to_chan_ordered_reordering = visibility_spectrum_order.argsort()
	return grab_order_for_visibility_spectrum_ordered_to_chan_ordered_reordering

def calc_vis_selection_numbers_ZM(nu,nv):
	required_chan_order = arange(nu*nv).reshape(nu,nv)
	visibility_spectrum_order = required_chan_order.T
	r = ((np.arange(nu)-nu/2).reshape(-1,1)**2. + (np.arange(nv)-nv/2).reshape(1,-1)**2.)**0.5
	non_excluded_values_mask = r>0.5 #true for everything other than the central 9 pix
	visibility_spectrum_order_ZM = visibility_spectrum_order[non_excluded_values_mask]
	grab_order_for_visibility_spectrum_ordered_to_chan_ordered_reordering_ZM = visibility_spectrum_order_ZM.argsort()
	return grab_order_for_visibility_spectrum_ordered_to_chan_ordered_reordering_ZM

def calc_vis_selection_numbers_SH(nu,nv, U_oversampling_Factor=1.0, V_oversampling_Factor=1.0):
	required_chan_order = arange(nu*nv).reshape(nu,nv)
	visibility_spectrum_order = required_chan_order.T
	r = ((np.arange(nu)-nu/2).reshape(-1,1)**2. + (np.arange(nv)-nv/2).reshape(1,-1)**2.)**0.5
	non_excluded_values_mask = r>1.5 #true for everything other than the central 9 pix
	visibility_spectrum_order_ZM_coarse_grid = visibility_spectrum_order[non_excluded_values_mask]
	grab_order_for_visibility_spectrum_ordered_to_chan_ordered_reordering_ZM_coarse_grid = visibility_spectrum_order_ZM_coarse_grid.argsort()
	grab_order_for_visibility_spectrum_ordered_to_chan_ordered_reordering_ZM_SH_grid = calc_vis_selection_numbers_ZM(3*U_oversampling_Factor, 3*V_oversampling_Factor)
	return grab_order_for_visibility_spectrum_ordered_to_chan_ordered_reordering_ZM_coarse_grid, grab_order_for_visibility_spectrum_ordered_to_chan_ordered_reordering_ZM_SH_grid

vis_grab_order = calc_vis_selection_numbers_ZM(nu,nv)
vals_per_chan = vis_grab_order.size

gridding_matrix_vis_ordered_to_chan_ordered = np.zeros([vals_per_chan*(nf), vals_per_chan*(nf)])
###
for i in range(nf):
	for j, vis_grab_val in enumerate(vis_grab_order):
			row_number = (i*vals_per_chan)+j
			grid_pix = i+vis_grab_val*(nf) #pixel to grab from vis-ordered vector and place as next chan-ordered value
			print i,j,vis_grab_val,row_number,grid_pix
			gridding_matrix_vis_ordered_to_chan_ordered[row_number, grid_pix]=1


###
# Test 3D with 1D zero mean
###
a=np.arange(nf*nu*nv).reshape(nf,nu,nv)
a1=a.T.flatten()
remove_indices = [np.where(a1==x) for x in (((nu*nv)/2)+arange(nf)*(nu*nv))]
a2=np.delete(a1, remove_indices)
a2 = a2.reshape(-1,1)

ga2 = np.dot(gridding_matrix_vis_ordered_to_chan_ordered, a2)
correct_order_zm = np.delete(arange(nf*nu*nv),[np.where(arange(nf*nu*nv)==x) for x in (((nu*nv)/2)+arange(nf)*(nu*nv))])

print abs(ga2.flatten()-correct_order_zm).max() #If the printout is 0.0 then gridding_matrix_vis_ordered_to_chan_ordered is correctly constructed! 

###
# Useful lemma!
###
gridding_matrix_chan_ordered_to_vis_ordered = gridding_matrix_vis_ordered_to_chan_ordered.T #NOTE: taking the transpose reverses the gridding. Important: This is what happens in dbar where Fz.conjugate().T is multiplied by d and the gridding_matrix_vis_ordered_to_chan_ordered.conjugate().T part of Fz changes d from being chan-ordered initially to vis-ordered after application (Note - .conjugate does nothing to the gridding matrix component of Fz, which is real, it only changes the 1D IDFT component into an 1D DFT).
np.dot(gridding_matrix_chan_ordered_to_vis_ordered, correct_order_zm)-a2.flatten() #If the printout is 0.0 then gridding_matrix_vis_ordered_to_chan_ordered is correctly constructed! 





# ---------------------------------------------
###
# Generate test sim data
###
k_z, k_y, k_x = np.mgrid[-(nf/2):(nf/2),-(nu/2):(nu/2)+1,-(nv/2):(nv/2)+1]
mod_k = (k_z**2. + k_y**2. + k_x**2.)**0.5
k_cube_power_scaling_cube = np.zeros([nf,nu,nv])

k_sigma_1 = 2.0
k_sigma_2 = 10.0
bin_limit = 2.0
bin_1_mask = np.logical_and(bin_limit>mod_k, mod_k>0.0)
bin_2_mask = mod_k>=bin_limit
k_cube_power_scaling_cube[bin_1_mask] = k_sigma_1
k_cube_power_scaling_cube[bin_2_mask] = k_sigma_2

random_im=np.random.normal(0.,1,nf*nu*nv).reshape(nf,nu,nv)
axes_tuple = (0,1,2)
random_k=numpy.fft.ifftshift(random_im+0j, axes=axes_tuple)
random_k=numpy.fft.fftn(random_k, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
random_k=numpy.fft.fftshift(random_k, axes=axes_tuple)
random_k = random_k/random_k.std()

k_perp_3D = (k_x**2.+k_y**2)**0.5
ZM_mask = k_perp_3D>0.0

bin_1_mask_and_ZM = np.logical_and.reduce((bin_1_mask, ZM_mask))
bin_2_mask_and_ZM = np.logical_and.reduce((bin_2_mask, ZM_mask))

# k_cube_signal = random_k.copy()
# k_cube_signal[bin_1_mask_and_ZM] *= k_sigma_1/k_cube_signal[bin_1_mask_and_ZM].std()
# k_cube_signal[bin_2_mask_and_ZM] *= k_sigma_2/k_cube_signal[bin_2_mask_and_ZM].std()

# print k_cube_signal[bin_1_mask_and_ZM].std()
# print k_cube_signal[bin_2_mask_and_ZM].std()
# print k_cube_signal[ZM_mask].std()

k_cube_signal = k_cube_power_scaling_cube*random_k

axes_tuple = (0,1,2)
im_power_scaling=numpy.fft.ifftshift(random_k+0j, axes=axes_tuple)
im_power_scaling=numpy.fft.ifftn(im_power_scaling, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
im_power_scaling=numpy.fft.fftshift(im_power_scaling, axes=axes_tuple)

s_im=numpy.fft.ifftshift(k_cube_signal+0j, axes=axes_tuple)
s_im=numpy.fft.ifftn(s_im, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
s_im=numpy.fft.fftshift(s_im, axes=axes_tuple)

s_im = s_im/(1./DFT2D_Fz_normalisation)
# s_im = s_im/im_power_scaling.std()
# s_im = s_im * (k_cube_signal.std()/s_im.std())


# import pylab as P
# P.imshow(s_im[1].real)
# P.colorbar()
# P.show()

s = np.dot(multi_chan_dft_array_noZMchan/(nu*nv)**0.5, s_im.flatten())
# s = np.dot(multi_chan_dft_array_noZMchan.conjugate()/(nu*nv)**0.5, s_im.flatten())




# ---------------------------------------------

# ---------------------------------------------
###
# Map out bins for power spectral coefficients
###
k_z, k_y, k_x = np.mgrid[-(nf/2):(nf/2),-(nu/2):(nu/2)+1,-(nv/2):(nv/2)+1]
mod_k = (k_z**2. + k_y**2. + k_x**2.)**0.5

bin_1_mask_vis_ordered = bin_1_mask.T.flatten()
bin_2_mask_vis_ordered = bin_2_mask.T.flatten()

k_perp_3D = (k_x**2.+k_y**2)**0.5
ZM_mask = k_perp_3D>0.0

bin_1_mask_correct_order_zm = np.delete(bin_1_mask.flatten(),[np.where(arange(nf*nu*nv)==x) for x in (((nu*nv)/2)+arange(nf)*(nu*nv))])
bin_2_mask_correct_order_zm = np.delete(bin_2_mask.flatten(),[np.where(arange(nf*nu*nv)==x) for x in (((nu*nv)/2)+arange(nf)*(nu*nv))])
np.dot(gridding_matrix_chan_ordered_to_vis_ordered, bin_1_mask_correct_order_zm)

ZM_2D_mask_vis_ordered = ZM_mask.T.flatten()

bin_1_mask_vis_ordered = bin_1_mask_vis_ordered[ZM_2D_mask_vis_ordered]
bin_2_mask_vis_ordered = bin_2_mask_vis_ordered[ZM_2D_mask_vis_ordered]

np.dot(gridding_matrix_chan_ordered_to_vis_ordered, bin_1_mask_correct_order_zm) - bin_1_mask_vis_ordered

# ---------------------------------------------


sigma=20.e-3
sigma_squared_array = np.ones(s.size)*sigma**2 + 0j*np.ones(s.size)*sigma**2
N = np.diag(sigma_squared_array)
Ninv = np.diag(1./sigma_squared_array)
logNDet=np.sum(np.log(sigma_squared_array))





	


import pylab

# for i in range(10):
# np.random.seed(123+i)
sigma_complex = sigma/2**0.5
noise_real = np.random.normal(0,sigma_complex,s.size)
# noise = (np.random.normal(0,sigma_complex,s.size)+1j*np.random.normal(0,sigma_complex,s.size))
noise_real[bin_1_mask_vis_ordered] *= sigma_complex/noise_real[bin_1_mask_vis_ordered].std()
noise_real[bin_2_mask_vis_ordered] *= sigma_complex/noise_real[bin_2_mask_vis_ordered].std()

# np.random.seed(4321-15*i)
noise_imag = np.random.normal(0,sigma_complex,s.size)
# noise = (np.random.normal(0,sigma_complex,s.size)+1j*np.random.normal(0,sigma_complex,s.size))
noise_imag[bin_1_mask_vis_ordered] *= sigma_complex/noise_imag[bin_1_mask_vis_ordered].std()
noise_imag[bin_2_mask_vis_ordered] *= sigma_complex/noise_imag[bin_2_mask_vis_ordered].std()

noise = noise_real+1j*noise_imag

d=s+noise

Fz = np.dot(gridding_matrix_vis_ordered_to_chan_ordered, multi_vis_idft_array_1D)
Fprime = multi_chan_idft_array_noZMchan
Finv = multi_chan_dft_array_noZMchan

Fprime_Fz = np.dot(Fprime,Fz)

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

use_PB_matrix=False
if use_PB_matrix:
	PB_Fprime_Fz = np.dot(PB_matrix,Fprime_Fz)
	T = np.dot(Finv,PB_Fprime_Fz)
else:
	T = np.dot(Finv,Fprime_Fz)

Ninv_d = np.dot(Ninv,d)
dbar = np.dot(T.conjugate().T,Ninv_d)

Ninv_T = np.dot(Ninv,T)
T_Ninv_T = np.dot(T.conjugate().T,Ninv_T)

pylab.figure()
pylab.imshow(abs(T_Ninv_T))
pylab.colorbar()
if Show:pylab.show()

# Ninv_T = np.dot(Ninv,T.conjugate().T)
# T_Ninv_T = np.dot(T,Ninv_T)



# import pylab as P
# P.imshow(abs(T_Ninv_T)[:,:])
# # P.imshow(abs(T_Ninv_T)[0:10,0:10])
# P.colorbar()
# P.show()

print np.dot(Fz.conjugate().T, s)[bin_1_mask_vis_ordered].std()
print np.dot(Fz.conjugate().T, s)[bin_2_mask_vis_ordered].std()

print np.dot(Fz.conjugate().T, d)[bin_1_mask_vis_ordered].std()
print np.dot(Fz.conjugate().T, d)[bin_2_mask_vis_ordered].std()


#######################
###
# k-cube recovery testing
###
axes_tuple = (1,2)
uvf=numpy.fft.ifftshift(s_im+0j, axes=axes_tuple)
uvf=numpy.fft.fftn(uvf, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
uvf=numpy.fft.fftshift(uvf, axes=axes_tuple)

uveta2 = np.dot(Fz.conjugate().T, s.flatten())
axes_tuple = (0,1,2)
uveta=numpy.fft.ifftshift(s_im+0j, axes=axes_tuple)
uveta=numpy.fft.fftn(uveta, axes=axes_tuple) #FFT (python pre-normalises correctly! -- see parsevals theorem for discrete fourier transform.)
uveta=numpy.fft.fftshift(uveta, axes=axes_tuple)

print uveta.T.flatten()[0:10]-(uveta2[0:10]*(3*Fz_normalisation))
print k_cube_signal.T.flatten()[0:10] - uveta2[0:10]

#######################







Sigma_Diag_Indices=np.diag_indices(shape(T_Ninv_T)[0])
Npar = shape(T_Ninv_T)[0]

x=[1.0]


#######################
###
# matrix inversion testing
###
# PowerI=np.ones(Npar)/x[0]**2.
# PhiI=PowerI
# Sigma=T_Ninv_T.copy()
# Sigma[Sigma_Diag_Indices]+=PhiI

# Sigmacho = scipy.linalg.cho_factor(Sigma)
# # Sigmacho = scipy.linalg.cholesky(Sigma.astype(np.complex256), lower=True)
# SigmaI_dbar = scipy.linalg.cho_solve(Sigmacho, dbar)
# # SigmaI_dbar = scipy.linalg.cho_solve((Sigmacho,True), dbar)
# dbarSigmaIdbar=np.dot(dbar.conjugate().T,SigmaI_dbar)

# q,r = scipy.linalg.qr(Sigma)
# SigmaI_dbar = scipy.linalg.cho_solve((r,False), dbar)
# dbarSigmaIdbar=np.dot(dbar.conjugate().T,SigmaI_dbar)

# SigmaI = scipy.linalg.inv(Sigma)
# SigmaI_dbar2 = np.dot(SigmaI, dbar)
# dbarSigmaIdbar2=np.dot(dbar.conjugate().T,SigmaI_dbar2)

#######################


from numpy import real
def log_likelihood(x, fit_single_elems=False, T_Ninv_T=T_Ninv_T, dbar=dbar):
	# print 'Hello'
	phi = [0.0]*len(x)
	# print 'x',x
	# print 'x[0]',x[0]
	try:
		PowerI=np.ones(Npar)/k_cube_signal[bin_1_mask].var()*5.
		# PowerI[bin_1_mask_vis_ordered] = 1./x[0]**2.
		# PowerI[maxLx_elements<20] = 1./x[0]**2.
		# print 'i_bin', i_bin
		if fit_single_elems:
			PowerI[i_bin] = 1./x[0]**2.
		else:
			# PowerI=np.ones(Npar)/x[0]**2.
			PowerI[bin_2_mask_vis_ordered] = 1./x[0]**2.
		# PowerI[(0*9):(22*9)]=1.e20
		# PowerI[(23*9):(40*9)]=1.e20
		# PhiI=PowerI/(2**0.5)
		PhiI=PowerI+0j*PowerI
		Sigma=T_Ninv_T.copy()
		Sigma[Sigma_Diag_Indices]+=PhiI
		# Sigma[Sigma_Diag_Indices]+=PhiI-9.

		# Sigma[Sigma_Diag_Indices] += exact_powers-2.5-2.5*1j

		# Sigmacho = scipy.linalg.cholesky(Sigma, lower=True).astype(np.complex256)
		# SigmaI_dbar = scipy.linalg.cho_solve((Sigmacho,True), dbar)

		SigmaI = scipy.linalg.inv(Sigma)
		SigmaI_dbar = np.dot(SigmaI, dbar)
		dbarSigmaIdbar=np.dot(dbar.conjugate().T,SigmaI_dbar)

		logSigmaDet=np.linalg.slogdet(Sigma)[1]
		# logSigmaDet=2.*np.sum(np.log(np.diag(Sigmacho)))
		# logPhiDet=-1*np.sum(np.log(PhiI)) #Only possible because Phi is diagonal (otherwise would need to calc np.linalg.slogdet(Phi)). -1 factor is to get logPhiDet from logPhiIDet
		logPhiDet=-1*np.sum(np.log(PhiI)).real #Only possible because Phi is diagonal (otherwise would need to calc np.linalg.slogdet(Phi)). -1 factor is to get logPhiDet from logPhiIDet. Note: taking the real part of this calculation matches the solution given by np.linalg.slogdet(Phi))

		dbarSigmaIdbar=np.dot(dbar.conjugate().T,SigmaI_dbar)

		MargLogL =  -0.5*logSigmaDet -0.5*logPhiDet + 0.5*dbarSigmaIdbar
		vals = map(real, (-0.5*logSigmaDet, -0.5*logPhiDet, 0.5*dbarSigmaIdbar, MargLogL))
		# print vals[0], vals[1], vals[2], vals[3]
		MargLogL =  MargLogL.real
		return (MargLogL.squeeze())*1.0, phi
	except:
		return -np.inf, phi



# correct_answer = k_cube_signal[bin_1_mask_and_ZM].std()
correct_answer = k_cube_signal[bin_2_mask_and_ZM].std()
# correct_answer = np.dot(T,s).std()

from numpy import linspace
param_array=[]
loglike_array=[]
# for i in linspace(correct_answer*0.8,correct_answer*1.2,100):
for i in linspace(0,2.e1,2000):
	loglike,phi=log_likelihood([i])
	# print i, log_likelihood([i]),'\n'
	param_array.append(i)
	loglike_array.append(loglike)


maxLx=[]
argmax =  np.array(loglike_array)[np.logical_not(np.isinf(loglike_array))].argmax()
maxLx.append(np.array(param_array)[np.logical_not(np.isinf(loglike_array))][argmax])
print 'maxLx[-1], np.dot(T.conjugate().T,s).std(), sigma:'
print maxLx[-1], np.dot(T.conjugate().T,s).std(), sigma
print maxLx[-1], correct_answer

print np.dot(T.conjugate().T, s)[bin_2_mask_vis_ordered].std()







#######################
###
# PolyChord setup
Bin1 = [0.0, 100.0]
# Bin2 = [0.0, 100.0]

priors_min_max=[Bin1,]
# priors_min_max=[Bin1,Bin2]

class PriorC(object):
	def __init__(self, priors_min_max):
		self.priors_min_max=priors_min_max

	def prior_func(self, cube):
		pmm = self.priors_min_max
		theta = []
        	for i_p in range(len(cube)):
			theta_i = pmm[i_p][0]+((pmm[i_p][1]-pmm[i_p][0])*cube[i_p])
        		theta.append(theta_i)
        	return theta


prior_c = PriorC(priors_min_max)
nDerived = 0
nDims = 1

base_dir = 'chains'
if not os.path.isdir(base_dir):
    	os.makedirs(base_dir+'/clusters/')

file_root = 'test_v1d0-'

PolyChord.mpi_notification()
PolyChord.run_nested_sampling(log_likelihood, nDims, nDerived, file_root=file_root, read_resume=False, prior=prior_c.prior_func)


#######################


print np.dot(T.conjugate().T, s)[bin_2_mask_vis_ordered].std()












###
# Plot likelihood array
###
import pylab
fig,ax=pylab.subplots()
ax.errorbar(param_array, loglike_array)
if Show:fig.show()




###
# Calculate power spectrum of each visability spectrum individually
###
import time
Start = time.time()
from numpy import linspace
maxLx_elements=[]

for i_bin in arange(0,Npar,1):
# for i_bin in arange(0,912,1):
	i_bin=int(i_bin)
	print 'Time taken:', (time.time()-Start)
	print i_bin, 'of', Npar
	param_array=[]
	loglike_array=[]
	for i in linspace(0,2.e1,1000):
		loglike=log_likelihood([i], True)
		# print i, log_likelihood([i]),'\n'
		param_array.append(i)
		loglike_array.append(loglike[0])

	argmax =  np.array(loglike_array)[np.logical_not(np.isinf(loglike_array))].argmax()
	maxLx_elements.append(np.array(param_array)[np.logical_not(np.isinf(loglike_array))][argmax])

maxLx_elements = np.array(maxLx_elements)
print 'maxLx_elements', maxLx_elements
print 'np.mean(maxLx_elements)', np.mean(maxLx_elements)
print 'maxLx[-1], np.dot(T.conjugate().T,s).std(), sigma', maxLx[-1], np.dot(T.conjugate().T,s).std(), sigma
print 'np.mean(maxLx_elements**2.)**0.5', np.mean(maxLx_elements**2.)**0.5 #NOTE: for equal step sizes, this should be very close to maxLx[-1]. It is important to note that the mean Power (or in this case it's sqrt) that should be the same in both cases, not the mean of the standard deviations, since it is the power that is entering the likelihood via PhiI, not sigma!!! 
print 'maxLx[-1], np.dot(T.conjugate().T,s).std(), sigma', maxLx[-1], np.dot(T.conjugate().T,s).std(), sigma

print (maxLx_elements[bin_1_mask_vis_ordered]**2.).mean()**0.5
print (maxLx_elements[bin_2_mask_vis_ordered]**2.).mean()**0.5

print np.dot(Fz.conjugate().T, s)[bin_1_mask_vis_ordered].std()
print np.dot(Fz.conjugate().T, s)[bin_2_mask_vis_ordered].std()













