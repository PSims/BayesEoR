import time
import numpy as np
from numpy import shape
import scipy
from numpy import real
from pdb import set_trace as brk
import BayesEoR.Params.params as p


try:
	import pycuda.autoinit
	import pycuda.driver as cuda
	for devicenum in range(cuda.Device.count()):
	    device=cuda.Device(devicenum)
	    attrs=device.get_attributes()
	    print("\n===Attributes for device %d"%devicenum)
	    for (key,value) in attrs.iteritems():
	        print("%s:%s"%(str(key),str(value)))

	import time
	import numpy as np
	import ctypes
	from numpy import ctypeslib
	from scipy import linalg
	from subprocess import os

	# wrapmzpotrf=ctypes.CDLL('/users/psims/EoR/Applications/magma-2.3.0/example/new_wrappers/temp/wrapmzpotrf.so')
	current_dir = os.getcwd()
	base_dir = '/'.join(current_dir.split('/')[:np.where(np.array(current_dir.split('/'))=='BayesEoR')[0][-1]])+'/BayesEoR/'
	GPU_wrap_dir = base_dir+'likelihood_tests/SimpleEoRtestWQ/GPU_wrapper/'
	wrapmzpotrf=ctypes.CDLL(GPU_wrap_dir+'wrapmzpotrf.so')
	nrhs=1
	wrapmzpotrf.cpu_interface.argtypes = [ctypes.c_int, ctypes.c_int, ctypeslib.ndpointer(np.complex128, ndim=2, flags='C'), ctypeslib.ndpointer(np.complex128, ndim=1, flags='C'), ctypes.c_int, ctypeslib.ndpointer(np.int, ndim=1, flags='C')]
	print 'Computing on GPU'







except Exception as e:
	print 'Exception loading GPU encountered...'
	print e
	print 'Computing on CPU instead...'
	p.useGPU=False

#--------------------------------------------
# Define posterior
#--------------------------------------------

class PowerSpectrumPosteriorProbability(object):
	def __init__(self, T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, Ninv, d_Ninv_d, fit_single_elems=False, **kwargs):
		##===== Defaults =======
		default_diagonal_sigma = False
		default_block_T_Ninv_T=[]
		default_log_priors = False
		default_dimensionless_PS=False
		default_inverse_LW_power= 0.0
		default_inverse_LW_power_zeroth_LW_term= 0.0
		default_inverse_LW_power_first_LW_term= 0.0
		default_inverse_LW_power_second_LW_term= 0.0
		default_Print=False
		default_debug=False
		default_Print_debug=False
		default_intrinsic_noise_fitting=False
		
		##===== Inputs =======
		self.diagonal_sigma=kwargs.pop('diagonal_sigma',default_diagonal_sigma)
		self.block_T_Ninv_T=kwargs.pop('block_T_Ninv_T',default_block_T_Ninv_T)
		self.log_priors=kwargs.pop('log_priors',default_log_priors)
		if self.log_priors:print 'Using log-priors'
		self.dimensionless_PS=kwargs.pop('dimensionless_PS',default_dimensionless_PS)
		if self.dimensionless_PS:print 'Calculating dimensionless_PS'
		self.inverse_LW_power=kwargs.pop('inverse_LW_power',default_inverse_LW_power)
		self.inverse_LW_power_zeroth_LW_term=kwargs.pop('inverse_LW_power_zeroth_LW_term',default_inverse_LW_power_zeroth_LW_term)
		self.inverse_LW_power_first_LW_term=kwargs.pop('inverse_LW_power_first_LW_term',default_inverse_LW_power_first_LW_term)
		self.inverse_LW_power_second_LW_term=kwargs.pop('inverse_LW_power_second_LW_term',default_inverse_LW_power_second_LW_term)
		self.Print=kwargs.pop('Print',default_Print)
		self.debug=kwargs.pop('debug',default_debug)
		self.Print_debug=kwargs.pop('Print_debug',default_Print_debug)
		self.intrinsic_noise_fitting=kwargs.pop('intrinsic_noise_fitting',default_intrinsic_noise_fitting)

		self.fit_single_elems = fit_single_elems
		self.T_Ninv_T = T_Ninv_T
		self.dbar = dbar
		self.Sigma_Diag_Indices = Sigma_Diag_Indices
		self.diagonal_sigma = False
		self.block_diagonal_sigma = False
		self.instantiation_time = time.time()
		self.count = 0
		self.Npar = Npar
		self.k_cube_voxels_in_bin = k_cube_voxels_in_bin
		self.nuv = nuv
		self.nu = nu
		self.nv = nv
		self.nx = nx
		self.ny = ny
		self.neta = neta
		self.nf = nf
		self.nq = nq
		self.masked_power_spectral_modes = masked_power_spectral_modes
		self.modk_vis_ordered_list = modk_vis_ordered_list
		self.Ninv = Ninv
		self.d_Ninv_d = d_Ninv_d
		self.print_rate = 1000
		self.alpha_prime = 1.0


	def add_power_to_diagonals(self, T_Ninv_T_block, PhiI_block, **kwargs):
		return T_Ninv_T_block+np.diag(PhiI_block)

	def calc_Sigma_block_diagonals(self, T_Ninv_T, PhiI, **kwargs):
		PhiI_blocks = np.split(PhiI, self.nuv)
		Sigma_block_diagonals = np.array([self.add_power_to_diagonals(T_Ninv_T[(self.neta+self.nq)*i_block:(self.neta+self.nq)*(i_block+1),(self.neta+self.nq)*i_block:(self.neta+self.nq)*(i_block+1)], PhiI_blocks[i_block]) for i_block in range(self.nuv)])
		return Sigma_block_diagonals

	def calc_SigmaI_dbar_wrapper(self, x, T_Ninv_T, dbar, **kwargs):
		block_T_Ninv_T=[]

		##===== Inputs =======
		if 'block_T_Ninv_T' in kwargs:
			block_T_Ninv_T=kwargs['block_T_Ninv_T']
		
		start = time.time()
		PowerI = self.calc_PowerI(x)
		PhiI=PowerI
		if self.Print:print 'Time taken: {}'.format(time.time()-start)

		do_block_diagonal_inversion = len(shape(block_T_Ninv_T))>1
		if do_block_diagonal_inversion:
			if self.Print:print 'Using block-diagonal inversion'
			start = time.time()
			if self.intrinsic_noise_fitting:
				Sigma_block_diagonals = self.calc_Sigma_block_diagonals(T_Ninv_T/self.alpha_prime**2., PhiI) #This is only valid if the data is uniformly weighted
			else:
				Sigma_block_diagonals = self.calc_Sigma_block_diagonals(T_Ninv_T, PhiI)
			if self.Print:print 'Time taken: {}'.format(time.time()-start)
			if self.Print:print 'nuv', self.nuv

			start = time.time()
			dbar_blocks = np.split(dbar, self.nuv)
			if p.useGPU:
				if self.Print:print 'Computing block diagonal inversion on GPU'
				SigmaI_dbar_blocks_and_logdet_Sigma = np.array([self.calc_SigmaI_dbar(Sigma_block_diagonals[i_block], dbar_blocks[i_block], x_for_error_checking=x)  for i_block in range(self.nuv)])
				SigmaI_dbar_blocks = np.array([SigmaI_dbar_block for SigmaI_dbar_block, logdet_Sigma in SigmaI_dbar_blocks_and_logdet_Sigma])
				# SigmaI_dbar_blocks = SigmaI_dbar_blocks_and_logdet_Sigma[:,0] #This doesn't work because the array size is lost / flatten fails
				logdet_Sigma_blocks = SigmaI_dbar_blocks_and_logdet_Sigma[:,1]
			else:
				SigmaI_dbar_blocks = np.array([self.calc_SigmaI_dbar(Sigma_block_diagonals[i_block], dbar_blocks[i_block], x_for_error_checking=x)  for i_block in range(self.nuv)])
			if self.Print:print 'Time taken: {}'.format(time.time()-start)
			
			SigmaI_dbar = SigmaI_dbar_blocks.flatten()
			dbarSigmaIdbar=np.dot(dbar.conjugate().T,SigmaI_dbar)
			if self.Print:print 'Time taken: {}'.format(time.time()-start)

			if p.useGPU:
				logSigmaDet = np.sum(logdet_Sigma_blocks)
			else:
				logSigmaDet=np.sum([np.linalg.slogdet(Sigma_block)[1] for Sigma_block in Sigma_block_diagonals])
				if self.Print:print 'Time taken: {}'.format(time.time()-start)

		else:
			if self.count%self.print_rate==0:print 'Not using block-diagonal inversion'
			start = time.time()
			###
			# Note: the following two lines can probably be speeded up by adding T_Ninv_T and np.diag(PhiI). (Should test this!) but this else statement only occurs on the GPU inversion so will deal with it later.
			###
			Sigma=T_Ninv_T.copy()
			if self.intrinsic_noise_fitting:
				Sigma = Sigma/self.alpha_prime**2.0

			Sigma[self.Sigma_Diag_Indices]+=PhiI
			if self.Print:print 'Time taken: {}'.format(time.time()-start)

			start = time.time()
			if p.useGPU:
				if self.Print:print 'Computing matrix inversion on GPU'
				SigmaI_dbar_and_logdet_Sigma = self.calc_SigmaI_dbar(Sigma, dbar, x_for_error_checking=x)
				SigmaI_dbar = SigmaI_dbar_and_logdet_Sigma[0]
				logdet_Sigma = SigmaI_dbar_and_logdet_Sigma[1]
			else:
				SigmaI_dbar = self.calc_SigmaI_dbar(Sigma, dbar, x_for_error_checking=x)
			if self.Print:print 'Time taken: {}'.format(time.time()-start)

			dbarSigmaIdbar=np.dot(dbar.conjugate().T,SigmaI_dbar)
			if self.Print:print 'Time taken: {}'.format(time.time()-start)

			if p.useGPU:
				logSigmaDet = logdet_Sigma
			else:
				logSigmaDet=np.linalg.slogdet(Sigma)[1]
			if self.Print:print 'Time taken: {}'.format(time.time()-start)
			# logSigmaDet=2.*np.sum(np.log(np.diag(Sigmacho)))

		return SigmaI_dbar, dbarSigmaIdbar, PhiI, logSigmaDet

	def calc_SigmaI_dbar(self, Sigma, dbar, **kwargs):
		##===== Defaults =======
		block_T_Ninv_T=[]
		default_x_for_error_checking = "Params haven't been recorded... use x_for_error_checking kwarg when calling calc_SigmaI_dbar to change this."

		##===== Inputs =======
		if 'block_T_Ninv_T' in kwargs:
			block_T_Ninv_T=kwargs['block_T_Ninv_T']
		if 'x_for_error_checking' in kwargs:
			x_for_error_checking=kwargs['x_for_error_checking']
		
		if not p.useGPU:
			# Sigmacho = scipy.linalg.cholesky(Sigma, lower=True).astype(np.complex256)
			# SigmaI_dbar = scipy.linalg.cho_solve((Sigmacho,True), dbar)
			SigmaI = scipy.linalg.inv(Sigma)
			SigmaI_dbar = np.dot(SigmaI, dbar)
			return SigmaI_dbar

		else:
			# brk()
			dbar_copy = dbar.copy()
			dbar_copy_copy = dbar.copy()
			self.GPU_error_flag = np.array([0])
			# wrapmzpotrf.cpu_interface(len(Sigma), nrhs, Sigma, dbar_copy, 1, self.GPU_error_flag) #to print debug
			wrapmzpotrf.cpu_interface(len(Sigma), nrhs, Sigma, dbar_copy, 0, self.GPU_error_flag)
			logdet_Magma_Sigma = np.sum(np.log(np.diag(abs(Sigma))))*2 #Note: After wrapmzpotrf, Sigma is actually SigmaCho (i.e. L with SigmaLL^T)
			# print logdet_Magma_Sigma
			SigmaI_dbar = linalg.cho_solve((Sigma.conjugate().T,True), dbar_copy_copy)
			if self.GPU_error_flag[0] != 0:
				logdet_Magma_Sigma=+np.inf #If the inversion doesn't work, zero-weight the sample (may want to stop computing if this occurs?)
				print 'GPU inversion error. Setting sample posterior probability to zero.'
				print 'Param values: ', x_for_error_checking
			return SigmaI_dbar, logdet_Magma_Sigma
			



		
	def calc_dimensionless_power_spectral_normalisation_ltl(self, i_bin, **kwargs):
		EoRVolume = 770937185.063917
		Omega_map = ((12*np.pi/180.)**2)
		dimensionless_PS_scaling = (EoRVolume*self.modk_vis_ordered_list[i_bin]**3.)/(Omega_map**4 * (self.nu*self.nv*self.nf)**4)
		return dimensionless_PS_scaling

	def calc_dimensionless_power_spectral_normalisation_21cmFAST(self, i_bin, **kwargs):
		###
		# NOTE: the physical size of the cosmological box is simulation dependent. The values here are matched to the following 21cmFAST simulation:
		# /users/psims/EoR/EoR_simulations/21cmFAST_512MPc_512pix_128pix/ and will require updating if the input signal is changed to a new source
		# 21cmFAST normalisation:
		#define BOX_LEN (float) 512 // in Mpc
		#define VOLUME (BOX_LEN*BOX_LEN*BOX_LEN) // in Mpc^3
		# p_box[ct] += pow(k_mag,3)*pow(cabs(deldel_T[HII_C_INDEX(n_x, n_y, n_z)]), 2)/(2.0*PI*PI*VOLUME);
		###
		EoR_x_full_pix = float(p.box_size_21cmFAST_pix_sc) #pix (defined by input to 21cmFAST simulation)
		EoR_y_full_pix = float(p.box_size_21cmFAST_pix_sc) #pix (defined by input to 21cmFAST simulation)
		EoR_z_full_pix = float(p.box_size_21cmFAST_pix_sc) #pix (defined by input to 21cmFAST simulation)
		EoR_x_full_Mpc = float(p.box_size_21cmFAST_Mpc_sc) #Mpc (defined by input to 21cmFAST simulation)
		EoR_y_full_Mpc = float(p.box_size_21cmFAST_Mpc_sc) #Mpc (defined by input to 21cmFAST simulation)
		EoR_z_full_Mpc = float(p.box_size_21cmFAST_Mpc_sc) #Mpc (defined by input to 21cmFAST simulation)
		# EoR_analysis_cube_x_pix = EoR_x_full_pix #Mpc Analysing the full FoV in x
		# EoR_analysis_cube_y_pix = EoR_y_full_pix #Mpc Analysing the full FoV in y
		# EoR_analysis_cube_z_pix = 38 #Mpc Analysing 38 of the 128 channels of the full EoR_simulations 
		EoR_analysis_cube_x_pix = float(p.EoR_analysis_cube_x_pix) #pix Analysing the full FoV in x
		EoR_analysis_cube_y_pix = float(p.EoR_analysis_cube_y_pix) #pix Analysing the full FoV in y
		EoR_analysis_cube_z_pix = self.nf #Mpc Analysing 38 of the 128 channels of the full EoR_simulations 
		EoR_analysis_cube_x_Mpc = float(p.EoR_analysis_cube_x_Mpc) #Mpc Analysing the full FoV in x
		EoR_analysis_cube_y_Mpc = float(p.EoR_analysis_cube_y_Mpc) #Mpc Analysing the full FoV in y
		EoR_analysis_cube_z_Mpc = EoR_z_full_Mpc*(float(EoR_analysis_cube_z_pix)/EoR_z_full_pix) #Mpc Analysing 38 of the 128 channels of the full simulation
		EoRVolume = EoR_analysis_cube_x_Mpc*EoR_analysis_cube_y_Mpc*EoR_analysis_cube_z_Mpc
		pixel_volume = EoR_analysis_cube_x_pix*EoR_analysis_cube_y_pix*EoR_analysis_cube_z_pix
		cosmo_fft_norm_factor = (2.*np.pi)**2. #This needs to be verified / replaced........!
		# dimensionless_PS_scaling = (self.modk_vis_ordered_list[i_bin]**3.)*(EoRVolume**1.0)/(2.*(np.pi**2)*pixel_volume**2.)*cosmo_fft_norm_factor
		dimensionless_PS_scaling = (self.modk_vis_ordered_list[i_bin]**3.)*(EoRVolume**1.0)/(2.*(np.pi**2)*pixel_volume**1.)
		# dimensionless_PS_scaling = (self.modk_vis_ordered_list[i_bin].mean()**3.)*(EoRVolume**1.0)/(2.*(np.pi**2)*pixel_volume**1.)
		if p.include_instrumental_effects:
			# e.g. http://www.mrao.cam.ac.uk/~kjbg1/lectures/lect1_1.pdf
			Omega_beam_Gaussian_sr = (p.FWHM_deg_at_ref_freq_MHz*np.pi/180.)**2. * np.pi/(4.*np.log(2.0))
			dimensionless_PS_scaling = dimensionless_PS_scaling * Omega_beam_Gaussian_sr**4.0

		return dimensionless_PS_scaling
	def calc_dimensionless_power_spectral_normalisation_21cmFAST_v2d0(self, i_bin, **kwargs):
		###
		# NOTE: the physical size of the cosmological box is simulation dependent. The values here are matched to the following 21cmFAST simulation:
		# /users/psims/EoR/EoR_simulations/21cmFAST_512MPc_512pix_128pix/ and will require updating if the input signal is changed to a new source
		# 21cmFAST normalisation:
		#define BOX_LEN (float) 512 // in Mpc
		#define VOLUME (BOX_LEN*BOX_LEN*BOX_LEN) // in Mpc^3
		# p_box[ct] += pow(k_mag,3)*pow(cabs(deldel_T[HII_C_INDEX(n_x, n_y, n_z)]), 2)/(2.0*PI*PI*VOLUME);
		###
		EoR_x_full_pix = float(p.box_size_21cmFAST_pix_sc) #pix (defined by input to 21cmFAST simulation)
		EoR_y_full_pix = float(p.box_size_21cmFAST_pix_sc) #pix (defined by input to 21cmFAST simulation)
		EoR_z_full_pix = float(p.box_size_21cmFAST_pix_sc) #pix (defined by input to 21cmFAST simulation)
		EoR_x_full_Mpc = float(p.box_size_21cmFAST_Mpc_sc) #Mpc (defined by input to 21cmFAST simulation)
		EoR_y_full_Mpc = float(p.box_size_21cmFAST_Mpc_sc) #Mpc (defined by input to 21cmFAST simulation)
		EoR_z_full_Mpc = float(p.box_size_21cmFAST_Mpc_sc) #Mpc (defined by input to 21cmFAST simulation)
		# EoR_analysis_cube_x_pix = EoR_x_full_pix #Mpc Analysing the full FoV in x
		# EoR_analysis_cube_y_pix = EoR_y_full_pix #Mpc Analysing the full FoV in y
		# EoR_analysis_cube_z_pix = 38 #Mpc Analysing 38 of the 128 channels of the full EoR_simulations 
		EoR_analysis_cube_x_pix = float(p.EoR_analysis_cube_x_pix) #pix Analysing the full FoV in x
		EoR_analysis_cube_y_pix = float(p.EoR_analysis_cube_y_pix) #pix Analysing the full FoV in y
		EoR_analysis_cube_z_pix = self.nf #Mpc Analysing 38 of the 128 channels of the full EoR_simulations 
		EoR_analysis_cube_x_Mpc = float(p.EoR_analysis_cube_x_Mpc) #Mpc Analysing the full FoV in x
		EoR_analysis_cube_y_Mpc = float(p.EoR_analysis_cube_y_Mpc) #Mpc Analysing the full FoV in y
		EoR_analysis_cube_z_Mpc = EoR_z_full_Mpc*(float(EoR_analysis_cube_z_pix)/EoR_z_full_pix) #Mpc Analysing 38 of the 128 channels of the full simulation
		EoRVolume = EoR_analysis_cube_x_Mpc*EoR_analysis_cube_y_Mpc*EoR_analysis_cube_z_Mpc
		pixel_volume = EoR_analysis_cube_x_pix*EoR_analysis_cube_y_pix*EoR_analysis_cube_z_pix
		cosmo_fft_norm_factor = (2.*np.pi)**2. #This needs to be verified / replaced........!

		# dimensionless_PS_scaling = (self.modk_vis_ordered_list[i_bin]**3.)*(EoRVolume**1.0)/(2.*(np.pi**2)*pixel_volume**1.)

		###
		# 21cmFast normalisation code
		###
		# // do the FFTs
		# plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)deltax, (fftwf_complex *)deltax, FFTW_ESTIMATE);
		# fftwf_execute(plan);
		# fftwf_destroy_plan(plan);
		# fftwf_cleanup();
		# for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++){
		#    deltax[ct] *= VOLUME/(HII_TOT_NUM_PIXELS+0.0);
		# }
		# p_box[ct] +=  pow(k_mag,3)*pow(cabs(deltax[HII_C_INDEX(n_x, n_y, n_z)]), 2) / (2.0*PI*PI*VOLUME);

		dimensionless_PS_scaling = (self.modk_vis_ordered_list[i_bin]**3.)*(EoRVolume/pixel_volume)**2./(2.*(np.pi**2)*EoRVolume)

		if p.include_instrumental_effects:
			# e.g. http://www.mrao.cam.ac.uk/~kjbg1/lectures/lect1_1.pdf
			Omega_beam_Gaussian_sr = (p.FWHM_deg_at_ref_freq_MHz*np.pi/180.)**2. * np.pi/(4.*np.log(2.0))
			dimensionless_PS_scaling = dimensionless_PS_scaling * Omega_beam_Gaussian_sr**4.0

		return dimensionless_PS_scaling
		
	def calc_dimensionless_power_spectral_normalisation_21cmFAST_v3d0(self, i_bin, **kwargs):
		###
		# NOTE: the physical size of the cosmological box is simulation dependent.
		# 21cmFAST normalisation:
		#define BOX_LEN (float) 512 // in Mpc
		#define VOLUME (BOX_LEN*BOX_LEN*BOX_LEN) // in Mpc^3
		# p_box[ct] += pow(k_mag,3)*pow(cabs(deldel_T[HII_C_INDEX(n_x, n_y, n_z)]), 2)/(2.0*PI*PI*VOLUME);
		###
		EoR_x_full_pix = float(p.box_size_21cmFAST_pix_sc) #pix (defined by input to 21cmFAST simulation)
		EoR_y_full_pix = float(p.box_size_21cmFAST_pix_sc) #pix (defined by input to 21cmFAST simulation)
		EoR_z_full_pix = float(p.box_size_21cmFAST_pix_sc) #pix (defined by input to 21cmFAST simulation)
		EoR_x_full_Mpc = float(p.box_size_21cmFAST_Mpc_sc) #Mpc (defined by input to 21cmFAST simulation)
		EoR_y_full_Mpc = float(p.box_size_21cmFAST_Mpc_sc) #Mpc (defined by input to 21cmFAST simulation)
		EoR_z_full_Mpc = float(p.box_size_21cmFAST_Mpc_sc) #Mpc (defined by input to 21cmFAST simulation)

		VOLUME = p.box_size_21cmFAST_Mpc_sc**3. #Full cube volume in Mpc^3
		HII_TOT_NUM_PIXELS = p.box_size_21cmFAST_pix_sc**3. #Full cube Npix
		amplitude_normalisation_21cmFast = (VOLUME/HII_TOT_NUM_PIXELS)
		explicit_21cmFast_power_spectrum_normalisation = 1./(2.0*np.pi**2.*VOLUME)
		full_21cmFast_power_spectrum_normalisation = VOLUME / (2.0*np.pi**2.*HII_TOT_NUM_PIXELS**2.)#amplitude_normalisation_21cmFast**2. * explicit_power_spectrum_normalisation = (VOLUME/HII_TOT_NUM_PIXELS)**2. / (2.0*np.pi**2.*VOLUME)

		############
		# subset_power_spectrum_normalisation:
		# 1. Image space full cube -> k-space subset cube,
		# values are (nf/512.)**0.5 times smaller than in 21cmFast.
		# Thus, to normalise, the amplitude spectrum should be scaled by (512./nf)**0.5 (subset cube component)
		# 
		# 2. k-space subset cube -> image space subset cube -> data = np.dot(Finv, image space subset cube),
		# values are nf**0.5 times larger than in np.dot(T, k-space subset cube) -> data.
		#[This is because in the k-space subset cube -> image space subset cube step (1) with a numpy ifft there is an effective division by nf**0.5 where as in the equivalent k-space subset cube -> image space subset cube component of T there is a division by nf**1.0, thus the values in np.dot(T, k-space subset cube) -> data end up being nf**0.5 times smaller.]
		# Thus, to normalise, the amplitude spectrum should be scaled by 1./nf**0.5  (matrix encoding component)
		# 
		# Thus, the overall subset + matrix encoding amplitude normalisation is: (512./nf)**0.5/nf**0.5 = 512**0.5
		############
		subset_power_spectrum_normalisation = float(p.box_size_21cmFAST_pix_sc) #See /home/peter/OSCAR_mnt/rdata/EoR/Python_Scripts/BayesEoR/git_version/BayesEoR/spec_model_tests/random/normalisation_testing/dimensionless_power_spectrum_21cmFast_comparison_normalisation_v2d0.py

		full_power_spectrum_normalisation = subset_power_spectrum_normalisation * full_21cmFast_power_spectrum_normalisation
		dimensionless_PS_scaling = (self.modk_vis_ordered_list[i_bin]**3.)*full_power_spectrum_normalisation

		# if p.include_instrumental_effects:
		# 	# e.g. http://www.mrao.cam.ac.uk/~kjbg1/lectures/lect1_1.pdf
		# 	Omega_beam_Gaussian_sr = (p.FWHM_deg_at_ref_freq_MHz*np.pi/180.)**2. * np.pi/(4.*np.log(2.0))
		# 	dimensionless_PS_scaling = dimensionless_PS_scaling * Omega_beam_Gaussian_sr**4.0

		return dimensionless_PS_scaling

	def calc_Npix_physical_power_spectrum_normalisation(self, i_bin, **kwargs):
		###
		# 21cmFAST normalisation:
		#define BOX_LEN (float) 512 // in Mpc
		#define VOLUME (BOX_LEN*BOX_LEN*BOX_LEN) // in Mpc^3
		# p_box[ct] += pow(k_mag,3)*pow(cabs(deldel_T[HII_C_INDEX(n_x, n_y, n_z)]), 2)/(2.0*PI*PI*VOLUME);
		###
		EoR_x_full_pix = float(p.box_size_21cmFAST_pix) #pix (defined by input to 21cmFAST simulation)
		EoR_y_full_pix = float(p.box_size_21cmFAST_pix) #pix (defined by input to 21cmFAST simulation)
		EoR_z_full_pix = float(p.box_size_21cmFAST_pix) #pix (defined by input to 21cmFAST simulation)
		EoR_x_full_Mpc = float(p.box_size_21cmFAST_Mpc) #Mpc (defined by input to 21cmFAST simulation)
		EoR_y_full_Mpc = float(p.box_size_21cmFAST_Mpc) #Mpc (defined by input to 21cmFAST simulation)
		EoR_z_full_Mpc = float(p.box_size_21cmFAST_Mpc) #Mpc (defined by input to 21cmFAST simulation)
		# EoR_analysis_cube_x_pix = EoR_x_full_pix #Mpc Analysing the full FoV in x
		# EoR_analysis_cube_y_pix = EoR_y_full_pix #Mpc Analysing the full FoV in y
		# EoR_analysis_cube_z_pix = 38 #Mpc Analysing 38 of the 128 channels of the full EoR_simulations 
		EoR_analysis_cube_x_pix = float(p.EoR_analysis_cube_x_pix) #pix Analysing the full FoV in x
		EoR_analysis_cube_y_pix = float(p.EoR_analysis_cube_y_pix) #pix Analysing the full FoV in y
		EoR_analysis_cube_z_pix = self.nf #Mpc Analysing 38 of the 128 channels of the full EoR_simulations 
		EoR_analysis_cube_x_Mpc = float(p.EoR_analysis_cube_x_Mpc) #Mpc Analysing the full FoV in x
		EoR_analysis_cube_y_Mpc = float(p.EoR_analysis_cube_y_Mpc) #Mpc Analysing the full FoV in y
		EoR_analysis_cube_z_Mpc = EoR_z_full_Mpc*(float(EoR_analysis_cube_z_pix)/EoR_z_full_pix) #Mpc Analysing 38 of the 128 channels of the full simulation
		EoRVolume = EoR_analysis_cube_x_Mpc*EoR_analysis_cube_y_Mpc*EoR_analysis_cube_z_Mpc
		pixel_volume = EoR_analysis_cube_x_pix*EoR_analysis_cube_y_pix*EoR_analysis_cube_z_pix
		cosmo_fft_norm_factor = (2.*np.pi)**2. #This needs to be verified / replaced........!
		PS_scaling = (EoRVolume**1.0)/(pixel_volume**1.0)

		random_scaling_factor = (9.0*np.pi/180.)**2. * np.pi/(4.*np.log(2.0))
		# PS_scaling = PS_scaling*0.3**3.*random_scaling_factor**4. #Random scaling factor to put the physical and dimensionless power spectrum on a comparable scale (to avoid having to use different priors for the two cases).
		PS_scaling = PS_scaling*0.1**3.*random_scaling_factor**4. #Random scaling factor to put the physical and dimensionless power spectrum on a comparable scale (to avoid having to use different priors for the two cases).

		return PS_scaling

	def calc_PowerI(self, x, **kwargs):
		###
		# Place restricions on the power in the long spectral scale model either for,
		# inverse_LW_power: constrain the amplitude distribution of all of the large spectral scale model components
		# inverse_LW_power_zeroth_LW_term: constrain the amplitude of monopole-term basis vector
		# inverse_LW_power_first_LW_term: constrain the amplitude of the model components of the 1st LW basis vector (e.g. linear model comp.)
		# inverse_LW_power_second_LW_term: constrain the amplitude of the model components of the 2nd LW basis vector (e.g. quad model comp.)
		# Note: The indices used are correct for the current ordering of basis vectors when nf is an even number...
		###
		
		PowerI=np.zeros(self.Npar)

		if p.include_instrumental_effects:
			q0_index = self.neta/2
		else:
			q0_index = self.nf/2-1
		q1_index = self.neta
		q2_index = self.neta+1

		# Constrain LW mode amplitude distribution
		dimensionless_PS_scaling = self.calc_dimensionless_power_spectral_normalisation_21cmFAST_v3d0(0)
		if p.use_LWM_Gaussian_prior:
			Fourier_mode_start_index = 3
			# PowerI[self.nf/2-1::self.nf]=np.mean(dimensionless_PS_scaling)/x[0] #set to zero for a uniform distribution
			# PowerI[self.nf-2::self.nf]=np.mean(dimensionless_PS_scaling)/x[1] #set to zero for a uniform distribution
			# PowerI[self.nf-1::self.nf]=np.mean(dimensionless_PS_scaling)/x[2] #set to zero for a uniform distribution
			PowerI[q0_index::(self.neta+self.nq)]=np.mean(dimensionless_PS_scaling)/x[0] #set to zero for a uniform distribution
			PowerI[q1_index::(self.neta+self.nq)]=np.mean(dimensionless_PS_scaling)/x[1] #set to zero for a uniform distribution
			PowerI[q2_index::(self.neta+self.nq)]=np.mean(dimensionless_PS_scaling)/x[2] #set to zero for a uniform distribution
		else:
			Fourier_mode_start_index = 0
			PowerI[q0_index::(self.neta+self.nq)]=self.inverse_LW_power #set to zero for a uniform distribution
			PowerI[q1_index::(self.neta+self.nq)]=self.inverse_LW_power #set to zero for a uniform distribution
			PowerI[q2_index::(self.neta+self.nq)]=self.inverse_LW_power #set to zero for a uniform distribution
			if self.inverse_LW_power==0.0:
				PowerI[q0_index::(self.neta+self.nq)]=self.inverse_LW_power_zeroth_LW_term #set to zero for a uniform distribution
				PowerI[q1_index::(self.neta+self.nq)]=self.inverse_LW_power_first_LW_term #set to zero for a uniform distribution
				PowerI[q2_index::(self.neta+self.nq)]=self.inverse_LW_power_second_LW_term #set to zero for a uniform distribution

		if self.dimensionless_PS:
			self.power_spectrum_normalisation_func = self.calc_dimensionless_power_spectral_normalisation_21cmFAST_v3d0
		else:
			self.power_spectrum_normalisation_func = self.calc_Npix_physical_power_spectrum_normalisation
			
		# Fit for Fourier mode power spectrum
		for i_bin in range(len(self.k_cube_voxels_in_bin)):
			power_spectrum_normalisation = self.power_spectrum_normalisation_func(i_bin)
			PowerI[self.k_cube_voxels_in_bin[i_bin]] = power_spectrum_normalisation/x[Fourier_mode_start_index+i_bin] #NOTE: fitting for power not std here

		return PowerI

	def posterior_probability(self, x, **kwargs):
		if self.debug:brk()

		##===== Defaults =======
		block_T_Ninv_T=self.block_T_Ninv_T
		fit_single_elems = self.fit_single_elems
		T_Ninv_T = self.T_Ninv_T
		dbar = self.dbar

		##===== Inputs =======
		if 'block_T_Ninv_T' in kwargs:
			block_T_Ninv_T=kwargs['block_T_Ninv_T']

		if self.intrinsic_noise_fitting:
			self.alpha_prime = x[0]
			x = x[1:]
			Ndat = len(np.diagonal(self.Ninv))
			log_det_N = Ndat*np.log(self.alpha_prime**2.0) #This is only valid if the data is uniformly weighted
			d_Ninv_d = self.d_Ninv_d/(self.alpha_prime**2.0) #This is only valid if the data is uniformly weighted
			dbar = dbar/(self.alpha_prime**2.0) #This is only valid if the data is uniformly weighted

		if self.log_priors:
			# print 'Using log-priors'
			x = 10.**np.array(x)

		# brk()
		phi = [0.0]
		do_block_diagonal_inversion = len(shape(block_T_Ninv_T))>1
		self.count+=1
		start = time.time()
		try:
			if do_block_diagonal_inversion:
				if self.Print:print 'Using block-diagonal inversion'
				SigmaI_dbar, dbarSigmaIdbar, PhiI, logSigmaDet = self.calc_SigmaI_dbar_wrapper(x, T_Ninv_T, dbar, block_T_Ninv_T=block_T_Ninv_T)
			else:
				if self.Print:print 'Not using block-diagonal inversion'
				SigmaI_dbar, dbarSigmaIdbar, PhiI, logSigmaDet = self.calc_SigmaI_dbar_wrapper(x, T_Ninv_T, dbar)

			#logPhiDet=-1*np.sum(np.log(PhiI[np.logical_not(self.masked_power_spectral_modes)])).real #Only possible because Phi is diagonal (otherwise would need to calc np.linalg.slogdet(Phi)). -1 factor is to get logPhiDet from logPhiIDet. Note: the real part of this calculation matches the solution given by np.linalg.slogdet(Phi))
			logPhiDet=-1*np.sum(np.log(PhiI)).real #Only possible because Phi is diagonal (otherwise would need to calc np.linalg.slogdet(Phi)). -1 factor is to get logPhiDet from logPhiIDet. Note: the real part of this calculation matches the solution given by np.linalg.slogdet(Phi))

			MargLogL =  -0.5*logSigmaDet -0.5*logPhiDet + 0.5*dbarSigmaIdbar
			vals = map(real, (-0.5*logSigmaDet, -0.5*logPhiDet, 0.5*dbarSigmaIdbar, MargLogL))
			if self.intrinsic_noise_fitting:
				MargLogL = MargLogL -0.5*d_Ninv_d -0.5*log_det_N
			MargLogL =  MargLogL.real
			if self.Print_debug:
				MargLogL_equation_string = 'MargLogL =  -0.5*logSigmaDet -0.5*logPhiDet + 0.5*dbarSigmaIdbar'
				if self.intrinsic_noise_fitting:
					print 'Using intrinsic noise fitting'
					MargLogL_equation_string+=' - 0.5*d_Ninv_d -0.5*log_det_N'
					print 'logSigmaDet, logPhiDet, dbarSigmaIdbar, d_Ninv_d, log_det_N', logSigmaDet, logPhiDet, dbarSigmaIdbar, d_Ninv_d, log_det_N
				else:
					print 'logSigmaDet, logPhiDet, dbarSigmaIdbar', logSigmaDet, logPhiDet, dbarSigmaIdbar
				print MargLogL_equation_string, MargLogL
				print 'MargLogL.real', MargLogL.real

			# brk()
			
			if self.nu>10:
				self.print_rate=100
			if self.count%self.print_rate==0:
				print 'count', self.count
				print 'Time since class instantiation: %f'%(time.time()-self.instantiation_time)
				print 'Time for this likelihood call: %f'%(time.time()-start)
			return (MargLogL.squeeze())*1.0, phi
		except Exception as e:
			print 'Exception encountered...'
			print e
			return -np.inf, phi


# 
# 





