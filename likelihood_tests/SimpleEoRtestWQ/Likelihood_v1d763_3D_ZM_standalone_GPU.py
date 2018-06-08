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
	wrapmzpotrf.cpu_interface.argtypes = [ctypes.c_int, ctypes.c_int, ctypeslib.ndpointer(np.complex128, ndim=2, flags='C'), ctypeslib.ndpointer(np.complex128, ndim=1, flags='C'), ctypes.c_int]
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
	def __init__(self, T_Ninv_T, dbar, Sigma_Diag_Indices, Npar, k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq, masked_power_spectral_modes, modk_vis_ordered_list, fit_single_elems=False, **kwargs):
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

		self.fit_single_elems      = fit_single_elems
		self.T_Ninv_T              = T_Ninv_T
		self.dbar                  = dbar
		self.Sigma_Diag_Indices    = Sigma_Diag_Indices
		self.diagonal_sigma        = False
		self.block_diagonal_sigma  = False
		self.instantiation_time    = time.time()
		self.count                 = 0
		self.Npar                  = Npar
		self.k_cube_voxels_in_bin  = k_cube_voxels_in_bin
		self.nuv  = nuv
		self.nu  = nu
		self.nv  = nv
		self.nx  = nx
		self.ny  = ny
		self.neta  = neta
		self.nf  = nf
		self.nq  = nq
		self.masked_power_spectral_modes  = masked_power_spectral_modes
		self.modk_vis_ordered_list = modk_vis_ordered_list

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
			Sigma_block_diagonals = self.calc_Sigma_block_diagonals(T_Ninv_T, PhiI)
			if self.Print:print 'Time taken: {}'.format(time.time()-start)
			if self.Print:print 'nuv', self.nuv

			start = time.time()
			dbar_blocks = np.split(dbar, self.nuv)
			if p.useGPU:
				SigmaI_dbar_blocks_and_logdet_Sigma = np.array([self.calc_SigmaI_dbar(Sigma_block_diagonals[i_block], dbar_blocks[i_block])  for i_block in range(self.nuv)])
				SigmaI_dbar_blocks = np.array([SigmaI_dbar_block for SigmaI_dbar_block, logdet_Sigma in SigmaI_dbar_blocks_and_logdet_Sigma])
				# SigmaI_dbar_blocks = SigmaI_dbar_blocks_and_logdet_Sigma[:,0] #This doesn't work because the array size is lost / flatten fails
				logdet_Sigma_blocks = SigmaI_dbar_blocks_and_logdet_Sigma[:,1]
			else:
				SigmaI_dbar_blocks = np.array([self.calc_SigmaI_dbar(Sigma_block_diagonals[i_block], dbar_blocks[i_block])  for i_block in range(self.nuv)])
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
			print 'Not using block-diagonal inversion'
			start = time.time()
			###
			# Note: the following two lines can probably be speeded up by adding T_Ninv_T and np.diag(PhiI). (Should test this!) but this else statement only occurs on the GPU inversion so will deal with it later.
			###
			Sigma=T_Ninv_T.copy()
			Sigma[self.Sigma_Diag_Indices]+=PhiI
			if self.Print:print 'Time taken: {}'.format(time.time()-start)

			start = time.time()
			if p.useGPU:
				SigmaI_dbar_and_logdet_Sigma = self.calc_SigmaI_dbar(Sigma, dbar)
				SigmaI_dbar = SigmaI_dbar_and_logdet_Sigma[0]
				logdet_Sigma = SigmaI_dbar_and_logdet_Sigma[1]
			else:
				SigmaI_dbar = self.calc_SigmaI_dbar(Sigma, dbar)
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

		##===== Inputs =======
		if 'block_T_Ninv_T' in kwargs:
			block_T_Ninv_T=kwargs['block_T_Ninv_T']
		
		if not p.useGPU:
			# Sigmacho = scipy.linalg.cholesky(Sigma, lower=True).astype(np.complex256)
			# SigmaI_dbar = scipy.linalg.cho_solve((Sigmacho,True), dbar)
			SigmaI = scipy.linalg.inv(Sigma)
			SigmaI_dbar = np.dot(SigmaI, dbar)
			return SigmaI_dbar

		else:
			dbar_copy = dbar.copy()
			dbar_copy_copy = dbar.copy()
			# wrapmzpotrf.cpu_interface(len(Sigma), nrhs, Sigma, dbar_copy, 1) #to print debug
			wrapmzpotrf.cpu_interface(len(Sigma), nrhs, Sigma, dbar_copy, 0)
			logdet_Magma_Sigma = np.sum(np.log(np.diag(abs(Sigma))))*2 #Note: After wrapmzpotrf, Sigma is actually SigmaCho (i.e. L with SigmaLL^T)
			# print logdet_Magma_Sigma
			SigmaI_dbar = linalg.cho_solve((Sigma.conjugate().T,True), dbar_copy_copy)
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
		# dimensionless_PS_scaling = (self.modk_vis_ordered_list[i_bin]**3.)*(EoRVolume**1.0)/(2.*(np.pi**2)*pixel_volume**2.)*cosmo_fft_norm_factor
		dimensionless_PS_scaling = (self.modk_vis_ordered_list[i_bin]**3.)*(EoRVolume**1.0)/(2.*(np.pi**2)*pixel_volume**1.)

		return dimensionless_PS_scaling

	def calc_PowerI(self, x, **kwargs):
		###
		# Place restricions on the power in the long spectral scale model either for,
		# inverse_LW_power: constrain the amplitude of all of the long spectral scale basis model components
		# inverse_LW_power_zeroth_LW_term: constrain the amplitude of monopole-term basis vector
		# inverse_LW_power_first_LW_term: constrain the amplitude of the model components of the 1st LW basis vector (e.g. linear model comp.)
		# inverse_LW_power_second_LW_term: constrain the amplitude of the model components of the 2nd LW basis vector (e.g. quad model comp.)
		# Note: The indices used are correct for the current ordering of basis vectors when nf is an even number...
		###
		PowerI=np.zeros(self.Npar)+self.inverse_LW_power #set to zero for a uniform distribution
		if self.inverse_LW_power==0.0:
			PowerI[self.nf/2-1::self.nf]=self.inverse_LW_power_zeroth_LW_term #set to zero for a uniform distribution
			PowerI[self.nf-2::self.nf]=self.inverse_LW_power_first_LW_term #set to zero for a uniform distribution
			PowerI[self.nf-1::self.nf]=self.inverse_LW_power_second_LW_term #set to zero for a uniform distribution

		if self.dimensionless_PS:
			for i_bin in range(len(self.k_cube_voxels_in_bin)):
				dimensionless_PS_scaling = self.calc_dimensionless_power_spectral_normalisation_21cmFAST(i_bin)
				PowerI[self.k_cube_voxels_in_bin[i_bin]] = dimensionless_PS_scaling/x[i_bin] #NOTE: fitting for power not std here
		else:
			for i_bin in range(len(self.k_cube_voxels_in_bin)):
				PowerI[self.k_cube_voxels_in_bin[i_bin]] = 1./x[i_bin]  #NOTE: fitting for power not std here

		# brk()
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

			logPhiDet=-1*np.sum(np.log(PhiI[np.logical_not(self.masked_power_spectral_modes)])).real #Only possible because Phi is diagonal (otherwise would need to calc np.linalg.slogdet(Phi)). -1 factor is to get logPhiDet from logPhiIDet. Note: the real part of this calculation matches the solution given by np.linalg.slogdet(Phi))

			MargLogL =  -0.5*logSigmaDet -0.5*logPhiDet + 0.5*dbarSigmaIdbar
			vals = map(real, (-0.5*logSigmaDet, -0.5*logPhiDet, 0.5*dbarSigmaIdbar, MargLogL))
			MargLogL =  MargLogL.real
			if self.Print_debug:
				print 'MargLogL =  -0.5*logSigmaDet -0.5*logPhiDet + 0.5*dbarSigmaIdbar', MargLogL
				print 'MargLogL.real', MargLogL.real
				print 'logSigmaDet, logPhiDet, dbarSigmaIdbar', logSigmaDet, logPhiDet, dbarSigmaIdbar


			print_rate = 10000
			# brk()
			
			if self.nu>10:
				print_rate=100
			if self.count%print_rate==0:
				print 'count', self.count
				print 'Time since class instantiation: %f'%(time.time()-self.instantiation_time)
				print 'Time for this likelihood call: %f'%(time.time()-start)
			return (MargLogL.squeeze())*1.0, phi
		except Exception as e:
			print 'Exception encountered...'
			print e
			return -np.inf, phi









