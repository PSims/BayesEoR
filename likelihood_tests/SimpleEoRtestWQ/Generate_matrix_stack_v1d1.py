import numpy as np
import numpy
from numpy import arange, shape
import scipy
from subprocess import os
import shutil
import sys
import pylab
import time
import h5py
from scipy.linalg import block_diag

from Linalg import IDFT_Array_IDFT_2D_ZM_SH, makeGaussian, Produce_Full_Coordinate_Arrays, Produce_Coordinate_Arrays_ZM
from Linalg import Produce_Coordinate_Arrays_ZM_Coarse, Produce_Coordinate_Arrays_ZM_SH, Calc_Coords_High_Res_Im_to_Large_uv
from Linalg import Calc_Coords_Large_Im_to_High_Res_uv, Restore_Centre_Pixel, Calc_Indices_Centre_3x3_Grid
from Linalg import Delete_Centre_3x3_Grid, Delete_Centre_Pix, N_is_Odd, Calc_Indices_Centre_NxN_Grid, Obtain_Centre_NxN_Grid
from Linalg import Restore_Centre_3x3_Grid, Restore_Centre_NxN_Grid, Generate_Combined_Coarse_plus_Subharmic_uv_grids
from Linalg import IDFT_Array_IDFT_2D_ZM_SH, DFT_Array_DFT_2D, IDFT_Array_IDFT_2D, DFT_Array_DFT_2D_ZM, IDFT_Array_IDFT_2D_ZM
from Linalg import Construct_Hermitian, Construct_Hermitian_Gridding_Matrix, Construct_Hermitian_Gridding_Matrix_CosSin
from Linalg import Construct_Hermitian_Gridding_Matrix_CosSin_SH_v4, IDFT_Array_IDFT_1D, IDFT_Array_IDFT_1D
from Linalg import generate_gridding_matrix_vis_ordered_to_chan_ordered

from SimData import generate_test_sim_signal, map_out_bins_for_power_spectral_coefficients
from SimData import generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector
from SimData import generate_test_sim_signal_with_large_spectral_scales_1
from SimData import map_out_bins_for_power_spectral_coefficients_WQ_v2

from Linalg import IDFT_Array_IDFT_1D_WQ, generate_gridding_matrix_vis_ordered_to_chan_ordered_WQ
from Linalg import IDFT_Array_IDFT_1D_WQ_ZM, generate_gridding_matrix_vis_ordered_to_chan_ordered_ZM

###
# NOTE: a (960*38)*(960*38) array requires ~10.75 GB of memory (960*38*969*38*(64./8)/1.e9 GB precisely for a numpy.float64 double precision array). With 128 GB of memory per node 11 matrices of this size to be held in memory simultaneously.
###

###
# NOTE: this file supersedes Likelihood_v1d6_3D_ZM__save_arrays_v1d1.py.
###

## ======================================================================================================
## ======================================================================================================

class BuildMatrixTree(object):
	def __init__(self, array_save_directory, **kwargs):
		self.array_save_directory = array_save_directory

		self.matrix_prerequisites_dictionary = {}
		self.matrix_prerequisites_dictionary['multi_vis_idft_array_1D'] = ['idft_array_1D']
		self.matrix_prerequisites_dictionary['multi_vis_idft_array_1D_WQ'] = ['idft_array_1D_WQ']
		self.matrix_prerequisites_dictionary['gridding_matrix_chan_ordered_to_vis_ordered'] = ['gridding_matrix_vis_ordered_to_chan_ordered']
		self.matrix_prerequisites_dictionary['Fz'] = ['gridding_matrix_vis_ordered_to_chan_ordered', 'multi_vis_idft_array_1D_WQ', 'multi_vis_idft_array_1D']
		self.matrix_prerequisites_dictionary['multi_chan_idft_array_noZMchan'] = ['idft_array']
		self.matrix_prerequisites_dictionary['multi_chan_dft_array_noZMchan'] = ['dft_array']
		self.matrix_prerequisites_dictionary['Fprime'] = ['multi_chan_idft_array_noZMchan']
		self.matrix_prerequisites_dictionary['Finv'] = ['multi_chan_dft_array_noZMchan']
		self.matrix_prerequisites_dictionary['Fprime_Fz'] = ['Fprime', 'Fz']
		self.matrix_prerequisites_dictionary['T'] = ['Finv','Fprime_Fz']
		self.matrix_prerequisites_dictionary['Ninv_T'] = ['Ninv','T']
		self.matrix_prerequisites_dictionary['T_Ninv_T'] = ['T', 'Ninv_T']
		self.matrix_prerequisites_dictionary['block_T_Ninv_T'] = ['T_Ninv_T']

	def check_for_prerequisites(self, parent_matrix):
		prerequisites_status = {}
		if parent_matrix in self.matrix_prerequisites_dictionary.keys():
			for child_matrix in self.matrix_prerequisites_dictionary[parent_matrix]:
				matrix_available = self.check_if_matrix_exists(child_matrix)
				prerequisites_status[child_matrix]=matrix_available
		return prerequisites_status

	def check_if_matrix_exists(self, matrix_name):
		matrix_available =  os.path.exists(self.array_save_directory+matrix_name+'.h5')
		return matrix_available

	def create_directory(self, Directory,**kwargs):
		"""
		Create output directory if it doesn't exist
		"""
		if not os.path.exists(Directory):
			print 'Directory not found: \n\n'+Directory+"\n"
			print 'Creating required directory structure..'
			os.makedirs(Directory)
		
		return 0

	def output_to_hdf5(self, output_array, output_directory, file_name, dataset_name):
		"""
		# Write array to HDF5 file
		"""
		start = time.time()
		self.create_directory(output_directory)
		output_path = '/'.join((output_directory,file_name))
		print "Writing data to", output_path
		with h5py.File(output_path, 'w') as hf:
		    hf.create_dataset(dataset_name,  data=output_array)
		print 'Time taken: {}'.format(time.time()-start)
		return 0

	def read_data_from_hdf5(self, file_path, dataset_name):
		"""
		# Read array from HDF5 file
		"""
		with h5py.File(file_path, 'r') as hf:
		    data = hf[dataset_name][:]
		return data


## ======================================================================================================
## ======================================================================================================

class BuildMatrices(BuildMatrixTree):
	def __init__(self, array_save_directory, nu, nv, nx, ny, neta, nf, nq, sigma, **kwargs):
		super(BuildMatrices, self).__init__(array_save_directory)

		##===== Defaults =======
		default_npl = 0
		
		##===== Inputs =======
		self.npl=kwargs.pop('npl',default_npl)

		print self.array_save_directory 
		print self.matrix_prerequisites_dictionary
		self.nu = nu
		self.nv = nv
		self.nx = nx
		self.ny = ny
		self.neta = neta
		self.nf = nf
		self.nq = nq
		self.sigma = sigma
		self.Fz_normalisation = self.nf**0.5
		self.DFT2D_Fz_normalisation = (self.nu*self.nv*self.nf)**0.5
		self.n_Fourier = (self.nu*self.nv-1)*self.nf
		self.n_quad = (self.nu*self.nv-1)*self.nq
		self.n_model = self.n_Fourier+self.n_quad
		self.n_dat = self.n_Fourier
		
		self.matrix_construction_methods_dictionary={}
		self.matrix_construction_methods_dictionary['T_Ninv_T'] = self.build_T_Ninv_T
		self.matrix_construction_methods_dictionary['idft_array_1D'] = self.build_idft_array_1D
		self.matrix_construction_methods_dictionary['idft_array_1D_WQ'] = self.build_idft_array_1D_WQ
		self.matrix_construction_methods_dictionary['gridding_matrix_vis_ordered_to_chan_ordered'] = self.build_gridding_matrix_vis_ordered_to_chan_ordered
		self.matrix_construction_methods_dictionary['gridding_matrix_chan_ordered_to_vis_ordered'] = self.build_gridding_matrix_chan_ordered_to_vis_ordered
		self.matrix_construction_methods_dictionary['Fz'] = self.build_Fz
		self.matrix_construction_methods_dictionary['multi_vis_idft_array_1D_WQ'] = self.build_multi_vis_idft_array_1D_WQ
		self.matrix_construction_methods_dictionary['multi_vis_idft_array_1D'] = self.build_multi_vis_idft_array_1D
		self.matrix_construction_methods_dictionary['idft_array'] = self.build_idft_array
		self.matrix_construction_methods_dictionary['multi_chan_dft_array_noZMchan'] = self.build_multi_chan_dft_array_noZMchan
		self.matrix_construction_methods_dictionary['dft_array'] = self.build_dft_array
		self.matrix_construction_methods_dictionary['Fprime'] = self.build_Fprime
		self.matrix_construction_methods_dictionary['multi_chan_idft_array_noZMchan'] = self.build_multi_chan_idft_array_noZMchan
		self.matrix_construction_methods_dictionary['Finv'] = self.build_Finv
		self.matrix_construction_methods_dictionary['multi_chan_dft_array_noZMchan'] = self.build_multi_chan_dft_array_noZMchan
		self.matrix_construction_methods_dictionary['Fprime_Fz'] = self.build_Fprime_Fz
		self.matrix_construction_methods_dictionary['T'] = self.build_T
		self.matrix_construction_methods_dictionary['Ninv'] = self.build_Ninv
		self.matrix_construction_methods_dictionary['Ninv_T'] = self.build_Ninv_T
		self.matrix_construction_methods_dictionary['block_T_Ninv_T'] = self.build_block_T_Ninv_T
		self.matrix_construction_methods_dictionary['N'] = self.build_N

	def load_prerequisites(self, matrix_name):
		prerequisite_matrices_dictionary = {}
		print 'About to check and load any prerequisites for', matrix_name
		print 'Checking for prerequisites'
		prerequisites_status = self.check_for_prerequisites(matrix_name)
		if prerequisites_status=={}:
			print matrix_name, 'has no prerequisites. Continuing...'
		else:
			for child_matrix,matrix_available in prerequisites_status.iteritems():
				if matrix_available:
					print child_matrix, 'is available. Loading...'
				else:
					print child_matrix, 'is not available. Building...'
					self.matrix_construction_methods_dictionary[child_matrix]()
					print child_matrix, 'is now available. Loading...'
				###
				# Load prerequisite matrix into prerequisite_matrices_dictionary
				###
				file_path = self.array_save_directory+child_matrix+'.h5'
				dataset_name = child_matrix
				start = time.time()
				data = self.read_data_from_hdf5(file_path, dataset_name)
				prerequisite_matrices_dictionary[child_matrix] = data
				print 'Time taken: {}'.format(time.time()-start)
		
		return prerequisite_matrices_dictionary

	def build_T(self):
		matrix_name='T'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		T = np.dot(pmd['Finv'],pmd['Fprime_Fz'])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(T, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_Ninv_T(self):
		matrix_name='Ninv_T'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		Ninv_T = np.dot(pmd['Ninv'],pmd['T'])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(Ninv_T, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_Fprime_Fz(self):
		matrix_name='Fprime_Fz'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		Fprime_Fz = np.dot(pmd['Fprime'],pmd['Fz'])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(Fprime_Fz, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_multi_chan_dft_array_noZMchan(self):
		matrix_name='multi_chan_dft_array_noZMchan'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		multi_chan_dft_array_noZMchan = block_diag(*[pmd['dft_array'].T for i in range(self.nf)])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(multi_chan_dft_array_noZMchan, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_Finv(self):
		matrix_name='Finv'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		Finv = pmd['multi_chan_dft_array_noZMchan']
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(Finv, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_multi_chan_idft_array_noZMchan(self):
		matrix_name='multi_chan_idft_array_noZMchan'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		multi_chan_idft_array_noZMchan = block_diag(*[pmd['idft_array'].T for i in range(self.nf)])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(multi_chan_idft_array_noZMchan, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_Fprime(self):
		matrix_name='Fprime'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		Fprime = pmd['multi_chan_idft_array_noZMchan']
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(Fprime, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_dft_array(self):
		matrix_name='dft_array'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		dft_array = DFT_Array_DFT_2D_ZM(self.nu,self.nv,self.nx,self.ny)
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(dft_array, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_multi_chan_dft_array_noZMchan(self):
		matrix_name='multi_chan_dft_array_noZMchan'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		multi_chan_dft_array_noZMchan = block_diag(*[pmd['dft_array'].T for i in range(self.nf)])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(multi_chan_dft_array_noZMchan, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_idft_array(self):
		matrix_name='idft_array'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		idft_array = IDFT_Array_IDFT_2D_ZM(self.nu,self.nv,self.nx,self.ny)
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(idft_array, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_multi_chan_idft_array_noZMchan(self):
		matrix_name='multi_chan_idft_array_noZMchan'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		multi_chan_idft_array_noZMchan = block_diag(*[pmd['idft_array'].T for i in range(self.nf)])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(multi_chan_idft_array_noZMchan, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_multi_vis_idft_array_1D(self):
		matrix_name='multi_vis_idft_array_1D'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		multi_vis_idft_array_1D = block_diag(*[pmd['idft_array_1D'] for i in range(self.nu*self.nv-1)])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(multi_vis_idft_array_1D, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_multi_vis_idft_array_1D_WQ(self):
		matrix_name='multi_vis_idft_array_1D_WQ'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		multi_vis_idft_array_1D_WQ = block_diag(*[pmd['idft_array_1D_WQ'].T for i in range(self.nu*self.nv-1)])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(multi_vis_idft_array_1D_WQ, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_Fz(self):
		matrix_name='Fz'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		if self.nq==0:
			# raw_input('in nq=0. continue?')
			Fz = np.dot(pmd['gridding_matrix_vis_ordered_to_chan_ordered'],pmd['multi_vis_idft_array_1D'])
		else:
			Fz = np.dot(pmd['gridding_matrix_vis_ordered_to_chan_ordered'],pmd['multi_vis_idft_array_1D_WQ'])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(Fz, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_gridding_matrix_chan_ordered_to_vis_ordered(self):
		matrix_name='gridding_matrix_chan_ordered_to_vis_ordered'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		gridding_matrix_chan_ordered_to_vis_ordered = pmd['gridding_matrix_vis_ordered_to_chan_ordered'].T #NOTE: taking the transpose reverses the gridding. This is what happens in dbar where Fz.conjugate().T is multiplied by d and the gridding_matrix_vis_ordered_to_chan_ordered.conjugate().T part of Fz transforms d from chan-ordered initially to vis-ordered (Note - .conjugate does nothing to the gridding matrix component of Fz, which is real, it only transforms the 1D IDFT to a DFT).
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(gridding_matrix_chan_ordered_to_vis_ordered, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_gridding_matrix_vis_ordered_to_chan_ordered(self):
		matrix_name='gridding_matrix_vis_ordered_to_chan_ordered'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		gridding_matrix_vis_ordered_to_chan_ordered = generate_gridding_matrix_vis_ordered_to_chan_ordered(self.nu,self.nv,self.nf)
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(gridding_matrix_vis_ordered_to_chan_ordered, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_idft_array_1D(self):
		matrix_name='idft_array_1D'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		idft_array_1D=IDFT_Array_IDFT_1D(self.nf, self.neta)*self.Fz_normalisation #Note: the nf**0.5 normalisation factor results in a symmetric transform and is necessary for correctly estimating the power spectrum. However, it is not consistent with Python's asymmetric DFTs, therefore this factor needs to be removed when comparing to np.fft.fftn cross-checks!
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(idft_array_1D, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_idft_array_1D_WQ(self):
		matrix_name='idft_array_1D_WQ'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		idft_array_1D_WQ=IDFT_Array_IDFT_1D_WQ(self.nf, self.neta, self.nq, npl=self.npl)*self.Fz_normalisation #Note: the nf**0.5 normalisation factor results in a symmetric transform and is necessary for correctly estimating the power spectrum. However, it is not consistent with Python's asymmetric DFTs, therefore this factor needs to be removed when comparing to np.fft.fftn cross-checks!
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(idft_array_1D_WQ, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_N(self):
		matrix_name='N'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		s_size = (self.nu*self.nv-1)*self.nf
		sigma_squared_array = np.ones(s_size)*self.sigma**2 + 0j*np.ones(s_size)*self.sigma**2
		N = np.diag(sigma_squared_array)
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(N, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_Ninv(self):
		matrix_name='Ninv'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		s_size = (self.nu*self.nv-1)*self.nf
		sigma_squared_array = np.ones(s_size)*self.sigma**2 + 0j*np.ones(s_size)*self.sigma**2
		Ninv = np.diag(1./sigma_squared_array)
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(Ninv, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_N(self):
		matrix_name='N'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		s_size = (self.nu*self.nv-1)*self.nf
		sigma_squared_array = np.ones(s_size)*self.sigma**2 + 0j*np.ones(s_size)*self.sigma**2
		N = np.diag(sigma_squared_array)
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(N, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_T_Ninv_T(self):
		matrix_name='T_Ninv_T'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		T_Ninv_T = np.dot(pmd['T'].conjugate().T,pmd['Ninv_T'])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(T_Ninv_T, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_block_T_Ninv_T(self):
		matrix_name='block_T_Ninv_T'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		self.nuv = (self.nu*self.nv-1)
		block_T_Ninv_T = np.array([np.hsplit(block,self.nuv) for block in np.vsplit(pmd['T_Ninv_T'],self.nuv)])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5
		###
		self.output_to_hdf5(block_T_Ninv_T, self.array_save_directory, matrix_name+'.h5', matrix_name)

	def build_matrix_if_it_doesnt_already_exist(self, matrix_name):
		matrix_available = self.check_if_matrix_exists(matrix_name)
		if not matrix_available:
			self.matrix_construction_methods_dictionary[matrix_name]()
		
	def prepare_matrix_stack_for_deletion(self, src, overwrite_existing_matrix_stack):
		if overwrite_existing_matrix_stack:
			if src[-1]=='/':src=src[:-1]
			head,tail = os.path.split(src)
			dst = os.path.join(head,'delete_'+tail)
			print 'Archiving existing matrix stack to:', dst
			del_error_flag = 0
			try:
				shutil.move(src, dst)
			except:
				print 'Archive path already existed. Deleting the previous archive.'
				self.delete_old_matrix_stack(dst, 'y')
				self.prepare_matrix_stack_for_deletion(self.array_save_directory, self.overwrite_existing_matrix_stack)
			return dst

	def delete_old_matrix_stack(self, path_to_old_matrix_stack, confirm_deletion):
		if confirm_deletion.lower()=='y' or confirm_deletion.lower()=='yes':
			shutil.rmtree(path_to_old_matrix_stack)
		else:
			print 'Prior matrix tree archived but not deleted. \nPath to archive:', path_to_old_matrix_stack

	def build_minimum_sufficient_matrix_stack(self, **kwargs):
		default_overwrite_existing_matrix_stack = False
		
		##===== Inputs =======
		self.overwrite_existing_matrix_stack=kwargs.pop('overwrite_existing_matrix_stack',default_overwrite_existing_matrix_stack)

		matrix_stack_dir_exists =  os.path.exists(self.array_save_directory)
		if matrix_stack_dir_exists:
			dst = self.prepare_matrix_stack_for_deletion(self.array_save_directory, self.overwrite_existing_matrix_stack)
		self.build_matrix_if_it_doesnt_already_exist('T_Ninv_T')
		self.build_matrix_if_it_doesnt_already_exist('block_T_Ninv_T')
		self.build_matrix_if_it_doesnt_already_exist('N')
		if matrix_stack_dir_exists and self.overwrite_existing_matrix_stack:
			confirm_deletion = raw_input('Confirm deletion of archived matrix stack? y/n\n')
			self.delete_old_matrix_stack(dst, confirm_deletion)

		print 'Matrix stack complete'


## ======================================================================================================
## ======================================================================================================







