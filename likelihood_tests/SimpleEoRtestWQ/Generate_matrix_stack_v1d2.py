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
from pdb import set_trace as brk
from scipy import sparse

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

from BayesEoR.SimData import generate_test_sim_signal, map_out_bins_for_power_spectral_coefficients
from BayesEoR.SimData import generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector
from BayesEoR.SimData import map_out_bins_for_power_spectral_coefficients_WQ_v2

from BayesEoR.Linalg import IDFT_Array_IDFT_1D_WQ, generate_gridding_matrix_vis_ordered_to_chan_ordered_WQ
from BayesEoR.Linalg import IDFT_Array_IDFT_1D_WQ_ZM, generate_gridding_matrix_vis_ordered_to_chan_ordered_ZM
from BayesEoR.Linalg import nuDFT_Array_DFT_2D, make_Gaussian_beam, make_Uniform_beam, nuDFT_Array_DFT_2D_v2d0


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
		self.matrix_prerequisites_dictionary['Fprime_Fz'] = ['Fprime', 'Fz']
		self.matrix_prerequisites_dictionary['T'] = ['Finv','Fprime_Fz']
		self.matrix_prerequisites_dictionary['Ninv_T'] = ['Ninv','T']
		self.matrix_prerequisites_dictionary['T_Ninv_T'] = ['T', 'Ninv_T']
		self.matrix_prerequisites_dictionary['block_T_Ninv_T'] = ['T_Ninv_T']
		self.matrix_prerequisites_dictionary['Fprime'] = ['multi_chan_idft_array_noZMchan']
		if p.include_instrumental_effects:
			self.matrix_prerequisites_dictionary['Finv'] = ['multi_chan_nudft', 'multi_chan_P']
		else:
			self.matrix_prerequisites_dictionary['Finv'] = ['multi_chan_dft_array_noZMchan']


	def check_for_prerequisites(self, parent_matrix):
		prerequisites_status = {}
		if parent_matrix in self.matrix_prerequisites_dictionary.keys():
			for child_matrix in self.matrix_prerequisites_dictionary[parent_matrix]:
				matrix_available = self.check_if_matrix_exists(child_matrix)
				prerequisites_status[child_matrix]=matrix_available
		return prerequisites_status

	def check_if_matrix_exists(self, matrix_name):
		"""
		# Check is hdf5 or npz file with matrix_name exists.
		If it does, return 1 for an hdf5 file or 2 for an npz
		"""
		hdf5_matrix_available =  os.path.exists(self.array_save_directory+matrix_name+'.h5')
		if hdf5_matrix_available:
			matrix_available = 1
		else:
			npz_matrix_available = os.path.exists(self.array_save_directory+matrix_name+'.npz')
			if npz_matrix_available:
				matrix_available = 2
				if not p.use_sparse_matrices:
					print 'Only the sparse matrix representation is available.'
					print 'Using sparse representation and setting p.use_sparse_matrices=True'
					p.use_sparse_matrices=True
			else:
				matrix_available = 0
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

	def output_data(self, output_array, output_directory, file_name, dataset_name):
		"""
		# Check if the data is an array or sparse matrix and call the corresponding method to output to HDF5 or npz
		"""
		output_array_is_sparse = sparse.issparse(output_array)
		if output_array_is_sparse:
			self.output_sparse_matrix_to_npz(output_array, output_directory, file_name+'.npz', dataset_name)
		else:
			self.output_to_hdf5(output_array, output_directory, file_name+'.h5', dataset_name)
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

	def output_sparse_matrix_to_npz(self, output_array, output_directory, file_name, dataset_name):
		"""
		# Write sparse matrix to npz (note: to maintain sparse matrix attributes need to use sparse.save_npz rather than np.savez)
		"""
		start = time.time()
		self.create_directory(output_directory)
		output_path = '/'.join((output_directory,file_name))
		print "Writing data to", output_path
		sparse.save_npz(output_path, output_array)
		print 'Time taken: {}'.format(time.time()-start)
		return 0

	def read_data_s2d(self, file_path, dataset_name):
		"""
		# Check if the data is an array (.h5) or sparse matrix (.npz) and call the corresponding method to read it in, then convert matrix to numpy array if it is sparse.
		"""
		data = self.read_data(file_path, dataset_name)
		data = self.convert_sparse_matrix_to_dense_numpy_array(data)
		return data

	def read_data(self, file_path, dataset_name):
		"""
		# Check if the data is an array (.h5) or sparse matrix (.npz) and call the corresponding method to read it in
		"""
		if file_path.count('.h5'):
			data = self.read_data_from_hdf5(file_path, dataset_name)
		elif file_path.count('.npz'):
			data = self.read_data_from_npz(file_path, dataset_name)
		else:
			###
			# If no file extension is given, look to see if an hdf5 or npz with the file name exists
			###
			found_npz = os.path.exists(file_path+'.npz')
			if found_npz:
				data = self.read_data_from_npz(file_path+'.npz', dataset_name)
			else:
				found_hdf5 = os.path.exists(file_path+'.h5')
				if found_hdf5:
					data = self.read_data_from_hdf5(file_path+'.h5', dataset_name)

		return data

	def read_data_from_hdf5(self, file_path, dataset_name):
		"""
		# Read array from HDF5 file
		"""
		with h5py.File(file_path, 'r') as hf:
			data = hf[dataset_name][:]
		return data

	def read_data_from_npz(self, file_path, dataset_name):
		"""
		# Read sparse matrix from npz (note: to maintain sparse matrix attributes need to use sparse.load_npz rather than np.loadz)
		"""
		data = sparse.load_npz(file_path)
		return data




## ======================================================================================================
## ======================================================================================================

class BuildMatrices(BuildMatrixTree):
	def __init__(self, array_save_directory, nu, nv, nx, ny, n_vis, neta, nf, nq, sigma, **kwargs):
		super(BuildMatrices, self).__init__(array_save_directory)

		##===== Defaults =======
		default_npl = 0

		##===== Inputs =======
		self.npl=kwargs.pop('npl',default_npl)
		if p.include_instrumental_effects:
			self.uvw_multi_time_step_array_meters = kwargs.pop('uvw_multi_time_step_array_meters')
			self.uvw_multi_time_step_array_meters_vectorised = kwargs.pop('uvw_multi_time_step_array_meters_vectorised')
			###
			# Currently only using uv-coordinates so exclude w for now
			###
			#---
			self.uvw_multi_time_step_array_meters = self.uvw_multi_time_step_array_meters[:,:,:2]
			self.uvw_multi_time_step_array_meters_vectorised = self.uvw_multi_time_step_array_meters_vectorised[:,:2]
			#---
			self.baseline_redundancy_array_time_vis_shaped = kwargs.pop('baseline_redundancy_array_time_vis_shaped')
			self.baseline_redundancy_array_vectorised = kwargs.pop('baseline_redundancy_array_vectorised')
			self.beam_type = kwargs.pop('beam_type')
			self.beam_peak_amplitude = kwargs.pop('beam_peak_amplitude')
			self.FWHM_deg_at_ref_freq_MHz = kwargs.pop('FWHM_deg_at_ref_freq_MHz')
			self.PB_ref_freq_MHz = kwargs.pop('PB_ref_freq_MHz')


		print self.array_save_directory
		print self.matrix_prerequisites_dictionary
		self.nu = nu
		self.nv = nv
		self.nx = nx
		self.ny = ny
		self.n_vis = n_vis
		self.neta = neta
		self.nf = nf
		self.nq = nq
		self.sigma = sigma

		# Fz normalizations
		# self.delta_kz_inv_Mpc = 2 * np.pi / (p.box_size_21cmFAST_Mpc_sc * p.nf / p.box_size_21cmFAST_pix_sc)
		# self.Fz_normalisation = self.nf**0.5 * self.delta_kz_inv_Mpc
		# self.delta_eta_inv_Mpc = 1.0 / (p.box_size_21cmFAST_Mpc_sc * p.nf / p.box_size_21cmFAST_pix_sc)
		# self.delta_eta_inv_Hz = 1.0 / ((p.nf - 1) * p.channel_width_MHz * 1.0e6)
		# According to np.fft.fftfreq, deta should be equal to 1 / (nf * df)
		self.delta_eta_inv_Hz = 1.0 / (p.nf * p.channel_width_MHz * 1.0e6)
		self.Fz_normalisation = self.nf * self.delta_eta_inv_Hz

		# Fprime normalizations
		# self.delta_kx_inv_Mpc = 2 * np.pi / (p.box_size_21cmFAST_Mpc_sc)
		# self.delta_ky_inv_Mpc = 2 * np.pi / (p.box_size_21cmFAST_Mpc_sc)
		# self.Fprime_normalisation = (self.nu * self.nv - 1)**0.5 * self.delta_kx_inv_Mpc * self.delta_ky_inv_Mpc
		self.delta_u_inv_rad = p.uv_pixel_width_wavelengths
		self.Fprime_normalisation = (self.nu * self.nv) * self.delta_u_inv_rad**2

		# Finv normalizations
		self.dA_sr = p.sky_model_pixel_area_sr
		self.Finv_normalisation = self.dA_sr

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
		self.matrix_construction_methods_dictionary['multi_chan_nudft'] = self.build_multi_chan_nudft
		self.matrix_construction_methods_dictionary['multi_chan_P'] = self.build_multi_chan_P

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
					#Re-check that that the matrix now exists and whether it is dense (hdf5; matrix_available=1) or sparse (npz; matrix_available=2)
					matrix_available = self.check_if_matrix_exists(child_matrix)
					print child_matrix, 'is now available. Loading...'
				###
				# Load prerequisite matrix into prerequisite_matrices_dictionary
				###
				if matrix_available == 1:
					file_extension = '.h5'
				elif matrix_available == 2:
					file_extension = '.npz'
				else:
					file_extension = '.h5'
				matrix_available
				file_path = self.array_save_directory+child_matrix+file_extension
				dataset_name = child_matrix
				start = time.time()
				data = self.read_data(file_path, dataset_name)
				prerequisite_matrices_dictionary[child_matrix] = data
				print 'Time taken: {}'.format(time.time()-start)

		return prerequisite_matrices_dictionary

	def dot_product(self, matrix_A, matrix_B):
		"""
		Calculate the dot product of matrix_A and matrix_B correctly whether either or both of A and B sparse or dense.
		"""
		matrix_A_is_sparse = sparse.issparse(matrix_A)
		matrix_B_is_sparse = sparse.issparse(matrix_B)
		if not (matrix_A_is_sparse or matrix_B_is_sparse):
			AB = np.dot(matrix_A, matrix_B) #Use np.dot to calculate the dot product of np.arrays
		else: #One of the matrices is sparse - need to use python matrix syntax (i.e. * for dot product)
			#NOTE:sparse*dense=dense, dense*sparse=dense, sparse*sparse=sparse
			print self.convert_sparse_to_dense_matrix(matrix_A).shape
			print self.convert_sparse_to_dense_matrix(matrix_B).shape
			AB = matrix_A * matrix_B
		return AB

	def convert_sparse_to_dense_matrix(self, matrix_A):
		"""
		Convert scipy.sparse matrix to dense matrix
		"""
		matrix_A_is_sparse = sparse.issparse(matrix_A)
		if matrix_A_is_sparse:
			matrix_A_dense = matrix_A.todense()
		else:
			matrix_A_dense = matrix_A
		return matrix_A_dense

	def convert_sparse_matrix_to_dense_numpy_array(self, matrix_A):
		"""
		Convert scipy.sparse matrix to dense numpy array
		"""
		matrix_A_dense = self.convert_sparse_to_dense_matrix(matrix_A)
		matrix_A_dense_np_array = np.array(matrix_A_dense)
		return matrix_A_dense_np_array

	def sd_block_diag(self, block_matrices_list):
		"""
		Generate block diagonal matrix from blocks in block_matrices_list
		"""
		if p.use_sparse_matrices:
			return sparse.block_diag(block_matrices_list)
		else:
			return block_diag(*block_matrices_list)

	def build_T(self):
		matrix_name='T'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		T = self.dot_product(pmd['Finv'],pmd['Fprime_Fz'])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(T, self.array_save_directory, matrix_name, matrix_name)

	def build_Ninv_T(self):
		matrix_name='Ninv_T'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		Ninv_T = self.dot_product(pmd['Ninv'],pmd['T'])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(Ninv_T, self.array_save_directory, matrix_name, matrix_name)

	def build_Fprime_Fz(self):
		matrix_name='Fprime_Fz'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		Fprime_Fz = self.dot_product(pmd['Fprime'],pmd['Fz'])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(Fprime_Fz, self.array_save_directory, matrix_name, matrix_name)

	def build_multi_chan_dft_array_noZMchan(self):
		matrix_name='multi_chan_dft_array_noZMchan'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		multi_chan_dft_array_noZMchan = self.sd_block_diag([pmd['dft_array'].T for i in range(self.nf)])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(multi_chan_dft_array_noZMchan, self.array_save_directory, matrix_name, matrix_name)

	def build_multi_chan_nudft(self):
		"""
		Build nudft array from (l,m,f) to (u,v,f) sampled by the instrument.
		If use_nvis_nt_nchan_ordering: model visibilities will be ordered (nvis*nt) per chan for all channels (this is the old default).
		If use_nvis_nchan_nt_ordering: model visibilities will be ordered (nvis*nchan) per time step for all time steps (this ordering is required when using a drift scan primary beam).
		"""
		matrix_name='multi_chan_nudft'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		nu_array_MHz = p.nu_min_MHz+np.arange(p.nf)*p.channel_width_MHz
		if p.use_nvis_nt_nchan_ordering:
			sampled_uvw_coords_m = self.uvw_multi_time_step_array_meters_vectorised.copy()
			multi_chan_nudft = self.sd_block_diag([nuDFT_Array_DFT_2D(self.nu,self.nv,self.nx,self.ny, chan_freq_MHz, sampled_uvw_coords_m).T for chan_freq_MHz in nu_array_MHz]) #Matrix shape is (nx*ny*nf,nv*nt*nf)
		elif p.use_nvis_nchan_nt_ordering: #This will be used if a drift scan primary beam is included in the data model (i.e. p.model_drift_scan_primary_beam=True)
			sampled_uvw_coords_m = self.uvw_multi_time_step_array_meters.copy() #NOTE: p.uvw_multi_time_step_array_meters is in (nvis_per_chan,nchan) order (unlike p.uvw_multi_time_step_array_meters_vectorised which is used in nuDFT_Array_DFT_2D and is a vector of nvis_per_chan*nchan (u,v) coords) to make it simpler to index the array to calculate that the visibilities for all frequencies for single time steps in sampled_uvw_coords_wavelengths_all_freqs_all_times and then index over time steps to make multi_chan_nudft with nt block diagonal matrices, themselves each consisting of nchan blocks with each block the nudft from image to sampled visibilities as that time and channel frequency.
			sampled_uvw_coords_wavelengths_all_freqs_all_times = np.array([sampled_uvw_coords_m/(p.speed_of_light/(chan_freq_MHz*1.e6)) for chan_freq_MHz in nu_array_MHz]) # Convert uv-coordinates from meters to wavelengths at frequency chan_freq_MHz for all chan_freq_MHz in nu_array_MHz
			sampled_uvw_coords_inverse_pixel_units_all_freqs_all_times = sampled_uvw_coords_wavelengths_all_freqs_all_times/p.uv_pixel_width_wavelengths #Convert uv-coordinates from wavelengths to inverse pixel units

			multi_chan_nudft = self.sd_block_diag( [ self.sd_block_diag([nuDFT_Array_DFT_2D_v2d0(self.nu,self.nv,self.nx,self.ny, sampled_uvw_coords_inverse_pixel_units_all_freqs_all_times[freq_i,time_i,:,:].reshape(-1,2)).T for freq_i in range(p.nf)]) for time_i in range(p.nt) ] ) #Matrix shape is (nx*ny*nf*nt,nv*nf*nt)

		else:
			sampled_uvw_coords_m = self.uvw_multi_time_step_array_meters_vectorised
			multi_chan_nudft = self.sd_block_diag([nuDFT_Array_DFT_2D(self.nu,self.nv,self.nx,self.ny, sampled_uvw_coords_m, chan_freq_MHz).T for chan_freq_MHz in nu_array_MHz])

		# Multiply by sky model pixel area to get the units of the model visibilities correct
		multi_chan_nudft *= self.Finv_normalisation

		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(multi_chan_nudft, self.array_save_directory, matrix_name, matrix_name)

	def build_multi_chan_P(self):
		matrix_name='multi_chan_P'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		nu_array_MHz = p.nu_min_MHz+np.arange(p.nf)*p.channel_width_MHz
		image_size_pix = p.nx
		beam_peak_amplitude = p.beam_peak_amplitude
		deg_to_pix = image_size_pix / float(p.simulation_FoV_deg)
		FWHM_deg_at_chan_freq_MHz = [p.FWHM_deg_at_ref_freq_MHz*(float(p.PB_ref_freq_MHz)/chan_freq_MHz) for chan_freq_MHz in nu_array_MHz]
		FWHM_pix_at_chan_freq_MHz = [FWHM_deg*deg_to_pix for FWHM_deg in FWHM_deg_at_chan_freq_MHz]
		if not p.model_drift_scan_primary_beam:
			if p.beam_type.lower() == 'gaussian':
				multi_chan_P = self.sd_block_diag([np.diag(make_Gaussian_beam(image_size_pix,FWHM_pix,beam_peak_amplitude).flatten()) for FWHM_pix in FWHM_pix_at_chan_freq_MHz])
			elif p.beam_type.lower() == 'uniform':
				multi_chan_P = self.sd_block_diag([np.diag(make_Uniform_beam(image_size_pix,beam_peak_amplitude).flatten()) for FWHM_pix in FWHM_pix_at_chan_freq_MHz])
			else:
				multi_chan_P = self.sd_block_diag([np.diag(make_Uniform_beam(image_size_pix,beam_peak_amplitude).flatten()) for FWHM_pix in FWHM_pix_at_chan_freq_MHz])
		else: #Model the time dependence of the primary beam pointing for a drift scan (i.e. change in zenith angle with time due to Earth rotation).
			if p.beam_type.lower() == 'gaussian':
				degrees_per_minute_of_time = 360./(24*60)
				time_step_range = range(np.round(-p.nt/2),np.round(p.nt/2),1)
				pointing_center_HA_pix_offset_array = np.array([i_min*p.integration_time_minutes*degrees_per_minute_of_time*deg_to_pix for i_min in time_step_range]) #Matches offset of pointing center from zenith as a function of time used when calculating the uv-coords in calc_UV_coords_v1d2.py (i.e. the uv-coords in self.uvw_multi_time_step_array_meters). The zenith (and hence PB center) coords will thus be the negative of this array (as used below).
				if not p.use_sparse_matrices:
					multi_chan_P_drift_scan = np.vstack([block_diag(*[np.diag(make_Gaussian_beam(image_size_pix,FWHM_pix,beam_peak_amplitude,center_pix=[image_size_pix/2-pointing_center_HA_pix_offset,image_size_pix/2] ).flatten()) for FWHM_pix in FWHM_pix_at_chan_freq_MHz]) for pointing_center_HA_pix_offset in pointing_center_HA_pix_offset_array])
				else:
					multi_chan_P_drift_scan = sparse.vstack([sparse.block_diag([sparse.diags(make_Gaussian_beam(image_size_pix,FWHM_pix,beam_peak_amplitude,center_pix=[image_size_pix/2-pointing_center_HA_pix_offset,image_size_pix/2] ).flatten()) for FWHM_pix in FWHM_pix_at_chan_freq_MHz]) for pointing_center_HA_pix_offset in pointing_center_HA_pix_offset_array])
				multi_chan_P = multi_chan_P_drift_scan
			elif p.beam_type.lower() == 'uniform': #Uniform beam is unaltered by drift scan modelling
				multi_chan_P = self.sd_block_diag([np.diag(make_Uniform_beam(image_size_pix,beam_peak_amplitude).flatten()) for FWHM_pix in FWHM_pix_at_chan_freq_MHz])
			else: #Uniform beam is unaltered by drift scan modelling
				multi_chan_P = self.sd_block_diag([np.diag(make_Uniform_beam(image_size_pix,beam_peak_amplitude).flatten()) for FWHM_pix in FWHM_pix_at_chan_freq_MHz])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(multi_chan_P, self.array_save_directory, matrix_name, matrix_name)

	def build_Finv(self):
		matrix_name='Finv'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		if p.include_instrumental_effects:
			Finv = self.dot_product(pmd['multi_chan_nudft'], pmd['multi_chan_P'])
		else:
			Finv = pmd['multi_chan_dft_array_noZMchan']
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(Finv, self.array_save_directory, matrix_name, matrix_name)

	def build_multi_chan_idft_array_noZMchan(self):
		matrix_name='multi_chan_idft_array_noZMchan'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		multi_chan_idft_array_noZMchan = self.sd_block_diag([pmd['idft_array'].T for i in range(self.nf)])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(multi_chan_idft_array_noZMchan, self.array_save_directory, matrix_name, matrix_name)

	def build_Fprime(self):
		matrix_name='Fprime'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		Fprime = pmd['multi_chan_idft_array_noZMchan']
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(Fprime, self.array_save_directory, matrix_name, matrix_name)

	def build_dft_array(self):
		matrix_name='dft_array'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		dft_array = DFT_Array_DFT_2D_ZM(self.nu,self.nv,self.nx,self.ny)
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(dft_array, self.array_save_directory, matrix_name, matrix_name)

	def build_multi_chan_dft_array_noZMchan(self):
		matrix_name='multi_chan_dft_array_noZMchan'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		multi_chan_dft_array_noZMchan = self.sd_block_diag([pmd['dft_array'].T for i in range(self.nf)])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(multi_chan_dft_array_noZMchan, self.array_save_directory, matrix_name, matrix_name)

	def build_idft_array(self):
		matrix_name='idft_array'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		idft_array = IDFT_Array_IDFT_2D_ZM(self.nu,self.nv,self.nx,self.ny) * self.Fprime_normalisation
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(idft_array, self.array_save_directory, matrix_name, matrix_name)

	# def build_multi_chan_idft_array_noZMchan(self):
	# 	matrix_name='multi_chan_idft_array_noZMchan'
	# 	pmd = self.load_prerequisites(matrix_name)
	# 	start = time.time()
	# 	print 'Performing matrix algebra'
	# 	multi_chan_idft_array_noZMchan = self.sd_block_diag([pmd['idft_array'].T for i in range(self.nf)])
	# 	print 'Time taken: {}'.format(time.time()-start)
	# 	###
	# 	# Save matrix to HDF5 or sparse matrix to npz
	# 	###
	# 	self.output_data(multi_chan_idft_array_noZMchan, self.array_save_directory, matrix_name, matrix_name)

	def build_multi_vis_idft_array_1D(self):
		matrix_name='multi_vis_idft_array_1D'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		if p.fit_for_monopole:
			multi_vis_idft_array_1D = self.sd_block_diag([pmd['idft_array_1D'] for i in range(self.nu*self.nv)])
		else:
			multi_vis_idft_array_1D = self.sd_block_diag([pmd['idft_array_1D'] for i in range(self.nu*self.nv-1)])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(multi_vis_idft_array_1D, self.array_save_directory, matrix_name, matrix_name)

	def build_multi_vis_idft_array_1D_WQ(self):
		matrix_name='multi_vis_idft_array_1D_WQ'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		if p.fit_for_monopole:
			multi_vis_idft_array_1D_WQ = self.sd_block_diag([pmd['idft_array_1D_WQ'].T for i in range(self.nu*self.nv)])
		else:
			multi_vis_idft_array_1D_WQ = self.sd_block_diag([pmd['idft_array_1D_WQ'].T for i in range(self.nu*self.nv-1)])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(multi_vis_idft_array_1D_WQ, self.array_save_directory, matrix_name, matrix_name)

	def build_Fz(self):
		matrix_name='Fz'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		if self.nq==0:
			# raw_input('in nq=0. continue?')
			Fz = self.dot_product(pmd['gridding_matrix_vis_ordered_to_chan_ordered'],pmd['multi_vis_idft_array_1D'])
		else:
			Fz = self.dot_product(pmd['gridding_matrix_vis_ordered_to_chan_ordered'],pmd['multi_vis_idft_array_1D_WQ'])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(Fz, self.array_save_directory, matrix_name, matrix_name)

	def build_gridding_matrix_chan_ordered_to_vis_ordered(self):
		matrix_name='gridding_matrix_chan_ordered_to_vis_ordered'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		gridding_matrix_chan_ordered_to_vis_ordered = pmd['gridding_matrix_vis_ordered_to_chan_ordered'].T #NOTE: taking the transpose reverses the gridding. This is what happens in dbar where Fz.conjugate().T is multiplied by d and the gridding_matrix_vis_ordered_to_chan_ordered.conjugate().T part of Fz transforms d from chan-ordered initially to vis-ordered (Note - .conjugate does nothing to the gridding matrix component of Fz, which is real, it only transforms the 1D IDFT to a DFT).
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(gridding_matrix_chan_ordered_to_vis_ordered, self.array_save_directory, matrix_name, matrix_name)

	def build_gridding_matrix_vis_ordered_to_chan_ordered(self):
		matrix_name='gridding_matrix_vis_ordered_to_chan_ordered'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		gridding_matrix_vis_ordered_to_chan_ordered = generate_gridding_matrix_vis_ordered_to_chan_ordered(self.nu,self.nv,self.nf)
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(gridding_matrix_vis_ordered_to_chan_ordered, self.array_save_directory, matrix_name, matrix_name)

	def build_idft_array_1D(self):
		matrix_name='idft_array_1D'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		idft_array_1D=IDFT_Array_IDFT_1D(self.nf, self.neta)*self.Fz_normalisation #Note: the nf**0.5 normalisation factor results in a symmetric transform and is necessary for correctly estimating the power spectrum. However, it is not consistent with Python's asymmetric DFTs, therefore this factor needs to be removed when comparing to np.fft.fftn cross-checks!
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(idft_array_1D, self.array_save_directory, matrix_name, matrix_name)

	def build_idft_array_1D_WQ(self):
		matrix_name='idft_array_1D_WQ'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		idft_array_1D_WQ=IDFT_Array_IDFT_1D_WQ(self.nf, self.neta, self.nq, npl=self.npl, nu_min_MHz=p.nu_min_MHz, channel_width_MHz=p.channel_width_MHz, beta=p.beta)*self.Fz_normalisation #Note: the nf**0.5 normalisation factor results in a symmetric transform and is necessary for correctly estimating the power spectrum. However, it is not consistent with Python's asymmetric DFTs, therefore this factor needs to be removed when comparing to np.fft.fftn cross-checks!
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(idft_array_1D_WQ, self.array_save_directory, matrix_name, matrix_name)

	def build_Ninv(self):
		matrix_name='Ninv'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		if p.include_instrumental_effects:
			if p.use_nvis_nt_nchan_ordering:
				baseline_redundancy_array = self.baseline_redundancy_array_vectorised #This array is channel_ordered and the covariance matrix assumes a channel_ordered data set (this vector should be re-ordered if the data is in a different order)
				s_size = self.n_vis*self.nf
				multifreq_baseline_redundancy_array = np.array([baseline_redundancy_array for i in range(p.nf)]).flatten()
				sigma_accounting_for_redundancy = self.sigma / multifreq_baseline_redundancy_array**0.5 #RMS drops as the squareroot of the number of redundant samples
				sigma_squared_array = np.ones(s_size)*sigma_accounting_for_redundancy**2 + 0j*np.ones(s_size)*sigma_accounting_for_redundancy**2
			elif p.use_nvis_nchan_nt_ordering:
				baseline_redundancy_array_time_vis_shaped = self.baseline_redundancy_array_time_vis_shaped
				baseline_redundancy_array_time_freq_vis = np.array([[baseline_redundancy_array_vis for i in range(p.nf)] for baseline_redundancy_array_vis in baseline_redundancy_array_time_vis_shaped]).flatten()
				s_size = self.n_vis*self.nf
				sigma_accounting_for_redundancy = self.sigma / baseline_redundancy_array_time_freq_vis**0.5 #RMS drops as the squareroot of the number of redundant samples
				sigma_squared_array = np.ones(s_size)*sigma_accounting_for_redundancy**2 + 0j*np.ones(s_size)*sigma_accounting_for_redundancy**2
		else:
			s_size = (self.nu*self.nv-1)*self.nf
			sigma_squared_array = np.ones(s_size)*self.sigma**2 + 0j*np.ones(s_size)*self.sigma**2

		if p.use_sparse_matrices:
			Ninv = sparse.diags(1./sigma_squared_array)
		else:
			Ninv = np.diag(1./sigma_squared_array)
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(Ninv, self.array_save_directory, matrix_name, matrix_name)

	def build_N(self):
		matrix_name='N'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		if p.include_instrumental_effects:
			if p.use_nvis_nt_nchan_ordering:
				baseline_redundancy_array = self.baseline_redundancy_array_vectorised #This array is channel_ordered and the covariance matrix assumes a channel_ordered data set (this vector should be re-ordered if the data is in a different order)
				s_size = self.n_vis*self.nf
				multifreq_baseline_redundancy_array = np.array([baseline_redundancy_array for i in range(p.nf)]).flatten()
				sigma_accounting_for_redundancy = self.sigma / multifreq_baseline_redundancy_array**0.5 #RMS drops as the squareroot of the number of redundant samples
				sigma_squared_array = np.ones(s_size)*sigma_accounting_for_redundancy**2 + 0j*np.ones(s_size)*sigma_accounting_for_redundancy**2
			elif p.use_nvis_nchan_nt_ordering:
				baseline_redundancy_array_time_vis_shaped = self.baseline_redundancy_array_time_vis_shaped
				baseline_redundancy_array_time_freq_vis = np.array([[baseline_redundancy_array_vis for i in range(p.nf)] for baseline_redundancy_array_vis in baseline_redundancy_array_time_vis_shaped]).flatten()
				s_size = self.n_vis*self.nf
				sigma_accounting_for_redundancy = self.sigma / baseline_redundancy_array_time_freq_vis**0.5 #RMS drops as the squareroot of the number of redundant samples
				sigma_squared_array = np.ones(s_size)*sigma_accounting_for_redundancy**2 + 0j*np.ones(s_size)*sigma_accounting_for_redundancy**2
		else:
			s_size = (self.nu*self.nv-1)*self.nf
			sigma_squared_array = np.ones(s_size)*self.sigma**2 + 0j*np.ones(s_size)*self.sigma**2

		if p.use_sparse_matrices:
			N = sparse.diags(sigma_squared_array)
		else:
			N = np.diag(sigma_squared_array)
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(N, self.array_save_directory, matrix_name, matrix_name)

	def build_T_Ninv_T(self):
		matrix_name='T_Ninv_T'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		T_Ninv_T = self.dot_product(pmd['T'].conjugate().T,pmd['Ninv_T'])
		T_Ninv_T = self.convert_sparse_matrix_to_dense_numpy_array(T_Ninv_T) #T_Ninv_T needs to be dense to pass to the GPU (note: if T_Ninv_T is already a dense / a numpy array it will be returned unchanged)
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(T_Ninv_T, self.array_save_directory, matrix_name, matrix_name)

	def build_block_T_Ninv_T(self):
		matrix_name='block_T_Ninv_T'
		pmd = self.load_prerequisites(matrix_name)
		start = time.time()
		print 'Performing matrix algebra'
		if p.fit_for_monopole:
			self.nuv = (self.nu*self.nv)
		else:
			self.nuv = (self.nu*self.nv-1)
		block_T_Ninv_T = np.array([np.hsplit(block,self.nuv) for block in np.vsplit(pmd['T_Ninv_T'],self.nuv)])
		print 'Time taken: {}'.format(time.time()-start)
		###
		# Save matrix to HDF5 or sparse matrix to npz
		###
		self.output_data(block_T_Ninv_T, self.array_save_directory, matrix_name, matrix_name)

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
		default_proceed_without_overwrite_confirmation = False #Set to true when submitting to cluster

		##===== Inputs =======
		self.overwrite_existing_matrix_stack=kwargs.pop('overwrite_existing_matrix_stack',default_overwrite_existing_matrix_stack)
		self.proceed_without_overwrite_confirmation=kwargs.pop('proceed_without_overwrite_confirmation',default_proceed_without_overwrite_confirmation)

		matrix_stack_dir_exists =  os.path.exists(self.array_save_directory)
		if matrix_stack_dir_exists:
			dst = self.prepare_matrix_stack_for_deletion(self.array_save_directory, self.overwrite_existing_matrix_stack)
		self.build_matrix_if_it_doesnt_already_exist('T_Ninv_T')
		self.build_matrix_if_it_doesnt_already_exist('block_T_Ninv_T')
		self.build_matrix_if_it_doesnt_already_exist('N')
		if matrix_stack_dir_exists and self.overwrite_existing_matrix_stack:
			if not self.proceed_without_overwrite_confirmation:
				confirm_deletion = raw_input('Confirm deletion of archived matrix stack? y/n\n')
			else:
				print 'Deletion of archived matrix stack has been pre-confirmed. Continuing...'
				confirm_deletion = 'y'
			self.delete_old_matrix_stack(dst, confirm_deletion)

		print 'Matrix stack complete'


## ======================================================================================================
## ======================================================================================================
