import numpy as np
from subprocess import os
import shutil
import time
import h5py
from scipy.linalg import block_diag
from scipy import sparse
from pathlib import Path

import BayesEoR.Params.params as p
from BayesEoR.Linalg import\
    DFT_Array_DFT_2D_ZM, IDFT_Array_IDFT_2D_ZM, IDFT_Array_IDFT_1D,\
    generate_gridding_matrix_vis_ordered_to_chan_ordered,\
    IDFT_Array_IDFT_1D_WQ,\
    nuDFT_Array_DFT_2D_v2d0,\
    make_Gaussian_beam, make_Uniform_beam
from BayesEoR.Linalg.healpix import Healpix


"""
    NOTE:
    A (960*38)*(960*38) array requires ~10.75 GB of memory
    (960*38*969*38*(64./8)/1.e9 GB precisely for a numpy.float64 double
    precision array). With 128 GB of memory per node 11 matrices of this
    size to be held in memory simultaneously.
"""

SECS_PER_HOUR = 60 * 60
SECS_PER_DAY = SECS_PER_HOUR * 24
DAYS_PER_SEC = 1.0 / SECS_PER_DAY
DEGREES_PER_HOUR = 360.0 / 24
DEGREES_PER_SEC = DEGREES_PER_HOUR * 1 / SECS_PER_HOUR
DEGREES_PER_MIN = DEGREES_PER_SEC * 60


class BuildMatrixTree(object):
    """
    Class containing system level functions used to create the matrices
    used in BayesEoR.
    """
    def __init__(self, array_save_directory, **kwargs):
        self.array_save_directory = array_save_directory

        self.matrix_prerequisites_dictionary = {
            'multi_vis_idft_array_1D': ['idft_array_1D'],
            'multi_vis_idft_array_1D_WQ': ['idft_array_1D_WQ'],
            'gridding_matrix_chan_ordered_to_vis_ordered':
                ['gridding_matrix_vis_ordered_to_chan_ordered'],
            'Fz':
                ['gridding_matrix_vis_ordered_to_chan_ordered',
                 'multi_vis_idft_array_1D_WQ',
                 'multi_vis_idft_array_1D'],
            'multi_chan_idft_array_noZMchan': ['idft_array'],
            'multi_chan_dft_array_noZMchan': ['dft_array'],
            'Fprime_Fz': ['Fprime', 'Fz'],
            'T': ['Finv', 'Fprime_Fz'],
            'Ninv_T': ['Ninv', 'T'],
            'T_Ninv_T': ['T', 'Ninv_T'],
            'block_T_Ninv_T': ['T_Ninv_T'],
            'Fprime': ['multi_chan_idft_array_noZMchan']
            }
        if p.include_instrumental_effects:
            self.matrix_prerequisites_dictionary['Finv'] =\
                ['phasor_matrix', 'multi_chan_nudft', 'multi_chan_P']
        else:
            self.matrix_prerequisites_dictionary['Finv'] =\
                ['multi_chan_dft_array_noZMchan']

        if p.include_instrumental_effects:
            self.beam_center = None

    def check_for_prerequisites(self, parent_matrix):
        prerequisites_status = {}
        if parent_matrix in self.matrix_prerequisites_dictionary.keys():
            for child_matrix\
                    in self.matrix_prerequisites_dictionary[parent_matrix]:
                matrix_available = self.check_if_matrix_exists(child_matrix)
                prerequisites_status[child_matrix]= matrix_available
        return prerequisites_status

    def check_if_matrix_exists(self, matrix_name):
        """
        Check is hdf5 or npz file with matrix_name exists.
        If it does, return 1 for an hdf5 file or 2 for an npz
        """
        hdf5_matrix_available = os.path.exists(
            self.array_save_directory+matrix_name+'.h5')
        if hdf5_matrix_available:
            matrix_available = 1
        else:
            npz_matrix_available = os.path.exists(
                self.array_save_directory+matrix_name+'.npz')
            if npz_matrix_available:
                matrix_available = 2
                if not p.use_sparse_matrices:
                    print('Only the sparse matrix'
                          ' representation is available.')
                    print('Using sparse representation and'
                          ' setting p.use_sparse_matrices=True')
                    p.use_sparse_matrices=True
            else:
                matrix_available = 0
        return matrix_available

    def create_directory(self, Directory, **kwargs):
        """
        Create output directory if it doesn't exist
        """
        if not os.path.exists(Directory):
            print('Directory not found: \n\n'+Directory+"\n")
            print('Creating required directory structure..')
            os.makedirs(Directory)
        return 0

    def output_data(
            self, output_array, output_directory, file_name, dataset_name):
        """
        Check if the data is an array or sparse matrix and call the
        corresponding method to output to HDF5 or npz
        """
        output_array_is_sparse = sparse.issparse(output_array)
        if output_array_is_sparse:
            self.output_sparse_matrix_to_npz(
                output_array, output_directory, file_name+'.npz', dataset_name)
        else:
            self.output_to_hdf5(
                output_array, output_directory, file_name+'.h5', dataset_name)
        return 0

    def output_to_hdf5(
            self, output_array, output_directory, file_name, dataset_name):
        """
        Write array to HDF5 file
        """
        start = time.time()
        self.create_directory(output_directory)
        output_path = '/'.join((output_directory, file_name))
        print('Writing data to', output_path)
        with h5py.File(output_path, 'w') as hf:
            hf.create_dataset(dataset_name, data=output_array)
        print('Time taken: {}'.format(time.time() - start))
        return 0

    def output_sparse_matrix_to_npz(self, output_array, output_directory,
                                    file_name, dataset_name):
        """
        Write sparse matrix to npz (note: to maintain sparse matrix
        attributes need to use sparse.save_npz rather than np.savez)
        """
        start = time.time()
        self.create_directory(output_directory)
        output_path = '/'.join((output_directory, file_name))
        print('Writing data to', output_path)
        sparse.save_npz(output_path, output_array)
        print('Time taken: {}'.format(time.time() - start))
        return 0

    def read_data_s2d(self, file_path, dataset_name):
        """
        Check if the data is an array (.h5) or sparse matrix (.npz)
        and call the corresponding method to read it in, then convert
        matrix to numpy array if it is sparse.
        """
        data = self.read_data(file_path, dataset_name)
        data = self.convert_sparse_matrix_to_dense_numpy_array(data)
        return data

    def read_data(self, file_path, dataset_name):
        """
        Check if the data is an array (.h5) or sparse matrix (.npz)
        and call the corresponding method to read it in
        """
        if (
                self.beam_center is not None
                and dataset_name in self.beam_matrix_names
        ):
            file_path = Path(file_path)
            file_path = file_path.with_name(
                file_path.stem
                + self.beam_center_str
                + file_path.suffix
                )
            file_path = str(file_path)
            dataset_name += self.beam_center_str

        if file_path.count('.h5'):
            data = self.read_data_from_hdf5(file_path, dataset_name)
        elif file_path.count('.npz'):
            data = self.read_data_from_npz(file_path, dataset_name)
        else:
            # If no file extension is given, look to see
            # if an hdf5 or npz with the file name exists
            found_npz = os.path.exists(file_path+'.npz')
            if found_npz:
                data = self.read_data_from_npz(
                    file_path+'.npz', dataset_name)
            else:
                found_hdf5 = os.path.exists(file_path+'.h5')
                if found_hdf5:
                    data = self.read_data_from_hdf5(
                        file_path+'.h5', dataset_name)
        return data

    def read_data_from_hdf5(self, file_path, dataset_name):
        """
        Read array from HDF5 file
        """
        with h5py.File(file_path, 'r') as hf:
            data = hf[dataset_name][:]
        return data

    def read_data_from_npz(self, file_path, dataset_name):
        """
        Read sparse matrix from npz (note: to maintain sparse matrix
        attributes need to use sparse.load_npz rather than np.loadz)
        """
        data = sparse.load_npz(file_path)
        return data


class BuildMatrices(BuildMatrixTree):
    """
    Child class used to create minimum sufficient matrix stack
    using BayesEoR.Linalg functions to create matrices.

    Parameters
    ----------
    array_save_directory : str
        Path to the directory where arrays will be saved.
    nu : int
        Number of pixels on a side for the u axis in the model uv-plane.
    nv : int
        Number of pixels on a side for the v axis in the model uv-plane.
    nx : int
        Number of pixels on a side for the l axis in the sky model.
        Potentially deprecated parameter.
    ny : int
        Number of pixels on a side for the m axis in the sky model.
        Potentially deprecated parameter.
    n_vis : int
        Number of visibilities per channel, i.e. number of
        redundant baselines * number of time steps.
    neta : int
        Number of LoS FT modes.
    nf : int
        Number of frequency channels.
    nq : int
        Number of large spectral scale quadratic modes.
    sigma : float
        Expected noise level in the data vector = signal + noise.
    npl : int
        Number of power law coefficients for the large
        spectral scale model.
    uvw_multi_time_step_array_meters : np.ndarray of floats
        Array containing the (u(t), v(t), w(t)) coordinates
        of the instrument model with shape (nt, nbls, 3).
    uvw_multi_time_step_array_meters_vectorised : np.ndarray of floats
        Reshaped `uvw_multi_time_step_array_meters` with shape
        (nt * nbls, 3).  Each set of nbls entries contain
        the (u, v, w) coordinates for a single integration.
    baseline_redundancy_array_time_vis_shaped : np.ndarray of floats
        Array containing the number of redundant baselines
        at each (u(t), v(t), w(t)) in the instrument model
        with shape (nt, nbls, 1).
    baseline_redundancy_array_vectorised : np.ndarray of floats
        Reshaped `baseline_redundancy_array_time_vis_shaped`
        with shape (nt * nbls, 1).  Each set of nbls entries
        contain the redundancy of each (u, v, w) for a
        single integration.
    phasor_vector : np.ndarray of complex floats
        Array with shape (ndata,) that contains the phasor term
        used to phase visibilities after performing the nuDFT
        from HEALPix (l, m, f) to instrumentally sampled,
        unphased (u, v, f).
    beam_type : str
        Can be either 'uniform' or 'gaussian' (case insensitive).
    beam_peak_amplitude : float
        Peak amplitude of the beam.
    beam_center : tuple of floats
        Beam center in (RA, DEC) coordinates and units of degrees.
        Assumed to be an tuple of offsets along the RA and DEC axes
        relative to the pointing center of the sky model determined
        from the instrument model parameters `telescope_latlonalt`
        and `central_jd`.
    FWHM_deg_at_ref_freq_MHz : float
        FWHM of the beam if using a Gaussian beam.
    PB_ref_freq_MHz : float
        If using a chromatic beam, sets the reference frequency
        that the beam is scaled against spectrally.
    effective_noise : np.ndarray of complex floats
        If the data vector being analyzed contains signal + noise,
        the effective_noise vector contains the estimate of the
        noise in the data vector.  Must have the shape and ordering
        of the data vector, i.e. (ndata,).
    """
    def __init__(self, array_save_directory, nu, nv, nx, ny,
                 n_vis, neta, nf, nq, sigma, **kwargs):
        super(BuildMatrices, self).__init__(array_save_directory)

        # ===== Defaults =====
        default_npl = 0

        # ===== Inputs =====
        self.npl = kwargs.pop('npl', default_npl)
        if p.include_instrumental_effects:
            self.uvw_multi_time_step_array_meters =\
                kwargs.pop('uvw_multi_time_step_array_meters')
            self.uvw_multi_time_step_array_meters_vectorised =\
                kwargs.pop('uvw_multi_time_step_array_meters_vectorised')
            # Currently only using uv-coordinates so exclude w for now
            self.uvw_multi_time_step_array_meters =\
                self.uvw_multi_time_step_array_meters[:, :, :2]
            self.uvw_multi_time_step_array_meters_vectorised =\
                self.uvw_multi_time_step_array_meters_vectorised[:, :2]
            self.baseline_redundancy_array_time_vis_shaped =\
                kwargs.pop('baseline_redundancy_array_time_vis_shaped')
            self.baseline_redundancy_array_vectorised =\
                kwargs.pop('baseline_redundancy_array_vectorised')
            # Load in phasor data vector to phase data after performing
            # the nuDFT from lmf -> instrumentally sampled uvf
            self.phasor_vector = kwargs.pop('phasor_vector')
            self.beam_type = kwargs.pop('beam_type')
            self.beam_peak_amplitude = kwargs.pop('beam_peak_amplitude')
            self.beam_center = kwargs.pop('beam_center', None)
            self.FWHM_deg_at_ref_freq_MHz =\
                kwargs.pop('FWHM_deg_at_ref_freq_MHz')
            self.PB_ref_freq_MHz = kwargs.pop('PB_ref_freq_MHz')
            # Estimate for the noise vector in the data if input data
            # vector contains noise
            self.effective_noise = kwargs.pop('effective_noise', None)

            # Set up Healpix instance
            self.hp = Healpix(
                fov_deg=p.simulation_FoV_deg,
                nside=p.nside,
                telescope_latlonalt=p.telescope_latlonalt,
                central_jd=p.central_jd,
                nt=p.nt,
                int_time=p.integration_time_minutes * 60,
                beam_type=self.beam_type,
                peak_amp=self.beam_peak_amplitude,
                fwhm_deg=self.FWHM_deg_at_ref_freq_MHz,
                beam_center=self.beam_center,
                rel=True
                )

        # Set necessary / useful parameter values
        self.nu = nu
        self.nv = nv
        self.nx = nx
        self.ny = ny
        self.n_vis = n_vis
        self.neta = neta
        self.nf = nf
        self.nq = nq
        self.sigma = sigma

        # Fz normalization
        self.delta_eta_inv_Hz = 1.0 / (p.nf * p.channel_width_MHz * 1.0e6)
        self.Fz_normalisation = self.nf * self.delta_eta_inv_Hz

        # Fprime normalization
        self.delta_u_inv_rad = p.uv_pixel_width_wavelengths
        self.Fprime_normalisation = (
                (self.nu * self.nv) * self.delta_u_inv_rad**2
            )

        # Finv normalization
        self.dA_sr = p.sky_model_pixel_area_sr
        self.Finv_normalisation = self.dA_sr

        self.matrix_construction_methods_dictionary = {
            'T_Ninv_T':
                self.build_T_Ninv_T,
            'idft_array_1D':
                self.build_idft_array_1D,
            'idft_array_1D_WQ':
                self.build_idft_array_1D_WQ,
            'gridding_matrix_vis_ordered_to_chan_ordered':
                self.build_gridding_matrix_vis_ordered_to_chan_ordered,
            'gridding_matrix_chan_ordered_to_vis_ordered':
                self.build_gridding_matrix_chan_ordered_to_vis_ordered,
            'Fz':
                self.build_Fz,
            'multi_vis_idft_array_1D_WQ':
                self.build_multi_vis_idft_array_1D_WQ,
            'multi_vis_idft_array_1D':
                self.build_multi_vis_idft_array_1D,
            'idft_array':
                self.build_idft_array,
            'multi_chan_dft_array_noZMchan':
                self.build_multi_chan_dft_array_noZMchan,
            'dft_array':
                self.build_dft_array,
            'Fprime':
                self.build_Fprime,
            'multi_chan_idft_array_noZMchan':
                self.build_multi_chan_idft_array_noZMchan,
            'Finv':
                self.build_Finv,
            'Fprime_Fz':
                self.build_Fprime_Fz,
            'T':
                self.build_T,
            'Ninv':
                self.build_Ninv,
            'Ninv_T':
                self.build_Ninv_T,
            'block_T_Ninv_T':
                self.build_block_T_Ninv_T,
            'N':
                self.build_N,
            'multi_chan_nudft':
                self.build_multi_chan_nudft,
            'multi_chan_P':
                self.build_multi_chan_P,
            'phasor_matrix':
                self.build_phasor_matrix
            }

        # Check if beam_center is not None
        # If a beam_center is passed (assumed to be in units of
        # degrees) rename matrices to include beam offset
        if self.beam_center is not None:
            # 1. Update matrix_prerequisites_dictionary
            # Update prereqs for multi_chan_P, Finv, T, Ninv_T, T_Ninv_T
            self.beam_center_str =\
                '_beam_center_RA0+{:.2f}_DEC0+{:.2f}'.format(
                    self.beam_center[0], self.beam_center[1]
                    )
            self.beam_matrix_names = [
                'multi_chan_P', 'Finv', 'T',
                'Ninv_T', 'T_Ninv_T', 'block_T_Ninv_T']
            dependencies = {
                'multi_chan_P' + self.beam_center_str : None,
                'Finv' + self.beam_center_str : [
                    'phasor_matrix',
                    'multi_chan_nudft',
                    'multi_chan_P' + self.beam_center_str
                    ],
                'T' + self.beam_center_str : [
                    'Finv' + self.beam_center_str,
                    'Fprime_Fz'
                    ],
                'Ninv_T' + self.beam_center_str : [
                    'Ninv',
                    'T' + self.beam_center_str
                    ],
                'T_Ninv_T' + self.beam_center_str : [
                    'T' + self.beam_center_str,
                    'Ninv_T' + self.beam_center_str
                    ],
                'block_T_Ninv_T' + self.beam_center_str: [
                    'T_Ninv_T' + self.beam_center_str
                    ]
                }
            for matrix_name in self.beam_matrix_names:
                if matrix_name in self.matrix_prerequisites_dictionary.keys():
                    self.matrix_prerequisites_dictionary.pop(matrix_name)
                key = matrix_name + self.beam_center_str
                if dependencies[key] is not None:
                    self.matrix_prerequisites_dictionary[key] =\
                        dependencies[matrix_name + self.beam_center_str]

                self.matrix_construction_methods_dictionary[key] =\
                    self.matrix_construction_methods_dictionary.pop(
                        matrix_name
                    )

            print('Matrix names : {}'.format(
                self.matrix_construction_methods_dictionary.keys()))
            print('Dependencies : ')
            for key in self.matrix_prerequisites_dictionary.keys():
                print('{} : {}'.format(
                    key, self.matrix_prerequisites_dictionary[key]
                        )
                    )

    def load_prerequisites(self, matrix_name):
        """
        Load any prerequisites for matrix_name if they exist,
        or build them if they don't.
        """
        prerequisite_matrices_dictionary = {}
        print('About to check and load any prerequisites for', matrix_name)
        print('Checking for prerequisites')
        prerequisites_status = self.check_for_prerequisites(matrix_name)
        if prerequisites_status == {}:
            # If matrix has no prerequisites
            print(matrix_name, 'has no prerequisites. Continuing...')

        else:
            # Matrix has prerequisites that
            # need to be built and/or loaded
            for child_matrix, matrix_available\
                    in prerequisites_status.items():
                if matrix_available:
                    print(child_matrix, 'is available. Loading...')
                else:
                    print(child_matrix, 'is not available. Building...')
                    self.matrix_construction_methods_dictionary[child_matrix]()
                    # Re-check that that the matrix now exists and
                    # whether it is dense (hdf5; matrix_available=1)
                    # or sparse (npz; matrix_available=2)
                    matrix_available =\
                        self.check_if_matrix_exists(child_matrix)
                    print(child_matrix, 'is now available. Loading...')

                # Load prerequisite matrix into
                # prerequisite_matrices_dictionary
                if matrix_available == 1:
                    file_extension = '.h5'
                elif matrix_available == 2:
                    file_extension = '.npz'
                else:
                    file_extension = '.h5'
                file_path = (self.array_save_directory
                             + child_matrix
                             + file_extension)
                dataset_name = child_matrix
                start = time.time()
                data = self.read_data(file_path, dataset_name)
                prerequisite_matrices_dictionary[child_matrix] = data
                print('Time taken: {}'.format(time.time() - start))

        return prerequisite_matrices_dictionary

    def dot_product(self, matrix_A, matrix_B):
        """
        Calculate the dot product of matrix_A and matrix_B correctly
        whether either or both of A and B are sparse or dense.
        """
        matrix_A_is_sparse = sparse.issparse(matrix_A)
        matrix_B_is_sparse = sparse.issparse(matrix_B)
        if not (matrix_A_is_sparse or matrix_B_is_sparse):
            # Both matrices are dense numpy.ndarrays
            # Use np.dot to calculate the dot product
            AB = np.dot(matrix_A, matrix_B)
        else:
            # One of the matrices is sparse - need to use
            # python matrix syntax (i.e. * for dot product)
            # NOTE:
            # sparse * dense = dense
            # dense * sparse = dense
            # sparse * sparse = sparse
            print(matrix_A.shape)
            print(matrix_B.shape)
            AB = matrix_A * matrix_B
        return AB

    def convert_sparse_to_dense_matrix(self, matrix_A):
        """
        Convert scipy.sparse matrix to dense matrix.
        """
        matrix_A_is_sparse = sparse.issparse(matrix_A)
        if matrix_A_is_sparse:
            matrix_A_dense = matrix_A.todense()
        else:
            matrix_A_dense = matrix_A
        return matrix_A_dense

    def convert_sparse_matrix_to_dense_numpy_array(self, matrix_A):
        """
        Convert scipy.sparse matrix to dense numpy array.
        """
        matrix_A_dense = self.convert_sparse_to_dense_matrix(matrix_A)
        matrix_A_dense_np_array = np.array(matrix_A_dense)
        return matrix_A_dense_np_array

    def sd_block_diag(self, block_matrices_list):
        """
        Generate block diagonal matrix from
        blocks in block_matrices_list.
        """
        if p.use_sparse_matrices:
            return sparse.block_diag(block_matrices_list)
        else:
            return block_diag(*block_matrices_list)

    # Finv functions
    def build_dft_array(self):
        """
        Construct a block for `multi_chan_dft_array_noZMchan` which
        is a uniform DFT matrix that takes a rectilinear sky model
        in (l, m) space and transforms it a rectilinear (u, v) plane
        for a single frequency channel.

        Used to construct `Finv` if
        `p.include_instrumental_effects = False`.
        dft_array has shape (nuv, nx * ny).
        """
        matrix_name = 'dft_array'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        dft_array = DFT_Array_DFT_2D_ZM(self.nu, self.nv, self.nx, self.ny)
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(dft_array,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_multi_chan_dft_array_noZMchan(self):
        """
        Constructs a block-diagonal uniform DFT matrix which takes
        a rectilinear model sky vector in (l, m, f) space and transforms
        it to a rectilinear (u, v, f) data space.

        Used to construct `Finv` if
        `p.include_instrumental_effects = False`.
        multi_chan_dft_array_noZMchan has shape
        (nuv * nf, nx * ny * nf).
        """
        matrix_name = 'multi_chan_dft_array_noZMchan'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        multi_chan_dft_array_noZMchan =\
            self.sd_block_diag([pmd['dft_array'].T for i in range(self.nf)])
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(
            multi_chan_dft_array_noZMchan,
            self.array_save_directory,
            matrix_name,
            matrix_name)

    def build_phasor_matrix(self):
        """
        Construct phasor matrix which is multiplied elementwise into
        the visibility vector from Finv, constructed using unphased
        (u, v, w) coordinates, to produce phased visibilities.

        The phasor matrix is a diagonal matrix with the
        `e^i*phi(t, u, v)` phase elements on the diagonal.
        The phasor vector must be a part of the instrument
        model being used.

        NOTE:
            This function assumes that
            `use_nvis_nchan_nt_ordering = True`

        Used to construct `Finv`.
        phasor_matrix has shape (ndata, ndata).
        """
        matrix_name = 'phasor_matrix'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if p.use_sparse_matrices:
            phasor_matrix = sparse.diags(self.phasor_vector)
        else:
            phasor_matrix = np.diag(self.phasor_vector)
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(
            phasor_matrix,
            self.array_save_directory,
            matrix_name,
            matrix_name)

    def build_multi_chan_nudft(self):
        """
        Construct block-diagonal non-uniform DFT array from (l, m, f)
        to unphased (u, v, f) in the instrument model.

        If use_nvis_nt_nchan_ordering:
            model visibilities will be ordered (nvis*nt) per chan for
            all channels (this is the old default).
        If use_nvis_nchan_nt_ordering:
            model visibilities will be ordered (nvis*nchan) per time
            step for all time steps (this ordering is required when
            using a drift scan primary beam).

        Used to construct `Finv`.
        `multi_chan_nudft` has shape (ndata, npix * nf * nt).
        """
        matrix_name = 'multi_chan_nudft'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        nu_array_MHz = (
                p.nu_min_MHz
                + np.arange(p.nf)*p.channel_width_MHz
            )
        sampled_uvw_coords_m = self.uvw_multi_time_step_array_meters.copy()
        # Convert uv-coordinates from meters to wavelengths at frequency
        # chan_freq_MHz for all chan_freq_MHz in nu_array_MHz
        sampled_uvw_coords_wavelengths = \
            np.array([sampled_uvw_coords_m
                      / (p.speed_of_light / (chan_freq_MHz * 1.e6))
                      for chan_freq_MHz in nu_array_MHz])
        if p.use_nvis_nt_nchan_ordering:
            # Used if p.model_drift_scan_primary_beam = False

            # Get (l, m) coordinates from Healpix object
            ls_rad, ms_rad = self.hp.calc_lm_from_radec(
                self.hp.pointing_centers[self.hp.nt//2],
                self.hp.north_poles[self.hp.nt//2]
                )
            sampled_lm_coords_radians = np.vstack((ls_rad, ms_rad)).T

            multi_chan_nudft =\
                self.sd_block_diag(
                    [nuDFT_Array_DFT_2D_v2d0(
                        sampled_lm_coords_radians,
                        sampled_uvw_coords_wavelengths[
                            freq_i, 0, :, :
                        ].reshape(-1, 2))
                     for freq_i in range(p.nf)
                     ]
                    )
        elif p.use_nvis_nchan_nt_ordering:
            # This will be used if a drift scan primary
            # beam is included in the data model
            # (i.e. p.model_drift_scan_primary_beam=True)

            multi_chan_nudft = self.sd_block_diag([self.sd_block_diag([
                nuDFT_Array_DFT_2D_v2d0(
                    np.vstack(
                        self.hp.calc_lm_from_radec(
                            self.hp.pointing_centers[time_i],
                            self.hp.north_poles[time_i]
                            ) # gets (l(t), m(t))
                        ).T,
                    sampled_uvw_coords_wavelengths[
                        freq_i, time_i, :, :
                    ].reshape(-1, 2))
                    for freq_i in range(p.nf)])
                for time_i in range(p.nt)])

        # Multiply by sky model pixel area to get the
        # units of the model visibilities correct
        multi_chan_nudft *= self.Finv_normalisation

        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(multi_chan_nudft,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_multi_chan_P(self):
        """
        Constructs a stacked, block-diagonal matrix containing the beam
        amplitude of each (l(t), m(t), f) pixel in the sky model.
        Each block-diagonal matrix in the stack contains a block
        diagonal matrix with nf blocks that contain the beam amplitude
        at each (l(t_i), m(t_i), f_i).

        Used to construct `Finv`.
        `multi_chan_P` has shape (npix * nf * nt, nuv * nf).
        """
        matrix_name = 'multi_chan_P'
        if self.beam_center is not None:
            matrix_name += self.beam_center_str
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')

        if not p.model_drift_scan_primary_beam:
            # Needs to be updated to use HEALPix coordinates
            multi_chan_P = self.sd_block_diag([
                np.diag(
                    self.hp.get_beam_vals(
                        *self.hp.calc_lm_from_radec(
                            center=self.hp.pointing_centers[p.nt//2],
                            north=self.hp.north_poles[p.nt//2]
                            )
                        )
                    )
                for _ in range(p.nf)])
        else:
            # Model the time dependence of the primary beam pointing
            # for a drift scan (i.e. change in zenith angle with time
            # due to Earth rotation).

            # Need to change this from using vstack to block_diag
            # when reintroducing the time axis
            if not p.use_sparse_matrices:
                # Stack dense block diagonal
                # matrices using np.vstack
                # Use Healpix class functions
                multi_chan_P_drift_scan = np.vstack([
                    block_diag(*[
                        np.diag(
                            self.hp.get_beam_vals(
                                *self.hp.calc_lm_from_radec(
                                    center=self.hp.pointing_centers[time_i],
                                    north=self.hp.north_poles[time_i]
                                    )
                                )
                            )
                        for _ in range(p.nf)])
                    for time_i in range(p.nt)])
            else:
                # Stack spares block diagonal
                # matrices using sparse.vstack
                # Use Healpix class functions
                multi_chan_P_drift_scan = sparse.vstack([
                    sparse.block_diag([
                        sparse.diags(
                            self.hp.get_beam_vals(
                                *self.hp.calc_lm_from_radec(
                                    center=self.hp.pointing_centers[time_i],
                                    north=self.hp.north_poles[time_i]
                                    )
                                )
                            )
                        for _ in range(p.nf)])
                    for time_i in range(p.nt)])

            multi_chan_P = multi_chan_P_drift_scan

        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(multi_chan_P,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_Finv(self):
        """
        Construct a non-uniform DFT matrix that takes a vector of
        (l, m, f) syk model pixel amplitudes and:
          1. applies a beam per time and frequency via `multi_chan_P`
          2. transforms to insttrumentally sampled, unphased (u, v, f)
             coordinates from the instrument model
          3. applies a phasor vector from the instrument model to phase
             the visibilities to the central time step

        `Finv` has shape (ndata, nuv * nf).
        """
        matrix_name = 'Finv'
        if self.beam_center is not None:
            matrix_name += self.beam_center_str
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if p.include_instrumental_effects:
            if self.beam_center is None:
                Finv = self.dot_product(
                    pmd['multi_chan_nudft'],
                    pmd['multi_chan_P']
                    )
            else:
                Finv = self.dot_product(
                    pmd['multi_chan_nudft'],
                    pmd['multi_chan_P' + self.beam_center_str]
                    )
            Finv = self.dot_product(pmd['phasor_matrix'], Finv)
        else:
            Finv = pmd['multi_chan_dft_array_noZMchan']
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(Finv,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    # Fprime functions
    # Fprime will need to be remade using vstack to transform
    # to (l(t), m(t)) instead of a single set of (l, m) coords
    def build_idft_array(self):
        """
        Construct a block for `multi_chan_idft_array_noZMchan` which
        is a non-uniform DFT matrix that takes a rectilinear, model
        (u, v) plane and transforms it to HEALPix sky model (l, m)
        space for a single frequency channel.

        Used to construct `Fprime`.
        `idft_array` has shape (npix, nuv).
        """
        matrix_name = 'idft_array'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        # Get (l, m) coordinates from Healpix object
        ls_rad, ms_rad = self.hp.calc_lm_from_radec(
            self.hp.pointing_centers[p.nt//2],
            self.hp.north_poles[p.nt//2]
            )
        sampled_lm_coords_radians = np.vstack((ls_rad, ms_rad)).T

        idft_array = IDFT_Array_IDFT_2D_ZM(
            self.nu, self.nv,
            sampled_lm_coords_radians)
        idft_array *= self.Fprime_normalisation
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(idft_array,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_multi_chan_idft_array_noZMchan(self):
        """
        Construct a non-uniform, block-diagonal DFT matrix which
        takes a rectilinear (u, v, f) model and transforms it to
        HEALPix sky model (l, m, f) space.  Each block corresponds
        to a single frequency channel.

        Used to construct `Fprime`.
        `multi_chan_idft_array_noZMchan` has shape
        (npix * nf, nuv * nf).
        """
        matrix_name = 'multi_chan_idft_array_noZMchan'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        multi_chan_idft_array_noZMchan = self.sd_block_diag(
            [pmd['idft_array'].T for i in range(self.nf)]
            )
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(multi_chan_idft_array_noZMchan,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_Fprime(self):
        """
        Construct a non-uniform, block-diagonal DFT matrix which
        takes a rectilinear (u, v, f) model and transforms it to
        HEALPix sky model (l, m, f) space.

        `Fprime` has shape (npix * nf, nuv * nf).
        """
        matrix_name = 'Fprime'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        Fprime = pmd['multi_chan_idft_array_noZMchan']
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(Fprime,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    # Fz functions
    def build_idft_array_1D(self):
        """
        Construct a 1D LoS DFT matrix for a single (u, v) pixel
        with `nq = 0`.  Constructs one block within
        `multi_vis_idft_array_1D`.

        Used to construct `Fz` if `nq = 0`.
        `idft_array_1D` has shape (nf, neta).
        """
        matrix_name = 'idft_array_1D'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        idft_array_1D = IDFT_Array_IDFT_1D(self.nf, self.neta)
        idft_array_1D *= self.Fz_normalisation
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(idft_array_1D,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_multi_vis_idft_array_1D(self):
        """
        Construct a block diagonal matrix of 1D LoS DFT matrices for
        every (u, v) pixel in the model uv-plane.

        Used to construct `Fz` if `nq = 0`.
        `multi_vis_idft_array_1D` has shape (nuv * nf, nuv * neta).
        """
        matrix_name = 'multi_vis_idft_array_1D'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if p.fit_for_monopole:
            multi_vis_idft_array_1D = self.sd_block_diag(
                [pmd['idft_array_1D'] for i in range(self.nu*self.nv)]
                )
        else:
            multi_vis_idft_array_1D = self.sd_block_diag(
                [pmd['idft_array_1D'] for i in range(self.nu*self.nv-1)]
                )
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(multi_vis_idft_array_1D,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_idft_array_1D_WQ(self):
        """
        Construct a 1D LoS DFT matrix for a single (u, v) pixel
        with `nq > 0`.  Constructs one block within
        `multi_vis_idft_array_1D_WQ`.

        Used to construct `Fz` if `nq > 0`.
        `idft_array_1D_WQ` has shape (nf, neta + nq).
        """
        matrix_name = 'idft_array_1D_WQ'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        idft_array_1D_WQ = IDFT_Array_IDFT_1D_WQ(
            self.nf,
            self.neta,
            self.nq,
            npl=self.npl,
            nu_min_MHz=p.nu_min_MHz,
            channel_width_MHz=p.channel_width_MHz,
            beta=p.beta)
        idft_array_1D_WQ *= self.Fz_normalisation
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(idft_array_1D_WQ,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_multi_vis_idft_array_1D_WQ(self):
        """
        Construct a block diagonal matrix of 1D LoS DFT matrices
        with `nq > 0` for every (u, v) pixel in the model uv-plane.

        Used to construct `Fz` if `nq > 0`.
        `multi_vis_idft_array_1D_WQ` has shape
        (nuv * nf, nuv * (neta + nq))
        """
        matrix_name = 'multi_vis_idft_array_1D_WQ'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if p.fit_for_monopole:
            multi_vis_idft_array_1D_WQ = self.sd_block_diag(
                [pmd['idft_array_1D_WQ'].T for i in range(self.nu*self.nv)]
                )
        else:
            multi_vis_idft_array_1D_WQ = self.sd_block_diag(
                [pmd['idft_array_1D_WQ'].T for i in range(self.nu*self.nv - 1)]
                )
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(multi_vis_idft_array_1D_WQ,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_gridding_matrix_vis_ordered_to_chan_ordered(self):
        """
        Constructs a matrix which takes a vector that is vis ordered:
          - the first `nf` (`nf + nq` if `nq > 0`) entries correspond
            to the spectrum of a single model (u, v) pixel
          - the second `nf` (`nf + nq` if `nq > 0`) entries correspond
            to the spectrum of the next model (u, v) pixel
          - etc.
        and converts it to chan ordered:
          - the first 'nuv' (`nuv + nq` if `nq > 0`) entries correspond
            to the values of the model (u, v) plane at the zeroth
            frequency channel
          - the second 'nuv' (`nuv + nq` if `nq > 0`) entries correspond
            to the values of the model (u, v) plane at the first
            frequency channel
          - etc.

        `gridding_matrix_vis_ordered_to_chan_ordered` has shape
        (nuv * (neta + nq), nuv * (neta + nq)).
        """
        matrix_name = 'gridding_matrix_vis_ordered_to_chan_ordered'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        gridding_matrix_vis_ordered_to_chan_ordered =\
            generate_gridding_matrix_vis_ordered_to_chan_ordered(self.nu,
                                                                 self.nv,
                                                                 self.nf)
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(gridding_matrix_vis_ordered_to_chan_ordered,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_Fz(self):
        """
        Constructs a 1D LoS DFT matrix which takes a vis ordered
        (u, v, eta) space data vector and transforms it to a chan
        ordered (u, v, f) space data vector.

        `Fz` has shape (nuv * nf, nuv * (neta + nq)).
        """
        matrix_name = 'Fz'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if self.nq == 0:
            Fz = self.dot_product(
                pmd['gridding_matrix_vis_ordered_to_chan_ordered'],
                pmd['multi_vis_idft_array_1D'])
        else:
            Fz = self.dot_product(
                pmd['gridding_matrix_vis_ordered_to_chan_ordered'],
                pmd['multi_vis_idft_array_1D_WQ'])
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(Fz,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_Fprime_Fz(self):
        """
        Matrix product of `Fprime * Fz` with
        shape (npix, nuv * (neta + nq)).
        """
        matrix_name = 'Fprime_Fz'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        Fprime_Fz = self.dot_product(pmd['Fprime'], pmd['Fz'])
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(
            Fprime_Fz,
            self.array_save_directory,
            matrix_name,
            matrix_name)

    def build_gridding_matrix_chan_ordered_to_vis_ordered(self):
        """
        Constructs a matrix which takes a vector that is chan ordered:
          - the first 'nuv' entries correspond to the values of the
            model (u, v) plane at the zeroth frequency channel
          - the second 'nuv' entries correspond to the values of the
            model (u, v) plane at the first frequency channel
          - etc.
        and converts it to vis ordered:
          - the first `nf` (`nf + nq` if `nq > 0`) entries correspond
            to the spectrum of a single model (u, v) pixel
          - the second `nf` (`nf + nq` if `nq > 0`) entries correspond
            to the spectrum of the next model (u, v) pixel
          - etc.

        `gridding_matrix_chan_ordered_to_vis_ordered` has shape
        (nuv * (neta + nq), nuv * (neta + nq)).
        """
        matrix_name = 'gridding_matrix_chan_ordered_to_vis_ordered'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        # NOTE: taking the transpose reverses the gridding. This is what
        # happens in dbar where Fz.conjugate().T is multiplied by d and
        # the gridding_matrix_vis_ordered_to_chan_ordered.conjugate().T
        # part of Fz transforms d from chan-ordered initially to
        # vis-ordered.
        # NOTE: conjugate does nothing to the gridding matrix component
        # of Fz, which is real, it only transforms the 1D IDFT to a DFT)
        gridding_matrix_chan_ordered_to_vis_ordered =\
            pmd['gridding_matrix_vis_ordered_to_chan_ordered'].T
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(gridding_matrix_chan_ordered_to_vis_ordered,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    # Covariance matrix functions
    def build_Ninv(self):
        """
        Constructs a sparse diagonal inverse covariance matrix.
        Each diagonal component contains an estimate of the
        `1 / noise_amplitude**2` in the data vector
        at the index of the diagonal entry, i.d. data[i] for Ninv[i, i].

        `Ninv` has shape (ndata, ndata).
        """
        matrix_name = 'Ninv'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if p.include_instrumental_effects:
            if p.use_nvis_nt_nchan_ordering:
                # This array is channel_ordered and the covariance
                # matrix assumes a channel_ordered data set
                # (this vector should be re-ordered if
                # the data is in a different order)
                baseline_redundancy_array =\
                    self.baseline_redundancy_array_vectorised
                s_size = self.n_vis * self.nf
                multifreq_baseline_redundancy_array = np.array(
                    [baseline_redundancy_array for i in range(p.nf)]
                    ).flatten()

                # RMS drops as the square root of
                # the number of redundant samples
                sigma_redundancy = (
                        self.sigma
                        / multifreq_baseline_redundancy_array**0.5)

                sigma_squared_array = (
                        np.ones(s_size) * sigma_redundancy**2
                        + 0j*np.ones(s_size)*sigma_redundancy**2
                    )
            elif p.use_nvis_nchan_nt_ordering:
                if self.effective_noise is None:
                    red_array_time_vis_shaped =\
                        self.baseline_redundancy_array_time_vis_shaped
                    baseline_redundancy_array_time_freq_vis = np.array([
                        [red_array_vis for i in range(p.nf)]
                        for red_array_vis in red_array_time_vis_shaped
                        ]).flatten()
                    s_size = self.n_vis * self.nf

                    # RMS drops as the square root of
                    # the number of redundant samples
                    sigma_redundancy = (
                            self.sigma
                            / baseline_redundancy_array_time_freq_vis**0.5)

                    sigma_squared_array = (
                            np.ones(s_size) * sigma_redundancy**2
                            + 0j*np.ones(s_size)*sigma_redundancy**2
                        )
                else:
                    sigma_squared_array = (
                            np.abs(self.effective_noise)**2
                            + 0j*np.abs(self.effective_noise)**2
                        )
        else:
            s_size = (self.nu*self.nv - 1) * self.nf
            sigma_squared_array = (
                    np.ones(s_size)*self.sigma**2
                    + 0j*np.ones(s_size)*self.sigma**2
                )

        if p.use_sparse_matrices:
            Ninv = sparse.diags(1./sigma_squared_array)
        else:
            Ninv = np.diag(1./sigma_squared_array)
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(Ninv,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_N(self):
        """
        Constructs a sparse diagonal inverse covariance matrix.
        Each diagonal component contains an estimate of the
        `noise_amplitude**2` in the data vector at the index of the
        diagonal entry, i.d. data[i] for N[i, i].

        `N` has shape (ndata, ndata).
        """
        matrix_name = 'N'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if p.include_instrumental_effects:
            if p.use_nvis_nt_nchan_ordering:
                # This array is channel_ordered and the covariance
                # matrix assumes a channel_ordered data set
                # (this vector should be re-ordered if
                # the data is in a different order)
                baseline_redundancy_array = \
                    self.baseline_redundancy_array_vectorised
                s_size = self.n_vis * self.nf
                multifreq_baseline_redundancy_array = np.array(
                    [baseline_redundancy_array for i in range(p.nf)]
                    ).flatten()

                # RMS drops as the square root of
                # the number of redundant samples
                sigma_redundancy = (
                        self.sigma
                        / multifreq_baseline_redundancy_array ** 0.5)

                sigma_squared_array = (
                        np.ones(s_size) * sigma_redundancy ** 2
                        + 0j * np.ones(s_size) * sigma_redundancy ** 2
                    )
            elif p.use_nvis_nchan_nt_ordering:
                if self.effective_noise is None:
                    red_array_time_vis_shaped = \
                        self.baseline_redundancy_array_time_vis_shaped
                    baseline_redundancy_array_time_freq_vis = np.array([
                        [red_array_vis for i in range(p.nf)]
                        for red_array_vis in red_array_time_vis_shaped
                        ]).flatten()
                    s_size = self.n_vis * self.nf

                    # RMS drops as the square root of
                    # the number of redundant samples
                    sigma_redundancy = (
                            self.sigma
                            / baseline_redundancy_array_time_freq_vis ** 0.5)

                    sigma_squared_array = (
                            np.ones(s_size) * sigma_redundancy ** 2
                            + 0j * np.ones(s_size) * sigma_redundancy ** 2
                        )
                else:
                    sigma_squared_array = (
                            np.abs(self.effective_noise) ** 2
                            + 0j * np.abs(self.effective_noise) ** 2
                        )
        else:
            s_size = (self.nu * self.nv - 1) * self.nf
            sigma_squared_array = (
                    np.ones(s_size) * self.sigma ** 2
                    + 0j * np.ones(s_size) * self.sigma ** 2
                )

        if p.use_sparse_matrices:
            N = sparse.diags(sigma_squared_array)
        else:
            N = np.diag(sigma_squared_array)
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(N,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    # T functions
    def build_T(self):
        """
        Construct `T = Finv * Fprime * Fz` which takes a model
        (u, v, eta) space data vector and transforms it to:
          1. model (u, v, f) space via Fz
          2. model (l, m, f) HEALPix space via Fprime
          3. data (u, v, f) space via Finv

        T has shape (ndata, nuv * (neta + nq)).
        """
        matrix_name = 'T'
        if self.beam_center is not None:
            matrix_name += self.beam_center_str
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if self.beam_center is None:
            T = self.dot_product(pmd['Finv'],
                                 pmd['Fprime_Fz'])
        else:
            T = self.dot_product(pmd['Finv' + self.beam_center_str],
                                 pmd['Fprime_Fz'])
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(
            T,
            self.array_save_directory,
            matrix_name,
            matrix_name)

    def build_Ninv_T(self):
        """
        Matrix product of Ninv * T.  Can be used to take a (u, v, eta)
        data vector and compute a noise weighted vector in data space.

        Ninv_T has shape (ndata, nuv * (neta + nq)).
        """
        matrix_name = 'Ninv_T'
        if self.beam_center is not None:
            matrix_name += self.beam_center_str
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if self.beam_center is None:
            Ninv_T = self.dot_product(pmd['Ninv'],
                                      pmd['T'])
        else:
            Ninv_T = self.dot_product(pmd['Ninv'],
                                      pmd['T' + self.beam_center_str])
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(
            Ninv_T,
            self.array_save_directory,
            matrix_name,
            matrix_name)

    def build_T_Ninv_T(self):
        """
        Matrix product of T.conjugate().T * Ninv * T
        with shape (nuv * (neta + nq), nuv * (neta + nq)).
        """
        matrix_name = 'T_Ninv_T'
        if self.beam_center is not None:
            matrix_name += self.beam_center_str
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if self.beam_center is None:
            T_Ninv_T = self.dot_product(
                pmd['T'].conjugate().T,
                pmd['Ninv_T']
                )
        else:
            T_Ninv_T = self.dot_product(
                pmd['T' + self.beam_center_str].conjugate().T,
                pmd['Ninv_T' + self.beam_center_str]
                )
        # T_Ninv_T needs to be dense to pass to the GPU (note: if
        # T_Ninv_T is already a dense / a numpy array it will be
        # returned unchanged)
        T_Ninv_T = self.convert_sparse_matrix_to_dense_numpy_array(T_Ninv_T)
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(T_Ninv_T,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_block_T_Ninv_T(self):
        matrix_name = 'block_T_Ninv_T'
        if self.beam_center is not None:
            matrix_name += self.beam_center_str
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if p.fit_for_monopole:
            self.nuv = (self.nu*self.nv)
        else:
            self.nuv = (self.nu*self.nv - 1)
        if self.beam_center is None:
            block_T_Ninv_T = np.array(
                [np.hsplit(block,self.nuv)
                 for block in np.vsplit(pmd['T_Ninv_T'],self.nuv)]
                )
        else:
            block_T_Ninv_T = np.array(
                [np.hsplit(block, self.nuv)
                 for block in np.vsplit(
                    pmd['T_Ninv_T' + self.beam_center_str],
                    self.nuv
                    )
                 ]
                )
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(block_T_Ninv_T,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_matrix_if_it_doesnt_already_exist(self, matrix_name):
        matrix_available = self.check_if_matrix_exists(matrix_name)
        if not matrix_available:
            self.matrix_construction_methods_dictionary[matrix_name]()

    def prepare_matrix_stack_for_deletion(
            self, src, overwrite_existing_matrix_stack):
        if overwrite_existing_matrix_stack:
            if src[-1] == '/':
                src = src[:-1]
            head, tail = os.path.split(src)
            dst = os.path.join(head, 'delete_'+tail)
            print('Archiving existing matrix stack to:', dst)
            del_error_flag = 0
            try:
                shutil.move(src, dst)
            except:
                print('Archive path already existed.'
                      ' Deleting the previous archive.')
                self.delete_old_matrix_stack(dst, 'y')
                self.prepare_matrix_stack_for_deletion(
                    self.array_save_directory,
                    self.overwrite_existing_matrix_stack)
            return dst

    def delete_old_matrix_stack(
            self, path_to_old_matrix_stack, confirm_deletion):
        if (confirm_deletion.lower() == 'y'
                or confirm_deletion.lower() == 'yes'):
            shutil.rmtree(path_to_old_matrix_stack)
        else:
            print('Prior matrix tree archived but not deleted.'
                  ' \nPath to archive:', path_to_old_matrix_stack)

    def build_minimum_sufficient_matrix_stack(self, **kwargs):
        # ===== Defaults =====
        default_overwrite_existing_matrix_stack = False
        # Set to true when submitting to cluster
        default_proceed_without_overwrite_confirmation = False

        # ===== Inputs =====
        self.overwrite_existing_matrix_stack = kwargs.pop(
            'overwrite_existing_matrix_stack',
            default_overwrite_existing_matrix_stack)
        self.proceed_without_overwrite_confirmation = kwargs.pop(
            'proceed_without_overwrite_confirmation',
            default_proceed_without_overwrite_confirmation)

        # Prepare matrix directory
        matrix_stack_dir_exists =  os.path.exists(self.array_save_directory)
        if matrix_stack_dir_exists:
            dst = self.prepare_matrix_stack_for_deletion(
                self.array_save_directory,
                self.overwrite_existing_matrix_stack)
        # Build matrices
        if self.beam_center is None:
            self.build_matrix_if_it_doesnt_already_exist('T_Ninv_T')
            self.build_matrix_if_it_doesnt_already_exist('block_T_Ninv_T')
        else:
            self.build_matrix_if_it_doesnt_already_exist(
                'T_Ninv_T' + self.beam_center_str
                )
            self.build_matrix_if_it_doesnt_already_exist(
                'block_T_Ninv_T' + self.beam_center_str
                )
        self.build_matrix_if_it_doesnt_already_exist('N')
        if matrix_stack_dir_exists and self.overwrite_existing_matrix_stack:
            if not self.proceed_without_overwrite_confirmation:
                confirm_deletion = raw_input(
                    'Confirm deletion of archived matrix stack? y/n\n')
            else:
                print('Deletion of archived matrix stack has'
                      ' been pre-confirmed. Continuing...')
                confirm_deletion = 'y'
            self.delete_old_matrix_stack(dst, confirm_deletion)

        print('Matrix stack complete')
