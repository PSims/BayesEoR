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
    IDFT_Array_IDFT_2D_ZM, IDFT_Array_IDFT_1D,\
    generate_gridding_matrix_vis_ordered_to_chan_ordered,\
    IDFT_Array_IDFT_1D_WQ,\
    nuDFT_Array_DFT_2D_v2d0,\
    idft_array_idft_1d_sh, IDFT_Array_IDFT_2D_ZM_SH
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
            'multi_chan_dft_array_noZMchan': ['dft_array'],
            'Fprime_Fz': ['Fprime', 'Fz'],
            'T': ['Finv', 'Fprime_Fz'],
            'Ninv_T': ['Ninv', 'T'],
            'T_Ninv_T': ['T', 'Ninv_T'],
            'block_T_Ninv_T': ['T_Ninv_T'],
            'Fprime': ['multi_chan_idft_array_noZMchan'],
            'multi_chan_idft_array_noZMchan': ['idft_array']
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
                prerequisites_status[child_matrix] = matrix_available
        return prerequisites_status

    def check_if_matrix_exists(self, matrix_name):
        """
        Check is hdf5 or npz file with `matrix_name` exists.

        Parameters
        ----------
        matrix_name : str
            Name of matrix.

        Returns
        -------
        matrix_available : int
            If matrix exists, 1 or 2 if matrix is an hdf5 or npz file,
            respectively.
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
                    p.use_sparse_matrices = True
            else:
                matrix_available = 0
        return matrix_available

    def create_directory(self, directory):
        """
        Create output directory if it doesn't already exist.

        Parameters
        ----------
        directory : str
            Name of directory.

        """
        if not os.path.exists(directory):
            print('Directory not found:\n\n' + directory + "\n")
            print('Creating required directory structure..')
            os.makedirs(directory)
        return 0

    def output_data(
            self, output_array, output_directory, file_name, dataset_name):
        """
        Check if the data is an array or sparse matrix and call the
        corresponding method to output to HDF5 or npz.

        Parameters
        ----------
        output_array : array
            Array to be written to disk.
        output_directory : str
            Directory in which to write `output_array`.
        file_name : str
            Filename to use for `output_array`.
        dataset_name : str
            If saving as hdf5, the key used to access `output_array`.

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
        Write array to HDF5 file.

        Parameters
        ----------
        output_array : array
            Array to be written to disk.
        output_directory : str
            Directory in which to write `output_array`.
        file_name : str
            Filename to use for `output_array`.
        dataset_name : str
            If saving as hdf5, the key used to access `output_array`.

        """
        start = time.time()
        self.create_directory(output_directory)
        output_path = '/'.join((output_directory, file_name))
        print('Writing data to', output_path)
        with h5py.File(output_path, 'w') as hf:
            hf.create_dataset(dataset_name, data=output_array)
        print('Time taken: {}'.format(time.time() - start))
        return 0

    def output_sparse_matrix_to_npz(
            self, output_array, output_directory, file_name, dataset_name):
        """
        Write sparse matrix to npz (note: to maintain sparse matrix
        attributes need to use sparse.save_npz rather than np.savez).

        Parameters
        ----------
        output_array : array
            Array to be written to disk.
        output_directory : str
            Directory in which to write `output_array`.
        file_name : str
            Filename to use for `output_array`.
        dataset_name : str
            If saving as hdf5, the key used to access `output_array`.

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

        Parameters
        ----------
        file_path : str
            Path to array file.
        dataset_name : str
            If reading an hdf5 file, the key used to access the dataset.

        """
        data = self.read_data(file_path, dataset_name)
        data = self.convert_sparse_matrix_to_dense_numpy_array(data)
        return data

    def read_data(self, file_path, dataset_name):
        """
        Check if the data is an array (.h5) or sparse matrix (.npz)
        and call the corresponding method to read it in.

        Parameters
        ----------
        file_path : str
            Path to array file.
        dataset_name : str
            If reading an hdf5 file, the key used to access the dataset.

        """
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
        Read array from HDF5 file.

        Parameters
        ----------
        file_path : str
            Path to array file.
        dataset_name : str
            If reading an hdf5 file, the key used to access the dataset.

        """
        with h5py.File(file_path, 'r') as hf:
            data = hf[dataset_name][:]
        return data

    def read_data_from_npz(self, file_path, dataset_name):
        """
        Read sparse matrix from npz (note: to maintain sparse matrix
        attributes need to use sparse.load_npz rather than np.loadz).

        Parameters
        ----------
        file_path : str
            Path to array file.
        dataset_name : str
            If reading an hdf5 file, the key used to access the dataset.

        """
        data = sparse.load_npz(file_path)
        return data


class BuildMatrices(BuildMatrixTree):
    """
    Child class used to create a minimum sufficient matrix stack
    using BayesEoR.Linalg matrix creation functions.

    Parameters
    ----------
    array_save_directory : str
        Path to the directory where arrays will be saved.
    nu : int
        Number of pixels on a side for the u axis in the model uv-plane.
    nv : int
        Number of pixels on a side for the v axis in the model uv-plane.
    n_vis : int
        Number of visibilities per channel, i.e. number of redundant
        baselines * number of time steps.
    neta : int
        Number of Line of Sight (LoS) Fourier modes.
    nf : int
        Number of frequency channels.
    nt : int
        Number of times.
    nq : int
        Number of quadratic modes in the Large Spectral Scale Model (LSSM).
    npl : int
        Number of power law coefficients which replace quadratic modes in
        the LSSM.
    sigma : float
        Expected noise amplitude in the data vector = signal + noise.
    uvw_array_m : :class:`numpy.ndarray`
        Array containing the (u(t), v(t), w(t)) coordinates of the instrument
        model with shape (nt, nbls, 3).
    uvw_array_m_vec : :class:`numpy.ndarray`
        Reshaped `uvw_array_m` with shape (nt * nbls, 3).
        Each set of nbls entries contain the (u, v, w) coordinates for a
        single integration.
    bl_red_array : :class:`numpy.ndarray`
        Array containing the number of redundant baselines at each
        (u(t), v(t), w(t)) in the instrument model with shape (nt, nbls, 1).
    bl_red_array_vec : :class:`numpy.ndarray`
        Reshaped `bl_red_array` with shape
        (nt * nbls, 1).  Each set of nbls entries contain the redundancy of
        each (u, v, w) for a single integration.
    phasor_vec : :class:`numpy.ndarray`
        Array with shape (ndata,) that contains the phasor term used to phase
        visibilities after performing the nuDFT from HEALPix (l, m, f) to
        instrumentally sampled, unphased (u, v, f).
    fov_ra_deg : float
        Field of view in degrees of the RA axis of the sky model.
    fov_dec_deg : float
        Field of view in degrees of the DEC axis of the sky model.
    nside : int
        HEALPix nside parameter.
    telescope_latlonalt : tuple
        The latitude, longitude, and altitude of the telescope in degrees,
        degrees, and meters, respectively.
    central_jd : float
        Central time step of the observation in JD2000 format.
    int_time : float
        Integration time in seconds.
    beam_type : {'uniform', 'gaussian', 'airy'}
        Beam type to use.
    beam_peak_amplitude : float
        Peak amplitude of the beam.
    beam_center : tuple of floats
        Beam center in (RA, DEC) coordinates and units of degrees.  Assumed to
        be an tuple of offsets along the RA and DEC axes relative to the
        pointing center of the sky model determined from the instrument model
        parameters `telescope_latlonalt` and `central_jd`.
    fwhm_deg : float
        Full Width at Half Maximum (FWHM) of the beam if using a Gaussian beam,
        or the effective FWHM of the main lobe of an Airy beam from which the
        diameter of the aperture is calculated.
    antenna_diameter : float
        Antenna (aperture) diameter in meters..  Used in the calculation of an
        Airy beam pattern or when using a Gaussian beam with a FWHM that varies
        as a function of frequency.  The FWHM evolves according to the
        effective FWHM of the main lobe of an Airy beam.
    effective_noise : :class:`numpy.ndarray`
        If the data vector being analyzed contains signal + noise, the
        effective_noise vector contains the estimate of the noise in the data
        vector.  Must have the shape and ordering of the data vector,
        i.e. (ndata,).
    delta_eta_iHz : float
        Fourier mode spacing along the eta (line of sight, frequency) axis in
        inverse Hz.
    delta_u_irad : float
        Fourier mode spacing along the u axis in inverse radians of the
        model uv-plane.
    delta_v_irad : float
        Fourier mode spacing along the v axis in inverse radians of the
        model uv-plane.
    use_shg : bool, optional
        If `True`, use the SubHarmonic Grid (SHG) in the model uv-plane.
    fit_for_shg_amps : bool, optional
        if `True`, fit explicitly for the amplitudes of the individual SHG
        pixels per frequency.
    nu_sh : int, optional
        Number of pixels on a side for the u-axis in the subharmonic model
        uv-plane.
    nv_sh : int, optional
        Number of pixels on a side for the v-axis in the subharmonic model
        uv-plane.
    nq_sh : int, optional
        Number of large spectral scale modes for each pixel in the subharmonic
        grid.
    npl_sh : int, optional
        Number of power law coefficients used in the large spectral scale model
        for each pixel in the subharmonic grid.

    """
    def __init__(self, array_save_directory, nu, nv,
                 n_vis, neta, nf, nq, sigma, **kwargs):
        super(BuildMatrices, self).__init__(array_save_directory)

        # ===== Defaults =====
        default_npl = 0

        # ===== Inputs =====
        self.npl = kwargs.pop('npl', default_npl)
        if p.include_instrumental_effects:
            self.uvw_array_m = kwargs.pop('uvw_array_m')
            self.bl_red_array = kwargs.pop('bl_red_array')
            self.bl_red_array_vec = kwargs.pop('bl_red_array_vec')
            self.phasor_vec = kwargs.pop('phasor_vec')
            self.beam_type = kwargs.pop('beam_type')
            self.beam_peak_amplitude = kwargs.pop('beam_peak_amplitude')
            self.beam_center = kwargs.pop('beam_center', None)
            self.fwhm_deg = kwargs.pop('fwhm_deg')
            self.antenna_diameter = kwargs.pop('antenna_diameter', None)
            self.effective_noise = kwargs.pop('effective_noise', None)

            # Set up Healpix instance
            self.hp = Healpix(
                fov_ra_deg=p.fov_ra_deg,
                fov_dec_deg=p.fov_dec_deg,
                nside=p.nside,
                telescope_latlonalt=p.telescope_latlonalt,
                central_jd=p.central_jd,
                nt=p.nt,
                int_time=p.integration_time_minutes * 60,
                beam_type=self.beam_type,
                peak_amp=self.beam_peak_amplitude,
                fwhm_deg=self.fwhm_deg,
                diam=self.antenna_diameter
                )

        # Set necessary / useful parameter values
        self.nu = nu
        self.nv = nv
        self.n_vis = n_vis
        self.neta = neta
        self.nf = nf
        self.nq = nq
        self.sigma = sigma
        # SHG params
        self.use_shg = kwargs.pop('use_shg', False)
        self.fit_for_shg_amps = kwargs.pop('fit_for_shg_amps', False)
        self.nu_sh = kwargs.pop('nu_sh', 0)
        self.nv_sh = kwargs.pop('nv_sh', 0)
        self.nq_sh = kwargs.pop('nq_sh', 0)
        self.npl_sh = kwargs.pop('npl_sh', 0)

        # Fz normalization
        self.delta_eta_iHz = kwargs.pop('delta_eta_iHz')
        self.Fz_normalization = self.nf * self.delta_eta_iHz

        # Fprime normalization
        self.delta_u_irad = kwargs.pop('delta_u_irad')
        self.delta_v_irad = kwargs.pop('delta_v_irad')
        self.Fprime_normalization = (
                self.nu * self.nv * self.delta_u_irad * self.delta_v_irad
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

        if self.use_shg:
            # Add SHG matrices to matrix calculations
            self.matrix_prerequisites_dictionary.update({
                'multi_vis_idft_array_1D': [
                    'idft_array_1D', 'idft_array_1d_sh'
                    ],
                'multi_vis_idft_array_1D_WQ': [
                    'idft_array_1D_WQ', 'idft_array_1d_sh'
                    ],
                'Fprime': [
                    'multi_chan_idft_array_noZMchan', 'idft_array_sh'
                    ]
            })
            self.matrix_construction_methods_dictionary.update({
                'idft_array_1d_sh': self.build_idft_array_1d_sh,
                'idft_array_sh': self.build_idft_array_sh
            })

    def load_prerequisites(self, matrix_name):
        """
        Load any prerequisites for matrix_name if they exist,
        or build them if they don't.

        Parameters
        ----------
        matrix_name : str
            Name of matrix.

        Returns
        -------
        prerequisite_matrices_dictionary : dict
            Dictionary containing any and all loaded matrix prerequisites.
        """
        prerequisite_matrices_dictionary = {}
        print('About to check and load any prerequisites for', matrix_name)
        print('Checking for prerequisites')
        prerequisites_status = self.check_for_prerequisites(matrix_name)
        if prerequisites_status == {}:
            print(matrix_name, 'has no prerequisites. Continuing...')
        else:
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

    def dot_product(self, matrix_a, matrix_b):
        """
        Calculate the dot product of matrix_a and matrix_b correctly
        whether either or both of A and B are sparse or dense.

        Parameters
        ----------
        matrix_a : array
            First argument.
        matrix_b : array
            Second argument.

        Returns
        -------
        ab : array
            Matrix dot product of `matrix_a` and `matrix_b`.

        Notes
        -----
        For dot products of sparse and dense matrices:
        * dot(sparse, dense) = dense
        * dot(dense, sparse) = dense
        * dot(sparse, sparse) = sparse
        """
        matrix_a_is_sparse = sparse.issparse(matrix_a)
        matrix_b_is_sparse = sparse.issparse(matrix_b)
        print(matrix_a.shape)
        print(matrix_b.shape)
        if not (matrix_a_is_sparse or matrix_b_is_sparse):
            ab = np.dot(matrix_a, matrix_b)
        else:
            ab = matrix_a * matrix_b
        return ab

    def convert_sparse_to_dense_matrix(self, matrix_a):
        """
        Convert scipy.sparse matrix to dense matrix.

        Parameters
        ----------
        matrix_a : :class:`scipy.sparse`
            Sparse matrix.

        Returns
        -------
        matrix_a_dense : :class:`numpy.matrix`
            Dense representation of `matrix_a`.
        """
        matrix_a_is_sparse = sparse.issparse(matrix_a)
        if matrix_a_is_sparse:
            matrix_a_dense = matrix_a.todense()
        else:
            matrix_a_dense = matrix_a
        return matrix_a_dense

    def convert_sparse_matrix_to_dense_numpy_array(self, matrix_a):
        """
        Convert scipy.sparse matrix to dense numpy array.

        Parameters
        ----------
        matrix_a : :class:`scipy.sparse`
            Sparse matrix.

        Returns
        -------
        matrix_a_dense_np_array : :class:`numpy.ndarray`
            Dense representation of `matrix_a`.
        """
        matrix_a_dense = self.convert_sparse_to_dense_matrix(matrix_a)
        matrix_a_dense_np_array = np.array(matrix_a_dense)
        return matrix_a_dense_np_array

    def sd_block_diag(self, block_matrices_list):
        """
        Generate a block diagonal matrix from blocks in `block_matrices_list`.

        Parameters
        ----------
        block_matrices_list : list
            List of input matrices.

        Returns
        -------
        block_diag_matrix : array
            If ``p.use_sparse_matrices = True``, return a sparse matrix.
            Otherwise, return a dense numpy array.

        """
        if p.use_sparse_matrices:
            block_diag_matrix = sparse.block_diag(block_matrices_list)
        else:
            block_diag_matrix = block_diag(*block_matrices_list)
        return block_diag_matrix

    def sd_vstack(self, matrices_list):
        """
        Generate a vertically stacked matrix from a list of matrices
        in `matrices_list`.

        Parameters
        ----------
        matrices_list : list
            List of input matrices.

        Returns
        -------
        vstack_matrix : array
            If ``p.use_sparse_matrices = True``, return a sparse matrix.
            Otherwise, return a dense numpy array.

        """
        if p.use_sparse_matrices:
            vstack_matrix = sparse.vstack(matrices_list)
        else:
            vstack_matrix = np.vstack(matrices_list)
        return vstack_matrix

    def sd_hstack(self, matrices_list):
        """
        Generate a horizontally stacked matrix from a list of matrices
        in `matrices_list`.

        Parameters
        ----------
        matrices_list : list
            List of input matrices.

        Returns
        -------
        hstack_matrix : array
            If ``p.use_sparse_matrices = True``, return a sparse matrix.
            Otherwise, return a dense numpy array.

        """
        if p.use_sparse_matrices:
            hstack_matrix = sparse.hstack(matrices_list)
        else:
            hstack_matrix = np.hstack(matrices_list)
        return hstack_matrix

    def sd_diags(self, diagonal_vals):
        """
        Generate a diagonal matrix from a list of entries in `diagonal_vals`.

        Parameters
        ----------
        diagonal_vals : array
            Input values to be placed on the matrix diagonal.

        Returns
        -------
        diagonal_matrix : array
            If ``p.use_sparse_matrices = True``, return a sparse matrix.
            Otherwise, return a dense numpy array.

        """
        if p.use_sparse_matrices:
            diagonal_matrix = sparse.diags(diagonal_vals)
        else:
            diagonal_matrix = np.diag(diagonal_vals)
        return diagonal_matrix

    # Finv functions
    def build_phasor_matrix(self):
        """
        Construct a phasor matrix which is multiplied elementwise into the
        visibility vector from Finv, constructed using unphased (u, v, w)
        coordinates, to produce phased visibilities.

        The phasor matrix is a diagonal matrix with the `e^(i*phi(t, u, v))`
        phase elements on the diagonal.  The phasor vector must be a part of
        the instrument model being used.

        Notes
        -----
        * Used to construct `Finv`.
        * phasor_matrix has shape (ndata, ndata).
        * This function assumes that `use_nvis_nchan_nt_ordering = True`.

        """
        matrix_name = 'phasor_matrix'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if p.use_sparse_matrices:
            phasor_matrix = sparse.diags(self.phasor_vec)
        else:
            phasor_matrix = np.diag(self.phasor_vec)
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(
            phasor_matrix,
            self.array_save_directory,
            matrix_name,
            matrix_name)

    def build_multi_chan_nudft(self):
        """
        Construct block-diagonal non-uniform DFT array from
        (l(t), m(t), n(t), f) to unphased (u, v, f) from the instrument model.

        Notes
        -----
        * If ``use_nvis_nt_nchan_ordering = True``: model visibilities will be
          ordered (nvis*nt) per chan for all channels (old default).
        * If ``use_nvis_nchan_nt_ordering = True``: model visibilities will be
          ordered (nvis*nchan) per time step for all time steps.  This ordering
          is required when using a drift scan primary beam (current default).
        * Used to construct `Finv`.
        * `multi_chan_nudft` has shape (ndata, npix * nf * nt).

        """
        matrix_name = 'multi_chan_nudft'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        nu_array_MHz = (
                p.nu_min_MHz
                + np.arange(p.nf)*p.channel_width_MHz
            )
        sampled_uvw_coords_m = self.uvw_array_m.copy()
        # Convert uv-coordinates from meters to wavelengths at frequency
        # chan_freq_MHz for all chan_freq_MHz in nu_array_MHz
        sampled_uvw_coords_wavelengths = \
            np.array([sampled_uvw_coords_m
                      / (p.speed_of_light / (chan_freq_MHz * 1.e6))
                      for chan_freq_MHz in nu_array_MHz])
        if p.use_nvis_nt_nchan_ordering:
            # Used if p.model_drift_scan_primary_beam = False

            # Get (l, m, n) coordinates from Healpix object
            ls_rad, ms_rad, ns_rad = self.hp.calc_lmn_from_radec(
                self.hp.jds[p.nt // 2],
                radec_offset=self.beam_center
                )
            sampled_lmn_coords_radians = np.vstack((ls_rad, ms_rad, ns_rad)).T

            multi_chan_nudft = \
                self.sd_block_diag(
                    [
                        nuDFT_Array_DFT_2D_v2d0(
                            sampled_lmn_coords_radians,
                            sampled_uvw_coords_wavelengths[
                                freq_i, 0, :, :
                            ].reshape(-1, 3))
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
                            self.hp.calc_lmn_from_radec(
                                self.hp.jds[time_i],
                                radec_offset=self.beam_center
                                )  # gets (l(t), m(t), n(t))
                            ).T,
                        sampled_uvw_coords_wavelengths[
                            freq_i, time_i, :, :
                        ].reshape(-1, 3))
                    for freq_i in range(p.nf)])
                for time_i in range(p.nt)])

        # Multiply by sky model pixel area to get the units of the
        # model visibilities correct
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

        Notes
        -----
        * Used to construct `Finv`.
        * `multi_chan_P` has shape (npix * nf * nt, nuv * nf).

        """
        matrix_name = 'multi_chan_P'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        # nu_array_Hz = np.ones(p.nf) * p.nu_min_MHz * 1.0e6
        nu_array_Hz = (
            p.nu_min_MHz + np.arange(p.nf)*p.channel_width_MHz
        ) * 1.0e6
        if not p.model_drift_scan_primary_beam:
            multi_chan_P = self.sd_block_diag([
                np.diag(
                    self.hp.get_beam_vals(
                        *self.hp.calc_lmn_from_radec(
                            self.hp.jds[p.nt//2],
                            radec_offset=self.beam_center,
                            return_azza=True,
                            )[3:],  # Only need az, za
                        freq=freq
                        )
                    )
                for freq in nu_array_Hz])
        else:
            # Model the time dependence of the primary beam pointing
            # for a drift scan (i.e. change in zenith angle with time
            # due to Earth rotation).
            multi_chan_P = self.sd_vstack([
                self.sd_block_diag([
                    self.sd_diags(
                        self.hp.get_beam_vals(
                            *self.hp.calc_lmn_from_radec(
                                self.hp.jds[time_i],
                                radec_offset=self.beam_center,
                                return_azza=True
                                )[3:],  # Only need az, za
                            freq=freq
                            )
                        )
                    for freq in nu_array_Hz])
                for time_i in range(p.nt)])

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

        Notes
        -----
        * `Finv` has shape (ndata, nuv * nf).

        """
        matrix_name = 'Finv'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if p.include_instrumental_effects:
            Finv = self.dot_product(
                pmd['multi_chan_nudft'],
                pmd['multi_chan_P']
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
    def build_idft_array(self):
        """
        Construct a block for `multi_chan_idft_array_noZMchan` which
        is a non-uniform DFT matrix that takes a rectilinear, model
        (u, v) plane and transforms it to HEALPix sky model (l, m)
        space for a single frequency channel.

        Notes
        -----
        * Used to construct `Fprime`.
        * `idft_array` has shape (npix, nuv).

        """
        matrix_name = 'idft_array'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        # Get (l, m) coordinates from Healpix object
        ls_rad, ms_rad, _ = self.hp.calc_lmn_from_radec(
            self.hp.jds[p.nt//2]
            )
        sampled_lm_coords_radians = np.vstack((ls_rad, ms_rad)).T
        idft_array = IDFT_Array_IDFT_2D_ZM(
            self.nu, self.nv,
            sampled_lm_coords_radians,
            exclude_mean=(not p.fit_for_monopole))
        idft_array *= self.Fprime_normalization
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

        Notes
        -----
        * Used to construct `Fprime`.
        * `multi_chan_idft_array_noZMchan` has shape (npix * nf, nuv * nf).

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

    def build_idft_array_sh(self):
        """
        Construct a block diagonal matrix used in Fprime when using the
        subharmonic grid (SHG).  Each block in the matrix is a non-uniform
        DFT matrix that takes the SHG model (u, v) plane and transforms
        it to HEALPix sky model (l, m) space at a single frequency.

        Notes
        -----
        * Used to construct `Fprime` if using the SHG.
        * `idft_array_sh` has shape
          (npix*nf, nuv_sh*fit_for_shg_amps + nuv_sh*nq_sh).

        """
        matrix_name = 'idft_array_sh'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        # Get (l, m) coordinates from Healpix object
        ls_rad, ms_rad, _ = self.hp.calc_lmn_from_radec(
            self.hp.jds[p.nt//2]
            )
        sampled_lm_coords_radians = np.vstack((ls_rad, ms_rad)).T

        idft_array_sh_block = IDFT_Array_IDFT_2D_ZM_SH(
            self.nu_sh, self.nv_sh,
            sampled_lm_coords_radians)
        idft_array_sh_block *= self.Fprime_normalization / (self.nu*self.nv)
        idft_array_sh = self.sd_block_diag(
            [idft_array_sh_block for i in range(self.nf)]
        )
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(idft_array_sh,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_Fprime(self):
        """
        Construct a non-uniform, block-diagonal DFT matrix which takes a
        rectilinear (u, v, f) model and transforms it to HEALPix sky model
        (l, m, f) space.

        Notes
        -----
        * `Fprime` has shape (npix * nf, nuv * nf).

        """
        matrix_name = 'Fprime'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if self.use_shg:
            Fprime = self.sd_hstack(
                [pmd['multi_chan_idft_array_noZMchan'], pmd['idft_array_sh']]
            )
        else:
            Fprime = pmd['multi_chan_idft_array_noZMchan']
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(Fprime,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    # Fz functions
    def build_idft_array_1d_sh(self):
        """
        Construct a 1D LoS DFT matrix for a each (u, v) pixel in the
        SubHarmonic Grid (SHG).

        Notes
        -----
        * Used to construct `Fz` if using the SHG.

        """
        matrix_name = 'idft_array_1d_sh'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        idft_array_1d_sh_block = idft_array_idft_1d_sh(
            self.nf,
            self.neta,
            self.nq_sh,
            self.npl_sh,
            fit_for_shg_amps=self.fit_for_shg_amps,
            nu_min_MHz=p.nu_min_MHz,
            channel_width_MHz=p.channel_width_MHz,
            beta=p.beta)
        idft_array_1d_sh_block *= self.Fz_normalization
        nuv_sh = self.nu_sh*self.nv_sh - 1
        idft_array_1d_sh = self.sd_block_diag(
            [idft_array_1d_sh_block for i in range(nuv_sh)]
        )
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(idft_array_1d_sh,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)
        return idft_array_1d_sh

    def build_idft_array_1D(self):
        """
        Construct a 1D LoS DFT matrix for a single (u, v) pixel with
        ``nq = 0``.  Constructs one block within `multi_vis_idft_array_1D`.

        Notes
        -----
        * Used to construct `Fz` if ``nq = 0``.
        * `idft_array_1D` has shape (nf, neta).

        """
        matrix_name = 'idft_array_1D'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        idft_array_1D = IDFT_Array_IDFT_1D(self.nf, self.neta)
        idft_array_1D *= self.Fz_normalization
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(idft_array_1D,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_multi_vis_idft_array_1D(self):
        """
        Construct a block diagonal matrix of 1D LoS DFT matrices for every
        (u, v) pixel in the model uv-plane.

        Notes
        -----
        * Used to construct `Fz` if `nq` = 0.
        * `multi_vis_idft_array_1D` has shape (nuv * nf, nuv * neta).

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
        if self.use_shg:
            # Use subarhomic grid
            multi_vis_idft_array_1D = self.sd_block_diag(
                [multi_vis_idft_array_1D, pmd['idft_array_1d_sh']]
            )
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(multi_vis_idft_array_1D,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_idft_array_1D_WQ(self):
        """
        Construct a 1D LoS DFT matrix for a single (u, v) pixel with
        ``nq > 0``.  Constructs one block within `multi_vis_idft_array_1D_WQ`.

        Notes
        -----
        * Used to construct `Fz` if ``nq > 0``.
        * `idft_array_1D_WQ` has shape (nf, neta + nq).

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
        idft_array_1D_WQ *= self.Fz_normalization
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(idft_array_1D_WQ,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_multi_vis_idft_array_1D_WQ(self):
        """
        Construct a block diagonal matrix of 1D LoS DFT matrices with
        ``nq > 0`` for every (u, v) pixel in the model uv-plane.

        Notes
        -----
        * Used to construct `Fz` if ``nq > 0``.
        * `multi_vis_idft_array_1D_WQ` has shape (nuv * nf, nuv * (neta + nq)).

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
        if self.use_shg:
            # Use subarhomic grid
            multi_vis_idft_array_1D_WQ = self.sd_block_diag(
                [multi_vis_idft_array_1D_WQ, pmd['idft_array_1d_sh']]
            )
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(multi_vis_idft_array_1D_WQ,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_gridding_matrix_vis_ordered_to_chan_ordered(self):
        """
        Constructs a matrix which takes a (u, v, f) space vector that is vis
        ordered:
          - the first `nf` entries correspond to the spectrum of a single
            model (u, v) pixel
          - the second `nf` entries correspond to the spectrum of the next
            model (u, v) pixel
          - etc.
        and converts it to chan ordered:
          - the first 'nuv' entries correspond to the values of the model
            (u, v) plane at the zeroth frequency channel
          - the second 'nuv' entries correspond to the values of the model
            (u, v) plane at the first frequency channel
          - etc.

        Notes
        -----
        * `gridding_matrix_vis_ordered_to_chan_ordered` has shape
          (nuv*nf, nuv*nf).
        """
        matrix_name = 'gridding_matrix_vis_ordered_to_chan_ordered'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        gridding_matrix_vis_ordered_to_chan_ordered =\
            generate_gridding_matrix_vis_ordered_to_chan_ordered(
                self.nu, self.nv, self.nf,
                exclude_mean=(not p.fit_for_monopole))
        if self.use_shg:
            gridding_matrix_vo_to_co_sh =\
                generate_gridding_matrix_vis_ordered_to_chan_ordered(
                    self.nu_sh, self.nv_sh, self.nf
                )
            gridding_matrix_vis_ordered_to_chan_ordered = self.sd_block_diag(
                [gridding_matrix_vis_ordered_to_chan_ordered,
                 gridding_matrix_vo_to_co_sh]
            )
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

        Notes
        -----
        * `Fz` has shape (nuv * nf, nuv * (neta + nq)).

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
        Constructs `Fprime_Fz`.

        Notes
        -----
        * `Fprime_Fz` has shape (npix, nuv * (neta + nq)).

        """
        matrix_name = 'Fprime_Fz'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if self.use_shg and sparse.issparse(pmd['Fprime']):
            pmd['Fprime'] = pmd['Fprime'].toarray()
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

        Notes
        -----
        * `gridding_matrix_chan_ordered_to_vis_ordered` has shape
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
        Each diagonal component contains an estimate of
        1 / noise_amplitude**2 in the data vector
        at the index of the diagonal entry, i.e. data[i] for Ninv[i, i].

        Notes
        -----
        * `Ninv` has shape (ndata, ndata).

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
                baseline_redundancy_array = self.bl_red_array_vec
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
                    red_array_time_vis_shaped = self.bl_red_array
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

        Notes
        -----
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
                baseline_redundancy_array = self.bl_red_array_vec
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
                    red_array_time_vis_shaped = self.bl_red_array
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
        Construct `T` as a matrix product of `Finv`, `Fprime`, and Fz`.  `T`
        takes a model (u, v, eta) space data vector and transforms it to:
          1. model (u, v, f) space via Fz
          2. model (l, m, n, f) HEALPix space via Fprime
          3. data (u, v, w, f) space via Finv

        Notes
        -----
        * `T` has shape (ndata, nuv * (neta + nq)).

        """
        matrix_name = 'T'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        T = self.dot_product(pmd['Finv'],
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
        Matrix product of `Ninv` and `T`.  Can be used to take a (u, v, eta)
        data vector and compute a noise weighted vector in data space.

        Notes
        -----
        * `Ninv_T` has shape (ndata, nuv * (neta + nq)).

        """
        matrix_name = 'Ninv_T'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        Ninv_T = self.dot_product(pmd['Ninv'],
                                  pmd['T'])
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(
            Ninv_T,
            self.array_save_directory,
            matrix_name,
            matrix_name)

    def build_T_Ninv_T(self):
        """
        Matrix product of `T.conjugate().T`, `Ninv`, and `T`.

        Notes
        -----
        * `T_Ninv_T` has shape (nuv * (neta + nq), nuv * (neta + nq)).

        """
        matrix_name = 'T_Ninv_T'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        T_Ninv_T = self.dot_product(
            pmd['T'].conjugate().T,
            pmd['Ninv_T']
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
        """
        Constructs a block diagonal representation of `T_Ninv_T`.  Only used
        if ``p.use_instrumental_effects = False``.
        """
        matrix_name = 'block_T_Ninv_T'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if p.fit_for_monopole:
            self.nuv = (self.nu*self.nv)
        else:
            self.nuv = (self.nu*self.nv - 1)
        block_T_Ninv_T = np.array(
            [np.hsplit(block, self.nuv)
             for block in np.vsplit(pmd['T_Ninv_T'], self.nuv)]
            )
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(block_T_Ninv_T,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_matrix_if_it_doesnt_already_exist(self, matrix_name):
        """
        Constructs a matrix with name `matrix_name` if it doesn't
        already exist.

        This function doesn't return anything.  It instead calls the
        corresponding build matrix function.

        Parameters
        ----------
        matrix_name : str
            Name of matrix.

        """
        matrix_available = self.check_if_matrix_exists(matrix_name)
        if not matrix_available:
            self.matrix_construction_methods_dictionary[matrix_name]()

    def prepare_matrix_stack_for_deletion(
            self, src, overwrite_existing_matrix_stack):
        """
        Archive an existing matrix stack on disk by prepending 'delete_'
        to the child directory.

        Parameters
        ----------
        src : str
            Path to existing matrix stack directory.
        overwrite_existing_matrix_stack : bool
            If `True`, overwrite a previously archived matrix stack.

        Returns
        -------
        dst : str
            If ``overwrite_existing_matrix_stack = True``, path to matrix
            stack directory to be deleted.

        """
        if overwrite_existing_matrix_stack:
            if src[-1] == '/':
                src = src[:-1]
            head, tail = os.path.split(src)
            dst = os.path.join(head, 'delete_'+tail)
            print('Archiving existing matrix stack to:', dst)
            try:
                shutil.move(src, dst)
            except Exception:
                print('Archive path already existed. '
                      'Deleting the previous archive.')
                self.delete_old_matrix_stack(dst, 'y')
                self.prepare_matrix_stack_for_deletion(
                    self.array_save_directory,
                    self.overwrite_existing_matrix_stack)
            return dst

    def delete_old_matrix_stack(
            self, path_to_old_matrix_stack, confirm_deletion):
        """
        Delete or archive an existing matrix stack.

        Parameters
        ----------
        path_to_old_matrix_stack : str
            Path to the existing matrix stack.
        confirm_deletion : str
            If 'y', delete existing matrix stack.  Otherwise, archive the
            matrix stack.

        """
        if (confirm_deletion.lower() == 'y'
                or confirm_deletion.lower() == 'yes'):
            shutil.rmtree(path_to_old_matrix_stack)
        else:
            print('Prior matrix tree archived but not deleted.'
                  ' \nPath to archive:', path_to_old_matrix_stack)

    def build_minimum_sufficient_matrix_stack(
            self,
            overwrite_existing_matrix_stack=False,
            proceed_without_overwrite_confirmation=False):
        """
        Construct a minimum sufficient matrix stack needed to run BayesEoR.

        Parameters
        ----------
        overwrite_existing_matrix_stack : bool
            If `True`, overwrite the existing matrix stack.
        proceed_without_overwrite_confirmation : bool
            If `True`, delete the old matrix stack without user input.
            If `False`, prompt the user to specify wether the matrix stack
            should be deleted ('y') or archived ('n').

        """
        self.overwrite_existing_matrix_stack = (
            overwrite_existing_matrix_stack)
        self.proceed_without_overwrite_confirmation = (
            proceed_without_overwrite_confirmation)

        # Prepare matrix directory
        matrix_stack_dir_exists = os.path.exists(self.array_save_directory)
        if matrix_stack_dir_exists:
            dst = self.prepare_matrix_stack_for_deletion(
                self.array_save_directory,
                self.overwrite_existing_matrix_stack)
        # Build matrices
        self.build_matrix_if_it_doesnt_already_exist('T_Ninv_T')
        if not p.include_instrumental_effects:
            self.build_matrix_if_it_doesnt_already_exist('block_T_Ninv_T')
        self.build_matrix_if_it_doesnt_already_exist('N')
        if matrix_stack_dir_exists and self.overwrite_existing_matrix_stack:
            if not self.proceed_without_overwrite_confirmation:
                confirm_deletion = input(
                    'Confirm deletion of archived matrix stack? (y/n)\n')
            else:
                print('Deletion of archived matrix stack has '
                      'been pre-confirmed. Continuing...')
                confirm_deletion = 'y'
            self.delete_old_matrix_stack(dst, confirm_deletion)

        print('Matrix stack complete')
