import numpy as np
from subprocess import os
import shutil
import time
import h5py
from scipy.linalg import block_diag
from scipy import sparse
from scipy.signal import windows
from pathlib import Path
from astropy.constants import c

from .funcs import (
    nuidft_matrix_2d, idft_matrix_1d,
    build_lssm_basis_vectors,
    generate_gridding_matrix_vo2co,
    nuDFT_Array_DFT_2D_v2d0,
    idft_array_idft_1d_sh,
    IDFT_Array_IDFT_2D_ZM_SH,
    sampled_uv_vectors,
    build_nudft_array
)
from ..model.healpix import Healpix
from .. import __version__


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
try:
    from threadpoolctl import threadpool_info
    NTHREADS = threadpool_info()[0]['num_threads']
except:
    NTHREADS = 1


class BuildMatrixTree(object):
    """
    Class for building and manipulating BayesEoR matrices.

    Parameters
    ----------
    array_save_directory : str
        Path to the directory where arrays will be saved.
    include_instrumental_effects : bool
        If True, include instrumental effects like frequency dependent (u, v)
        sampling and the primary beam.
    use_sparse_matrices : bool
        If True, use sparse matrices in place of numpy arrays.
    Finv_Fprime : bool, optional
        If True (default), construct the matrix product Finv_Fprime in place
        from the dense matrices comprising Finv and Fprime to minimize the
        memory and time required to build the matrix stack.  In this case,
        only the matrix product Finv_Fprime is written to disk.  Otherwise,
        construct Finv and Fprime independently and save both matrices to disk.
    verbose : bool, optional
        Verbose output. Defaults to False.

    """
    def __init__(
        self,
        array_save_directory,
        include_instrumental_effects,
        use_sparse_matrices,
        Finv_Fprime=True,
        verbose=False
    ):
        self.array_save_directory = array_save_directory
        self.include_instrumental_effects = include_instrumental_effects
        self.use_sparse_matrices = use_sparse_matrices
        self.Finv_Fprime = Finv_Fprime
        self.verbose = verbose

        # Dictionary with keys for each parent matrix required to build the
        # full matrix stack and values of list containing names of child
        # matrices required to build each parent matrix
        self.matrix_prereqs = {
            'Finv': ['multi_chan_nudft', 'multi_chan_beam'],
            'Fprime': ['multi_chan_nuidft', 'multi_chan_nuidft_fg'],
            'multi_chan_nuidft': ['nuidft_array'],
            'multi_vis_idft_array_1d': ['idft_array_1d'],
            'gridding_matrix_co2vo': ['gridding_matrix_vo2co'],
            'Fz': [
                'gridding_matrix_vo2co',
                'multi_vis_idft_array_1d',
                'gridding_matrix_vo2co_fg',
                'idft_array_1d_fg'
            ],
            'Ninv_T': ['Ninv', 'T'],
            'T_Ninv_T': ['T', 'Ninv_T'],
            'block_T_Ninv_T': ['T_Ninv_T'],
        }
        if Finv_Fprime:
            self.matrix_prereqs.update(
                {'T': ['Finv_Fprime', 'Fz'],}
            )
        else:
            self.matrix_prereqs.update(
                {
                    'Fprime_Fz': ['Fprime', 'Fz'],
                    'T': ['Finv', 'Fprime_Fz']
                }
            )

    def check_for_prerequisites(self, parent_matrix):
        """
        Check if `parent_matrix` requires any child matrices to be built.

        Parameters
        ----------
        parent_matrix : str
            Name of matrix.

        Returns
        -------
        prerequisites_status : dict
            Dictionary containing any required child matrices as keys and
            integers as values specifying whether each child matrix has been
            constructed (1 for dense, 2 for sparse) or not (0). If no child
            matrices are required to build `parent_matrix`,
            `prerequisites_status` is returned as an empty dictionary.

        """
        prerequisites_status = {}
        if parent_matrix in self.matrix_prereqs.keys():
            for child_matrix\
                    in self.matrix_prereqs[parent_matrix]:
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
                if not self.use_sparse_matrices:
                    if self.verbose:
                        print(
                            'Only the sparse matrix representation is '
                            'available.\nUsing sparse representation and '
                            'setting self.use_sparse_matrices=True.'
                        )
                    self.use_sparse_matrices = True
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
            if self.verbose:
                print('Directory not found:\n\n' + directory + "\n")
                print('Creating required directory structure..')
            os.makedirs(directory)
        return 0

    def output_data(
        self, output_array, output_directory, file_name, dataset_name
    ):
        """
        Write matrix to disk.

        Checks if the data is a dense or sparse matrix and calls the
        corresponding output method.

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
                output_array, output_directory, file_name+'.npz')
        else:
            self.output_to_hdf5(
                output_array, output_directory, file_name+'.h5', dataset_name)
        return 0

    def output_to_hdf5(
        self, output_array, output_directory, file_name, dataset_name
    ):
        """
        Write matrix to HDF5 file.

        Parameters
        ----------
        output_array : array
            Array to be written to disk.
        output_directory : str
            Directory in which to write `output_array`.
        file_name : str
            Filename to use for `output_array`.
        dataset_name : str
            Key used to access `output_array`.

        """
        if self.verbose:
            start = time.time()
        self.create_directory(output_directory)
        output_path = Path(output_directory) / file_name
        if self.verbose:
            print('Writing data to', output_path)
        with h5py.File(output_path, 'w') as hf:
            hf.create_dataset(dataset_name, data=output_array)
        if self.verbose:
            print('Time taken: {}'.format(time.time() - start))
        return 0

    def output_sparse_matrix_to_npz(
        self, output_array, output_directory, file_name
    ):
        """
        Write sparse matrix to npz file.

        Parameters
        ----------
        output_array : array
            Array to be written to disk.
        output_directory : str
            Directory in which to write `output_array`.
        file_name : str
            Filename to use for `output_array`.

        Notes
        -----
        * To maintain sparse matrix attributes, use `sparse.save_npz` rather
          than `numpy.savez`.

        """
        if self.verbose:
            start = time.time()
        self.create_directory(output_directory)
        output_path = Path(output_directory) / file_name
        if self.verbose:
            print('Writing data to', output_path)
        sparse.save_npz(output_path, output_array.tocsr())
        if self.verbose:
            print('Time taken: {}'.format(time.time() - start))
        return 0

    def read_data_s2d(self, matrix_name):
        """
        Read matrix from disk and, if necessary, convert to dense matrix.

        Parameters
        ----------
        matrix_name : str
            Matrix name. For example, to load Finv from disk, `matrix_name`
            would be 'Finv'.

        """
        data = self.read_data(matrix_name)
        data = self.convert_sparse_matrix_to_dense_numpy_array(data)
        return data
    
    def read_data(self, matrix_name):
        """
        Read matrix from disk as dense or sparse matrix.

        This function searches `self.array_save_directory` for a file with a
        prefix matching `matrix_name`. It first searches for a sparse matrix,
        i.e. ``matrix_name + '.npz'``, and then searches for a dense matrix if
        a sparse matrix is not found, i.e. ``matrix_name + '.h5'``. If no
        valid file is found, an error is raised.

        Parameters
        ----------
        matrix_name : str
            Matrix name. For example, to load Finv from disk, `matrix_name`
            would be 'Finv'.

        Returns
        -------
        data : :class:`numpy.ndarray` or :class:`scipy.sparse`
            Loaded matrix as a dense or sparse array.

        """
        array_dir = Path(self.array_save_directory)
        if (array_dir / f"{matrix_name}.npz").exists():
            data = self.read_data_from_npz(array_dir / f"{matrix_name}.npz")
        elif (array_dir / f"{matrix_name}.h5").exists():
            data = self.read_data_from_hdf5(
                array_dir / f"{matrix_name}.h5", matrix_name
            )
        else:
            raise FileNotFoundError(
                f"File prefix {matrix_name} not found in "
                "self.array_save_directory"
            )
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

    def read_data_from_npz(self, file_path):
        """
        Read sparse matrix from npz file.

        Parameters
        ----------
        file_path : str
            Path to array file.

        Notes
        -----
        * To maintain sparse matrix attributes, use `sparse.load_npz` rather
          than `numpy.loadz`.

        """
        data = sparse.load_npz(file_path).tocsr()
        return data


class BuildMatrices(BuildMatrixTree):
    """
    Class for handling matrix construction and arithmetic.

    Parameters
    ----------
    array_save_directory : str
        Path to the directory where arrays will be saved.
    include_instrumental_effects : bool
        If True, include instrumental effects like frequency dependent (u, v)
        sampling and the primary beam.
    use_sparse_matrices : bool
        If True, use sparse matrices in place of numpy arrays.
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
    f_min : float
        Minimum frequency in megahertz.
    df : float
        Frequency channel width in megahertz.
    nt : int
        Number of times.
    nq : int
        Number of quadratic modes in the Large Spectral Scale Model (LSSM).
    sigma : float
        Expected noise amplitude in the data vector = signal + noise.
    fit_for_monopole : bool
        If True, fit for (u, v) = (0, 0).  Otherwise, exclude it from the fit.
    Finv_Fprime : bool
        If True (default), construct the matrix product Finv_Fprime in place
        from the dense matrices comprising Finv and Fprime to minimize the
        memory and time required to build the matrix stack.  In this case,
        only the matrix product Finv_Fprime is written to disk.  Otherwise,
        construct Finv and Fprime independently and save both matrices to disk.
    npl : int
        Number of power law coefficients which replace quadratic modes in
        the LSSM.
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
    phasor : :class:`numpy.ndarray`
        Array with shape (ndata,) that contains the phasor term used to phase
        visibilities after performing the nuDFT from HEALPix (l, m, f) to
        instrumentally sampled, unphased (u, v, f).  Defaults to None, i.e.
        modelling unphased visibilities.
    fov_ra_eor : float
        Field of view in degrees of the RA axis of the EoR sky model.
    fov_dec_eor : float
        Field of view in degrees of the DEC axis of the EoR sky model.
    fov_ra_fg : float
        Field of view in degrees of the RA axis of the FG sky model.
    fov_dec_fg : float
        Field of view in degrees of the DEC axis of the FG sky model.
    simple_za_filter : bool
        If passed, filter pixels in the sky model by zenith angle only.
        Otherwise, filter pixels in a rectangular region set by the FoV
        values along the RA and DEC axes (default).
    nside : int
        HEALPix nside parameter.
    telescope_latlonalt : tuple
        The latitude, longitude, and altitude of the telescope in degrees,
        degrees, and meters, respectively.
    central_jd : float
        Central time step of the observation in JD2000 format.
    dt : float
        Time cadence of observations in seconds.
    drift_scan : bool
        If True, model a drift scan primary beam, i.e. the beam center drifts
        across the image space model with time.
    beam_type : string
        Beam type to use.  Can be 'uniform', 'gaussian', 'airy', 'taperairy',
        or 'gausscosine'.
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
        Antenna (aperture) diameter in meters.  Used in the calculation of an
        Airy beam pattern or when using a Gaussian beam with a FWHM that varies
        as a function of frequency.  The FWHM evolves according to the
        effective FWHM of the main lobe of an Airy beam.
    cosfreq : float
        Cosine frequency in radians if using a 'gausscosine' beam.
    achromatic_beam : bool, optional
        If True, force the beam to be achromatic using `beam_ref_freq` as the
        reference frequency.
    beam_ref_freq : float, optional
        Beam reference frequency in MHz.  Used to fix the beam to be
        achromatic.
    effective_noise : :class:`numpy.ndarray`
        If the data vector being analyzed contains signal + noise, the
        effective_noise vector contains the estimate of the noise in the data
        vector.  Must have the shape and ordering of the data vector,
        i.e. (ndata,).
    deta : float
        Fourier mode spacing along the eta (line of sight, frequency) axis in
        inverse Hz.
    du_eor : float
        Fourier mode spacing along the u axis in inverse radians of the
        EoR model uv-plane.
    dv_eor : float
        Fourier mode spacing along the v axis in inverse radians of the
        EoR model uv-plane.
    du_fg : float
        Fourier mode spacing along the u axis in inverse radians of the
        FG model uv-plane.
    dv_fg : float
        Fourier mode spacing along the v axis in inverse radians of the
        FG model uv-plane.
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
    taper_func : str, optional
        Tapering function to apply to the frequency axis of the model
        visibilities.  Can be any valid argument to
        `scipy.signal.windows.get_window`.
    nu_fg : int
        Number of pixels on a side for the u-axis in the FG model uv-plane.
    nv_fg : int
        Number of pixels on a side for the v-axis in the FG model uv-plane.
    verbose : bool, optional
        Verbose output. Defaults to False.

    """
    def __init__(
        self,
        nu=None,  # required
        du_eor=None,  # required
        nv=None,  # required
        dv_eor=None,  # required
        nu_fg=None,  # required
        du_fg=None,  # required
        nv_fg=None,  # required
        dv_fg=None,  # required
        nf=None,   # required
        neta=None,   # required
        deta=None,  # required
        fit_for_monopole=False,
        use_shg=False,
        nu_sh=0,
        nv_sh=0,
        nq_sh=0,
        npl_sh=0,
        fit_for_shg_amps=False,
        f_min=None,  # required
        df=None,  # required
        nq=0,
        npl=0,
        beta=None,
        sigma=None,  # required
        nside=None,
        fov_ra_eor=None,
        fov_dec_eor=None,
        fov_ra_fg=None,
        fov_dec_fg=None,
        simple_za_filter=True,
        include_instrumental_effects=True,
        telescope_latlonalt=(0, 0, 0),
        nt=None,
        central_jd=None,
        dt=None,
        beam_type=None,
        beam_center=None,
        achromatic_beam=False,
        beam_peak_amplitude=None,
        fwhm_deg=None,
        antenna_diameter=None,
        cosfreq=None,
        beam_ref_freq=None,
        drift_scan=True,
        n_vis=None,  # required
        array_save_directory="./matrices/",
        use_sparse_matrices=True,
        uvw_array_m=None,
        bl_red_array=None,
        bl_red_array_vec=None,
        phasor=None,
        effective_noise=None,
        taper_func=None,
        Finv_Fprime=True,
        verbose=False,
    ):
        # TODO: it would be nice to come up with some way to check that the
        # required kwargs have been passed, but the list of required kwargs
        # depends on e.g. include_instrumental_effects, use_shg.
        required_kwargs = [
            nu,
            nv,
            nu_fg,
            nv_fg,
            n_vis,
            neta,
            nf,
            f_min,
            df,
            sigma,
            deta,
            du_eor,
            dv_eor,
            du_fg,
            dv_fg
        ]
        if not np.all([arg is not None for arg in required_kwargs]):
            raise ValueError(
                "The following kwargs are required and cannot be None: "
                "nu, nv, nu_fg, nv_fg, n_vis, neta, nf, f_min, df, sigma, "
                "deta, du_eor, dv_eor, du_fg, dv_fg"
            )

        # TODO: merge the BuildMatrixTree and BuildMatrices classes into one.
        # There's no need to have them separated as BuildMatrixTree was never
        # used independently and BuildMatrices inherits directly from it with
        # no modifications of the parent class's functions or attributes.
        """BuildMatrixTree.__init__"""
        # self.array_save_directory = array_save_directory
        # self.include_instrumental_effects = include_instrumental_effects
        # self.use_sparse_matrices = use_sparse_matrices
        # self.Finv_Fprime = Finv_Fprime
        # self.verbose = verbose

        # # Dictionary with keys for each parent matrix required to build the
        # # full matrix stack and values of list containing names of child
        # # matrices required to build each parent matrix
        # self.matrix_prereqs = {
        #     'Finv': ['multi_chan_nudft', 'multi_chan_beam'],
        #     'Fprime': ['multi_chan_nuidft', 'multi_chan_nuidft_fg'],
        #     'multi_chan_nuidft': ['nuidft_array'],
        #     'multi_vis_idft_array_1d': ['idft_array_1d'],
        #     'gridding_matrix_co2vo': ['gridding_matrix_vo2co'],
        #     'Fz': [
        #         'gridding_matrix_vo2co',
        #         'multi_vis_idft_array_1d',
        #         'gridding_matrix_vo2co_fg',
        #         'idft_array_1d_fg'
        #     ],
        #     'Ninv_T': ['Ninv', 'T'],
        #     'T_Ninv_T': ['T', 'Ninv_T'],
        #     'block_T_Ninv_T': ['T_Ninv_T'],
        # }
        # if Finv_Fprime:
        #     self.matrix_prereqs.update({'T': ['Finv_Fprime', 'Fz']})
        # else:
        #     self.matrix_prereqs.update(
        #         {'Fprime_Fz': ['Fprime', 'Fz'], 'T': ['Finv', 'Fprime_Fz']}
        #     )
        """BuildMatrixTree.__init__"""

        # Required params
        self.nu = nu
        self.nv = nv
        self.nu_fg = nu_fg
        self.nv_fg = nv_fg
        self.n_vis = n_vis
        self.neta = neta
        self.nf = nf
        self.f_min = f_min
        self.df = df
        self.nq = nq
        self.nt = nt
        self.dt = dt
        self.sigma = sigma
        self.fit_for_monopole = fit_for_monopole

        self.freqs_hertz = (self.f_min + np.arange(self.nf)*self.df) * 1e6

        if self.include_instrumental_effects:
            self.uvw_array_m = uvw_array_m
            self.nbls = uvw_array_m.shape[1]
            self.bl_red_array = bl_red_array
            self.bl_red_array_vec = bl_red_array_vec
            self.phasor = phasor
            self.fov_ra_eor = fov_ra_eor
            if fov_dec_eor is None:
                self.fov_dec_eor = self.fov_ra_eor
            else:
                self.fov_dec_eor = fov_dec_eor
            if fov_ra_fg is None:
                self.fov_ra_fg = fov_ra_eor
            else:
                self.fov_ra_fg = fov_ra_fg
            if fov_dec_fg is None:
                self.fov_dec_fg = self.fov_ra_fg
            else:
                self.fov_dec_fg = fov_dec_fg
            self.simple_za_filter = simple_za_filter
            self.nside = nside
            self.central_jd = central_jd
            self.telescope_latlonalt = telescope_latlonalt
            self.beam_type = beam_type
            self.beam_peak_amplitude = beam_peak_amplitude
            self.beam_center = beam_center
            self.fwhm_deg = fwhm_deg
            self.antenna_diameter = antenna_diameter
            self.cosfreq = cosfreq
            self.achromatic_beam = achromatic_beam
            self.beam_ref_freq = beam_ref_freq
            self.effective_noise = effective_noise

            self.hpx = Healpix(
                fov_ra_eor=self.fov_ra_eor,
                fov_dec_eor=self.fov_dec_eor,
                fov_ra_fg=self.fov_ra_fg,
                fov_dec_fg=self.fov_dec_fg,
                simple_za_filter=self.simple_za_filter,
                nside=self.nside,
                telescope_latlonalt=self.telescope_latlonalt,
                central_jd=self.central_jd,
                nt=self.nt,
                int_time=self.dt,
                beam_type=self.beam_type,
                peak_amp=self.beam_peak_amplitude,
                fwhm_deg=self.fwhm_deg,
                diam=self.antenna_diameter,
                cosfreq=self.cosfreq
            )

            self.drift_scan = drift_scan

        # FG model params
        self.nuv_fg = self.nu_fg*self.nv_fg - (not self.fit_for_monopole)
        self.npl = npl
        self.beta = beta

        # SHG params
        self.use_shg = use_shg
        self.fit_for_shg_amps = fit_for_shg_amps
        self.nu_sh = nu_sh
        self.nv_sh = nv_sh
        self.nq_sh = nq_sh
        self.npl_sh = npl_sh

        # Taper function
        self.taper_func = taper_func

        # Fz normalization
        self.deta = deta
        self.Fz_normalization = self.deta

        # Fprime normalization
        self.du_eor = du_eor
        self.dv_eor = dv_eor
        self.Fprime_normalization_eor = (
            self.nu * self.nv * self.du_eor * self.dv_eor
        )
        self.du_fg = du_fg
        self.dv_fg = dv_fg
        self.Fprime_normalization_fg = (
            self.nu_fg * self.nv_fg * self.du_fg * self.dv_fg
        )

        # Finv normalization
        # FIXME: this will error if include_instrumental_effects is False.
        self.Finv_normalisation = self.hpx.pixel_area_sr

        self.matrix_methods = {
            'idft_array_1d': self.build_idft_array_1d,
            'multi_vis_idft_array_1d': self.build_multi_vis_idft_array_1d,
            'gridding_matrix_vo2co': self.build_gridding_matrix_vo2co,
            'gridding_matrix_co2vo': self.build_gridding_matrix_co2vo,
            'idft_array_1d_fg': self.build_idft_array_1d_fg,
            'gridding_matrix_vo2co_fg': self.build_gridding_matrix_vo2co_fg,
            'Fz': self.build_Fz,
            'nuidft_array': self.build_nuidft_array,
            'multi_chan_nuidft': self.build_multi_chan_nuidft,
            'multi_chan_nuidft_fg': self.build_multi_chan_nuidft_fg,
            'Fprime': self.build_Fprime,
            'multi_chan_nudft': self.build_multi_chan_nudft,
            'multi_chan_beam': self.build_multi_chan_beam,
            'Finv': self.build_Finv,
            'Fprime_Fz': self.build_Fprime_Fz,
            'Finv_Fprime': self.build_Finv_Fprime,
            'T': self.build_T,
            'N': self.build_N,
            'Ninv': self.build_Ninv,
            'Ninv_T': self.build_Ninv_T,
            'T_Ninv_T': self.build_T_Ninv_T,
            'block_T_Ninv_T': self.build_block_T_Ninv_T,
        }
        
        if self.phasor is not None:
            self.matrix_prereqs.update(
                {'Finv': ['phasor_matrix'] + self.matrix_prereqs['Finv']}
            )
            self.matrix_methods.update(
                {'phasor_matrix': self.build_phasor_matrix}
            )

        if self.use_shg:
            self.matrix_prereqs.update(
                {
                    'multi_vis_idft_array_1d': ['idft_array_1d',
                                                'idft_array_1d_sh'],
                    'Fprime': ['multi_chan_nuidft',
                               'multi_chan_nuidft_fg',
                               'nuidft_array_sh']
                }
            )
            self.matrix_methods.update(
                {'idft_array_1d_sh': self.build_idft_array_1d_sh,
                 'nuidft_array_sh': self.build_nuidft_array_sh}
            )

        if self.taper_func is not None:
            self.matrix_prereqs.update(
                {'Finv': ['taper_matrix'] + self.matrix_prereqs['Finv']}
            )
            self.matrix_methods.update(
                {'taper_matrix': self.build_taper_matrix}
            )

    def load_prerequisites(self, matrix_name):
        """
        Load or build any prerequisites for a given matrix.

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
                    self.matrix_methods[child_matrix]()
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
                start = time.time()
                data = self.read_data(child_matrix)
                prerequisite_matrices_dictionary[child_matrix] = data
                print('Time taken: {}'.format(time.time() - start))

        return prerequisite_matrices_dictionary

    def dot_product(self, matrix_a, matrix_b):
        """
        Calculate the dot product of matrix_a and matrix_b.
        
        Uses sparse or dense matrix algebra based upon the type of each matrix.

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
        Generate a block diagonal matrix from a list of matrices.

        Parameters
        ----------
        block_matrices_list : list
            List of input matrices.

        Returns
        -------
        block_diag_matrix : array
            If ``self.use_sparse_matrices = True``, return a sparse matrix.
            Otherwise, return a dense numpy array.

        """
        if self.use_sparse_matrices:
            block_diag_matrix = sparse.block_diag(block_matrices_list)
        else:
            block_diag_matrix = block_diag(*block_matrices_list)
        return block_diag_matrix

    def sd_vstack(self, matrices_list):
        """
        Generate a vertically stacked matrix from a list of matrices.

        Parameters
        ----------
        matrices_list : list
            List of input matrices.

        Returns
        -------
        vstack_matrix : array
            If ``self.use_sparse_matrices = True``, return a sparse matrix.
            Otherwise, return a dense numpy array.

        """
        if self.use_sparse_matrices:
            vstack_matrix = sparse.vstack(matrices_list)
        else:
            vstack_matrix = np.vstack(matrices_list)
        return vstack_matrix

    def sd_hstack(self, matrices_list):
        """
        Generate a horizontally stacked matrix from a list of matrices.

        Parameters
        ----------
        matrices_list : list
            List of input matrices.

        Returns
        -------
        hstack_matrix : array
            If ``self.use_sparse_matrices = True``, return a sparse matrix.
            Otherwise, return a dense numpy array.

        """
        if self.use_sparse_matrices:
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
            If ``self.use_sparse_matrices = True``, return a sparse matrix.
            Otherwise, return a dense numpy array.

        """
        if self.use_sparse_matrices:
            diagonal_matrix = sparse.diags(diagonal_vals)
        else:
            diagonal_matrix = np.diag(diagonal_vals)
        return diagonal_matrix

    # Finv functions
    def build_taper_matrix(self):
        """
        Build a diagonal matrix containing a frequency taper function.

        The taper matrix is a diagonal matrix containing a tapering function
        on the diagonal multiplied elementwise into the visibility vector.

        Notes
        -----
        * Used to construct `Finv` if using a taper function.
        * taper_matrix has shape (ndata, ndata).
        * This function assumes that `use_nvis_nchan_nt_ordering = True`.

        """
        matrix_name = 'taper_matrix'
        start = time.time()
        print('Performing matrix algebra')
        taper = windows.get_window(self.taper_func, self.nf)
        nbls = self.uvw_array_m.shape[1]
        taper = np.repeat(taper[None, :], nbls, axis=0).flatten(order='F')
        taper = np.tile(taper, self.nt)
        if self.use_sparse_matrices:
            taper_matrix = sparse.diags(taper)
        else:
            taper_matrix = np.diag(taper)
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(
            taper_matrix,
            self.array_save_directory,
            matrix_name,
            matrix_name)

    def build_phasor_matrix(self):
        """
        Build a diagonal matrix used to phase visibilities.

        The phasor matrix is multiplied elementwise into the visibility vector
        from Finv, constructed using unphased (u, v, w) coordinates, to produce
        phased visibilities.

        The phasor matrix is constructed as a diagonal matrix of
        $e^(i*phi(u(t), v(t), w(t)))$ phasor terms from the optional phasor
        vector in the instrument model.

        Notes
        -----
        * Used to construct `Finv` if modelling phased visibilities.
        * phasor_matrix has shape (ndata, ndata).
        * This function assumes that `use_nvis_nchan_nt_ordering = True`.

        """
        matrix_name = 'phasor_matrix'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if self.use_sparse_matrices:
            phasor_matrix = sparse.diags(self.phasor)
        else:
            phasor_matrix = np.diag(self.phasor)
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(
            phasor_matrix,
            self.array_save_directory,
            matrix_name,
            matrix_name)

    def build_multi_chan_nudft(self):
        """
        Build a multi-frequency NUDFT matrix for image to measurement space.

        Each block in this block-diagonal matrix transforms a set of
        time-dependent image-space (l(t), m(t), n(t)) HEALPix coordinates to
        unphased, instrumentall sampled, frequency dependent
        (u(f), v(f), w(f)).

        Notes
        -----
        * Used to construct `Finv`.
        * If ``use_nvis_nt_nchan_ordering = True``: model visibilities will be
          ordered (nvis*nt) per chan for all channels (old default).
        * If ``use_nvis_nchan_nt_ordering = True``: model visibilities will be
          ordered (nvis*nchan) per time step for all time steps.  This ordering
          is required when using a drift scan primary beam (current default).
        * `multi_chan_nudft` has shape (ndata, npix * nf * nt).

        """
        matrix_name = 'multi_chan_nudft'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        sampled_uvw_coords_m = self.uvw_array_m.copy()
        # Convert uv-coordinates from meters to wavelengths per frequency
        sampled_uvw_coords_wavelengths = np.array([
            sampled_uvw_coords_m / (c.to('m/s').value / freq)
            for freq in self.freqs_hertz
        ])
        if not self.drift_scan:
            # Used if self.drift_scan = False
            # Get (l, m, n) coordinates from Healpix object
            ls_rad, ms_rad, ns_rad = self.hpx.calc_lmn_from_radec(
                self.hpx.jds[self.nt//2],
                self.hpx.ra_fg,
                self.hpx.dec_fg,
                radec_offset=self.beam_center
            )
            sampled_lmn_coords_radians = np.vstack((ls_rad, ms_rad, ns_rad)).T

            multi_chan_nudft = self.sd_block_diag([
                nuDFT_Array_DFT_2D_v2d0(
                    sampled_lmn_coords_radians,
                    sampled_uvw_coords_wavelengths[
                        freq_i, 0, :, :
                    ].reshape(-1, 3))
                for freq_i in range(self.nf)
            ])
        else:
            # This will be used if a drift scan primary beam is included in
            # the data model (i.e. self.drift_scan=True)
            multi_chan_nudft = self.sd_block_diag([
                self.sd_block_diag([
                    nuDFT_Array_DFT_2D_v2d0(
                        np.vstack(
                            self.hpx.calc_lmn_from_radec(
                                self.hpx.jds[time_i],
                                self.hpx.ra_fg,
                                self.hpx.dec_fg,
                                radec_offset=self.beam_center
                            )
                        ).T,
                        sampled_uvw_coords_wavelengths[
                            freq_i, time_i, :, :
                        ].reshape(-1, 3))
                    for freq_i in range(self.nf)
                ])
                for time_i in range(self.nt)
            ])

        # Multiply by sky model pixel area to get the units of the
        # model visibilities correct
        multi_chan_nudft *= self.Finv_normalisation

        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(multi_chan_nudft,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_multi_chan_beam(self):
        """
        Build a matrix contating image space beam amplitudes.

        Each block-diagonal entry contains the beam amplitude at each HEALPix
        sampled (l(t), m(t), n(t)) for a single time and frequency.  Each stack
        contains `nf` block-diagonal entries containing the beam amplitudes at
        all frequencies for a single time.

        Notes
        -----
        * Used to construct `Finv`.
        * `multi_chan_beam` has shape (npix * nf * nt, npix * nf).

        """
        matrix_name = 'multi_chan_beam'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if self.achromatic_beam:
            freq_array = np.ones_like(self.freqs_hertz) * self.beam_ref_freq
            freq_array *= 1e6  # MHz --> Hz
        else:
            freq_array = self.freqs_hertz
        if not self.drift_scan:
            multi_chan_beam = self.sd_block_diag([
                np.diag(
                    self.hpx.get_beam_vals(
                        *self.hpx.calc_lmn_from_radec(
                            self.hpx.jds[self.nt//2],
                            self.hpx.ra_fg,
                            self.hpx.dec_fg,
                            radec_offset=self.beam_center,
                            return_azza=True,
                            )[3:],  # Only need az, za
                        freq=freq
                        )
                    )
                for freq in freq_array])
        else:
            # Model the time dependence of the primary beam pointing
            # for a drift scan (i.e. change in zenith angle with time
            # due to Earth rotation).
            multi_chan_beam = self.sd_vstack([
                self.sd_block_diag([
                    self.sd_diags(
                        self.hpx.get_beam_vals(
                            *self.hpx.calc_lmn_from_radec(
                                self.hpx.jds[time_i],
                                self.hpx.ra_fg,
                                self.hpx.dec_fg,
                                radec_offset=self.beam_center,
                                return_azza=True
                                )[3:],  # Only need az, za
                            freq=freq
                            )
                        )
                    for freq in freq_array])
                for time_i in range(self.nt)])

        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(multi_chan_beam,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_Finv(self):
        """
        Build a multi-frequency NUDFT matrix for image to measurement space.

        Finv is a a non-uniform DFT matrix that takes a vector of
        (l, m, n) syk model pixel amplitudes and

          #. Applies a beam per time and frequency via `multi_chan_beam`
          #. Transforms to insttrumentally sampled, unphased (u(f), v(f), w(f))
             coordinates from the instrument model
          #. If modelling phased visibilities, applies a phasor vector from the
             instrument model to phase the visibilities to the central time
             step

        Notes
        -----
        * `Finv` has shape (ndata, npix * nf).

        """
        matrix_name = 'Finv'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        Finv = self.dot_product(
            pmd['multi_chan_nudft'], pmd['multi_chan_beam']
        )
        if self.phasor is not None:
            Finv = self.dot_product(pmd['phasor_matrix'], Finv)
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(Finv,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    # Fprime functions
    def build_nuidft_array(self):
        """
        Build a NUIDFT matrix for uv to image space.

        This matrix forms a block in `multi_chan_nuidft` and transforms the EoR
        model uv-plane to image space at a single frequency. Specifically,
        `nuidft_array` transforms a rectilinear (u, v) grid to HEALPix sampled
        (l, m).  The model uv-plane has w=0, so no w or n terms are included
        in this transformation.

        Notes
        -----
        * Used for the EoR model in `Fprime`.
        * `nuidft_array` has shape (npix, nuv).
        * If the EoR and FG models have different FoV values, `nuidft_array` is
          reshaped to match the dimensions of the FG model (FoV_FG >= FoV_EoR).
          The HEALPix pixel ordering must be preserved in this reshaping so
          that shared pixels between the EoR and FG models are summed together
          in image-space.

        """
        matrix_name = 'nuidft_array'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        # Get (l, m) coordinates from Healpix object
        ls_rad, ms_rad, _ = self.hpx.calc_lmn_from_radec(
            self.hpx.jds[self.nt//2],
            self.hpx.ra_eor,
            self.hpx.dec_eor
        )
        nuidft_array = nuidft_matrix_2d(
            self.nu, self.nv, self.du_eor, self.dv_eor, ls_rad, ms_rad
        )
        if not self.hpx.fovs_match:
            nuidft_array_fg_pix = np.zeros(
                (self.hpx.npix_fov, nuidft_array.shape[1]), dtype=complex
            )
            nuidft_array_fg_pix[self.hpx.eor_to_fg_pix] = (
                nuidft_array
            )
            nuidft_array = nuidft_array_fg_pix
        nuidft_array *= self.Fprime_normalization_eor
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(nuidft_array,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_multi_chan_nuidft(self):
        """
        Build a multi-frequency NUIDFT matrix for uv to image space.
        
        `multi_chan_nuidft` is constructed as a block-diagonal matrix.  Each
        block is constructed via `build_nuidft_array` and represents a 2D
        non-uniform DFT matrix from rectilinear (u, v) to HEALPix (l, m) for a
        single frequency.

        Notes
        -----
        * Used for the EoR model in `Fprime`.
        * `multi_chan_nuidft` has shape (npix_eor * nf, nuv_eor * nf).

        """
        matrix_name = 'multi_chan_nuidft'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        multi_chan_nuidft = self.sd_block_diag(
            [pmd['nuidft_array'] for i in range(self.nf)]
        )
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(multi_chan_nuidft,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_multi_chan_nuidft_fg(self):
        """
        Build a multi-frequency NUIDFT matrix for uv to image space.

        `multi_chan_nuidft_fg` is constructed as a block-diagonal matrix.  Each
        block is a 2D non-uniform DFT matrix from rectilinear (u, v) to
        HEALPix (l, m) at a single frequency.

        Notes
        -----
        * Used for the FG model in `Fprime`.
        * `multi_chan_nuidft_fg` has shape
          (npix_fg * nf, nuv_fg * nf)

        """
        matrix_name = 'multi_chan_nuidft_fg'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        ls_rad, ms_rad, _ = self.hpx.calc_lmn_from_radec(
            self.hpx.jds[self.nt//2],
            self.hpx.ra_fg,
            self.hpx.dec_fg
        )
        nuidft_array = nuidft_matrix_2d(
            self.nu_fg, self.nv_fg, self.du_fg, self.dv_fg,
            ls_rad, ms_rad, exclude_mean=False
        )
        nuidft_array *= self.Fprime_normalization_fg
        if self.fit_for_monopole:
            mp_col = nuidft_array[:, self.nuv_fg//2].copy().reshape(-1, 1)
        nuidft_array = np.delete(nuidft_array, self.nuv_fg//2, axis=1)
        if self.fit_for_monopole:
            nuidft_array = np.hstack((nuidft_array, mp_col))
        multi_chan_nuidft_fg = self.sd_block_diag(
            [nuidft_array for i in range(self.nf)]
        )
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(multi_chan_nuidft_fg,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_nuidft_array_sh(self):
        """
        Build a multi-frequency NUIDFT matrix for uv to image space.
        
        `nuidft_array_sh` is constructed as a block diagonal matrix.  Each
        block transforms the SubHarmonic Grid (SHG) model uv-plane to HEALPix
        sampled (l, m) at a single frequency.

        Notes
        -----
        * Used for the SHG model in `Fprime`.
        * `nuidft_array_sh` has shape
          (npix*nf, nuv_sh*fit_for_shg_amps + nuv_sh*nq_sh).

        """
        matrix_name = 'nuidft_array_sh'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        # Get (l, m) coordinates from Healpix object
        ls_rad, ms_rad, _ = self.hpx.calc_lmn_from_radec(
            self.hpx.jds[self.nt//2]
        )
        sampled_lm_coords_radians = np.vstack((ls_rad, ms_rad)).T

        nuidft_array_sh_block = IDFT_Array_IDFT_2D_ZM_SH(
            self.nu_sh, self.nv_sh,
            sampled_lm_coords_radians)
        nuidft_array_sh_block *= self.Fprime_normalization / (self.nu*self.nv)
        nuidft_array_sh = self.sd_block_diag(
            [nuidft_array_sh_block for i in range(self.nf)]
        )
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(nuidft_array_sh,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_Fprime(self):
        """
        Build a multi-frequency NUIDFT matrix fo uv to image space.
        
        Fprime takes a rectilinear (u, v) model as a channel ordered vector
        and transforms it to HEALPix sky model (l, m) space.  Fprime is
        constructed as a block-diagonal matrix with blocks for the EoR and FG
        models.

        Notes
        -----
        * `Fprime` has shape
          ((npix_eor + npix_fg) * nf, (nuv_eor + nuv_fg) * nf).

        """
        matrix_name = 'Fprime'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        Fprime = self.sd_hstack([
            pmd['multi_chan_nuidft'], pmd['multi_chan_nuidft_fg']
        ])
        if self.use_shg:
            Fprime = self.sd_hstack([Fprime, pmd['nuidft_array_sh']])
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(Fprime,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    # Fz functions
    def build_idft_array_1d_sh(self):
        """
        Build a block-diagonal IDFT matrix for eta to frequency space.

        `idft_array_1d_sh` is constructted as a block-diagonal matrix.  Each
        block is a 1D IDFT matrix for the eta spectrum of each (u, v) pixel in
        the SubHarmonic Grid (SHG) model uv-plane.

        Notes
        -----
        * Used for the SHG model in `Fz`.

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
            nu_min_MHz=self.f_min,
            channel_width_MHz=self.df,
            beta=self.beta)
        idft_array_1d_sh_block *= self.Fz_normalization * self.neta
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

    def build_idft_array_1d(self):
        """
        Build an IDFT matrix for eta to frequency space.
        
        Constructs one block within `multi_vis_idft_array_1d`.

        Notes
        -----
        * Used for the EoR model in `Fz`.
        * `idft_array` has shape (nf, neta - 1).
        * Excludes eta=0 which belongs to the FG model.

        """
        matrix_name = 'idft_array_1d'
        start = time.time()
        print('Performing matrix algebra')
        idft_array_1d = idft_matrix_1d(
            self.nf,
            self.neta,
            include_eta0=False
        )
        idft_array_1d *= self.Fz_normalization
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(idft_array_1d,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_multi_vis_idft_array_1d(self):
        """
        Build a block-diagonal IDFT matrix from eta to frequency space.

        Notes
        -----
        * Used for the EoR model in `Fz`.
        * `multi_vis_idft_array_1d` has shape
          (nuv_eor * nf, nuv_eor * (neta - 1)).

        """
        matrix_name = 'multi_vis_idft_array_1d'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        multi_vis_idft_array_1d = self.sd_block_diag(
            [pmd['idft_array_1d'] for i in range(self.nu*self.nv - 1)]
        )
        if self.use_shg:
            multi_vis_idft_array_1d = self.sd_block_diag(
                [multi_vis_idft_array_1d, pmd['idft_array_1d_sh']]
            )
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(multi_vis_idft_array_1d,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_idft_array_1d_fg(self):
        """
        Build a block-diagonal IDFT matrix for eta to freuqency space.

        Notes
        -----
        * Used for the FG model in `Fz`.
        * `idft_array_1d_fg` has shape (nuv_fg * nf,
          nuv_fg*(1 + nq) + fit_for_monopole*(neta + nq)).

        """
        matrix_name = 'idft_array_1d_fg'
        start = time.time()
        print('Performing matrix algebra')
        eta0_to_frequency = np.ones((self.nf, 1), dtype=complex)
        eta0_to_frequency *= self.deta * self.neta
        neta0_blocks = self.nuv_fg - self.fit_for_monopole
        idft_array_1d_eta0 = self.sd_block_diag([
            eta0_to_frequency for _ in range(neta0_blocks)
        ])
        idft_array_1d_fg = idft_array_1d_eta0
        if self.nq > 0:
            lssm_basis_vecs = build_lssm_basis_vectors(
                self.nf, nq=self.nq, npl=self.npl, f_min=self.f_min,
                df=self.df, beta=self.beta
            )
            lssm_basis_vecs *= self.deta * self.neta
            lssm_array = self.sd_block_diag([
                lssm_basis_vecs for _ in range(neta0_blocks)
            ])
            idft_array_1d_fg = self.sd_hstack([
                idft_array_1d_fg, lssm_array
            ])
        if self.fit_for_monopole:
            idft_array_1d_mp = idft_matrix_1d(
                self.nf, self.neta, nq=self.nq, npl=self.npl,
                f_min=self.f_min, df=self.df, beta=self.beta
            )
            idft_array_1d_mp *= self.deta
            idft_array_1d_mp[:, self.neta//2] *= self.neta
            idft_array_1d_mp[:, self.neta:] *= self.neta
            idft_array_1d_fg = self.sd_block_diag([
                idft_array_1d_fg, idft_array_1d_mp
            ])
        print('Time taken: {}'.format(time.time() - start))
        self.output_data(idft_array_1d_fg,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_gridding_matrix_vo2co(self):
        """
        Build a vis ordered to chan ordered gridding matrix.

        The gridding matrix takes a (u, v, f) space vector that is vis ordered:
          - the first `neta` entries correspond to the spectrum of the zeroth
            index model (u, v) pixel
          - the second `neta` entries correspond to the spectrum of the first
            index model (u, v) pixel
          - etc.
        and converts it to chan ordered:
          - the first 'nuv' entries correspond to the values of the model
            (u, v) plane at the zeroth frequency channel
          - the second 'nuv' entries correspond to the values of the model
            (u, v) plane at the first frequency channel
          - etc.

        Notes
        -----
        * Used for the EoR model in `Fz`.
        * `gridding_matrix_vo2co` has shape (nuv*nf, nuv*nf).

        """
        matrix_name = 'gridding_matrix_vo2co'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        gridding_matrix = generate_gridding_matrix_vo2co(
            self.nu, self.nv, self.nf, use_sparse=self.use_sparse_matrices
        )
        if self.use_shg:
            gridding_matrix_vo_to_co_sh = generate_gridding_matrix_vo2co(
                self.nu_sh, self.nv_sh, self.nf,
                use_sparse=self.use_sparse_matrices
            )
            gridding_matrix = self.sd_block_diag(
                [gridding_matrix, gridding_matrix_vo_to_co_sh]
            )
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(gridding_matrix,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)
    
    def build_gridding_matrix_vo2co_fg(self):
        """
        Build a vis ordered to chan ordered gridding matrix.

        The gridding matrix takes a (u, v, f) space vector that is vis ordered:
          - the first `neta` entries correspond to the spectrum of the zeroth
            index model (u, v) pixel
          - the second `neta` entries correspond to the spectrum of the first
            index model (u, v) pixel
          - etc.
        and converts it to chan ordered:
          - the first 'nuv' entries correspond to the values of the model
            (u, v) plane at the zeroth frequency channel
          - the second 'nuv' entries correspond to the values of the model
            (u, v) plane at the first frequency channel
          - etc.

        Notes
        -----
        * Used for the FG model in `Fz`.
        * `gridding_matrix_vo2co_fg` is a square matrix with dimension
          (nuv_fg - (not fit_for_monopole)) * nf.

        """
        matrix_name = 'gridding_matrix_vo2co_fg'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        gridding_matrix = generate_gridding_matrix_vo2co(
            self.nu_fg, self.nv_fg, self.nf,
            exclude_mean=(not self.fit_for_monopole),
            use_sparse=self.use_sparse_matrices
        )
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(gridding_matrix,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_Fz(self):
        """
        Build a block-diagonal IDFT matrix for eta to frequency space.

        `Fz` is constructed as a block-diagonal matrix.  Each block is a 1D
        IDFT matrix which takes a vis ordered eta space vector and
        transforms it to a chan ordered frequency space data vector.

        Notes
        -----
        * `Fz` has shape ((nuv_eor + nuv_fg)*nf,
           nuv_eor*(neta - 1) + nuv_fg*(1 + nq)
           + fit_for_monopole*(neta + nq)).

        """
        matrix_name = 'Fz'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        Fz_eor = self.dot_product(
            pmd['gridding_matrix_vo2co'],
            pmd['multi_vis_idft_array_1d']
        )
        Fz_fg = self.dot_product(
            pmd['gridding_matrix_vo2co_fg'],
            pmd['idft_array_1d_fg']
        )
        Fz = self.sd_block_diag((Fz_eor, Fz_fg))
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(Fz,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_Finv_Fprime(self):
        """
        Build the matrix product Finv @ Fprime from their dense blocks.

        This function has been added to reduce the RAM required to build
        Finv and Fprime when the foreground sky model FoV is large.  Building
        the matrix product Finv @ Fprime directly from their dense blocks
        requires much less memory than constructing Finv and Fprime
        independently.  If more than one thread is available and threading is
        available within numpy, this method also significantly reduces the time
        required to compute the matrix product Finv @ Fprime.
        
        Notes
        -----
        * `Finv_Fprime` has shape
          - (ndata, nf*nuv_eor) if modelling the EoR only
          - (ndata, nf*(nuv_eor + nuv_fg)) if modelling EoR + foregrounds
          - (ndata, nf*(nuv_eor + nuv_fg) + nuv_sh*(nq_sh +
            fit_for_shg_amps)) if modelling EoR + foregrounds + SHG.  Please
            note that the SHG is not currently supported (see issue #50).

        """
        nuv_eor = self.nu*self.nv - 1
        model_fgs = self.nq > 0 or self.fit_for_monopole
        if model_fgs:
            nuv_fg = self.nu_fg*self.nv_fg - (not self.fit_for_monopole)
        # FIXME: add support for the SHG uv-plane (issue #50)
        # if self.use_shg:
        #     nuv_sh = self.nu_sh*self.nv_sh - 1

        Finv_Fprime_shape = [
            # Measurement space axis
            # Number of visibilities
            self.nbls * self.nt * self.nf,
            # Combined EoR + FG model (u, v, f) cube axis
            # Number of EoR model parameters
            nuv_eor * self.nf
        ]
        if model_fgs:
            # Number of FG model parameters
            Finv_Fprime_shape[1] += nuv_fg * self.nf
        # FIXME: add support for the SHG uv-plane (issue #50)
        # if self.use_shg:
        #     # Number of SHG model parameters
        #     Finv_Fprime_shape[1] += nuv_sh*(self.nq_sh + self.fit_for_shg_amps)
        Finv_Fprime_shape = tuple(Finv_Fprime_shape)

        # Get time-dependent (l, m, n, az, za) coordinates
        model_lmnazza_rad = [
            self.hpx.calc_lmn_from_radec(
                self.hpx.jds[i_t],
                self.hpx.ra,
                self.hpx.dec,
                return_azza=True,
                radec_offset=self.beam_center,
            )
            for i_t in range(self.nt)
        ]
        # Shape (nt, nax, npix)
        # nax index mapping:
        # 0: l
        # 1: m
        # 2: n
        # 3: az
        # 4: za
        model_lmnazza_rad = np.array(model_lmnazza_rad)

        # Frequency-dependent (u, v, w) sampled by the instrument
        inst_uvw_m = self.uvw_array_m.copy()
        # Convert from meters to inverse radians per frequency
        inst_uvw_irad = [
            inst_uvw_m / (c.to('m/s').value / freq)
            for freq in self.freqs_hertz
        ]
        # Shape (nf, nt, nbls, 3)
        inst_uvw_irad = np.array(inst_uvw_irad)
        inst_uvw_irad = np.swapaxes(inst_uvw_irad, -2, -1)

        # Construct the dense NUIDFT matrices in Fprime relating the model
        # uv-plane (u, v, f) to the image space (l, m, f) coordinates.
        # EoR model uv-plane components and matrix:
        # Flattened vectors of (u, v) in the EoR model uv-plane
        model_us_eor_irad, model_vs_eor_irad = sampled_uv_vectors(
            self.nu,
            self.nv,
            du=self.du_eor,
            dv=self.dv_eor,
            exclude_mean=True
        )
        model_uv_eor_irad = np.vstack((model_us_eor_irad, model_vs_eor_irad))
        nuidft_uv_to_lm_eor = build_nudft_array(
            model_uv_eor_irad,
            model_lmnazza_rad[self.nt//2, :2, self.hpx.eor_to_fg_pix].T,
            sign=-1
        )
        # FIXME: Fprime normalization no longer needs to include nu*nv norm
        # when using the build_nudft_array function.  The nuidft_matrix_2d
        # function follows the numpy approach and uses 1/Npix for the inverse
        # transform which is why the nu*nv normalization is currently
        # included in self.Fprime_normalization_eor.
        nuidft_uv_to_lm_eor *= (
            self.Fprime_normalization_eor / (self.nu * self.nv)
        )
        
        if model_fgs:
            model_us_fg_irad, model_vs_fg_irad = sampled_uv_vectors(
                self.nu_fg,
                self.nv_fg,
                du=self.du_fg,
                dv=self.dv_fg,
                exclude_mean=(not self.fit_for_monopole)
            )
            model_uv_fg_irad = np.vstack((model_us_fg_irad, model_vs_fg_irad))
            nuidft_uv_to_lm_fg = build_nudft_array(
                model_uv_fg_irad,
                model_lmnazza_rad[self.nt//2, :2],
                sign=-1
            )
            # FIXME: Fprime normalization no longer needs to include nu*nv norm
            # when using the build_nudft_array function.  The nuidft_matrix_2d
            # function follows the numpy approach and uses 1/Npix for the
            # inverse transform which is why the nu*nv normalization is
            # currently included in self.Fprime_normalization_fg.
            nuidft_uv_to_lm_fg *= (
                self.Fprime_normalization_fg / (self.nu_fg * self.nv_fg)
            )
            # Move the (u, v) = (0, 0) pixel to the rightmost column in
            # nuidft_uv_to_lm_fg so that the LSSM components are all clustered
            # together in the model vector (recall that (u, v) = (0, 0) acts
            # like the constant term in the LSSM).
            mp_column = nuidft_uv_to_lm_fg[:, nuv_fg//2].copy()
            mp_ind = nuv_fg//2
            nuidft_uv_to_lm_fg[:, mp_ind:-1] = nuidft_uv_to_lm_fg[:, mp_ind+1:]
            nuidft_uv_to_lm_fg[:, -1] = mp_column

        # FIXME: add support for the SHG uv-plane (issue #50)
        # if self.use_shg:
        #     nuidft_uv_to_lm_shg = ?

        # Instantiate empty containers for matrices
        Finv_Fprime = sparse.lil_matrix(Finv_Fprime_shape, dtype=complex)
        
        matrix_name = 'Finv_Fprime'
        # FIXME: for now, there are no prerequisites for this matrix.  But,
        # this might change in the future if we add a kwarg for writing out
        # intermediate matrices to disk (e.g. the dense blocks that comprise
        # Fprime).
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        
        for i_f in range(self.nf):
            uv_Fprime_inds_eor = slice(
                i_f*nuv_eor, (i_f + 1)*nuv_eor
            )
            if model_fgs:
                uv_Fprime_inds_fg = slice(
                    self.nf*nuv_eor + i_f*nuv_fg,
                    self.nf*nuv_eor + (i_f + 1)*nuv_fg
                )
            # FIXME: add support for the SHG uv-plane (issue #50)
            # if self.use_shg:
            #     uv_Fprime_inds_shg = slice(
            #         self.nf*(nuv_eor + nuv_fg) + i_f*?,
            #         self.nf*(nuv_eor + nuv_fg) + (i_f + 1)*?
            #     )
            for i_t in range(self.nt):
                vis_inds = slice(
                    i_t*self.nf*self.nbls + i_f*self.nbls,
                    i_t*self.nf*self.nbls + (i_f + 1)*self.nbls
                )

                if self.drift_scan:
                    # For a drift scan observation, the beam drifts across the
                    # modelled patch of sky, so the (l, m, n) coordinates for
                    # a given HEALPix pixel change with time.
                    lmn_rad = model_lmnazza_rad[i_t, :3]
                    az_rad = model_lmnazza_rad[i_t, 3]
                    za_rad = model_lmnazza_rad[i_t, 4]
                else:
                    # For modelling phased observations, the beam is always
                    # phased to the center of the model FoV.  The (l, m, n)
                    # coordinates in this case are independent of time.
                    lmn_rad = model_lmnazza_rad[self.nt//2, :3]
                    az_rad = model_lmnazza_rad[self.nt//2, 3]
                    za_rad = model_lmnazza_rad[self.nt//2, 4]

                # NUDFT matrix for image space (l, m, n) to measurement
                # space (u, v, w) sampled by the instrument
                nudft_lmn_to_vis = build_nudft_array(
                    lmn_rad,
                    inst_uvw_irad[i_f, 0],
                    sign=+1
                )
                nudft_lmn_to_vis *= self.Finv_normalisation
                # Beam amplitude at each (l, m, n)
                beam_vals = self.hpx.get_beam_vals(
                    az_rad, za_rad, freq=self.freqs_hertz[i_f]
                )
                # Equivalent to np.dot(nudft_lmn_to_vis, np.diag(beam_vals))
                Finv_block = nudft_lmn_to_vis * beam_vals[None, :]

                # FIXME: add support for modelling phased visibilities via
                # the phasor vector.

                Finv_Fprime[vis_inds, uv_Fprime_inds_eor] = np.dot(
                    Finv_block[:, self.hpx.eor_to_fg_pix],
                    nuidft_uv_to_lm_eor
                )
                if model_fgs:
                    Finv_Fprime[vis_inds, uv_Fprime_inds_fg] = np.dot(
                        Finv_block,
                        nuidft_uv_to_lm_fg
                    )
                # FIXME: add support for the SHG uv-plane (issue #50)
                # if self.use_shg:
                #     Finv_Fprime[vis_inds, uv_Fprime_inds_shg] = ?

        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(Finv_Fprime,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    def build_Fprime_Fz(self):
        """
        Build `Fprime_Fz = Fprime * Fz`.

        Notes
        -----
        * `Fprime_Fz` has shape (npix, nuv_eor * (neta - 1)
           + (nuv_fg - fit_for_monopole) * (1 + nq)
           + fit_for_monopole * (neta + nq)).

        """
        matrix_name = 'Fprime_Fz'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if self.use_shg and sparse.issparse(pmd['Fprime']):
            pmd['Fprime'] = pmd['Fprime'].toarray()
        Fprime_Fz = self.dot_product(pmd['Fprime'], pmd['Fz'])
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5
        # Read, write, and dot product times are all shorter for Fprime_Fz
        # as a dense matrix
        self.output_data(
            Fprime_Fz.toarray(),
            self.array_save_directory,
            matrix_name,
            matrix_name)

    def build_gridding_matrix_co2vo(self):
        """
        Build a chan ordered to vis ordered gridding matrix.

        This matrix is the transposition of `gridding_matrix_vo2co`.

        Notes
        -----
        * `gridding_matrix_co2vo` has shape
          (nuv * (neta + nq), nuv * (neta + nq)).

        """
        matrix_name = 'gridding_matrix_co2vo'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        # NOTE: taking the transpose reverses the gridding. This is what
        # happens in dbar where Fz.conjugate().T is multiplied by d and
        # the gridding_matrix_vo2co.conjugate().T
        # part of Fz transforms d from chan-ordered initially to
        # vis-ordered.
        # NOTE: conjugate does nothing to the gridding matrix component
        # of Fz, which is real, it only transforms the 1D IDFT to a DFT)
        gridding_matrix_co2vo = pmd['gridding_matrix_vo2co'].T
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(gridding_matrix_co2vo,
                         self.array_save_directory,
                         matrix_name,
                         matrix_name)

    # Covariance matrix functions
    def build_Ninv(self):
        """
        Build a diagonal inverse covariance matrix.

        Each diagonal component contains an estimate of `1 / |noise|**2`
        in the data vector.

        Notes
        -----
        * `Ninv` has shape (ndata, ndata).

        """
        matrix_name = 'Ninv'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if self.include_instrumental_effects:
            if not self.drift_scan:
                # This array is channel_ordered and the covariance
                # matrix assumes a channel_ordered data set
                # (this vector should be re-ordered if
                # the data is in a different order)
                baseline_redundancy_array = self.bl_red_array_vec
                s_size = self.n_vis * self.nf
                multifreq_baseline_redundancy_array = np.array(
                    [baseline_redundancy_array for i in range(self.nf)]
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
            else:
                if self.effective_noise is None:
                    red_array_time_vis_shaped = self.bl_red_array
                    baseline_redundancy_array_time_freq_vis = np.array([
                        [red_array_vis for i in range(self.nf)]
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
                    # FIXME: see BayesEoR issue #55
                    # The docstring for `effective_noise` suggests it stores
                    # the noise visibilities, but this code assumes it instead
                    # stores an estimate for the standard deviation of the
                    # noise
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

        if self.use_sparse_matrices:
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
        Build a diagonal covariance matrix.

        Each diagonal component contains an estimate of `|noise|**2` in the
        data vector.

        Notes
        -----
        * `N` has shape (ndata, ndata).

        """
        matrix_name = 'N'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if self.include_instrumental_effects:
            if not self.drift_scan:
                # This array is channel_ordered and the covariance
                # matrix assumes a channel_ordered data set
                # (this vector should be re-ordered if
                # the data is in a different order)
                baseline_redundancy_array = self.bl_red_array_vec
                s_size = self.n_vis * self.nf
                multifreq_baseline_redundancy_array = np.array(
                    [baseline_redundancy_array for i in range(self.nf)]
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
            else:
                if self.effective_noise is None:
                    red_array_time_vis_shaped = self.bl_red_array
                    baseline_redundancy_array_time_freq_vis = np.array([
                        [red_array_vis for i in range(self.nf)]
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

        if self.use_sparse_matrices:
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
        Build `T = Finv * Fprime * Fz`.

        `T` takes a model (u(eta), v(eta)) space data vector and transforms it
        to

          #. uv space via `Fz`
          #. image space via `Fprime`
          #. measurement space via `Finv`

        Notes
        -----
        * `T` has shape (ndata, nuv * (neta + nq)).

        """
        matrix_name = 'T'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if self.Finv_Fprime:
            print('Constructing T = Finv_Fprime * Fz')
            # Fz is mostly sparse (<3% of the entries are non-zero).
            # This matrix product Finv_Fprime * Fz is quick to compute
            # even using Fz as a sparse matrix which only usese a
            # single thread for matrix multiplications.
            T = self.dot_product(pmd['Finv_Fprime'], pmd['Fz']).todense()
        else:
            print('Constructing T = Finv * Fprime_Fz')
            # Finv is a stack of block diagonal matrices.  Each
            # component of the stack encodes the sky --> visibilities
            # transformation at a single time.  If we treat Finv as a
            # sparse matrix, we use scipy's sparse dot product which is
            # limited to a single core/thread.  Numpy's dot product is
            # typically configured for threading.  If we have more than
            # 1 thread available, it's much faster to do a dense-dense
            # dot product.  Finv has a lot of zeros, though, so it's
            # inefficient in terms of memory to cast it as a single
            # dense matrix.  Instead, it's more efficient in time and
            # memory to add a couple for loops and multiply the dense
            # block diagonals of Finv times the corresponding
            # components in Fprime_Fz.
            if NTHREADS == 1:
                T = self.dot_product(pmd['Finv'], pmd['Fprime_Fz'])
            else:
                T = np.zeros(
                    (pmd['Finv'].shape[0], pmd['Fprime_Fz'].shape[1]),
                    dtype=complex
                )
                for i_t in range(self.nt):
                    time_inds = slice(
                        i_t*self.nbls*self.nf, (i_t + 1)*self.nbls*self.nf
                    )
                    for i_f in range(self.nf):
                        vis_inds = slice(i_f*self.nbls, (i_f + 1)*self.nbls)
                        sky_inds = slice(
                            i_f*self.hpx.npix_fov,
                            (i_f + 1)*self.hpx.npix_fov
                        )
                        inds = [vis_inds, sky_inds]
                        T[time_inds][vis_inds] = np.dot(
                            pmd['Finv'][time_inds][*inds].toarray(),
                            pmd['Fprime_Fz'][sky_inds]
                        )
        print('Time taken: {}'.format(time.time() - start))
        # Save matrix to HDF5 or sparse matrix to npz
        self.output_data(
            T,
            self.array_save_directory,
            matrix_name,
            matrix_name)

    def build_Ninv_T(self):
        """
        Build `Ninv_T = Ninv * T`.

        `Ninv_T` computes the inverse covariance weighted vector in data space
        from an eta space data vector.

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
        Build `T_Ninv_T = T.conjugate().T * Ninv * T`.

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
        if ``self.use_instrumental_effects = False``.
        """
        matrix_name = 'block_T_Ninv_T'
        pmd = self.load_prerequisites(matrix_name)
        start = time.time()
        print('Performing matrix algebra')
        if self.fit_for_monopole:
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
            self.matrix_methods[matrix_name]()

    def prepare_matrix_stack_for_deletion(
            self, src, clobber_matrices):
        """
        Archive an existing matrix stack on disk by prepending 'delete_'
        to the child directory.

        Parameters
        ----------
        src : str
            Path to existing matrix stack directory.
        clobber_matrices : bool
            If `True`, overwrite a previously archived matrix stack.

        Returns
        -------
        dst : str
            If ``clobber_matrices = True``, path to matrix
            stack directory to be deleted.

        """
        if clobber_matrices:
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
                    self.clobber_matrices)
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
        if confirm_deletion.lower() in ['y', 'yes']:
            shutil.rmtree(path_to_old_matrix_stack)
        else:
            print('Prior matrix tree archived but not deleted.'
                  ' \nPath to archive:', path_to_old_matrix_stack)
    
    def write_version_info(self):
        """
        Write version info to disk.

        """
        fp = Path(self.array_save_directory) / 'version.txt'
        if not fp.exists():
            with open(fp, 'w') as f:
                f.write(f'{__version__}\n')

    def build_minimum_sufficient_matrix_stack(
            self,
            clobber_matrices=False,
            force_clobber=False):
        """
        Construct a minimum sufficient matrix stack needed to run BayesEoR.

        Parameters
        ----------
        clobber_matrices : bool
            If `True`, overwrite the existing matrix stack.
        force_clobber : bool
            If `True`, delete the old matrix stack without user input.
            If `False`, prompt the user to specify wether the matrix stack
            should be deleted ('y') or archived ('n').

        """
        self.clobber_matrices = clobber_matrices
        self.force_clobber = force_clobber

        # Prepare matrix directory
        matrix_stack_dir_exists = os.path.exists(self.array_save_directory)
        if matrix_stack_dir_exists:
            dst = self.prepare_matrix_stack_for_deletion(
                self.array_save_directory,
                self.clobber_matrices)
        # Build matrices
        self.build_matrix_if_it_doesnt_already_exist('T_Ninv_T')
        if not self.include_instrumental_effects:
            self.build_matrix_if_it_doesnt_already_exist('block_T_Ninv_T')
        self.build_matrix_if_it_doesnt_already_exist('N')
        if matrix_stack_dir_exists and self.clobber_matrices:
            if not self.force_clobber:
                confirm_deletion = input(
                    'Confirm deletion of archived matrix stack? (y/n)\n')
            else:
                print('Deletion of archived matrix stack has '
                      'been pre-confirmed. Continuing...')
                confirm_deletion = 'y'
            self.delete_old_matrix_stack(dst, confirm_deletion)

        self.write_version_info()
        print('Matrix stack complete')
