import time
import numpy as np
import scipy
from pdb import set_trace as brk
import h5py
from mpi4py import MPI

import BayesEoR.Params.params as p
from BayesEoR.Utils import Cosmology, mpiprint


"""
Potentially useful links:
http://www.mrao.cam.ac.uk/~kjbg1/lectures/lect1_1.pdf
"""


rank = MPI.COMM_WORLD.Get_rank()

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    import ctypes
    from numpy import ctypeslib

    # Get path to installation of BayesEoR
    base_dir = '/'.join(__file__.split('/')[:-2]) + '/'
    # Load MAGMA GPU Wrapper Functions
    GPU_wrap_dir = base_dir+'Likelihood/GPU_wrapper/'

    device = cuda.Device(0)
    if 'p100' in device.name().lower():
        gpu_arch = 'p100'
    elif 'v100' in device.name().lower():
        gpu_arch = 'v100'
    mpiprint('Found GPU with {} architecture'.format(gpu_arch), rank=rank)
    mpiprint('Loading shared library from {}'.format(
        GPU_wrap_dir + 'wrapmzpotrf_{}.so'.format(gpu_arch)
        ),
        rank=rank
    )
    wrapmzpotrf = ctypes.CDLL(
        GPU_wrap_dir + 'wrapmzpotrf_{}.so'.format(gpu_arch)
        )
    nrhs = 1
    wrapmzpotrf.cpu_interface.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypeslib.ndpointer(np.complex128, ndim=2, flags='C'),
        ctypeslib.ndpointer(np.complex128, ndim=1, flags='C'),
        ctypes.c_int,
        ctypeslib.ndpointer(np.int, ndim=1, flags='C')]
    mpiprint('Computing on {} GPUs'.format(gpu_arch), rank=rank)

except Exception as e:
    mpiprint('Exception loading GPU encountered...', rank=rank)
    mpiprint(repr(e), rank=rank)
    mpiprint('Computing on CPU instead...', rank=rank)
    p.useGPU = False


# --------------------------------------------
# Define posterior
# --------------------------------------------
class PowerSpectrumPosteriorProbability(object):
    """
    Class containing posterior calculation functions.

    Parameters
    ----------
    T_Ninv_T : np.ndarray
        Complex matrix product `T.conjugate().T * Ninv * T`.  Used to construct
        `Sigma` which is used to solve for the ML 3D Fourier space vector and
        LSSM amplitudes.  `T_Ninv_T` must be a Hermitian, positive-definite
        matrix.
    dbar : np.ndarray
        Noise weighted representation of the data (signal + noise) vector of
        visibilities in model (u, v, eta) space.
    Sigma_Diag_Indices : np.ndarray
        Diagonal indices of `T_Ninv_T`.
    Npar : int
        Total number of model parameters being fit.
    k_cube_voxels_in_bin : list
        List containing sublists containing the flattened 3D k-space cube index
        of all |k| that fall within a given k-bin.
    nuv : int
        Number of model uv-plane points per frequency channel.  Computed as
        `nuv = nu*nv - 1*np.logical_not(p.fit_for_monopole)`.
    nu : int
        Number of pixels on a side for the u-axis in the model uv-plane.
    nv : int
        Number of pixels on a side for the v-axis in the model uv-plane.
    neta : int
        Number of Line of Sight (LoS, frequency axis) Fourier modes.
    nf : int
        Number of frequency channels.
    nq : int
        Number of quadratic modes in the Larse Spectral Scale Model (LSSM).
    masked_power_spectral_modes : np.ndarray
        Boolean array used to mask additional (u, v, eta) amplitudes from
        being included in the posterior calculations.
    modk_vis_ordered_list : list
        List of sublists containing the |k| values for each |k| that falls
        within a given k-bin.
    Ninv : np.ndarray
        Covariance matrix of the data (signal + noise) vector of visibilities.
    d_Ninv_d : np.ndarray
        Single complex number computed as `d.conjugate() * Ninv * d`.
    k_vals : np.ndarray of floats
        Array containing the mean k for each k-bin.
    ps_box_size_ra_Mpc : float
        Right ascension (RA) axis extent of the cosmological volume in Mpc from
        which the power spectrum is estimated.
    ps_box_size_dec_Mpc : float
        Declination (DEC) axis extent of the cosmological volume in Mpc from
        which the power spectrum is estimated.
    ps_box_size_para_Mpc : float
        LoS extent of the cosmological volume in Mpc from which the power
        spectrum is estimated.
    block_T_Ninv_T : list
        Block diagonal representation of `T_Ninv_T`.  Only used if
        ``p.use_instrumental_effects = False``.  Defaults to `[]`.
    log_priors : boolean
        If `True`, power spectrum k-bin amplitudes are assumed to be in log
        units, otherwise they will be treated using linear units.
    dimensionless_PS : boolean
        If `True`, use a dimensionless power spectrum normalization
        `Delta**2 ~ mK**2`, otherwise use a dimensionful power spectrum
        normalization `P(k) ~ mK**2 Mpc**3`.
    inverse_LW_power : float
        Prior over the long wavelength modes in the large spectral scale model
        (LSSM).  Defaults to 0.0.
    inverse_LW_power_zeroth_LW_term : float
        Prior for the zeroth (monopole) term in the LSSM.  Defaults to 0.0.
    inverse_LW_power_first_LW_term : float
        Prior for the first (linear) term in the LSSM.  Defaults to 0.0.
    inverse_LW_power_second_LW_term : float
        Prior for the second (quadratic) term in the LSSM.  Defaults to 0.0.
    Print : boolean
        If `True`, print execution time messages.  Defaults to `False`.
    debug : boolean
        If `True`, execute break statements for debugging.  Defaults to
        `False`.
    Print_debug : boolean
        If `True`, print debug related messages.  Defaults to `False`.
    intrinsic_noise_fitting : boolean
        If `True`, fit for the amplitude of the noise in the data instead of
        using the covariance estimate as is in `Ninv`.  Defaults to `False`.
    return_Sigma : boolean
        If `True`, break and return the matrix `Sigma = T_Ninv_T + PhiI`.
        Defaults to `False`.
    fit_for_spectral_model_parameters : boolean
        If `True`, fit for the LSSM parameter values instead of using the
        values in `p.beta`.  Defaults to `False`.
    n_uniform_prior_k_bins : int
        Number of k-bins, counting up from the lowest k_bin, that will use a
        prior which is uniform in the amplitude.  The remaining k-bins will use
        log-uniform priors.  Defaults to 0.
    uniform_priors : bool
        If `True`, all k-bins use a prior uniform in the amplitude.  Otherwise,
        all k-bins use a log-uniform prior.
    use_shg : bool
        If `True`, use the SubHarmonic Grid (SHG) in the model uv-plane.
    fit_for_shg_amps : bool
        if `True`, fit explicitly for the amplitudes of the individual SHG
        pixels per frequency.
    nuv_sh : int
        Number of pixels in the SHG model uv-plane `nuv_sh = nu_sh*nv_sh - 1`.
    nu_sh : int
        Number of pixels on a side for the u-axis in the subharmonic model
        uv-plane.
    nv_sh : int
        Number of pixels on a side for the v-axis in the subharmonic model
        uv-plane.
    nq_sh : int
        Number of LSSM quadratic modes for each pixel in the subharmonic grid.
    rank : int
        MPI rank.

    """
    def __init__(
            self, T_Ninv_T, dbar, Sigma_Diag_Indices, Npar,
            k_cube_voxels_in_bin, nuv, nu, nv, neta, nf, nq,
            masked_power_spectral_modes, modk_vis_ordered_list,
            Ninv, d_Ninv_d, k_vals, ps_box_size_ra_Mpc,
            ps_box_size_dec_Mpc, ps_box_size_para_Mpc,
            block_T_Ninv_T=[], log_priors=False, dimensionless_PS=False,
            inverse_LW_power=0.0,
            inverse_LW_power_zeroth_LW_term=0.0,
            inverse_LW_power_first_LW_term=0.0,
            inverse_LW_power_second_LW_term=0.0,
            Print=False, debug=False, Print_debug=False,
            intrinsic_noise_fitting=False, return_Sigma=False,
            fit_for_spectral_model_parameters=False,
            n_uniform_prior_k_bins=0, uniform_priors=False,
            use_shg=False, fit_for_shg_amps=False,
            nuv_sh=None, nu_sh=None, nv_sh=None, nq_sh=None,
            rank=0
            ):
        self.rank = rank
        self.block_T_Ninv_T = block_T_Ninv_T
        self.log_priors = log_priors
        if self.log_priors:
            mpiprint('Using log-priors', rank=self.rank)
        self.dimensionless_PS = dimensionless_PS
        if self.dimensionless_PS:
            mpiprint('Calculating dimensionless_PS', rank=self.rank)
        self.inverse_LW_power = inverse_LW_power
        self.inverse_LW_power_zeroth_LW_term = inverse_LW_power_zeroth_LW_term
        self.inverse_LW_power_first_LW_term = inverse_LW_power_first_LW_term
        self.inverse_LW_power_second_LW_term = inverse_LW_power_second_LW_term
        self.Print = Print
        self.debug = debug
        self.Print_debug = Print_debug
        self.intrinsic_noise_fitting = intrinsic_noise_fitting
        self.return_Sigma = return_Sigma
        self.fit_for_spectral_model_parameters =\
            fit_for_spectral_model_parameters
        self.k_vals = k_vals
        self.n_uniform_prior_k_bins = n_uniform_prior_k_bins
        self.uniform_priors = uniform_priors
        self.ps_box_size_ra_Mpc = ps_box_size_ra_Mpc
        self.ps_box_size_dec_Mpc = ps_box_size_dec_Mpc
        self.ps_box_size_para_Mpc = ps_box_size_para_Mpc
        self.use_shg = use_shg
        self.fit_for_shg_amps = fit_for_shg_amps
        self.nuv_sh = nuv_sh
        self.nu_sh = nu_sh
        self.nv_sh = nv_sh
        self.nq_sh = nq_sh

        self.T_Ninv_T = T_Ninv_T
        self.dbar = dbar
        self.Sigma_Diag_Indices = Sigma_Diag_Indices
        self.block_diagonal_sigma = False
        self.instantiation_time = time.time()
        self.count = 0
        self.Npar = Npar
        self.k_cube_voxels_in_bin = k_cube_voxels_in_bin
        self.nuv = nuv
        self.nu = nu
        self.nv = nv
        self.neta = neta
        self.nf = nf
        self.nq = nq
        self.masked_power_spectral_modes = masked_power_spectral_modes
        self.modk_vis_ordered_list = modk_vis_ordered_list
        self.Ninv = Ninv
        self.d_Ninv_d = d_Ninv_d
        self.print_rate = 1000
        self.alpha_prime = 1.0
        self.spectral_model_parameters_array_storage_dir =\
            '/gpfs/data/jpober/psims/EoR/Python_Scripts/BayesEoR/'\
            'git_version/BayesEoR/spec_model_tests/array_storage/'\
            'FgSpecMOptimisation/Likelihood_v1d76_3D_ZM_nu_9_nv_9_'\
            'neta_38_nq_2_npl_2_b1_2.00E+00_b2_3.00E+00_sigma_8d5E+04'\
            '_instrumental/HERA_331_baselines_shorter_than_29d3_for_30'\
            '_0d5_min_time_steps_Gaussian_beam_peak_amplitude_1d0_beam_'\
            'width_9d0_deg_at_150d0_MHz/'

    def add_power_to_diagonals(self, T_Ninv_T_block, PhiI_block):
        """
        Add a matrix and a vector reshaped as a diagonal matrix.

        Parameters
        ----------
        T_Ninv_T_block : np.ndarray
            Single block matrix from block_T_Ninv_T.
        PhiI_block : np.ndarray
            Vector of the estimated inverse variance of each |k| voxel present
            in `T_Ninv_T_block`.

        Returns
        -------
        matrix_sum : np.ndarray
            Sum of `T_Ninv_T_block` and a diagonal matrix constructed from
            `PhiI_block`.

        """
        matrix_sum = T_Ninv_T_block + np.diag(PhiI_block)
        return matrix_sum

    def calc_physical_dimensionless_power_spectral_normalisation(self, i_bin):
        """
        This normalization will calculate PowerI, an estimate for one over
        the variance of a in units of 1 / (mK**2 sr**2 Hz**2).

        Parameters
        ----------
        i_bin : int
            Input spherically averaged k-bin index.

        Returns
        -------
        dmps_norm : float
            Dimensionless power spectrum normalization with units of
            1 / (sr**2 Hz**2).
        """
        volume = (
            self.ps_box_size_ra_Mpc
            * self.ps_box_size_dec_Mpc
            * self.ps_box_size_para_Mpc
        )

        # Normalization calculated relative to mean
        # of vector k within the i_bin-th k-bin
        dmps_norm = self.k_vals[i_bin]**3./(2*np.pi**2) / volume

        # Redshift dependent quantities
        nu_array_MHz = p.nu_min_MHz + np.arange(p.nf)*p.channel_width_MHz
        cosmo = Cosmology()
        z = cosmo.f2z(nu_array_MHz.mean()*1e6)
        inst_to_cosmo_vol = cosmo.inst_to_cosmo_vol(z)
        dmps_norm *= inst_to_cosmo_vol**2

        return dmps_norm

    def calc_PowerI(self, x):
        """
        Calculate an estimate of the variance of the k-cube (uveta cube) from
        a set of power spectrum k-bin amplitudes `x`.

        Place restrictions on the power in the long spectral scale
        model either for,

        Parameters
        ----------
        x : array_like
            Input power spectrum amplitudes per |k|-bin with length `nDims`.

        Returns
        -------
        PhiI : np.ndarray
            Vector with estimates of the inverse variance of each |k| voxel.

        Notes
        -----
        The indices used are correct for the current ordering of basis vectors
        when nf is an even number.

        """
        PowerI = np.zeros(self.Npar)

        if p.include_instrumental_effects:
            q0_index = self.neta//2
        else:
            q0_index = self.nf//2 - 1
        q1_index = self.neta
        q2_index = self.neta + 1
        if self.use_shg:
            cg_end = self.nuv*self.nf
        else:
            cg_end = None

        # Constrain LW mode amplitude distribution
        dimensionless_PS_scaling =\
            self.calc_physical_dimensionless_power_spectral_normalisation(0)
        if p.use_LWM_Gaussian_prior:
            Fourier_mode_start_index = 3
            PowerI[:cg_end][q0_index::self.neta+self.nq] =\
                np.mean(dimensionless_PS_scaling) / x[0]
            PowerI[:cg_end][q1_index::self.neta+self.nq] =\
                np.mean(dimensionless_PS_scaling) / x[1]
            PowerI[:cg_end][q2_index::self.neta+self.nq] =\
                np.mean(dimensionless_PS_scaling) / x[2]
        else:
            Fourier_mode_start_index = 0
            # Set to zero for a uniform distribution
            PowerI[:cg_end][q0_index::self.neta+self.nq] =\
                self.inverse_LW_power
            PowerI[:cg_end][q1_index::self.neta+self.nq] =\
                self.inverse_LW_power
            PowerI[:cg_end][q2_index::self.neta+self.nq] =\
                self.inverse_LW_power

            if self.inverse_LW_power == 0.0:
                # Set to zero for a uniform distribution
                PowerI[:cg_end][q0_index::self.neta+self.nq] =\
                    self.inverse_LW_power_zeroth_LW_term
                PowerI[:cg_end][q1_index::self.neta+self.nq] =\
                    self.inverse_LW_power_first_LW_term
                PowerI[:cg_end][q2_index::self.neta+self.nq] =\
                    self.inverse_LW_power_second_LW_term

        if self.use_shg:
            # This should not be a permanent fix
            # Is a minimal prior on the SHG amplitudes
            # and LSSM the right choice?
            PowerI[cg_end:] = self.inverse_LW_power

        if self.dimensionless_PS:
            self.power_spectrum_normalisation_func =\
                self.calc_physical_dimensionless_power_spectral_normalisation
        else:
            self.power_spectrum_normalisation_func =\
                self.calc_Npix_physical_power_spectrum_normalisation

        # Fit for Fourier mode power spectrum
        for i_bin in range(len(self.k_cube_voxels_in_bin)):
            power_spectrum_normalisation =\
                self.power_spectrum_normalisation_func(i_bin)
            # NOTE: fitting for power not std here
            PowerI[self.k_cube_voxels_in_bin[i_bin]] = (
                    power_spectrum_normalisation
                    / x[Fourier_mode_start_index+i_bin])

        return PowerI

    def calc_Sigma_block_diagonals(self, T_Ninv_T, PhiI):
        """
        Constructs Sigma using block diagonal components like `block_T_Ninv_T`.

        Parameters
        ----------
        T_Ninv_T : np.ndarray
            Complex matrix product `T.conjugate().T * Ninv * T`.
        PhiI : np.ndarray
            Vector with estimates of the inverse variance of each |k| voxel.

        Returns
        -------
        Sigma_block_diagonals : np.ndarray
            Array containing a block diagonal representation of
            `Sigma = T_Ninv_T + PhiI`.  Each block is an entry along the zeroth
            axis of `Sigma_block_diagonals`.  This allows `Sigma` to be
            inverted numerically block by block as opposed to all at once.

        """
        PhiI_blocks = np.split(PhiI, self.nuv)
        Sigma_block_diagonals = np.array(
            [self.add_power_to_diagonals(
                T_Ninv_T[
                    (self.neta + self.nq)*i_block:
                        (self.neta + self.nq)*(i_block+1),
                    (self.neta + self.nq)*i_block:
                        (self.neta + self.nq)*(i_block+1)
                    ],
                PhiI_blocks[i_block]
                )
                for i_block in range(self.nuv)]
            )
        return Sigma_block_diagonals

    def calc_SigmaI_dbar_wrapper(self, x, T_Ninv_T, dbar, block_T_Ninv_T=[]):
        """
        Wrapper of `calc_SigmaI_dbar` which calculates `SigmaI_dbar` via an
        inversion per block diagonal element or as a single block-diagonal
        matrix via `calc_SigmaI_dbar`.

        Parameters
        ----------
        x : array_like
            Input power spectrum amplitudes per |k|-bin with length `nDims`.
        T_Ninv_T : np.ndarray
            Complex matrix product `T.conjugate().T * Ninv * T`.
        dbar : np.ndarray
            Noise weighted representation of the data (signal + noise) vector
            of visibilities in model (u, v, eta) space.
        block_T_Ninv_T : list
            Block diagonal representation of `T_Ninv_T`.  Only used if
            ``p.use_instrumental_effects = False``.  Defaults to `[]`.

        Returns
        -------
        SigmaI_dbar : np.ndarray
            Complex array of maximum likelihood (u, v, eta) and Large Spectral
            Scale Model (LSSM) amplitudes.  Used to compute the model data
            vector via `m = T * SigmaI_dbar`.
        dbarSigmaIdbar : np.ndarray
            Complex array product `dbar * SigmaI_dbar`.
        PhiI : np.ndarray
            Vector with estimates of the inverse variance of each |k| voxel.
        logSigmaDet : np.ndarray
            Natural logarithm of the determinant of `Sigma = T_Ninv_T + PhiI`.

        """
        start = time.time()
        PowerI = self.calc_PowerI(x)
        PhiI = PowerI
        if self.Print:
            mpiprint('\tPhiI time: {}'.format(time.time()-start),
                     rank=self.rank)

        do_block_diagonal_inversion = len(np.shape(block_T_Ninv_T)) > 1
        if do_block_diagonal_inversion:
            if self.Print:
                mpiprint('Using block-diagonal inversion', rank=self.rank)
            start = time.time()
            if self.intrinsic_noise_fitting:
                # This is only valid if the data is uniformly weighted
                Sigma_block_diagonals = self.calc_Sigma_block_diagonals(
                    T_Ninv_T/self.alpha_prime**2., PhiI)
            else:
                Sigma_block_diagonals = self.calc_Sigma_block_diagonals(
                    T_Ninv_T, PhiI)
            if self.Print:
                mpiprint('Time taken: {}'.format(time.time()-start),
                         rank=self.rank)

            start = time.time()
            dbar_blocks = np.split(dbar, self.nuv)
            if p.useGPU:
                if self.Print:
                    mpiprint('Computing block diagonal inversion on GPU',
                             rank=self.rank)
                SigmaI_dbar_blocks_and_logdet_Sigma = np.array(
                    [self.calc_SigmaI_dbar(
                        Sigma_block_diagonals[i_block],
                        dbar_blocks[i_block],
                        x_for_error_checking=x
                        )
                     for i_block in range(self.nuv)]
                    )
                SigmaI_dbar_blocks = np.array(
                    [SigmaI_dbar_block
                     for SigmaI_dbar_block, logdet_Sigma
                     in SigmaI_dbar_blocks_and_logdet_Sigma]
                    )
                logdet_Sigma_blocks = SigmaI_dbar_blocks_and_logdet_Sigma[:, 1]
            else:
                SigmaI_dbar_blocks = np.array(
                    [self.calc_SigmaI_dbar(
                        Sigma_block_diagonals[i_block],
                        dbar_blocks[i_block],
                        x_for_error_checking=x
                        )
                     for i_block in range(self.nuv)]
                    )
            if self.Print:
                mpiprint('Time taken: {}'.format(time.time()-start),
                         rank=self.rank)

            SigmaI_dbar = SigmaI_dbar_blocks.flatten()
            dbarSigmaIdbar = np.dot(dbar.conjugate().T, SigmaI_dbar)
            if self.Print:
                mpiprint('Time taken: {}'.format(time.time()-start),
                         rank=self.rank)

            if p.useGPU:
                logSigmaDet = np.sum(logdet_Sigma_blocks)
            else:
                logSigmaDet = np.sum(
                    [np.linalg.slogdet(Sigma_block)[1]
                     for Sigma_block in Sigma_block_diagonals]
                    )
                if self.Print:
                    mpiprint('Time taken: {}'.format(time.time()-start),
                             rank=self.rank)

        else:
            if self.count % self.print_rate == 0:
                mpiprint('Not using block-diagonal inversion', rank=self.rank)
            start = time.time()
            # Note: the following two lines can probably be speeded up
            # by adding T_Ninv_T and np.diag(PhiI). (Should test this!)
            # but this else statement only occurs on the GPU inversion
            # so will deal with it later.
            Sigma = T_Ninv_T.copy()
            if self.intrinsic_noise_fitting:
                Sigma = Sigma/self.alpha_prime**2.0

            Sigma[self.Sigma_Diag_Indices] += PhiI
            if self.Print:
                mpiprint('\tSigma build time: {}'.format(time.time()-start),
                         rank=self.rank)
            if self.return_Sigma:
                return Sigma

            start = time.time()
            if p.useGPU:
                if self.Print:
                    mpiprint('Computing matrix inversion on GPU',
                             rank=self.rank)
                SigmaI_dbar_and_logdet_Sigma = self.calc_SigmaI_dbar(
                    Sigma, dbar, x_for_error_checking=x)
                SigmaI_dbar = SigmaI_dbar_and_logdet_Sigma[0]
                logdet_Sigma = SigmaI_dbar_and_logdet_Sigma[1]
            else:
                SigmaI_dbar = self.calc_SigmaI_dbar(
                    Sigma, dbar, x_for_error_checking=x)
            if self.Print:
                mpiprint(
                    '\tcalc_SigmaI_dbar time: {}'.format(time.time()-start),
                    rank=self.rank
                )

            start = time.time()
            dbarSigmaIdbar = np.dot(dbar.conjugate().T, SigmaI_dbar)
            if self.Print:
                mpiprint(
                    '\tdbarSigmaIdbar time: {}'.format(time.time()-start),
                    rank=self.rank
                )

            start = time.time()
            if p.useGPU:
                logSigmaDet = logdet_Sigma
            else:
                logSigmaDet = np.linalg.slogdet(Sigma)[1]
            if self.Print:
                mpiprint(
                    '\tlogSigmaDet time: {}'.format(time.time()-start),
                    rank=self.rank
                )

        return SigmaI_dbar, dbarSigmaIdbar, PhiI, logSigmaDet

    def calc_SigmaI_dbar(self, Sigma, dbar, x_for_error_checking=[]):
        """
        Solves the linear system `Sigma * a = dbar` by calculating the Cholesky
        decomposition (if ``p.useGPU = True``) of the matrix'
        `Sigma = T_Ninv_T + PhiI`.  If not using GPUs to perform the matrix
        inversion, `scipy.linalg.inv` will be used.  This is not desired as the
        CPU based `scipy.linalg.inv` function does not always return the "true"
        matrix inverse for the matrices used here.

        Parameters
        ----------
        Sigma : np.ndarray
            Complex matrix `Sigma = T_Ninv_T + PhiI`.
        dbar : np.ndarray
            Noise weighted representation of the data (signal + noise) vector
            of visibilities in model (u, v, eta) space.
        x_for_error_checking : array_like
            Input power spectrum amplitudes per |k|-bin with length `nDims`
            used for error checking of the matrix inversion.  Defaults to `[]`.

        Returns
        -------
        SigmaI_dbar : np.ndarray
            Complex array of maximum likelihood (u, v, eta) and Large Spectral
            Scale Model (LSSM) amplitudes.  Used to compute the model data
            vector via `m = T * SigmaI_dbar`.
        logdet_Magma_Sigma : np.ndarray
            Natural logarith of the determinant of `Sigma`.  Only returned if
            ``p.useGPU = True``.

        """
        if not p.useGPU:
            # Sigmacho = scipy.linalg.cholesky(
            #         Sigma, lower=True
            #     ).astype(np.complex256)
            # SigmaI_dbar = scipy.linalg.cho_solve((Sigmacho, True), dbar)
            # scipy.linalg.inv is not numerically stable
            # USE WITH CAUTION
            SigmaI = scipy.linalg.inv(Sigma)
            SigmaI_dbar = np.dot(SigmaI, dbar)
            return SigmaI_dbar

        else:
            # brk()
            dbar_copy = dbar.copy()
            dbar_copy_copy = dbar.copy()
            self.GPU_error_flag = np.array([0])
            if self.Print:
                start = time.time()
            # Replace 0 with 1 to pring debug in the following command
            wrapmzpotrf.cpu_interface(
                len(Sigma), nrhs, Sigma, dbar_copy, 0, self.GPU_error_flag)
            if self.Print:
                mpiprint(
                    '\t\tCholesky decomposition time: {}'.format(
                        time.time() - start
                    ),
                    rank=self.rank
                )
            # Note: After wrapmzpotrf, Sigma is actually
            # SigmaCho (i.e. L with Sigma = LL^T)
            logdet_Magma_Sigma = np.sum(np.log(np.diag(abs(Sigma)))) * 2
            if self.Print:
                start = time.time()
            SigmaI_dbar = scipy.linalg.cho_solve(
                (Sigma.conjugate().T, True), dbar_copy_copy)
            if self.Print:
                mpiprint(
                    '\t\tscipy cho_solve time: {}'.format(
                        time.time() - start
                    ),
                    rank=self.rank
                )
            if self.GPU_error_flag[0] != 0:
                # If the inversion doesn't work, zero-weight the
                # sample (may want to stop computing if this occurs?)
                logdet_Magma_Sigma = +np.inf
                print(self.rank, ':', 'GPU inversion error. Setting sample '
                      'posterior probability to zero.')
                print(self.rank, ':', 'Param values: ', x_for_error_checking)
                print(self.rank, ':',
                      'GPU_error_flag = {}'.format(self.GPU_error_flag))

            return SigmaI_dbar, logdet_Magma_Sigma

    def posterior_probability(self, x, block_T_Ninv_T=[]):
        """
        Computes the posterior probability of a given set of power spectrum
        amplitudes per |k|-bin.

        Parameters
        ----------
        x : array_like
            Input power spectrum amplitudes per |k|-bin with length `nDims`.
        block_T_Ninv_T : list
            Block diagonal representation of `T_Ninv_T`.  Only used if
            ``p.use_instrumental_effects = False``.  Defaults to `[]`.

        Returns
        -------
        MargLogL.squeeze()*1.0, phi
        MargLogL : float
            Posterior probability of `x`.
        phi :
            Only used if sampling with `PolyChord`.

        """
        if self.debug:
            brk()
        phi = [0.0]
        T_Ninv_T = self.T_Ninv_T
        dbar = self.dbar

        if self.fit_for_spectral_model_parameters:
            Print = self.Print
            self.Print = True
            pl_params = x[:2]
            x = x[2:]
            if self.Print:
                mpiprint('pl_params', pl_params, rank=self.rank)
            if self.Print:
                mpiprint('p.pl_grid_spacing, p.pl_max',
                         p.pl_grid_spacing, p.pl_max,
                         rank=self.rank)
            b1 = pl_params[0]
            b2 = (b1
                  + p.pl_grid_spacing
                  + (p.pl_max - b1 - p.pl_grid_spacing)*pl_params[1]
                  )
            # Round derived pl indices to nearest p.pl_grid_spacing
            b1, b2 = (p.pl_grid_spacing
                      * np.round(np.array([b1, b2]) / p.pl_grid_spacing, 0))
            if self.Print:
                mpiprint('b1, b2', b1, b2, rank=self.rank)

            # Load matrices associated with sampled beta params
            T_Ninv_T_dataset_name =\
                'T_Ninv_T_b1_{}_b2_{}'.format(b1, b2).replace('.', 'd')
            T_Ninv_T_file_path = (
                    self.spectral_model_parameters_array_storage_dir
                    + T_Ninv_T_dataset_name
                    + '.h5')
            if self.count % self.print_rate == 0:
                mpiprint('Replacing T_Ninv_T with:', T_Ninv_T_file_path,
                         rank=self.rank)
            start = time.time()
            with h5py.File(T_Ninv_T_file_path, 'r') as hf:
                T_Ninv_T = hf[T_Ninv_T_dataset_name][:]
                # alpha_prime = p.sigma/170000.0
                # This is only valid if the data is uniformly weighted
                # T_Ninv_T = T_Ninv_T/(alpha_prime**2.0)
                if self.count % self.print_rate == 0:
                    mpiprint('Time taken: {}'.format(time.time() - start),
                             rank=self.rank)

            dbar_dataset_name =\
                'dbar_b1_{}_b2_{}'.format(b1, b2).replace('.', 'd')
            dbar_file_path = (
                    self.spectral_model_parameters_array_storage_dir
                    + dbar_dataset_name
                    + '.h5')
            if self.count % self.print_rate == 0:
                mpiprint('Replacing dbar with:', dbar_file_path,
                         rank=self.rank)
            start = time.time()
            with h5py.File(dbar_file_path, 'r') as hf:
                dbar = hf[dbar_dataset_name][:]
                # alpha_prime = p.sigma/170000.0
                # This is only valid if the data is uniformly weighted
                # dbar = dbar/(alpha_prime**2.0)
                # if self.count % self.print_rate == 0:
                #     mpiprint('alpha_prime = ', alpha_prime, rank=self.rank)
                if self.count % self.print_rate == 0:
                    mpiprint('Time taken: {}'.format(time.time() - start),
                             rank=self.rank)
            self.Print = Print

        if self.intrinsic_noise_fitting:
            self.alpha_prime = x[0]
            x = x[1:]
            Ndat = self.Ninv.diagonal().size
            # The following is only valid if
            # the data is uniformly weighted
            log_det_N = Ndat * np.log(self.alpha_prime**2.0)
            d_Ninv_d = self.d_Ninv_d / (self.alpha_prime**2.0)
            dbar = dbar / (self.alpha_prime**2.0)

        if self.log_priors:
            x = 10.**np.array(x)

        # brk()
        do_block_diagonal_inversion = len(np.shape(block_T_Ninv_T)) > 1
        self.count += 1
        start_call = time.time()
        try:
            if do_block_diagonal_inversion:
                if self.Print:
                    mpiprint('Using block-diagonal inversion', rank=self.rank)
                SigmaI_dbar, dbarSigmaIdbar, PhiI, logSigmaDet =\
                    self.calc_SigmaI_dbar_wrapper(
                        x, T_Ninv_T, dbar, block_T_Ninv_T=block_T_Ninv_T)
            else:
                if self.Print:
                    mpiprint('Not using block-diagonal inversion',
                             rank=self.rank)
                start = time.time()
                SigmaI_dbar, dbarSigmaIdbar, PhiI, logSigmaDet =\
                    self.calc_SigmaI_dbar_wrapper(x, T_Ninv_T, dbar)
                if self.Print:
                    mpiprint(
                        'calc_SigmaI_dbar_wrapper time: {}'.format(
                            time.time() - start
                        ),
                        rank=self.rank
                    )

            # Only possible because Phi is diagonal (otherwise would
            # need to calc np.linalg.slogdet(Phi)). -1 factor is to get
            # logPhiDet from logPhiIDet. Note: the real part of this
            # calculation matches the solution given by
            # np.linalg.slogdet(Phi))
            start = time.time()
            logPhiDet = -1 * np.sum(np.log(PhiI)).real
            if self.Print:
                mpiprint('logPhiDet time: {}'.format(time.time() - start),
                         rank=self.rank)

            start = time.time()
            MargLogL = -0.5*logSigmaDet - 0.5*logPhiDet + 0.5*dbarSigmaIdbar
            if np.abs(self.n_uniform_prior_k_bins) > 0:
                # Specific bins use a uniform prior
                if self.n_uniform_prior_k_bins > 0:
                    MargLogL += np.sum(
                        np.log(x[:self.n_uniform_prior_k_bins])
                    )
                else:
                    MargLogL += np.sum(
                        np.log(x[self.n_uniform_prior_k_bins:])
                    )
            elif self.uniform_priors:
                # All bins use a uniform prior
                MargLogL += np.sum(np.log(x))
            if self.intrinsic_noise_fitting:
                MargLogL = MargLogL - 0.5*d_Ninv_d - 0.5*log_det_N
            MargLogL = MargLogL.real
            if self.Print:
                mpiprint('MargLogL time: {}'.format(time.time() - start),
                         rank=self.rank)
            if self.Print_debug:
                MargLogL_equation_string = \
                    'MargLogL = -0.5*logSigmaDet '\
                    '-0.5*logPhiDet + 0.5*dbarSigmaIdbar'
                if self.intrinsic_noise_fitting:
                    print(self.rank, ':', 'Using intrinsic noise fitting')
                    MargLogL_equation_string +=\
                        ' - 0.5*d_Ninv_d -0.5*log_det_N'
                    print(
                        self.rank, ':',
                        'logSigmaDet, logPhiDet, dbarSigmaIdbar, '
                        'd_Ninv_d, log_det_N',
                        logSigmaDet, logPhiDet, dbarSigmaIdbar,
                        d_Ninv_d, log_det_N
                    )
                else:
                    print(
                        self.rank, ':',
                        'logSigmaDet, logPhiDet, dbarSigmaIdbar',
                        logSigmaDet, logPhiDet, dbarSigmaIdbar
                    )
                print(self.rank, ':', MargLogL_equation_string, MargLogL)
                print(self.rank, ':', 'MargLogL.real', MargLogL.real)

            # brk()

            if self.nu > 10:
                self.print_rate = 100
            if self.count % self.print_rate == 0:
                mpiprint('count', self.count, rank=self.rank)
                print(
                    self.rank, ':',
                    'Time since class instantiation: {}'.format(
                        time.time() - self.instantiation_time
                    )
                )
                print(
                    self.rank, ':',
                    'Time for this likelihood call: {}'.format(
                        time.time() - start_call
                    )
                )
            return MargLogL.squeeze()*1.0, phi
        except Exception as e:
            # This won't catch a warning if, for example, PhiI contains
            # any zeros in np.sum(np.log(PhiI))
            print(self.rank, ':', 'Exception encountered...')
            print(self.rank, ':', repr(e))
            return -np.inf, -1
