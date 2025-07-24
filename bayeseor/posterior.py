import time
import numpy as np
import scipy
from pdb import set_trace as brk
import h5py

from .cosmology import Cosmology
from .gpu import GPUInterface
from .utils import mpiprint


"""
Potentially useful links:
http://www.mrao.cam.ac.uk/~kjbg1/lectures/lect1_1.pdf
"""


class PriorC(object):
    """
    Prior class for MultiNest and PolyChord sampler compatibility.

    """
    def __init__(self, priors_min_max):
        self.priors_min_max = priors_min_max

    def prior_func(self, cube):
        pmm = self.priors_min_max
        theta = []
        for i_p in range(len(cube)):
            theta_i = pmm[i_p][0] + ((pmm[i_p][1] - pmm[i_p][0]) * cube[i_p])
            theta.append(theta_i)
        return theta

class PowerSpectrumPosteriorProbability(object):
    r"""
    Class containing posterior calculation functions.

    Parameters
    ----------
    T_Ninv_T : numpy.ndarray
        Complex matrix product
        :math:`\mathbf{T}^\dagger\mathbf{N}^{-1}\mathbf{T}`. Used to construct
        `Sigma` which is used to solve for the maximum a posteriori 3D Fourier
        space vector and large spectral scale model amplitudes. `T_Ninv_T`
        must be a Hermitian, positive-definite matrix.
    dbar : numpy.ndarray
        Noise weighted representation of the data vector (signal + noise) of
        visibilities in model (u, v, eta) space computed as
        :math:`\bar{\boldsymbol{d}}=\mathbf{T}^\dagger\mathbf{N}^{-1}\mathbf{d}`.
    k_vals : numpy.ndarray
        Mean of each :math:`k` bin.
    k_cube_voxels_in_bin : list
        List containing sublists for each :math:`k` bin.  Each sublist contains
        the flattened 3D :math:`k`-cube index of all :math:`\vec{k}` that fall
        within a given :math:`k` bin.
    nuv : int
        Number of model uv-plane pixels per frequency channel in the EoR model.
    neta : int
        Number of line-of-sight (frequency axis) Fourier modes.
    nf : int
        Number of frequency channels.
    nq : int
        Number of quadratic modes in the large spectral scale model.
    Ninv : numpy.ndarray
        Noise covariance matrix.
    d_Ninv_d : numpy.ndarray
        Matrix-vector product
        :math:`\boldsymbol{d}^\dagger\mathbf{N}^{-1}\boldsymbol{d}`.
    ps_box_size_ra_Mpc : float
        Right ascension axis extent of the cosmological volume in Mpc from
        which the power spectrum is estimated.
    ps_box_size_dec_Mpc : float
        Declination axis extent of the cosmological volume in Mpc from which
        the power spectrum is estimated.
    ps_box_size_para_Mpc : float
        Line-of-sight extent of the cosmological volume in Mpc from which the
        power spectrum is estimated.
    include_instrumental_effects : bool, optional
        Forward model an instrument. Defaults to True.
    log_priors : bool, optional
        Assume priors on power spectrum coefficients are in log_10 units.
        Defaults to False.
    uprior_inds : numpy.ndarray, optional
        Boolean 1D array that is True for any :math:`k` bins using a uniform
        prior. False entries use a log-uniform prior. Defaults to None (all
        :math:`k` bins use a log-uniform prior).
    use_LWM_Gaussian_prior : bool, optional
        Use a Gaussian prior on the large spectral scale model (NOT
        IMPLEMENTED). Otherwise, use a uniform prior. Defaults to False.
    inverse_LW_power : float, optional
        Inverse prior on the large spectral scale model modes.  Defaults to
        1e-16 (infinite variance on the large spectral scale model modes).
    dimensionless_PS : bool, optional
        Fit for the dimensionless power spectrum, :math:`\Delta^2(k)` (True),
        or the power spectrum, :math:`P(k)` (False). Defaults to True.
    block_T_Ninv_T : list, optional
        Block diagonal representation of `T_Ninv_T`. Only used if
        `include_instrumental_effects` is False. Defaults to None.
    intrinsic_noise_fitting : bool, optional
        Fit for the amplitude of the noise in the data instead of using the
        noise covariance matrix, `Ninv`. Defaults to False.
    fit_for_spectral_model_parameters : bool, optional
        Fit for the large spectral scale model parameter values.  Defaults to
        False.
    pl_max : float, optional
        Maximum brightness temperature spectral index when fitting for the
        optimal large spectral scale model spectral indices. Defaults to None.
    pl_grid_spacing : float, optional
        Grid spacing for the power law spectral index axis when fitting for the
        large spectral scale model parameter values. Defaults to None.
    use_shg : bool, optional
        Use the subharmonic grid. Defaults to False.
    return_Sigma : bool, optional
        Break and return the matrix `Sigma = T_Ninv_T + PhiI`, where `PhiI` is
        the inverse prior covariance matrix. Defaults to False.
    rank : int, optional
        MPI rank. Defaults to 0.
    use_gpu : bool, optional
        Use GPUs for the Cholesky decomposition of `Sigma`. Otherwise, use
        CPUs (inadvisable due to potential inaccuracy of CPU matrix inversion).  
        Defaults to True.
    verbose : bool, optional
        Verbose output. Defaults to False.
    print_rate : int, optional
        Number of iterations between print statements. Defaults to 100.
    debug : bool, optional
        Execute break statements for debugging. Defaults to False.
    print_debug : bool, optional
        Print debug related messages that are more detailed than verbose
        output. Defaults to False.

    """
    def __init__(
        self,
        T_Ninv_T,
        dbar,
        k_vals,
        k_cube_voxels_in_bin,
        nuv,
        neta,
        nf,
        nq,
        Ninv,
        d_Ninv_d,
        redshift,
        ps_box_size_ra_Mpc,
        ps_box_size_dec_Mpc,
        ps_box_size_para_Mpc,
        include_instrumental_effects=True,
        log_priors=False,
        uprior_inds=None,
        use_LWM_Gaussian_prior=False,
        inverse_LW_power=1e-16,
        dimensionless_PS=True,
        block_T_Ninv_T=None,
        intrinsic_noise_fitting=False,
        fit_for_spectral_model_parameters=False,
        pl_max=None,
        pl_grid_spacing=None,
        use_shg=False,
        return_Sigma=False,
        rank=0,
        use_gpu=True,
        verbose=False,
        print_rate=100,
        debug=False,
        print_debug=False
    ):
        self.instantiation_time = time.time()

        # Required params
        self.T_Ninv_T = T_Ninv_T
        self.dbar = dbar
        self.k_vals = k_vals
        self.k_cube_voxels_in_bin = k_cube_voxels_in_bin
        self.nuv = nuv
        self.neta = neta
        self.nf = nf
        self.nq = nq
        self.Ninv = Ninv
        self.d_Ninv_d = d_Ninv_d
        self.redshift = redshift
        self.ps_box_size_ra_Mpc = ps_box_size_ra_Mpc
        self.ps_box_size_dec_Mpc = ps_box_size_dec_Mpc
        self.ps_box_size_para_Mpc = ps_box_size_para_Mpc
        # Optional params
        self.include_instrumental_effects = include_instrumental_effects
        self.log_priors = log_priors
        self.uprior_inds = uprior_inds
        self.use_LWM_Gaussian_prior = use_LWM_Gaussian_prior
        self.inverse_LW_power = inverse_LW_power
        self.dimensionless_PS = dimensionless_PS
        if block_T_Ninv_T is None:
            self.block_T_Ninv_T = []
        else:
            self.block_T_Ninv_T = block_T_Ninv_T
        self.intrinsic_noise_fitting = intrinsic_noise_fitting
        self.fit_for_spectral_model_parameters =\
            fit_for_spectral_model_parameters
        if self.fit_for_spectral_model_parameters:
            req_params = np.all(
                [x is not None for x in [pl_max, pl_grid_spacing]]
            )
            if self.rank == 0:
                assert req_params, (
                    "If fit_for_spectral_model_parameters is True, must pass "
                    "both pl_max and pl_grid_spacing."
                )
            self.pl_max = pl_max
            self.pl_grid_spacing = pl_grid_spacing
        self.use_shg = use_shg
        self.return_Sigma = return_Sigma
        self.rank = rank
        self.verbose = verbose
        self.print_rate = print_rate
        self.debug = debug
        self.print_debug = print_debug
        # Auxiliary params
        self.count = 0  # Iteration counter
        self.neor_uveta = self.nuv*(self.neta - 1)  # EoR model parameters
        self.Npar = T_Ninv_T.shape[0]  # EoR+FG model parameters
        self.Sigma_diag_inds = np.diag_indices(self.Npar)
        self.alpha_prime = 1.0  # Initial noise fitting parameter value
        # Conversion from observational k-cube voxel units [sr Hz] to comoving 
        # voxel units [Mpc^3]
        self.inst_to_cosmo_vol = Cosmology().inst_to_cosmo_vol(self.redshift)
        # Volume of model image cube in comoving coordinates
        self.cosmo_volume = (
            self.ps_box_size_ra_Mpc
            * self.ps_box_size_dec_Mpc
            * self.ps_box_size_para_Mpc
        )
        # FIXME: add argument for location of pre-computed(?) arrays when
        # fitting for spectral model parameters?
        self.spectral_model_parameters_array_storage_dir = ''

        if self.verbose:
            if self.log_priors:
                mpiprint('Using log-priors', rank=self.rank)
            if self.dimensionless_PS:
                mpiprint(
                    'Calculating the dimensionless power spectrum',
                    rank=self.rank
                )
            mpiprint(
                f"Setting inverse_LW_power to {self.inverse_LW_power}",
                rank=self.rank
            )
        
        if use_gpu:
            # Initialize the GPU interface
            self.gpu = GPUInterface(rank=self.rank, verbose=self.verbose)
            if self.gpu.gpu_initialized:
                self.use_gpu = True
            else:
                # GPU initilization failed, use CPU methods instead
                self.use_gpu = False
        else:
            self.use_gpu = False

    def add_power_to_diagonals(self, T_Ninv_T_block, PhiI_block):
        """
        Add a matrix and a vector reshaped as a diagonal matrix.

        Used only if `T_Ninv_T` has been constructed as a block-diagonal
        matrix, i.e. if ignoring instrumental effects.

        Parameters
        ----------
        T_Ninv_T_block : numpy.ndarray
            Single block matrix from block_T_Ninv_T.
        PhiI_block : numpy.ndarray
            Vector of the estimated inverse variance of each :math:`k` voxel
            present in `T_Ninv_T_block`.

        Returns
        -------
        matrix_sum : numpy.ndarray
            Sum of `T_Ninv_T_block` and a diagonal matrix constructed from
            `PhiI_block`.

        """
        matrix_sum = T_Ninv_T_block + np.diag(PhiI_block)
        return matrix_sum

    def calc_physical_dimensionless_power_spectral_normalisation(self, i_bin):
        """
        Dimensionless power spectrum normalization in physical units.

        This normalization will return `PowerI`, an estimate for one over the
        variance of `a`, in units of 1 / (mK^2 sr^2 Hz^2).

        Parameters
        ----------
        i_bin : int
            Spherically averaged :math:`k` bin index.

        Returns
        -------
        dmps_norm : float
            Dimensionless power spectrum normalization with units of
            1 / (sr**2 Hz**2).

        """
        # Normalization calculated relative to mean of vector k within the
        # i_bin-th k bin
        dmps_norm = self.k_vals[i_bin]**3./(2*np.pi**2) / self.cosmo_volume

        # Redshift dependent quantities
        dmps_norm *= self.inst_to_cosmo_vol**2

        return dmps_norm

    def calc_PowerI(self, x):
        """
        Calculate an estimate of the inverse variance of the model.

        Given a power spectrum, this function returns an estimate of the
        inverse variance of each voxel in the (u, v, eta) model which is used
        as a prior in the posterior probability calculation.

        Parameters
        ----------
        x : array_like
            Input power spectrum amplitudes per :math:`k` bin with length
            `nDims`.

        Returns
        -------
        PhiI : numpy.ndarray
            Vector with estimates of the inverse variance of each model
            (u, v, eta) voxel.

        """
        PowerI = np.zeros(self.Npar)

        # FIXME: these variables, q?_index, are remnants of the old model
        # vector ordering scheme.  This function needs to be updated and have
        # the deprecated variables removed.
        if self.include_instrumental_effects:
            q0_index = self.neta//2
        else:
            q0_index = self.nf//2 - 1
        q1_index = self.neta
        q2_index = self.neta + 1
        if self.use_shg:
            # FIXME: Needs to be updated for the new separate EoR and FG
            # models if used in the future!!!
            cg_end = self.nuv*(self.neta - 1)
        else:
            cg_end = None

        # Constrain LW mode amplitude distribution
        dimensionless_PS_scaling =\
            self.calc_physical_dimensionless_power_spectral_normalisation(0)
        # FIXME: Need to update the code in this if/else block to account
        # for the separate EoR and FG models if used in the future.  There is
        # currently no way to implement separate priors on the individual large
        # spectral scale model (LSSM) parameters.  Currently, the prior is set
        # to be identical accross all LSSM parameters.
        if self.use_LWM_Gaussian_prior:
            Fourier_mode_start_index = 3
            PowerI[:cg_end][q0_index::self.neta+self.nq] =\
                np.mean(dimensionless_PS_scaling) / x[0]
            PowerI[:cg_end][q1_index::self.neta+self.nq] =\
                np.mean(dimensionless_PS_scaling) / x[1]
            PowerI[:cg_end][q2_index::self.neta+self.nq] =\
                np.mean(dimensionless_PS_scaling) / x[2]
        else:
            Fourier_mode_start_index = 0
            # Set to zero (1e-16) for a uniform distribution
            PowerI[self.neor_uveta:] = self.inverse_LW_power

        if self.use_shg:
            # FIXME: Needs to be updated for the new separate EoR and FG
            # models if used in the future!!!
            # This should not be a permanent fix
            # Is a minimal prior on the SHG amplitudes
            # and LSSM the right choice?
            PowerI[cg_end:] = self.inverse_LW_power

        if self.dimensionless_PS:
            self.power_spectrum_normalisation_func =\
                self.calc_physical_dimensionless_power_spectral_normalisation
        else:
            # FIXME: do we want to add support for modelling P(k)?
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

        Used only if `T_Ninv_T` has been constructed as a block-diagonal
        matrix, i.e. if ignoring instrumental effects.

        Parameters
        ----------
        T_Ninv_T : numpy.ndarray
            Complex matrix product ``T.conjugate().T * Ninv * T``.
        PhiI : numpy.ndarray
            Vector with estimates of the inverse variance of each model
            (u, v, eta) voxel.

        Returns
        -------
        Sigma_block_diagonals : numpy.ndarray
            Array containing a block diagonal representation of
            ``Sigma = T_Ninv_T + PhiI``.  Each block is an entry along the
            zeroth axis of `Sigma_block_diagonals`.  This allows `Sigma` to be
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
        Wrapper for `calc_SigmaI_dbar`.

        Constructs `Sigma` and calculates `SigmaI_dbar` and other quantities
        derived from `SigmaI_dbar`.  The actual solve step to obtain
        `SigmaI_dbar` is performed in `calc_SigmaI_sbar`.

        Parameters
        ----------
        x : array_like
            Input power spectrum amplitudes per :math:`k` bin with length
            `nDims`.
        T_Ninv_T : numpy.ndarray
            Complex matrix product ``T.conjugate().T * Ninv * T``.
        dbar : numpy.ndarray
            Noise weighted representation of the data (signal + noise) vector
            of visibilities in model (u, v, eta) space.
        block_T_Ninv_T : list
            Block diagonal representation of `T_Ninv_T`.  Only used if ignoring
            instrumental effects.  Defaults to an empty list.

        Returns
        -------
        SigmaI_dbar : numpy.ndarray
            Complex array of maximum a posteriori (u, v, eta) and Large
            Spectral Scale Model (LSSM) amplitudes.  Used to compute the model
            data vector via ``m = T * SigmaI_dbar``.
        dbarSigmaIdbar : numpy.ndarray
            Complex array product ``dbar * SigmaI_dbar``.
        PhiI : numpy.ndarray
            Vector with estimates of the inverse variance of each model
            (u, v, eta) voxel.
        logSigmaDet : numpy.ndarray
            Natural logarithm of the determinant of
            ``Sigma = T_Ninv_T + PhiI``.

        """
        start = time.time()
        PowerI = self.calc_PowerI(x)
        PhiI = PowerI
        if self.verbose:
            mpiprint('\tPhiI time: {}'.format(time.time()-start),
                     rank=self.rank)

        do_block_diagonal_inversion = len(np.shape(block_T_Ninv_T)) > 1
        if do_block_diagonal_inversion:
            if self.verbose:
                mpiprint('Using block-diagonal inversion', rank=self.rank)
            start = time.time()
            if self.intrinsic_noise_fitting:
                # This is only valid if the data is uniformly weighted
                Sigma_block_diagonals = self.calc_Sigma_block_diagonals(
                    T_Ninv_T/self.alpha_prime**2., PhiI)
            else:
                Sigma_block_diagonals = self.calc_Sigma_block_diagonals(
                    T_Ninv_T, PhiI)
            if self.verbose:
                mpiprint('Time taken: {}'.format(time.time()-start),
                         rank=self.rank)

            start = time.time()
            dbar_blocks = np.split(dbar, self.nuv)
            if self.use_gpu:
                if self.verbose:
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
            if self.verbose:
                mpiprint('Time taken: {}'.format(time.time()-start),
                         rank=self.rank)

            SigmaI_dbar = SigmaI_dbar_blocks.flatten()
            dbarSigmaIdbar = np.dot(dbar.conjugate().T, SigmaI_dbar)
            if self.verbose:
                mpiprint('Time taken: {}'.format(time.time()-start),
                         rank=self.rank)

            if self.use_gpu:
                logSigmaDet = np.sum(logdet_Sigma_blocks)
            else:
                logSigmaDet = np.sum(
                    [np.linalg.slogdet(Sigma_block)[1]
                     for Sigma_block in Sigma_block_diagonals]
                    )
                if self.verbose:
                    mpiprint('Time taken: {}'.format(time.time()-start),
                             rank=self.rank)

        else:
            if self.count % self.print_rate == 0 and self.verbose:
                mpiprint('Not using block-diagonal inversion', rank=self.rank)
            start = time.time()
            # Note: the following two lines can probably be speeded up
            # by adding T_Ninv_T and np.diag(PhiI). (Should test this!)
            # but this else statement only occurs on the GPU inversion
            # so will deal with it later.
            Sigma = T_Ninv_T.copy()
            if self.intrinsic_noise_fitting:
                Sigma = Sigma/self.alpha_prime**2.0

            Sigma[self.Sigma_diag_inds] += PhiI
            if self.verbose:
                mpiprint('\tSigma build time: {}'.format(time.time()-start),
                         rank=self.rank)
            if self.return_Sigma:
                return Sigma

            start = time.time()
            if self.use_gpu:
                if self.verbose:
                    mpiprint('Computing matrix inversion on GPU',
                             rank=self.rank)
                SigmaI_dbar_and_logdet_Sigma = self.calc_SigmaI_dbar(
                    Sigma, dbar, x_for_error_checking=x)
                SigmaI_dbar = SigmaI_dbar_and_logdet_Sigma[0]
                logdet_Sigma = SigmaI_dbar_and_logdet_Sigma[1]
            else:
                SigmaI_dbar = self.calc_SigmaI_dbar(
                    Sigma, dbar, x_for_error_checking=x)
            if self.verbose:
                mpiprint(
                    '\tcalc_SigmaI_dbar time: {}'.format(time.time()-start),
                    rank=self.rank
                )

            start = time.time()
            dbarSigmaIdbar = np.dot(dbar.conjugate().T, SigmaI_dbar)
            if self.verbose:
                mpiprint(
                    '\tdbarSigmaIdbar time: {}'.format(time.time()-start),
                    rank=self.rank
                )

            start = time.time()
            if self.use_gpu:
                logSigmaDet = logdet_Sigma
            else:
                logSigmaDet = np.linalg.slogdet(Sigma)[1]
            if self.verbose:
                mpiprint(
                    '\tlogSigmaDet time: {}'.format(time.time()-start),
                    rank=self.rank
                )

        return SigmaI_dbar, dbarSigmaIdbar, PhiI, logSigmaDet

    def calc_SigmaI_dbar(self, Sigma, dbar, x_for_error_checking=[]):
        """
        Solves the linear system `Sigma * a = dbar`.
        
        Solved by calculating the Cholesky decomposition (if `self.use_gpu` is
        True) of the matrix ``Sigma = T_Ninv_T + PhiI``.  If not using GPUs to
        perform the matrix inversion, ``scipy.linalg.inv`` will be used.  This
        is not desired as the CPU based ``scipy.linalg.inv`` function does not
        always return the "true" matrix inverse for the matrices used here.

        Parameters
        ----------
        Sigma : numpy.ndarray
            Complex matrix ``Sigma = T_Ninv_T + PhiI``.
        dbar : numpy.ndarray
            Noise weighted representation of the data (signal + noise) vector
            of visibilities in model (u, v, eta) space.
        x_for_error_checking : array_like
            Input power spectrum amplitudes per :math:`k` bin with length
            `nDims` used for error checking of the matrix inversion. Defaults
            to an empty list (no error checking).

        Returns
        -------
        SigmaI_dbar : numpy.ndarray
            Complex array of maximum a posteriori (u, v, eta) and Large
            Spectral Scale Model (LSSM) amplitudes.  Used to compute the model
            data vector via ``m = T * SigmaI_dbar``.
        logdet_Magma_Sigma : numpy.ndarray
            Natural logarith of the determinant of `Sigma`.  Only returned if
            `self.use_gpu` is True.

        """
        if not self.use_gpu:
            # scipy.linalg.inv is not numerically stable
            # USE WITH CAUTION
            SigmaI = scipy.linalg.inv(Sigma)
            SigmaI_dbar = np.dot(SigmaI, dbar)
            return SigmaI_dbar

        else:
            self.magma_info = np.array([0])
            if self.verbose:
                start = time.time()
            self.gpu.magma_init()
            # The MAGMA docs suggest that uplo = 121 corresponds to the
            # upper-triangular component of the matrix being store in
            # memory but testing shows 121 actually stores the lower-
            # triangular matrix in memory, which is what we use below.
            exit_status = self.gpu.magma_zpotrf(
                121,
                Sigma.shape[1],
                Sigma,
                Sigma.shape[0],
                self.magma_info
            )
            self.gpu.magma_finalize()
            if self.verbose:
                mpiprint(
                    f'\t\tCholesky decomposition time: {time.time() - start}',
                    rank=self.rank
                )
            # Note: After wrapmzpotrf, Sigma is actually
            # SigmaCho (i.e. L with Sigma = LL^T)
            logdet_Magma_Sigma = np.sum(np.log(np.diag(abs(Sigma)))) * 2
            if self.verbose:
                start = time.time()
            SigmaI_dbar = scipy.linalg.cho_solve((Sigma, True), dbar)
            if self.verbose:
                mpiprint(
                    '\t\tscipy cho_solve time: {}'.format(
                        time.time() - start
                    ),
                    rank=self.rank
                )
            if exit_status != 0:
                # If the inversion doesn't work, zero-weight the
                # sample (may want to stop computing if this occurs?)
                logdet_Magma_Sigma = np.inf
                print(self.rank, ':',
                      'GPU inversion error. '
                      'Setting sample posterior probability to zero.')
                print(self.rank, ':', 'Param values: ', x_for_error_checking)
                print(self.rank, ':', f'info = {self.magma_info[0]}')

            return SigmaI_dbar, logdet_Magma_Sigma

    def posterior_probability(self, x, block_T_Ninv_T=[]):
        """
        Computes the posterior probability for a given power spectrum sample.

        Parameters
        ----------
        x : array_like
            Input power spectrum amplitudes per :math:`k` bin with length
            `nDims`.
        block_T_Ninv_T : list
            Block diagonal representation of `T_Ninv_T`.  Only used if ignoring
            instrumental effects.  Defaults to an empty list.

        Returns
        -------
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
            pl_params = x[:2]
            x = x[2:]
            if self.verbose:
                mpiprint('pl_params', pl_params, rank=self.rank)
            if self.verbose:
                mpiprint('pl_grid_spacing, pl_max',
                         self.pl_grid_spacing, self.pl_max,
                         rank=self.rank)
            b1 = pl_params[0]
            b2 = (b1
                  + self.pl_grid_spacing
                  + (self.pl_max - b1 - self.pl_grid_spacing)*pl_params[1]
                  )
            # Round derived pl indices to nearest self.pl_grid_spacing
            b1, b2 = (self.pl_grid_spacing
                      * np.round(np.array([b1, b2]) / self.pl_grid_spacing, 0))
            if self.verbose:
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
                # This is only valid if the data is uniformly weighted
                # dbar = dbar/(alpha_prime**2.0)
                # if self.count % self.print_rate == 0:
                #     mpiprint('alpha_prime = ', alpha_prime, rank=self.rank)
                if self.count % self.print_rate == 0:
                    mpiprint('Time taken: {}'.format(time.time() - start),
                             rank=self.rank)

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

        do_block_diagonal_inversion = len(np.shape(block_T_Ninv_T)) > 1
        self.count += 1
        start_call = time.time()
        try:
            if do_block_diagonal_inversion:
                if self.verbose:
                    mpiprint('Using block-diagonal inversion', rank=self.rank)
                SigmaI_dbar, dbarSigmaIdbar, PhiI, logSigmaDet =\
                    self.calc_SigmaI_dbar_wrapper(
                        x, T_Ninv_T, dbar, block_T_Ninv_T=block_T_Ninv_T)
            else:
                if self.verbose:
                    mpiprint('Not using block-diagonal inversion',
                             rank=self.rank)
                start = time.time()
                SigmaI_dbar, dbarSigmaIdbar, PhiI, logSigmaDet =\
                    self.calc_SigmaI_dbar_wrapper(x, T_Ninv_T, dbar)
                if self.verbose:
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
            if self.verbose:
                mpiprint('logPhiDet time: {}'.format(time.time() - start),
                         rank=self.rank)

            start = time.time()
            MargLogL = -0.5*logSigmaDet - 0.5*logPhiDet + 0.5*dbarSigmaIdbar
            if self.uprior_inds is not None:
                MargLogL += np.sum(np.log(x[self.uprior_inds]))
            if self.intrinsic_noise_fitting:
                MargLogL = MargLogL - 0.5*d_Ninv_d - 0.5*log_det_N
            MargLogL = MargLogL.real
            if self.verbose:
                mpiprint('MargLogL time: {}'.format(time.time() - start),
                         rank=self.rank)
            if self.print_debug:
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
