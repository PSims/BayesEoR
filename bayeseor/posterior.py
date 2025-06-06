import time
import numpy as np
import scipy
from pdb import set_trace as brk
import h5py
from pathlib import Path

from ..gpu import GPUInterface
from ..utils import Cosmology, mpiprint


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
        Noise weighted representation of the data vector (signal + noise) of
        visibilities in model (u, v, eta) space.
    k_vals : np.ndarray of floats
        Array containing the mean k for each k bin.
    k_cube_voxels_in_bin : list
        List of sublists containing the flattened 3D k-space cube index of all
        k that fall within a given k bin.
    nuv : int
        Number of model uv-plane points per frequency channel.
    neta : int
        Number of Line of Sight (LoS, frequency axis) Fourier modes.
    nf : int
        Number of frequency channels.
    nq : int
        Number of quadratic modes in the Larse Spectral Scale Model (LSSM).
    Ninv : np.ndarray
        Covariance matrix of the data (signal + noise) vector of visibilities.
    d_Ninv_d : np.ndarray
        Single complex number computed as `d.conjugate() * Ninv * d`.
    ps_box_size_ra_Mpc : float
        Right ascension (RA) axis extent of the cosmological volume in Mpc from
        which the power spectrum is estimated.
    ps_box_size_dec_Mpc : float
        Declination (DEC) axis extent of the cosmological volume in Mpc from
        which the power spectrum is estimated.
    ps_box_size_para_Mpc : float
        LoS extent of the cosmological volume in Mpc from which the power
        spectrum is estimated.
    include_instrumental_effects : bool
        If True, include instrumental effects like frequency dependent (u, v)
        sampling and the primary beam.  Defaults to `True`.
    log_priors : boolean
        If `True`, power spectrum k bin amplitudes are assumed to be in log
        units, otherwise they will be treated using linear units.
    uprior_inds : array
        Boolean array with shape `len(k_vals)`.  If True (False), k bin uses a
        uniform (log-uniform) prior.
    masked_power_spectral_modes : np.ndarray
        Boolean array used to mask additional (u, v, eta) amplitudes from
        being included in the posterior calculations.  Defaults to using all
        EoR model modes.
    use_LWM_Gaussian_prior : bool
        If True, use a Gaussian prior on the LSSM (NOT IMPLEMENTED).
        Otherwise, use a uniform prior (default).
    inverse_LW_power : float
        Prior over the long wavelength modes in the LSSM.  Defaults to 0.0.
    dimensionless_PS : boolean
        If `True`, use a dimensionless power spectrum normalization
        `Delta**2 ~ mK**2`, otherwise use a dimensionful power spectrum
        normalization `P(k) ~ mK**2 Mpc**3`.
    block_T_Ninv_T : list
        Block diagonal representation of `T_Ninv_T`.  Only used if ignoring
        instrumental effects.  Defaults to `[]`.
    intrinsic_noise_fitting : boolean
        If `True`, fit for the amplitude of the noise in the data instead of
        using the covariance estimate as is in `Ninv`.  Defaults to `False`.
    fit_for_spectral_model_parameters : boolean
        If `True`, fit for the LSSM parameter values.  Defaults to `False`.
    pl_max : float
        Maximum brightness temperature spectral index when fitting for the
        optimal LSSM spectral indices.  Defaults to `None`.
    pl_grid_spacing : float
        Grid spacing for the power law spectral index axis when fitting for the
        LSSM parameter values.  Defaults to `None`.
    use_shg : bool
        If `True`, use the SubHarmonic Grid (SHG) in the model uv-plane.
    return_Sigma : boolean
        If `True`, break and return the matrix `Sigma = T_Ninv_T + PhiI`.
        Defaults to `False`.
    rank : int
        MPI rank.
    use_gpu : bool
        If True, try and use GPUs for Cholesky decomposition.  Otherwise, use
        CPUs (inadvisable due to inaccuracy of CPU matrix inversion).  
        Defaults to True.
    print : boolean
        If `True`, print execution time messages.  Defaults to `False`.
    print_rate : int
        Number of iterations between print statements.  Defaults to 100.
    debug : boolean
        If `True`, execute break statements for debugging.  Defaults to
        `False`.
    print_debug : boolean
        If `True`, print debug related messages.  Defaults to `False`.

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
        masked_power_spectral_modes=None,
        use_LWM_Gaussian_prior=False,
        inverse_LW_power=0.0,
        dimensionless_PS=False,
        block_T_Ninv_T=[],
        intrinsic_noise_fitting=False,
        fit_for_spectral_model_parameters=False,
        pl_max=None,
        pl_grid_spacing=None,
        use_shg=False,
        return_Sigma=False,
        rank=0,
        use_gpu=True,
        print=False,
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
        self.masked_power_spectral_modes = masked_power_spectral_modes
        self.use_LWM_Gaussian_prior = use_LWM_Gaussian_prior
        self.inverse_LW_power = inverse_LW_power
        self.dimensionless_PS = dimensionless_PS
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
        self.print = print
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

        if self.log_priors:
            mpiprint('Using log-priors', rank=self.rank)
        if self.dimensionless_PS:
            mpiprint('Calculating dimensionless_PS', rank=self.rank)
        mpiprint(
            f"Setting inverse_LW_power to {self.inverse_LW_power}",
            rank=self.rank
        )
        
        if use_gpu:
            # Initialize the GPU interface
            self.gpu = GPUInterface(rank=self.rank)
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
        T_Ninv_T_block : np.ndarray
            Single block matrix from block_T_Ninv_T.
        PhiI_block : np.ndarray
            Vector of the estimated inverse variance of each k voxel present
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
        Dimensionless power spectrum normalization in physical units.

        This normalization will return `PowerI`, an estimate for one over the
        variance of `a`, in units of 1 / (mK^2 sr^2 Hz^2).

        Parameters
        ----------
        i_bin : int
            Spherically averaged k bin index.

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
            Input power spectrum amplitudes per k bin with length `nDims`.

        Returns
        -------
        PhiI : np.ndarray
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
        T_Ninv_T : np.ndarray
            Complex matrix product ``T.conjugate().T * Ninv * T``.
        PhiI : np.ndarray
            Vector with estimates of the inverse variance of each model
            (u, v, eta) voxel.

        Returns
        -------
        Sigma_block_diagonals : np.ndarray
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
            Input power spectrum amplitudes per k bin with length `nDims`.
        T_Ninv_T : np.ndarray
            Complex matrix product ``T.conjugate().T * Ninv * T``.
        dbar : np.ndarray
            Noise weighted representation of the data (signal + noise) vector
            of visibilities in model (u, v, eta) space.
        block_T_Ninv_T : list
            Block diagonal representation of `T_Ninv_T`.  Only used if ignoring
            instrumental effects.  Defaults to an empty list.

        Returns
        -------
        SigmaI_dbar : np.ndarray
            Complex array of maximum a posteriori (u, v, eta) and Large
            Spectral Scale Model (LSSM) amplitudes.  Used to compute the model
            data vector via ``m = T * SigmaI_dbar``.
        dbarSigmaIdbar : np.ndarray
            Complex array product ``dbar * SigmaI_dbar``.
        PhiI : np.ndarray
            Vector with estimates of the inverse variance of each model
            (u, v, eta) voxel.
        logSigmaDet : np.ndarray
            Natural logarithm of the determinant of
            ``Sigma = T_Ninv_T + PhiI``.

        """
        start = time.time()
        PowerI = self.calc_PowerI(x)
        PhiI = PowerI
        if self.print:
            mpiprint('\tPhiI time: {}'.format(time.time()-start),
                     rank=self.rank)

        do_block_diagonal_inversion = len(np.shape(block_T_Ninv_T)) > 1
        if do_block_diagonal_inversion:
            if self.print:
                mpiprint('Using block-diagonal inversion', rank=self.rank)
            start = time.time()
            if self.intrinsic_noise_fitting:
                # This is only valid if the data is uniformly weighted
                Sigma_block_diagonals = self.calc_Sigma_block_diagonals(
                    T_Ninv_T/self.alpha_prime**2., PhiI)
            else:
                Sigma_block_diagonals = self.calc_Sigma_block_diagonals(
                    T_Ninv_T, PhiI)
            if self.print:
                mpiprint('Time taken: {}'.format(time.time()-start),
                         rank=self.rank)

            start = time.time()
            dbar_blocks = np.split(dbar, self.nuv)
            if self.use_gpu:
                if self.print:
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
            if self.print:
                mpiprint('Time taken: {}'.format(time.time()-start),
                         rank=self.rank)

            SigmaI_dbar = SigmaI_dbar_blocks.flatten()
            dbarSigmaIdbar = np.dot(dbar.conjugate().T, SigmaI_dbar)
            if self.print:
                mpiprint('Time taken: {}'.format(time.time()-start),
                         rank=self.rank)

            if self.use_gpu:
                logSigmaDet = np.sum(logdet_Sigma_blocks)
            else:
                logSigmaDet = np.sum(
                    [np.linalg.slogdet(Sigma_block)[1]
                     for Sigma_block in Sigma_block_diagonals]
                    )
                if self.print:
                    mpiprint('Time taken: {}'.format(time.time()-start),
                             rank=self.rank)

        else:
            if self.count % self.print_rate == 0 and self.print:
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
            if self.print:
                mpiprint('\tSigma build time: {}'.format(time.time()-start),
                         rank=self.rank)
            if self.return_Sigma:
                return Sigma

            start = time.time()
            if self.use_gpu:
                if self.print:
                    mpiprint('Computing matrix inversion on GPU',
                             rank=self.rank)
                SigmaI_dbar_and_logdet_Sigma = self.calc_SigmaI_dbar(
                    Sigma, dbar, x_for_error_checking=x)
                SigmaI_dbar = SigmaI_dbar_and_logdet_Sigma[0]
                logdet_Sigma = SigmaI_dbar_and_logdet_Sigma[1]
            else:
                SigmaI_dbar = self.calc_SigmaI_dbar(
                    Sigma, dbar, x_for_error_checking=x)
            if self.print:
                mpiprint(
                    '\tcalc_SigmaI_dbar time: {}'.format(time.time()-start),
                    rank=self.rank
                )

            start = time.time()
            dbarSigmaIdbar = np.dot(dbar.conjugate().T, SigmaI_dbar)
            if self.print:
                mpiprint(
                    '\tdbarSigmaIdbar time: {}'.format(time.time()-start),
                    rank=self.rank
                )

            start = time.time()
            if self.use_gpu:
                logSigmaDet = logdet_Sigma
            else:
                logSigmaDet = np.linalg.slogdet(Sigma)[1]
            if self.print:
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
        Sigma : np.ndarray
            Complex matrix ``Sigma = T_Ninv_T + PhiI``.
        dbar : np.ndarray
            Noise weighted representation of the data (signal + noise) vector
            of visibilities in model (u, v, eta) space.
        x_for_error_checking : array_like
            Input power spectrum amplitudes per k bin with length `nDims`
            used for error checking of the matrix inversion.  Defaults to an
            empty list (no error checking).

        Returns
        -------
        SigmaI_dbar : np.ndarray
            Complex array of maximum a posteriori (u, v, eta) and Large
            Spectral Scale Model (LSSM) amplitudes.  Used to compute the model
            data vector via ``m = T * SigmaI_dbar``.
        logdet_Magma_Sigma : np.ndarray
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
            if self.print:
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
            if self.print:
                mpiprint(
                    '\t\tCholesky decomposition time: {time.time() - start}',
                    rank=self.rank
                )
            # Note: After wrapmzpotrf, Sigma is actually
            # SigmaCho (i.e. L with Sigma = LL^T)
            logdet_Magma_Sigma = np.sum(np.log(np.diag(abs(Sigma)))) * 2
            if self.print:
                start = time.time()
            SigmaI_dbar = scipy.linalg.cho_solve((Sigma, True), dbar)
            if self.print:
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
            Input power spectrum amplitudes per k bin with length `nDims`.
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
            if self.print:
                mpiprint('pl_params', pl_params, rank=self.rank)
            if self.print:
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
            if self.print:
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
                if self.print:
                    mpiprint('Using block-diagonal inversion', rank=self.rank)
                SigmaI_dbar, dbarSigmaIdbar, PhiI, logSigmaDet =\
                    self.calc_SigmaI_dbar_wrapper(
                        x, T_Ninv_T, dbar, block_T_Ninv_T=block_T_Ninv_T)
            else:
                if self.print:
                    mpiprint('Not using block-diagonal inversion',
                             rank=self.rank)
                start = time.time()
                SigmaI_dbar, dbarSigmaIdbar, PhiI, logSigmaDet =\
                    self.calc_SigmaI_dbar_wrapper(x, T_Ninv_T, dbar)
                if self.print:
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
            if self.print:
                mpiprint('logPhiDet time: {}'.format(time.time() - start),
                         rank=self.rank)

            start = time.time()
            MargLogL = -0.5*logSigmaDet - 0.5*logPhiDet + 0.5*dbarSigmaIdbar
            if self.uprior_inds is not None:
                MargLogL += np.sum(np.log(x[self.uprior_inds]))
            if self.intrinsic_noise_fitting:
                MargLogL = MargLogL - 0.5*d_Ninv_d - 0.5*log_det_N
            MargLogL = MargLogL.real
            if self.print:
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
