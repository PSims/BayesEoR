import time
import numpy as np
import scipy
from pdb import set_trace as brk
import h5py

import BayesEoR.Params.params as p
from BayesEoR.Utils import Cosmology


"""
Potentially useful links:
http://www.mrao.cam.ac.uk/~kjbg1/lectures/lect1_1.pdf
"""


try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    import ctypes
    from numpy import ctypeslib

    # Get path to installation of BayesEoR
    base_dir = '/'.join(__file__.split('/')[:-2]) + '/'
    # Load MAGMA GPU Wrapper Functions
    GPU_wrap_dir = base_dir+'Likelihood/GPU_wrapper/'
    wrapmzpotrf = ctypes.CDLL(GPU_wrap_dir+'wrapmzpotrf.so')
    nrhs = 1
    wrapmzpotrf.cpu_interface.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypeslib.ndpointer(np.complex128, ndim=2, flags='C'),
        ctypeslib.ndpointer(np.complex128, ndim=1, flags='C'),
        ctypes.c_int,
        ctypeslib.ndpointer(np.int, ndim=1, flags='C')]
    print('Computing on GPU')

except Exception as e:
    print('Exception loading GPU encountered...')
    print(e)
    print('Computing on CPU instead...')
    p.useGPU = False


#--------------------------------------------
# Define posterior
#--------------------------------------------

class PowerSpectrumPosteriorProbability(object):
    def __init__(
            self, T_Ninv_T, dbar, Sigma_Diag_Indices, Npar,
            k_cube_voxels_in_bin, nuv, nu, nv, nx, ny, neta, nf, nq,
            masked_power_spectral_modes, modk_vis_ordered_list,
            Ninv, d_Ninv_d, fit_single_elems=False, **kwargs):

        # ===== Defaults =====
        default_diagonal_sigma = False
        default_block_T_Ninv_T = []
        default_log_priors = False
        default_dimensionless_PS = False
        default_inverse_LW_power = 0.0
        default_inverse_LW_power_zeroth_LW_term = 0.0
        default_inverse_LW_power_first_LW_term = 0.0
        default_inverse_LW_power_second_LW_term = 0.0
        default_Print = False
        default_debug = False
        default_Print_debug = False
        default_intrinsic_noise_fitting = False
        default_return_Sigma = False
        default_fit_for_spectral_model_parameters = False

        # ===== Inputs =====
        self.diagonal_sigma = kwargs.pop(
            'diagonal_sigma', default_diagonal_sigma)
        self.block_T_Ninv_T = kwargs.pop(
            'block_T_Ninv_T', default_block_T_Ninv_T)
        self.log_priors = kwargs.pop(
            'log_priors', default_log_priors)
        if self.log_priors:
            print('Using log-priors')
        self.dimensionless_PS = kwargs.pop(
            'dimensionless_PS', default_dimensionless_PS)
        if self.dimensionless_PS:
            print('Calculating dimensionless_PS')
        self.inverse_LW_power = kwargs.pop(
            'inverse_LW_power', default_inverse_LW_power)
        self.inverse_LW_power_zeroth_LW_term = kwargs.pop(
            'inverse_LW_power_zeroth_LW_term',
            default_inverse_LW_power_zeroth_LW_term)
        self.inverse_LW_power_first_LW_term = kwargs.pop(
            'inverse_LW_power_first_LW_term',
            default_inverse_LW_power_first_LW_term)
        self.inverse_LW_power_second_LW_term = kwargs.pop(
            'inverse_LW_power_second_LW_term',
            default_inverse_LW_power_second_LW_term)
        self.Print = kwargs.pop(
            'Print', default_Print)
        self.debug = kwargs.pop(
            'debug', default_debug)
        self.Print_debug = kwargs.pop(
            'Print_debug', default_Print_debug)
        self.intrinsic_noise_fitting = kwargs.pop(
            'intrinsic_noise_fitting', default_intrinsic_noise_fitting)
        self.return_Sigma = kwargs.pop(
            'return_Sigma', default_return_Sigma)
        self.fit_for_spectral_model_parameters = kwargs.pop(
            'fit_for_spectral_model_parameters',
            default_fit_for_spectral_model_parameters)
        # Must pass k_vals with new normalization (v4d0)
        self.k_vals = kwargs.pop('k_vals')

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
        self.spectral_model_parameters_array_storage_dir =\
            '/gpfs/data/jpober/psims/EoR/Python_Scripts/BayesEoR/'\
            'git_version/BayesEoR/spec_model_tests/array_storage/'\
            'FgSpecMOptimisation/Likelihood_v1d76_3D_ZM_nu_9_nv_9_'\
            'neta_38_nq_2_npl_2_b1_2.00E+00_b2_3.00E+00_sigma_8d5E+04'\
            '_instrumental/HERA_331_baselines_shorter_than_29d3_for_30'\
            '_0d5_min_time_steps_Gaussian_beam_peak_amplitude_1d0_beam_'\
            'width_9d0_deg_at_150d0_MHz/'


    def add_power_to_diagonals(self, T_Ninv_T_block, PhiI_block, **kwargs):
        return T_Ninv_T_block+np.diag(PhiI_block)

    def calc_Sigma_block_diagonals(self, T_Ninv_T, PhiI, **kwargs):
        PhiI_blocks = np.split(PhiI, self.nuv)
        Sigma_block_diagonals = np.array(
            [self.add_power_to_diagonals(
                T_Ninv_T[
                    (self.neta + self.nq)*i_block
                    : (self.neta + self.nq)*(i_block+1),
                    (self.neta + self.nq)*i_block
                    : (self.neta + self.nq)*(i_block+1)
                    ],
                PhiI_blocks[i_block]
                )
                for i_block in range(self.nuv)]
            )
        return Sigma_block_diagonals

    def calc_SigmaI_dbar_wrapper(self, x, T_Ninv_T, dbar, **kwargs):
        block_T_Ninv_T = []

        # ===== Inputs =====
        if 'block_T_Ninv_T' in kwargs:
            block_T_Ninv_T = kwargs['block_T_Ninv_T']

        start = time.time()
        PowerI = self.calc_PowerI(x)
        PhiI = PowerI
        if self.Print:
            print('Time taken: {}'.format(time.time()-start))

        do_block_diagonal_inversion = len(np.shape(block_T_Ninv_T)) > 1
        if do_block_diagonal_inversion:
            if self.Print:
                print('Using block-diagonal inversion')
            start = time.time()
            if self.intrinsic_noise_fitting:
                # This is only valid if the data is uniformly weighted
                Sigma_block_diagonals = self.calc_Sigma_block_diagonals(
                    T_Ninv_T/self.alpha_prime**2., PhiI)
            else:
                Sigma_block_diagonals = self.calc_Sigma_block_diagonals(
                    T_Ninv_T, PhiI)
            if self.Print:
                print('Time taken: {}'.format(time.time()-start))
            if self.Print:
                print('nuv', self.nuv)

            start = time.time()
            dbar_blocks = np.split(dbar, self.nuv)
            if p.useGPU:
                if self.Print:
                    print('Computing block diagonal inversion on GPU')
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
                # This doesn't work because the array
                # size is lost / flatten fails
                # SigmaI_dbar_blocks = SigmaI_dbar_blocks_and_logdet_Sigma[:, 0]
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
                print('Time taken: {}'.format(time.time()-start))

            SigmaI_dbar = SigmaI_dbar_blocks.flatten()
            dbarSigmaIdbar = np.dot(dbar.conjugate().T, SigmaI_dbar)
            if self.Print:
                print('Time taken: {}'.format(time.time()-start))

            if p.useGPU:
                logSigmaDet = np.sum(logdet_Sigma_blocks)
            else:
                logSigmaDet = np.sum(
                    [np.linalg.slogdet(Sigma_block)[1]
                     for Sigma_block in Sigma_block_diagonals]
                    )
                if self.Print:
                    print('Time taken: {}'.format(time.time()-start))

        else:
            if self.count % self.print_rate == 0:
                print('Not using block-diagonal inversion')
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
                print('Time taken: {}'.format(time.time()-start))
            if self.return_Sigma:
                return Sigma

            start = time.time()
            if p.useGPU:
                if self.Print:
                    print('Computing matrix inversion on GPU')
                SigmaI_dbar_and_logdet_Sigma = self.calc_SigmaI_dbar(
                    Sigma, dbar, x_for_error_checking=x)
                SigmaI_dbar = SigmaI_dbar_and_logdet_Sigma[0]
                logdet_Sigma = SigmaI_dbar_and_logdet_Sigma[1]
            else:
                SigmaI_dbar = self.calc_SigmaI_dbar(
                    Sigma, dbar, x_for_error_checking=x)
            if self.Print:
                print('Time taken: {}'.format(time.time()-start))

            dbarSigmaIdbar = np.dot(dbar.conjugate().T, SigmaI_dbar)
            if self.Print:
                print('Time taken: {}'.format(time.time()-start))

            if p.useGPU:
                logSigmaDet = logdet_Sigma
            else:
                logSigmaDet = np.linalg.slogdet(Sigma)[1]
            if self.Print:
                print('Time taken: {}'.format(time.time()-start))
            # logSigmaDet = 2.*np.sum(np.log(np.diag(Sigmacho)))

        return SigmaI_dbar, dbarSigmaIdbar, PhiI, logSigmaDet

    def calc_SigmaI_dbar(self, Sigma, dbar, **kwargs):
        # ===== Defaults =====
        block_T_Ninv_T = []
        default_x_for_error_checking = \
            "Params haven't been recorded... use x_for_error_checking "\
            "kwarg when calling calc_SigmaI_dbar to change this."

        # ===== Inputs =====
        if 'block_T_Ninv_T' in kwargs:
            block_T_Ninv_T = kwargs['block_T_Ninv_T']
        if 'x_for_error_checking' in kwargs:
            x_for_error_checking = kwargs['x_for_error_checking']

        if not p.useGPU:
            # Sigmacho = scipy.linalg.cholesky(
            #         Sigma, lower=True
            #     ).astype(np.complex256)
            # SigmaI_dbar = scipy.linalg.cho_solve((Sigmacho, True), dbar)
            # scipy.linalg.inv is not numerically stable
            # use with caution
            SigmaI = scipy.linalg.inv(Sigma)
            SigmaI_dbar = np.dot(SigmaI, dbar)
            return SigmaI_dbar

        else:
            # brk()
            dbar_copy = dbar.copy()
            dbar_copy_copy = dbar.copy()
            self.GPU_error_flag = np.array([0])
            # Replace 0 with 1 to pring debug in the following command
            wrapmzpotrf.cpu_interface(
                len(Sigma), nrhs, Sigma, dbar_copy, 0, self.GPU_error_flag)
            # Note: After wrapmzpotrf, Sigma is actually
            # SigmaCho (i.e. L with Sigma = LL^T)
            logdet_Magma_Sigma = np.sum(np.log(np.diag(abs(Sigma)))) * 2
            # print(logdet_Magma_Sigma)
            SigmaI_dbar = scipy.linalg.cho_solve(
                (Sigma.conjugate().T, True), dbar_copy_copy)
            if self.GPU_error_flag[0] != 0:
                # If the inversion doesn't work, zero-weight the
                # sample (may want to stop computing if this occurs?)
                logdet_Magma_Sigma = +np.inf
                print('GPU inversion error. Setting sample posterior probability to zero.')
                print('Param values: ', x_for_error_checking)
                print('GPU_error_flag = {}'.format(self.GPU_error_flag))

            return SigmaI_dbar, logdet_Magma_Sigma

    def calc_physical_dimensionless_power_spectral_normalisation(
            self, i_bin, **kwargs):
        """
        This normalization will calculate PowerI, an estimate for one over
        the variance of a in units of 1 / (mK**2 sr**2 Hz**2)
        """
        # Temporary fix for varying the FoV while keeping the bandwidth fixed
        VOLUME = p.box_size_21cmFAST_Mpc_sc**2 * 2048.0
        # Frequency axis is truncated by p.nf
        VOLUME *= 1.0 * p.nf / p.box_size_21cmFAST_pix_sc

        full_power_spectrum_normalisation = 1.0 / (2.0*np.pi**2.*VOLUME)

        # If using |vector k| in the normalization
        # dimensionless_PS_scaling = (
        #         self.modk_vis_ordered_list[i_bin]**3.
        #         * full_power_spectrum_normalisation)
        # If using mean k-bin centers in the normalization
        dimensionless_PS_scaling = (
                self.k_vals[i_bin]**3.
                * full_power_spectrum_normalisation)

        if not p.include_instrumental_effects:
            # Account for / undo the extra (vfft1[0].size**0.5) scaling
            # factor that is currently in the non-instrumental data
            # creation functions
            dimensionless_PS_scaling = (
                    dimensionless_PS_scaling
                    * float(p.box_size_21cmFAST_pix_sc)**2.)

        # Redshift dependent quantities
        nu_array_MHz = p.nu_min_MHz + np.arange(p.nf)*p.channel_width_MHz
        cosmo = Cosmology()
        redshift =\
            cosmo.Convert_from_21cmFrequency_to_Redshift(nu_array_MHz.mean())
        X2Y = cosmo.X2Y(redshift)
        dimensionless_PS_scaling *= X2Y**2

        return dimensionless_PS_scaling

    def calc_PowerI(self, x, **kwargs):
        """
            Place restrictions on the power in the long spectral scale
            model either for,
            inverse_LW_power:
                constrain the amplitude distribution of all of
                the large spectral scale model components
            inverse_LW_power_zeroth_LW_term:
                constrain the amplitude of monopole-term basis vector
            inverse_LW_power_first_LW_term:
                constrain the amplitude of the model components
                of the 1st LW basis vector (e.g. linear model comp.)
            inverse_LW_power_second_LW_term:
                constrain the amplitude of the model components
                of the 2nd LW basis vector (e.g. quad model comp.)

            Note: The indices used are correct for the current
            ordering of basis vectors when nf is an even number...
        """
        PowerI = np.zeros(self.Npar)

        if p.include_instrumental_effects:
            # Updated for python 3: floor division
            q0_index = self.neta//2
        else:
            # Updated for python 3: floor division
            q0_index = self.nf//2 - 1
        q1_index = self.neta
        q2_index = self.neta + 1

        # Constrain LW mode amplitude distribution
        dimensionless_PS_scaling =\
            self.calc_physical_dimensionless_power_spectral_normalisation(0)
        if p.use_LWM_Gaussian_prior:
            Fourier_mode_start_index = 3
            # Set to zero for a uniform distribution
            # Updated for python 3: floor division
            # PowerI[self.nf//2 - 1 :: self.nf] =\
            #     np.mean(dimensionless_PS_scaling) / x[0]
            # # Updated for python 3: floor division
            # PowerI[self.nf - 2 :: self.nf] =\
            #     np.mean(dimensionless_PS_scaling) / x[1]
            # # Updated for python 3: floor division
            # PowerI[self.nf - 1 :: self.nf] =\
            #     np.mean(dimensionless_PS_scaling) / x[2]
            PowerI[q0_index :: self.neta+self.nq] =\
                np.mean(dimensionless_PS_scaling) / x[0]
            PowerI[q1_index :: self.neta+self.nq] =\
                np.mean(dimensionless_PS_scaling) / x[1]
            PowerI[q2_index :: self.neta+self.nq] =\
                np.mean(dimensionless_PS_scaling) / x[2]
        else:
            Fourier_mode_start_index = 0
            # Set to zero for a uniform distribution
            PowerI[q0_index :: self.neta+self.nq] = self.inverse_LW_power
            PowerI[q1_index :: self.neta+self.nq] = self.inverse_LW_power
            PowerI[q2_index :: self.neta+self.nq] = self.inverse_LW_power

            if self.inverse_LW_power == 0.0:
                # Set to zero for a uniform distribution
                PowerI[q0_index :: self.neta+self.nq] =\
                    self.inverse_LW_power_zeroth_LW_term
                PowerI[q1_index :: self.neta+self.nq] =\
                    self.inverse_LW_power_first_LW_term
                PowerI[q2_index :: self.neta+self.nq] =\
                    self.inverse_LW_power_second_LW_term

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
            PowerI[self.k_cube_voxels_in_bin[i_bin]] =(
                    power_spectrum_normalisation
                    / x[Fourier_mode_start_index+i_bin])

        return PowerI

    def posterior_probability(self, x, **kwargs):
        if self.debug:
            brk()
        phi = [0.0]

        # ===== Defaults =====
        block_T_Ninv_T = self.block_T_Ninv_T
        fit_single_elems = self.fit_single_elems
        T_Ninv_T = self.T_Ninv_T
        dbar = self.dbar

        # ===== Inputs =====
        if 'block_T_Ninv_T' in kwargs:
            block_T_Ninv_T = kwargs['block_T_Ninv_T']

        if self.fit_for_spectral_model_parameters:
            Print = self.Print
            self.Print = True
            pl_params = x[:2]
            x = x[2:]
            if self.Print:
                print('pl_params', pl_params)
            if self.Print:
                print('p.pl_grid_spacing, p.pl_max',
                      p.pl_grid_spacing, p.pl_max)
            b1 = pl_params[0]
            b2 = (b1
                  + p.pl_grid_spacing
                  + (p.pl_max - b1 - p.pl_grid_spacing)*pl_params[1]
                  )
            # Round derived pl indices to nearest p.pl_grid_spacing
            b1, b2 = (p.pl_grid_spacing
                      * np.round(np.array([b1, b2]) / p.pl_grid_spacing, 0))
            if self.Print:
                print('b1, b2', b1, b2)

            # Load matrices associated with sampled beta params
            T_Ninv_T_dataset_name =\
                'T_Ninv_T_b1_{}_b2_{}'.format(b1, b2).replace('.', 'd')
            T_Ninv_T_file_path = (
                    self.spectral_model_parameters_array_storage_dir
                    + T_Ninv_T_dataset_name
                    + '.h5')
            if self.count % self.print_rate == 0:
                print('Replacing T_Ninv_T with:', T_Ninv_T_file_path)
            start = time.time()
            with h5py.File(T_Ninv_T_file_path, 'r') as hf:
                T_Ninv_T = hf[T_Ninv_T_dataset_name][:]
                # alpha_prime = p.sigma/170000.0
                # This is only valid if the data is uniformly weighted
                # T_Ninv_T = T_Ninv_T/(alpha_prime**2.0)
                if self.count % self.print_rate == 0:
                    print('Time taken: {}'.format(time.time() - start))

            dbar_dataset_name =\
                'dbar_b1_{}_b2_{}'.format(b1, b2).replace('.', 'd')
            dbar_file_path = (
                    self.spectral_model_parameters_array_storage_dir
                    + dbar_dataset_name
                    + '.h5')
            if self.count % self.print_rate == 0:
                print('Replacing dbar with:', dbar_file_path)
            start = time.time()
            with h5py.File(dbar_file_path, 'r') as hf:
                dbar = hf[dbar_dataset_name][:]
                # alpha_prime = p.sigma/170000.0
                # This is only valid if the data is uniformly weighted
                # dbar = dbar/(alpha_prime**2.0)
                # if self.count % self.print_rate == 0:
                #     print('alpha_prime = ', alpha_prime)
                if self.count % self.print_rate == 0:
                    print('Time taken: {}'.format(time.time() - start))
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
            # print('Using log-priors')
            x = 10.**np.array(x)

        # brk()
        do_block_diagonal_inversion = len(np.shape(block_T_Ninv_T)) > 1
        self.count += 1
        start = time.time()
        try:
            if do_block_diagonal_inversion:
                if self.Print:
                    print('Using block-diagonal inversion')
                SigmaI_dbar, dbarSigmaIdbar, PhiI, logSigmaDet =\
                    self.calc_SigmaI_dbar_wrapper(
                        x, T_Ninv_T, dbar, block_T_Ninv_T=block_T_Ninv_T)
            else:
                if self.Print:
                    print('Not using block-diagonal inversion')
                SigmaI_dbar, dbarSigmaIdbar, PhiI, logSigmaDet =\
                    self.calc_SigmaI_dbar_wrapper(x, T_Ninv_T, dbar)

            # Only possible because Phi is diagonal (otherwise would
            # need to calc np.linalg.slogdet(Phi)). -1 factor is to get
            # logPhiDet from logPhiIDet. Note: the real part of this
            # calculation matches the solution given by
            # np.linalg.slogdet(Phi))
            # logPhiDet = -1 * np.sum(np.log(
            #     PhiI[np.logical_not(self.masked_power_spectral_modes)]
            #     )).real
            logPhiDet = -1 * np.sum(np.log(PhiI)).real

            MargLogL = -0.5*logSigmaDet - 0.5*logPhiDet + 0.5*dbarSigmaIdbar
            if self.intrinsic_noise_fitting:
                MargLogL = MargLogL - 0.5*d_Ninv_d - 0.5*log_det_N
            MargLogL = MargLogL.real
            if self.Print_debug:
                MargLogL_equation_string = \
                    'MargLogL = -0.5*logSigmaDet '\
                    '-0.5*logPhiDet + 0.5*dbarSigmaIdbar'
                if self.intrinsic_noise_fitting:
                    print('Using intrinsic noise fitting')
                    MargLogL_equation_string +=\
                        ' - 0.5*d_Ninv_d -0.5*log_det_N'
                    print('logSigmaDet, logPhiDet, dbarSigmaIdbar, '
                          'd_Ninv_d, log_det_N',
                          logSigmaDet, logPhiDet, dbarSigmaIdbar,
                          d_Ninv_d, log_det_N)
                else:
                    print('logSigmaDet, logPhiDet, dbarSigmaIdbar',
                          logSigmaDet, logPhiDet, dbarSigmaIdbar)
                print(MargLogL_equation_string, MargLogL)
                print('MargLogL.real', MargLogL.real)

            # brk()

            if self.nu > 10:
                self.print_rate = 100
            if self.count % self.print_rate == 0:
                print('count', self.count)
                print('Time since class instantiation: {}'.format(
                    time.time() - self.instantiation_time))
                print('Time for this likelihood call: {}'.format(
                    time.time() - start))
            return MargLogL.squeeze()*1.0, phi
        except Exception as e:
            print('Exception encountered...')
            print(e)
            return -np.inf, -1
