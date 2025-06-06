""" Class and functions for performing maximum a posteriori calculations. """
import numpy as np
from pathlib import Path
import json
from jsonargparse import Namespace
import matplotlib.pyplot as plt

from ..matrices import BuildMatrixTree
from ..model import (
    load_inst_model,
    generate_k_cube_in_physical_coordinates,
    mask_k_cube,
    generate_k_cube_model_spherical_binning,
    calc_mean_binned_k_vals,
    generate_data_and_noise_vector_instrumental
)
from ..params import calculate_derived_params, parse_uprior_inds
from ..posterior import PowerSpectrumPosteriorProbability
from .utils import mpiprint, vector_is_hermitian

class MaximumAPosteriori(object):
    """
    Class for performing maximum a posteriori calculations with BayesEoR.

    Parameters
    ----------
    data_path : str or Path
        Path to a numpy readable visibility data file in units of mK sr.
    array_dir : str or Path
        Path to a directory containing T, Ninv, and T_Ninv_T.
    output_dir : str or Path
        Path to a BayesEoR output directory which contains 'args.json' and
        'k-vals.txt' files.
    verbose : bool
        If True (default), print timing and status messages.
    
    """
    def __init__(
        self,
        data_path,
        array_dir,
        output_dir,
        verbose=True
    ):        
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        with open(output_dir / "args.json", "r") as f:
            cl_args = Namespace(**json.load(f))
        args = calculate_derived_params(cl_args)

        # Calculate k bin values
        mpiprint(
            "\nSpherical k bins:", style="bold", rank=1-int(verbose)
        )
        mod_k = generate_k_cube_in_physical_coordinates(
            args.nu,
            args.nv,
            args.neta,
            args.ps_box_size_ra_Mpc,
            args.ps_box_size_dec_Mpc,
            args.ps_box_size_para_Mpc
        )[0]
        mod_k_vo = mask_k_cube(mod_k)
        k_cube_voxels_in_bin = generate_k_cube_model_spherical_binning(
            mod_k_vo,
            args.ps_box_size_para_Mpc
        )[0]
        self.k_vals = calc_mean_binned_k_vals(
            mod_k_vo,
            k_cube_voxels_in_bin,
            rank=1-int(verbose)
        )

        # Load matrices
        self.bm = BuildMatrixTree(
            array_dir.as_posix(),
            args.include_instrumental_effects,
            args.use_sparse_matrices
        )
        Ninv = self.bm.read_data((array_dir / "Ninv.npz").as_posix(), "Ninv")
        self.Ndata = Ninv.shape[0]
        T = self.bm.read_data((array_dir / "T.h5").as_posix(), "T")
        T_Ninv_T = self.bm.read_data(
            (array_dir / "T_Ninv_T.h5").as_posix(), "T_Ninv_T"
        )

        data = np.load(data_path, allow_pickle=True)
        if data.dtype.kind == "O":
            # Data and noise vector saved in a single dictionary
            dict_format = True
            data = data.item()
            if "noise" in data and "sigma" not in args:
                gen_noise = False
            else:
                gen_noise = True
        else:
            # Data and noise vector saved as separate arrays
            dict_format = False
            if "noise_data_path" in args:
                gen_noise = False
            else:
                gen_noise = True
        if gen_noise:
            effective_noise = None
        else:
            if dict_format:
                effective_noise = data["noise"]
            else:
                effective_noise = np.load(args.noise_data_path)
            args.sigma = effective_noise.std()
        if args.include_instrumental_effects:
            uvw_array_m, bl_red_array, phasor_vec = load_inst_model(
                args.inst_model
            )
            if phasor_vec is not None and args.drift_scan:
                phasor_vec = None
            bl_red_array = bl_red_array*0 + bl_red_array.min()
            avg_bl_red = np.mean(bl_red_array)
            if gen_noise:
                args.sigma *= avg_bl_red**0.5
            if dict_format:
                d_eor = data["data"]
            else:
                d_eor = data
            if gen_noise:
                d, effective_noise, bl_conj_map = \
                    generate_data_and_noise_vector_instrumental(
                        args.sigma,
                        d_eor,
                        args.nf,
                        args.nt,
                        uvw_array_m[0],
                        bl_red_array[0],
                        random_seed=args.noise_seed,
                        rank=1-int(verbose)
                    )
            else:
                d = d_eor.copy()
                _, _, bl_conj_map =\
                    generate_data_and_noise_vector_instrumental(
                        0,
                        d_eor,
                        args.nf,
                        args.nt,
                        uvw_array_m[0],
                        bl_red_array[0],
                        rank=1-int(verbose)
                    )
        mpiprint(
            "\nHermitian symmetry checks:", style="bold", rank=1-int(verbose)
        )
        mpiprint(
            "signal is Hermitian:",
            vector_is_hermitian(
                d_eor, bl_conj_map, args.nt, args.nf, uvw_array_m.shape[1]
            ),
            rank=1-int(verbose)
        )
        mpiprint(
            "signal + noise is Hermitian:",
            vector_is_hermitian(
                d, bl_conj_map, args.nt, args.nf, uvw_array_m.shape[1]
            ),
            rank=1-int(verbose)
        )

        mpiprint("\nSNR:", style="bold", rank=1-int(verbose))
        mpiprint(f"Stddev(signal) = {d_eor.std():.4e}", rank=1-int(verbose))
        effective_noise_std = effective_noise.std()
        mpiprint(
            f"Stddev(noise) = {effective_noise_std:.4e}",
            rank=1-int(verbose)
        )
        mpiprint(
            f"SNR = {(d_eor.std() / effective_noise_std):.4e}\n",
            rank=1-int(verbose)
        )

        Ninv_d = Ninv * d  # Ninv is sparse, so we use * instead of np.dot
        self.d_Ninv_d = np.dot(d.conjugate(), Ninv_d)
        dbar = np.dot(T.conjugate().T, Ninv_d)
        nDims = len(k_cube_voxels_in_bin)
        if args.use_intrinsic_noise_fitting:
            nDims += 1
        if args.use_LWM_Gaussian_prior:
            nDims += 3
        if args.include_instrumental_effects:
            block_T_Ninv_T = []
        
        if args.uprior_bins != "":
            args.uprior_inds = parse_uprior_inds(args.uprior_bins, nDims)
            mpiprint(
                "\nUniform prior k-bin indices:",
                f"{np.where(args.uprior_inds)[0]}\n",
                rank=1-int(verbose)
            )
        else:
            args.uprior_inds = None

        self.pspp = PowerSpectrumPosteriorProbability(
            T_Ninv_T,
            dbar,
            self.k_vals,
            k_cube_voxels_in_bin,
            args.nuv,
            args.neta,
            args.nf,
            args.nq,
            Ninv,
            self.d_Ninv_d,
            args.redshift,
            args.ps_box_size_ra_Mpc,
            args.ps_box_size_dec_Mpc,
            args.ps_box_size_para_Mpc,
            include_instrumental_effects=args.include_instrumental_effects,
            log_priors=args.log_priors,
            uprior_inds=args.uprior_inds,
            inverse_LW_power=args.inverse_LW_power,
            dimensionless_PS=True,  # Currently only support \Delta^2(k)
            block_T_Ninv_T=block_T_Ninv_T,
            intrinsic_noise_fitting=args.use_intrinsic_noise_fitting,
            use_shg=args.use_shg,
            rank=0,
            use_gpu=True,
            print=args.verbose
        )

        self.args = args
        self.d = d
        self.d_eor = d_eor
        if gen_noise:
            self.noise = effective_noise
        self.T = T
        self.uvw_array_m = uvw_array_m
        self.bl_red_array = bl_red_array
    
    def map_estimate(
        self,
        ps=None,
        dmps=None,
        return_prior_cov=False
    ):
        """
        Calculate the maximum a posteriori (MAP) model coefficients.

        Must pass one of `ps` or `dmps`.  Both cannot be none.  These variables
        are used to calculate the prior covariance matrix.

        Parameters
        ----------
        ps : float or array-like
            Expected power spectrum, P(k).  Can be a single float (for a flat
            P(k)) or an array-like with shape `(self.k_vals.size,)`.
        dmps : float or array-like
            Expected dimensionless power spectrum, \Delta^2(k).  Can be a
            single float or an array-like with shape `(self.k_vals.size,)`.
        return_prior_cov : bool, optional
            If True, return the prior covariance matrix in addition to the
            MAP estimate.  Defaults to False.
        
        Returns
        -------
        map_coeffs : ndarray
            MAP coefficients of the model.
        map_vis : ndarray
            MAP model visibilities.
        log_post : float
            Log posterior probability.
        prior_cov : ndarray
            Diagonal of the prior covariance matrix.  Only returned if
            `return_prior_cov` is True.
        
        Notes
        -----
        * This function does not currently support noise fitting

        """
        dmps = self._calculate_dmps(ps=ps, dmps=dmps)
        map_coeffs, dbar_SigmaI_dbar, prior_cov, log_det_Sigma = \
            self.pspp.calc_SigmaI_dbar_wrapper(
                dmps,
                self.pspp.T_Ninv_T,
                self.pspp.dbar
            )
        map_vis = np.dot(self.T, map_coeffs)

        log_det_prior_cov = -1 * np.sum(np.log(prior_cov)).real
        log_post = (
            -0.5*log_det_Sigma
            - 0.5*log_det_prior_cov
            + 0.5*dbar_SigmaI_dbar
        )
        if self.pspp.uprior_inds is not None:
            log_post += np.sum(np.log(dmps[self.pspp.uprior_inds]))
        log_post = log_post.real

        if not return_prior_cov:
            return map_coeffs, map_vis, log_post
        else:
            return map_coeffs, map_vis, log_post, prior_cov

    def calculate_prior_covariance(
        self,
        ps=None,
        dmps=None,
    ):
        """
        Calculate the prior covariance matrix \Phi^{-1}.

        Parameters
        ----------
        ps : float or array-like, optional
            Expected power spectrum, P(k).  Can be a single float (for a flat
            P(k)) or an array-like with shape `(self.k_vals.size,)`.
        dmps : float or array-like, optional
            Expected dimensionless power spectrum, \Delta^2(k).  Can be a
            single float or an array-like with shape `(self.k_vals.size,)`.

        Returns
        -------
        PhiI : ndarray
            Diagonal of the prior covariance matrix.

        """
        if ps is None and dmps is None:
            raise TypeError("One of 'ps' or 'dmps' must not be None")
        dmps = self._calculate_dmps(ps=ps, dmps=dmps)
        PhiI = self.pspp.calc_PowerI(dmps)

        return PhiI
    
    def _calculate_dmps(self, ps=None, dmps=None):
        """
        Calculated the expected dimensionless power spectrum.

        Parameters
        ----------
        ps : float or array-like, optional
            Expected power spectrum, P(k).  Can be a single float (for a flat
            P(k)) or an array-like with shape `(self.k_vals.size,)`.
        dmps : float or array-like, optional
            Expected dimensionless power spectrum, \Delta^2(k).  Can be a
            single float or an array-like with shape `(self.k_vals.size,)`.

        Returns
        -------
        dmps : ndarray
            Array of dimensionless power spectrum amplitudes with shape
            `(self.k_vals.size,)`.

        """        
        if ps is not None:
            if hasattr(ps, "__iter__"):
                ps = np.array(ps)
                assert ps.size == self.k_vals.size, (
                    f"'ps' has a size {ps.size} which is not "
                    f"equal to the number of k bins, {self.k_vals.size}"
                )
            else:
                # Flat P(k)
                ps *= np.ones_like(self.k_vals)
            dmps = self.k_vals**3 / (2 * np.pi**2) * ps
        if dmps is not None:
            if hasattr(dmps, "__iter__"):
                dmps = np.array(dmps)
                assert dmps.size == self.k_vals.size, (
                    f"'dmps' has a size {dmps.size} which is "
                    f"not equal to the number of k bins, {self.k_vals.size}"
                )
            else:
                # Flat \Delta^2(k)
                dmps *= np.ones_like(self.k_vals)

        return dmps