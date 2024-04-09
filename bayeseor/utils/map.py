""" Class and functions for performing maximum a posteriori calculations. """
import numpy as np
from pathlib import Path
import json
from jsonargparse import Namespace
import matplotlib.pyplot as plt

from ..matrices import BuildMatrixTree
from ..model import (
    generate_k_cube_in_physical_coordinates,
    mask_k_cube,
    generate_k_cube_model_spherical_binning,
    calc_mean_binned_k_vals
)
from ..params import calculate_derived_params
from ..posterior import PowerSpectrumPosteriorProbability

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
        k_vals = calc_mean_binned_k_vals(
            mod_k_vo,
            k_cube_voxels_in_bin,
            rank=1-int(verbose)
        )

        # Load matrices
        bm = BuildMatrixTree(
            array_dir.as_posix(),
            args.include_instrumental_effects,
            args.use_sparse_matrices
        )
        Ninv = bm.read_data(array_dir / "Ninv.npz", "Ninv")
        T = bm.read_data(array_dir / "T.h5", "T")
        T_Ninv_T = bm.read_data(array_dir / "T_Ninv_T.h5", "T_Ninv_T")

        d = np.load(data_path)
        Ninv_d = np.dot(Ninv, d)
        d_Ninv_d = np.dot(d.conjugate(), Ninv_d)
        dbar = np.dot(T.conjugate().T, Ninv_d)
        
        self.pspp = PowerSpectrumPosteriorProbability(
            T_Ninv_T,
            dbar,
            k_vals,
            k_cube_voxels_in_bin,
            args.nuv,
            args.neta,
            args.nf,
            args.nq,
            Ninv,
            d_Ninv_d,
            args.redshift,
            args.ps_box_size_ra_Mpc,
            args.ps_box_size_dec_Mpc,
            args.ps_box_size_para_Mpc,
            include_instrumental_effects=args.include_instrumental_effects,
            log_priors=args.log_priors,
            uprior_inds=args.uprior_inds,
            inverse_LW_power=args.inverse_LW_power,
            dimensionless_PS=True,  # Currently only support \Delta^2(k)
            block_T_Ninv_T=None,
            intrinsic_noise_fitting=args.use_intrinsic_noise_fitting,
            use_shg=args.use_shg,
            rank=0,
            use_gpu=True,
            print=args.verbose
        )

        self.args = args
    
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
        map_estimate : ndarray
            MAP coefficients of the model.
        prior_cov : ndarray
            Diagonal of the prior covariance matrix.

        """
        dmps = self._calculate_dmps(ps=ps, dmps=dmps)
        map_estimate, _, prior_cov, _ = self.pspp.calc_SigmaI_dbar_wrapper(
            dmps,
            self.pspp.T_Ninv_T,
            self.pspp.dbar
        )

        if not return_prior_cov:
            return map_estimate
        else:
            return map_estimate, prior_cov

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