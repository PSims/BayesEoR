""" Class and functions for performing maximum a posteriori calculations. """
import numpy as np
from numpy.typing import ArrayLike
from astropy import units
from astropy.time import Time
from pathlib import Path
from pprint import pprint
import matplotlib.pyplot as plt

from rich.panel import Panel

from ..params import BayesEoRParser
from ..setup import run_setup
from ..utils import mpiprint

class MaximumAPosteriori(object):
    """
    Class for performing maximum a posteriori calculations.

    Parameters
    ----------
    config : str
        Path to a BayesEoR yaml configuration file.
    data_path : str, optional
        Path to either a pyuvdata-compatible file containing visibilities or
        a numpy-compatible file containing a preprocessed visibility vector.
        Supersedes the `data_path` in `config`. Defaults to None (use the
        `data_path` in `config`).
    array_dir : str, optional
        Path to a directory containing T, Ninv, and T_Ninv_T. Supersedes the
        `array_dir` created based on the parameters in `config`. Defaults to
        None (use the `array_dir` created from the parameters in `config`).
    verbose : bool
        Verbose output. Defaults to False.
    
    """
    def __init__(
        self,
        *,
        config : Path | str,
        data_path : str | None = None,
        array_dir : str | None = None,
        verbose : bool = False
    ):
        print_rank = 1 - verbose

        if not Path(config).exists():
            raise FileNotFoundError(f"{config} does not exist")
        parser = BayesEoRParser()
        args = parser.parse_args(["--config", config])
        args.verbose = verbose
        if verbose:
            mpiprint("\n", Panel("Parameters"))
            pprint(args.__dict__)

        if data_path is not None:
            if not Path(data_path).exists():
                raise FileNotFoundError(f"{data_path} does not exist")
            data_paths_differ = (
                Path(data_path).absolute().as_posix()
                != Path(args.data_path).absolute().as_posix()
            )
            if data_paths_differ:
                mpiprint("\nReplacing data_path:", style="bold", rank=print_rank)
                mpiprint(f"{args.data_path = }", rank=print_rank)
                mpiprint(f"{data_path      = }", rank=print_rank)
                args.data_path = data_path
            update_pspp = data_paths_differ
        else:
            update_pspp = False
        
        if array_dir is not None:
            if not Path(array_dir).exists():
                raise FileNotFoundError(f"{array_dir} does not exist")
        
        pspp, _, vis_dict, bm = run_setup(
            **args,
            return_vis=True,
            return_bm=True,
            save_k_vals=False,
            mkdir=False
        )

        if array_dir is not None:
            array_dirs_differ = (
                Path(array_dir).absolute().as_posix()
                != Path(bm.array_dir).absolute().as_posix()
            )
            if array_dirs_differ:
                mpiprint(
                    f"\nReplacing array_dir:",
                    style="bold",
                    end="\n\n",
                    rank=print_rank
                )
                mpiprint(f"{bm.array_dir = }", rank=print_rank)
                mpiprint(f"{array_dir    = }", rank=print_rank)
                bm.array_dir = array_dir
                update_pspp = True

        if update_pspp:
            mpiprint("\nCalculating dbar and d_Ninv_d...", rank=print_rank)
            vis_noisy = vis_dict["vis_noisy"]

            bm_verbose = bm.verbose
            bm.verbose = False
            Ninv = bm.read_data("Ninv")
            T = bm.read_data("T")
            Ninv_d = bm.dot_product(Ninv, vis_noisy)
            dbar = bm.dot_product(T.conj().T, Ninv_d)
            d_Ninv_d = np.dot(vis_noisy.conj(), Ninv_d)
            T_Ninv_T = bm.read_data("T_Ninv_T")
            bm.verbose = bm_verbose

            pspp.Ninv = Ninv
            pspp.dbar = dbar
            pspp.d_Ninv_d = d_Ninv_d
            pspp.T_Ninv_T = T_Ninv_T
            self.T = T
        else:
            T = bm.read_data("T")
            self.T = T
        
        self.args = args
        self.pspp = pspp
        self.bm = bm
        self.k_vals = pspp.k_vals
        if "vis" in vis_dict:
            self.s = vis_dict["vis"]
        if "noise" in vis_dict:
            self.n = vis_dict["noise"]
        self.d = vis_dict["vis_noisy"]
        self.freqs = vis_dict["freqs"] * units.Hz
        self.jds = Time(vis_dict["jds"], format="jd")
        self.uvws = vis_dict["uvws"]
        self.redundancy = vis_dict["redundancy"]
        if "antpairs" in vis_dict:
            self.antpairs = vis_dict["antpairs"]
    
    def map_estimate(
        self,
        ps : float | ArrayLike | None = None,
        dmps : float | ArrayLike | None = None,
        return_prior_cov : bool = False
    ):
        r"""
        Calculate the maximum a posteriori (MAP) model coefficients.

        Must pass one of `ps` or `dmps` which is required to calculate the
        prior covariance matrix.

        Parameters
        ----------
        ps : float or array-like, optional
            Expected power spectrum, :math:`P(k)`. Can be a single float (for
            a flat power spectrum) or an array-like with shape
            `(self.k_vals.size,)`. Required if `dmps` is None.
        dmps : float or array-like, optional
            Expected dimensionless power spectrum, :math:`\Delta^2(k)`. Can be
            a single float (for a flat dimensionless power spectrum) or an
            array-like with shape `(self.k_vals.size,)`. Required if `ps` is
            None.
        return_prior_cov : bool, optional
            Return the diagonal of the prior covariance matrix. Defaults to
            False.
        
        Returns
        -------
        map_coeffs : numpy.ndarray
            MAP coefficients of the model.
        map_vis : numpy.ndarray
            MAP model visibilities.
        log_post : float
            Log posterior probability.
        prior_cov : numpy.ndarray
            Diagonal of the prior covariance matrix.  Only returned if
            `return_prior_cov` is True.
        
        Notes
        -----
        * This function does not currently support noise fitting

        """
        if ps is None and dmps is None:
            raise ValueError("One of 'ps' or 'dmps' must not be None")
        dmps = self.calculate_dmps(ps=ps, dmps=dmps)

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
        ps : float | ArrayLike | None = None,
        dmps : float | ArrayLike | None = None
    ):
        r"""
        Calculate the prior covariance matrix :math:`\Phi^{-1}`.

        Parameters
        ----------
        ps : float or array-like, optional
            Expected power spectrum, :math:`P(k)`. Can be a single float (for
            a flat power spectrum) or an array-like with shape
            `(self.k_vals.size,)`. Required if `dmps` is None.
        dmps : float or array-like, optional
            Expected dimensionless power spectrum, :math:`\Delta^2(k)`. Can be
            a single float (for a flat dimensionless power spectrum) or an
            array-like with shape `(self.k_vals.size,)`. Required if `ps` is
            None.

        Returns
        -------
        PhiI : ndarray
            Diagonal of the prior covariance matrix.

        """
        if ps is None and dmps is None:
            raise ValueError("One of 'ps' or 'dmps' must not be None")
        dmps = self.calculate_dmps(ps=ps, dmps=dmps)
        PhiI = self.pspp.calc_PowerI(dmps)

        return PhiI
    
    def calculate_dmps(
        self,
        ps : float | ArrayLike | None = None,
        dmps : float | ArrayLike | None = None
    ):
        r"""
        Calculated the expected dimensionless power spectrum.

        Parameters
        ----------
        ps : float or array-like, optional
            Expected power spectrum, :math:`P(k)`. Can be a single float (for
            a flat power spectrum) or an array-like with shape
            `(self.k_vals.size,)`. Required if `dmps` is None.
        dmps : float or array-like, optional
            Expected dimensionless power spectrum, :math:`\Delta^2(k)`. Can be
            a single float (for a flat dimensionless power spectrum) or an
            array-like with shape `(self.k_vals.size,)`. Required if `ps` is
            None.

        Returns
        -------
        dmps : numpy.ndarray
            Array of dimensionless power spectrum amplitudes with shape
            `(self.k_vals.size,)`.

        """
        if ps is None and dmps is None:
            raise ValueError("One of 'ps' or 'dmps' must not be None")

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

    def extract_bl_vis(self, vis_vec : np.ndarray):
        """
        Form a dictionary of visibility waterfalls indexed by baseline.

        The input visibility vector should have shape `(nbls*nf*nt,)` where
        `nbls` is the number of baselines, `nf` the number of frequencies, and
        `nt` the number of times. The ordering of the visibility vector is
        expected to follow the ordering of the visibilities in `Finv`, i.e.
        the first `nbls` visibilities correspond to all baselines at the
        0th frequency and 0th time, the second `nbls` visibilities
        correspond to all baselines at the 1st frequency and 0th time, etc.
        The baseline axis evolves most rapidly, then frequency, and then
        time. In this ordering, a visibility waterfall for the i-th indexed
        baseline, `i_bl`, with shape `(nt, nf)` can be obtained via

        .. code_block:: python

            bl_vis = vis_vec[i_bl::nbls].reshape(nt, nf)

        Parameters
        ----------
        vis_vec : numpy.ndarray
            Visibility vector with shape `(nbls*nf*nt,)`.

        Returns
        -------
        bl_vis : dict
            Dictionary with keys of either baseline index or antenna pair tuple
            if `self.antpairs` exists.
        """
        nf = self.freqs.size
        nt = self.jds.size
        nbls = self.uvws.shape[1]

        if vis_vec.shape != (nbls*nf*nt,):
            raise ValueError(
                f"Shape mistmatch: `vis_vec` is expected to have shape "
                f"{(nbls*nf*nt,)} but has shape {vis_vec.shape}"
            )

        bl_vis = {}
        for i_bl in range(nbls):
            if hasattr(self, "antpairs"):
                key = self.antpairs[i_bl]
            else:
                key = i_bl
            bl_vis[key] = vis_vec[i_bl::nbls].reshape(nt, nf)

        return bl_vis