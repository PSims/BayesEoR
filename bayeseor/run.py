import numpy as np
from collections.abc import Sequence
from pathlib import Path
from pymultinest.solve import solve
import time

from .posterior import PriorC, PowerSpectrumPosteriorProbability
from .utils import mpiprint

def run(
    *,
    pspp : PowerSpectrumPosteriorProbability,
    priors : Sequence[float],
    n_live_points : int | None = None,
    calc_avg_eval : bool = False,
    avg_iters : int = 10,
    out_dir : Path | str = "./",
    sampler : str = "multinest",
    rank : int = 0
):
    """
    Run a power spectrum analysis.

    Parameters
    ----------
    pspp : :class:`bayeseor.posterior.PowerSpectrumPosteriorProbability`
        Power spectrum posterior class.
    priors : list of float
        Prior [min, max] for each k bin as a list, e.g. [[min1, max1],
        [min2, max2], ...].
    n_live_points : int, optional
        Number of live points. Defaults to None (sets the number of live
        points to `25 * pspp.k_vals.size`).
    calc_avg_eval : bool, optional
        Estimate the average evaluation time by calculating the posterior
        probability of the mean value of each prior range `avg_iters` times.
        Only computed if `rank` is 0. Defaults to False.
    avg_iters : int, optional
        Number of iterations to use to calculate the average posterior
        probability evaluation time. Used only if `calc_avg_eval` is True.
        Defaults to 10.
    out_dir : pathlib.Path or str, optional
        Sampler output directory. Must be less than 100 characters for
        compatibility with Multinest (the fortran code only supports a string
        with character length <= 100). It is encouraged you run BayesEoR from
        the directory where you would like the outputs to be written for this
        reason. Defaults to "./".
    sampler : {'multinest', 'polychord'}, optional
        Case insensitive string specifying the sampler, one of 'multinest' or
        'polychord'. Defaults to 'multinest'.
    rank : int, optional
        MPI rank. Defaults to 0.

    """
    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sampler_output_base = (out_dir / "data-").as_posix()

    sampler = sampler.lower()
    if sampler == "multinest":
        assert len(out_dir.as_posix()) <= 100, (
            "When using MultiNest, the path to the sampler output directory "
            "`out_dir` must be <= 100 characters in length"
        )
        # Log-likelihood wrapper function for MultiNest
        def loglikelihood(theta, calc_likelihood=pspp.posterior_probability):
            return calc_likelihood(theta)[0]
    elif sampler == "polychord":
        raise NotImplementedError("PolyChord will be supported in the future.")
    else:
        raise ValueError("sampler must be one of 'multinest' or 'polychord'")

    nkbins = pspp.k_vals.size
    if n_live_points is None:
        n_live_points = 25 * nkbins
    prior_c = PriorC(priors)
    # Suppress verbose output for power spectrum analysis only
    pspp.verbose = False

    if calc_avg_eval:
        # Compute the average posterior calculation time for
        # reference and check that this calculation returns
        # a finite value for the posterior probability
        mpiprint(
            "\nCalculating average posterior probability evaulation time:",
            style="bold"
        )
        start = time.time()
        pspp_verbose = pspp.verbose
        pspp.verbose = False
        for _ in range(avg_iters):
            post = pspp.posterior_probability(np.array(priors).mean(axis=1))[0]
            if not np.isfinite(post):
                mpiprint(
                    "WARNING: Infinite value returned in posterior calculation!",
                    style="bold red",
                    rank=rank
                )
        avg_eval_time = (time.time() - start) / avg_iters
        mpiprint(
            f"Average evaluation time: {avg_eval_time} s",
            rank=rank,
            end="\n\n"
        )

    if sampler == "multinest":
        result = solve(
            LogLikelihood=loglikelihood,
            Prior=prior_c.prior_func,
            n_dims=nkbins,
            outputfiles_basename=sampler_output_base,
            n_live_points=n_live_points
        )
    mpiprint("\nSampling complete!", rank=rank, end="\n\n")
