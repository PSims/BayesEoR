import numpy as np
from collections.abc import Sequence
from pathlib import Path
from pymultinest.solve import solve
from rich.panel import Panel

from .posterior import PriorC, PowerSpectrumPosteriorProbability
from .utils import mpiprint

def run(
    *,
    pspp : PowerSpectrumPosteriorProbability,
    priors : Sequence[float],
    n_live_points : int | None = None,
    out_dir : Path | str = "./",
    sampler : str = "multinest",
    verbose : bool = False,
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
    out_dir : pathlib.Path or str, optional
        Sampler output directory. Must be less than 100 characters for
        compatibility with Multinest (the fortran code only supports a string
        with character length <= 100). It is encouraged you run BayesEoR from
        the directory where you would like the outputs to be written for this
        reason. Defaults to "./".
    sampler : {'multinest', 'polychord'}, optional
        Case insensitive string specifying the sampler, one of 'multinest' or
        'polychord'. Defaults to 'multinest'.
    verbose : bool, optional
        Verbose output. Defaults to False.
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
    pspp.verbose = verbose

    mpiprint("\n", Panel("Analysis"), rank=rank)
    if sampler == "multinest":
        result = solve(
            LogLikelihood=loglikelihood,
            Prior=prior_c,
            n_dims=nkbins,
            outputfiles_basename=sampler_output_base,
            n_live_points=n_live_points
        )
    mpiprint("\nSampling complete!", rank=rank, end="\n\n")
