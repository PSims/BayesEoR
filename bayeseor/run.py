import numpy as np

from .posterior import PowerSpectrumPosteriorProbability
from .utils import mpiprint

def run(
    *,
    pspp : PowerSpectrumPosteriorProbability,
    out_dir : str = "./",
    sampler : str = "multinest",
    verbose : bool = False,
    rank : int = 0
):
    """
    Run a power spectrum analysis.

    Parameters
    ----------
    pspp : :class:`bayeseor.posterior.PowerSpectrumPosteriorProbability`
        Instantiated posterior class.
    out_dir : str
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
    required_kwargs = [pspp, out_dir, sampler]
    if not np.all([arg is not None for arg in required_kwargs]):
        raise ValueError(
            "The following kwargs are required and cannot be None: "
            "pspp, out_dir, sampler"
        )

    sampler = sampler.lower()
    if sampler == "multinest":
        def loglikelihood(theta, calc_likelihood=pspp.posterior_probability):
            return calc_likelihood(theta)[0]
    elif sampler == "polychord":
        raise NotImplementedError(
            "PolyChord will be supported in the future."
        )
    else:
        raise ValueError("sampler must be one of 'multinest' or 'polychord'")
    
    # print_rank will only trigger print if verbose is True and rank == 0
    print_rank = 1 - (verbose and rank == 0)

    mpiprint(
        "\nRunning power spectrum analysis...",
        style="bold",
        justify="center",
        rank=rank,
        end="\n\n"
    )
