""" Example driver script for running BayesEoR """
import numpy as np
import os
from pathlib import Path
from pprint import pprint
from rich.panel import Panel
import sys
import time

from bayeseor.params import BayesEoRParser
from bayeseor.run import run
from bayeseor.setup import run_setup
from bayeseor.utils import mpiprint, write_log_files

parser = BayesEoRParser()
cl_args = parser.parse_args(derived_params=False)
# Calculate derived parameters from command line arguments. For now,
# calculate_derived_params returns a new jsonargparse.Namespace instance.
# Attributes of the Namespace must be linked to a parser argument for
# jsonargparse.ArgumentParser.save to function properly and this
# save function is currently used in bayeseor.utils.write_log_files.
args = parser.calculate_derived_params(cl_args)

if __name__ == "__main__":
    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    mpiprint(f"\nmpi_size: {mpi_size}", rank=rank, end="\n\n")
    if args.use_Multinest:
        from pymultinest.solve import solve
    else:
        import PyPolyChord.PyPolyChord as PolyChord
else:
    # Skip mpi and other imports that can cause crashes in ipython
    rank = 0

if rank == 0 and args.verbose:
    mpiprint(Panel("Parameters"), style="bold")
    if args.config:
        mpiprint(
            f"\nConfig file: {Path(args.config[0]).absolute().as_posix()}",
            end="\n\n"
        )
    pprint(args.__dict__)

# Run all steps required to instantiate the
# PowerSpectrumPosteriorProbability class
pspp, out_dir = run_setup(**args)

if "SLURM_JOB_ID" in os.environ:
    # Create empty file named with the SLURM Job ID
    (out_dir / os.environ["SLURM_JOB_ID"]).touch()

if not args.run:
    mpiprint("\nSkipping sampling, exiting...", rank=rank, end="\n\n")
else:
    if args.use_gpu and not pspp.gpu.gpu_initialized:
        mpiprint(
            f"\nERROR: GPU initialization failed on rank {rank}. Aborting.\n",
            style="bold red",
            justify="center",
            rank=0
        )
        if mpi_comm is not None:
            mpi_comm.Abort(1)
        else:
            sys.exit()
    
    if rank == 0:
        # Compute the average posterior calculation time for
        # reference and check that this calculation returns
        # a finite value for the posterior probability
        start = time.time()
        pspp_verbose = pspp.verbose
        pspp.verbose = False
        for iter in range(10):
            L = pspp.posterior_probability(np.array(args.priors).mean(axis=1))
            if not np.isfinite(L):
                mpiprint(
                    "WARNING: Infinite value returned in posterior calculation!",
                    style="bold red",
                    justify="center",
                    rank=rank
                )
        avg_eval_time = (time.time() - start) / (iter + 1)
        mpiprint(f"\nAverage evaluation time {avg_eval_time} s\n\n", rank=rank)

        # Write log files containing analysis parameters
        # and git version info for posterity
        write_log_files(parser, cl_args, verbose=args.verbose)
    
    if args.use_Multinest:
        sampler = "multinest"
    else:
        sampler = "polychord"

    # Run the power spectrum analysis
    run(
        pspp,
        args.priors,
        out_dir=out_dir,
        sampler=sampler,
        verbose=args.verbose,
        rank=rank
    )
