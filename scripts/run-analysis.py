""" Example driver script for running BayesEoR """
import numpy as np
import os
from pathlib import Path
from pprint import pprint
from rich.panel import Panel
import sys

import sys
sys.path.insert(0, "/home/jburba/src/BayesEoR/")
print(f"{sys.path = }")

from bayeseor.params import BayesEoRParser
from bayeseor.run import run
from bayeseor.setup import run_setup
from bayeseor.utils import mpiprint, write_log_files

if __name__ == "__main__":
    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    mpiprint(f"\nmpi_size: {mpi_size}", rank=rank, end="\n\n")
else:
    # Skip mpi and other imports that can cause crashes in ipython
    rank = 0

parser = BayesEoRParser()
args = parser.parse_args()
if rank == 0 and args.verbose:
    mpiprint(Panel("Parameters"))
    if rank == 0 and args.config:
        mpiprint(
            f"\nConfig file: {Path(args.config[0]).absolute().as_posix()}",
            end="\n\n",
            rank=rank
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
    if rank == 0 and args.verbose:
        mpiprint("\n", Panel("Analysis"))
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
        # Write log files containing analysis parameters
        # and git version info for posterity
        write_log_files(parser, args, out_dir=out_dir, verbose=args.verbose)
    
    if args.use_Multinest:
        sampler = "multinest"
    else:
        sampler = "polychord"

    # Run the power spectrum analysis
    run(
        pspp=pspp,
        priors=args.priors,
        calc_avg_eval=True,
        out_dir=out_dir,
        sampler=sampler,
        rank=rank
    )
