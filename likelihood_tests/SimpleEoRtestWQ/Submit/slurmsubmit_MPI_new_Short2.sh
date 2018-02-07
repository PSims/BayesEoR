#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=01:59:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=24

# Specify a job name:
#SBATCH -J PMNtest

# Specify an output file
#SBATCH -o BayesEoR-%j.out
#SBATCH -e BayesEoR-%j.out

# Setting mem=0 defaults to using all available memory on the node
# (see https://slurm.schedmd.com/sbatch.html)
#SBATCH --mem=0

##SBATCH --qos=jpober-condo


module load  openblas gcc/4.9.2 mvapich2
env
module list

export LD_PRELOAD=/gpfs/runtime/opt/mvapich2/2.0rc1/lib/libmpichf90.so.12:/gpfs/runtime/opt/mvapich2/2.0rc1/lib/libmpich.so.12:$LD_PRELOAD



cd /users/psims/EoR/Python_Scripts/BayesEoR/SpatialPS/PolySpatialPS/likelihood_tests/SimpleEoRtestWQ/
srun -n 2 python Likelihood_v1d763_3D_ZM_small.py --nq=2





