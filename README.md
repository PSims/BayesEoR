# BayesEoR

A Bayesian approach to estimating the power spectrum of the Epoch of Reionization (EoR) from interferometric observations.

BayesEoR provides a means of performing a joint Bayesian analysis of models for large-spectral-scale foreground emission and a stochastic signal from redshifted 21-cm emission emitted by neutral Hydrogen during the EoR.

For a detailed description of the methodology, see [Sims et al. 2016](https://ui.adsabs.harvard.edu/link_gateway/2016MNRAS.462.3069S/doi:10.1093/mnras/stw1768) and [Sims et al. 2019](https://ui.adsabs.harvard.edu/link_gateway/2019MNRAS.484.4152S/doi:10.1093/mnras/stz153). For more detail on the methodology and a demonstration using simulated data, see [Sims and Pober 2019](https://ui.adsabs.harvard.edu/link_gateway/2019MNRAS.488.2904S/doi:10.1093/mnras/stz1888).

# Installation

## Hardware/Software Dependencies

BayesEoR relies on GPUs to perform a Cholesky decomposition on large matrices using the Matrix Algebra on GPU and Multicore Architectures (MAGMA) library. As currently implemented, the following software dependencies must be installed to run BayesEoR:
- [MAGMA](https://icl.cs.utk.edu/magma/)
- [CUDA](https://developer.nvidia.com/cuda-toolkit)
- MPI
- [MultiNest](https://github.com/JohannesBuchner/MultiNest)
- [PolyChord](https://cobaya.readthedocs.io/en/latest/sampler_polychord.html) (better performance than MultiNest for large parameter spaces)

BayesEoR has been tested with:
- NVIDIA p100 and v100 GPU architectures
- MAGMA versions 2.4.0 and 2.5.4
- `conda`'s default MPI (mpich) and OpenMPI 4.0.5
- Cuda 9.1.85.1 and 11.1.1
- `conda`'s default MultiNest and a source installation

There has been no observable difference in execution time of BayesEoR when using the `conda` installation of MPI or MultiNest versus using source installations.

## Python Dependencies

BayesEoR is written primarily in python, with the exception of the MAGMA interface which is written in C (and wrapped in python). The required python dependencies are:
- python>=3.6
- numpy>=1.19.2
- pymultinest>=2.10 (python wrapper of MultiNest, a nested sampler)
- astropy>=4.0
- pyuvdata
- scipy>=1.0.1
- astropy-healpix>=0.5
- h5py
- gcc_linux-64

The following additional required dependencies must be installed with `pip`:
- pycuda 
- mpi4py>=3.0.0 (see [this article](https://researchcomputing.princeton.edu/support/knowledge-base/mpi4py) to ensure mpi4py points to the desired MPI)
Otherwise, `conda` will install it's own versions of CUDA and MPI which might be undesirable. If using a custom implementation of MultiNest, pymultinest can also be installed with `pip` and forced to point to a particular installation by including the MultiNest installation in `LD_LIBRARY_PATH`.  See the pymultinest [documentation](https://johannesbuchner.github.io/PyMultiNest/install.html) for more details.

For the `conda` dependencies, a yaml file `environment.yml` has been included in the main directory of BayesEoR for convenience.  A new `conda` environment can be created using
```
conda env create -f environment.yml
```
If installing CUDA and MPI with `conda`, you must uncomment the last two lines of the yaml file before exeucuting the above `conda` command.  Otherwise, those dependencies must be installed manually by the user.

## Running BayesEoR

There are two ways to interface with variables in BayesEoR: command line arguments or `Params/params.py`.  Any arguments passed via the command line interface `Params/command_line_arguments.py` will overwrite the corresponding value in the params file.  For a list of available command line arguments, run:
```
python run_EoR.py --help
```

`run_EoR.py` provides an example driver script for running BayesEoR.  This file contains all of the necessary steps to set up the `PowerSpectrumPosteriorProbability` class and to run MultiNest (or PolyChord for large parameter spaces).