BayesEoR
========

A Bayesian approach to estimating the power spectrum of the Epoch of Reionization (EoR) from interferometric observations.

BayesEoR provides a means of performing a joint Bayesian analysis of models for large-spectral-scale foreground emission and a stochastic signal from redshifted 21-cm emission emitted by neutral Hydrogen during the EoR.

For a detailed description of the methodology, see [Sims et al. 2016](https://ui.adsabs.harvard.edu/link_gateway/2016MNRAS.462.3069S/doi:10.1093/mnras/stw1768) and [Sims et al. 2019](https://ui.adsabs.harvard.edu/link_gateway/2019MNRAS.484.4152S/doi:10.1093/mnras/stz153). For more detail on the methodology and demonstrations using simulated data, see [Sims and Pober 2019](https://ui.adsabs.harvard.edu/link_gateway/2019MNRAS.488.2904S/doi:10.1093/mnras/stz1888) and [Burba et al. 2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.4443B/abstract).

# Installation

### Hardware/Software Dependencies

BayesEoR relies on GPUs to perform a Cholesky decomposition on large matrices using the Matrix Algebra on GPU and Multicore Architectures (MAGMA) library. As currently implemented, the following software dependencies must be installed to run BayesEoR:
- [MAGMA](https://icl.cs.utk.edu/magma/)
- [CUDA](https://developer.nvidia.com/cuda-toolkit)
- MPI
- [MultiNest](https://github.com/JohannesBuchner/MultiNest)
<!-- - [PolyChord](https://cobaya.readthedocs.io/en/latest/sampler_polychord.html) (better performance than MultiNest for large parameter spaces) -->

BayesEoR has been succesfully run with:
- **GPUs:** NVIDIA P100, V100, and A100 architectures
- **MAGMA:** 2.4.0, 2.5.4, and 2.7.1
- **MPI:** `conda` installation (mpich) and OpenMPI 4.0.5
- **CUDA:** 9.1.85.1 and 11.1.1
- **MultiNest:** `conda` installation and a source installation

This is not an exhaustive list of software versions which are compatible with our analysis, just a guide of what versions we have used succesfully in our BayesEoR analyses.

#### A Note on Using CPUs

While it is in principle possible to run BayesEoR on CPUs, we strongly suggest using GPUs due to their increased speed and precision relative to CPU-based methods.

### Python Dependencies

BayesEoR is written primarily in python, with the exception of the MAGMA interface which is written in C (and wrapped in python). The required python dependencies are

- astropy
- astropy-healpix
- gcc_linux-64
- h5py
- jsonargparse
- mpi4py>=3.0.0
- numpy
- pip
- pycuda
- pymultinest
- python
- pyuvdata
- rich
- scipy
- setuptools
- setuptools_scm
- sphinx

If you with to install all of these dependencies with `conda`, you can do so using the included `environment.yaml` file via
```
conda env create -f environment.yaml
```

If you have pre-configured installations of CUDA or MPI, e.g. installations optimized/configured for a compute cluster, we suggest installing `pycuda` and/or `mpi4py` via `pip` (and commenting out `pycuda` and `mpi4py` in the `environment.yaml` file).  If you install these dependencies with `conda`, `conda` will install its own CUDA and MPI binaries which may not be desirable.  For `pycuda`, you need only have the path to your cuda binaries in your bash `PATH` variable prior to `pip` installation.  For `mpi4py`, see [this article](https://researchcomputing.princeton.edu/support/knowledge-base/mpi4py) to ensure `mpi4py` points to the desired MPI installation.

Similarly, if using a pre-configured implementation of MultiNest, pymultinest can also be installed with `pip` and forced to point to a particular installation by including the MultiNest installation in your `LD_LIBRARY_PATH`.  See the pymultinest [documentation](https://johannesbuchner.github.io/PyMultiNest/install.html) for more details.


# Documentation

Documentation on how to estimate the power spectrum of redshifted 21-cm emission in a radio interferometric data set using BayesEoR is hosted on [ReadTheDocs](https://bayeseor.readthedocs.io/en/docs/).


# Citation

Users of the code are requested to cite the BayesEoR papers: 

- https://ui.adsabs.harvard.edu/abs/2016MNRAS.462.3069S/abstract
- https://ui.adsabs.harvard.edu/abs/2019MNRAS.484.4152S/abstract
- https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2904S/abstract
- https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.4443B/abstract

in their publications.


# Running BayesEoR

There are two ways to interface with variables in BayesEoR: command line arguments or config files.  For a list of available command line arguments and their descriptions, run
```
python run-analysis.py --help
```
or see the [documentation](https://bayeseor.readthedocs.io/en/docs/) for more info on the analysis parameters used by BayesEoR.

The `jsonargparse` package allows for all of these command line arguments to be set via a yaml configuration file.  An example yaml file has been provided (`example-config.yaml`).  Any variable that can be set via a command line argument can also be set in this yaml configuration file (command line arguments containing dashes in the variable name must be replaced with underscores, i.e. the command line argument `--data-path` can be set in the configuration file via `data_path: "/path/to/data.npy"`).  The example configuration file also specifies the minimally sufficient variables that must be set for a BayesEoR analysis.

`run-analysis.py` provides an example driver script for running BayesEoR.  This file contains all of the necessary steps to set up the `PowerSpectrumPosteriorProbability` class and to run MultiNest and obtain power spectrum posteriors.  If using a configuration file, this driver script can be run via
```
python run-analysis.py --config /path/to/config.yaml
```
