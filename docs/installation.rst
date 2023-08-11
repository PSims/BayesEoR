Installation
============

Hardware/Software Dependencies
------------------------------

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

A Note on Using CPUs
^^^^^^^^^^^^^^^^^^^^

While it is in principle possible to run BayesEoR on CPUs, we strongly suggest using GPUs due to their increased speed and precision relative to CPU-based methods.



Python Dependencies
-------------------

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