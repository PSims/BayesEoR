Installation
============

Hardware/Software Dependencies
------------------------------

BayesEoR relies on GPUs to perform a Cholesky decomposition on large matrices using the Matrix Algebra on GPU and Multicore Architectures (MAGMA) library. As currently implemented, the following software dependencies must be installed to run BayesEoR:

- `MAGMA <https://icl.cs.utk.edu/magma/>`_
- `CUDA <https://developer.nvidia.com/cuda-toolkit>`_
- MPI
- `MultiNest <https://github.com/JohannesBuchner/MultiNest>`_

BayesEoR has been succesfully run with

- GPUs: NVIDIA P100, V100, and A100 architectures
- MAGMA: 2.4.0, 2.5.4, and 2.7.1
- MPI: ``conda`` installation (mpich) and OpenMPI 4.0.5
- CUDA: 9.1.85.1 and 11.1.1
- MultiNest: ``conda`` installation and a source installation

This is not an exhaustive list of software versions which are compatible with our analysis, just a guide of what versions we have used succesfully in our BayesEoR analyses.



A Note on Using CPUs
""""""""""""""""""""

While it is in principle possible to run BayesEoR on CPUs, we strongly suggest using GPUs due to their increased speed and precision relative to CPU-based methods.



MAGMA Install Notes
"""""""""""""""""""

We have found that MAGMA performs best when used with Intel's math kernel library (`MKL <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html>`_) when running on Intel-based machines.  It is also possible to build MAGMA with `OpenBLAS <https://www.openblas.net/>`_.  For more configuration options when installing MAGMA, consult the README included with your source download.  Below, we also provide some installation notes that proved useful when installing MAGMA.

- `Download <https://icl.cs.utk.edu/magma/>`_ the latest version (2.7.1 at the time of this writing) and checkout README for some simple quick start instructions.  Upon extracting the files, this makes a directory `magma-<version>/`.
- There are a variety of example make.inc files in the `magma-<version>/make.inc-examples` directory.  For reference, we have used the `make.inc.mkl-gcc` file.  Copy whichever example make file you'd like to use to `magma-<version>/make.inc`.
- If you're using ``CUDA`` and ``MKL``, make sure to uncomment the lines in the `make.inc` file that set the ``MKLROOT`` and ``CUDADIR`` bash environment variables.  Otherwise, uncomment the appropriate lines for the libraries you are using.  These lines are commented by default.  Set these variables to their appropriate paths on your machine.
- We also found it necessary to append the path to our installed ``CUDA`` binaries to the bash ``PATH`` variable, in our case by executing ``export PATH=/usr/local/cuda/bin:$PATH``, because the installer requires use of NVIDIA's ``nvcc`` compiler.
- Inside the `Makefile`, set the prefix variable to install ``MAGMA`` to your desired location.
- Install
- Note that running ``make`` might take a while.  We have seen ``make`` take as long as 1-1.5 hours.  Consider running ``make`` in a screen instance to avoid network issues.



Shared Library Creation
^^^^^^^^^^^^^^^^^^^^^^^

Included within our repo is a file containing the MAGMA calls written in C.  This C code must be compiled into a shared library and placed inside the directory `bayeseor/posterior/gpu_wrapper/`.  BayesEoR automatically looks for a shared library in this directory with name `wrapmzpotrf.so`.



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

If you with to install all of these dependencies with ``conda``, you can do so using the included ``environment.yaml`` file via

.. code-block:: bash

    conda env create -f environment.yaml


If you have pre-configured installations of CUDA or MPI, e.g. installations optimized/configured for a compute cluster, we suggest installing ``pycuda`` and/or ``mpi4py`` via ``pip`` (and commenting out ``pycuda`` and ``mpi4py`` in the ``environment.yaml`` file).  If you install these dependencies with ``conda``, ``conda`` will install its own CUDA and MPI binaries which may not be desirable.  For ``pycuda``, you need only have the path to your cuda binaries in your bash ``PATH`` variable prior to ``pip`` installation.  For ``mpi4py``, see `this article <https://researchcomputing.princeton.edu/support/knowledge-base/mpi4py>`_ to ensure ``mpi4py`` points to the desired MPI installation.

Similarly, if using a pre-configured implementation of MultiNest, pymultinest can also be installed with ``pip`` and forced to point to a particular installation by including the MultiNest installation in your ``LD_LIBRARY_PATH``.  See the `pymultinest documentation <https://johannesbuchner.github.io/PyMultiNest/install.html>`_ for more details.



Installing BayesEoR
-------------------

Once you have satisfied the above dependencies, you can install the BayesEoR python package (``bayeseor``) via ``pip`` with

.. code-block:: bash

    pip install .
