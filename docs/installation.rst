Installation
============

Hardware/Software Dependencies
------------------------------

BayesEoR relies on GPUs to perform a Cholesky decomposition on large matrices using the Matrix Algebra on GPU and Multicore Architectures (MAGMA) library. As currently implemented, the following software dependencies must be installed to run BayesEoR:

- `MAGMA <https://icl.cs.utk.edu/magma/>`_
- `CUDA <https://developer.nvidia.com/cuda-toolkit>`_
- MPI
- `MultiNest <https://github.com/JohannesBuchner/MultiNest>`_

All of these dependcies can be installed via ``conda`` for ease of use.  Please see the section below on :ref:`python-dependencies` for more information.

BayesEoR has been succesfully run with

- GPUs: NVIDIA P100, V100, and A100 architectures
- MAGMA: ``conda`` and source installations
- MPI: ``conda`` installation (mpich) and OpenMPI 4.0.5
- CUDA: 9.1.85.1 and 11.1.1
- MultiNest: ``conda`` and source installations

This is not an exhaustive list of software versions which are compatible with our analysis, just a guide of what versions we have used succesfully in our own analyses.



A Note on Using CPUs
""""""""""""""""""""

While it is in principle possible to run BayesEoR on CPUs, we strongly suggest using GPUs due to their increased speed and precision relative to CPU-based methods.


.. _python-dependencies:

Python Dependencies
-------------------

BayesEoR is written in python. The required python dependencies are

- astropy
- astropy-healpix
- cuda
- gcc_linux-64
- h5py
- jsonargparse
- magma
- matplotlib
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

Alternatively, if you use ``mamba`` (recommended, more info `here <https://mamba.readthedocs.io/en/latest/>`_), you can simply replace ``conda`` with ``mamba`` in the above command, i.e.

.. code-block:: bash

    mamba env create -f environment.yaml

If you have a pre-configure installation of CUDA, we suggest commenting out ``cuda`` and ``pycuda`` in the ``environment.yaml`` file prior to executing the above ``conda`` command.  Similarly, to use a pre-configured MPI installation, comment out ``mpi4py`` (and ``pymultinest`` as it also installs a ``conda`` binary, see the paragraph below for installation instructions for ``pymultinest`` in this case) in the ``environment.yaml`` file.  You can then install ``mpi4py`` via ``pip``.  If you install these dependencies with ``conda``, ``conda`` will install its own CUDA and MPI binaries which may not be desirable.  For ``pycuda``, you need only have the path to your cuda binaries in your bash ``PATH`` variable prior to ``pip`` installation.  For ``mpi4py``, see `this article <https://researchcomputing.princeton.edu/support/knowledge-base/mpi4py>`_ to ensure ``mpi4py`` points to the desired MPI installation.

Similarly, if using a pre-configured implementation of MultiNest, pymultinest can also be installed with ``pip`` and forced to point to a particular installation by including the MultiNest installation in your ``LD_LIBRARY_PATH``.  See the `pymultinest documentation <https://johannesbuchner.github.io/PyMultiNest/install.html>`_ for more details.



Installing BayesEoR
-------------------

Once you have satisfied the above dependencies, you can install the BayesEoR python package (``bayeseor``) via ``pip`` with

.. code-block:: bash

    pip install .
