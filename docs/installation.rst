Installation
============

Dependencies
------------

Hardware
^^^^^^^^

While it is in principle possible to run BayesEoR on CPUs, we strongly suggest using GPUs due to their increased speed and precision relative to CPU-based methods.


Software
^^^^^^^^

BayesEoR is written primarily in python.  It also uses a small wrapper to interface with the Matrix Algebra on GPU and Multicore Architectures (MAGMA) library written in C. As currently implemented, the following software dependencies must be installed to run BayesEoR on GPUs:

- `MAGMA <https://icl.cs.utk.edu/magma/>`_
- `CUDA <https://developer.nvidia.com/cuda-toolkit>`_
- MPI
- `MultiNest <https://github.com/JohannesBuchner/MultiNest>`_

On CPUs, the linear algebra functionality is implemented via ``scipy`` and thus ``MAGMA`` and ``CUDA`` are not required.  However, as mentioned above, we recommend running BayesEoR on GPUs due to their speed and precision.

All of the required dependencies can be installed via ``mamba`` (recommended, more info `here <https://mamba.readthedocs.io/en/latest/>`_) or ``conda`` using the provided ``environment.yaml`` file:

.. literalinclude:: ../environment.yaml
    :language: yaml

You can install all of these dependencies with ``mamba`` from the root directory of the BayesEoR repo via

.. code-block:: bash

    mamba env create -f environment.yaml

If you use ``conda``, simply replace ``conda`` with ``mamba`` in the above command.

If you wish to use an existing installation of ``CUDA``, please comment out ``cuda`` in ``environment.yaml`` before installing the dependencies with ``mamba`` / ``conda``.  As long as a valid path to the existing ``CUDA`` binaries, e.g. ``/usr/local/cuda/bin``, is present in your bash ``PATH`` variable, ``pycuda`` will be automatically configured to point to the existing ``CUDA``.

If you wish to use an existing installation of ``MPI``, please comment out ``mpi4py`` and ``pymultinest`` in ``environment.yaml`` before installing the dependencies with ``mamba`` / ``conda``.  ``pymultinest`` requires ``mpi4py`` as a dependency, so it too must be commented out.  You can then install ``mpi4py`` via ``pip``.  For ``mpi4py``, see `this article <https://researchcomputing.princeton.edu/support/knowledge-base/mpi4py>`_ to ensure ``mpi4py`` points to the desired ``MPI`` installation.  Once ``mpi4py`` has been installed with ``pip``, you can proceed with installing ``pymultinest`` via ``mamba`` / ``conda``.

If you wish to use an existing installation of ``MultiNest``, please comment out ``pymultinest`` in ``environment.yaml`` before installing the dependencies with ``mamba`` / ``conda``.  ``pymultinest`` can be installed with ``pip`` and forced to point to a particular installation by including the ``MultiNest`` installation in your ``LD_LIBRARY_PATH``.  See the `pymultinest documentation <https://johannesbuchner.github.io/PyMultiNest/install.html>`_ for more details.



Installing BayesEoR
-------------------

Once you have satisfied the above dependencies, you can install the BayesEoR python package (``bayeseor``) via ``pip`` with

.. code-block:: bash

    pip install .


Updating BayesEoR
^^^^^^^^^^^^^^^^^

If you have recently pulled changes and wish to update your version of ``bayeseor``, it is `suggested <https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html#caching-and-troubleshooting>`_ that you remove the ``build``, ``dist``, ``bayeseor.egg-info``, and ``bayeseor/_version.py`` files before attempting to update via

.. code-block:: bash

    pip install .