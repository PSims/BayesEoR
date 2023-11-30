Running BayesEoR
================

``run-analysis.py`` provides an example driver script for running BayesEoR.  This file contains all of the necessary steps to set up the :class:`bayeseor.posterior.PowerSpectrumPosteriorProbability` class and to run MultiNest and obtain power spectrum posteriors.  If using a configuration file, this driver script can be run via

.. code-block:: Bash
    
    python run-analysis.py --config /path/to/config.yaml


For additional help with running BayesEoR and setting analysis parameters, please see :ref:`setting-parameters`.  More information on running BayesEoR can be found below in :ref:`test-data`.

.. _test-data:

Test Dataset
------------

The BayesEoR repository provides a set of test data and an example yaml configuration file.  The test data contain mock EoR only simulated visibilities with a Gaussian beam and a full width at half maximum of 9.3 degrees.  For more information on the test data, see Section 3 of `Burba et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.4443B/abstract>`_.

There are currently two steps required to run a BayesEoR analysis

1. Build the required matrices (uses only CPUs, no GPUs required)
2. Run the power spectrum analysis (GPUs required)

To build the matrices (which will require ~17 GB of disk space) using the provided example yaml and test data, first navigate to the root directory of the BayesEoR repository and run

.. code-block:: Bash

    python run-analysis.py --config example-config.yaml --cpu

Note that with ``jsonargparse``, command line arguments that come after the `--config` flag overwrite the value of the argument in the configuration file.  In the above example, the `--cpu` flag placed after the `--config` flag will force the code to use CPUs only.

Once the matrices are built, you can run the power spectrum analysis (which requires GPUs) via

.. code-block:: Bash

    python run-analysis.py --config example-config.yaml --gpu

The expected power spectrum amplitude for the mock EoR signal in the test data is `P(k) = 214777.66068216303 mK^2 Mpc^3`.  BayesEoR outputs the dimensionless power spectrum which can be obtained from `P(k)` via Equation 13 in `Burba et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.4443B/abstract>`_.  The `k` bin values required to obtain the dimensionless power spectrum are written to disk automatically by BayesEoR (the default save location for the `k` values is `./k_vals/`).
