.. _running-bayeseor:

Running BayesEoR
================

``run-analysis.py`` provides an example driver script for running BayesEoR.  This file contains all of the necessary steps to set up the :class:`bayeseor.posterior.PowerSpectrumPosteriorProbability` class and to run MultiNest and obtain power spectrum posteriors.

There are currently two steps required to run a BayesEoR analysis

1. Build the required matrices (uses only CPUs, no GPUs required)
2. Run the power spectrum analysis (double precision GPUs required)

Below, we provide some useful information about these two steps.  For additional help with running BayesEoR and setting analysis parameters, please see :ref:`setting-parameters`.  More information on running BayesEoR can also be found below in :ref:`test-data`.


Building the matrix stack
^^^^^^^^^^^^^^^^^^^^^^^^^

If using a configuration file (recommended), this driver script can be run to build the matrices via

.. code-block:: Bash
    
    python run-analysis.py --config /path/to/config.yaml --cpu

Note that with ``jsonargparse``, command line arguments that come after the ``--config`` flag overwrite the value of the argument in the configuration file.  In the above example, the ``--cpu`` flag placed after the ``--config`` flag will force the code to use CPUs only.

BayesEoR automatically creates a directory in which to store the matrix stack if one does not already exist.  The name of the matrix stack directory is set automatically based on the chosen analysis parameters.  The prefix for this matrix stack directory can be set via the ``array_dir_prefix`` argument in the configuration yaml or the ``--array-dir-prefix`` flag on the command line.  The matrix stack is saved in a subdirectory within ``array_dir_prefix``.  The default matrix stack prefix is `./array-storage/`.


Running the power spectrum analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the matrices are built, you can run the power spectrum analysis (which requires double precision GPUs) via

.. code-block:: Bash
    
    python run-analysis.py --config /path/to/config.yaml --gpu

As above, the trailing ``--gpu`` flag will force the code to use GPUs.  The power spectrum analysis will only run if at least one GPU is found and the GPU initialization is succesful.



.. _output-location:

Outputs
-------

The location for the outputs of a BayesEoR analysis can be set via the ``output_dir`` argument in the configuration yaml or the ``--output-dir`` flag on the command line.  The output files from BayesEoR will be placed in a subdirectory of ``output_dir`` and the name of the subdirectory is set automatically based on the chosen analysis parameters.  The default output directory prefix is `./chains/`.

BayesEoR outputs a few key files.  All of these outputs are written to the same directory as the sampler outputs (see :ref:`output-location` above for more information).

1. ``args.json``: This file contains all of the configuration / command line arguments used for each analysis.

2.  ``k-vals*.txt``: These files contain information about the spherically-averaged k bins.  The files ``k-vals.txt``, ``k-vals-bins.txt``, and ``k-vals-nsamples.txt`` contain the central values of each k bin, the bin edges of each k bin, and the number of model k-cube voxels included in each k bin when spherically averaging.

3.  ``data-*``: These files contain the outputs of the sampler.  The most important of these files is ``data-.txt``.  This file contains the sampler output and has the power spectrum amplitude samples for each iteration.  For MultiNest outputs, this file has `Nkbins` + 2 columns where `Nkbins` is the number of spherically-averaged k bins.  The columns of interest in this file are the columns with index 0 and >= 2.  The 0th column contains the joint posterior probability value per iteration.  The columns with index >= 2 contain the power spectrum amplitude samples for each k bin.

For convenience, we have provided a class to aid in analyzing the aforementioned outputs of BayesEoR.  For more information on this class, please see :ref:`post-analysis-class`.



.. _test-data:

Test Dataset
------------

The BayesEoR repository provides a set of test data and an example yaml configuration file.  The test data contain mock EoR only simulated visibilities with a Gaussian beam and a full width at half maximum of 9.3 degrees.  For more information on the test data, see Section 3 of `Burba et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.4443B/abstract>`_.

To build the matrices (which will require ~17 GB of disk space) using the provided example configuration yaml and test data, first navigate to the root directory of the BayesEoR repository and run

.. code-block:: Bash

    python run-analysis.py --config example-config.yaml --cpu

Once the matrices are built, you can run the power spectrum analysis via

.. code-block:: Bash

    python run-analysis.py --config example-config.yaml --gpu

The mock EoR signal in the provided test data was generated as Gaussian white noise which has a flat power spectrum, `P(k) = 214777.66068216303 mK^2 Mpc^3`.  BayesEoR outputs the dimensionless power spectrum which can be obtained from `P(k)` via e.g. Equation 13 `Burba et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.4443B/abstract>`_.  The `k` bin values required to obtain the dimensionless power spectrum are written to disk automatically by BayesEoR.  The k bin values are written to the same directory as the sampler outputs (please see :ref:`running-bayeseor` above for more information).



.. _post-analysis-class:

Analyzing BayesEoR Outputs
--------------------------

We have provided a basic class for analyzing the outputs of BayesEoR.  The minimum requirement to instantiate the class is a list of directory names containing the BayesEoR output files.  There are also several kwargs you can set to calculate various quantities, compare the results with an expected power spectrum, and/or modify the attributes of the created plots.  Please see the class definition below for more information.

As an example, let us consider the case of analyzing the outputs of an analysis using the provided test data.

.. code-block:: python

    from pathlib import Path
    from bayeseor.utils.analyze_results import DataContainer

    dir_prefix = Path('./chains/')
    dirnames = ['MN-Test-15-15-38-0-2-6.2E-03-2.63-2.82-lp-dPS-v1']
    expected_ps = 214777.66068216303  # mK^2 Mpc^3

    data = DataContainer(
        dirnames, dir_prefix=dir_prefix, expected_ps=expected_ps, labels=['v1']
    )
    fig = data.plot_power_spectra_and_posteriors(
        suptitle='Test Data Analysis', plot_fracdiff=True
    )

In this example, we've assumed the default output location `./chains/`.  The subdirectory containing the BayesEoR output files is `./chains/MN-Test-15-15-38-0-2-6.2E-03-2.63-2.82-lp-dPS-v1/`.  Here, we are only analyzing the output from a single analysis.  If you wish to compare multiple analyses within the same directory, i.e. you have multiple subdirectories containing output files in `./chains/`, you can add more entries to `dirnames` e.g.

.. code-block:: python

    dirnames = ['MN-Test-15-15-38-0-2-6.2E-03-2.63-2.82-lp-dPS-v1',
                'MN-Test-15-15-38-0-2-6.2E-03-2.63-2.82-lp-dPS-v2',
                'MN-Test-15-15-38-0-2-6.2E-03-2.63-2.82-lp-dPS-v3']

The variable ``expected_ps`` in the example above has been set specifically for the test dataset.  The mock EoR signal in the test dataset has a flat power spectrum, P(k) (more info in the seciton above on :ref:`test-data`).  We thus only need to specify a floating point number for the expected P(k).  The class will internally convert this P(k) into the dimensionless power spectrum, or vice versa, based on the combination of the ``ps_kind`` kwarg (``'ps'`` for power spectrum or ``'dmps'`` for the dimensionless power spectrum) and the ``expected_ps`` or ``expected_dmps`` kwargs.  The default value of ``ps_kind`` is ``'dmps'``, but we've passed the class the ``expected_ps`` kwarg corresponding to the power spectrum.  The class will thus automatically convert this floating point P(k) into the corresponding dimensionless power spectrum using the k bins files in each output directory.

The ``DataContainer`` class also provides a few plotting functions.  In the example above, we're using the ``plot_power_spectra_and_posteriors`` function which creates a summary plot containing a subplot for the power spectrum estimates, an optional difference or fractional difference subplot if providing a known input power spectrum, and a subplot for the posterior of each k bin.  This code snippet will produce the following output if the analysis has been run correctly:

.. image:: ../test_data/test_data_results.png
    :width: 600


DataContainer Class Definition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: bayeseor.utils.analyze_results.DataContainer
    :members:
