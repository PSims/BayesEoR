Running BayesEoR
================

``run-analysis.py`` provides an example driver script for running BayesEoR.  This file contains all of the necessary steps to set up the :class:`bayeseor.posterior.PowerSpectrumPosteriorProbability` class and to run MultiNest and obtain power spectrum posteriors.  If using a configuration file, this driver script can be run via

.. code-block:: Bash
    
    python run-analysis.py --config /path/to/config.yaml


For additional help with running BayesEoR and setting analysis parameters, please see :ref:`setting-parameters`.
