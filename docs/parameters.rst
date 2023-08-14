Parameters
==========

All user-definable analysis parameters in BayesEoR are handled via the :class:`bayeseor.params.BayesEoRParser` class:

.. autoclass:: bayeseor.params.BayesEoRParser
    :members:


.. _setting-parameters:

Setting Parameters
------------------

The ``jsonargparse`` package allows for all of these parameters to be set via the command line or a yaml configuration file.  An example yaml file has been provided (``example-config.yaml``)

.. literalinclude:: ../example-config.yaml
    :language: yaml

Any variable that can be set via a command line argument can also be set in this yaml configuration file (command line arguments containing dashes in the variable name must be replaced with underscores, i.e. the command line argument ``--data-path`` can be set in the configuration file via ``data_path: "/path/to/data.npy"``).  The example configuration file also specifies the minimally sufficient variables that must be set for a BayesEoR analysis.

The command line interface in BayesEoR uses several classes from the ``jsonargparse`` package, notably the ``ArgumentParser`` and ``ActionYesNo`` classes.  For more information on these classes, please refer to the ``jsonargparse`` `documentation <https://jsonargparse.readthedocs.io/en/stable/>`_.