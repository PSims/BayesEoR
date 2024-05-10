.. BayesEoR documentation master file, created by
   sphinx-quickstart on Thu Aug 10 16:18:23 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BayesEoR
========

A Bayesian approach to estimating the power spectrum of the Epoch of Reionization (EoR) from interferometric observations.

Observations of the highly redshifted 21-cm signal from neutral hydrogen during the EoR has proven difficult for the 21-cm community.  This is primarily due to the presence of bright, contaminating sources in between us and the 21-cm signal we wish to detect, e.g. our galaxy and other galaxies.  We refer to these contaminating sources as foregrounds.  Because of the large dynamic range between the EoR and foreground signals (~5 orders of magnitude), dealing with foregrounds during power spectrum estimation requires great care.  While Bayesian analyses of interferometric 21-cm data have become more popular in recent years (see e.g. `Ghosh et al. 2015 <https://ui.adsabs.harvard.edu/abs/2015MNRAS.452.1587G/abstract>`_, `Zhang et al. 2016 <https://ui.adsabs.harvard.edu/abs/2016ApJS..222....3Z/abstract>`_, `Ghosh et al. 2020 <https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.2813G/abstract>`_, `Kennedy et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023ApJS..266...23K/abstract>`_), the existing approaches lack the ability to account for the covariance between the observed EoR and foreground signals.  Intrinsically, the EoR and foregrounds signals are uncorrelated.  The instrument modulates both signals identically during observation, however, making them covariant.  This covariance can be accounted for by forward modelling both signals, a key advantage of our approach in `BayesEoR`. Please see the introduction of `Burba et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.4443B/abstract>`_ for a more detailed discussion of the differences and advantages of our approach to power spectrum estimation.

BayesEoR provides a means of performing a joint Bayesian analysis of models for large-spectral-scale foreground emission and a stochastic signal from redshifted 21-cm emission emitted by neutral Hydrogen during the EoR.  For a detailed description of the methodology, see `Sims et al. 2016 <https://ui.adsabs.harvard.edu/link_gateway/2016MNRAS.462.3069S/doi:10.1093/mnras/stw1768>`_ and `Sims et al. 2019 <https://ui.adsabs.harvard.edu/link_gateway/2019MNRAS.484.4152S/doi:10.1093/mnras/stz153>`_. For more detail on the methodology and demonstrations using simulated data, see `Sims and Pober 2019 <https://ui.adsabs.harvard.edu/link_gateway/2019MNRAS.488.2904S/doi:10.1093/mnras/stz1888>`_ and `Burba et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.4443B/abstract>`_.

Please note that BayesEoR is designed for use on high performance computing systems which utilize double precision GPUs.  While some components of the code can run on CPUs, the power spectrum analysis requires the speed and precision of GPUs.

Table of Contents
"""""""""""""""""

.. toctree::
   :maxdepth: 2
   
   installation
   parameters
   usage
   class_list
