---
title: 'BayesEoR: Bayesian 21 cm Power Spectrum Estimation from Interferometric Visibilities'
tags:
 - radio astronomy
 - interferometry
 - gpu
 - power spectrum
 - bayes
 - epoch of reionization
authors:
 - name: Peter Sims
   orcid: 0000-0002-2871-0413
   affiliation: 1
#    equal-contrib: true
 - name: Jacob Burba
   orcid: 0000-0002-8465-9341
   affiliation: 2
   corresponding: true
 - name: Jonathan Pober
   orcid: 0000-0002-3492-0433
   affiliation: 3
affiliations:
 - name: McGill Space Institute, McGill University, Canada
   index: 1
 - name: Department of Physics and Astronomy, University of Manchester, UK
   index: 2
 - name: Department of Physics, Brown University, USA
   index: 3
date: 17 January 2024
bibliography: paper.bib

---

# Summary

The highly redshifted 21 cm signal from neutral hydrogen from the early universe contains a wealth of information about dark matter, dark energy, and the evolution of the first stars and galaxies.  Modern interferometers like LoFAR (https://www.mpifr-bonn.mpg.de/en/lofar), MWA (http://www.mwatelescope.org/), and HERA (http://reionization.org/) have been designed to observe with many antennas simultaneously to maximize their sensitivity to the 21 cm signal.  These experiments have shown that detecting this signal is rife with difficulty [@lofar:2020; @mwa:2020; @hera:2020], primarily due to the presence of bright, contaminating sources in between us and the cosmological signal referred to as "foregrounds".  BayesEoR is a GPU-accelerated, python-based pipeline designed to overcome this difficulty by jointly modelling the 21 cm and foreground signals and forward modelling the instrument with which these signals are observed in a Bayesian framework to extract statistically robust and unbiased estimates of the 21 cm signal in the form of its power spectrum [@sims:2016; @sims:2019a; @sims:2019b; @burba:2023].  

# Acknowledgements

The authors acknowledge support from NSF Awards 1636646 and 1907777, as well as Brown University's Richard B. Salomon Faculty Research Award Fund. JB also acknowledges support from a NASA RI Space Grant Graduate Fellowship. PHS was supported in part by a McGill Space Institute fellowship and funding from the Canada 150 Research Chairs Program.  This result is part of a project that has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (Grant agreement No. 948764; JTB).  This research was conducted using computational resources and services at the Center for Computation and Visualization, Brown University.

# References
