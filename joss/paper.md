---
title: 'BayesEoR: Bayesian 21-cm Power Spectrum Estimation from Interferometric Visibilities'
tags:
 - radio astronomy
 - interferometry
 - gpu
 - power spectrum
 - bayes
 - epoch of reionization
authors:
 - name: Peter H. Sims
   orcid: 0000-0002-2871-0413
   affiliation: 1
   equal-contrib: true
 - name: Jacob Burba
   orcid: 0000-0002-8465-9341
   affiliation: 2
   equal-contrib: true
   corresponding: true
 - name: Jonathan C. Pober
   orcid: 0000-0002-3492-0433
   affiliation: 3
   equal-contrib: true
affiliations:
 - name: School of Earth and Space Exploration, Arizona State University, USA
   index: 1
 - name: Department of Physics and Astronomy, University of Manchester, UK
   index: 2
 - name: Department of Physics, Brown University, USA
   index: 3
date: 17 January 2024
bibliography: paper.bib

---

# Statement of need

The highly redshifted 21-cm signal from neutral hydrogen in the early universe contains a wealth of information about the state of the intergalactic medium during the first Gyr of cosmic history from which properties of the first stars and galaxies can be inferred. Modern interferometers like HERA (http://reionization.org/), LoFAR (https://www.mpifr-bonn.mpg.de/en/lofar), and the MWA (http://www.mwatelescope.org/), have been designed to observe with many antennas simultaneously to maximize their sensitivity to the 21-cm signal. These experiments have shown that detecting this signal is rife with difficulty [@hera:2022; @lofar:2020; @mwa:2020], primarily due to the presence of bright, contaminating sources between us and the cosmological signal, referred to as "foregrounds."  Because these foregrounds are several orders of magnitude brighter than the 21-cm signal we wish to detect, advanced data analysis tools are needed to extract accurate estimates of the faint, 21-cm signal.

# Summary

`BayesEoR` is a GPU-accelerated, Python-based implementation of a Bayesian framework designed to jointly model the 21-cm and foreground signals and forward model the instrument with which these signals are observed. Using these combined techniques, we can overcome the aforementioned difficulties associated with extracting a faint, background signal in the presence of bright foregrounds.  `BayesEoR` enables one to sample directly from the marginal posterior distribution of the 21-cm power spectrum of the underlying 21-cm signal in interferometric data, enabling recovery of statistically robust and unbiased estimates of the power spectrum and its uncertainties [@sims:2016; @sims:2019a; @sims:2019b; @burba:2023].  The outputs of an analysis using the [test dataset](https://bayeseor.readthedocs.io/en/latest/usage.html#test-dataset) and [plotting code](https://bayeseor.readthedocs.io/en/latest/usage.html#analyzing-bayeseor-outputs) provided with `BayesEoR` can be found in \autoref{fig:example}.

![Example outputs from a `BayesEoR` analysis of the provided test dataset with a known power spectrum.  The top left subplot shows the inferred power spectrum estimates in blue and the expected power spectrum as the black, dashed line.  The bottom left subplot shows the fractional difference between the recovered and expected power spectra.  The right subplots show the posterior distribution of each power spectrum coefficient in blue with the expected power spectrum amplitude as the black, vertical, dashed lines. \label{fig:example}](../test_data/test_data_results.png)

# Acknowledgements

The authors acknowledge support from NSF Awards 1636646 and 1907777, as well as Brown University's Richard B. Salomon Faculty Research Award Fund. JB also acknowledges support from a NASA RI Space Grant Graduate Fellowship. PHS was supported in part by a McGill Space Institute fellowship and funding from the Canada 150 Research Chairs Program. This result is part of a project that has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (Grant agreement No. 948764; JTB). This research was conducted using computational resources and services at the Center for Computation and Visualization, Brown University.

# References
