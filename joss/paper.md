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

# Summary 

`BayesEoR` is a GPU-accelerated, MPI-compatible Python package for estimating the power spectrum of redshifted 21-cm emission from interferometric observations of the Epoch of Reionization (EoR). Utilizing a Bayesian framework, `BayesEoR` jointly fits for the 21-cm EoR power spectrum and a "foreground" model, referring to bright, contaminating emission between us and the cosmological signal, and forward models the instrument with which these signals are observed.  To perform the sampling, we use `MultiNest` [@buchner:2014] which calculates the Bayesian evidence as part of the analysis.  Thus, `BayesEoR` can also be used as a tool for model selection [see e.g. @sims:2019a].

# Statement of need

Neutral hydrogen can undergo a spin-flip transition in which the quantum spins of the proton and electron transition from an aligned to an anti-aligned state, or vice versa, resulting in emission or absorption of a photon with a wavelength of 21-cm. The hydrogen 21-cm spin temperature quantifies the relative number densities of atoms in the aligned and anti-aligned states. Interferometric 21-cm cosmology experiments aim to measure the contrast between the 21-cm spin temperature of neutral hydrogen and the radio background temperature in the early Universe. By observing this signal at high redshift, we can learn a wealth of information about the state of the intergalactic medium during the first billion years of cosmic history.  This information can, in turn, be used to infer properties of the first stars and galaxies that transformed the hydrogen intergalactic medium from a cold neutral gas to a hot ionised plasma during the Epoch of Reionization (EoR). Modern interferometers like [HERA](http://reionization.org/), [LOFAR](https://www.mpifr-bonn.mpg.de/en/lofar), and the [MWA](http://www.mwatelescope.org/), have been designed to observe with many antennas simultaneously to maximize their sensitivity to the 21-cm signal from the EoR. These experiments have shown that detecting this signal is rife with difficulty [@hera:2022; @lofar:2020; @mwa:2020].  This is primarily due to the coupling of bright contaminating sources between us and the cosmological signal, referred to as "foregrounds", with the spectral structure imparted by the instrument. Existing approaches to recovering the 21-cm signal from the data lack direct modelling of the observed covariance between the 21-cm and foreground signals in the data.  Intrinsically, the 21-cm and foreground signals are uncorrelated.  The instrument modulates both signals identically during observation, however, making them covariant. This covariance can be accounted for by forward modelling both signals, a key advantage of our approach in `BayesEoR`.  For a detailed comparison of `BayesEoR` with other existing methods, please see section 7.1 of @sims:2019a and section 1 of @burba:2023.

`BayesEoR` is a GPU-accelerated, MPI-compatible Python implementation of a Bayesian framework designed to jointly model the 21-cm and foreground signals and forward model the instrument with which these signals are observed. Using these combined techniques, we can overcome the aforementioned difficulties associated with extracting a faint, background signal in the presence of bright foregrounds. `BayesEoR` enables one to sample directly from the marginal posterior distribution of the power spectrum of the underlying 21-cm signal in interferometric data, enabling recovery of statistically robust and unbiased[^1] estimates of the 21-cm power spectrum and its uncertainties [@sims:2016; @sims:2019a; @sims:2019b; @burba:2023].  The power spectrum estimates ($\Delta^2(k)$) from an analysis using the [test dataset](https://bayeseor.readthedocs.io/en/latest/usage.html#test-dataset) and [plotting code](https://bayeseor.readthedocs.io/en/latest/usage.html#analyzing-bayeseor-outputs) provided with `BayesEoR` can be found in \autoref{fig:example}.  This figure demonstrates the primary output of `BayesEoR`: a posterior distribution of the dimensionless power spectrum amplitude of the 21-cm EoR signal for each spherically-averaged $k$ bin in the model (right subplots in \autoref{fig:example}).  From these posteriors, we can derive power spectrum estimates and uncertainties (top left subplot in \autoref{fig:example}).  Mathematically, the spherically-averaged power spectrum $P(k)$ is calculated as
\begin{equation}
P(k_i) = \frac{1}{N_{k,i}}\sum_{\mathbf{k}}P(\mathbf{k})
\end{equation}
where $i$ indexes the spherically-averaged $k$ bins, the sum is performed over all $\mathbf{k}$ in a spherical shell satisfying $k_i \leq |\mathbf{k}| < k_i + \Delta k_i$, and $N_{k, i}$ is the number of voxels in the $i$-th spherical shell.  The spherically-averaged dimensionless power spectrum, $\Delta^2(k)$, which we infer in `BayesEoR`, is related to $P(k)$ via
\begin{equation}
\Delta^2(k_i) = \frac{k_i^3}{2\pi^2}P(k_i)
\end{equation}

# Running `BayesEoR`

Running a `BayesEoR` analysis requires an input dataset, a model of the instrument, and a set of analysis parameters.  A script is provided with `BayesEoR` for convenience which pre-processes a `pyuvdata`-compatible dataset [@hazelton:2017] of visibilities per baseline, time, and frequency into a one-dimensional data vector, the required form of the input dataset to `BayesEoR`.  As part of the inference, `BayesEoR` forward models the instrument which requires an instrument model containing the primary beam response of a "baseline" (pair of antennas) and the "uv sampling" (the length and orientation of each baseline in the data).  The primary beam response is passed via a configuration file or command line argument.  The uv sampling is generated by the aforementioned convenience script to ensure that the ordering of the baseline in the instrument model matches that in the visibility data vector.  Analysis parameters must also be set by the user to specify file paths to input and output data products and model parameters used to construct the data model (e.g. the number of frequencies and times in the data, the number of voxels in the model Fourier domain cube, the field of view of the sky model).  Note however that these analysis parameters must be chosen carefully based on the data to be analyzed (please see section 2.3 of @burba:2023 for more details on choosing model parameters).  Accordingly, because `BayesEoR` forward models the instrument, we generate a model of the sky as part of our model visibilities calculation.  When the EoR and foregrounds can be adequately described by a sky model with a field of view equal to the width of the primary beam, the memory requirements for a `BayesEoR` analysis are on the order of 10 GB.  This is the case for the provided test dataset for which the disk and RAM requirements are ~17 GB and ~12 GB, respectively.  The field of view for the EoR and foreground sky models can be set independently, however, which allows for the EoR to be modelled within the primary field of view of the telescope while the foregrounds can be modelled across the whole sky.  We wish to note however that, in this fashion, modelling the whole sky can be computationally demanding depending upon the data being analyzed.  For example, in @burba:2023 we show that analyzing a relatively modest dataset (compared to those typically analyzed by HERA, LoFAR, or the MWA) can require ~400 GB of RAM[^2]. Please see section 6.1 of @burba:2023 for more details.

[^1]: Recovery of unbiased estimates of the 21-cm power spectrum requires that the field of view of the foreground model encompasses the region of sky from which instrument-weighted foregrounds contribute significantly to the observed data [@burba:2023].
[^2]: `BayesEoR` uses a routine from the Matrix Algebra for GPU and Multicore Architectures ([MAGMA](https://icl.utk.edu/magma/)) library which allows for the use of matrices with a memory footprint larger than the available RAM on a GPU.

![Example outputs from a `BayesEoR` analysis of the provided test dataset with a known power spectrum. The top left subplot shows the inferred dimensionless power spectrum estimates with $1\sigma$ uncertanties in blue ($\Delta^2(k)$ as a function of spherically-averaged Fourier mode, $k$) and the expected dimensionless power spectrum as the black, dashed line.  The bottom left subplot shows the fractional difference between the recovered and expected power spectra. The right subplots show the posterior distribution of each power spectrum coefficient in the top left plot ($\varphi_i$ where $i$ indexes the spherically-averaged $k$ bins) in blue with the expected power spectrum amplitude as the black, vertical, dashed lines. \label{fig:example}](../test_data/test_data_results.png)

# Acknowledgements

The authors acknowledge support from NSF Awards 1636646 and 1907777, as well as Brown University's Richard B. Salomon Faculty Research Award Fund. JB also acknowledges support from a NASA RI Space Grant Graduate Fellowship. PHS was supported in part by a McGill Space Institute fellowship and funding from the Canada 150 Research Chairs Program. This result is part of a project that has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (Grant agreement No. 948764; JTB). This research was conducted using computational resources and services at the Center for Computation and Visualization, Brown University.

# References
