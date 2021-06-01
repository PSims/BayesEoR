import argparse
import numpy as np
from astropy import constants

"""
Analysis settings
"""

# --------------------------------------------
# User editable parameters
# --------------------------------------------

###
# MultiNest params
###
# If None, MultiNest will start a new analysis.
# Otherwise, resume analysis from ``file_root``.
file_root = None

###
# k-cube params
###
nf = 38
neta = 38
nu = 9
nv = None
nx = 9
ny = 9
nq = 0
# Subharmonic grid (SHG) params
nu_sh = 0
nv_sh = None
nq_sh = 0
npl_sh = 0

###
# EoR sim params
###
EoR_npz_path = ''
EoR_npz_path_sc = ''

###
# Frequency params
###
nu_min_MHz = 158.304048743
channel_width_MHz = 0.237618986858

###
# Sky model params
###
fov_ra_deg = 12.9080728652
fov_dec_deg = None
nside = 256

# --------------------------------------------
# Parameters below this shouldn't require editing
# --------------------------------------------

###
# GDSE foreground params
###
# Matches beta_150_408 in Mozden, Bowman et al. 2016
beta_experimental_mean = 2.63 + 0
# A conservative over-estimate of the dbeta_150_408=0.01
# (dbeta_90_190=0.02) in Mozden, Bowman et al. 2016
beta_experimental_std = 0.02
# Revise to match published values
gamma_mean = -2.7
# Revise to match published values
gamma_sigma = 0.3

# Matches GSM mean in region A
# Tb_experimental_mean_K = 194.0
# Matches GSM mean in region considered in S19b
# (see GSM_map_std_at_-30_dec_v1d3.ipynb)
Tb_experimental_mean_K = 471.0
# 70th percentile 12 deg.**2 region at 56 arcmin res. centered on
# -30. deg declination (see GSM_map_std_at_-30_dec_v1d0.ipynb)
Tb_experimental_std_K = 62.0
Tb_experimental_std_K = (
        Tb_experimental_std_K *
        (nu_min_MHz / 163.) ** (-beta_experimental_mean))

# Matches EoR sim (note: use closest odd val., so 127 rather than 128,
# for easier FFT normalisation)
# simulation_resolution_deg = simulation_FoV_deg / 511.
fits_storage_dir = ''
# HF_nu_min_MHz_array = [210,220,230]
HF_nu_min_MHz_array = [220]

###
# Diffuse free-free foreground params
###
beta_experimental_mean_ff = 2.15 + 0
beta_experimental_std_ff = 1.e-10
gamma_mean_ff = -2.59
gamma_sigma_ff = 0.04

Tb_experimental_mean_K_ff = Tb_experimental_mean_K / 100.0
Tb_experimental_std_K_ff = Tb_experimental_std_K / 100.0

nu_min_MHz_ff = 163.0 - 4.0
Tb_experimental_std_K_ff = (
        Tb_experimental_std_K_ff *
        (nu_min_MHz_ff / 163.) ** (-beta_experimental_mean_ff))

channel_width_MHz_ff = 0.2
simulation_FoV_deg_ff = 12.0  # Matches EoR simulation
# Matches EoR sim (note: use closest odd val., so 127 rather than 128,
# for easier FFT normalisation)
simulation_resolution_deg_ff = simulation_FoV_deg_ff / 511.
fits_storage_dir_ff = ''
# HF_nu_min_MHz_array = [210,220,230]
HF_nu_min_MHz_array_ff = [210]

###
# Extragalactic source foreground params
###
EGS_npz_path = ''

###
# Spectral model params
###
beta = [2.63, 2.82]
if beta:
    if type(beta) == list:
        npl = len(beta)
    else:
        npl = 1
else:
    npl = 0

###
# Accelerate likelihood on GPU
###
useGPU = True

###
# Useful constants
###
speed_of_light = constants.c.value

###
# Instrumental effects params
###
include_instrumental_effects = True
# Include minimal prior over LW modes to
# ensure numerically stable posterior
inverse_LW_power = 1.e-16

if include_instrumental_effects:
    ###
    # Obs params
    ###
    nt = 17
    integration_time_seconds = 11
    integration_time_minutes = integration_time_seconds/60
    integration_time_minutes_str = '{}'.format(
        integration_time_minutes).replace('.', 'd')
    instrument_model_directory = (
        '/users/jburba/data/jburba/bayes/BayesEoR/Instrument_Model/'
        'Hex_61-10m_healvis_model_for_{}_{}_sec_time_'
        'steps_bl_less_than_41m/'.format(nt, integration_time_seconds))
    # instrument_model_directory = (
    #     '/users/jburba/data/jburba/bayes/BayesEoR/Instrument_Model/'
    #     'Hex_61-10m_healvis_model_for_{}_{}_min_time_'
    #     'steps_bl_less_than_41m/'.format(nt, integration_time_minutes_str))
    telescope_latlonalt = (-30.72152777777791,
                           21.428305555555557,
                           1073.0000000093132)
    central_jd = 2458098.5521759833

    ###
    # Primary beam params
    ###
    beam_type = 'Airy'
    beam_peak_amplitude = 1.0
    FWHM_deg_at_ref_freq_MHz = None  # degrees
    PB_ref_freq_MHz = 150.0  # MHz
    antenna_diameter = None
    # Set the primary beam pointing center in (RA, DEC)
    # If None, will use the pointing center at zenith according to
    # telescope_latlonalt and central_jd. Otherwise, must be a tuple of
    # offsets in degrees along the RA and DEC axes defined relative to
    # the pointing center at zenith according to telescope_latlonalt
    # and central_jd.
    beam_center = None

    model_drift_scan_primary_beam = True
    if model_drift_scan_primary_beam:
        use_nvis_nt_nchan_ordering = False
        use_nvis_nchan_nt_ordering = True
    else:
        use_nvis_nt_nchan_ordering = True
        use_nvis_nchan_nt_ordering = False

###
# Intrinsic noise fitting params
###
use_intrinsic_noise_fitting = False

###
# Simulated signals in analysis
###
use_EoR_cube = True
use_GDSE_foreground_cube = False
use_freefree_foreground_cube = False
use_EGS_cube = False

###
# Prior on long wavelength modes
###
use_LWM_Gaussian_prior = False

###
# Fit for global signal jointly with the power spectrum
# See e.g. http://adsabs.harvard.edu/abs/2015ApJ...809...18P
###
# fit_for_monopole = True

###
# Uniform prior(s)
###
n_uniform_prior_k_bins = 0

###
# Fit for the optimal the large spectral scale model parameters
###
fit_for_spectral_model_parameters = False
pl_min = 2.0
pl_max = 3.0
pl_grid_spacing = 0.1

###
# Use sparse matrices to reduce storage requirements
# when constructing the data model
###
use_sparse_matrices = True
