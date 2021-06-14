""" Analysis settings """
import argparse
import numpy as np
from astropy import constants

# Useful constants
speed_of_light = constants.c.value


""" Compute Params """
# Accelerate likelihood on GPU
useGPU = True

# Use sparse matrices to reduce storage requirements
# when constructing the data model
use_sparse_matrices = True

# MultiNest params
# If None, MultiNest will start a new analysis.
# Otherwise, resume analysis from ``file_root``.
file_root = None


""" Model Params """
# k-cube params
nf = 38
neta = 38
nu = 9
nv = None
nq = 0
# Subharmonic grid (SHG) params
nu_sh = 0
nv_sh = None
nq_sh = 0
npl_sh = 0

# Sky model params
fov_ra_deg = 12.9080728652
fov_dec_deg = None
nside = 256

# Simulated signals in analysis
use_EoR_cube = True
# EoR sim params
eor_sim_path = ''

# Frequency params
nu_min_MHz = 158.304048743
channel_width_MHz = 0.237618986858

# Spectral model params
beta = [2.63, 2.82]
if beta:
    if type(beta) == list:
        npl = len(beta)
    else:
        npl = 1
else:
    npl = 0
# Fit for the optimal the large spectral scale model parameters
fit_for_spectral_model_parameters = False
pl_min = 2.0
pl_max = 3.0
pl_grid_spacing = 0.1

# Instrumental effects params
include_instrumental_effects = True
# Include minimal prior over LW modes to
# ensure numerically stable posterior
inverse_LW_power = 1.e-16

if include_instrumental_effects:
    # Obs params
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

    # Primary beam params
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

# Sets the number of k-bins which use a prior which is uniform
# in the amplitude.  Indexed from the zeroth k-bin.
# Setting to -1 will use uniform priors on all k-bins.
n_uniform_prior_k_bins = 0

# Intrinsic noise fitting params
use_intrinsic_noise_fitting = False

# Prior on long wavelength modes
use_LWM_Gaussian_prior = False
