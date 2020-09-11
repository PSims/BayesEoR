import argparse
import numpy as np

"""
Analysis settings
"""

###
# Define analysis parameters here rather than in driver and util files...!
###

# --------------------------------------------
# User editable parameters
# --------------------------------------------

###
# k-cube params
###
nf = 38
neta = 38
nu = 9  # 9 for 12.9 deg FoV, 7 for 9.68 deg FoV, 5 for 6.45 deg FoV
nv = 9
nx = 9
ny = 9
nq = 0

###
# Data noise estimate
###
# sigma=50.e-1*250.0 #Noise level in S19b
# sigma=50.e-1*1000.0


###
# EoR sim params
###
EoR_npz_path = ''
# The following must match EoR_npz_path parameters
box_size_21cmFAST_pix = 128
box_size_21cmFAST_Mpc = 512

EoR_npz_path_sc = '/users/jburba/data/shared/PSims/BayesEoR_files_P/EoRsims/' \
                  'Hoag19/21cm_mK_z7.600_nf0.883_useTs0.0_aveTb21.06_cube_' \
                  'side_pix512_cube_side_Mpc2048.npz'
# Must match EoR_npz_path_sc parameters
box_size_21cmFAST_pix_sc = 512
box_size_21cmFAST_Mpc_sc = 2048
# Cosmological angular distance corresponding to a field of view of 12.0
# degrees and bandwidth of 7.6 MHz centered on 162.7 MHz
# box_size_21cmFAST_Mpc_sc = 1903.92479627


###
# Frequency params
###
# Corresponds to the lower edge of a ~9 MHz bandwidth centered on
# 162.7 MHz (z~7) for a 21cmFAST cube LOS size of 2048 * 38 / 512 Mpc
nu_min_MHz = 158.304048743
# Corresponds to the channel width for a LOS distance of 2048 * 38 / 512
# Mpc at redshift z~7 with nf=38 frequency channels
channel_width_MHz = 0.237618986858

###
# FoV parameters
###
# Corresponds to the angular extent of a 2048 Mpc patch of sky at
# redshift z~7 (band centerd on 162.7 MHz)
simulation_FoV_deg = 12.9080728652

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
simulation_resolution_deg = simulation_FoV_deg / 511.
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
from astropy import constants

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
    nt = 1
    integration_time_minutes = 0.5
    integration_time_minutes_str = '{}'.format(
        integration_time_minutes).replace('.', 'd')
    instrument_model_directory = (
        '/users/jburba/data/jburba/bayes/BayesEoR/Instrument_Model/'
        'HERA_469-2.4m_healvis_model_for_{}_{}_min_time_steps'
        '_bl_less_than_29.3m/'.format(nt, integration_time_minutes_str))

    ###
    # Primary beam params
    ###
    FWHM_deg_at_ref_freq_MHz = 2.0  # degrees
    PB_ref_freq_MHz = 150.0  # MHz
    # beam_type = 'Uniform'
    beam_type = 'Gaussian'
    beam_peak_amplitude = 1.0

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
fit_for_monopole = False

###
# Normalisation params
###
# Transverse size of the analysis cube in pixels
EoR_analysis_cube_x_pix = box_size_21cmFAST_pix_sc
EoR_analysis_cube_y_pix = box_size_21cmFAST_pix_sc
# Transverse size of the analysis cube in Mpc
EoR_analysis_cube_x_Mpc = box_size_21cmFAST_Mpc_sc
EoR_analysis_cube_y_Mpc = box_size_21cmFAST_Mpc_sc

###
# k_z uniform prior
###
# If True, don't use the min_kz voxels (eta \propto 1/B),
# which have significant correlation with the Fg model,
# in estimates of the low-k power spectrum
use_uniform_prior_on_min_k_bin = False

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

###
# Other parameter types
# fg params
# spectral params
# uv params
# ...
# etc.
###
