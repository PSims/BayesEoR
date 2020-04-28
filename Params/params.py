import argparse
import numpy as np
"""
Analysis settings
"""

###
# Define analysis parameters here rather than in driver and util files...!
###

#--------------------------------------------
# User editable parameters
#--------------------------------------------

###
#k-cube params
###
nf=38
neta=38
nu=9
nv=9
nx=9
ny=9
nq=0

###
# Data noise estimate
###
# sigma=50.e-1*250.0 #Noise level in S19b
# sigma=50.e-1*1000.0


###
# EoR sim params
###
EoR_npz_path = '/users/psims/EoR/EoR_simulations/21cmFAST_512MPc_512pix_128pix/Fits/21cm_z10d2_mK.npz'
box_size_21cmFAST_pix = 128 #Must match EoR_npz_path parameters
box_size_21cmFAST_Mpc = 512 #Must match EoR_npz_path parameters

# EoR_npz_path_sc = '/users/psims/EoR/EoR_simulations/21cmFAST_2048MPc_2048pix_512pix_AstroParamExploration1/Fits/npzs/Zeta10.0_Tvir1.0e+05_mfp22.2_Taue0.041_zre-1.000_delz-1.000_512_2048Mpc/21cm_mK_z7.600_nf0.883_useTs0.0_aveTb21.06_cube_side_pix512_cube_side_Mpc2048.npz'
EoR_npz_path_sc = '/users/jburba/data/shared/PSims/BayesEoR_files_P/EoRsims/Hoag19/21cm_mK_z7.600_nf0.883_useTs0.0_aveTb21.06_cube_side_pix512_cube_side_Mpc2048.npz'
box_size_21cmFAST_pix_sc = 512 #Must match EoR_npz_path parameters
box_size_21cmFAST_Mpc_sc = 2048 #Must match EoR_npz_path parameters
# box_size_21cmFAST_Mpc_sc = 1903.92479627 # Cosmological angular distance corresponding to a field of view of 12.0 degrees and bandwidth of 7.6 MHz centered on 162.7 MHz


#--------------------------------------------
# Parameters below this shouldn't require editing
#--------------------------------------------

###
# GDSE foreground params
###
beta_experimental_mean = 2.63+0   #Matches beta_150_408 in Mozden, Bowman et al. 2016
beta_experimental_std  = 0.02      #A conservative over-estimate of the dbeta_150_408=0.01 (dbeta_90_190=0.02) in Mozden, Bowman et al. 2016
gamma_mean             = -2.7     #Revise to match published values
gamma_sigma            = 0.3      #Revise to match published values
# Tb_experimental_mean_K = 194.0    #Matches GSM mean in region A
Tb_experimental_mean_K = 471.0    #Matches GSM mean in region considered in S19b (see GSM_map_std_at_-30_dec_v1d3.ipynb)
Tb_experimental_std_K  = 62.0     #70th percentile 12 deg.**2 region at 56 arcmin res. centered on -30. deg declination (see GSM_map_std_at_-30_dec_v1d0.ipynb)
# nu_min_MHz             = 163.0-4.0
nu_min_MHz = 158.304048743 # Corresponds to the lower edge of a ~9 MHz bandwidth centered on 162.7 MHz (z~7) for a 21cmFAST cube LOS size of 2048 * nf / 512 Mpc
Tb_experimental_std_K = Tb_experimental_std_K*(nu_min_MHz/163.)**-beta_experimental_mean
# channel_width_MHz      = 0.2
channel_width_MHz = 0.237618986858 # Corresponds to the channel width for a LOS distance of 2048 * nf / 512 Mpc at redshift z~7 with nf=38 frequency channels
# simulation_FoV_deg = 12.0             #Matches EoR simulation
# simulation_FoV_deg = 22.918311805232932 # chosen as uv_pixel_wavelengths = 2.5 and fov = np.rad2deg(1 / 2.5) ~ 22 degrees # 12.0 deg Matches EoR simulation
# simulation_FoV_deg = 30.0
simulation_FoV_deg = 12.9080728652 # Corresponds to the angular extent of a 2048 Mpc patch of sky at redshift z~7 (band centerd on 162.7 MHz)
simulation_resolution_deg = simulation_FoV_deg/511. #Matches EoR sim (note: use closest odd val., so 127 rather than 128, for easier FFT normalisation)
fits_storage_dir = 'fits_storage/multi_frequency_band_pythonPStest1/Jelic_nu_min_MHz_{}_TbStd_{}_beta_{}_dbeta{}/'.format(nu_min_MHz, Tb_experimental_std_K, beta_experimental_mean, beta_experimental_std).replace('.','d')
# HF_nu_min_MHz_array = [210,220,230]
HF_nu_min_MHz_array = [220]

sky_model_pixel_area_sr = np.deg2rad(simulation_FoV_deg / nx)**2 # pixel area in steradians of the sky model
# sigma = 50.e-1*1000.0 * sky_model_pixel_area_sr
# sigma = 0.001555210668254031 # 0.3 * np.std(hv_data in mK sr) - calculated for flatspec/uniform beam data
# sigma = 0.00018327540647584584 # for 1.5 deg FWHM beam data
# sigma = 0.00019604703375601735 # for 2 deg FWHM beam data
# sigma = 0.000019604703375601735 # for 2 deg FWHM beam data / 10
# sigma = 0.0003413612494755223 # 0.3 * np.std(hv_data with gaussian beam in mK sr)
# sigma = 0.0006385239458905794 # 0.9 sigma for 2 deg FWHM beam data
# sigma = 0.001064206576484299 # 1.5 sigma for 2 deg FWHM beam data
# sigma = 0.0014189421019790652 # 2 sigma for 2 deg FWHM beam data
# sigma = 0.0050441458536927535 # 0.9 sigma for uniform beam data
# sigma = 0.00840690975615459 # 1.5 sigma for uniform beam data
sigma = 0.011209213008206119 # 2.0 sigma for uniform beam data
# sigma = 0.014011516260257649 # 2.5 sigma for uniform beam data
# sigma = 0.01681381951230918 # 3.0 sigma for uniform beam data
# sigma = 0.0029141736440333335 # 0.5 sigma for internal model data
# sigma = 0.00524551255926 # 0.9 sigma for internal model data
# sigma = 0.0116566945761 # 2.0 sigma for internal model data
# sigma = 0.0145708682202 # 2.5 sigma for internal model data
# sigma = 0.0174850418642 # 3.0 sigma for internal model data

###
# diffuse free-free foreground params
###
beta_experimental_mean_ff = 2.15+0
beta_experimental_std_ff  = 1.e-10
gamma_mean_ff             = -2.59
gamma_sigma_ff            = 0.04
Tb_experimental_mean_K_ff = Tb_experimental_mean_K/100.0
Tb_experimental_std_K_ff  = Tb_experimental_std_K/100.0
print 'Hi!', Tb_experimental_std_K, Tb_experimental_std_K_ff
nu_min_MHz_ff             = 163.0-4.0
Tb_experimental_std_K_ff = Tb_experimental_std_K_ff*(nu_min_MHz_ff/163.)**-beta_experimental_mean_ff
channel_width_MHz_ff      = 0.2
simulation_FoV_deg_ff = 12.0             #Matches EoR simulation
simulation_resolution_deg_ff = simulation_FoV_deg_ff/511. #Matches EoR sim (note: use closest odd val., so 127 rather than 128, for easier FFT normalisation)
fits_storage_dir_ff = 'fits_storage/free_free_emission/Free_free_nu_min_MHz_{}_TbStd_{}_beta_{}_dbeta{}/'.format(nu_min_MHz_ff, Tb_experimental_std_K_ff, beta_experimental_mean_ff, beta_experimental_std_ff).replace('.','d')
# HF_nu_min_MHz_array = [210,220,230]
HF_nu_min_MHz_array_ff = [210]


###
# Extragalactic source foreground params
###
EGS_npz_path = '/users/psims/Cav/EoR/Missing_Radio_Flux/Surveys/Flux_Variance_Maps/S_Cubed/S_163_10nJy_Image_Cube_v34_18_deg_NV_1JyCN_With_Synchrotron_Self_Absorption/Fits/Flux_Density_Upper_Lim_1.0__Flux_Density_Lower_Lim_0.0/mk_cube/151_Flux_values_10NanoJansky_limit_data_result_18_Degree_Cube_RA_Dec_Degrees_and__10_pow_LogFlux_Columns_and_Source_Redshifts_and_Source_SI_and_Source_AGN_Type_Comb__mk.npz' #Low intensity EGS sim in S19b

# EGS_npz_path = '/users/psims/Cav/EoR/Missing_Radio_Flux/Surveys/Flux_Variance_Maps/S_Cubed/S_163_10nJy_Image_Cube_v34_18_deg_NV_40JyCN_With_Synchrotron_Self_Absorption/Fits/Flux_Density_Upper_Lim_40.0__Flux_Density_Lower_Lim_0.0/mk_cube/151_Flux_values_10NanoJansky_limit_data_result_18_Degree_Cube_RA_Dec_Degrees_and__10_pow_LogFlux_Columns_and_Source_Redshifts_and_Source_SI_and_Source_AGN_Type_Comb__mk.npz' #High intensity EGS sim in S19b


###
# Spectral model params
###
nu_min_MHz = nu_min_MHz #Match spectral range of simulated signals
channel_width_MHz = channel_width_MHz #Match spectral range of simulated signals
beta = [2.63, 2.82]
if beta:
	if type(beta)==list:
		npl = len(beta)
	else:
		npl=1
else:
	npl=0


###
# Accelerate likelihood on GPU
###
useGPU = True #Use GPU if available


###
# Useful constants
###
from astropy import constants
speed_of_light = constants.c.value


###
# Instrumental effects params
###
include_instrumental_effects = True
inverse_LW_power = 1.e-16 #Include minimal prior over LW modes to ensure numerically stable posterior *250
if include_instrumental_effects:
	###
	# Obs params
	###
	nt = 30
	integration_time_minutes = 0.5
	integration_time_minutes_str = '{}'.format(integration_time_minutes).replace('.','d')
	# instrument_model_directory = '/users/psims/EoR/Python_Scripts/BayesEoR/git_version/BayesEoR/Instrument_Model/HERA_331_baselines_shorter_than_29d3_for_{}_{}_min_time_steps/'.format(nt, integration_time_minutes_str)
	instrument_model_directory = '/users/jburba/data/jburba/bayes/BayesEoR/Instrument_Model/HERA_19_healvis_model_for_{}_{}_min_time_steps/'.format(nt, integration_time_minutes_str)
	# uv_pixel_width_wavelengths = 2.5 #Define a fixed pixel width in wavelengths
	uv_pixel_width_wavelengths = 1.0 / np.deg2rad(simulation_FoV_deg) # originally set to 2.5 for a 12.0 deg FoV
	###
	# Primary beam params
	###
	FWHM_deg_at_ref_freq_MHz = 2.0 # degrees
	PB_ref_freq_MHz = 150.0 #150 MHz
	beam_type = 'Uniform'
	# beam_type = 'Gaussian'
	beam_peak_amplitude = 1.0
	# beam_info_str = ''
	# if beam_type.lower() == 'Uniform'.lower():
	# 	beam_info_str += '{}_beam_peak_amplitude_{}'.format(beam_type, str(beam_peak_amplitude).replace('.','d'))
	# if beam_type.lower() == 'Gaussian'.lower():
	# 	beam_info_str += '{}_beam_peak_amplitude_{}_beam_width_{}_deg_at_{}_MHz'.format(beam_type, str(beam_peak_amplitude).replace('.','d'), str(FWHM_deg_at_ref_freq_MHz).replace('.','d'), str(PB_ref_freq_MHz).replace('.','d'))
	#
	# instrument_model_directory_plus_beam_info = instrument_model_directory[:-1]+'_{}/'.format(beam_info_str)
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
fit_for_monopole = False


###
# Normalisation params
###
EoR_analysis_cube_x_pix = box_size_21cmFAST_pix_sc #pix Analysing the full FoV in x
EoR_analysis_cube_y_pix = box_size_21cmFAST_pix_sc #pix Analysing the full FoV in y
EoR_analysis_cube_x_Mpc = box_size_21cmFAST_Mpc_sc #Mpc Analysing the full FoV in x
EoR_analysis_cube_y_Mpc = box_size_21cmFAST_Mpc_sc #Mpc Analysing the full FoV in y


###
# k_z uniform prior
###
use_uniform_prior_on_min_k_bin = False
# use_uniform_prior_on_min_k_bin = True #Don't use the min_kz voxels (eta \propto 1/B), which have significant correlation with the Fg model, in estimates of the low-k power spectrum


###
# Fit for the optimal the large spectral scale model parameters
###
fit_for_spectral_model_parameters = False
# fit_for_spectral_model_parameters = True
pl_min = 2.0
pl_max = 3.0
pl_grid_spacing = 0.1
# pl_max = 5.20
# pl_grid_spacing = 0.5
# pl_grid_spacing = 1.0


###
#Use sparse matrices to reduce storage requirements when constructing the data model
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
