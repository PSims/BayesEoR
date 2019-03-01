import argparse
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
# nf=76
# neta=76
# nu=17
# nv=17
# nx=17
# ny=17
# nu=15
# nv=15
# nx=15
# ny=15
# nu=13
# nv=13
# nx=13
# ny=13
nu=9
nv=9
nx=9
ny=9
# nu=7
# nv=7
# nx=7
# ny=7
# nu=11
# nv=11
# nx=11
# ny=11
# nu=31
# nv=31
# nx=31
# ny=31
# nu=21
# nv=21
# nx=21
# ny=21

use_uniform_prior_on_min_k_bin = False
# use_uniform_prior_on_min_k_bin = True #Don't use the min_kz voxels (eta \propto 1/B), which have significant correlation with the Fg model, in estimates of the low-k power spectrum 


###
# EoR sim params
###
EoR_npz_path = '/users/psims/EoR/EoR_simulations/21cmFAST_512MPc_512pix_128pix/Fits/21cm_z10d2_mK.npz'
box_size_21cmFAST_pix = 128 #Must match EoR_npz_path parameters
box_size_21cmFAST_Mpc = 512 #Must match EoR_npz_path parameters

# Big box 12 deg. 120 MHz higher res. (so nf=48 ~10 MHz). Downsampled from a high res. (3072 pix).
# EoR_npz_path_sc = '/users/psims/EoR/EoR_simulations/21cmFAST_2048MPc_3072pix_512pix_v2/Fits/21cm_mK_z7.600_nf0.459_useTs0.0_aveTb9.48_cube_side_pix512_cube_side_Mpc2048.npz'
EoR_npz_path_sc = '/users/psims/EoR/EoR_simulations/21cmFAST_2048MPc_2048pix_512pix_AstroParamExploration1/Fits/npzs/Zeta10.0_Tvir1.0e+05_mfp22.2_Taue0.041_zre-1.000_delz-1.000_512_2048Mpc/21cm_mK_z7.600_nf0.883_useTs0.0_aveTb21.06_cube_side_pix512_cube_side_Mpc2048.npz'
box_size_21cmFAST_pix_sc = 512 #Must match EoR_npz_path parameters
box_size_21cmFAST_Mpc_sc = 2048 #Must match EoR_npz_path parameters

###
# Normalisation params
###
EoR_analysis_cube_x_pix = box_size_21cmFAST_pix_sc #pix Analysing the full FoV in x
EoR_analysis_cube_y_pix = box_size_21cmFAST_pix_sc #pix Analysing the full FoV in y
EoR_analysis_cube_x_Mpc = box_size_21cmFAST_Mpc_sc #Mpc Analysing the full FoV in x
EoR_analysis_cube_y_Mpc = box_size_21cmFAST_Mpc_sc #Mpc Analysing the full FoV in y

#--------------------------------------------
# Parameters below this shouldn't require editing
#--------------------------------------------

###
# GDSE foreground params
###
beta_experimental_mean = 2.63+0   #Matches beta_150_408 in Mozden, Bowman et al. 2016
# beta_experimental_mean = 2.53+0   #Matches beta_150_408 in Mozden, Bowman et al. 2016
# beta_experimental_mean = 2.60+0   #Matches beta_150_408 in Mozden, Bowman et al. 2016
beta_experimental_std  = 0.02      #A conservative over-estimate of the dbeta_150_408=0.01 (dbeta_90_190=0.02) in Mozden, Bowman et al. 2016
# beta_experimental_std  = 1.e-10      #A conservative over-estimate of the dbeta_150_408=0.01 (dbeta_90_190=0.02) in Mozden, Bowman et al. 2016
gamma_mean             = -2.7     #Revise to match published values
gamma_sigma            = 0.3      #Revise to match published values
Tb_experimental_mean_K = 194.0    #Matches GSM mean in region A
Tb_experimental_std_K  = 62.0     #70th percentile 12 deg.**2 region at 56 arcmin res. centered on -30. deg declination (see GSM_map_std_at_-30_dec_v1d0.ipynb)
# Tb_experimental_std_K  = 62.0   #Median std at at 0.333 degree resolution in 50 deg by 50 deg maps centered on Dec=-30.0
nu_min_MHz             = 163.0-4.0
Tb_experimental_std_K = Tb_experimental_std_K*(nu_min_MHz/163.)**-beta_experimental_mean
channel_width_MHz      = 0.2
simulation_FoV_deg = 12.0             #Matches EoR simulation
simulation_resolution_deg = simulation_FoV_deg/511. #Matches EoR sim (note: use closest odd val., so 127 rather than 128, for easier FFT normalisation)
fits_storage_dir = 'fits_storage/multi_frequency_band_pythonPStest1/Jelic_nu_min_MHz_{}_TbStd_{}_beta_{}_dbeta{}/'.format(nu_min_MHz, Tb_experimental_std_K, beta_experimental_mean, beta_experimental_std).replace('.','d')
# HF_nu_min_MHz_array = [210,220,230]
HF_nu_min_MHz_array = [220]

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
EGS_npz_path = '/users/psims/Cav/EoR/Missing_Radio_Flux/Surveys/Flux_Variance_Maps/S_Cubed/S_163_10nJy_Image_Cube_v34_18_deg_NV_15JyCN_With_Synchrotron_Self_Absorption/Fits/Flux_Density_Upper_Lim_15.0__Flux_Density_Lower_Lim_0.0/mk_cube/151_Flux_values_10NanoJansky_limit_data_result_18_Degree_Cube_RA_Dec_Degrees_and__10_pow_LogFlux_Columns_and_Source_Redshifts_and_Source_SI_and_Source_AGN_Type_Comb__mk.npz'

###
# Spectral model params
###
nu_min_MHz = nu_min_MHz #Match spectral range of simulated signals
channel_width_MHz = channel_width_MHz #Match spectral range of simulated signals
# beta = 2.63
# beta = -2.0
# beta = [2.63, 2.82, 2.0]
beta = [2.63, 2.82]
# beta = [-1.0, -2.0]
# beta = [2.4, 3.0]
# beta = False

if beta:
	if type(beta)==list:
		npl = len(beta)
	else:
		npl=1
else:
	npl=0


def BayesEoRParser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-nq", "--nq", help="Number of LWM basis vectors (0-2)", default=2)
	parser.add_argument("-beta", "--beta", help="Power law spectral index used in data model", default=beta)
	args = parser.parse_args() #Parse command line arguments
	return args

###
# Accelerate likelihood on GPU
###
useGPU = True #Use GPU if available


###
# Useful constants
###
from astropy import constants
speed_of_light = constants.c.value


###--
###
# Instrumental effects params
###

include_instrumental_effects = True
# include_instrumental_effects = False
inverse_LW_power = 1.e-8 #Include minimal prior over LW modes to ensure numerically stable posterior


###
# NUDFT params
###

# Load uvw_multi_time_step_array_meters_reshaped inside a function to avoid creating extraneous params variables (like files_dir, file_name etc.)
import pickle
def load_uvw_instrument_sampling_m(instrument_model_directory):
	file_dir = instrument_model_directory
	file_name = "uvw_multi_time_step_array_meters_reshaped" #HERA 331 sub-100 m baselines (i.e. H37 baselines) uv-sampling in meters
	f = open(file_dir+file_name,'r')  
	uvw_multi_time_step_array_meters_reshaped =  pickle.load(f)
	return uvw_multi_time_step_array_meters_reshaped

def load_baseline_redundancy_array(instrument_model_directory):
	file_dir = instrument_model_directory
	file_name = "unique_H37_baseline_hermitian_redundancy_multi_time_step_array_reshaped" #HERA 331 sub-100 m baselines (i.e. H37 baselines) baseline redundancy
	f = open(file_dir+file_name,'r')  
	unique_H37_baseline_hermitian_redundancy_multi_time_step_array_reshaped =  pickle.load(f)
	return unique_H37_baseline_hermitian_redundancy_multi_time_step_array_reshaped




if include_instrumental_effects:
	# instrument_model_directory = '/users/psims/EoR/Python_Scripts/Calculate_HERA_UV_Coords/output_products/HERA_331_baselines_shorter_than_14d7_for_30_0d5_min_time_steps/'
	# instrument_model_directory = '/users/psims/EoR/Python_Scripts/Calculate_HERA_UV_Coords/output_products/HERA_331_baselines_shorter_than_14d7_for_60_0d5_min_time_steps/'
	# instrument_model_directory = '/users/psims/EoR/Python_Scripts/Calculate_HERA_UV_Coords/output_products/HERA_331_baselines_shorter_than_14d7_for_120_0d5_min_time_steps/'
	instrument_model_directory = '/users/psims/EoR/Python_Scripts/Calculate_HERA_UV_Coords/output_products/HERA_331_baselines_shorter_than_29d3_for_30_0d5_min_time_steps/'
	uvw_multi_time_step_array_meters_reshaped = load_uvw_instrument_sampling_m(instrument_model_directory)
	baseline_redundancy_array = load_baseline_redundancy_array(instrument_model_directory)
	uv_pixel_width_wavelengths = 2.5 #Define a fixed pixel width in wavelengths
	n_vis = len(uvw_multi_time_step_array_meters_reshaped) #Number of visibilities per channel (i.e. number of redundant baselines * number of time steps)


###
# Primary beam params
###

if include_instrumental_effects:
	FWHM_deg_at_ref_freq_MHz = 9.0 #9 degrees
	PB_ref_freq_MHz = 150.0 #150 MHz
	# beam_type = 'Uniform'
	beam_type = 'Gaussian'
	beam_peak_amplitude = 1.0
	beam_info_str = ''
	if beam_type.lower() == 'Uniform'.lower():
		beam_info_str += '{}_beam_peak_amplitude_{}'.format(beam_type, str(beam_peak_amplitude).replace('.','d'))		
	if beam_type.lower() == 'Gaussian'.lower():
		beam_info_str += '{}_beam_peak_amplitude_{}_beam_width_{}_deg_at_{}_MHz'.format(beam_type, str(beam_peak_amplitude).replace('.','d'), str(FWHM_deg_at_ref_freq_MHz).replace('.','d'), str(PB_ref_freq_MHz).replace('.','d'))		

	# instrument_model_directory = '/users/psims/EoR/Python_Scripts/Calculate_HERA_UV_Coords/output_products/HERA_331_baselines_shorter_than_14d7_for_30_0d5_min_time_steps/'
	# instrument_model_directory = '/users/psims/EoR/Python_Scripts/Calculate_HERA_UV_Coords/output_products/HERA_331_baselines_shorter_than_29d3_for_30_0d5_min_time_steps/'
	instrument_model_directory = instrument_model_directory[:-1]+'_{}/'.format(beam_info_str)

###--




###
# Other parameter types
# fg params
# spectral params
# uv params 
# ...
# etc.
###

