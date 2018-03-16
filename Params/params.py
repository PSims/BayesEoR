"""
Analysis settings
"""

###
# Define analysis parameters here rather than in driver and util files...!
###

###
#k-cube params
###
nf=48
neta=48
nu=9
nv=9
nx=9
ny=9

###
# EoR sim params
###
EoR_npz_path = '/users/psims/EoR/EoR_simulations/21cmFAST_512MPc_512pix_128pix/Fits/21cm_z10d2_mK.npz'
box_size_21cmFAST_pix = 128 #Must match EoR_npz_path parameters
box_size_21cmFAST_Mpc = 512 #Must match EoR_npz_path parameters

#Original box
# EoR_npz_path_sc = '/users/psims/EoR/EoR_simulations/21cmFAST_512MPc_512pix_128pix/Fits/21cm_z10d2_mK.npz'
# box_size_21cmFAST_pix_sc = 128 #Must match EoR_npz_path parameters
# box_size_21cmFAST_Mpc_sc = 512 #Must match EoR_npz_path parameters

#Small box 3 deg. 30 MHz
# EoR_npz_path_sc = '/users/psims/EoR/EoR_simulations/21cmFAST_512MPc_128pix_32pix/Fits/21cm_mK_z7.600_nf0.423_useTs0.0_aveTb9.94_cube_side_pix32_cube_side_Mpc512.npz'
# box_size_21cmFAST_pix_sc = 32 #Must match EoR_npz_path parameters
# box_size_21cmFAST_Mpc_sc = 512 #Must match EoR_npz_path parameters

# Big box 12 deg. 120 MHz lower res. (so nf=12 ~10 MHz)
# EoR_npz_path_sc = '/users/psims/EoR/EoR_simulations/21cmFAST_2048MPc_1024pix_128pix/Fits/21cm_mK_z7.600_nf0.420_useTs0.0_aveTb9.75_cube_side_pix128_cube_side_Mpc2048.npz'
# box_size_21cmFAST_pix_sc = 128 #Must match EoR_npz_path parameters
# box_size_21cmFAST_Mpc_sc = 2048 #Must match EoR_npz_path parameters

# Big box 12 deg. 120 MHz higher res. (so nf=24 ~10 MHz)
# EoR_npz_path_sc = '/users/psims/EoR/EoR_simulations/21cmFAST_2048MPc_1024pix_256pix/Fits/21cm_mK_z7.600_nf0.442_useTs0.0_aveTb9.47_cube_side_pix256_cube_side_Mpc2048.npz'
# box_size_21cmFAST_pix_sc = 256 #Must match EoR_npz_path parameters
# box_size_21cmFAST_Mpc_sc = 2048 #Must match EoR_npz_path parameters

# Big box 12 deg. 120 MHz higher res. (so nf=24 ~10 MHz). Downsampled from a higher res. (3072 pix) box than the one above (1024 pix) but should be identical. Test this! It's not...............
# EoR_npz_path_sc = '/users/psims/EoR/EoR_simulations/21cmFAST_2048MPc_3072pix_256pix_v2/Fits/21cm_mK_z7.600_nf0.431_useTs0.0_aveTb9.65_cube_side_pix256_cube_side_Mpc2048.npz'
# box_size_21cmFAST_pix_sc = 256 #Must match EoR_npz_path parameters
# box_size_21cmFAST_Mpc_sc = 2048 #Must match EoR_npz_path parameters

# Big box 12 deg. 120 MHz higher res. (so nf=48 ~10 MHz). Downsampled from a high res. (3072 pix).
# EoR_npz_path_sc = '/users/psims/EoR/EoR_simulations/21cmFAST_2048MPc_3072pix_512pix_v2/Fits/21cm_mK_z7.600_nf0.459_useTs0.0_aveTb9.48_cube_side_pix512_cube_side_Mpc2048.npz'
# box_size_21cmFAST_pix_sc = 512 #Must match EoR_npz_path parameters
# box_size_21cmFAST_Mpc_sc = 2048 #Must match EoR_npz_path parameters

# Big box 12 deg. 120 MHz higher res. (so nf=48 ~10 MHz). Downsampled from a high res. (3072 pix). Redshift 10.8 so more Gaussian signal.
# EoR_npz_path_sc = '/users/psims/EoR/EoR_simulations/21cmFAST_2048MPc_3072pix_512pix_v2/Fits/21cm_mK_z10.800_nf0.843_useTs0.0_aveTb23.83_cube_side_pix512_cube_side_Mpc2048.npz'
# box_size_21cmFAST_pix_sc = 512 #Must match EoR_npz_path parameters
# box_size_21cmFAST_Mpc_sc = 2048 #Must match EoR_npz_path parameters

# Big box 12 deg. 120 MHz higher res. (so nf=48 ~10 MHz). Downsampled from a high res. (3072 pix). Redshift 10.6 so also fairly Gaussian signal.
# EoR_npz_path_sc = '/users/psims/EoR/EoR_simulations/21cmFAST_2048MPc_3072pix_512pix_v2/Fits/21cm_mK_z10.600_nf0.829_useTs0.0_aveTb23.15_cube_side_pix512_cube_side_Mpc2048.npz'
# box_size_21cmFAST_pix_sc = 512 #Must match EoR_npz_path parameters
# box_size_21cmFAST_Mpc_sc = 2048 #Must match EoR_npz_path parameters

# Big box 12 deg. 120 MHz higher res. (so nf=48 ~10 MHz). Downsampled from a high res. (3072 pix). Redshift 11.0 so also fairly Gaussian signal.
# EoR_npz_path_sc = '/users/psims/EoR/EoR_simulations/21cmFAST_2048MPc_3072pix_512pix_v2/Fits/21cm_mK_z11.000_nf0.856_useTs0.0_aveTb24.49_cube_side_pix512_cube_side_Mpc2048.npz'
# box_size_21cmFAST_pix_sc = 512 #Must match EoR_npz_path parameters
# box_size_21cmFAST_Mpc_sc = 2048 #Must match EoR_npz_path parameters

# Big box 12 deg. 120 MHz higher res. (so nf=48 ~10 MHz). Downsampled from a high res. (3072 pix). Redshift 10.0 so also fairly Gaussian signal.
EoR_npz_path_sc = '/users/psims/EoR/EoR_simulations/21cmFAST_2048MPc_3072pix_512pix_v2/Fits/21cm_mK_z10.000_nf0.782_useTs0.0_aveTb20.93_cube_side_pix512_cube_side_Mpc2048.npz'
box_size_21cmFAST_pix_sc = 512 #Must match EoR_npz_path parameters
box_size_21cmFAST_Mpc_sc = 2048 #Must match EoR_npz_path parameters




###
# Posterior params
###
EoR_analysis_cube_x_pix = box_size_21cmFAST_pix_sc #pix Analysing the full FoV in x
EoR_analysis_cube_y_pix = box_size_21cmFAST_pix_sc #pix Analysing the full FoV in y
EoR_analysis_cube_x_Mpc = box_size_21cmFAST_Mpc_sc #Mpc Analysing the full FoV in x
EoR_analysis_cube_y_Mpc = box_size_21cmFAST_Mpc_sc #Mpc Analysing the full FoV in y


###
# GDSE foreground params
###
beta_experimental_mean = 2.63+0   #Matches beta_150_408 in Mozden, Bowman et al. 2016
beta_experimental_std  = 0.02      #A conservative over-estimate of the dbeta_150_408=0.01 (dbeta_90_190=0.02) in Mozden, Bowman et al. 2016
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
HF_nu_min_MHz_array = [210]


###
# GDSE foreground params
###
beta_experimental_mean_ff = 2.15+0   #Matches beta_150_408 in Mozden, Bowman et al. 2016
beta_experimental_std_ff  = 1.e-10   #A conservative over-estimate of the dbeta_150_408=0.01 (dbeta_90_190=0.02) in Mozden, Bowman et al. 2016
gamma_mean_ff             = -2.59    #Revise to match published values
gamma_sigma_ff            = 0.04     #Revise to match published values
Tb_experimental_mean_K_ff = Tb_experimental_mean_K/100.0    #Matches GSM mean in region A
Tb_experimental_std_K_ff  = Tb_experimental_std_K/100.0     #70th percentile 12 deg.**2 region at 56 arcmin res. centered on -30. deg declination (see GSM_map_std_at_-30_dec_v1d0.ipynb)
print 'Hi!', Tb_experimental_std_K, Tb_experimental_std_K_ff
print 
# Tb_experimental_std_K  = 62.0   #Median std at at 0.333 degree resolution in 50 deg by 50 deg maps centered on Dec=-30.0
nu_min_MHz_ff             = 163.0-4.0
Tb_experimental_std_K_ff = Tb_experimental_std_K_ff*(nu_min_MHz_ff/163.)**-beta_experimental_mean_ff
channel_width_MHz_ff      = 0.2
simulation_FoV_deg_ff = 12.0             #Matches EoR simulation
simulation_resolution_deg_ff = simulation_FoV_deg_ff/511. #Matches EoR sim (note: use closest odd val., so 127 rather than 128, for easier FFT normalisation)
fits_storage_dir_ff = 'fits_storage/free_free_emission/Free_free_nu_min_MHz_{}_TbStd_{}_beta_{}_dbeta{}/'.format(nu_min_MHz_ff, Tb_experimental_std_K_ff, beta_experimental_mean_ff, beta_experimental_std_ff).replace('.','d')
# HF_nu_min_MHz_array = [210,220,230]
HF_nu_min_MHz_array_ff = [210]




###
# Other parameter types
# fg params
# spectral params
# uv params 
# ...
# etc.
###

