###
# Imports
###
# import matplotlib
# matplotlib.use('pdf') #No pop-ups (comment out to interact with plots)
import numpy as np
from numpy import arange, shape, log10, pi, real
import scipy
from subprocess import os
import sys
# import pylab
import time
from scipy.linalg import block_diag
from pprint import pprint
from pdb import set_trace as brk

from .Linalg import\
    IDFT_Array_IDFT_2D_ZM_SH, makeGaussian,\
    Produce_Full_Coordinate_Arrays, Produce_Coordinate_Arrays_ZM,\
    Produce_Coordinate_Arrays_ZM_Coarse, Produce_Coordinate_Arrays_ZM_SH,\
    Calc_Coords_High_Res_Im_to_Large_uv, Calc_Coords_Large_Im_to_High_Res_uv,\
    Restore_Centre_Pixel, Calc_Indices_Centre_3x3_Grid,\
    Delete_Centre_3x3_Grid, Delete_Centre_Pix, N_is_Odd,\
    Calc_Indices_Centre_NxN_Grid, Obtain_Centre_NxN_Grid,\
    Restore_Centre_3x3_Grid, Restore_Centre_NxN_Grid,\
    Generate_Combined_Coarse_plus_Subharmic_uv_grids,\
    IDFT_Array_IDFT_2D_ZM_SH, DFT_Array_DFT_2D, IDFT_Array_IDFT_2D,\
    DFT_Array_DFT_2D_ZM, IDFT_Array_IDFT_2D_ZM, Construct_Hermitian,\
    Construct_Hermitian_Gridding_Matrix,\
    Construct_Hermitian_Gridding_Matrix_CosSin,\
    Construct_Hermitian_Gridding_Matrix_CosSin_SH_v4,\
    generate_gridding_matrix_vis_ordered_to_chan_ordered,\
    IDFT_Array_IDFT_1D, IDFT_Array_IDFT_1D_WQ, IDFT_Array_IDFT_1D_WQ_ZM,\
    generate_gridding_matrix_vis_ordered_to_chan_ordered_WQ,\
    generate_gridding_matrix_vis_ordered_to_chan_ordered_ZM,\
    nuDFT_Array_DFT_2D, make_Gaussian_beam, make_Uniform_beam

from .SimData import\
    generate_test_sim_signal,\
    map_out_bins_for_power_spectral_coefficients,\
    map_out_bins_for_power_spectral_coefficients_WQ_v2,\
    generate_k_cube_in_physical_coordinates,\
    map_out_bins_for_power_spectral_coefficients_HERA_Binning,\
    map_out_bins_for_power_spectral_coefficients_WQ_v2_HERA_Binning,\
    generate_test_sim_signal_with_large_spectral_scales_2_HERA_Binning,\
    generate_k_cube_in_physical_coordinates_21cmFAST,\
    generate_k_cube_in_physical_coordinates_21cmFAST_v2d0,\
    generate_test_sim_signal_with_large_spectral_scales_2_21cmFAST_Binning,\
    GenerateForegroundCube, generate_masked_coordinate_cubes,\
    generate_k_cube_model_cylindrical_binning,\
    generate_k_cube_model_spherical_binning,\
    construct_GRN_unitary_hermitian_k_cube, calc_mean_binned_k_vals,\
    generate_k_cube_model_spherical_binning_v2d0,\
    generate_k_cube_model_spherical_binning_v2d1,\
    generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2\
    as generate_data_and_noise_vector,\
    generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_instrumental_v1\
    as generate_data_and_noise_vector_instrumental

from .likelihood_tests.SimpleEoRtestWQ.Generate_matrix_stack_v1d2\
    import BuildMatrices
from .likelihood_tests.SimpleEoRtestWQ.Likelihood_v1d763_3D_ZM_standalone_GPU_v2d0\
    import PowerSpectrumPosteriorProbability

from .Utils import\
    PriorC, DataUnitConversionmKAndJyPerPix,\
    WriteDataToFits, ExtractDataFrom21cmFASTCube,\
    generate_output_file_base, load_uvw_instrument_sampling_m,\
    load_baseline_redundancy_array, write_log_file

from .GenerateForegroundCube import\
    generate_Jelic_cube,\
    generate_data_from_loaded_EoR_cube,\
    generate_data_from_loaded_EoR_cube_v2d0,\
    generate_test_signal_from_image_cube,\
    top_hat_average_temperature_cube_to_lower_res_31x31xnf_cube,\
    generate_data_from_loaded_EGS_cube,\
    generate_white_noise_signal_instrumental_k_2_vis,\
    generate_white_noise_signal_instrumental_im_2_vis,\
    generate_EoR_signal_instrumental_im_2_vis,\
    generate_Jelic_cube_instrumental_im_2_vis,\
    calculate_subset_cube_power_spectrum_v1d0,\
    calculate_subset_cube_power_spectrum_v2d0,\
    generate_data_from_loaded_EGS_cube_im_2_vis,\
    calculate_21cmFAST_EoR_cube_power_spectrum_in_subset_cube_bins_v1d0,\
    generate_Jelic_cube_instrumental_im_2_vis_v2d0,\
    generate_data_from_loaded_EGS_cube_im_2_vis_v2d0

from .Params import update_params_with_command_line_arguments








