###
# Imports
###
import matplotlib
matplotlib.use('pdf') #No pop-ups (comment out to interact with plots)
import numpy as np
import numpy
from numpy import arange, shape, log10, pi
import scipy
from subprocess import os
import sys
import pylab
import time
from scipy.linalg import block_diag
from pprint import pprint
from pdb import set_trace as brk
import time

from Linalg import IDFT_Array_IDFT_2D_ZM_SH, makeGaussian, Produce_Full_Coordinate_Arrays, Produce_Coordinate_Arrays_ZM
from Linalg import Produce_Coordinate_Arrays_ZM_Coarse, Produce_Coordinate_Arrays_ZM_SH, Calc_Coords_High_Res_Im_to_Large_uv
from Linalg import Calc_Coords_Large_Im_to_High_Res_uv, Restore_Centre_Pixel, Calc_Indices_Centre_3x3_Grid
from Linalg import Delete_Centre_3x3_Grid, Delete_Centre_Pix, N_is_Odd, Calc_Indices_Centre_NxN_Grid, Obtain_Centre_NxN_Grid
from Linalg import Restore_Centre_3x3_Grid, Restore_Centre_NxN_Grid, Generate_Combined_Coarse_plus_Subharmic_uv_grids
from Linalg import IDFT_Array_IDFT_2D_ZM_SH, DFT_Array_DFT_2D, IDFT_Array_IDFT_2D, DFT_Array_DFT_2D_ZM, IDFT_Array_IDFT_2D_ZM
from Linalg import Construct_Hermitian, Construct_Hermitian_Gridding_Matrix, Construct_Hermitian_Gridding_Matrix_CosSin
from Linalg import Construct_Hermitian_Gridding_Matrix_CosSin_SH_v4, IDFT_Array_IDFT_1D, IDFT_Array_IDFT_1D
from Linalg import generate_gridding_matrix_vis_ordered_to_chan_ordered
from Linalg import IDFT_Array_IDFT_1D_WQ, generate_gridding_matrix_vis_ordered_to_chan_ordered_WQ
from Linalg import IDFT_Array_IDFT_1D_WQ_ZM, generate_gridding_matrix_vis_ordered_to_chan_ordered_ZM
from Linalg import nuDFT_Array_DFT_2D

from SimData import generate_test_sim_signal, map_out_bins_for_power_spectral_coefficients
# from SimData import generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector
from SimData import generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_v2
from SimData import map_out_bins_for_power_spectral_coefficients_WQ_v2, generate_k_cube_in_physical_coordinates
from SimData import map_out_bins_for_power_spectral_coefficients_HERA_Binning
from SimData import map_out_bins_for_power_spectral_coefficients_WQ_v2_HERA_Binning
from SimData import generate_test_sim_signal_with_large_spectral_scales_2_HERA_Binning
from SimData import generate_k_cube_in_physical_coordinates_21cmFAST, generate_k_cube_in_physical_coordinates_21cmFAST_v2d0
from SimData import generate_test_sim_signal_with_large_spectral_scales_2_21cmFAST_Binning
from SimData import GenerateForegroundCube
from SimData import generate_masked_coordinate_cubes, generate_k_cube_model_cylindrical_binning
from SimData import generate_k_cube_model_spherical_binning, construct_GRN_unitary_hermitian_k_cube
from SimData import calc_mean_binned_k_vals
from SimData import generate_visibility_covariance_matrix_and_noise_realisation_and_the_data_vector_instrumental_v1

from likelihood_tests.SimpleEoRtestWQ.Generate_matrix_stack_v1d1 import BuildMatrices
from likelihood_tests.SimpleEoRtestWQ.Likelihood_v1d763_3D_ZM_standalone_GPU import PowerSpectrumPosteriorProbability

from Utils import PriorC, ParseCommandLineArguments, DataUnitConversionmkandJyperpix, WriteDataToFits
from Utils import ExtractDataFrom21cmFASTCube, plot_signal_vs_MLsignal_residuals
from Utils import generate_output_file_base, RenormaliseMatricesForScaledNoise

from Utils import remove_unused_header_variables, construct_aplpy_image_from_fits

from GenerateForegroundCube import generate_Jelic_cube, generate_data_from_loaded_EoR_cube, generate_data_from_loaded_EoR_cube_v2d0
from GenerateForegroundCube import generate_test_signal_from_image_cube
from GenerateForegroundCube import top_hat_average_temperature_cube_to_lower_res_31x31xnf_cube
from GenerateForegroundCube import generate_data_from_loaded_EGS_cube, generate_white_noise_signal





