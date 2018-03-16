
###
# Imports
###
import numpy as np
import numpy
from numpy import arange, shape, log10, pi
import scipy
import pylab
from scipy.linalg import block_diag
from subprocess import os
import sys
from scipy import stats
from pdb import set_trace as brk

from SimData_v1d4 import GenerateForegroundCube, update_Tb_experimental_std_K_to_correct_for_normalisation_resolution
from Utils_v1d0 import PriorC, ParseCommandLineArguments, DataUnitConversionmkandJyperpix, WriteDataToFits
from Utils_v1d0 import ExtractDataFrom21cmFASTCube



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
	# fits_storage_dir = 'fits_storage/multi_frequency_band_pythonPStest1/Jelic_nu_min_MHz_{}_TbStd_{}_beta_{}_dbeta{}/'.format(nu_min_MHz, Tb_experimental_std_K, beta_experimental_mean, beta_experimental_std).replace('.','d')
	fits_storage_dir = 'fits_storage/free_free_emission/Jelic_nu_min_MHz_{}_TbStd_{}_beta_{}_dbeta{}/'.format(nu_min_MHz, Tb_experimental_std_K, beta_experimental_mean, beta_experimental_std).replace('.','d')
	# HF_nu_min_MHz_array = [210,220,230]
	HF_nu_min_MHz_array = [210]
	foreground_outputs = generate_Jelic_cube(nu,nv,nx,ny,nf,neta,nq,k_x, k_y, k_z,Show, beta_experimental_mean,beta_experimental_std,gamma_mean,gamma_sigma,Tb_experimental_mean_K,Tb_experimental_std_K,nu_min_MHz,channel_width_MHz, generate_additional_extrapolated_HF_foreground_cube=True, fits_storage_dir=fits_storage_dir, HF_nu_min_MHz_array=HF_nu_min_MHz_array, simulation_FoV_deg=simulation_FoV_deg, simulation_resolution_deg=simulation_resolution_deg,random_seed=314211)
	fg, s, Tb_nu, beta_experimental_mean,beta_experimental_std,gamma_mean,gamma_sigma,Tb_experimental_mean_K,Tb_experimental_std_K,nu_min_MHz,channel_width_MHz, HF_Tb_nu = foreground_outputs
	foreground_outputs = []



