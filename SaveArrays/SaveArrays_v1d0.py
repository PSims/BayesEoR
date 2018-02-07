import numpy as np
import numpy
from numpy import arange, shape
import scipy
from subprocess import os
import sys
sys.path.append(os.path.expanduser('~/EoR/PolyChord1d9/PolyChord_WorkingInitSetup_Altered/'))
sys.path.append(os.path.expanduser('~/EoR/Python_Scripts/BayesEoR/SpatialPS/PolySpatialPS/'))
import pylab
import h5py



def create_directory(Directory,**kwargs):
	"""
	Create output directory if it doesn't exist
	"""
	if not os.path.exists(Directory):
		print 'Directory not found: \n\n'+Directory+"\n"
		print 'Creating required directory structure..'
		os.makedirs(Directory)
	
	return 0


def output_to_hdf5(output_array, output_directory, file_name, dataset_name):
	"""
	# Write array to HDF5 file
	"""
	create_directory(output_directory)
	output_path = '/'.join((output_directory,file_name))
	print "Writing data to", output_path
	with h5py.File(output_path, 'w') as hf:
	    hf.create_dataset(dataset_name,  data=output_array)
	return 0


def read_data_from_hdf5(file_path, dataset_name):
	"""
	# Read array from HDF5 file
	"""
	with h5py.File(file_path, 'r') as hf:
	    data = hf[dataset_name][:]
	return data













