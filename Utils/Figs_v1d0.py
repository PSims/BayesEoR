import numpy as np
from numpy import pi,log,sqrt,array
from numpy import arange, shape, log10
import subprocess
from subprocess import os
import sys, getopt
from scipy import integrate
from astropy.io import fits
import shutil
import pylab
from pylab import cm
from pdb import set_trace as brk
from astropy.io import fits


## ======================================================================================================
## ======================================================================================================

def remove_unused_header_variables(Output_Fits_Directory, Fits_File_Name):
	hdulist = fits.open(Output_Fits_Directory+Fits_File_Name+'.fits')
	header_items_to_remove = ['CTYPE4', 'CRVAL4', 'CDELT4', 'CRPIX4', 'CUNIT4']
	for item in header_items_to_remove:
		try:
			hdulist[0].header.remove(item)
		except Exception as e:
			print e
			print 'header variable', item, 'not present...?'
	hdulist[0].header.update(CTYPE1='RA',CTYPE2='DEC' )
	new_fits_file_name = Fits_File_Name+'_header_update'
	hdulist.writeto(Output_Fits_Directory+new_fits_file_name+'.fits', overwrite=True)
	return new_fits_file_name


## ======================================================================================================
## ======================================================================================================

def convert_from_mK_to_K(Output_Fits_Directory, Fits_File_Name):
	hdulist = fits.open(Output_Fits_Directory+Fits_File_Name+'.fits')
	data_mK = hdulist[0].data.copy()
	data_K  = data_mK/1.e3
	hdulist[0].data = data_K
	new_fits_file_name = Fits_File_Name+'_convert_mK_to_K'
	hdulist.writeto(Output_Fits_Directory+new_fits_file_name+'.fits', overwrite=True)
	return new_fits_file_name


## ======================================================================================================
## ======================================================================================================

def construct_aplpy_image_from_fits(Output_Fits_Directory, Fits_File_Name, **kwargs):

	##===== Defaults =======
	default_run_convert_from_mK_to_K = True
	default_run_remove_unused_header_variables = True
	
	##===== Inputs =======
	run_convert_from_mK_to_K=kwargs.pop('run_convert_from_mK_to_K',default_run_convert_from_mK_to_K)
	run_remove_unused_header_variables=kwargs.pop('run_remove_unused_header_variables',default_run_remove_unused_header_variables)

	if run_convert_from_mK_to_K: Fits_File_Name = convert_from_mK_to_K(Output_Fits_Directory, Fits_File_Name)
	if run_remove_unused_header_variables: Fits_File_Name = remove_unused_header_variables(Output_Fits_Directory, Fits_File_Name)
	#
	from pyfits import getheader
	Template = Output_Fits_Directory+Fits_File_Name+'.fits'
	Template_hdr = getheader(Template, 0)
	for key in Template_hdr:print key, Template_hdr[key]
	#
	print 'hi1'
	import aplpy
	import pylab as plt
	from pylab import cm
	# EGSIm = aplpy.FITSFigure(Output_Fits_Directory+Fits_File_Name+'.fits', dimensions=[1,2], slices=[0], convention='wells')
	EGSIm = aplpy.FITSFigure(Output_Fits_Directory+Fits_File_Name+'.fits', dimensions=[0,1], slices=[0])
	EGSIm.show_colorscale(cmap=cm.jet)
	#EGSIm.show_colorscale(cmap=cm.gray)
	EGSIm.show_colorbar()
	EGSIm._wcs.xaxis_coord_type='longitude'
	EGSIm._wcs.yaxis_coord_type='latitude'
	EGSIm.set_tick_labels_format(xformat='hh:mm',yformat='dd:mm')
	print 'hi2'
	from pdb import set_trace
	# set_trace()
	EGSIm.save(Output_Fits_Directory+Fits_File_Name+'.png')
	#
	Show=True
	print 'hi3'
	if Show: plt.show()


## ======================================================================================================
## ======================================================================================================


# class test():
# 	B=2
# 	def __init__(self):
# 		self.a=1
# 	def print_a(self):
# 		print self.a
# 	@classmethod
# 	def print_b(cls):
# 		print cls.B, 3
# 	@staticmethod
# 	def print_c():
# 		print 4

# t = test()
# getattr(test, 'print_a')
# getattr(t, 'print_a')()
# getattr(test, 'print_b')()
# getattr(test, 'print_c')()





