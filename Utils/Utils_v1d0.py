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
from pdb import set_trace as brk
import argparse
import BayesEoR.Params.params as p


## ======================================================================================================
## ======================================================================================================

class PriorC(object):
	def __init__(self, priors_min_max):
		self.priors_min_max=priors_min_max

	def prior_func(self, cube):
		pmm = self.priors_min_max
		theta = []
		for i_p in range(len(cube)):
			theta_i = pmm[i_p][0]+((pmm[i_p][1]-pmm[i_p][0])*cube[i_p])
			theta.append(theta_i)
		return theta


## ======================================================================================================
## ======================================================================================================

class ParseCommandLineArguments(object):
	def __init__(self):
		#---Defaults---
		self.nq = 2

	def print_true_false(self, variable, variable_name):
		if variable:
			print variable_name+' set to True'
		else:
			print variable_name+' set to False'
	
	def parse(self, argv):
		print argv
		try:
		    opts, args = getopt.getopt(argv,'',["nq=",])
		    print opts, args
		    true_list = ['true', 't']
		    for opt, arg in opts:
		    	if opt=='--nq':
		    		self.nq = int(arg)
		except getopt.GetoptError:
		    sys.exit(2)
		return self.nq


## ======================================================================================================
## ======================================================================================================

class DataUnitConversionmkandJyperpix():
	def __init__(self, **kwargs):
		#
		##===== Defaults =======
		default_Print=0
		
		##===== Inputs =======
		self.Print=kwargs.pop('Print', default_Print)

	def Brightness_RJ(self, Frequency_Hz, Temperature_K, **kwargs):
		#Rayleigh-Jeans Law approximation
		#Define Constants
		K=1.3806503E-23
		c=299792458 #~3.e8 m/s
		Nu=Frequency_Hz
		T=Temperature_K

		default_Print=True
		Print = kwargs.pop('Print', default_Print)

		if Print: print '\nK = ', K, '\nc = ', c, '\nT = ', T, '\nFrequency = ', Nu
		B= ( (2*(Nu**2)*K*T)/(c**2) )
		if Print: print 'B = ', B, 'Wm^-2Hz^-1sr^-1'
		return B

	def convert_from_mK_to_Jy_per_pix(self, Data_mK, Channel_Frequencies_Array_Hz, Pixel_Height_rad, **kwargs):
		Data_K=Data_mK/1.e3
		Pixel_Width_rad=Pixel_Height_rad
		Pixel_Area_sr=Pixel_Width_rad*Pixel_Height_rad
		Data_Jy_per_Pixel = np.zeros(Data_K.shape)
		for i_channel in range(len(Channel_Frequencies_Array_Hz)):
			Channel_Frequency_Hz = Channel_Frequencies_Array_Hz[i_channel]
			Temperature_K = 1.0
			Image_Conversion_Factor_K_to_Jy_per_Pixel=Pixel_Area_sr*self.Brightness_RJ(Channel_Frequency_Hz, Temperature_K, Print=False)*1.e26
			Data_Jy_per_Pixel[i_channel]=Data_K[i_channel]*Image_Conversion_Factor_K_to_Jy_per_Pixel
		return Data_Jy_per_Pixel




## ======================================================================================================
## ======================================================================================================

class Cosmology():
	"""
	Class for performing cosmological distance calculations
	"""
	def __init__(self, **kwargs):
		#
		##===== Defaults =======
		self.z1=0.
		self.z2=10.
		self.Print=0
		
		##===== Inputs =======
		if 'z1' in kwargs:
			self.z1=kwargs['z1']	
		if 'z2' in kwargs:
			self.z2=kwargs['z2']	
		if 'Print' in kwargs:
			self.Print=kwargs['Print']	
		
		self.Omega_m=0.279
		self.Omega_lambda=0.721
		self.Omega_k=0.0
		self.c=(299792458.) #km/s
		self.c_km_per_sec=(299792458./1.e3) #km/s
		self.H_0=70.0 #km/s/Mpc
		self.f_21=1420.40575177 #1420.40575177 MHz
		self.E_z2=(self.Omega_m*((1.+self.z2)**3) + self.Omega_lambda)**0.5 #Hubble Parameter at redshift z2
		
	def Comoving_Distance_Mpc_Integrand(self, z, **kwargs):
		#
		E_z=(self.Omega_m*((1.+z)**3) + self.Omega_lambda)**0.5 #Hubble parameter
		self.Hubble_Distance=self.c_km_per_sec/self.H_0 #Distance in Mpc
		#
		return (self.Hubble_Distance/E_z)
	
	###
	#Calculate 21cmFast Box size in degrees at a given redshift
	###
	def Calculate_Comoving_Distance_Mpc_Between_Redshifts_z1_and_z2(self, **kwargs):
		#
		self.Comoving_Distance_Mpc, self.Comoving_convergence_uncertainty=integrate.quad(self.Comoving_Distance_Mpc_Integrand,self.z1,self.z2)
		
		return self.Comoving_Distance_Mpc, self.Comoving_convergence_uncertainty
	
	
	###
	#Calculate 21cmFast frequency depth at a given redshift using Morales & Hewitt 2004 eqn. 
	###
	#
	#Example run command: Functions.Cosmology(z2=10.,Print=1).Convert_from_Comoving_Mpc_to_Delta_Frequency_at_Redshift_z2()
	#
	def Convert_from_Comoving_Mpc_to_Delta_Frequency_at_Redshift_z2(self, **kwargs):
		
		#
		##===== Defaults =======
		self.Box_Side_cMpc=3000.
		
		##===== Inputs =======
		if 'Box_Side_cMpc' in kwargs:
			self.Box_Side_cMpc=kwargs['Box_Side_cMpc']	
		#
		
		if self.Print:
			print 'Convert_from_Comoving_Mpc_to_Delta_Frequency_at_Redshift_z2 at \nRedshift z =', self.z2, '\nBox depth in cMpc, Box_Side_cMpc =', self.Box_Side_cMpc
		
		
		self.Delta_f_MHz = (self.H_0*self.f_21*self.E_z2*self.Box_Side_cMpc)/(self.c_km_per_sec*(1.+self.z2)**2.)
		self.Delta_f_Hz = self.Delta_f_MHz*1.e6
		
		if self.Print:
			print 'Delta_f_MHz = ', self.Delta_f_MHz
		
		return self.Delta_f_MHz
	
	###
	#Calculate 21cmFast k_parallel - space values
	###
	def Convert_from_Tau_to_Kz(self, Tau_Array, **kwargs):
		#
		self.K_z_Array = ((2.*pi*self.H_0*self.f_21*self.E_z2)/(self.c_km_per_sec*(1.+self.z2)**2.))*Tau_Array #Eta_Array=array of spatial wavelengths fitted to channels in frequency space.
		
		return self.K_z_Array
	
	###
	#Calculate 21cmFast k_perp - space values
	###
	def Convert_from_U_to_Kx(self, U_Array, **kwargs):
		#
		Comoving_Distance_Mpc, Comoving_convergence_uncertainty=self.Calculate_Comoving_Distance_Mpc_Between_Redshifts_z1_and_z2()
		self.K_x_Array=((2.*pi)/Comoving_Distance_Mpc)*U_Array #h*cMPc^-1
		
		return self.K_x_Array
	
	###
	#Calculate 21cmFast k_perp - space values
	###
	def Convert_from_V_to_Ky(self, V_Array, **kwargs):
		#
		Comoving_Distance_Mpc, Comoving_convergence_uncertainty=self.Calculate_Comoving_Distance_Mpc_Between_Redshifts_z1_and_z2()
		self.K_y_Array=((2.*pi)/Comoving_Distance_Mpc)*V_Array #h*cMPc^-1
		
		return self.K_y_Array
	
	###
	#Convert from Frequency to Redshift
	###
	def Convert_from_21cmFrequency_to_Redshift(self, Frequency_Array_MHz, **kwargs):
		#
		One_plus_z_Array=self.f_21/Frequency_Array
		self.z_Array=One_plus_z_Array-1.
		
		return self.z_Array

	###
	#Convert from Redshift to Frequency
	###
	def Convert_from_Redshift_to_21cmFrequency(self, Redshift, **kwargs):
		#
		One_plus_z=1.+Redshift
		self.redshifted_21cm_frequency = self.f_21/One_plus_z
		
		return self.redshifted_21cm_frequency

	###
	#Calculate angular separation of 21cmFast box at a given redshift  
	###
	#
	#Example run command: Cosmology(z1=0.0,z2=10.,Print=1).Convert_from_Comoving_Mpc_to_Delta_Angle_at_Redshift_z2(Box_Side_cMpc=512)
	#
	def Convert_from_Comoving_Mpc_to_Delta_Angle_at_Redshift_z2(self, **kwargs):
		###
		# Angular diameter distance
		# An object of size x at redshift z that appears to have angular size \delta\theta has the angular diameter distance of d_A(z)=x/\delta\theta.
		# Angular diameter distance:
		# d_A(z)  = \frac{d_M(z)}{1+z}
		# with d_M(z) = Comoving_Distance_Mpc
		###

		#
		##===== Defaults =======
		self.Box_Side_cMpc=3000.
		
		##===== Inputs =======
		if 'Box_Side_cMpc' in kwargs:
			self.Box_Side_cMpc=kwargs['Box_Side_cMpc']	
				
		Comoving_Distance_Mpc, Comoving_convergence_uncertainty = self.Calculate_Comoving_Distance_Mpc_Between_Redshifts_z1_and_z2()
		angular_diameter_distance_Mpc = Comoving_Distance_Mpc/(1+(self.z2-self.z1))
		Box_width_proper_distance_Mpc = self.Box_Side_cMpc/(1+(self.z2-self.z1))

		if self.Print:
			print 'Convert_from_Comoving_Mpc_to_Delta_Angle_at_Redshift_z2 at \nRedshift z =', self.z2, '\nBox depth in cMpc, Box_Side_cMpc =', self.Box_Side_cMpc, '\nBox proper depth in Mpc, Box proper width =', Box_width_proper_distance_Mpc, '\nComoving distance between z1={} and z2={}: {}'.format(self.z1,self.z2,Comoving_Distance_Mpc), '\nAngular diameter distance between z1={} and z2={}: {}'.format(self.z1,self.z2,angular_diameter_distance_Mpc)
		
		###
		#tan(theta) = Box_Side_cMpc/Comoving_Distance_Mpc
		###
		# self.theta_rad = np.arctan2(Box_width_proper_distance_Mpc,angular_diameter_distance_Mpc)
		self.theta_rad = (Box_width_proper_distance_Mpc/angular_diameter_distance_Mpc) #From Hogg 1999 (although no discussion of applicability for large theta there)
		self.theta_deg = self.theta_rad*180./np.pi
		
		if self.Print:
			print 'theta_deg = ', self.theta_deg
		return self.theta_deg


## ======================================================================================================
## ======================================================================================================

class ExtractDataFrom21cmFASTCube():
	def __init__(self, **kwargs):
		#
		##===== Defaults =======
		default_Print=0
		default_plot_data=False
		
		##===== Inputs =======
		self.Print=kwargs.pop('Print', default_Print)
		self.plot_data=kwargs.pop('plot_data', default_plot_data)

	def extract_21cmFAST_data(self, FilePath, **kwargs):
		with open(FilePath,'rb') as File:
			Data = numpy.fromfile(File,'f')

		cube_width_pix = int(Data.size**(1./3)+0.5)
		Data = Data.reshape(cube_width_pix,cube_width_pix,cube_width_pix)
		if self.plot_data:
			self.plot_21cmFAST_cube_channel(Data)
		return Data

	def plot_21cmFAST_cube_channel(self, Data, **kwargs):
		channel_number = 0
		pylab.imshow(Data[channel_number])
		pylab.colorbar()
		pylab.show()


	def calculate_box_size_in_degrees_and_MHz(self, Box_Side_cMpc, simulation_redshift, **kwargs):
		bandwidth_MHz = Cosmology(z2=simulation_redshift,Print=1).Convert_from_Comoving_Mpc_to_Delta_Frequency_at_Redshift_z2(Box_Side_cMpc=Box_Side_cMpc)
		central_frequency_MHz = Cosmology().Convert_from_Redshift_to_21cmFrequency(simulation_redshift)
		output_21cmFast_box_width_deg = Cosmology(z1=0.0,z2=simulation_redshift,Print=1).Convert_from_Comoving_Mpc_to_Delta_Angle_at_Redshift_z2(Box_Side_cMpc=Box_Side_cMpc)

		if self.Print: print 'bandwidth_MHz', bandwidth_MHz
		if self.Print: print 'central_frequency_MHz', central_frequency_MHz
		if self.Print: print 'output_21cmFast_box_width_deg', output_21cmFast_box_width_deg

		return bandwidth_MHz, central_frequency_MHz, output_21cmFast_box_width_deg

	def Brightness_RJ(self, Frequency_Hz, Temperature_K, **kwargs):
		#Rayleigh-Jeans Law approximation
		#Define Constants
		K=1.3806503E-23
		c=299792458 #~3.e8 m/s
		Nu=Frequency_Hz
		T=Temperature_K

		default_Print=True
		Print = kwargs.pop('Print', default_Print)

		if Print: print '\nK = ', K, '\nc = ', c, '\nT = ', T, '\nFrequency = ', Nu
		B= ( (2*(Nu**2)*K*T)/(c**2) )
		if Print: print 'B = ', B, 'Wm^-2Hz^-1sr^-1'
		return B

	def convert_from_mK_to_Jy_per_pix(self, Data_mK, Channel_Frequencies_Array_Hz, Pixel_Height_rad, **kwargs):
		Data_K=Data_mK/1.e3
		Pixel_Width_rad=Pixel_Height_rad
		Pixel_Area_sr=Pixel_Width_rad*Pixel_Height_rad
		Data_Jy_per_Pixel = np.zeros(Data_K.shape)
		for i_channel in range(len(Channel_Frequencies_Array_Hz)):
			Channel_Frequency_Hz = Channel_Frequencies_Array_Hz[i_channel]
			Temperature_K = 1.0
			Image_Conversion_Factor_K_to_Jy_per_Pixel=Pixel_Area_sr*self.Brightness_RJ(Channel_Frequency_Hz, Temperature_K, Print=False)*1.e26
			Data_Jy_per_Pixel[i_channel]=Data_K[i_channel]*Image_Conversion_Factor_K_to_Jy_per_Pixel
		return Data_Jy_per_Pixel



## ======================================================================================================
## ======================================================================================================

class WriteDataToFits(ExtractDataFrom21cmFASTCube):
	def __init__(self, **kwargs):
		
		##===== Defaults =======
		default_Print=True
		
		##===== Inputs =======
		self.Print=kwargs.pop('Print', default_Print)

	def findbackupdir(self, startdir):
		"""
		Recursively search for a backup location and, when found,
		assign it to global backupdir
		"""
		global backupdir
		if os.path.exists(startdir):
			head, tail = os.path.split(os.path.normpath(startdir))
			if tail.count('_backup'):
				t1,t2 = tail.split('_backup')
				self.findbackupdir(head+'/'+t1+'_backup'+str(int(t2)+1))
			else:
				self.findbackupdir(head+'/'+tail+'_backup0')
		else:
			backupdir=startdir


	def Create_Directory(self, Directory,**kwargs):
		
		if not os.path.exists(Directory):
			print 'Directory not found: \n\n'+Directory+"\n"
			print 'Creating required directory structure..'
			os.makedirs(Directory)
		
		return 0

	def check_path_includes_the_directory(self, output_path):
		head, tail = os.path.split(os.path.normpath(output_path))
		if head=='':
			output_path=os.getcwd()+'/'+output_path
		else:
			self.Create_Directory(head)
		return output_path

	def write_data(self, Data, output_path, **kwargs):
		
		##===== Defaults =======
		# default_Box_Side_cMpc=4096.
		# default_simulation_redshift=8.682
		default_Box_Side_cMpc=3072.
		default_simulation_redshift=7.6
		
		##===== Inputs =======
		self.Box_Side_cMpc=kwargs.pop('Box_Side_cMpc', default_Box_Side_cMpc)
		self.simulation_redshift=kwargs.pop('simulation_redshift', default_simulation_redshift)

		nz,ny,nx = Data.shape

		bandwidth_MHz, central_frequency_MHz, output_21cmFast_box_width_deg = self.calculate_box_size_in_degrees_and_MHz(self.Box_Side_cMpc, self.simulation_redshift)

		# subset_bandwidth_MHz = bandwidth_MHz*38./nx
		subset_bandwidth_MHz =8.0

		hdu = fits.PrimaryHDU(Data)
		hdulist = fits.HDUList([hdu])
		hdulist[0].header.set('CTYPE1', 'RA---CAR')
		hdulist[0].header.set('CRVAL1', 0.0)
		hdulist[0].header.set('CDELT1', -output_21cmFast_box_width_deg/float(nx))
		hdulist[0].header.set('CRPIX1', int((nx+1)/2))
		hdulist[0].header.set('CUNIT1', 'deg')
		hdulist[0].header.set('CTYPE2', 'DEC--CAR')
		hdulist[0].header.set('CRVAL2', -30.0)
		hdulist[0].header.set('CDELT2', output_21cmFast_box_width_deg/float(ny))
		hdulist[0].header.set('CRPIX2', int((ny+1)/2))
		hdulist[0].header.set('CUNIT2', 'deg')
		hdulist[0].header.set('CTYPE3', 'FREQ')
		hdulist[0].header.set('CRVAL3', (central_frequency_MHz-(subset_bandwidth_MHz/2.))*1.e6)
		hdulist[0].header.set('CDELT3', (subset_bandwidth_MHz/float(nz))*1.e6)
		hdulist[0].header.set('CRPIX3', 1.0)
		hdulist[0].header.set('CUNIT3', 'Hz')
		hdulist[0].header.set('CTYPE4', 'STOKES')
		hdulist[0].header.set('CRVAL4', 1.0)
		hdulist[0].header.set('CDELT4', 1.0)
		hdulist[0].header.set('CRPIX4', 1.0)
		hdulist[0].header.set('CUNIT4', '')

		output_path = self.check_path_includes_the_directory(output_path)
		self.findbackupdir(output_path)
		if backupdir!=output_path:
			shutil.move(output_path, backupdir)

		hdu.writeto(output_path, overwrite=True)
	

## ======================================================================================================
## ======================================================================================================

def plot_signal_vs_MLsignal_residuals(true_signal, MLsignal, sigma, output_path='', Show=False, **kwargs):
	##===== Defaults =======
	demarcate_baseline = False
	##===== Inputs =======
	if 'demarcate_baseline' in kwargs:
		demarcate_baseline=kwargs['demarcate_baseline']

	pylab.close('all')
	fig,ax = pylab.subplots(nrows=4, ncols=2, figsize=(20,20))
	ax[0,0].errorbar(arange(len(true_signal)), true_signal.real, color='black')
	ax[0,0].errorbar(arange(len(MLsignal)), MLsignal.real, yerr=np.ones(len(MLsignal))*sigma,  color='red', fmt='+')
	ax[0,0].legend(['Re(signal)', 'Re(ML fit)'], fontsize=20)
	if demarcate_baseline:
		for l in (arange(1,(nu*nv-1),1)*nf)-1: ax[0,0].axvline(l, color='r', linestyle='--', lw=2, alpha=0.5)
	ax[0,0].set_ylabel('Amplitude, arbitrary units', fontsize=20)
	ax[0,0].set_xlabel('$uv$-cell', fontsize=20)
	ax[0,1].errorbar(arange(len(true_signal)), true_signal.imag, color='black')
	ax[0,1].errorbar(arange(len(MLsignal)), MLsignal.imag, yerr=np.ones(len(MLsignal))*sigma,  color='blue', fmt='+')
	ax[0,1].legend(['Im(signal)', 'Im(ML fit)'], fontsize=20)
	if demarcate_baseline:
		for l in (arange(1,(nu*nv-1),1)*nf)-1: ax[0,1].axvline(l, color='b', linestyle='--', lw=2, alpha=0.5)
	residuals = true_signal-MLsignal
	ax[0,1].set_xlabel('$uv$-cell', fontsize=20)
	ax[1,0].errorbar(arange(len(MLsignal)), residuals.real, yerr=np.ones(len(residuals))*sigma,  color='red', fmt='+')
	ax[1,0].legend(['$Re(s-m)$ residual'], fontsize=20)
	ax[1,0].errorbar(arange(len(true_signal)), np.zeros(len(true_signal)), color='black')
	ax[1,0].set_ylabel('residual, arbitrary units', fontsize=20)
	ax[1,0].set_xlabel('$uv$-cell', fontsize=20)
	ax[1,1].errorbar(arange(len(MLsignal)), residuals.imag, yerr=np.ones(len(residuals))*sigma,  color='blue', fmt='+')
	ax[1,1].legend(['$Im(s-m)$ residual'], fontsize=20)
	ax[1,1].errorbar(arange(len(true_signal)), np.zeros(len(true_signal)), color='black')
	ax[1,1].set_xlabel('$uv$-cell', fontsize=20)
	hist,bin_edges,patches = ax[2,0].hist(residuals.real, bins=20, color='red', edgecolor = "black")
	ax[2,0].legend(['$\mu={}$,\n$\sigma={}$, \n$\sigma_{}={}$'.format(np.round(residuals.real.mean(),1), np.round(residuals.real.std(),1), r'\mathrm{T}', np.round(sigma/2**0.5,1))], fontsize=20)
	ax[2,0].set_ylabel('count', fontsize=20)
	ax[2,0].set_xlabel('residual, arbitrary units', fontsize=20)
	hist,bin_edges,patches = ax[2,1].hist(residuals.imag, bins=20, color='blue', edgecolor = "black")
	ax[2,1].legend(['$\mu={}$,\n$\sigma={}$, \n$\sigma_{}={}$'.format(np.round(residuals.imag.mean(),1), np.round(residuals.imag.std(),1), r'\mathrm{T}', np.round(sigma/2**0.5,1))], fontsize=20)
	ax[2,1].set_ylabel('count', fontsize=20)
	ax[2,1].set_xlabel('residual, arbitrary units', fontsize=20)

	nuv = (p.nu*p.nv)-1
	max_res_index_real = residuals.real.reshape(nuv,p.nf).argmax()/p.nf
	# max_res_index_real = 100
	max_res_real = residuals.real.reshape(nuv,p.nf)[max_res_index_real]
	max_res_index_imag = residuals.imag.reshape(nuv,p.nf).argmax()/p.nf
	# max_res_index_imag = 100
	max_res_imag = residuals.imag.reshape(nuv,p.nf)[max_res_index_imag]
	alpha=0.3

	ax[3,0].errorbar(arange(len(max_res_real)), max_res_real, yerr=np.ones(len(max_res_real))*sigma,  color='red', fmt='+')
	ax[3,0].errorbar(arange(len(max_res_real)), max_res_real,  color='red', fmt='-')
	ax[3,0].legend(['$Re(s-m)$ residual'], fontsize=20)
	ax[3,0].errorbar(arange(len(max_res_real)), true_signal.real.reshape(nuv,p.nf)[max_res_index_real], color='black', fmt='-', alpha=alpha)
	ax[3,0].errorbar(arange(len(max_res_real)), MLsignal.real.reshape(nuv,p.nf)[max_res_index_real], color='black', fmt='--', alpha=alpha)
	ax[3,0].set_ylabel('residual, arbitrary units', fontsize=20)
	ax[3,0].set_xlabel('$uv$-cell', fontsize=20)
	ax[3,1].errorbar(arange(len(max_res_imag)), max_res_imag, yerr=np.ones(len(max_res_imag))*sigma,  color='blue', fmt='+')
	ax[3,1].errorbar(arange(len(max_res_imag)), max_res_imag,  color='blue', fmt='-')
	ax[3,1].legend(['$Im(s-m)$ residual'], fontsize=20)
	ax[3,1].errorbar(arange(len(max_res_real)), true_signal.imag.reshape(nuv,p.nf)[max_res_index_imag], color='black', fmt='-', alpha=alpha)
	ax[3,1].errorbar(arange(len(max_res_real)), MLsignal.imag.reshape(nuv,p.nf)[max_res_index_imag], color='black', fmt='--', alpha=alpha)
	ax[3,1].set_xlabel('$uv$-cell', fontsize=20)
	for axi in ax.ravel(): axi.tick_params(labelsize=20)
	if not output_path=='':fig.savefig(output_path)
	if Show:fig.show()


## ======================================================================================================
## ======================================================================================================

def generate_output_file_base(file_root, **kwargs):
	##===== Defaults =======
	default_version_number = '1'
	##===== Inputs =======
	if 'version_number' in kwargs:
		version_number=kwargs.pop('version_number',default_version_number)

	file_name_exists = os.path.isfile('chains/'+file_root+'_phys_live.txt') or os.path.isfile('chains/'+file_root+'.resume') or os.path.isfile('chains/'+file_root+'resume.dat')
	while file_name_exists:
		fr1,fr2 = file_root.split('-v')
		fr21,fr22 = fr2.split('-')
		next_version_number = str(int(fr21)+1)
		file_root=file_root.replace('v'+version_number+'-', 'v'+next_version_number+'-')
		version_number = next_version_number
		file_name_exists = os.path.isfile('chains/'+file_root+'_phys_live.txt') or os.path.isfile('chains/'+file_root+'.resume') or os.path.isfile('chains/'+file_root+'resume.dat')
	return file_root


## ======================================================================================================
## ======================================================================================================

class RenormaliseMatricesForScaledNoise(object):
	def __init__(self):
		self.sigma_init = 1.0
		self.updated_sigma = 1.0
		self.matrix_renormalisation_scale_factor = 1.0
		self.have_run_calc_matrix_renormalisation_scale_factor = False

	def calc_updated_225_sigma(self, S18a_163_sigma):
		# Tsys = 100 K + 60*(lambda/1 m)**2.55 (Ewall-Wice et al. 2016, pg. 11 below eqn. 22)
		# Tsys_225 = 100+60*(3.e8/225.e6)**2.55
		# Tsys_163 = 100+60*(3.e8/163.e6)**2.55
		Tsys_225 = 245+60*(3.e8/225.e6)**2.55
		Tsys_163 = 245+60*(3.e8/163.e6)**2.55
		# S18a_163_sigma = 100.e-1 #BayesEoR sigma used in Sims et al 2018a
		updated_225_sigma = S18a_163_sigma * (Tsys_225/Tsys_163)
		updated_225_sigma = np.round(updated_225_sigma, 1)
		print 'updated_225_sigma', updated_225_sigma
		return updated_225_sigma

	def calc_matrix_renormalisation_scale_factor(self, sigma_init, updated_sigma):
		# For diagonal N: N \propto 1./sigma**2
		self.sigma_init = sigma_init
		self.updated_sigma = updated_sigma
		self.matrix_renormalisation_scale_factor  = sigma_init**2./updated_sigma**2.
		self.have_run_calc_matrix_renormalisation_scale_factor = True
		print 'Input sigma_init:', self.sigma_init
		print 'Input updated_sigma:', self.updated_sigma
		print 'Derived matrix_renormalisation_scale_factor:', self.matrix_renormalisation_scale_factor
		return self.matrix_renormalisation_scale_factor

	def update_matrix_renormalisation_scale_factor(self, matrix_renormalisation_scale_factor):
		self.matrix_renormalisation_scale_factor  = matrix_renormalisation_scale_factor

	def renormalise_matrices_for_scaled_noise(self, T_Ninv_T, block_T_Ninv_T, Ninv, **kwargs):
		#
		##===== Defaults =======
		default_matrix_renormalisation_scale_factor=self.matrix_renormalisation_scale_factor
		
		##===== Inputs =======
		self.matrix_renormalisation_scale_factor=kwargs.pop('matrix_renormalisation_scale_factor', default_matrix_renormalisation_scale_factor)
		
		if self.have_run_calc_matrix_renormalisation_scale_factor:
			print "About to update likelihood matrices to from a constructed noise level of {} to a newly assumed noise in the data of {} using a scale factor sigma_init**2./updated_sigma**2. = {}".format(self.sigma_init, self.updated_sigma, (self.sigma_init**2./self.updated_sigma**2.))
		else:
			print "Warning, the calc_matrix_renormalisation_scale_factor hasn't been used to derive this updated matrix_renormalisation_scale_factor so the output noise level isn't being tracked by this class...\nUpdating likelihood matrices with the manually input scale factor: {}...".format(self.matrix_renormalisation_scale_factor)

		print 'Renormalising T_Ninv_T'
		T_Ninv_T       = T_Ninv_T*self.matrix_renormalisation_scale_factor
		print 'Renormalising block_T_Ninv_T'
		block_T_Ninv_T = block_T_Ninv_T*self.matrix_renormalisation_scale_factor
		print 'Renormalising Ninv'
		Ninv           = Ninv*self.matrix_renormalisation_scale_factor
		return T_Ninv_T, block_T_Ninv_T, Ninv


## ======================================================================================================
## ======================================================================================================

def write_log_file(array_save_directory, file_root, args):
	import subprocess
	
	# make log file directory if it doesn't exist
	log_dir = 'log_files/'
	if not os.path.exists(log_dir):
		os.mkdir(log_dir)

	# Get git version and hash info
	version_info = {}
	version_info['git_origin'] = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'], stderr=subprocess.STDOUT)
	version_info['git_hash'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT)
	version_info['git_description'] = subprocess.check_output(['git', 'describe', '--dirty', '--tag', '--always'])
	version_info['git_branch'] = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.STDOUT)

	log_file = log_dir + file_root + '.log'
	dashed_line = '-'*44
	
	# Write array directories and command line arguments
	with open(log_file, 'ab') as f:
		f.write('#' + dashed_line + '\n# GitHub Info\n#' + dashed_line +'\n')
		for key in version_info.keys():
			f.write('%s: %s' %(key, version_info[key]))
		f.write('\n\n')
		f.write('#' + dashed_line + '\n# Directories\n#' + dashed_line +'\n')
		f.write('Array save directory:\t%s\n' %(array_save_directory))
		f.write('Multinest output file root:\t%s\n' %(file_root))
		f.write('\n\n')
		f.write('#' + dashed_line + '\n# Command Line Arguments\n#' + dashed_line +'\n')
		for key in vars(args).keys():
			f.write(' %s = %s\n' %(key, vars(args)[key]))
		f.write('\n\n')
		f.write('#' + dashed_line + '\n# Params/params.py\n#' + dashed_line +'\n')
	
	# Write params file to log file
	subprocess.Popen('cat Params/params.py >> %s' %log_file, shell=True)
	
	print 'Log file written successfully to %s' %(log_file)












