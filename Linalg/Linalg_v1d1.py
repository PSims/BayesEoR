
import matplotlib as mpl
# import pymultinest
import numpy
from numpy import *
import pylab as P
import time
import sys
import os
import numpy as np

import BayesEoR.Params.params as p

#####
#
#Useful DFT link:
#https://www.cs.cf.ac.uk/Dave/Multimedia/node228.html
#
#####



###
#2D DFT testing
###

 
def makeGaussian(size, fwhm = 3, center=None):
	""" Make a square gaussian kernel.
	size is the length of a side of the square
	fwhm is full-width-half-maximum, which
	can be thought of as an effective radius.
	"""
	x = np.arange(0, size, 1, float)
	y = x[:,np.newaxis]
	if center is None:
		x0 = y0 = size // 2
	else:
		x0 = center[0]
		y0 = center[1]
	return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2) 




###
def make_Gaussian_beam(image_size_pix, fwhm_pix, beam_peak_amplitude, center_pix=[]):
	""" 
	Make a square gaussian kernel centered on center_pix=[x0, y0].
	"""
	x = np.arange(0, image_size_pix, 1, float)
	y = x[:,np.newaxis]
	if not center_pix:
		x0 = y0 = image_size_pix // 2
	else:
		x0 = center_pix[0]
		y0 = center_pix[1]
	gaussian_beam = beam_peak_amplitude * np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm_pix**2) 
	return gaussian_beam




###
def make_Uniform_beam(image_size_pix, beam_peak_amplitude = 1.0):
	""" Make a square uniform kernel.
	"""
	uniform_beam = beam_peak_amplitude * np.ones([image_size_pix, image_size_pix])
	return uniform_beam




###
def Produce_Full_Coordinate_Arrays(nu, nv, nx, ny):
	#U_oversampling_Factor=nu/float(nx) #Keeps uv-plane size constant and oversampled rather than DFTing to a larger uv-plane
	#V_oversampling_Factor=nv/float(ny) #Keeps uv-plane size constant and oversampled rather than DFTing to a larger uv-plane
	#
	i_y_Vector=(np.arange(ny)-ny/2)
	#i_y_Vector=numpy.fft.fftshift(arange(ny)) #This puts the centre of x,y grid: 0,0 at the centre of the vector rather than the start
	i_y_Vector=i_y_Vector.reshape(1,ny)
	i_y_Array=np.tile(i_y_Vector,ny)
	i_y_Array_Vectorised=i_y_Array.reshape(nx*ny,1)
	i_y_AV=i_y_Array_Vectorised
	#
	i_x_Vector=(np.arange(nx)-nx/2)
	#i_x_Vector=numpy.fft.fftshift(arange(nx)) #This puts the centre of x,y grid: 0,0 at the centre of the vector rather than the start
	i_x_Vector=i_x_Vector.reshape(nx,1)
	i_x_Array=np.tile(i_x_Vector,nx)
	i_x_Array_Vectorised=i_x_Array.reshape(nx*ny,1)
	i_x_AV=i_x_Array_Vectorised
	#
	i_v_Vector=(np.arange(nu)-nu/2)
	#i_v_Vector= numpy.fft.fftshift(arange(nu)) #This puts the centre of u,v grid: 0,0 at the centre of the vector rather than the start
	i_v_Vector=i_v_Vector.reshape(1,nu)
	i_v_Array=np.tile(i_v_Vector,nv)
	i_v_Array_Vectorised=i_v_Array.reshape(1,nu*nv)
	i_v_AV=i_v_Array_Vectorised
	#
	i_u_Vector=(np.arange(nv)-nv/2)
	#i_u_Vector=numpy.fft.fftshift(arange(nv)) #This puts the centre of u,v grid: 0,0 at the centre of the vector rather than the start
	i_u_Vector=i_u_Vector.reshape(nv,1)
	i_u_Array=np.tile(i_u_Vector,nu)
	i_u_Array_Vectorised=i_u_Array.reshape(1,nv*nu)
	i_u_AV=i_u_Array_Vectorised
	#
	#
	#ExponentArray=np.exp(-2.0*np.pi*1j*( (i_x_AV*i_u_AV/float(nx)) +  (i_v_AV*i_y_AV/float(ny)) ))
	return i_x_AV, i_y_AV, i_u_AV, i_v_AV




###
def Produce_Coordinate_Arrays_ZM(nu, nv, nx, ny, **kwargs):
	##===== Defaults =======
	default_exclude_mean = True

	##===== Inputs =======
	exclude_mean=kwargs.pop('exclude_mean',default_exclude_mean)
	#
	i_y_Vector=(np.arange(ny)-ny/2)
	#i_y_Vector=numpy.fft.fftshift(arange(ny)) #This puts the centre of x,y grid: 0,0 at the centre of the vector rather than the start
	i_y_Vector=i_y_Vector.reshape(1,ny)
	i_y_Array=np.tile(i_y_Vector,ny)
	i_y_Array_Vectorised=i_y_Array.reshape(nx*ny,1)
	i_y_AV=i_y_Array_Vectorised
	#
	i_x_Vector=(np.arange(nx)-nx/2)
	#i_x_Vector=numpy.fft.fftshift(arange(nx)) #This puts the centre of x,y grid: 0,0 at the centre of the vector rather than the start
	i_x_Vector=i_x_Vector.reshape(nx,1)
	i_x_Array=np.tile(i_x_Vector,nx)
	i_x_Array_Vectorised=i_x_Array.reshape(nx*ny,1)
	i_x_AV=i_x_Array_Vectorised
	#
	i_v_Vector=(np.arange(nu)-nu/2)
	#i_v_Vector= numpy.fft.fftshift(arange(nu)) #This puts the centre of u,v grid: 0,0 at the centre of the vector rather than the start
	i_v_Vector=i_v_Vector.reshape(1,nu)
	i_v_Array=np.tile(i_v_Vector,nv)
	i_v_Array_Vectorised=i_v_Array.reshape(1,nu*nv)
	i_v_AV=i_v_Array_Vectorised
	if exclude_mean:
		i_v_AV=numpy.delete(i_v_AV,[i_v_AV.size/2]) #Remove the centre uv-pix
	#
	i_u_Vector=(np.arange(nv)-nv/2)
	#i_u_Vector=numpy.fft.fftshift(arange(nv)) #This puts the centre of u,v grid: 0,0 at the centre of the vector rather than the start
	i_u_Vector=i_u_Vector.reshape(nv,1)
	i_u_Array=np.tile(i_u_Vector,nu)
	i_u_Array_Vectorised=i_u_Array.reshape(1,nv*nu)
	i_u_AV=i_u_Array_Vectorised
	if exclude_mean:
		i_u_AV=numpy.delete(i_u_AV,[i_u_AV.size/2]) #Remove the centre uv-pix
	#
	#
	#ExponentArray=np.exp(-2.0*np.pi*1j*( (i_x_AV*i_u_AV/float(nx)) +  (i_v_AV*i_y_AV/float(ny)) ))
	return i_x_AV, i_y_AV, i_u_AV, i_v_AV




###
def Produce_Coordinate_Arrays_ZM_Coarse(nu, nv, nx, ny):
	#U_oversampling_Factor=nu/float(nx) #Keeps uv-plane size constant and oversampled rather than DFTing to a larger uv-plane
	#V_oversampling_Factor=nv/float(ny) #Keeps uv-plane size constant and oversampled rather than DFTing to a larger uv-plane
	#
	i_y_Vector=(np.arange(ny)-ny/2)
	#i_y_Vector=numpy.fft.fftshift(arange(ny)) #This puts the centre of x,y grid: 0,0 at the centre of the vector rather than the start
	i_y_Vector=i_y_Vector.reshape(1,ny)
	i_y_Array=np.tile(i_y_Vector,ny)
	i_y_Array_Vectorised=i_y_Array.reshape(nx*ny,1)
	i_y_AV=i_y_Array_Vectorised
	#
	i_x_Vector=(np.arange(nx)-nx/2)
	#i_x_Vector=numpy.fft.fftshift(arange(nx)) #This puts the centre of x,y grid: 0,0 at the centre of the vector rather than the start
	i_x_Vector=i_x_Vector.reshape(nx,1)
	i_x_Array=np.tile(i_x_Vector,nx)
	i_x_Array_Vectorised=i_x_Array.reshape(nx*ny,1)
	i_x_AV=i_x_Array_Vectorised
	#
	i_v_Vector=(np.arange(nu)-nu/2)
	#i_v_Vector= numpy.fft.fftshift(arange(nu)) #This puts the centre of u,v grid: 0,0 at the centre of the vector rather than the start
	i_v_Vector=i_v_Vector.reshape(1,nu)
	i_v_Array=np.tile(i_v_Vector,nv)
	i_v_Array_Vectorised=i_v_Array.reshape(1,nu*nv)
	i_v_AV=i_v_Array_Vectorised
	#
	GridSize=i_v_AV.size
	GridIndex, MaskOuterPoints=Calc_Indices_Centre_3x3_Grid(GridSize)
	InnerSubgridIndices=GridIndex[MaskOuterPoints]
	#
	Centre_v_CoordIndex=i_v_AV.size/2
	i_v_AV=numpy.delete(i_v_AV,InnerSubgridIndices) #Remove the centre 3x3 uv-grid (to be replaced by subharmonic grid)
	#i_v_AV=numpy.delete(i_v_AV,range(Centre_v_CoordIndex-4, Centre_v_CoordIndex+5,1)) #Remove the centre 3x3 uv-grid (to be replaced by subharmonic grid)
	#
	i_u_Vector=(np.arange(nv)-nv/2)
	#i_u_Vector=numpy.fft.fftshift(arange(nv)) #This puts the centre of u,v grid: 0,0 at the centre of the vector rather than the start
	i_u_Vector=i_u_Vector.reshape(nv,1)
	i_u_Array=np.tile(i_u_Vector,nu)
	i_u_Array_Vectorised=i_u_Array.reshape(1,nv*nu)
	i_u_AV=i_u_Array_Vectorised
	Centre_u_CoordIndex=i_u_AV.size/2
	i_u_AV=numpy.delete(i_u_AV,InnerSubgridIndices) #Remove the centre 3x3 uv-grid (to be replaced by subharmonic grid)
	#i_u_AV=numpy.delete(i_u_AV,range(Centre_u_CoordIndex-4, Centre_u_CoordIndex+5,1)) #Remove the centre 3x3 uv-grid (to be replaced by subharmonic grid)
	#
	#
	#ExponentArray=np.exp(-2.0*np.pi*1j*( (i_x_AV*i_u_AV/float(nx)) +  (i_v_AV*i_y_AV/float(ny)) ))
	return i_x_AV, i_y_AV, i_u_AV, i_v_AV








###
def Produce_Coordinate_Arrays_ZM_SH(nu, nv, nx, ny):
	#U_oversampling_Factor=nu/float(nx) #Keeps uv-plane size constant and oversampled rather than DFTing to a larger uv-plane
	#V_oversampling_Factor=nv/float(ny) #Keeps uv-plane size constant and oversampled rather than DFTing to a larger uv-plane
	#
	i_y_Vector=(np.arange(ny)-ny/2)
	#i_y_Vector=numpy.fft.fftshift(arange(ny)) #This puts the centre of x,y grid: 0,0 at the centre of the vector rather than the start
	i_y_Vector=i_y_Vector.reshape(1,ny)
	i_y_Array=np.tile(i_y_Vector,ny)
	i_y_Array_Vectorised=i_y_Array.reshape(nx*ny,1)
	i_y_AV=i_y_Array_Vectorised
	#
	i_x_Vector=(np.arange(nx)-nx/2)
	#i_x_Vector=numpy.fft.fftshift(arange(nx)) #This puts the centre of x,y grid: 0,0 at the centre of the vector rather than the start
	i_x_Vector=i_x_Vector.reshape(nx,1)
	i_x_Array=np.tile(i_x_Vector,nx)
	i_x_Array_Vectorised=i_x_Array.reshape(nx*ny,1)
	i_x_AV=i_x_Array_Vectorised
	#
	i_v_Vector=(np.arange(nu)-nu/2)
	#i_v_Vector= numpy.fft.fftshift(arange(nu)) #This puts the centre of u,v grid: 0,0 at the centre of the vector rather than the start
	i_v_Vector=i_v_Vector.reshape(1,nu)
	i_v_Array=np.tile(i_v_Vector,nv)
	i_v_Array_Vectorised=i_v_Array.reshape(1,nu*nv)
	i_v_AV=i_v_Array_Vectorised
	Centre_v_CoordIndex=i_v_AV.size/2
	i_v_AV=numpy.delete(i_v_AV,[Centre_v_CoordIndex]) #Remove the centre uv-pix
	#
	i_u_Vector=(np.arange(nv)-nv/2)
	#i_u_Vector=numpy.fft.fftshift(arange(nv)) #This puts the centre of u,v grid: 0,0 at the centre of the vector rather than the start
	i_u_Vector=i_u_Vector.reshape(nv,1)
	i_u_Array=np.tile(i_u_Vector,nu)
	i_u_Array_Vectorised=i_u_Array.reshape(1,nv*nu)
	i_u_AV=i_u_Array_Vectorised
	Centre_u_CoordIndex=i_u_AV.size/2
	i_u_AV=numpy.delete(i_u_AV,[Centre_u_CoordIndex]) #Remove the centre uv-pix
	#
	#
	#ExponentArray=np.exp(-2.0*np.pi*1j*( (i_x_AV*i_u_AV/float(nx)) +  (i_v_AV*i_y_AV/float(ny)) ))
	return i_x_AV, i_y_AV, i_u_AV, i_v_AV





###
def Calc_Coords_High_Res_Im_to_Large_uv(i_x_AV, i_y_AV, i_u_AV, i_v_AV, X_oversampling_Factor=1.0, Y_oversampling_Factor=1.0):
	#
	Y_oversampling_Factor=float(Y_oversampling_Factor)
	X_oversampling_Factor=float(X_oversampling_Factor)
	#
	i_y_AV=i_y_AV/Y_oversampling_Factor #Keeps xy-plane size constant and oversampled rather than DFTing from a larger xy-plane
	i_x_AV=i_x_AV/X_oversampling_Factor #Keeps xy-plane size constant and oversampled rather than DFTing from a larger xy-plane
	#
	return i_x_AV, i_y_AV, i_u_AV, i_v_AV




###
def Calc_Coords_Large_Im_to_High_Res_uv(i_x_AV, i_y_AV, i_u_AV, i_v_AV, U_oversampling_Factor=1.0, V_oversampling_Factor=1.0):
	#
	V_oversampling_Factor=float(V_oversampling_Factor)
	U_oversampling_Factor=float(U_oversampling_Factor)
	#
	i_v_AV=i_v_AV/V_oversampling_Factor #Keeps uv-plane size constant and oversampled rather than DFTing to a larger uv-plane
	i_u_AV=i_u_AV/U_oversampling_Factor #Keeps uv-plane size constant and oversampled rather than DFTing to a larger uv-plane
	#
	return i_x_AV, i_y_AV, i_u_AV, i_v_AV







def Restore_Centre_Pixel(Array, MeanVal=0.0):
	#
	Restored_Array=numpy.insert(Array, [Array.size/2],[MeanVal])
	
	return Restored_Array





def Calc_Indices_Centre_3x3_Grid(GridSize):
	#
	GridLength=int(GridSize**0.5)
	#
	LenX=LenY=GridLength
	#
	GridIndex=arange(LenX*LenY).reshape(LenX,LenY)
	Mask=zeros(LenX*LenY).reshape(LenX,LenY)
	Mask[len(Mask)/2-1:len(Mask)/2+2,len(Mask[0])/2-1:len(Mask[0])/2+2]=1
	#
	MaskOuterPoints=Mask.astype('bool')
	#
	return GridIndex, MaskOuterPoints



def Delete_Centre_3x3_Grid(Array):
	#
	GridSize=Array.size
	GridIndex, MaskOuterPoints=Calc_Indices_Centre_3x3_Grid(GridSize)
	#
	OuterArray=np.delete(Array, GridIndex[MaskOuterPoints])
	#
	return OuterArray




def Delete_Centre_Pix(Array):
	#
	Array=numpy.delete(Array, [Array.size/2])
	
	return Array



def N_is_Odd(N):
	return N%2


def Calc_Indices_Centre_NxN_Grid(GridSize, N):
	#
	GridLength=int(GridSize**0.5)
	#
	LenX=LenY=GridLength
	#
	GridIndex=arange(LenX*LenY).reshape(LenX,LenY)
	Mask=zeros(LenX*LenY).reshape(LenX,LenY)
	if N_is_Odd(N):
		Mask[len(Mask)/2-(N/2):len(Mask)/2+(N/2+1),len(Mask[0])/2-(N/2):len(Mask[0])/2+(N/2+1)]=1
	else:
		Mask[len(Mask)/2-(N/2):len(Mask)/2+(N/2),len(Mask[0])/2-(N/2):len(Mask[0])/2+(N/2)]=1
	#
	MaskOuterPoints=Mask.astype('bool')
	#
	return GridIndex, MaskOuterPoints


def Obtain_Centre_NxN_Grid(Array, N):
	#
	GridSize=Array.size
	GridIndex, MaskOuterPoints=Calc_Indices_Centre_NxN_Grid(GridSize, N)
	#
	Centre_NxN_Grid=Array.flatten()[GridIndex[MaskOuterPoints]]
	#
	return Centre_NxN_Grid



def Restore_Centre_3x3_Grid(Array, MeanVal=0.0):
	#
	LenRestoredArray=Array.size+9
	#
	GridSize=LenRestoredArray
	GridIndex, MaskOuterPoints=Calc_Indices_Centre_3x3_Grid(GridSize)
	#
	CurrentPointsIndex=GridIndex[np.where(np.logical_not(MaskOuterPoints))]
	RestoredPointsIndex=GridIndex[np.where((MaskOuterPoints))]
	#
	ConcatIndices=np.concatenate((CurrentPointsIndex,RestoredPointsIndex))
	SortedIndices=ConcatIndices.argsort()
	#
	Restored_Array_Unsorted=numpy.append(Array,[MeanVal]*9)
	Restored_Array=Restored_Array_Unsorted[SortedIndices]
	#
	return Restored_Array
	


def Restore_Centre_NxN_Grid(Array1, Array2, N):
	#
	LenRestoredArray=Array1.size+N*N
	#
	GridSize=LenRestoredArray
	GridIndex, MaskOuterPoints=Calc_Indices_Centre_NxN_Grid(GridSize, N)
	#
	CurrentPointsIndex=GridIndex[np.where(np.logical_not(MaskOuterPoints))]
	RestoredPointsIndex=GridIndex[np.where((MaskOuterPoints))]
	#
	ConcatIndices=np.concatenate((CurrentPointsIndex,RestoredPointsIndex))
	SortedIndices=ConcatIndices.argsort()
	#
	Restored_Array_Unsorted=numpy.append(Array1,Array2)
	Restored_Array=Restored_Array_Unsorted[SortedIndices]
	#
	return Restored_Array
	


def Generate_Combined_Coarse_plus_Subharmic_uv_grids(nu, nv, nx, ny, X_oversampling_Factor, Y_oversampling_Factor, U_oversampling_Factor, V_oversampling_Factor, ReturnSeparateCoarseandSHarrays=False):
	#U_oversampling_Factor and V_oversampling_Factor are the factors by which the subharmonic grid is oversampled relative to the coarse grid.
	#
	nu_SH=3*int(U_oversampling_Factor)
	nv_SH=3*int(V_oversampling_Factor)
	n_SH=(nu_SH*nv_SH)-1
	#
	i_x_AV_C, i_y_AV_C, i_u_AV_C, i_v_AV_C = Produce_Coordinate_Arrays_ZM_Coarse(nu, nv, nx, ny)
	i_x_AV_SH, i_y_AV_SH, i_u_AV_SH, i_v_AV_SH = Produce_Coordinate_Arrays_ZM_SH(nu_SH, nv_SH, nx, ny)
	#
	if U_oversampling_Factor!=1.0:
		i_x_AV_SH, i_y_AV_SH, i_u_AV_SH, i_v_AV_SH = Calc_Coords_Large_Im_to_High_Res_uv(i_x_AV_SH, i_y_AV_SH, i_u_AV_SH, i_v_AV_SH, U_oversampling_Factor, V_oversampling_Factor)
	if X_oversampling_Factor!=1.0:
		i_x_AV_C, i_y_AV_C, i_u_AV_C, i_v_AV_C = Calc_Coords_High_Res_Im_to_Large_uv(i_x_AV_C, i_y_AV_C, i_u_AV_C, i_v_AV_C, X_oversampling_Factor, Y_oversampling_Factor)
	#
	#Combine Coarse and subharmic uv-grids.
	i_u_AV=np.concatenate((i_u_AV_C, i_u_AV_SH))
	i_v_AV=np.concatenate((i_v_AV_C, i_v_AV_SH))
	#
	i_x_AV=i_x_AV_C
	i_y_AV=i_y_AV_C
	#
	if not ReturnSeparateCoarseandSHarrays:
		return i_u_AV, i_v_AV, i_x_AV, i_y_AV
	else:
		return i_u_AV_C, i_u_AV_SH, i_v_AV_C, i_v_AV_SH, i_x_AV, i_y_AV
	


def IDFT_Array_IDFT_2D_ZM_SH(nu, nv, nx, ny, X_oversampling_Factor=1.0, Y_oversampling_Factor=1.0, U_oversampling_Factor=1.0, V_oversampling_Factor=1.0):
	#U_oversampling_Factor and V_oversampling_Factor are the factors by which the subharmonic grid is oversampled relative to the coarse grid.
	#
	i_u_AV, i_v_AV, i_x_AV, i_y_AV = Generate_Combined_Coarse_plus_Subharmic_uv_grids(nu, nv, nx, ny, X_oversampling_Factor, Y_oversampling_Factor, U_oversampling_Factor, V_oversampling_Factor)
	#
	ExponentArray=np.exp(+2.0*np.pi*1j*( (i_x_AV*i_u_AV/float(nu)) +  (i_v_AV*i_y_AV/float(nv)) ))
	#
	NormalisedExponentArray=ExponentArray.T
	NormalisedExponentArray=NormalisedExponentArray/((nu*nv))
	
	return NormalisedExponentArray
	




#######
#Generates a DFT matrix (DFTArray) where upon dot producting with the input image (TestData) yields FFTTestData_Hermitian:
#FFTTestData_Hermitian = np.dot(TestData.reshape(1,nx*ny), DFTArrayF2).reshape(nu,nv) 
#(= np.dot(DFTArrayF2.T,TestData.reshape(1,nx*ny).T).reshape(nu,nv))

#And is equivalent in numpy to:

#ShiftedTestData    = numpy.fft.ifftshift(TestData+0j, axes=(0,1))
#FFTTestData        = numpy.fft.fftn(ShiftedTestData, axes=(0,1))
#ShiftedFFTTestData = numpy.fft.fftshift(FFTTestData, axes=(0,1))
#######

###
def DFT_Array_DFT_2D(nu, nv, nx, ny, X_oversampling_Factor=1.0, Y_oversampling_Factor=1.0, U_oversampling_Factor=1.0, V_oversampling_Factor=1.0):
	#
	i_x_AV, i_y_AV, i_u_AV, i_v_AV = Produce_Full_Coordinate_Arrays(nu, nv, nx, ny)
	#
	if U_oversampling_Factor!=1.0:
		i_x_AV, i_y_AV, i_u_AV, i_v_AV = Calc_Coords_Large_Im_to_High_Res_uv(i_x_AV, i_y_AV, i_u_AV, i_v_AV, U_oversampling_Factor, V_oversampling_Factor)
	if X_oversampling_Factor!=1.0:
		i_x_AV, i_y_AV, i_u_AV, i_v_AV = Calc_Coords_High_Res_Im_to_Large_uv(i_x_AV, i_y_AV, i_u_AV, i_v_AV, U_oversampling_Factor, V_oversampling_Factor)
	#
	ExponentArray=np.exp(-2.0*np.pi*1j*( (i_x_AV*i_u_AV/float(nx)) +  (i_v_AV*i_y_AV/float(ny)) ))
	return ExponentArray





###
# non-uniform DFT from image space to uv-coordinates given by p.uvw_multi_time_step_array_meters_reshaped (for examples, the sub-100m baselines sampled by HERA 331).
###
def nuDFT_Array_DFT_2D(nu, nv, nx, ny, chan_freq_MHz, sampled_uvw_coords_m, X_oversampling_Factor=1.0, Y_oversampling_Factor=1.0, U_oversampling_Factor=1.0, V_oversampling_Factor=1.0):
	#
	# sampled_uvw_coords_m = p.uvw_multi_time_step_array_meters_reshaped
	sampled_uvw_coords_wavelengths = sampled_uvw_coords_m/(p.speed_of_light/(chan_freq_MHz*1.e6)) # Convert uv-coordinates from meters to wavelengths at frequency chan_freq_MHz
	sampled_uvw_coords_inverse_pixel_units = sampled_uvw_coords_wavelengths/p.uv_pixel_width_wavelengths #Convert uv-coordinates from wavelengths to inverse pixel units

	i_x_AV, i_y_AV, i_u_AV, i_v_AV = Produce_Full_Coordinate_Arrays(nu, nv, nx, ny)
	# Overwrite gridded uv coords with instrumental uv coords loaded in params
	i_u_AV = sampled_uvw_coords_inverse_pixel_units[:,0].reshape(1,-1)
	i_v_AV = sampled_uvw_coords_inverse_pixel_units[:,1].reshape(1,-1)
	#
	if U_oversampling_Factor!=1.0:
		i_x_AV, i_y_AV, i_u_AV, i_v_AV = Calc_Coords_Large_Im_to_High_Res_uv(i_x_AV, i_y_AV, i_u_AV, i_v_AV, U_oversampling_Factor, V_oversampling_Factor)
	if X_oversampling_Factor!=1.0:
		i_x_AV, i_y_AV, i_u_AV, i_v_AV = Calc_Coords_High_Res_Im_to_Large_uv(i_x_AV, i_y_AV, i_u_AV, i_v_AV, U_oversampling_Factor, V_oversampling_Factor)
	#
	ExponentArray=np.exp(+2.0*np.pi*1j*( (i_x_AV*i_u_AV/float(nx)) +  (i_v_AV*i_y_AV/float(ny)) ))
	return ExponentArray




###
# non-uniform DFT from image space to uv-coordinates given by sampled_uvw_coords_inverse_pixel_units (for examples, the sub-100m baselines sampled by HERA 331).
###
def nuDFT_Array_DFT_2D_v2d0(nu, nv, nx, ny, sampled_uvw_coords_inverse_pixel_units, X_oversampling_Factor=1.0, Y_oversampling_Factor=1.0, U_oversampling_Factor=1.0, V_oversampling_Factor=1.0):

	i_x_AV, i_y_AV, i_u_AV, i_v_AV = Produce_Full_Coordinate_Arrays(nu, nv, nx, ny)
	# Overwrite gridded uv coords with instrumental uv coords loaded in params
	i_u_AV = sampled_uvw_coords_inverse_pixel_units[:,0].reshape(1,-1)
	i_v_AV = sampled_uvw_coords_inverse_pixel_units[:,1].reshape(1,-1)
	#
	if U_oversampling_Factor!=1.0:
		i_x_AV, i_y_AV, i_u_AV, i_v_AV = Calc_Coords_Large_Im_to_High_Res_uv(i_x_AV, i_y_AV, i_u_AV, i_v_AV, U_oversampling_Factor, V_oversampling_Factor)
	if X_oversampling_Factor!=1.0:
		i_x_AV, i_y_AV, i_u_AV, i_v_AV = Calc_Coords_High_Res_Im_to_Large_uv(i_x_AV, i_y_AV, i_u_AV, i_v_AV, U_oversampling_Factor, V_oversampling_Factor)
	#
	ExponentArray=np.exp(+2.0*np.pi*1j*( (i_x_AV*i_u_AV/float(nx)) +  (i_v_AV*i_y_AV/float(ny)) ))
	return ExponentArray


# a = np.ones([9,9])
# nuidft = nuDFT_Array_DFT_2D(nu, nv, nx, ny, 159.0)
# b3 = np.dot(nudft.T, a.reshape(-1,1))
# md_nudft =  block_diag(*[nuDFT_Array_DFT_2D(nu, nv, nx, ny, chan_freq_MHz) for chan_freq_MHz in np.linspace(159.,168.,p.nf)])






#######
#Generates a IDFT matrix (IDFTArray) where upon dot producting with the input image (TestData) yields FFTTestData_Hermitian:
#FFTTestData_Hermitian = np.dot(TestData.reshape(1,nx*ny), DFTArrayF2).reshape(nu,nv) 
#(= np.dot(DFTArrayF2.T,TestData.reshape(1,nx*ny).T).reshape(nu,nv))

#And is equivalent in numpy to:

#ShiftedTestData    = numpy.fft.ifftshift(TestData+0j, axes=(0,1))
#FFTTestData        = numpy.fft.fftn(ShiftedTestData, axes=(0,1))
#ShiftedFFTTestData = numpy.fft.fftshift(FFTTestData, axes=(0,1))
#######

###
def IDFT_Array_IDFT_2D(nu, nv, nx, ny, X_oversampling_Factor=1.0, Y_oversampling_Factor=1.0, U_oversampling_Factor=1.0, V_oversampling_Factor=1.0):
	#
	i_x_AV, i_y_AV, i_u_AV, i_v_AV = Produce_Full_Coordinate_Arrays(nu, nv, nx, ny)
	#
	if U_oversampling_Factor!=1.0:
		i_x_AV, i_y_AV, i_u_AV, i_v_AV = Calc_Coords_Large_Im_to_High_Res_uv(i_x_AV, i_y_AV, i_u_AV, i_v_AV, U_oversampling_Factor, V_oversampling_Factor)
	if X_oversampling_Factor!=1.0:
		i_x_AV, i_y_AV, i_u_AV, i_v_AV = Calc_Coords_High_Res_Im_to_Large_uv(i_x_AV, i_y_AV, i_u_AV, i_v_AV, U_oversampling_Factor, V_oversampling_Factor)
	#
	ExponentArray=np.exp(+2.0*np.pi*1j*( (i_x_AV*i_u_AV/float(nu)) +  (i_v_AV*i_y_AV/float(nv)) ))
	#return ExponentArray.T/(nu*U_oversampling_Factor*nv*V_oversampling_Factor)
	return ExponentArray.T/(nu*U_oversampling_Factor*nv*V_oversampling_Factor)






###
def DFT_Array_DFT_2D_ZM(nu, nv, nx, ny, X_oversampling_Factor=1.0, Y_oversampling_Factor=1.0, U_oversampling_Factor=1.0, V_oversampling_Factor=1.0):
	#
	exclude_mean = True
	if p.fit_for_monopole:
		exclude_mean = False
	i_x_AV, i_y_AV, i_u_AV, i_v_AV = Produce_Coordinate_Arrays_ZM(nu, nv, nx, ny, exclude_mean = exclude_mean)
	#
	if U_oversampling_Factor!=1.0:
		i_x_AV, i_y_AV, i_u_AV, i_v_AV = Calc_Coords_Large_Im_to_High_Res_uv(i_x_AV, i_y_AV, i_u_AV, i_v_AV, U_oversampling_Factor, V_oversampling_Factor)
	if X_oversampling_Factor!=1.0:
		i_x_AV, i_y_AV, i_u_AV, i_v_AV = Calc_Coords_High_Res_Im_to_Large_uv(i_x_AV, i_y_AV, i_u_AV, i_v_AV, U_oversampling_Factor, V_oversampling_Factor)
	#
	ExponentArray=np.exp(-2.0*np.pi*1j*( (i_x_AV*i_u_AV/float(nx)) +  (i_v_AV*i_y_AV/float(ny)) ))
	return ExponentArray





###
def IDFT_Array_IDFT_2D_ZM(nu, nv, nx, ny, X_oversampling_Factor=1.0, Y_oversampling_Factor=1.0, U_oversampling_Factor=1.0, V_oversampling_Factor=1.0):
	#
	exclude_mean = True
	if p.fit_for_monopole:
		exclude_mean = False
	i_x_AV, i_y_AV, i_u_AV, i_v_AV = Produce_Coordinate_Arrays_ZM(nu, nv, nx, ny, exclude_mean = exclude_mean)
	#
	if U_oversampling_Factor!=1.0:
		i_x_AV, i_y_AV, i_u_AV, i_v_AV = Calc_Coords_Large_Im_to_High_Res_uv(i_x_AV, i_y_AV, i_u_AV, i_v_AV, U_oversampling_Factor, V_oversampling_Factor)
	if X_oversampling_Factor!=1.0:
		i_x_AV, i_y_AV, i_u_AV, i_v_AV = Calc_Coords_High_Res_Im_to_Large_uv(i_x_AV, i_y_AV, i_u_AV, i_v_AV, U_oversampling_Factor, V_oversampling_Factor)
	#
	ExponentArray=np.exp(+2.0*np.pi*1j*( (i_x_AV*i_u_AV/float(nu)) +  (i_v_AV*i_y_AV/float(nv)) ))
	#return ExponentArray.T/(nu*U_oversampling_Factor*nv*V_oversampling_Factor)
	return ExponentArray.T/(nu*U_oversampling_Factor*nv*V_oversampling_Factor)






#####


def Construct_Hermitian(Tri_Real, Tri_Imag):
	Nx=int(((len(Tri_Real)*2)-1)**0.5)
	#
	Full_Real=np.concatenate((Tri_Real,Tri_Real[:-1][::-1])).reshape(Nx,Nx)
	Full_Imag=np.concatenate((Tri_Imag,-1.0*Tri_Imag[:-1][::-1])).reshape(Nx,Nx)
	#
	return Full_Real+1j*Full_Imag
	



###
#1. Construct G
###
#Assumes parameters are ordered: all real then all imaginary and the parameters are the upper triangular values of the uv-plane (which describes the whole uv-plane since it is Hermitian so UV[i,j]=UV[-i,-j] (or UV[-1-i,-1-j] in python because of zero indexing).
def Construct_Hermitian_Gridding_Matrix(nu,nv):
	#n_par=((nu*(nv-1)/2))*2
	n_par=((nu*nv/2)+1)*2
	#n_par=((nu*nv/2))*2 #Not including the offset term
	n_par_div2=n_par/2
	#
	G=np.zeros([nu*nv,n_par])+0j
	#
	###
	#Real Part
	###
	G[:n_par/2,:nu*nv/2+1] = np.identity(n_par/2) #Fill the first half of the matrix -- Upper tri (transposed) including the centre -- with the  real param values
	G[n_par/2:,:nu*nv/2+1] = np.identity(n_par/2)[:-1][::-1] #Fill the second half of the matrix -- Lower tri (transposed) minus the centre -- with the  real param values in reverse order (reverse order done by the [::-1])
	###
	#Imag Part
	###
	G[:n_par/2,nu*nv/2+1:] = +1j*np.identity(n_par/2) #Fill the first half of the matrix -- Upper tri (transposed) including the centre -- with the  imag param values
	G[n_par/2:,nu*nv/2+1:] = -1j*np.identity(n_par/2)[:-1][::-1] #Fill the second half of the matrix -- Lower tri (transposed) minus the centre -- with the  imag param values in reverse order (reverse order done by the [::-1])
	return G





###
#1. Construct G
###
#Assumes parameters are ordered: all real then all imaginary and the parameters are the upper triangular values of the uv-plane (which describes the whole uv-plane since it is Hermitian so UV[i,j]=UV[-i,-j] (or UV[-1-i,-1-j] in python because of zero indexing).
def Construct_Hermitian_Gridding_Matrix_CosSin(nu,nv):
	#n_par=((nu*nv/2)+1)*2
	n_par=((nu*nv/2))*2 #Not including the offset term
	n_par_div2=n_par/2
	#
	G=np.zeros([nu*nv*2,n_par])
	#
	###
	#Real Part
	###
	G[:nu*nv/2,:n_par/2] = np.identity(n_par/2) #Fill the first half of the matrix -- Upper tri (transposed) including the centre -- with the  real param values
	G[nu*nv/2+1:nu*nv,:n_par/2] = np.identity(n_par/2)[::-1] #Fill the second half of the matrix -- Lower tri (transposed) minus the centre -- with the  real param values in reverse order (reverse order done by the [::-1])
	###
	#Imag Part
	###
	G[nu*nv:3*nu*nv/2,n_par/2:] = np.identity(n_par/2) #Fill the first half of the matrix -- Upper tri (transposed) including the centre -- with the  imag param values
	G[3*nu*nv/2+1:nu*nv*2,n_par/2:] = -1*np.identity(n_par/2)[::-1] #Fill the second half of the matrix -- Lower tri (transposed) minus the centre -- with the  imag param values in reverse order (reverse order done by the [::-1])
	return G








###
#1. Construct G
###
#Assumes parameters are ordered: Coarse Grid - all real then all imaginary and the parameters are the upper triangular values of the uv-plane (which describes the whole uv-plane since it is Hermitian so UV[i,j]=UV[-i,-j] (or UV[-1-i,-1-j] in python because of zero indexing) minus the lines corresponding to the 9 central pixels. Followed by Subharmonic Grid  - all real then all imaginary and the parameters are the upper triangular values of the oversampled centre 9 coords of the uv-plane.
def Construct_Hermitian_Gridding_Matrix_CosSin_SH_v4(nu,nv, U_oversampling_Factor, V_oversampling_Factor):
	n_par=(nu*nv-9) #Complete coarse grid
	#
	nu_SH=3*U_oversampling_Factor
	nv_SH=3*V_oversampling_Factor
	n_par_SH=(nu_SH*nv_SH-1) #Zero mean (ie. missing centre pix) subharmonic grid
	#
	G=np.zeros([n_par*2+n_par_SH*2,n_par+n_par_SH])
	#
	###
	#Real Part Coarse grid
	###
	G[:n_par/2,:n_par/2] = np.identity(n_par/2) #Fill the first half of the matrix -- Upper tri (transposed) including the centre -- with the  real param values
	G[n_par/2:n_par,:n_par/2] = np.identity(n_par/2)[::-1] #Fill the second half of the matrix -- Lower tri (transposed) minus the centre -- with the  real param values in reverse order (reverse order done by the [::-1])
	###
	#Real Part Subharmonic grid
	###
	G[n_par:n_par+n_par_SH/2, n_par/2:n_par/2+n_par_SH/2] = np.identity(n_par_SH/2) #Fill the first half of the matrix -- Upper tri (transposed) including the centre -- with the  real param values
	G[n_par+n_par_SH/2:n_par+n_par_SH, n_par/2:n_par/2+n_par_SH/2] = np.identity(n_par_SH/2)[::-1] #Fill the second half of the matrix -- Lower tri (transposed) minus the centre -- with the  real param values in reverse order (reverse order done by the [::-1])
	###
	#
	###
	#Imag Part Coarse grid
	###
	G[n_par+n_par_SH:3*n_par/2+n_par_SH,n_par/2+n_par_SH/2:n_par+n_par_SH/2] = np.identity(n_par/2) #Fill the first half of the matrix -- Upper tri (transposed) including the centre -- with the  imag param values
	G[3*n_par/2+n_par_SH:n_par*2+n_par_SH,n_par/2+n_par_SH/2:n_par+n_par_SH/2] = -1*np.identity(n_par/2)[::-1] #Fill the second half of the matrix -- Lower tri (transposed) minus the centre -- with the  imag param values in reverse order (reverse order done by the [::-1])
	#Imag Part Subharmonic grid
	###
	G[n_par*2+n_par_SH:n_par*2+3*n_par_SH/2, n_par+n_par_SH/2:n_par+n_par_SH] = np.identity(n_par_SH/2) #Fill the first half of the matrix -- Upper tri (transposed) including the centre -- with the  imag param values
	G[n_par*2+3*n_par_SH/2:n_par*2+n_par_SH*2,n_par+n_par_SH/2:n_par+n_par_SH] = -1*np.identity(n_par_SH/2)[::-1] #Fill the second half of the matrix -- Lower tri (transposed) minus the centre -- with the  imag param values in reverse order (reverse order done by the [::-1])
	
	return G







###
def IDFT_Array_IDFT_1D(nf, neta):
	i_f=(np.arange(nf)-nf/2).reshape(-1,1)
	i_eta=(np.arange(neta)-neta/2).reshape(1,-1)
	ExponentArray=np.exp(+2.0*np.pi*1j*( (i_eta*i_f/float(nf)) ))
	return ExponentArray/float(nf)




# ###
# def quadratic_array_linear_plus_quad_modes_only_v2(neta, nq=2):
# 	quadratic_array = np.zeros([nq,neta])+0j
# 	# quadratic_array[0] = arange(neta)
# 	# quadratic_array[1] = arange(neta)**2.0
# 	# quadratic_array[2] = 1j*arange(neta)
# 	# quadratic_array[3] = 1j*arange(neta)**2
# 	quadratic_array[0] = arange(neta)+1j*arange(neta)
# 	quadratic_array[1] = arange(neta)**2.0+1j*arange(neta)**2
# 	return quadratic_array




###
def quadratic_array_linear_plus_quad_modes_only_v2(nf, nq=2, **kwargs):
	##===== Defaults =======
	default_npl = 0
	default_nu_min_MHz = (163.0-4.0)
	default_channel_width_MHz = 0.2
	default_beta = 2.63
	
	##===== Inputs =======
	npl=kwargs.pop('npl',default_npl)
	nu_min_MHz=kwargs.pop('nu_min_MHz',default_nu_min_MHz)
	channel_width_MHz=kwargs.pop('channel_width_MHz',default_channel_width_MHz)
	beta=kwargs.pop('beta',default_beta)

	quadratic_array = np.zeros([nq,nf])+0j
	if nq==1:
		x=arange(nf)-(nf/2.)
		quadratic_array[0] = x
		if npl==1:
			###
			# This function needs modifying so that the following information is passed to it rather than being hard coded!
			###
			# nu_min_MHz = (163.0-4.0)
			# channel_width_MHz = 0.2
			# # beta_experimental_mean = 2.63+0
			# beta = 2.63
			###
			nu_array_MHz = nu_min_MHz+np.arange(float(nf))*channel_width_MHz
			m_pl = np.array([(nu_array_MHz[i_nu]/nu_min_MHz)**-beta for i_nu in range(len(nu_array_MHz))])
			quadratic_array[0] = m_pl
			print '\nLinear LW mode replaced with power-law model'
			print 'nu_min_MHz = ', nu_min_MHz
			print 'channel_width_MHz = ', channel_width_MHz
			print 'beta = ', beta, '\n'
	if nq==2:
		x=arange(nf)-(nf/2.)
		quadratic_array[0] = x
		quadratic_array[1] = x**2
		# 
		if npl==1:
			###
			# This function needs modifying so that the following information is passed to it rather than being hard coded!
			###
			# nu_min_MHz = (163.0-4.0)
			# channel_width_MHz = 0.2
			# # beta_experimental_mean = 2.63+0
			# beta = 2.63
			###
			nu_array_MHz = nu_min_MHz+np.arange(float(nf))*channel_width_MHz
			m_pl = np.array([(nu_array_MHz[i_nu]/nu_min_MHz)**-beta for i_nu in range(len(nu_array_MHz))])
			quadratic_array[1] = m_pl
			print '\nQuadratic LW mode replaced with power-law model'
			print 'nu_min_MHz = ', nu_min_MHz
			print 'channel_width_MHz = ', channel_width_MHz
			print 'beta = ', beta, '\n'
		if npl==2:
			###
			# This function needs modifying so that the following information is passed to it rather than being hard coded!
			###
			# nu_min_MHz = (163.0-4.0)
			# channel_width_MHz = 0.2
			# # beta_experimental_mean = 2.63+0
			# beta = 2.63
			###
			nu_array_MHz = nu_min_MHz+np.arange(float(nf))*channel_width_MHz
			m_pl1 = np.array([(nu_array_MHz[i_nu]/nu_min_MHz)**-beta[0] for i_nu in range(len(nu_array_MHz))])
			quadratic_array[0] = m_pl1
			print '\nLinear LW mode replaced with power-law model'
			print 'beta1 = ', beta[0], '\n'	
			m_pl2 = np.array([(nu_array_MHz[i_nu]/nu_min_MHz)**-beta[1] for i_nu in range(len(nu_array_MHz))])
			quadratic_array[1] = m_pl2
			print '\nQuadratic LW mode replaced with power-law model'
			print 'nu_min_MHz = ', nu_min_MHz
			print 'channel_width_MHz = ', channel_width_MHz
			print 'beta2 = ', beta[1], '\n'	
		# 
	if nq==3:
		x=arange(nf)-(nf/2.)
		quadratic_array[0] = x
		quadratic_array[1] = x**2
		quadratic_array[1] = x**3
		# 
		if npl==1:
			###
			# This function needs modifying so that the following information is passed to it rather than being hard coded!
			###
			# nu_min_MHz = (163.0-4.0)
			# channel_width_MHz = 0.2
			# # beta_experimental_mean = 2.63+0
			# beta = 2.63
			###
			nu_array_MHz = nu_min_MHz+np.arange(float(nf))*channel_width_MHz
			m_pl = np.array([(nu_array_MHz[i_nu]/nu_min_MHz)**-beta for i_nu in range(len(nu_array_MHz))])
			quadratic_array[1] = m_pl
			print '\nQuadratic LW mode replaced with power-law model'
			print 'nu_min_MHz = ', nu_min_MHz
			print 'channel_width_MHz = ', channel_width_MHz
			print 'beta = ', beta, '\n'
		if npl==2:
			nu_array_MHz = nu_min_MHz+np.arange(float(nf))*channel_width_MHz
			m_pl1 = np.array([(nu_array_MHz[i_nu]/nu_min_MHz)**-beta[0] for i_nu in range(len(nu_array_MHz))])
			quadratic_array[0] = m_pl1
			print '\nLinear LW mode replaced with power-law model'
			print 'beta1 = ', beta[0], '\n'	
			m_pl2 = np.array([(nu_array_MHz[i_nu]/nu_min_MHz)**-beta[1] for i_nu in range(len(nu_array_MHz))])
			quadratic_array[1] = m_pl2
			print '\nQuadratic LW mode replaced with power-law model'
			print 'nu_min_MHz = ', nu_min_MHz
			print 'channel_width_MHz = ', channel_width_MHz
			print 'beta2 = ', beta[1], '\n'	
		if npl==3:
			nu_array_MHz = nu_min_MHz+np.arange(float(nf))*channel_width_MHz
			m_pl1 = np.array([(nu_array_MHz[i_nu]/nu_min_MHz)**-beta[0] for i_nu in range(len(nu_array_MHz))])
			quadratic_array[0] = m_pl1
			print '\nLinear LW mode replaced with power-law model'
			print 'beta1 = ', beta[0], '\n'	
			m_pl2 = np.array([(nu_array_MHz[i_nu]/nu_min_MHz)**-beta[1] for i_nu in range(len(nu_array_MHz))])
			quadratic_array[1] = m_pl2
			print '\nQuadratic LW mode replaced with power-law model'
			print 'beta2 = ', beta[1], '\n'	
			m_pl3 = np.array([(nu_array_MHz[i_nu]/nu_min_MHz)**-beta[1] for i_nu in range(len(nu_array_MHz))])
			quadratic_array[2] = m_pl3
			print '\nCubic LW mode replaced with power-law model'
			print 'nu_min_MHz = ', nu_min_MHz
			print 'channel_width_MHz = ', channel_width_MHz
			print 'beta3 = ', beta[2], '\n'	
		# 
	if nq==4:
		quadratic_array[0] = arange(nf)
		quadratic_array[1] = arange(nf)**2.0
		quadratic_array[2] = 1j*arange(nf)
		quadratic_array[3] = 1j*arange(nf)**2
	return quadratic_array




###
def quadratic_array_linear_plus_quad_modes_only_v2_ZM(nf, nq=2):
	quadratic_array = np.zeros([nq,nf-1])+0j
	if nq==2:
		# x=arange(nf)-(nf/2)
		# quadratic_array[0] = x+0j*x
		# quadratic_array[1] = x**2.0+0j*x**2
		# x=arange(nf)+0.0
		x=arange(nf-1)-0.
		# x=arange(nf)-(nf/2)
		quadratic_array[0] = x
		quadratic_array[1] = x**2
		# quadratic_array[0] = x+1j*x
		# quadratic_array[1] = 1j*x**2
		# quadratic_array[1] = x**2.0+1j*x**2
	if nq==3:
		x=arange(nf-1)
		# x=arange(nf)-(nf/2)
		quadratic_array[0] = np.ones(len(x))
		quadratic_array[1] = x+1j*x
		quadratic_array[2] = x**2.0+1j*x**2
	if nq==4:
		quadratic_array[0] = arange(nf-1)
		quadratic_array[1] = arange(nf-1)**2.0
		quadratic_array[2] = 1j*arange(nf-1)
		quadratic_array[3] = 1j*arange(nf-1)**2
	return quadratic_array




###
def IDFT_Array_IDFT_1D_WQ(nf, neta, nq, los_distances_inv_pix, **kwargs):
	##===== Defaults =======
	default_npl = 0
	default_nu_min_MHz = (163.0-4.0)
	default_channel_width_MHz = 0.2
	default_beta = 2.63
	
	##===== Inputs =======
	npl=kwargs.pop('npl',default_npl)
	nu_min_MHz=kwargs.pop('nu_min_MHz',default_nu_min_MHz)
	channel_width_MHz=kwargs.pop('channel_width_MHz',default_channel_width_MHz)
	beta=kwargs.pop('beta',default_beta)
	
	i_eta=(np.arange(neta)-neta/2).reshape(1,-1)
	ExponentArray = np.exp( +2.0*np.pi*1j*(i_eta * los_distances_inv_pix) )
	ExponentArray = ExponentArray/float(nf)
	quadratic_array = quadratic_array_linear_plus_quad_modes_only_v2(nf, nq, npl=npl, nu_min_MHz=nu_min_MHz, channel_width_MHz=channel_width_MHz, beta=beta)
	print quadratic_array
	# quadratic_array = quadratic_array_linear_plus_quad_modes_only(neta)
	Exponent_plus_quadratic_array = np.hstack((ExponentArray,quadratic_array.T))
	return Exponent_plus_quadratic_array.T




###
def IDFT_Array_IDFT_1D_WQ_ZM(nf, neta, nq):
	i_f=(np.arange(nf)-nf/2)
	i_eta=(np.arange(neta)-neta/2)

	# i_f=np.delete(i_f,np.where(i_f==0))
	i_eta=np.delete(i_eta,np.where(i_eta==0))

	i_f=i_f.reshape(-1,1)
	i_eta=i_eta.reshape(1,-1)

	ExponentArray=np.exp(+2.0*np.pi*1j*( (i_eta*i_f/float(nf)) ))
	ExponentArray = ExponentArray/float(nf)
	quadratic_array = quadratic_array_linear_plus_quad_modes_only_v2_ZM(neta, nq)
	print quadratic_array
	# quadratic_array = quadratic_array_linear_plus_quad_modes_only(neta)
	Exponent_plus_quadratic_array = np.vstack((ExponentArray,quadratic_array))
	return Exponent_plus_quadratic_array.T




###
def IDFT_Array_IDFT_1D_ZM(nf, neta):
	i_eta=(np.arange(neta)-neta/2).reshape(1,-1)
	i_f=(np.arange(nf)-nf/2)
	i_f=i_f[i_f!=0].reshape(-1,1) #Remove the centre uv-pix
	ExponentArray=np.exp(+2.0*np.pi*1j*( (i_eta*i_f/float(nf)) ))
	return ExponentArray/float(nf)





#------------------------------------------
#------Construct gridding matrix-----------
#------------------------------------------
###
# Note: the gridding logic works as follows:
# - In visibility order there are 37 complex numbers per vis (in the ZM case) follwing the 1D DFT (eta -> frequency)
# - Following gridding there will be (nu*nv-1) coarse grid points per channel
# - If d1 is the data in channel order with the numbers:
# [[0, 1, 2],
#  [3, 4, 5],
#  [6, 7, 8]]
# in d1[0], however, 4 is removed by the ZM; then the order of the first values in the visibility spectra in d1.T.flatten() will be: 0,3,6,1,7,2,5,8. So, the gridder needs to grab values in the order: 0*37+i_chan, 3*37+i_chan, 6*37+i_chan, 1*37+i_chan, etc. up to 8.

# could pre-build a list with e.g. 0,3,6,1,7,2,5,8 in it (generalised to the relevant nu and nv size) for selection from, rather than generating on the fly!?

def calc_vis_selection_numbers(nu,nv):
	required_chan_order = arange(nu*nv).reshape(nu,nv)
	visibility_spectrum_order = required_chan_order.T
	grab_order_for_visibility_spectrum_ordered_to_chan_ordered_reordering = visibility_spectrum_order.argsort()
	return grab_order_for_visibility_spectrum_ordered_to_chan_ordered_reordering

def calc_vis_selection_numbers_v2d0(nu,nv):
	required_chan_order = arange(nu*nv).reshape(nu,nv)
	visibility_spectrum_order = required_chan_order.T
	r = ((np.arange(nu)-nu/2).reshape(-1,1)**2. + (np.arange(nv)-nv/2).reshape(1,-1)**2.)**0.5
	non_excluded_values_mask = r>=0.0 #No values should be masked if the mean is included
	visibility_spectrum_order_ZM = visibility_spectrum_order[non_excluded_values_mask]
	grab_order_for_visibility_spectrum_ordered_to_chan_ordered_reordering_ZM = visibility_spectrum_order_ZM.argsort()
	return grab_order_for_visibility_spectrum_ordered_to_chan_ordered_reordering_ZM

def calc_vis_selection_numbers_ZM(nu,nv):
	required_chan_order = arange(nu*nv).reshape(nu,nv)
	visibility_spectrum_order = required_chan_order.T
	r = ((np.arange(nu)-nu/2).reshape(-1,1)**2. + (np.arange(nv)-nv/2).reshape(1,-1)**2.)**0.5
	non_excluded_values_mask = r>0.5 #true for everything other than the central pixel (note r==r.T)
	visibility_spectrum_order_ZM = visibility_spectrum_order[non_excluded_values_mask]
	grab_order_for_visibility_spectrum_ordered_to_chan_ordered_reordering_ZM = visibility_spectrum_order_ZM.argsort()
	return grab_order_for_visibility_spectrum_ordered_to_chan_ordered_reordering_ZM

def calc_vis_selection_numbers_SH(nu,nv, U_oversampling_Factor=1.0, V_oversampling_Factor=1.0):
	required_chan_order = arange(nu*nv).reshape(nu,nv)
	visibility_spectrum_order = required_chan_order.T
	r = ((np.arange(nu)-nu/2).reshape(-1,1)**2. + (np.arange(nv)-nv/2).reshape(1,-1)**2.)**0.5
	non_excluded_values_mask = r>1.5 #true for everything other than the central 9 pix
	visibility_spectrum_order_ZM_coarse_grid = visibility_spectrum_order[non_excluded_values_mask]
	grab_order_for_visibility_spectrum_ordered_to_chan_ordered_reordering_ZM_coarse_grid = visibility_spectrum_order_ZM_coarse_grid.argsort()
	grab_order_for_visibility_spectrum_ordered_to_chan_ordered_reordering_ZM_SH_grid = calc_vis_selection_numbers_ZM(3*U_oversampling_Factor, 3*V_oversampling_Factor)
	return grab_order_for_visibility_spectrum_ordered_to_chan_ordered_reordering_ZM_coarse_grid, grab_order_for_visibility_spectrum_ordered_to_chan_ordered_reordering_ZM_SH_grid



def generate_gridding_matrix_vis_ordered_to_chan_ordered(nu,nv,nf):
	if p.fit_for_monopole:
		vis_grab_order = calc_vis_selection_numbers_v2d0(nu,nv)
	else:
		vis_grab_order = calc_vis_selection_numbers_ZM(nu,nv)
	vals_per_chan = vis_grab_order.size

	gridding_matrix_vis_ordered_to_chan_ordered = np.zeros([vals_per_chan*(nf), vals_per_chan*(nf)])
	###
	for i in range(nf):
		for j, vis_grab_val in enumerate(vis_grab_order):
				row_number = (i*vals_per_chan)+j
				grid_pix = i+vis_grab_val*(nf) #pixel to grab from vis-ordered vector and place as next chan-ordered value
				print i,j,vis_grab_val,row_number,grid_pix
				gridding_matrix_vis_ordered_to_chan_ordered[row_number, grid_pix]=1
	return gridding_matrix_vis_ordered_to_chan_ordered



def generate_gridding_matrix_vis_ordered_to_chan_ordered_ZM(nu,nv,nf):
	if p.fit_for_monopole:
		vis_grab_order = calc_vis_selection_numbers_v2d0(nu,nv)
	else:
		vis_grab_order = calc_vis_selection_numbers_ZM(nu,nv)
	vals_per_chan = vis_grab_order.size

	gridding_matrix_vis_ordered_to_chan_ordered = np.zeros([vals_per_chan*(nf-1), vals_per_chan*(nf-1)])
	###
	for i in range(nf-1):
		for j, vis_grab_val in enumerate(vis_grab_order):
				row_number = (i*vals_per_chan)+j
				grid_pix = i+vis_grab_val*(nf-1) #pixel to grab from vis-ordered vector and place as next chan-ordered value
				print i,j,vis_grab_val,row_number,grid_pix
				gridding_matrix_vis_ordered_to_chan_ordered[row_number, grid_pix]=1
	return gridding_matrix_vis_ordered_to_chan_ordered



def generate_gridding_matrix_vis_ordered_to_chan_ordered_WQ(nu,nv,nf):
	"""
	Re-order matrix from vis-ordered to chan-ordered and place Fourier modes at the top and quadratic modes at the bottom.
	"""
	if p.fit_for_monopole:
		vis_grab_order = calc_vis_selection_numbers_v2d0(nu,nv)
	else:
		vis_grab_order = calc_vis_selection_numbers_ZM(nu,nv)
	vals_per_chan = vis_grab_order.size
	Fourier_vals_per_chan = vis_grab_order.size
	quadratic_vals_per_chan = 2

	gridding_matrix_vis_ordered_to_chan_ordered = np.zeros([vals_per_chan*(nf+2), vals_per_chan*(nf+2)])
	###
	for i in range(nf):
		for j, vis_grab_val in enumerate(vis_grab_order):
				row_number = (i*Fourier_vals_per_chan)+j
				grid_pix = i+vis_grab_val*(nf+2) #pixel to grab from vis-ordered vector and place as next chan-ordered value
				print i,j,vis_grab_val,row_number,grid_pix
				gridding_matrix_vis_ordered_to_chan_ordered[row_number, grid_pix]=1
	print 'Hello'
	for j, vis_grab_val in enumerate(vis_grab_order):
		for i in range(2):
				n_fourier_modes = (nf*Fourier_vals_per_chan) #Place quadratic modes after all of the Fourier modes in the resulting vector 
				row_number = n_fourier_modes+(j*quadratic_vals_per_chan)+i #Place quadratic modes after all of the Fourier modes in the resulting vector
				grid_pix = nf+i+vis_grab_val*(nf+2) #pixel to grab from vis-ordered vector and place as next chan-ordered value
				print i,j,vis_grab_val,row_number,grid_pix
				gridding_matrix_vis_ordered_to_chan_ordered[row_number, grid_pix]=1
	return gridding_matrix_vis_ordered_to_chan_ordered




# nu=3
# nv=3
# nf=6
# gridding_matrix_vis_ordered_to_chan_ordered_ZM = generate_gridding_matrix_vis_ordered_to_chan_ordered_ZM(nu,nv,nf)

# ###
# # Test 3D with 1D zero mean
# ###
# a=np.arange((nf-1)*nu*nv).reshape((nf-1),nu,nv)
# a1=a.T.flatten()
# remove_indices = [np.where(a1==x) for x in (((nu*nv)/2)+arange(nf-1)*(nu*nv))]
# a2=np.delete(a1, remove_indices)

# ga2 = np.dot(gridding_matrix_vis_ordered_to_chan_ordered_ZM, a2.reshape(-1,1))




# correct_order_zm = np.delete(arange(nf*nu*nv),[np.where(arange(nf*nu*nv)==x) for x in (((nu*nv)/2)+arange(nf)*(nu*nv))])
# print abs(ga2.flatten()-correct_order_zm).max() #If the printout is 0.0 then gridding_matrix_vis_ordered_to_chan_ordered is correctly constructed! 






# nu=5
# nv=5
# nf=6
# gridding_matrix_vis_ordered_to_chan_ordered_WQ = generate_gridding_matrix_vis_ordered_to_chan_ordered_WQ(nu,nv,nf)

# ###
# # Test 3D with 1D zero mean
# ###
# a=np.arange(nf*nu*nv).reshape(nf,nu,nv)
# a1=a.T.flatten()
# remove_indices = [np.where(a1==x) for x in (((nu*nv)/2)+arange(nf)*(nu*nv))]
# a2=np.delete(a1, remove_indices)
# a2 = a2.reshape(-1,1)
# a3 = a2.reshape(-1,6)
# a4 = np.array([np.concatenate((x,(100+2*i,101+2*i))) for i, x in enumerate(a3)])

# ga2 = np.dot(gridding_matrix_vis_ordered_to_chan_ordered_WQ, a4.reshape(-1,1))




# # correct_order_zm = np.delete(arange(nf*nu*nv),[np.where(arange(nf*nu*nv)==x) for x in (((nu*nv)/2)+arange(nf)*(nu*nv))])
# # print abs(ga2.flatten()-correct_order_zm).max() #If the printout is 0.0 then gridding_matrix_vis_ordered_to_chan_ordered is correctly constructed! 






