

#Gamma, k defined by power-law source-count distribution: n(S)=kS**(-Gamma), q=S_0/Sigma_c ie. required S/N ratio to constitute a detection
def Calc_Confusion_Sigma(Beam_FWHM_Rad, Gamma, k, q, **kwargs):
	#
	# Omega_b=( (pi*(Beam_FWHM_Rad**2.))/(4.*log(2.)) )
	Omega_b=( ((Beam_FWHM_Rad**2.)) )
	print 'Omega_b', Omega_b
	Omega_e=( Omega_b/(Gamma-1) )
	print 'Omega_e', Omega_e
	Confusion_Sigma=( (((q**(3-Gamma))/(3.-Gamma))**(1./(Gamma-1.))) * ((k*Omega_e)**(1./(Gamma-1.))) )
	print 'Confusion_Sigma', Confusion_Sigma
	return Confusion_Sigma



#HERA beam width and source-count from S16 
Beam_FWHM_Rad=1./(87./(3.e8/126.e6))
Gamma=2.51
k=4.0e3
q=5.
Confusion_Sigma=Calc_Confusion_Sigma(Beam_FWHM_Rad, Gamma, k, q)






#HERA beam width and source-count from for S18a (now with maximum 40 m baselines)
# Beam_FWHM_Rad=1./(87./(3.e8/163.e6))
Beam_FWHM_Rad=1./(40./(3.e8/163.e6))
Gamma=2.51
k=4.0e3*(163./151)**-0.82
q=5.
Confusion_Sigma=Calc_Confusion_Sigma(Beam_FWHM_Rad, Gamma, k, q)




#HERA beam width and source-count for 225 MHz in S18b
Beam_FWHM_Rad=1./(87./(3.e8/225.e6))
Gamma=2.51
k=4.0e3*(225./151)**-0.82
q=5.
Confusion_Sigma=Calc_Confusion_Sigma(Beam_FWHM_Rad, Gamma, k, q)







#HERA beam width and source-count from for S18a (now with H331 maximum baseline length of 20*14.5 = 290 m)
# Beam_FWHM_Rad=1./(87./(3.e8/163.e6))
Beam_FWHM_Rad=1./(290./(3.e8/163.e6))
Gamma=2.51
k=4.0e3*(163./151)**-0.82
q=5.
Confusion_Sigma=Calc_Confusion_Sigma(Beam_FWHM_Rad, Gamma, k, q)




