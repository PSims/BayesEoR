import numpy as np

###
# Ali 2015 power spectrum
###
data_Ali = np.loadtxt('Ali2015_PS.png.unknown.PSpoints')
data_Ali1 = data_Ali
x_Ali = data_Ali1[1::3][:,0]
y_Ali = data_Ali1[1::3][:,1]
y_Ali_err_upper = data_Ali1[2::3][:,1]-y_Ali
y_Ali_err_lower = y_Ali-data_Ali1[0::3][:,1]
y_Ali_lower = np.maximum(10.0, y_Ali - y_Ali_err_lower)
y_Ali_lower[y_Ali_lower==10.0] = 1.e-2
y_Ali_err_lower = y_Ali-y_Ali_lower
y_Ali_err_lower[0]*=2
y_Ali_err=[y_Ali_err_lower, y_Ali_err_upper]

import pylab as P
fig,ax=P.subplots()
ax.errorbar(x_Ali,y_Ali,yerr=[y_Ali_err_lower, y_Ali_err_upper], fmt='+')
ax.set_yscale('log')
ax.set_ylim(bottom=1.0, top=1.e6)
fig.show()

###
# EoR signal 
###
data_EoR = np.loadtxt('Ali2015_PS.png.unknown.EoRpoints')
# data_EoR = np.loadtxt('Ali2015_PS.png.unknown')
data_EoR1 = data_EoR
x_EoR = data_EoR1[:,0]
y_EoR = data_EoR1[:,1]

import pylab as P
fig,ax=P.subplots()
ax.errorbar(x_Ali,y_Ali,yerr=[y_Ali_err_lower, y_Ali_err_upper], fmt='+')
ax.errorbar(x_EoR,y_EoR, fmt='--')
ax.set_yscale('log')
ax.set_ylim(bottom=1.0, top=1.e6)
fig.show()


###
# Ali 2015 noise estimate
###
data_Noise = np.loadtxt('Ali2015_PS.png.unknown.Noisepoints')
# data_Noise = np.loadtxt('Ali2015_PS.png.unknown')
data_Noise1 = data_Noise
x_Noise = data_Noise1[:,0]
y_Noise = data_Noise1[:,1]

import pylab as P
fig,ax=P.subplots()
ax.errorbar(x_Ali,y_Ali,yerr=[y_Ali_err_lower, y_Ali_err_upper], fmt='+', color='green')
ax.errorbar(x_EoR,y_EoR, fmt='--', color='purple')
ax.errorbar(x_Noise,y_Noise, fmt='--', color='orange')
ax.set_yscale('log')
ax.set_ylim(bottom=1.0, top=1.e6)
ax.legend(['A15, $z=8.4$', 'sim EoR', 'A15 noise', ])
ax.set_xlabel('$k\ [h\mathrm{Mpc^{-1}}]$')
ax.set_ylabel('$\\Delta^2(k)\ [\mathrm{mK^{2}}]$')
fig.show()



###
# Patil 2017 power spectrum 10.1
###
data_Patil_10d1 = np.loadtxt('Patil2017_PS.png.unknown.PSpoints10d1')
x_Patil_10d1 = data_Patil_10d1[1::3][:,0]
y_Patil_10d1 = data_Patil_10d1[1::3][:,1]
y_Patil_10d1_err_upper = data_Patil_10d1[2::3][:,1]-y_Patil_10d1
y_Patil_10d1_err_lower = y_Patil_10d1-data_Patil_10d1[0::3][:,1]
y_Patil_10d1_lower = np.maximum(10.0, y_Patil_10d1 - y_Patil_10d1_err_lower)
y_Patil_10d1_lower[y_Patil_10d1_lower==10.0] = 1.e-2
y_Patil_10d1_err_lower = y_Patil_10d1-y_Patil_10d1_lower
y_Patil_10d1_err=[y_Patil_10d1_err_lower, y_Patil_10d1_err_upper]

import pylab as P
fig,ax=P.subplots()
ax.errorbar(x_Patil_10d1,y_Patil_10d1,yerr=[y_Patil_10d1_err_lower, y_Patil_10d1_err_upper], fmt='+')
ax.set_yscale('log')
ax.set_ylim(bottom=1.0, top=1.e6)
fig.show()



###
# Patil 2017 power spectrum 9.1
###
data_Patil_9d1 = np.loadtxt('Patil2017_PS.png.unknown.PSpoints9d1')
x_Patil_9d1 = data_Patil_9d1[1::3][:,0]
y_Patil_9d1 = data_Patil_9d1[1::3][:,1]
y_Patil_9d1_err_upper = data_Patil_9d1[2::3][:,1]-y_Patil_9d1
y_Patil_9d1_err_lower = y_Patil_9d1-data_Patil_9d1[0::3][:,1]
y_Patil_9d1_lower = np.maximum(10.0, y_Patil_9d1 - y_Patil_9d1_err_lower)
y_Patil_9d1_lower[y_Patil_9d1_lower==10.0] = 1.e-2
y_Patil_9d1_err_lower = y_Patil_9d1-y_Patil_9d1_lower
y_Patil_9d1_err=[y_Patil_9d1_err_lower, y_Patil_9d1_err_upper]

import pylab as P
fig,ax=P.subplots()
ax.errorbar(x_Patil_9d1,y_Patil_9d1,yerr=[y_Patil_9d1_err_lower, y_Patil_9d1_err_upper], fmt='+')
ax.set_yscale('log')
ax.set_ylim(bottom=1.0, top=1.e6)
fig.show()



###
# Patil 2017 power spectrum 8.3
###
data_Patil_8d3 = np.loadtxt('Patil2017_PS.png.unknown.PSpoints8d3')
x_Patil_8d3 = data_Patil_8d3[1::3][:,0]
y_Patil_8d3 = data_Patil_8d3[1::3][:,1]
y_Patil_8d3_err_upper = data_Patil_8d3[2::3][:,1]-y_Patil_8d3
y_Patil_8d3_err_lower = y_Patil_8d3-data_Patil_8d3[0::3][:,1]
y_Patil_8d3_lower = np.maximum(10.0, y_Patil_8d3 - y_Patil_8d3_err_lower)
y_Patil_8d3_lower[y_Patil_8d3_lower==10.0] = 1.e-2
y_Patil_8d3_err_lower = y_Patil_8d3-y_Patil_8d3_lower
y_Patil_8d3_err=[y_Patil_8d3_err_lower, y_Patil_8d3_err_upper]

import pylab as P
fig,ax=P.subplots()
ax.errorbar(x_Patil_8d3,y_Patil_8d3,yerr=[y_Patil_8d3_err_lower, y_Patil_8d3_err_upper], fmt='+')
ax.set_yscale('log')
ax.set_ylim(bottom=1.0, top=1.e6)
fig.show()





import pylab as P
fig,ax=P.subplots()
ax.errorbar(x_Ali,y_Ali,yerr=[y_Ali_err_lower, y_Ali_err_upper], fmt='+', color='green')
ax.errorbar(x_EoR,y_EoR, fmt='--', color='purple')
ax.errorbar(x_Noise,y_Noise, fmt='--', color='orange')
ax.errorbar(x_Patil_10d1,y_Patil_10d1,yerr=[y_Patil_10d1_err_lower, y_Patil_10d1_err_upper], fmt='+', color='black')
ax.errorbar(x_Patil_9d1,y_Patil_9d1,yerr=[y_Patil_9d1_err_lower, y_Patil_9d1_err_upper], fmt='+', color='darkgrey')
ax.errorbar(x_Patil_8d3,y_Patil_8d3,yerr=[y_Patil_8d3_err_lower, y_Patil_8d3_err_upper], fmt='+', color='grey')
ax.set_yscale('log')
ax.set_ylim(bottom=1.0, top=1.e6)
ax.legend(['A15, $z=8.4$', 'sim EoR', 'A15 noise', 'P17,  $z=10.1$', 'P17,  $z=9.1$', 'P17,  $z=8.3$', ])
ax.set_xlabel('$k\ [h\mathrm{Mpc^{-1}}]$')
ax.set_ylabel('$\\Delta^2(k)\ [\mathrm{mK^{2}}]$')
fig.show()







