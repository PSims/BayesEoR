import argparse
import BayesEoR.Params.params as p




def BayesEoRParser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-nq", "--nq", help="Number of LWM basis vectors (0-2)", default=2, type=int)
	parser.add_argument("-beta", "--beta", help="Power law spectral index used in data model", default=[2.63,2.82])
	parser.add_argument('--sigma_val',
					    type = float,
					    help = "RMS of noise in the visibilities (overwrites sigma in driver file).")
	parser.add_argument('--HERA_data_path',
						type = str,
						help = "Path to data file for analysis.")
	parser.add_argument('--beam_type',
						type = str,
						default = 'Gaussian',
						help = "Can be either 'Gaussian' or 'Uniform'. Defaults to 'Gaussian'.")
	parser.add_argument('--beam_peak_amplitude',
					    type = float,
					    help = "Peak amplitude of the beam.")
	parser.add_argument('--FWHM_deg_at_ref_freq_MHz',
					    type = float,
					    help = "FWHM of beam at the reference frequency in degrees.")
	parser.add_argument('--PB_ref_freq_MHz',
					    type = float,
					    help = "Reference frequency for primary beam in MHz.")

	args = parser.parse_args() #Parse command line arguments
	return args


args = BayesEoRParser()
print args
print args.nq
print args.nq*2




#--------------------------------------------
# Set analysis parameters
#--------------------------------------------
# Model Params
args = p.BayesEoRParser()
print args
npl = p.npl
print 'p.beta', p.beta
if args.beta:
	if type(args.beta)==str:
		if args.beta.count('[') and args.beta.count(']'):
			p.beta = map(float, args.beta.replace('[','').replace(']','').split(',')) #Overwrite parameter file beta with value chosen from the command line if it is included
			npl = len(p.beta) #Overwrites quadratic term when nq=2, otherwise unused.
		else:
			p.beta = float(args.beta) #Overwrite parameter file beta with value chosen from the command line if it is included 
			npl = 1
	elif type(args.beta)==list:
		p.beta = args.beta #Overwrite parameter file beta with value chosen from the command line if it is included 
		npl = len(p.beta)
	else:
		print 'No value for betas given, using defaults.'

print 'args.beta', args.beta
print 'p.beta', p.beta
print 'args.nq', args.nq



