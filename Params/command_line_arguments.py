import argparse
import params as p


def BayesEoRParser():
	parser = argparse.ArgumentParser()
	parser.add_argument("-nq", "--nq",
						type = int,
						help="Number of LWM basis vectors (0-2)")
	parser.add_argument("-beta", "--beta",
						help="Power law spectral index used in data model")
	parser.add_argument('--sigma',
					    type = float,
					    help = "RMS of the visibility noise.")
	parser.add_argument('--HERA_data_path',
						type = str,
						help = "Path to data file for analysis.")
	parser.add_argument('--beam_type',
						type = str,
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


def update_params_with_command_line_arguments():
	args = BayesEoRParser()
	cla_keys = [key for key in args.__dict__.keys() if not (key[0]=='_' and key[-1]=='_')]
	params_keys = [key for key in p.__dict__.keys() if not (key[0]=='_' and key[-1]=='_')]

	for key in cla_keys:
		if not args.__dict__[key] == None:
			if key == 'beta':
				if type(args.beta)==str:
					if args.beta.count('[') and args.beta.count(']'):
						p.beta = map(float, args.beta.replace('[','').replace(']','').split(',')) #Overwrite parameter file beta with value chosen from the command line if it is included
						p.npl = len(p.beta) #Overwrites quadratic term when nq=2, otherwise unused.
					else:
						p.beta = float(args.beta) #Overwrite parameter file beta with value chosen from the command line if it is included 
						p.npl = 1
					print 'Overwriting params p.{} = {} with command line argument {} = {}'.format(key, p.__dict__[key], key, args.__dict__[key])
				else:
					print 'No value for betas given, using defaults.'
			else:
				if key in params_keys:
					print 'Overwriting params p.{} = {} with command line argument {} = {}'.format(key, p.__dict__[key], key, args.__dict__[key])
				else:
					print 'Setting params p.{} = {}'.format(key, args.__dict__[key])
				p.__dict__[key] = args.__dict__[key]





