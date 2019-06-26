from mpi4py import MPI
comm = MPI.COMM_WORLD


import dyPolyChord.python_likelihoods as likelihoods  # Import some example python likelihoods
import dyPolyChord.python_priors as priors  # Import some example python priors
import dyPolyChord.pypolychord_utils
import dyPolyChord


# Definte the distribution to sample (likelihood, prior, number of dimensions)
ndim = 10
likelihood = likelihoods.Gaussian(sigma=1.0)
prior = priors.Gaussian(sigma=10.0)

# Make a callable for running PolyChord
my_callable = dyPolyChord.pypolychord_utils.RunPyPolyChord(
    likelihood, prior, ndim)

# Specify sampler settings (see run_dynamic_ns.py documentation for more details)
dynamic_goal = 1.0  # whether to maximise parameter estimation or evidence accuracy. 
ninit = 100          # number of live points to use in initial exploratory run.
nlive_const = 500   # total computational budget is the same as standard nested sampling with nlive_const live points. 
settings_dict = {'file_root': 'gaussian',
                 'base_dir': 'chains',
                 'seed': 1}

# # Run dyPolyChord
# dyPolyChord.run_dypolychord(my_callable, dynamic_goal, settings_dict,
#                             ninit=ninit, nlive_const=nlive_const)


# Run dyPolyChord with MPI
dyPolyChord.run_dypolychord(my_callable, dynamic_goal, settings_dict,
                            ninit=ninit, nlive_const=nlive_const, comm=comm)

