"""
Preprocess a pyuvdata compatible dataset and get it into the format
expected by BayesEoR which is a 1d np.ndarray with shape

    (ntimes * nfreqs * nbls,)

The data are ordered first by baseline, then frequency, then time such
that the first nbls entries in the data vector are the visibilities
for all baselines at the zeroth frequency channel, the next nbls entries
are the visibilities for all baselines at the first frequency channel,
etc.  Correspondingly, the first nfreqs * nbls entries are the
visibilities for all frequencies and baselines at the zeroth time
integration.

NOTE: Currently only working for simulated healvis datasets and will
      require modification to process HERA data again.  Specifically,
      in the creation of the noise estimate and the weighted averaging.
"""

import BayesEoR # only need the __file__ attribute
import numpy as np
import pickle
import copy
import os
import sys
import glob
import optparse

# import BayesEoR.Params.params as p
import pyuvdata.utils as uvutils

from pathlib import Path
from datetime import datetime
from matplotlib.gridspec import GridSpec
from hera_cal.io import HERAData
from pyuvdata import UVData
from astropy.time import Time
from scipy import sparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

plt.rcParams.update({'font.size': 16, 'figure.figsize': (12, 8)})
DEFAULT_SAVE_DIR = str(Path(BayesEoR.__file__).parent
                       / 'scripts')

# ----------------- Option Parser -----------------
o = optparse.OptionParser()

o.add_option(
    '--data_path',
    type=str,
    help='Path to the pyuvdata compatible data file for preprocessing.'
    )

o.add_option(
    '--filename',
    type=str,
    help='Filename in opts.data_path to use for preprocessing.'
    )

o.add_option(
    '--calfits_file',
    type=str,
    help='Filename in opts.data_path to use for calibration.'
    )

o.add_option(
    '--save_model',
    action='store_true',
    help='If passed, save the generated uvw and redundancy models.'
    )

o.add_option(
    '--inst_model_dir',
    type=str,
    default=None,
    help='Path to the BayesEoR/Instrument_Model directory.'
    )

o.add_option(
    '--uniform_red_model',
    action='store_true',
    help='If passed, replace the redundancy model'
         'with a uniform model (all ones).'
    )

o.add_option(
    '--plot_intermediate',
    action='store_true',
    help='If passed, produce plots showing baseline reordering and '
         'the redundancy model in the uv-plane.'
    )

o.add_option(
    '--plot_data',
    action='store_true',
    help='If passed, plot data for all baselines kept in model.'
    )

o.add_option(
    '--start_freq_MHz',
    type=float,
    help=('Starting frequency in MHz from which 76 right-adjacent '
          'frequency channels will be extracted. Default is 150 MHz.'),
    default=150.0
    )

o.add_option(
    '--save_data',
    action='store_true',
    dest='save',
    default=True
    )

o.add_option(
    '--no_save_data',
    action='store_false',
    dest='save',
    help="If passed, don't save data."
    )

o.add_option(
    '--ant_str',
    type=str,
    help='If passed, keep only baselines specified by ant_str '
         'according to UVData.select syntax.'
    )

o.add_option(
    '--flag_bad_bls',
    action='store_true',
    help='If passed, ignore baselines which are fully flagged.'
    )

o.add_option(
    '--single_bls',
    action='store_true',
    help='If passed, create data files for each baseline.  '
         'If passed with --ant_str, only make data files '
         'for the baselines contained in --ant_str.')

o.add_option(
    '--bl_type',
    type=str,
    help='Baseline type string for selecting from data.  '
         'Given as a {baseline_length}_{orientation}. '
         'For example, to keep 14.6 meter EW baselines --bl_type=14d6_EW.'
    )

o.add_option(
    '--bl_cutoff_m',
    type=float,
    default=29.3,
    help='Baseline cutoff length in meters.  Any baselines in the raw dataset'
         ' with |b| > <bl_cutoff_m> will be excluded from the written data.'
    )

o.add_option(
    '--all_bl_noise',
    action='store_true',
    help='If passed, generate noise estimate from all '
         'baselines within a redundant group.'
    )

o.add_option(
    '--clobber',
    action='store_true',
    help='If passed, clobber existing data file(s).'
    )

o.add_option(
    '--save_dir',
    type=str,
    default=DEFAULT_SAVE_DIR,
    help='Filepath in which the data will be saved. '
         'Defaults to the BayesEoR/scripts directory.'
    )

o.add_option(
    '--no_phase',
    action='store_true',
    help='If passed, do not phase data.  Otherwise, data is phased '
           'to the central time step.'
    )

opts, args = o.parse_args(sys.argv[1:])
print(o.values)


# ----------------- Functions -----------------

def elementwise_avg(*args):
    """
    Returns the elementwise average of a set of np.ndarrays.

    Parameters
    ----------
    args : sequence of ndarrays
        Sequence of np.ndarray objects with identical shapes.

    Returns
    -------
    avg : np.ndarray
        Elementwise average.
    """
    nargs = len(args)
    args_sum = np.zeros_like(args[0])
    for i in range(nargs):
        args_sum += args[i]
    avg = args_sum / float(nargs)
    return avg


def weighted_avg_and_std(values, weights):
    """
    Return the weighted standard deviation.

    Parameters
    ----------
    values : np.ndarray
        Input array.
    weights : np.ndarray
        Weights of each element in values.
        Must have the same shape as values.

    Returns
    -------
    average : np.ndarray or float
        Weighted average of values.
        If values is a 1d array, will return a single float.
        Otherwise returns the weighted average along the zeroth axis.
    stddev : np.ndarray or float
        Weighted standard deviation of values.
        If values is a 1d array, will return a single float.
        Otherwise returns the weighted standard deviation
        along the zeroth axis.
    """
    average = np.average(values, axis=0, weights=weights)
    variance = np.average(
        (values-average) * (values-average).conj(),
        axis=0,
        weights=weights)
    return average, np.sqrt(variance)


def Jy_to_Kstr(data_array, frequencies):
    """
    Convert visibilities from units of Janskys to Kelvin steradians.

    Parameters
    ----------
    data_array : np.ndarray
        Array of visibility data in units of Janskys.
    frequencies : 1d np.ndarray
        Array of frequencies for data contained in data_array
        in units of Hertz.
    """
    # Tile frequencies to match shape of data=(nblts, nfreqs)
    freq_array = np.tile(
            frequencies,
            data_array.shape[0]
        ).reshape((data_array.shape[0], len(frequencies)))

    # CGS unit constants
    c = 29979245800.0 # cm / s
    kb = 1.380658e-16 # erg / K
    erg_to_Jy = 1.0e23 # Jy / erg
    conversion_array = c**2/(2*freq_array**2*kb*erg_to_Jy)

    return data_array * conversion_array


# ----------------- Main -----------------

# File information
data_path = opts.data_path
filename = opts.filename
phase_data = True
if opts.no_phase:
    phase_data = False

# Load data
uvd = UVData()
print('Reading data from %s...\n' %(data_path + filename))
uvd.read(data_path + filename)

times_to_keep = np.unique(uvd.time_array)

# Perform initial select
print('-'*60)
print('Starting initial select at {}'.format(datetime.utcnow()), end='\n\n')

# Remove autocorrelations
print('Removing autocorrelations...\n')
uvd.select(ant_str='cross')

# Keep only baselines with |b| < opts.bl_cutoff_m meters
baseline_lengths = np.sqrt(uvd.uvw_array[:, 0]**2 + uvd.uvw_array[:, 1]**2)
inds = np.where(baseline_lengths <= opts.bl_cutoff_m)[0]
print('Selecting only baselines <= {} meters:'.format(opts.bl_cutoff_m))
print('-'*32)
print('Shape before select:', uvd.data_array.shape)
print('Nbls before select:', uvd.Nbls)
print('Ntimes before select:', uvd.Ntimes)

# select to |b| < opts.bl_cutoff_m meters and half the times
uvd.select(blt_inds=inds, times=times_to_keep)

print('\nConjugating baselines to u > 0 convention.')
uvd.conjugate_bls(convention='u>0', uvw_tol=1.0)

# # Select only good baselines and frequencies
# uvd.select(ant_str=good_bls_str,
#            frequencies=uvd.freq_array[0, freq_inds])

print('Nbls after select:', uvd.Nbls)
print('Ntimes after select:', uvd.Ntimes)
print('Shape after select:', uvd.data_array.shape)
print('-'*32)
print('')

print('Initial select finished at {}'.format(datetime.utcnow()))
print('-'*60, end='\n\n')

DEFAULT_SAVE_DIR = __file__


def data_processing(
        uvd_select,
        opts,
        filename,
        save_dir=DEFAULT_SAVE_DIR,
        inst_model_dir=None,
        uvd_all_bls=None):

    # ------------------- DATA PREPROCESSING -------------------

    outfile = filename.replace(
        'uvh5',
        'start_freq_{:.2f}_Nbls_{}_'.format(
            opts.start_freq_MHz,
            uvd_select.Nbls
            )
        + 'weighted_average_phased_flattened_mK_sr.npy'
        )
    if not phase_data:
        outfile = outfile.replace('phased', 'unphased')

    if uvd_select.Nbls == 1:
        antnums = uvd_select.baseline_to_antnums(uvd_select.baseline_array[0])
        outfile = outfile.replace(
            'Nbls_{}_'.format(uvd_select.Nbls),
            'Nbls_{}_ants_{}_{}_'.format(
                uvd_select.Nbls, antnums[0], antnums[1]
                )
            )
    elif opts.bl_type:
        outfile = outfile.replace(
            'Nbls_{}_'.format(uvd_select.Nbls),
            'Nbls_{}_{}_'.format(uvd_select.Nbls, opts.bl_type)
            )
    # What about the case where I only keep certain baselines within a
    # redundant baseline type? Do I need some sort of unique identifier
    # for chosen baselines when I choose two separate sets of Nbls?

    # Check if data already exists in save_dir and return or clobber it
    if (
            os.path.exists(os.path.join(save_dir, outfile))
            and not opts.clobber and not opts.save_model
    ):
        print('Data already exists at {}'.format(
            os.path.join(save_dir, outfile))
            )
        return

    # Phase data to central time step
    # Create a copy of the object which can be phased
    uvd = copy.deepcopy(uvd_select)
    # Create a copy of uvd_select to be phased and compressed
    uvd_comp = uvd_select.compress_by_redundancy(inplace=False)
    if phase_data:
        # Create a copy of uvd_select which is unphased for uvw model
        uvd_comp_unphased = copy.deepcopy(uvd_comp)

    # Create a copy of uvd_select to be
    # phased and used as the phasor vector
    uvd_comp_phasor = copy.deepcopy(uvd_comp)
    # Modify data array of phasor uvdataobject to get phase vector
    phasor_array = np.ones(uvd_comp_phasor.data_array.shape) + 0j
    uvd_comp_phasor.data_array = phasor_array
    time_to_phase = np.unique(uvd_select.time_array)[uvd.Ntimes // 2]
    uvd_comp_phasor.phase_to_time(Time(time_to_phase, format='jd'))

    if phase_data:
        print('Phasing data to central time step')
        uvd.phase_to_time(Time(time_to_phase, format='jd'))
        uvd_comp.phase_to_time(Time(time_to_phase, format='jd'))
        if opts.all_bl_noise:
            uvd_all_bls.phase_to_time(Time(time_to_phase, format='jd'))

    # Average over redundant baselines

    # Containers for averaged data and noise estimates
    data_array_shape = uvd_comp.data_array.shape

    # Shrink frequency axis by two for averraging
    # This might need to be updated when pyuvdata switches to
    # flexible spws (will remove the spw axis from UVData.data_array
    data_array_phased_avg = np.zeros((data_array_shape[0],
                                      data_array_shape[2]), dtype='complex128')
    data_array_phasor_avg = np.zeros((data_array_shape[0],
                                      data_array_shape[2]), dtype='complex128')

    # Get redundancy info
    baseline_groups, vec_bin_centers, lengths = uvd_select.get_redundancies()

    # Create averaged data arrays for all baselines in a redundant group
    for i_bl, bl_group in enumerate(baseline_groups):
        bl_group_data_container = []
        bl_group_nsamples_container = []

        # Collect data from each baseline group
        for bl in bl_group:
            # Get data for baseline bl
            data = uvd.get_data(bl)

            # Average data over adjacent frequency channels
            bl_group_data_container.append(data)

            # Get nsamples for bl data for weighted
            # average and noise estimate
            nsamples = uvd.get_nsamples(bl)
            bl_group_nsamples_container.append(nsamples)

        # Cast lists to arrays for easier manipulation
        bl_group_data_container = np.array(bl_group_data_container)
        bl_group_nsamples_container = np.array(bl_group_nsamples_container)

        # Pull phasor data
        data_phasor = uvd_comp_phasor.get_data(bl_group[0])

        # Estimate noise for each baseline group

        # compute the weighted average and standard
        # deviation from data and nsamples
        # avg_data, stddev_data = weighted_avg_and_std(
        #     bl_group_data_container, bl_group_nsamples_container)

        # For now, I'm using a perfectly redundant instrument model
        # so every baseline has numerically identical data
        avg_data = bl_group_data_container[0]

        # Add averaged data and noise estimates
        # to their corresponding arrays
        arr_inds = [i_bl * uvd.Ntimes, (i_bl + 1) * uvd.Ntimes]
        # data_array_phased_avg[:ntimes] contains the data for a
        # single redundant baseline group across all frequencies
        # and times with shape (ntimes, nfreqs)
        data_array_phased_avg[arr_inds[0] : arr_inds[1]] = avg_data
        data_array_phasor_avg[arr_inds[0] : arr_inds[1]] = data_phasor

    # Reorder / flatten data

    # Convert data & noise arrays to mK sr from Jy
    frequencies = uvd.freq_array[0]
    data_array_phased_avg = Jy_to_Kstr(
        data_array_phased_avg, frequencies
        )
    data_array_phased_avg *= 1.0e3 # K sr to mK sr

    # Double the bl axis size to account for each redundant
    # baseline group and its conjugate
    data_array_reordered = np.zeros((data_array_phased_avg.shape[0]*2,
                                     data_array_phased_avg.shape[1]),
                                    dtype='complex128')
    phasor_array_reordered = np.zeros((data_array_phasor_avg.shape[0]*2,
                                       data_array_phasor_avg.shape[1]),
                                      dtype='complex128')

    # Reshape data_array to (nbls
    for i_t in range(uvd_comp.Ntimes):
        # Get data for every baseline at the i_t-th time index
        # data has shape (nbls, nfreqs)
        data = data_array_phased_avg[i_t::uvd_comp.Ntimes]
        phasor = data_array_phasor_avg[i_t::uvd_comp.Ntimes]
        # Indices in the reordered array
        # accounting for mirroring of baselines
        inds = [i_t*2*uvd_comp.Nbls, (i_t + 1)*2*uvd_comp.Nbls]
        # Fill reordered data array with the (u, v)
        # data and mirrored (-u, -v) data
        # data_array_reordered[inds[0]:inds[1]] contains the
        # data for 2*nbls baselines across all frequencies at the
        # i_t-th time index with shape (2*nbls, nfreqs)
        data_array_reordered[inds[0]:inds[1]] = np.vstack(
                (data,
                 data.conjugate())
            )
        phasor_array_reordered[inds[0]:inds[1]] = np.vstack(
                (phasor,
                 phasor.conjugate())
            )

    # Flatten data in time -> frequency -> baseline ordering
    data_array_flattened = np.zeros(data_array_reordered.size,
                                    dtype='complex128')
    phasor_array_flattened = np.zeros(phasor_array_reordered.size,
                                      dtype='complex128')

    for i_t in range(uvd_comp.Ntimes):
        # Get data for every baseline at the i_t-th time step
        inds = [i_t*2*uvd_comp.Nbls, (i_t+1)*2*uvd_comp.Nbls]
        # Store data for all baselines at the zeroth frequency first,
        # then data for all baselines at the first frequency, etc.
        flat_inds = [i_t*2*uvd_comp.Nbls*uvd_comp.Nfreqs,
                     (i_t + 1)*2*uvd_comp.Nbls*uvd_comp.Nfreqs]
        # Flattening data_array_reordered[inds[0]:inds[1]] in
        # Fortran ordering flattens along columns, i.e. along the
        # baseline axis which returns a data vector in which
        # every nbls entries contain the visibility data for all
        # baselines at each frequency
        data_array_flattened[flat_inds[0]:flat_inds[1]] =\
            data_array_reordered[inds[0]:inds[1]].flatten(order='F')
        phasor_array_flattened[flat_inds[0]:flat_inds[1]] =\
            phasor_array_reordered[inds[0]:inds[1]].flatten(order='F')

    print('data_array_flattened.std() = {}'.format(data_array_flattened.std()))

    if opts.save:
        if opts.clobber:
            print('Clobbering files, if they exist.')

        if (
                (
                    os.path.exists(os.path.join(save_dir, outfile))
                    and opts.clobber
                )
                or not os.path.exists(os.path.join(save_dir, outfile))):
            print('\nWriting data to {}...'.format(
                os.path.join(save_dir, outfile))
                )
            np.save(
                os.path.join(save_dir, outfile),
                data_array_flattened
                )

    # ------------------- INSTRUMENT MODEL -------------------
    # Generate a (u, v, w) and redundancy model to be used
    # as the BayesEoR instrument model

    # Construct uvw model from phased data
    uvw_model_unphased = np.zeros((uvd.Ntimes, 2*uvd_comp.Nbls, 3))
    colors = plt.cm.viridis(np.linspace(0, 1, uvd.Ntimes))

    if opts.plot_intermediate:
        fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        for ax in axs:
            ax.set_xlabel('u [m]')
            ax.set_ylabel('v [m]')
            ax.set_aspect('equal')
        axs[0].set_title('Unphased UVW Model from\nMirrored healvis Data')
        axs[1].set_title('UVW Ordering from\npyuvdata.get_redundancies()')
        axs[2].set_title('Redundancy Model')

    for i_t, t in enumerate(np.unique(uvd_select.time_array)):
        time_inds = uvd_comp.time_array == t
        if phase_data:
            uvws = uvd_comp_unphased.uvw_array[time_inds]
        else:
            uvws = uvd_comp.uvw_array[time_inds]
        uvws_stacked = np.vstack((uvws, -uvws))
        uvw_model_unphased[i_t] = uvws_stacked

        if opts.plot_intermediate and i_t == 0:
            ax = axs[0]
            ax.scatter(uvw_model_unphased[i_t, :, 0],
                       uvw_model_unphased[i_t, :, 1],
                       c=colors[i_t].reshape(1, -1),
                       marker='o')
            for i_uv, uvw_vec in enumerate(uvws_stacked):
                ax.annotate(str(i_uv), (uvw_vec[0], uvw_vec[1]))

    if opts.plot_intermediate:
        ax = axs[1]
        ax.scatter(vec_bin_centers[:, 0],
                   vec_bin_centers[:, 1],
                   marker='o')
        for i_uv, uvw_vec in enumerate(vec_bin_centers):
            ax.annotate(str(i_uv), (uvw_vec[0], uvw_vec[1]))

    # Construct redundancy model
    if opts.uniform_red_model:
        redundancy_model = np.ones((uvw_model_unphased.shape[0],
                                    uvw_model_unphased.shape[1],
                                    1))
        redundancy_vec = np.ones(uvw_model_unphased.shape[1])
    else:
        redundancy_model = np.zeros((uvw_model_unphased.shape[0],
                                     uvw_model_unphased.shape[1],
                                     1))
        blgp_redundancies = np.array(
            [len(bl_group) for bl_group in baseline_groups]
            )
        redundancy_vec = np.hstack((blgp_redundancies, blgp_redundancies))
        for i_t in range(redundancy_model.shape[0]):
            redundancy_model[i_t] = redundancy_vec[:, np.newaxis]

    if opts.plot_intermediate:
        ax = axs[2]
        sc = ax.scatter(
            uvw_model_unphased[uvw_model_unphased.shape[0]//2, :, 0],
            uvw_model_unphased[uvw_model_unphased.shape[0]//2, :, 1],
            c=redundancy_vec,
            cmap=plt.cm.viridis)
        for i, uvw in enumerate(uvw_model_unphased[0]):
            ax.annotate(str(i), (uvw[0], uvw[1]))
        fig.colorbar(sc, ax=ax, label='Baseline Redundancy')

        fig.tight_layout()
        plt.show()

    if opts.save_model:
        print('\nSaving model to {}...'.format(inst_model_dir))

        if not os.path.exists(inst_model_dir):
            os.mkdir(inst_model_dir)

        # Save uvw model
        with open(
                Path(inst_model_dir) / 'uvw_multi_time_step_array_meters',
                'wb'
                ) as f:
            pickle.dump(uvw_model_unphased, f)

        # Save redundancy model
        red_file =\
            'unique_H37_baseline_hermitian_redundancy_multi_time_step_array'
        with open(Path(inst_model_dir) / red_file, 'wb') as f:
            pickle.dump(redundancy_model, f)

        # Save phasor array
        phasor_filename = 'phasor_vector.npy'
        np.save(Path(inst_model_dir) / phasor_filename,
                phasor_array_flattened)


if opts.bl_type:
    # Only keep a specific baseline type (unique? to H1C IDR2.2 data)
    # Need to change this since this references a very specific
    # file path unique to my machine
    bl_info_dic = np.load(
            '/users/jburba/data/shared/bayeseor_files/'
            'idr2d2/bl_info_dic_Nbls_143.npy',
            allow_pickle=True
        ).item()
    bl_to_keep = []
    for key in bl_info_dic.keys():
        if bl_info_dic[key] == opts.bl_type:
            bl_to_keep.append(key)
    opts.ant_str = ','.join(bl_to_keep)

if opts.all_bl_noise:
    # Keep a copy of the uvdata object with all baselines
    print('\nKeeping copy of data with all baselines for noise calculation.',
          end='\n\n')
    uvd_all_bls = copy.deepcopy(uvd)
else:
    uvd_all_bls = None

if opts.ant_str:
    # Process each baseline individually in uvd_select
    print('Keeping only specified baselines in ant_str = {}'.format(
        opts.ant_str)
        )
    uvd.select(ant_str=opts.ant_str)
    print('Nbls after removal:', uvd.Nbls, end='\n\n')

if opts.single_bls:
    print('-'*60)
    print('Starting single baseline runs for {} baselines at {}'.format(
        uvd.Nbls, datetime.utcnow())
        )

    # dictionary for baseline orientation
    # angles to cardinal directions
    orientation_dic = {
        120.0: 'NW',
        150.0: 'NW',
        0.0: 'EW',
        60.0: 'NE',
        90.0: 'NS',
        30.0: 'NE'
        }

    for bl in np.unique(uvd.baseline_array):
        # Select single baseline from original uvdata object
        bl = uvd.baseline_to_antnums(bl)
        uvd_select = uvd.select(bls=bl, inplace=False)

        print('Baseline {}:\n'.format(bl) + '-'*32)

        if opts.save_model:
            inst_model_dir = opts.inst_model_dir
            inst_model_dir = inst_model_dir.replace(
                'steps/',
                'steps_single_bl_{}_{}/'.format(bl[0], bl[1])
                )

            if not os.path.exists(inst_model_dir):
                os.mkdir(inst_model_dir)

            # perform data for single baseline
            data_processing(
                uvd_select,
                opts,
                filename,
                save_dir=opts.save_dir,
                inst_model_dir=inst_model_dir,
                uvd_all_bls=uvd_all_bls)
        else:
            data_processing(
                uvd_select,
                opts,
                filename,
                save_dir=opts.save_dir,
                uvd_all_bls=uvd_all_bls)
        print('')
    print('Single baseline runs finished at {}'.format(datetime.utcnow()))
    print('-'*60)
else:
    # Keeping all baselines in one data vector / instrument model
    if opts.bl_type:
        opts.inst_model_dir = opts.inst_model_dir.replace(
            'steps/',
            'steps_{}_bls/'.format(opts.bl_type)
            )
        if not os.path.exists(opts.inst_model_dir):
            os.mkdir(opts.inst_model_dir)

    data_processing(uvd, opts, filename,
                    inst_model_dir=opts.inst_model_dir,
                    uvd_all_bls=uvd_all_bls,
                    save_dir=opts.save_dir)
    print('')
