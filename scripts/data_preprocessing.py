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

import BayesEoR
import numpy as np
import pickle
import copy
import os
import sys
import optparse
import warnings
import subprocess

from pathlib import Path
from datetime import datetime
from pyuvdata import UVData
from astropy.time import Time, TimeDelta
from astropy import units
from astropy.units import Quantity
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


plt.rcParams.update({'font.size': 16, 'figure.figsize': (12, 8)})
DEFAULT_SAVE_DIR = str(
    Path(BayesEoR.__file__).parent / 'scripts'
)


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
    '--save_data',
    action='store_true',
    dest='save',
    default=True
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
    '--telescope_name',
    type=str,
    default=None,
    help='Telescope name to use for the instrument model.'
)
o.add_option(
    '--uniform_red_model',
    action='store_true',
    help='If passed, replace the redundancy model'
         'with a uniform model (all ones).'
)
o.add_option(
    '--plot_inst_model',
    action='store_true',
    help='If passed, produce plots showing baseline reordering and '
         'the redundancy model in the uv-plane.'
)
o.add_option(
    '--ant_str',
    type=str,
    help='If passed, keep only baselines specified by ant_str '
         'according to UVData.select syntax.'
)
o.add_option(
    '--single_bls',
    action='store_true',
    help='If passed, create data files for each baseline.  '
         'If passed with --ant_str, only make data files '
         'for the baselines contained in --ant_str.'
)
o.add_option(
    '--bl_type',
    type=str,
    help='Baseline type string for selecting from data.  '
         'Given as a {baseline_length}_{orientation}.  '
         'For example, to keep 14.6 meter EW baselines --bl_type=14d6_EW.  '
         'Must be passed with --bl_dict_path.'
)
o.add_option(
    '--bl_dict_path',
    type=str,
    help='Path to a numpy readable dictionary containing a set of keys '
         'of antenna pair tuples, i.e. (1, 2), each with a value of '
         '{baseline_length}_{orientation}, i.e. \'14.6_EW\'.'
)
o.add_option(
    '--bl_cutoff_m',
    type=float,
    help='Baseline cutoff length in meters.  Any baselines in the raw dataset'
         ' with |b| > <bl_cutoff_m> will be excluded from the written data.'
)
o.add_option(
    '--start_freq_MHz',
    type=float,
    help='Starting frequency in MHz from which 76 right-adjacent '
         'frequency channels will be extracted. Defaults to the first '
         'frequency channel in `filename`.'
)
o.add_option(
    '--nf',
    type=int,
    help='Number of frequency channels to include in the data vector.  '
         'Defaults to keeping all frequencies in the data file.'
)
o.add_option(
    '--avg_adj_freqs',
    action='store_true',
    help='If passed, include 2*nf frequency channels and average two '
         'adjacent frequency channels together to form nf frequencies '
         'in the data vector.'
)
o.add_option(
    '--nt',
    type=int,
    help='Number of integrations to include in the data vector.  '
         'Defaults to keeping all integrations in the data file.'
)
o.add_option(
    '--start_int',
    type=int,
    default=0,
    help='Integration number (zero indexed) from which the next consecutive '
         '`nt` integrations will be taken from the UVData object.  Defaults '
         'to zero.'
)
o.add_option(
    '--unphased',
    action='store_true',
    help='If passed, do not phase data.  Otherwise, data is phased '
           'to `phase_time_jd` or the central time step in the dataset.'
)
o.add_option(
    '--phase_time_jd',
    type=float,
    default=None,
    help='Time to which data will be phased.  Must be a valid Julian Date.'
)
o.add_option(
    '--form_pI',
    action='store_true',
    help="If passed, form pI visibilities from the 'xx' and/or 'yy' pols."
)
o.add_option(
    '--pI_norm',
    type=float,
    default=1.,
    help='Normalization N used in the formation of pI = N * (XX + YY).'
         'Defaults to 1.0.'
)
o.add_option(
    '--all_bl_noise',
    action='store_true',
    help='If passed, generate noise estimate from all '
         'baselines within a redundant group.'
)
opts, args = o.parse_args(sys.argv[1:])


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


def jy_to_ksr(data, freqs):
    """
    Convert visibilities from units of Janskys to Kelvin steradians.

    Parameters
    ----------
    data : np.ndarray
        Array of visibility data in units of Janskys.
    freqs : 1d np.ndarray
        Array of frequencies for data contained in data_array
        in units of Hertz.
    """
    # Tile frequencies to match shape of data=(nblts, nfreqs)
    if not isinstance(freqs, Quantity):
        freqs = Quantity(freqs, units.Hz)

    equiv = units.brightness_temperature(freqs, beam_area=1*units.sr)
    conv_factor = (1*units.Jy).to(units.K, equivalencies=equiv)
    conv_factor *= units.sr / units.Jy

    return data * conv_factor[np.newaxis, :].value


def data_processing(
        uvd_select,
        opts,
        filename,
        min_freq_MHz,
        save_dir=DEFAULT_SAVE_DIR,
        inst_model_dir=None,
        uvd_all_bls=None
):
    """
    Takes a UVData object and produces a 1-dimensional visibility data vector.
    This function returns nothing but produces and saves the following:
        - data_array_flattened : visibility data vector
        - phasor_array_flattened : vector of phasor values used to phase a set
          of unphased visibilities to the central time step
        - uvw_model_unphased : array of (u, v, w) coordinates per baseline
          and time
        - redundancy_model : array of redundancy values per baseline and time,
          i.e. the number of baselines that were averaged together into a
          redundant group

    Parameters
    ----------
    uvd_select : UVData object
        UVData object from which the visibility data vector will be formed.
    opts : dict
        Dictionary containing command line arguments.
    filename : str
        Input filename of UVData compatible file.
    save_dir : str
        Path where the data vector will be written.  Defaults to the `scripts/`
        subdirectory within the installation path of `BayesEoR`.
    inst_model_dir : str
        Path where the instrument model will be written.
    uvd_all_bls : UVData object
        UVData object containing all baselines.  Used for estimating the noise
        from all baselines but downselecting the data vector to a subset of
        baselines.

    """
    outfile = filename.replace(
        '.uvh5',
        '-start-freq-{:.2f}-nf-{}-nbls-{}'.format(
            min_freq_MHz, opts.nf, uvd_select.Nbls*2
        )
    )
    if opts.avg_adj_freqs:
        outfile = outfile.replace(
            '-nf-{}'.format(opts.nf),
            '-nf-{}-adj-freq-avg'.format(opts.nf)
        )
    outfile += '-phased.npy'
    if opts.unphased:
        outfile = outfile.replace('phased', 'unphased')
    elif opts.phase_time_jd is not None:
        outfile = outfile.replace(
            '.npy',
            '-phase-time-{}.npy'.format(opts.phase_time_jd)
        )

    if uvd_select.Nbls == 1:
        antnums = uvd_select.baseline_to_antnums(uvd_select.baseline_array[0])
        outfile = outfile.replace(
            'nbls-{}-'.format(uvd_select.Nbls),
            'bl-{}-{}-'.format(
                uvd_select.Nbls, antnums[0], antnums[1]
            )
        )
    elif opts.bl_type:
        outfile = outfile.replace(
            'nbls-{}-'.format(uvd_select.Nbls),
            'nbls-{}-{}-'.format(uvd_select.Nbls, opts.bl_type)
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

    uvd = copy.deepcopy(uvd_select)  # copy of uvd_select for phasing
    uvd_comp = uvd_select.compress_by_redundancy(inplace=False)
    uvd_comp_phasor = copy.deepcopy(uvd_comp)  # used for the phasor vector
    phasor_array = np.ones(uvd_comp_phasor.data_array.shape) + 0j
    uvd_comp_phasor.data_array = phasor_array

    if not opts.unphased:
        if opts.phase_time_jd is not None:
            print('Phasing data to JD {}'.format(opts.phase_time_jd))
            time_to_phase = opts.phase_time_jd
        else:
            print('Phasing data to central time step')
            time_to_phase = np.unique(uvd_select.time_array)[uvd.Ntimes // 2]
        uvd.phase_to_time(Time(time_to_phase, format='jd'))
        uvd_comp.phase_to_time(Time(time_to_phase, format='jd'))
        uvd_comp_phasor.phase_to_time(
            Time(time_to_phase, format='jd', scale='utc')
            )
        if opts.all_bl_noise:
            uvd_all_bls.phase_to_time(Time(time_to_phase, format='jd'))

    # Average Over Redundant Baselines
    data_array_shape = uvd_comp.data_array.shape
    data_array_shape_avg = [data_array_shape[0], data_array_shape[2]]
    if opts.avg_adj_freqs:
        # Reduce frequency axis by a factor of two
        data_array_shape_avg[1] = data_array_shape_avg[1]//2
    data_array_phased_avg = np.zeros(
        data_array_shape_avg,
        dtype='complex128'
    )
    noise_array_phased_avg = np.zeros_like(data_array_phased_avg)
    phasor_array_phased_avg = np.zeros_like(data_array_phased_avg)

    baseline_groups, vec_bin_centers, _ = uvd_select.get_redundancies()

    for i_bl, bl_group in enumerate(baseline_groups):
        blgp_data_container = []
        blgp_nsamples_container = []

        for bl in bl_group:
            data = uvd.get_data(bl)
            nsamples = uvd.get_nsamples(bl)
            if opts.avg_adj_freqs:
                data = (data[:, ::2] + data[:, 1::2]) / 2
                nsamples = (nsamples[:, ::2] + nsamples[:, 1::2]) / 2
            blgp_data_container.append(data)
            blgp_nsamples_container.append(nsamples)

        blgp_data_container = np.array(blgp_data_container)
        blgp_nsamples_container = np.array(blgp_nsamples_container)
        # Only need to pull phasor info from one baseline per redudant group
        data_phasor = uvd_comp_phasor.get_data(bl_group[0])
        if opts.avg_adj_freqs:
            data_phasor = (data_phasor[:, ::2] + data_phasor[:, 1::2]) / 2

        # Estimate noise for each baseline group
        avg_data, _ = weighted_avg_and_std(
            blgp_data_container, blgp_nsamples_container
        )
        blgp_eo_diff = (
            blgp_data_container[:, ::2] - blgp_data_container[:, 1::2]
        )
        if blgp_eo_diff.shape[0] > 1:
            blgp_eo_noise = np.std(blgp_eo_diff, axis=0)
        else:
            blgp_eo_noise = np.sqrt(np.abs(blgp_eo_diff)**2).squeeze()
        blgp_noise_estimate = np.zeros_like(avg_data)
        blgp_noise_estimate[::2] = blgp_eo_noise
        blgp_noise_estimate[1::2] = blgp_eo_noise

        arr_inds = slice(i_bl * uvd.Ntimes, (i_bl + 1) * uvd.Ntimes)
        # data_array_phased_avg[:ntimes] contains the data for a
        # single redundant baseline group across all frequencies
        # and times with shape (ntimes, nfreqs)
        data_array_phased_avg[arr_inds] = avg_data
        noise_array_phased_avg[arr_inds] = blgp_noise_estimate
        phasor_array_phased_avg[arr_inds] = data_phasor

    # Convert data & noise arrays to K sr from Jy
    frequencies = uvd.freq_array[0]
    nf = frequencies.size
    if opts.avg_adj_freqs:
        frequencies = (frequencies[::2] + frequencies[1::2]) / 2
        nf = frequencies.size
    data_array_phased_avg = jy_to_ksr(
        data_array_phased_avg, frequencies
    )
    data_array_phased_avg *= 1.0e3  # K sr to mK sr

    # Reorder and Flatten Data
    # Double the bl axis size to account for each redundant
    # baseline group and its conjugate
    data_array_reordered = np.zeros(
        (data_array_phased_avg.shape[0]*2, data_array_phased_avg.shape[1]),
        dtype='complex128'
    )
    noise_array_reordered = np.zeros_like(data_array_reordered)
    phasor_array_reordered = np.zeros_like(data_array_reordered)

    for i_t in range(uvd_comp.Ntimes):
        # Get data for every baseline at the i_t-th time index
        data = data_array_phased_avg[i_t::uvd_comp.Ntimes]  # (nbls, nfreqs)
        noise = noise_array_phased_avg[i_t::uvd_comp.Ntimes]
        phasor = phasor_array_phased_avg[i_t::uvd_comp.Ntimes]

        inds = slice(i_t*2*uvd_comp.Nbls, (i_t + 1)*2*uvd_comp.Nbls)
        # data_array_reordered[inds] contains the
        # data for 2*nbls baselines across all frequencies at the
        # i_t-th time index with shape (2*nbls, nfreqs)
        data_array_reordered[inds] = np.vstack((data, data.conjugate()))
        noise_array_reordered[inds] = np.vstack((noise, noise.conjugate()))
        phasor_array_reordered[inds] = np.vstack((phasor, phasor.conjugate()))

    data_array_flattened = np.zeros(data_array_reordered.size,
                                    dtype='complex128')
    noise_array_flattened = np.zeros_like(data_array_flattened)
    phasor_array_flattened = np.zeros_like(data_array_flattened)

    for i_t in range(uvd_comp.Ntimes):
        # Get data for every baseline at the i_t-th time index
        inds = slice(i_t*2*uvd_comp.Nbls, (i_t+1)*2*uvd_comp.Nbls)
        # Store data for all baselines at the zeroth frequency first,
        # then data for all baselines at the first frequency, etc.
        flat_inds = slice(
            i_t*2*uvd_comp.Nbls*nf,
            (i_t + 1)*2*uvd_comp.Nbls*nf
        )
        # Flattening data_array_reordered[inds[0]:inds[1]] in
        # Fortran ordering flattens along columns, i.e. along the
        # baseline axis which returns a data vector in which
        # every nbls entries contain the visibility data for all
        # baselines at each frequency
        data_array_flattened[flat_inds] = (
            data_array_reordered[inds].flatten(order='F')
        )
        noise_array_flattened[flat_inds] = (
            noise_array_reordered[inds].flatten(order='F')
        )
        phasor_array_flattened[flat_inds] = (
            phasor_array_reordered[inds].flatten(order='F')
        )

    print(
        'data_array_flattened.std() =', data_array_flattened.std()
    )
    print(
        'noise_array_flattened.std() =', noise_array_flattened.std(),
        end='\n\n'
    )

    if opts.save:
        if opts.clobber:
            print('Clobbering file, if it exists.')

        version_info = {}
        version_info['git_origin'] = subprocess.check_output(
            ['git', 'config', '--get', 'remote.origin.url'],
            stderr=subprocess.STDOUT)
        version_info['git_hash'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.STDOUT)
        version_info['git_description'] = subprocess.check_output(
            ['git', 'describe', '--dirty', '--tag', '--always'])
        version_info['git_branch'] = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.STDOUT)
        for key in version_info.keys():
            version_info[key] = version_info[key].decode('utf8').strip('\n')

        data_dict = {
            'data': data_array_flattened,
            'noise': noise_array_flattened,
            'opts': vars(opts),
            'version': version_info
        }
        datapath = os.path.join(save_dir, outfile)
        if not os.path.exists(datapath) or opts.clobber:
            print('Writing data to {}...'.format(datapath))
            np.save(datapath, data_dict)

        # if opts.all_bl_noise:
        #     noise_file = outfile.replace('.npy', '-eo-noise-all-bls.npy')
        # else:
        #     noise_file = outfile.replace('.npy', '-eo-noise.npy')
        # noisepath = os.path.join(save_dir, noise_file)
        # if not os.path.exists(noisepath) or opts.clobber:
        #     np.save(noisepath, noise_array_flattened)

    # Instrument Model
    # Generate a uvw and redundancy model to be used as the
    # BayesEoR instrument model
    uvw_model_unphased = np.zeros((uvd.Ntimes, 2*uvd_comp.Nbls, 3))

    # UVW coordinates must match the baseline ordering used in the flattened
    # data array which comes from UVData.get_baseline_redundancies
    uvws_stacked = np.vstack((vec_bin_centers, -vec_bin_centers))
    uvw_model_unphased = np.repeat(
        uvws_stacked[np.newaxis, :, :], uvd.Ntimes, axis=0
    )

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

    if opts.save_model:
        print('\nSaving model to {}...'.format(inst_model_dir))

        if not os.path.exists(inst_model_dir):
            os.mkdir(inst_model_dir)

        uvw_file = 'uvw_model.npy'
        np.save(Path(inst_model_dir) / uvw_file, uvw_model_unphased)

        red_file = 'redundancy_model.npy'
        np.save(Path(inst_model_dir) / red_file, redundancy_model)

        phasor_filename = 'phasor_vector.npy'
        np.save(Path(inst_model_dir) / phasor_filename, phasor_array_flattened)

    if opts.plot_inst_model:
        fig = plt.figure(figsize=(16, 8))
        grid = ImageGrid(
            fig,
            111,
            (1, 2),
            axes_pad=1.0,
            cbar_mode='single',
            cbar_pad=0.1,
            share_all=True
        )
        axs = grid.axes_all
        for ax in axs:
            ax.set_xlabel('u [m]')
            ax.set_ylabel('v [m]')
            ax.set_aspect('equal')

        ax = axs[0]
        ax.set_title(
            'UVW Model',
            fontsize=16,
        )
        ax.scatter(
            uvw_model_unphased[0, :, 0],
            uvw_model_unphased[0, :, 1],
            color='k',
            marker='o'
        )
        for i_uv, uvw_vec in enumerate(uvw_model_unphased[0]):
            ax.annotate(str(i_uv), (uvw_vec[0], uvw_vec[1]))
        u_max = np.abs(ax.get_xlim()[1])
        v_max = np.abs(ax.get_ylim()[1])
        max_uv = np.max((u_max, v_max))
        axlim = (-max_uv, max_uv)
        ax.set_xlim(axlim)
        ax.set_ylim(axlim)

        ax = axs[1]
        ax.set_title('Redundancy Model', fontsize=16)
        sc = ax.scatter(
            uvw_model_unphased[uvw_model_unphased.shape[0]//2, :, 0],
            uvw_model_unphased[uvw_model_unphased.shape[0]//2, :, 1],
            c=redundancy_vec,
            cmap=plt.cm.viridis
        )
        for i, uvw in enumerate(uvw_model_unphased[0]):
            ax.annotate(str(i), (uvw[0], uvw[1]))
        fig.colorbar(sc, ax=ax, cax=ax.cax, label='Redundancy')
        ax.set_xlim(axlim)
        ax.set_ylim(axlim)

        fig.tight_layout()


data_path = Path(opts.data_path)
filename = opts.filename

uvd = UVData()
print('\nReading data from', data_path / filename, end='\n\n')
uvd.read(data_path / filename, read_data=False)

print('Removing autocorrelations')
uvd.select(ant_str='cross')

if opts.bl_cutoff_m:
    print(
        'Selecting only baselines <= {} meters'.format(opts.bl_cutoff_m)
    )
    bl_lengths = np.sqrt(np.sum(uvd.uvw_array[:, :2]**2, axis=1))
    blt_inds = np.where(bl_lengths <= opts.bl_cutoff_m)[0]
    print('\tBaselines before length select:', uvd.Nbls)
    uvd.select(blt_inds=blt_inds)
    print('\tBaselines after length select: ', uvd.Nbls)
    bls_to_read = uvd.get_antpairs()

# Frequency selection
frequencies = uvd.freq_array[0]
if not opts.nf:
    opts.nf = frequencies.size
    if opts.avg_adj_freqs:
        opts.nf = opts.nf//2
nf = opts.nf
if opts.avg_adj_freqs:
    nf *= 2
if not opts.start_freq_MHz:
    opts.start_freq_MHz = uvd.freq_array[0, 0] / 1e6
print(
    'Selecting {} frequency channels >= {:.2f} MHz'.format(
        nf, opts.start_freq_MHz
    )
)
start_freq_ind = np.argmin(np.abs(frequencies - opts.start_freq_MHz*1e6))
if start_freq_ind + nf > uvd.Nfreqs:
    warnings.warn(
        'WARNING: the combination of start_freq_MHz and nf will result in '
        'fewer than nf frequencies being kept in the data vector.'
    )
frequencies = frequencies[start_freq_ind:start_freq_ind+nf]
min_freq_MHz = frequencies[0] / 1e6
print('\tMinimum frequency in data =', (frequencies[0]*units.Hz).to('MHz'))
df = Quantity(np.mean(np.diff(frequencies)), unit='Hz')
if opts.avg_adj_freqs:
    df *= 2
print('\tFrequency channel width in data =', df.to('MHz'))

# Time selection
times = np.unique(uvd.time_array)
if not opts.nt:
    opts.nt = times.size
nt = opts.nt
min_t_ind = opts.start_int
print(
    'Selecting {} times starting from integration number {}'.format(
        nt, min_t_ind
    )
)
if (min_t_ind + nt) > uvd.Ntimes:
    warnings.warn(
        'WARNING: start_int + nt > uvd.Ntimes.  This will result in '
        'fewer than nt integrations being kept in the data vector.'
    )
if uvd.Ntimes > nt:
    times = times[min_t_ind:min_t_ind+nt]
dt = TimeDelta(np.mean(np.diff(times)), format='jd')
print('\tIntegration time in data =', dt.sec, 's')
print('\tCentral JD in data =', times[times.size//2])
if opts.phase_time_jd:
    valid_phase_time = np.logical_and(
        opts.phase_time_jd >= times.min(),
        opts.phase_time_jd <= times.max()
    )
    if not valid_phase_time:
        warnings.warn(
            'WARNING: Supplied phase_time_jd argument lies outside the '
            'desired JD range {:.6f} -= {:.6f}'.format(
                times.min(), times.max()
            )
        )

print('Reading in data_array', end='\n\n')
uvd.read(
    data_path / filename,
    frequencies=frequencies,
    times=times,
    bls=bls_to_read
    )

if np.sum(uvd.flag_array) > 0:
    # Remove any fully flagged baselines
    antpairs = uvd.get_antpairs()
    good_antpairs = []
    for antpair in antpairs:
        flags = uvd.get_flags(antpair)  # + ('xx',) ?
        if np.sum(flags) < flags.size:
            good_antpairs.append(antpair)
    print('Removing fully flagged baselines')
    print('\tBaselines before flagging selection:', uvd.Nbls)
    uvd.select(bls=good_antpairs)
    print('\tBaselines after flagging selection: ', uvd.Nbls)

print('Conjugating baselines to u > 0 convention')
uvd.conjugate_bls(convention='u>0', uvw_tol=1.0)

if opts.form_pI:
    # This should work for now, but need to be more careful
    # about this in the future if/when polarization becomes important
    print('Forming pI visibilities')
    if np.all([pol in uvd.polarization_array for pol in [-5, -6]]):
        # Form pI as xx + yy
        xx_ind = np.where(uvd.polarization_array == -5)[0][0]
        yy_ind = np.where(uvd.polarization_array == -6)[0][0]
        uvd.data_array[..., xx_ind] += uvd.data_array[..., yy_ind]
        uvd.data_array *= opts.pI_norm
        pol_num = -5
    elif -5 in uvd.polarization_array:
        pol_num = -5
    elif -6 in uvd.polarization_array:
        pol_num = -6
    uvd.select(polarizations=pol_num)

if opts.all_bl_noise:
    print(
        'Keeping copy of data with all baselines for noise calculation.',
        end='\n\n'
    )
    uvd_all_bls = copy.deepcopy(uvd)
else:
    uvd_all_bls = None

if opts.bl_type:
    # Only keep a specific baseline type, i.e. length and orientation
    bl_info_dict = np.load(opts.bl_dict_path, allow_pickle=True).item()
    bls_to_keep = []
    for key in bl_info_dict.keys():
        if bl_info_dict[key] == opts.bl_type:
            if isinstance(key, tuple):
                key = '{}_{}'.format(*key)
            bls_to_keep.append(key)
    opts.ant_str = ','.join(bls_to_keep)

if opts.ant_str:
    # Preprocess each baseline in ant_str individually
    print(
        'Keeping only specified baselines in ant_str =', opts.ant_str
    )
    uvd.select(ant_str=opts.ant_str)
    print('Nbls after ant_str select:', uvd.Nbls, end='\n\n')

# Instrument model directory setup
inst_model_dir = str(Path(opts.inst_model_dir) / opts.telescope_name)
inst_model_dir += '-{}-{:.1f}sec-time-steps'.format(nt, dt.sec)
inst_model_dir += '-start-freq-{:.2f}-nf-{}'.format(
    min_freq_MHz, opts.nf
)
if opts.avg_adj_freqs:
    inst_model_dir += '-adj-freq-avg'
if not opts.single_bls:
    inst_model_dir += '-nbls-{}'.format(uvd.Nbls*2)
if opts.bl_cutoff_m and not (opts.single_bls or opts.bl_type):
    inst_model_dir += '-bl-cutoff-{}m'.format(opts.bl_cutoff_m)
inst_model_dir += '/'

if opts.single_bls:
    print('-'*60)
    print(
        'Starting single baseline runs for {} baselines at {}'.format(
            uvd.Nbls, datetime.utcnow()
        ),
        end='\n\n'
    )

    for bl in np.unique(uvd.baseline_array):
        bl = uvd.baseline_to_antnums(bl)
        uvd_select = uvd.select(bls=bl, inplace=False)

        print('Baseline {}:\n'.format(bl) + '-'*32)

        if opts.save_model:
            inst_model_dir_bl = (
                inst_model_dir[:-1]
                + '-bl-{}-{}/'.format(bl[0], bl[1])
            )
        else:
            inst_model_dir_bl = None

        data_processing(
            uvd_select,
            opts,
            filename,
            min_freq_MHz,
            save_dir=opts.save_dir,
            inst_model_dir=inst_model_dir_bl,
            uvd_all_bls=uvd_all_bls)
        print('')

    print('Single baseline runs finished at {}'.format(datetime.utcnow()))
    print('-'*60)
else:
    # Keeping all baselines in one data vector / instrument model
    if opts.bl_type:
        inst_model_dir = inst_model_dir[:-1] + '-{}/'.format(opts.bl_type)

    data_processing(
        uvd,
        opts,
        filename,
        min_freq_MHz,
        inst_model_dir=inst_model_dir,
        uvd_all_bls=uvd_all_bls,
        save_dir=opts.save_dir
    )
