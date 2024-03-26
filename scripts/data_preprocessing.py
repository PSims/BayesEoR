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

import bayeseor
import numpy as np
import copy
import os
import warnings

from pathlib import Path
from datetime import datetime
from pyuvdata import UVData, utils
from astropy.time import Time, TimeDelta
from astropy import units
from astropy.units import Quantity
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from jsonargparse import ArgumentParser, ActionConfigFile


plt.rcParams.update(
    {'font.size': 16, 'figure.figsize': (12, 8), 'figure.facecolor': 'w'}
)


parser = ArgumentParser()
parser.add_argument(
    '--data_path',
    type=str,
    help='Path to the pyuvdata compatible data file for preprocessing.'
)
parser.add_argument(
    '--filename',
    type=str,
    help='Filename in opts.data_path to use for preprocessing.'
)
parser.add_argument(
    '--clobber',
    action='store_true',
    help='If passed, clobber existing data file(s).'
)
parser.add_argument(
    '--save_dir',
    type=str,
    default='./',
    help='Filepath in which the data will be saved. Defaults to the current '
         'working directory.'
)
parser.add_argument(
    '--save_model',
    action='store_true',
    help='If passed, save the generated uvw and redundancy models.'
)
parser.add_argument(
    '--inst_model_dir',
    type=str,
    default=None,
    help='Directory in which the instrument model will be saved.'
)
parser.add_argument(
    '--telescope_name',
    type=str,
    default=None,
    help='Telescope name to use for the instrument model.'
)
parser.add_argument(
    '--uniform_red_model',
    action='store_true',
    help='If passed, replace the redundancy model with a uniform model (all '
         'ones).'
)
parser.add_argument(
    '--plot_inst_model',
    action='store_true',
    help='If passed, produce plots showing baseline reordering and '
         'the redundancy model in the uv-plane.'
)
parser.add_argument(
    '--ant_str',
    type=str,
    help='If passed, keep only baselines specified by ant_str '
         'according to UVData.select syntax.'
)
parser.add_argument(
    '--single_bls',
    action='store_true',
    help='If passed, create data files for each baseline.  '
         'If passed with --ant_str, only make data files '
         'for the baselines contained in --ant_str.'
)
parser.add_argument(
    '--bl_type',
    type=str,
    help='Baseline type string for selecting from data.  '
         'Given as a {baseline_length}_{orientation}.  '
         'For example, to keep 14.6 meter EW baselines --bl_type=14d6_EW.  '
         'Must be passed with --bl_dict_path.'
)
parser.add_argument(
    '--bl_dict_path',
    type=str,
    help='Path to a numpy readable dictionary containing a set of keys '
         'of antenna pair tuples, i.e. (1, 2), each with a value of '
         '{baseline_length}_{orientation}, i.e. \'14.6_EW\'.'
)
parser.add_argument(
    '--bl_cutoff_m',
    type=float,
    help='Baseline cutoff length in meters.  Any baselines in the raw dataset '
         'with |b| > <bl_cutoff_m> will be excluded from the written data.'
)
parser.add_argument(
    '--start_freq_MHz',
    type=float,
    help='Starting frequency in MHz from which `nf` right-adjacent '
         'frequency channels will be extracted. Defaults to the first '
         'frequency channel in `filename`.'
)
parser.add_argument(
    '--nf',
    type=int,
    help='Number of frequency channels to include in the data vector.  '
         'Defaults to keeping all frequencies in the data file.'
)
parser.add_argument(
    '--avg_adj_freqs',
    action='store_true',
    help='If passed, include 2*nf frequency channels and average two '
         'adjacent frequency channels together to form nf frequencies '
         'in the data vector.'
)
parser.add_argument(
    '--nt',
    type=int,
    help='Number of integrations to include in the data vector.  '
         'Defaults to keeping all integrations in the data file.'
)
parser.add_argument(
    '--start_int',
    type=int,
    default=0,
    help='Integration number (zero indexed) from which the next consecutive '
         '`nt` integrations will be taken from the UVData object.  Defaults '
         'to zero.'
)
parser.add_argument(
    '--phase',
    action='store_true',
    help='If passed, phase data.  If `phase_time_jd` is not specified, data '
         'will be phased to the central time step in the dataset.'
)
parser.add_argument(
    '--phase_time_jd',
    type=float,
    default=None,
    help='Time to which data will be phased.  Must be a valid Julian Date.'
)
parser.add_argument(
    '--form_pI',
    action='store_true',
    help="If passed, form pI visibilities from the 'xx' and/or 'yy' pols."
)
parser.add_argument(
    '--pI_norm',
    type=float,
    default=1.,
    help='Normalization N used in the formation of pI = N * (XX + YY). '
         'Defaults to 1.0.'
)
parser.add_argument(
    '--pol',
    type=str,
    default='xx',
    help='Polarization string to keep in the data vector.  Not used if forming'
         ' pI visibilities.  Defaults to \'xx\'.'
)
parser.add_argument(
    '--all_bl_noise',
    action='store_true',
    help='If passed, generate noise estimate from all baselines within a '
         'redundant group.'
)
parser.add_argument(
    "--config",
    action=ActionConfigFile
)
opts = parser.parse_args()


def add_mtime_to_filename(path, filename_in, join_char='-'):
    """
    Appends the mtime to a filename before the file suffix.

    Parameters
    ----------
    path : str
        Path to filename.
    filename_in : str
        Name of file.

    Returns
    -------
    filename_out : str
        Modified filename containing the mtime of the file.

    """
    fp = Path(path) / filename_in
    suffix = fp.suffix
    mtime = datetime.fromtimestamp(os.path.getmtime(fp))
    mtime = mtime.isoformat()
    filename_out = filename_in.replace(suffix, f'{join_char}{mtime}{suffix}')
    return filename_out


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


def jy_to_ksr(data, freqs, mK=False):
    """
    Convert visibilities from units of Janskys to Kelvin steradians.

    Parameters
    ----------
    data : np.ndarray
        Array of visibility data in units of Janskys.
    freqs : 1d np.ndarray
        Array of frequencies for data contained in data_array
        in units of Hertz.
    mK : bool
        If True, multiply by 1000 to convert from K sr to mK sr.
        Otherwise, return data in K sr.

    """
    # Tile frequencies to match shape of data=(nblts, nfreqs)
    if not isinstance(freqs, Quantity):
        freqs = Quantity(freqs, units.Hz)

    equiv = units.brightness_temperature(freqs, beam_area=1*units.sr)
    if mK:
        temp_unit = units.mK
    else:
        temp_unit = units.K
    conv_factor = (1*units.Jy).to(temp_unit, equivalencies=equiv)
    conv_factor *= units.sr / units.Jy

    return data * conv_factor[np.newaxis, :].value


def data_processing(
        uvd_select,
        opts,
        filename,
        min_freq_MHz,
        central_jd,
        save_dir="./",
        inst_model_dir=None,
        uvd_all_bls=None
):
    """
    Takes a UVData object and produces a 1-dimensional visibility data vector.
    This function returns nothing but produces and saves the following:
        - data_array_flattened : visibility data vector
        - phasor_array_flattened : vector of phasor values used to phase a set
          of unphased visibilities to the central time step
        - uvw_model : array of (u, v, w) coordinates per baseline
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
    min_freq_MHz : float
        Minimum frequency in the data.
    central_jd : float
        Central JD in the data.
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
    # wn-gsm-ptsrc-fov-30-start-freq-150.29-nf-180-nbls-30-pol-pI.npy
    outfile = filename.replace('.uvh5', '')
    # Frequency parameters
    outfile += f'-min-freq-{min_freq_MHz:.2f}MHz-Nfreqs-{opts.nf}'
    if opts.avg_adj_freqs:
        outfile += '-adj-freq-avg'
    # Baseline parameters
    if opts.bl_type:
        outfile += f'-Nbls-{uvd_select.Nbls*2}-{opts.bl_type}'
    else:
        if uvd_select.Nbls > 1:
            outfile += f'-Nbls-{uvd_select.Nbls*2}'
        elif uvd_select.Nbls == 1:
            outfile += f'-bl-{antnums[0]}-{antnums[1]}'
    # Time parameters
    outfile += f'-Ntimes-{opts.nt}-central-JD-{central_jd:.2f}'
    # Polarization parameters
    if opts.form_pI:
        outfile += '-pol-pI'
    else:
        outfile += f'-pol-{opts.pol}'
    # Phasing parameters
    if opts.phase:
        outfile += '-phased'
    elif opts.phase_time_jd is not None:
        outfile += f'-phase-time-{opts.phase_time_jd}'
    outfile += '.npy'
    # What about the case where I only keep certain baselines within a
    # redundant baseline type? Do I need some sort of unique identifier
    # for chosen baselines when I choose two separate sets of Nbls?

    uvd = copy.deepcopy(uvd_select)  # copy of uvd_select for phasing
    uvd_comp = uvd_select.compress_by_redundancy(inplace=False)

    if opts.phase:
        uvd_comp_phasor = copy.deepcopy(uvd_comp)  # used for the phasor vector
        phasor_array = np.ones(uvd_comp_phasor.data_array.shape) + 0j
        uvd_comp_phasor.data_array = phasor_array
        if opts.phase_time_jd is not None:
            print(f'Phasing data to JD {opts.phase_time_jd}')
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
    data_array_avg = np.zeros(data_array_shape_avg, dtype='complex128')
    noise_array_avg = np.zeros_like(data_array_avg)
    if opts.phase:
        phasor_array_avg = np.zeros_like(data_array_avg)

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
        if opts.phase:
            # Only need phasor info from one baseline per redudant group
            data_phasor = uvd_comp_phasor.get_data(bl_group[0])
            if opts.avg_adj_freqs:
                # I don't think this is the right thing to do
                # Needs to be corrected in the future if phasing data
                data_phasor = (data_phasor[:, ::2] + data_phasor[:, 1::2]) / 2

        # Estimate noise for each baseline group
        avg_data, _ = weighted_avg_and_std(
            blgp_data_container, blgp_nsamples_container
        )
        blgp_eo_diff = np.zeros((len(bl_group), uvd.Ntimes//2, uvd.Nfreqs))
        even_times = blgp_data_container[:, ::2]
        odd_times = np.zeros_like(even_times)
        if uvd.Ntimes % 2 == 1:
            odd_times[:, :-1] = blgp_data_container[:, 1::2]
            odd_times[:, -1] = blgp_data_container[:, -2]
        else:
            odd_times = blgp_data_container[:, 1::2]
        blgp_eo_diff = even_times - odd_times
        if blgp_eo_diff.shape[0] > 1:
            blgp_eo_noise = np.std(blgp_eo_diff, axis=0)
        else:
            blgp_eo_noise = np.sqrt(np.abs(blgp_eo_diff)**2).squeeze()
        blgp_noise_estimate = np.zeros_like(avg_data)
        blgp_noise_estimate[::2] = blgp_eo_noise
        if uvd.Ntimes % 2 == 1:
            blgp_noise_estimate[1::2] = blgp_eo_noise[:-1]
        else:
            blgp_noise_estimate[1::2] = blgp_eo_noise

        arr_inds = slice(i_bl * uvd.Ntimes, (i_bl + 1) * uvd.Ntimes)
        # data_array_avg[:ntimes] contains the data for a
        # single redundant baseline group across all frequencies
        # and times with shape (ntimes, nfreqs)
        data_array_avg[arr_inds] = avg_data
        noise_array_avg[arr_inds] = blgp_noise_estimate
        if opts.phase:
            phasor_array_avg[arr_inds] = data_phasor

    # Convert data & noise arrays to mK sr from Jy
    frequencies = uvd.freq_array[0]
    nf = frequencies.size
    if opts.avg_adj_freqs:
        frequencies = (frequencies[::2] + frequencies[1::2]) / 2
        nf = frequencies.size
    data_array_avg = jy_to_ksr(data_array_avg, frequencies, mK=True)

    # Reorder and Flatten Data
    # Double the bl axis size to account for each redundant
    # baseline group and its conjugate
    data_array_reordered = np.zeros(
        (data_array_avg.shape[0]*2, data_array_avg.shape[1]),
        dtype='complex128'
    )
    noise_array_reordered = np.zeros_like(data_array_reordered)
    if opts.phase:
        phasor_array_reordered = np.zeros_like(data_array_reordered)

    for i_t in range(uvd_comp.Ntimes):
        # Get data for every baseline at the i_t-th time index
        data = data_array_avg[i_t::uvd_comp.Ntimes]  # (nbls, nfreqs)
        noise = noise_array_avg[i_t::uvd_comp.Ntimes]
        if opts.phase:
            phasor = phasor_array_avg[i_t::uvd_comp.Ntimes]

        inds = slice(i_t*2*uvd_comp.Nbls, (i_t + 1)*2*uvd_comp.Nbls)
        # data_array_reordered[inds] contains the
        # data for 2*nbls baselines across all frequencies at the
        # i_t-th time index with shape (2*nbls, nfreqs)
        data_array_reordered[inds] = np.vstack((data, data.conjugate()))
        noise_array_reordered[inds] = np.vstack((noise, noise.conjugate()))
        if opts.phase:
            phasor_array_reordered[inds] = np.vstack(
                (phasor, phasor.conjugate())
            )

    data_array_flattened = np.zeros(data_array_reordered.size,
                                    dtype='complex128')
    noise_array_flattened = np.zeros_like(data_array_flattened)
    if opts.phase:
        phasor_array_flattened = np.zeros_like(data_array_flattened)

    for i_t in range(uvd_comp.Ntimes):
        # Get data for every baseline at the i_t-th time index
        inds = slice(i_t*2*uvd_comp.Nbls, (i_t+1)*2*uvd_comp.Nbls)
        # Store data for all baselines at the zeroth frequency first,
        # then data for all baselines at the first frequency, etc.
        flat_inds = slice(
            i_t*2*uvd_comp.Nbls*nf, (i_t + 1)*2*uvd_comp.Nbls*nf
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
        if opts.phase:
            phasor_array_flattened[flat_inds] = (
                phasor_array_reordered[inds].flatten(order='F')
            )

    print('data_array_flattened.std() =', data_array_flattened.std())
    print('noise_array_flattened.std() =', noise_array_flattened.std(),
          end='\n\n')

    if opts.clobber:
        print('Clobbering file if it exists')

    data_dict = {
        'data': data_array_flattened,
        'noise': noise_array_flattened,
        'opts': vars(opts),
        'version': bayeseor.__version__,
        'ctime': datetime.now().isoformat()
    }
    datapath = os.path.join(save_dir, outfile)
    print(f'Writing data to\n{datapath}')
    if not os.path.exists(datapath) or opts.clobber:
        np.save(datapath, data_dict)
    else:
        old_outfile = add_mtime_to_filename(save_dir, outfile)
        print(
            'Existing file found.  Moving to\n',
            os.path.join(save_dir, old_outfile)
        )
        os.rename(
            os.path.join(save_dir, outfile),
            os.path.join(save_dir, old_outfile)
        )
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
    # UVW coordinates must match the baseline ordering used in the flattened
    # data array which comes from UVData.get_baseline_redundancies
    uvws_stacked = np.vstack((vec_bin_centers, -vec_bin_centers))
    uvw_model = np.repeat(
        uvws_stacked[np.newaxis, :, :], uvd.Ntimes, axis=0
    )

    if opts.uniform_red_model:
        redundancy_model = np.ones((uvw_model.shape[0], uvw_model.shape[1], 1))
        redundancy_vec = np.ones(uvw_model.shape[1])
    else:
        redundancy_model = np.zeros(
            (uvw_model.shape[0], uvw_model.shape[1], 1)
        )
        blgp_redundancies = np.array(
            [len(bl_group) for bl_group in baseline_groups]
        )
        redundancy_vec = np.hstack((blgp_redundancies, blgp_redundancies))
        for i_t in range(redundancy_model.shape[0]):
            redundancy_model[i_t] = redundancy_vec[:, np.newaxis]

    if opts.save_model:
        print(f"\nInstrument model: {inst_model_dir.split('/')[-2]}")
        if opts.clobber:
            print('Clobbering files if they exist')

        if not os.path.exists(inst_model_dir):
            os.mkdir(inst_model_dir)

        data_dict = {
            'uvw_model': uvw_model,
            'redundancy_model': redundancy_model,
            'opts': vars(opts),
            'version': bayeseor.__version__,
            'ctime': datetime.now().isoformat()
        }
        if opts.phase:
            data_dict['phasor_vector'] = phasor_array_flattened
        outfile = 'instrument_model.npy'
        datapath = os.path.join(inst_model_dir, outfile)
        print(f'Writing instrument model to\n{datapath}')
        if not os.path.exists(datapath) or opts.clobber:
            np.save(datapath, data_dict)
        else:
            old_outfile = add_mtime_to_filename(
                inst_model_dir, outfile, join_char='_'
            )
            print(
                'Existing model found.  Moving to\n',
                os.path.join(inst_model_dir, old_outfile)
            )
            os.rename(
                os.path.join(inst_model_dir, outfile),
                os.path.join(inst_model_dir, old_outfile)
            )
            np.save(datapath, data_dict)

    if opts.plot_inst_model:
        fig = plt.figure(figsize=(16, 8))
        grid = ImageGrid(
            fig,
            111,
            (1, 2),
            axes_pad=1.0,
            share_all=True,
            cbar_mode='single',
            cbar_pad=0.1
        )
        axs = grid.axes_all
        for ax in axs:
            ax.set_xlabel('u [m]')
            ax.set_ylabel('v [m]')
            ax.set_aspect('equal')

        ax = axs[0]
        ax.set_title('UVW Model', fontsize=16)
        ax.scatter(
            uvw_model[0, :, 0],
            uvw_model[0, :, 1],
            color='k',
            marker='o'
        )
        for i_uv, uvw_vec in enumerate(uvw_model[0]):
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
            uvw_model[uvw_model.shape[0]//2, :, 0],
            uvw_model[uvw_model.shape[0]//2, :, 1],
            c=redundancy_vec,
            cmap=plt.cm.viridis
        )
        for i, uvw in enumerate(uvw_model[0]):
            ax.annotate(str(i), (uvw[0], uvw[1]))
        fig.colorbar(sc, ax=ax, cax=ax.cax, label='Redundancy')
        ax.set_xlim(axlim)
        ax.set_ylim(axlim)

        fig.tight_layout()
        plt.show()


data_path = Path(opts.data_path)
filename = opts.filename

uvd = UVData()
print('\nReading data from', data_path / filename, end='\n\n')
uvd.read(data_path / filename, read_data=False)

print('Removing autocorrelations')
uvd.select(ant_str='cross')

if opts.bl_cutoff_m:
    print(
        f'Selecting only baselines <= {opts.bl_cutoff_m} meters'
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
    f'Selecting {nf} frequency channels >= {opts.start_freq_MHz:.2f} MHz'
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
    f'Selecting {nt} times starting from integration number {min_t_ind}'
)
if (min_t_ind + nt) > uvd.Ntimes:
    warnings.warn(
        'WARNING: start_int + nt > uvd.Ntimes.  This will result in '
        'fewer than nt integrations being kept in the data vector.'
    )
if uvd.Ntimes > nt:
    times = times[min_t_ind:min_t_ind+nt]
dt = TimeDelta(np.mean(np.diff(times)), format='jd')
central_jd = times[times.size//2]
print('\tIntegration time in data =', dt.sec, 's')
print('\tCentral JD in data =', central_jd)
if opts.phase_time_jd:
    valid_phase_time = np.logical_and(
        opts.phase_time_jd >= times.min(),
        opts.phase_time_jd <= times.max()
    )
    if not valid_phase_time:
        warnings.warn(
            'WARNING: Supplied phase_time_jd argument lies outside the '
            f'desired JD range {times.min():.6f} - {times.max():.6f}'
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
else:
    pol_num = utils.polstr2num(opts.pol)
    assert pol_num in uvd.polarization_array, (
        f"Polarization {opts.pol} not present in polarization_array."
    )
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
                key = f'{key[0]}_{key[1]}'
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
inst_model_dir += f'-{nt}-{dt.sec:.1f}sec-time-steps'
if opts.phase:
    inst_model_dir += f'-min-freq-{min_freq_MHz:.2f}-Nfreqs-{opts.nf}'
    if opts.avg_adj_freqs:
        inst_model_dir += '-adj-freq-avg'
if not opts.single_bls:
    inst_model_dir += f'-Nbls-{uvd.Nbls*2}'
if opts.bl_cutoff_m and not (opts.single_bls or opts.bl_type):
    inst_model_dir += f'-bl-cutoff-{opts.bl_cutoff_m}m'
inst_model_dir += '/'

if opts.single_bls:
    print('-'*60)
    print(
        f'Starting single baseline runs for {uvd.Nbls} baselines '
        f'at {datetime.utcnow()}',
        end='\n\n'
    )

    for bl in np.unique(uvd.baseline_array):
        bl = uvd.baseline_to_antnums(bl)
        uvd_select = uvd.select(bls=bl, inplace=False)

        print(f'Baseline {bl}:\n' + '-'*32)

        if opts.save_model:
            inst_model_dir_bl = (
                inst_model_dir[:-1]
                + f'-bl-{bl[0]}-{bl[1]}/'
            )
        else:
            inst_model_dir_bl = None

        data_processing(
            uvd_select,
            opts,
            filename,
            min_freq_MHz,
            central_jd,
            save_dir=opts.save_dir,
            inst_model_dir=inst_model_dir_bl,
            uvd_all_bls=uvd_all_bls)
        print('')

    print(f'Single baseline runs finished at {datetime.utcnow()}')
    print('-'*60)
else:
    # Keeping all baselines in one data vector / instrument model
    if opts.bl_type:
        inst_model_dir = inst_model_dir[:-1] + f'-{opts.bl_type}/'

    data_processing(
        uvd,
        opts,
        filename,
        min_freq_MHz,
        central_jd,
        inst_model_dir=inst_model_dir,
        uvd_all_bls=uvd_all_bls,
        save_dir=opts.save_dir
    )
