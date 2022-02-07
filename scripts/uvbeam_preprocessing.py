"""
    Create a pStokes power HEALPix UVBeam object interpolated in frequency.
"""

import BayesEoR
import numpy as np
import argparse
import os
import subprocess

from pathlib import Path
from datetime import datetime
from pyuvdata import UVBeam
from astropy import units
from astropy.units import Quantity


parser = argparse.ArgumentParser()
parser.add_argument(
    'data_path',
    type=str,
    help='Path to a UVBeam compatible file.'
)
parser.add_argument(
    '--save_dir',
    type=str,
    help='Filepath in which the UVBeam file will be saved. '
         'Defaults to the directory of the supplied UVBeam compatible file.'
)
parser.add_argument(
    '--start_freq_MHz',
    type=float,
    help='Starting interpolation frequency in MHz.'
)
parser.add_argument(
    '--df_MHz',
    type=float,
    help='Interpolation frequency resolution in MHz.'
)
parser.add_argument(
    '--nf',
    type=int,
    help='Number of interpolated frequency channels.'
)
parser.add_argument(
    '--nside',
    type=int,
    help='HEALPix resolution for spatial interpolation.'
)
parser.add_argument(
    '--interp_func',
    type=str,
    default='az_za_simple',
    help='Spatial interpolation function.  See '
    '`UVBeam.interpolation_function_dict` for possible options. '
    'Defaults to \'az_za_simple\'.'
)
parser.add_argument(
    '--freq_interp_kind',
    type=str,
    default='quadratic',
    help='1D Frequency interpolation kind.  See `scipy.interpolate.interp1d` '
         'for all possible options.  Defaults to \'quadratic\'.'
)
args = parser.parse_args()


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
    filename_out = filename_in.replace(suffix, '{}{}{}'.format(
        join_char, mtime, suffix
    ))
    return filename_out


data_path = Path(args.data_path)
data_dir = data_path.parent
filename = data_path.name
if not args.save_dir:
    args.save_dir = data_dir

freqs = Quantity(
    args.start_freq_MHz + np.arange(args.nf) * args.df_MHz,
    unit='MHz'
)
freqs = freqs.to('Hz')

print(f'Reading in data from {data_path}')
uvb = UVBeam()
uvb.read_beamfits(data_path)
if not uvb.beam_type == 'efield':
    assert 1 in uvb.polarization_array, (
        "If operating on a 'power' beam, must be a pStokes power beam and "
        "include the pI polarization in `UVBeam.polarization_array`."
    )
else:
    print('Converting from E-field to pStokes power beam...')
    uvb.efield_to_pstokes()
uvb.interpolation_function = args.interp_func
uvb.freq_interp_kind = args.freq_interp_kind
print('Interpolating...')
uvb_hpx = uvb.interp(
    freq_array=freqs.to('Hz').value, healpix_nside=args.nside, new_object=True
)

outfile = filename.lower().strip('.fits')
outfile = outfile.replace('_', '-')
outfile = outfile.replace('efield', 'pstokes-power')
outfile += '-nside-{}-{:.2f}-{:.2f}MHz-nf-{}.fits'.format(
    uvb_hpx.nside, *uvb_hpx.freq_array[0, [0, -1]]/1e6, uvb_hpx.freq_array.size
)

# Append Git information to history
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
version_info_str = [
    '{}: {}'.format(key, version_info[key]) for key in version_info.keys()
]
version_info_str = ', '.join(version_info_str)
cla_str = [
    '{}: {}'.format(key, args.__dict__[key]) for key in args.__dict__.keys()
]
cla_str = ', '.join(cla_str)
uvb.history += (
    '\n\nPre-processed with BayesEoR/scripts/uvbeam_preprocessing.py.  '
    + 'Git config info - {}.  '.format(version_info_str)
    + 'Command line arguments - {}'.format(cla_str)
)

save_path = Path(args.save_dir) / outfile
print(f'Writing to:\n{save_path}')
if save_path.exists():
    old_outfile = add_mtime_to_filename(save_path.parent, save_path.name)
    print('Existing file found.  Moving to\n{}'.format(
        data_dir / old_outfile
    ))
    os.rename(
        os.path.join(data_dir, outfile),
        os.path.join(data_dir, old_outfile)
    )
uvb_hpx.write_beamfits(save_path)
