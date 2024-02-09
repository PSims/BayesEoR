"""
    Create a power HEALPix UVBeam object interpolated in frequency.
"""

import numpy as np
import argparse
import os

from pathlib import Path
from datetime import datetime
from pyuvdata import UVBeam
from astropy.units import Quantity

from bayeseor import __version__

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
    default=None,
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
parser.add_argument(
    '--norm',
    type=str,
    default='peak',
    help='Beam normalization to use.  Can be either \'peak\' or \'physical\'.'
         '  Defaults to \'peak\'.'
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
if uvb.beam_type == 'efield':
    print('Converting from E-field to power beam...')
    uvb.efield_to_power()
uvb.interpolation_function = args.interp_func
uvb.freq_interp_kind = args.freq_interp_kind
print('Interpolating...')
uvb_interp = uvb.interp(
    freq_array=freqs.to('Hz').value, healpix_nside=args.nside, new_object=True
)
if args.norm == 'peak':
    print('Peak normalizing...')
    uvb_interp.peak_normalize()

outfile = filename.lower().strip('.fits')
outfile = outfile.replace('_', '-')
outfile = outfile.replace('efield', 'power')
if args.norm == 'peak':
    outfile += '-peak-norm'
if not args.nside is None:
    outfile += f'-nside-{uvb_interp.nside}'
outfile += '-{:.2f}-{:.2f}MHz-nf-{}'.format(
    *uvb_interp.freq_array[0, [0, -1]]/1e6, uvb_interp.freq_array.size
)
outfile += '.fits'

# Append Git information to history
cla_str = [
    '{}: {}'.format(key, args.__dict__[key]) for key in args.__dict__.keys()
]
cla_str = ', '.join(cla_str)
uvb.history += (
    f'\n\nPre-processed with bayeseor version {__version__}\n\n'
    + f'Command line arguments - {cla_str}'
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
uvb_interp.write_beamfits(save_path)
