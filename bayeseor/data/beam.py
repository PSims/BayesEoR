import numpy as np
from pathlib import Path
from pyuvdata import UVBeam

from .vis import save_numpy_dict
from ..utils import mpiprint
from .. import __version__


def preprocess_uvbeam(
    fp,
    freqs=None,
    nside=None,
    interp_func="az_za_simple",
    freq_interp_kind="cubic",
    norm="peak",
    save=False,
    save_dir="./",
    clobber=False,
    verbose=False,
    rank=0
):
    """
    Create a frequency-interpolated power beam from a pyuvdata-compatible beam.

    Parameters
    ----------
    fp : Path or str
        Path to pyuvdata-compatible beam file.
    freqs : :class:`astropy.Quantity` or array_like of float, optional
        Interpolation frequencies in Hertz if not a Quantity. Defaults to None
        (keep native frequencies).
    nside : int, optional
        HEALPix resolution for spatial interpolation. Defaults to None (use
        native resolution).
    interp_func : str, optional
        Interpolation function string. Please see the documentation for
        `pyuvdata.UVBeam.interp` for more details on valid interpolation
        functions.  Defaults to 'az_za_simple'.
    freq_interp_kind : str, optional
        Frequency interpolation kind. Please see `scipy.interpolate.interp1d`
        for more details.  Defaults to 'cubic'.
    norm : str, optional
        Beam normalization string. Can be 'peak' or 'physical'. Defaults to
        'peak'.
    save : bool, optional
        Save preprocessed UVBeam object to `save_dir`. Defaults to False.
    save_dir : Path or str, optional
        Output directory if `save` is True. Defaults to './'.
    clobber : bool, optional
        Clobber files on disk if they exist. Defaults to False.
    verbose : bool, optional
        Print statements useful for debugging. Defaults to False.
    rank : int, optional
        MPI rank. Defaults to 0.

    """
    if not isinstance(fp, Path):
        fp = Path(fp)
    if not fp.exists():
        raise FileNotFoundError(f"No such file or directory: '{fp}'")
    
    if save:
        if save_dir is None:
            raise ValueError(
                "save_dir cannot be none if save_vis or save_model is True"
            )
        else:
            if not isinstance(save_dir, Path):
                save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
    
    if np.any([arg is None for arg in [freq_min, df, Nfreqs]])

    # print_rank will only trigger print if verbose is True and rank == 0
    print_rank = 1 - (verbose and rank==0)

    uvb = UVBeam()
    mpiprint(f"\nReading data from: {fp}", rank=print_rank)
    uvb.read(fp)
    if uvb.beam_type == "efield":
        mpiprint("Converting from E-field to power beam", rank=print_rank)
        uvb.efield_to_power()
    uvb.interpolation_function = interp_func
    uvb.freq_interp_kind = freq_interp_kind
