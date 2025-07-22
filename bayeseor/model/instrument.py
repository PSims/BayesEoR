import numpy as np
import os
from pathlib import Path

from ..utils import load_numpy_dict

def load_inst_model(
    inst_model_dir,
    uvw_file="uvw_model.npy",
    red_file="redundancy_model.npy",
    antpairs_file="antpairs.npy",
    phasor_file="phasor_vector.npy"
):
    """
    Load a BayesEoR instrument model.
    
    The instrument model consists of:
    - (u, v, w) array with shape `(nt, nbls, 3)` where `nt` is the number of
      times and `nbls` is the number of baselines
    - Number of redundantly-averaged baselines per (u, v, w) with shape
      `(nt, nbls, 1)`
    - (optional) a list of antenna pair tuples with length `nbls` matching the
      order of baselines in the (u, v, w) array
    - (optional) a phasor vector with shape (nf*nt*nbls,) which takes an
      unphased set of visibilities and phases them to a user-specified time
      in the observation

    This function first looks for an 'instrument_model.npy' pickled dictionary
    in `inst_model_dir`. If not found, it will then load the individual numpy
    arrays specified by `uvw_file`, `red_file`, and `phasor_file`.

    Parameters
    ----------
    inst_model_dir : pathlib.Path or str
        Path to the instrument model directory.
    uvw_file : str, optional
        File containing instrumentally sampled (u, v, w) coords. Defaults to
        'uvw_model.npy'.
    red_file : str, optional
        File containing baseline redundancy info. Defaults to
        'redundancy_model.npy'.
    antpairs_file : str, optional
        File containing baseline antenna pairs for each sampled (u, v, w).
        Defaults to 'antpairs.npy'
    phasor_file : str, optional
        File containing the phasor vector. Defaults to 'phasor_vector.npy'.

    Returns
    -------
    uvw_array_m : numpy.ndarray
        Sampled (u, v, w) coordinates in meters.
    bl_red_array : numpy.ndarray
        Number of redundantly-averaged baselines per (u, v, w).
    antpairs : list of tuple
        List of baseline antenna pair tuples if `antpairs_file` is found in
        `inst_model_dir`, otherwise None.
    phasor_vec : numpy.ndarray
        Phasor vector if `phasor_file` is found in `inst_model_dir`, otherwise
        None.

    """
    if not isinstance(inst_model_dir, Path):
        inst_model_dir = Path(inst_model_dir)

    if (inst_model_dir / "instrument_model.npy").exists():
        data_dict = np.load(
            inst_model_dir / "instrument_model.npy",
            allow_pickle=True
        ).item()
        uvw_array_m = data_dict["uvw_model"]
        bl_red_array = data_dict["redundancy_model"]
        if "antpairs" in data_dict:
            antpairs = data_dict["antpairs"]
        else:
            antpairs = None
        if "phasor_vector" in data_dict:
            phasor_vec = data_dict["phasor_vector"]
        else:
            phasor_vec = None
    else:
        uvw_array_m = load_numpy_dict(inst_model_dir / uvw_file)
        bl_red_array = load_numpy_dict(inst_model_dir / red_file)
        if (inst_model_dir / antpairs_file).exists():
            antpairs = load_numpy_dict(inst_model_dir / antpairs_file)
        else:
            antpairs = None
        if (inst_model_dir / phasor_file).exists():
            phasor_vec = load_numpy_dict(inst_model_dir / phasor_file)
        else:
            phasor_vec = None

    return uvw_array_m, bl_red_array, antpairs, phasor_vec