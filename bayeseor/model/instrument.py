import numpy as np
from pathlib import Path

def load_inst_model(
    inst_model_path,
    uvw_file="uvw_model.npy",
    red_file="redundancy_model.npy",
    phasor_file="phasor_vector.npy"
):
    """
    Load the instrument model consisting of
    - a (u, v, w) array with shape (nt, nbls, 3)
    - baseline redundancy array with shape (nt, nbls, 1)
    - a phasor vector with shape (ndata,) (if modelling phased visibilities)

    The phasor vector takes an unphased set of visibilities and phases them
    to the central time step in the observation.

    This function first looks for an 'instrument_model.npy' pickled dictionary
    in `inst_model_path`.  If not found, it will then load the individual numpy
    arrays specified by `uvw_file`, `red_file`, and `phasor_file`.

    Parameters
    ----------
    inst_model_path : str
        Path to the instrument model directory.
    uvw_file : str
        File containing instrumentally sampled (u, v, w) coords.  Defaults to
        'uvw_model.npy'.
    red_file : str
        File containing baseline redundancy info.  Defaults to
        'redundancy_model.npy'.
    phasor_file : str
        File containing the phasor vector.  Defaults to 'phasor_vector.npy'.

    Returns
    -------
    uvw_array_m : np.ndarray
        Array of (u, v, w) coordinates.
    bl_red_array : np.ndarray
        Array of baseline redundancies.
    phasor_vec : np.ndarray
        Array of phasor values.

    """
    inst_model_path = Path(inst_model_path)
    if (inst_model_path / "instrument_model.npy").exists():
        # Instrument model stored in a dictionary
        data_dict = np.load(
            inst_model_path / "instrument_model.npy",
            allow_pickle=True
        ).item()
        uvw_array_m = data_dict["uvw_model"]
        bl_red_array = data_dict["redundancy_model"]
        if "phasor_vector" in data_dict:
            phasor_vec = data_dict["phasor_vector"]
        else:
            phasor_vec = None
    else:
        # Instrument model stored as individual files
        uvw_array_m = np.load(inst_model_path / uvw_file)
        bl_red_array = np.load(inst_model_path / red_file)
        if (inst_model_path / phasor_file).exists():
            phasor_vec = np.load(inst_model_path / phasor_file)
        else:
            phasor_vec = None

    return uvw_array_m, bl_red_array, phasor_vec