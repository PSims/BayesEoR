""" Convenience functions for setting up a BayesEoR analysis. """

import numpy as np
from pathlib import Path


def run_setup(**kwargs):
    """
    Run setup steps.

    Parameters
    ----------
    ...

    """
    
    """
    Required steps:
    1.  [vis.read_data] Read visibility data (load directly via pyuvdata)
    2.  [..model.instrument] Load instrument model
    3.  [..utils.?] Create output directory
    4.  [..matrices.build_matrices] Build matrices
    5.  [..model.k_cube] Create model k cube
    #.  [driver?] Load T and T_Ninv_T (used where? why loaded here?)
    6.  [vis.mock_data_*] Create mock data (requires matrix stack for Finv)
    7.  [..model.noise] Add synthetic noise to data (optional)
    #.  [?] Apply taper matrix (leave this in?  it's not documented as working)
    8.  [vis.?] Hermitian data checks
    9.  [driver?] Calculate dbar and d_Ninv_d (why here? requires matrix stack)
    10. [..posterior] Set up priors
    11. [..posterior] Create posterior class (change name, too long)
    12. [driver?] Estimate posterior calculation time
    13. [DELETE] Write MAP dict
    14. [..run?] Define likelihood function for MultiNest/PolyChord
    15. [..run] Run analysis
    """