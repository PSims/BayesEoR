import numpy as np

import pytest

from bayeseor.params import BayesEoRParser
from bayeseor.matrices import BuildMatrices
from bayeseor.model import load_inst_model
from bayeseor.utils import get_array_dir_name

@pytest.fixture(scope="session")
def config():
    parser = BayesEoRParser()
    args = parser.parse_args(["--config", "tests-config.yaml"])

    # Add additional attributes required by a broad range of tests
    uvw_array_m, bl_red_array, phasor_vec = load_inst_model(args.inst_model)
    uvw_array_m_vec = np.reshape(uvw_array_m, (-1, 3))
    args.uvw_array_m = uvw_array_m
    args.n_vis = len(uvw_array_m_vec)
    args.bl_red_array = bl_red_array
    args.phasor_vec = phasor_vec
    
    yield args

    del args


@pytest.fixture(scope="session")
def build_matrices(config):
    cl_args = config

    array_dir = get_array_dir_name(cl_args)[0]

    args = [
        array_dir,
        cl_args.include_instrumental_effects,
        cl_args.use_sparse_matrices,
        cl_args.nu,
        cl_args.nv,
        cl_args.n_vis,
        cl_args.neta,
        cl_args.nf,
        cl_args.nu_min_MHz,
        cl_args.channel_width_MHz,
        cl_args.nq,
        cl_args.nt,
        cl_args.integration_time_seconds,
        cl_args.sigma,
        cl_args.fit_for_monopole,
    ]

    required_params = [
        'nside',
        'central_jd',
        'telescope_latlonalt',
        'drift_scan',
        'beam_type',
        'beam_peak_amplitude',
        'beam_center',
        'fwhm_deg',
        'antenna_diameter',
        'cosfreq',
        'achromatic_beam',
        'beam_ref_freq',
        'du_eor',
        'dv_eor',
        'du_fg',
        'dv_fg',
        'deta',
        'fov_ra_eor',
        'fov_dec_eor',
        'nu_fg',
        'nv_fg',
        'npl',
        'beta',
        'fov_ra_fg',
        'fov_dec_fg',
        'simple_za_filter',
        'uvw_array_m',
        'bl_red_array',
        'bl_red_array_vec',
        'phasor_vec',
        'use_shg',
        'fit_for_shg_amps',
        'nu_sh',
        'nv_sh',
        'nq_sh',
        'npl_sh',
        'effective_noise',
        'taper_func'
    ]
    kwargs = {key: cl_args.__dict__[key] for key in required_params}

    BM = BuildMatrices(*args, **kwargs)

    yield BM

    del BM