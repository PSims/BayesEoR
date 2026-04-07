from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import bayeseor.setup as bayes_setup


class FakeBuildMatrices:
    def __init__(self) -> None:
        self.verbose = False
        self.array_dir = "fake-array-dir"
        self.freqs_hertz = np.array([150e6, 151e6], dtype=float)

    def read_data(self, name: str) -> Any:
        if name == "Ninv":
            return np.eye(2)
        if name == "T":
            return np.eye(2, dtype=complex)
        if name == "T_Ninv_T":
            return np.eye(2)
        if name == "block_T_Ninv_T":
            return [np.eye(2)]
        raise KeyError(name)

    def dot_product(self, matrix: Any, vector: Any) -> Any:
        return np.dot(matrix, vector)


class FakePosterior:
    def __init__(self) -> None:
        self.k_vals = np.array([0.1, 0.2], dtype=float)


def test_get_vis_data_preprocessed_path_builds_axes(
    tmp_path: Path, monkeypatch: Any
) -> None:
    data_path = tmp_path / "vis.npy"
    noise_path = tmp_path / "noise.npy"
    inst_model = tmp_path / "inst"
    data_path.touch()
    noise_path.touch()
    inst_model.mkdir()
    (inst_model / "uvw_model.npy").touch()
    (inst_model / "redundancy_model.npy").touch()

    vis = np.array([1 + 1j, 2 + 2j], dtype=complex)
    noise = np.array([0.1 + 0.0j, 0.2 + 0.0j], dtype=complex)
    uvws = np.zeros((2, 1, 3), dtype=float)
    redundancy = np.ones((2, 1, 1), dtype=float)
    antpairs = np.array([(0, 1)])
    phasor = np.ones(2, dtype=complex)

    def fake_load_numpy_dict(path: Path) -> np.ndarray[Any, Any]:
        if path == data_path:
            return vis
        if path == noise_path:
            return noise
        raise AssertionError(f"Unexpected path {path}")

    monkeypatch.setattr(bayes_setup, "load_numpy_dict", fake_load_numpy_dict)
    monkeypatch.setattr(
        bayes_setup,
        "load_inst_model",
        lambda _: (uvws, redundancy, antpairs, phasor),
    )

    vis_dict = bayes_setup.get_vis_data(
        data_path=data_path,
        nf=2,
        df=1e6,
        freq_min=150e6,
        nt=2,
        dt=10.0,
        jd_min=2450000.0,
        calc_noise=True,
        noise_data_path=noise_path,
        inst_model=inst_model,
    )

    np.testing.assert_allclose(vis_dict["vis_noisy"], vis)
    np.testing.assert_allclose(vis_dict["noise"], noise)
    np.testing.assert_allclose(vis_dict["freqs"], np.array([150e6, 151e6]))
    np.testing.assert_allclose(
        vis_dict["jds"],
        np.array([2450000.0, 2450000.0 + 10.0 / 86400.0]),
    )
    np.testing.assert_array_equal(
        vis_dict["bl_conj_pairs_map"], np.arange(vis.size, dtype=int)
    )
    assert "uvd" not in vis_dict


def test_run_setup_return_vis_and_bm_preserves_tuple_contract(
    tmp_path: Path, monkeypatch: Any
) -> None:
    fake_vis_dict: bayes_setup.VisibilityData = {
        "vis_noisy": np.array([1 + 0j, 2 + 0j], dtype=complex),
        "noise": np.array([0.1 + 0j, 0.2 + 0j], dtype=complex),
        "bl_conj_pairs_map": np.array([0, 1], dtype=int),
        "uvws": np.zeros((1, 1, 3), dtype=float),
        "redundancy": np.ones((1, 1, 1), dtype=float),
        "freqs": np.array([150e6, 151e6], dtype=float),
        "df": 1e6,
        "jds": np.array([2450000.0, 2450000.1], dtype=float),
        "dt": 10.0,
        "vis": np.array([1 + 0j, 2 + 0j], dtype=complex),
    }

    fake_bm = FakeBuildMatrices()
    fake_pspp = FakePosterior()

    monkeypatch.setattr(bayes_setup, "get_vis_data", lambda **_: fake_vis_dict)
    monkeypatch.setattr(
        bayes_setup,
        "build_k_cube",
        lambda **_: (np.array([0.1, 0.2], dtype=float), [(np.array([0, 1]),)]),
    )
    monkeypatch.setattr(bayes_setup, "build_matrices", lambda **_: fake_bm)
    monkeypatch.setattr(bayes_setup, "build_posterior", lambda **_: fake_pspp)
    monkeypatch.setattr(
        bayes_setup.Cosmology,
        "f2z",
        lambda self, freq: 8.0,
    )
    monkeypatch.setattr(
        bayes_setup.Cosmology,
        "dL_df",
        lambda self, redshift: 1.0,
    )
    monkeypatch.setattr(
        bayes_setup.Cosmology,
        "dL_dth",
        lambda self, redshift: 1.0,
    )

    result = bayes_setup.run_setup(
        nu=2,
        nside=1,
        fov_ra_eor=10.0,
        beam_type="uniform",
        data_path=tmp_path / "vis.npy",
        priors=[[0.0, 1.0]],
        sigma=1.0,
        include_instrumental_effects=False,
        file_root_dir="run",
        output_dir=tmp_path,
        mkdir=False,
        return_vis=True,
        return_bm=True,
    )

    assert len(result) == 4
    pspp, sampler_dir, vis_dict, bm = result
    assert pspp is fake_pspp
    assert sampler_dir == tmp_path / "run"
    assert vis_dict is fake_vis_dict
    assert bm is fake_bm
