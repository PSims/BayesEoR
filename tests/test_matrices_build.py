from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import bayeseor.matrices.build as build_module


class FakeHealpix:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.pixel_area_sr = 0.25


class FakeMatrixHealpix:
    def __init__(self) -> None:
        self.jds = np.array([2450000.0], dtype=float)
        self.ra_eor = np.array([0.01, 0.02], dtype=float)
        self.dec_eor = np.array([0.03, 0.04], dtype=float)
        self.ra_fg = np.array([0.1, 0.2], dtype=float)
        self.dec_fg = np.array([0.3, 0.4], dtype=float)
        self.fovs_match = True
        self.npix_fov = 2
        self.eor_to_fg_pix = np.array([0, 1], dtype=int)

    def calc_lmn_from_radec(
        self,
        time: float,
        ra: np.ndarray[Any, Any],
        dec: np.ndarray[Any, Any],
        radec_offset: Any = None,
        return_azza: bool = False,
    ) -> tuple[np.ndarray[Any, Any], ...]:
        ls = np.array([1.0, 2.0], dtype=float)
        ms = np.array([3.0, 4.0], dtype=float)
        ns = np.array([5.0, 6.0], dtype=float)
        if return_azza:
            az = np.array([0.7, 0.8], dtype=float)
            za = np.array([0.9, 1.0], dtype=float)
            return ls, ms, ns, az, za
        return ls, ms, ns

    def get_beam_vals(
        self,
        az: np.ndarray[Any, Any],
        za: np.ndarray[Any, Any],
        freq: float,
    ) -> np.ndarray[Any, Any]:
        return np.array([freq, freq + 1.0], dtype=float)


def _base_build_kwargs() -> dict[str, Any]:
    return {
        "nu": 2,
        "du_eor": 0.1,
        "nv": 2,
        "dv_eor": 0.1,
        "nu_fg": 2,
        "du_fg": 0.2,
        "nv_fg": 2,
        "dv_fg": 0.2,
        "nf": 3,
        "neta": 2,
        "deta": 1e-6,
        "f_min": 100.0,
        "df": 1.0,
        "sigma": 1.5,
    }


def test_build_matrices_constructor_registers_optional_matrix_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build_module, "Healpix", FakeHealpix)

    bm = build_module.BuildMatrices(
        **_base_build_kwargs(),
        include_instrumental_effects=True,
        uvw_array_m=np.zeros((2, 3, 3), dtype=float),
        bl_red_array=np.ones((2, 3, 1), dtype=float),
        nside=8,
        fov_ra_eor=10.0,
        fov_dec_eor=12.0,
        jd_center=2450000.0,
        nt=2,
        dt=10.0,
        beam_type="uniform",
        phasor=np.ones(18, dtype=complex),
        use_shg=True,
        nu_sh=1,
        nv_sh=1,
        nq_sh=1,
        npl_sh=1,
        taper_func="hann",
    )

    np.testing.assert_allclose(bm.freqs_hertz, np.array([100e6, 101e6, 102e6]))
    assert bm.Finv_normalisation == 0.25
    assert "phasor_matrix" in bm.matrix_methods
    assert "taper_matrix" in bm.matrix_methods
    assert "idft_array_1d_sh" in bm.matrix_methods
    assert "nuidft_array_sh" in bm.matrix_methods
    assert "phasor_matrix" in bm.matrix_prereqs["Finv"]
    assert "taper_matrix" in bm.matrix_prereqs["Finv"]


def test_build_matrices_constructor_without_instrumental_effects_skips_healpix() -> None:
    bm = build_module.BuildMatrices(
        **_base_build_kwargs(),
        include_instrumental_effects=False,
    )

    np.testing.assert_allclose(bm.freqs_hertz, np.array([100e6, 101e6, 102e6]))
    assert bm.Finv_normalisation == 1.0
    assert not hasattr(bm, "hpx")


def test_build_matrices_requires_uvw_array_when_instrumental_effects_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build_module, "Healpix", FakeHealpix)

    with pytest.raises(AssertionError, match="uvw_array_m must not be None"):
        build_module.BuildMatrices(
            **_base_build_kwargs(),
            include_instrumental_effects=True,
            uvw_array_m=None,
            bl_red_array=np.ones((2, 3, 1), dtype=float),
            nside=8,
            fov_ra_eor=10.0,
            jd_center=2450000.0,
            nt=2,
            dt=10.0,
            beam_type="uniform",
        )


def test_build_taper_matrix_outputs_expected_diagonal() -> None:
    bm = object.__new__(build_module.BuildMatrices)
    bm.verbose = False
    bm.taper_func = "boxcar"
    bm.nf = 3
    bm.nt = 2
    bm.uvw_array_m = np.zeros((2, 2, 3), dtype=float)
    bm.use_sparse_matrices = False
    captured: dict[str, Any] = {}

    def fake_output_data(matrix: np.ndarray[Any, Any], name: str) -> None:
        captured["matrix"] = matrix
        captured["name"] = name

    cast_bm = bm  # retain local name for readability
    setattr(cast_bm, "output_data", fake_output_data)

    build_module.BuildMatrices.build_taper_matrix(bm)

    assert captured["name"] == "taper_matrix"
    expected_taper = np.tile(
        np.repeat(np.ones(3, dtype=float)[None, :], 2, axis=0).flatten(order="F"),
        2,
    )
    np.testing.assert_allclose(np.diag(captured["matrix"]), expected_taper)


def test_build_phasor_matrix_outputs_expected_diagonal() -> None:
    bm = object.__new__(build_module.BuildMatrices)
    bm.verbose = False
    bm.phasor = np.array([1.0 + 0.0j, 0.0 + 1.0j], dtype=complex)
    bm.use_sparse_matrices = False
    captured: dict[str, Any] = {}

    def fake_output_data(matrix: np.ndarray[Any, Any], name: str) -> None:
        captured["matrix"] = matrix
        captured["name"] = name

    setattr(bm, "output_data", fake_output_data)

    build_module.BuildMatrices.build_phasor_matrix(bm)

    assert captured["name"] == "phasor_matrix"
    np.testing.assert_allclose(np.diag(captured["matrix"]), bm.phasor)


def test_build_multi_chan_nudft_uses_scalar_time_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bm = object.__new__(build_module.BuildMatrices)
    bm.verbose = False
    bm.uvw_array_m = np.zeros((1, 2, 3), dtype=float)
    bm.freqs_hertz = np.array([100e6, 101e6], dtype=float)
    bm.drift_scan = False
    setattr(bm, "hpx", FakeMatrixHealpix())
    bm.nt = 1
    bm.nf = 2
    bm.beam_center = None
    bm.use_sparse_matrices = False
    bm.Finv_normalisation = 0.25
    bm.load_prerequisites = lambda matrix_name: {}
    captured: dict[str, Any] = {}

    def fake_output_data(matrix: np.ndarray[Any, Any], name: str) -> None:
        captured["matrix"] = matrix
        captured["name"] = name

    setattr(bm, "output_data", fake_output_data)

    def fake_nudft(
        sampled_lmn_coords_radians: np.ndarray[Any, Any],
        sampled_uvw_coords: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        return np.full(
            (sampled_uvw_coords.shape[0], sampled_lmn_coords_radians.shape[0]),
            2.0,
            dtype=float,
        )

    monkeypatch.setattr(build_module, "nuDFT_Array_DFT_2D_v2d0", fake_nudft)

    build_module.BuildMatrices.build_multi_chan_nudft(bm)

    assert captured["name"] == "multi_chan_nudft"
    expected = np.block(
        [
            [np.full((2, 2), 0.5, dtype=float), np.zeros((2, 2), dtype=float)],
            [np.zeros((2, 2), dtype=float), np.full((2, 2), 0.5, dtype=float)],
        ]
    )
    np.testing.assert_allclose(captured["matrix"], expected)


def test_build_multi_chan_beam_uses_scalar_time_center() -> None:
    bm = object.__new__(build_module.BuildMatrices)
    bm.verbose = False
    bm.achromatic_beam = False
    bm.freqs_hertz = np.array([100e6, 101e6], dtype=float)
    bm.drift_scan = False
    setattr(bm, "hpx", FakeMatrixHealpix())
    bm.nt = 1
    bm.beam_center = None
    bm.use_sparse_matrices = False
    bm.load_prerequisites = lambda matrix_name: {}
    captured: dict[str, Any] = {}

    def fake_output_data(matrix: np.ndarray[Any, Any], name: str) -> None:
        captured["matrix"] = matrix
        captured["name"] = name

    setattr(bm, "output_data", fake_output_data)

    build_module.BuildMatrices.build_multi_chan_beam(bm)

    assert captured["name"] == "multi_chan_beam"
    expected = np.diag([100e6, 100e6 + 1.0, 101e6, 101e6 + 1.0])
    np.testing.assert_allclose(captured["matrix"], expected)


def test_build_nuidft_array_uses_scalar_time_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bm = object.__new__(build_module.BuildMatrices)
    bm.verbose = False
    setattr(bm, "hpx", FakeMatrixHealpix())
    bm.nt = 1
    bm.nu = 2
    bm.nv = 2
    bm.du_eor = 0.1
    bm.dv_eor = 0.2
    bm.Fprime_normalization_eor = 0.5
    bm.load_prerequisites = lambda matrix_name: {}
    captured: dict[str, Any] = {}

    def fake_output_data(matrix: np.ndarray[Any, Any], name: str) -> None:
        captured["matrix"] = matrix
        captured["name"] = name

    setattr(bm, "output_data", fake_output_data)

    def fake_nuidft_matrix_2d(*args: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
        return np.full((2, 3), 2.0, dtype=float)

    monkeypatch.setattr(build_module, "nuidft_matrix_2d", fake_nuidft_matrix_2d)

    build_module.BuildMatrices.build_nuidft_array(bm)

    assert captured["name"] == "nuidft_array"
    np.testing.assert_allclose(captured["matrix"], np.full((2, 3), 1.0))


def test_build_multi_chan_nuidft_fg_preserves_monopole_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bm = object.__new__(build_module.BuildMatrices)
    bm.verbose = False
    setattr(bm, "hpx", FakeMatrixHealpix())
    bm.nt = 1
    bm.nf = 2
    bm.nu_fg = 2
    bm.nv_fg = 2
    bm.du_fg = 0.1
    bm.dv_fg = 0.2
    bm.Fprime_normalization_fg = 0.5
    bm.fit_for_monopole = True
    bm.nuv_fg = 4
    bm.use_sparse_matrices = False
    bm.load_prerequisites = lambda matrix_name: {}
    captured: dict[str, Any] = {}

    def fake_output_data(matrix: np.ndarray[Any, Any], name: str) -> None:
        captured["matrix"] = matrix
        captured["name"] = name

    setattr(bm, "output_data", fake_output_data)

    def fake_nuidft_matrix_2d(*args: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
        return np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], dtype=float)

    monkeypatch.setattr(build_module, "nuidft_matrix_2d", fake_nuidft_matrix_2d)

    build_module.BuildMatrices.build_multi_chan_nuidft_fg(bm)

    assert captured["name"] == "multi_chan_nuidft_fg"
    single_block = np.array([[0.5, 1.0, 2.0, 1.5], [5.0, 10.0, 20.0, 15.0]])
    expected = np.block(
        [
            [single_block, np.zeros_like(single_block)],
            [np.zeros_like(single_block), single_block],
        ]
    )
    np.testing.assert_allclose(captured["matrix"], expected)


def test_build_nuidft_array_sh_uses_eor_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bm = object.__new__(build_module.BuildMatrices)
    bm.verbose = False
    setattr(bm, "hpx", FakeMatrixHealpix())
    bm.nt = 1
    bm.nf = 2
    bm.nu = 2
    bm.nv = 2
    bm.nu_sh = 1
    bm.nv_sh = 1
    bm.du_eor = 3.0
    bm.dv_eor = 4.0
    bm.Fprime_normalization_eor = 0.5
    bm.use_sparse_matrices = False
    bm.load_prerequisites = lambda matrix_name: {}
    captured: dict[str, Any] = {}
    called: dict[str, Any] = {}

    def fake_output_data(matrix: np.ndarray[Any, Any], name: str) -> None:
        captured["matrix"] = matrix
        captured["name"] = name

    setattr(bm, "output_data", fake_output_data)

    def fake_idft_array(*args: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
        called["kwargs"] = kwargs
        return np.full((2, 1), 4.0, dtype=float)

    monkeypatch.setattr(build_module, "IDFT_Array_IDFT_2D_ZM_SH", fake_idft_array)

    build_module.BuildMatrices.build_nuidft_array_sh(bm)

    assert called["kwargs"]["delta_u_irad"] == 3.0
    assert called["kwargs"]["delta_v_irad"] == 4.0
    assert captured["name"] == "nuidft_array_sh"
    expected_block = np.full((2, 1), 0.5, dtype=float)
    expected = np.block(
        [
            [expected_block, np.zeros_like(expected_block)],
            [np.zeros_like(expected_block), expected_block],
        ]
    )
    np.testing.assert_allclose(captured["matrix"], expected)


def test_build_idft_array_1d_sh_uses_funcs_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bm = object.__new__(build_module.BuildMatrices)
    bm.verbose = False
    bm.nf = 3
    bm.neta = 2
    bm.nq_sh = 1
    bm.npl_sh = 0
    bm.fit_for_shg_amps = False
    bm.f_min = 100.0
    bm.df = 1.0
    bm.beta = None
    bm.Fz_normalization = 0.5
    bm.nu_sh = 2
    bm.nv_sh = 2
    bm.load_prerequisites = lambda matrix_name: {}
    bm.use_sparse_matrices = False
    captured: dict[str, Any] = {}
    called: dict[str, Any] = {}

    def fake_output_data(matrix: np.ndarray[Any, Any], name: str) -> None:
        captured["matrix"] = matrix
        captured["name"] = name

    setattr(bm, "output_data", fake_output_data)

    def fake_idft_array_idft_1d_sh(*args: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
        called["kwargs"] = kwargs
        return np.full((3, 2), 2.0, dtype=float)

    monkeypatch.setattr(build_module, "idft_array_idft_1d_sh", fake_idft_array_idft_1d_sh)

    build_module.BuildMatrices.build_idft_array_1d_sh(bm)

    assert called["kwargs"]["f_min"] == 100.0
    assert called["kwargs"]["df"] == 1.0
    assert captured["name"] == "idft_array_1d_sh"
    expected_block = np.full((3, 2), 2.0 * 0.5 * 2, dtype=float)
    expected = np.block(
        [
            [expected_block, np.zeros_like(expected_block), np.zeros_like(expected_block)],
            [np.zeros_like(expected_block), expected_block, np.zeros_like(expected_block)],
            [np.zeros_like(expected_block), np.zeros_like(expected_block), expected_block],
        ]
    )
    np.testing.assert_allclose(captured["matrix"], expected)


def test_build_idft_array_1d_fg_handles_monopole_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bm = object.__new__(build_module.BuildMatrices)
    bm.verbose = False
    bm.nf = 3
    bm.deta = 0.5
    bm.neta = 2
    bm.nuv_fg = 4
    bm.fit_for_monopole = True
    bm.nq = 1
    bm.npl = 0
    bm.f_min = 100.0
    bm.df = 1.0
    bm.beta = None
    bm.use_sparse_matrices = False
    captured: dict[str, Any] = {}

    def fake_output_data(matrix: np.ndarray[Any, Any], name: str) -> None:
        captured["matrix"] = matrix
        captured["name"] = name

    setattr(bm, "output_data", fake_output_data)

    def fake_build_lssm_basis_vectors(*args: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
        return np.array([[1.0], [2.0], [3.0]], dtype=complex)

    def fake_idft_matrix_1d(*args: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
        return np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=complex,
        )

    monkeypatch.setattr(build_module, "build_lssm_basis_vectors", fake_build_lssm_basis_vectors)
    monkeypatch.setattr(build_module, "idft_matrix_1d", fake_idft_matrix_1d)

    build_module.BuildMatrices.build_idft_array_1d_fg(bm)

    assert captured["name"] == "idft_array_1d_fg"
    assert captured["matrix"].shape[0] > 0


def test_build_ninv_without_instrumental_effects() -> None:
    bm = object.__new__(build_module.BuildMatrices)
    bm.verbose = False
    bm.include_instrumental_effects = False
    bm.nu = 2
    bm.nv = 3
    bm.nf = 2
    bm.sigma = 4.0
    bm.use_sparse_matrices = False
    bm.load_prerequisites = lambda matrix_name: {}
    captured: dict[str, Any] = {}

    def fake_output_data(matrix: np.ndarray[Any, Any], name: str) -> None:
        captured["matrix"] = matrix
        captured["name"] = name

    setattr(bm, "output_data", fake_output_data)

    build_module.BuildMatrices.build_Ninv(bm)

    assert captured["name"] == "Ninv"
    np.testing.assert_allclose(np.diag(captured["matrix"]), np.full(10, 1.0 / 16.0))


def test_build_n_without_instrumental_effects() -> None:
    bm = object.__new__(build_module.BuildMatrices)
    bm.verbose = False
    bm.include_instrumental_effects = False
    bm.nu = 2
    bm.nv = 3
    bm.nf = 2
    bm.sigma = 4.0
    bm.use_sparse_matrices = False
    bm.load_prerequisites = lambda matrix_name: {}
    captured: dict[str, Any] = {}

    def fake_output_data(matrix: np.ndarray[Any, Any], name: str) -> None:
        captured["matrix"] = matrix
        captured["name"] = name

    setattr(bm, "output_data", fake_output_data)

    build_module.BuildMatrices.build_N(bm)

    assert captured["name"] == "N"
    np.testing.assert_allclose(np.diag(captured["matrix"]), np.full(10, 16.0))
