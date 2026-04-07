from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import bayeseor.matrices.build as build_module


class FakeHealpix:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.pixel_area_sr = 0.25


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
