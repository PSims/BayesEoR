from __future__ import annotations

import numpy as np

from bayeseor.matrices.funcs import (
    IDFT_Array_IDFT_2D_ZM_SH,
    Produce_Coordinate_Arrays_ZM_SH,
    build_lssm_basis_vectors,
    idft_array_idft_1d_sh,
)


def test_idft_array_idft_2d_zm_sh_uses_explicit_linear_spacing() -> None:
    sampled_lm_coords_radians = np.array([[0.0, 0.0], [0.25, -0.5]], dtype=float)
    matrix = IDFT_Array_IDFT_2D_ZM_SH(
        3,
        3,
        sampled_lm_coords_radians,
        delta_u_irad=6.0,
        delta_v_irad=3.0,
    )

    u_vec, v_vec = Produce_Coordinate_Arrays_ZM_SH(3, 3)
    u_vec = u_vec.astype(float) * (6.0 / 3.0)
    v_vec = v_vec.astype(float) * (3.0 / 3.0)
    x_vec = sampled_lm_coords_radians[:, 0].reshape(-1, 1)
    y_vec = sampled_lm_coords_radians[:, 1].reshape(-1, 1)
    expected = np.exp(-2.0 * np.pi * 1j * (x_vec * u_vec + y_vec * v_vec))

    np.testing.assert_allclose(matrix, expected)


def test_build_lssm_basis_vectors_accepts_scalar_beta() -> None:
    basis_vectors = build_lssm_basis_vectors(
        nf=4, nq=1, npl=1, f_min=100.0, df=1.0, beta=2.63
    )

    freq_array = 100.0 + np.arange(4)
    expected = (freq_array / 100.0) ** -2.63
    np.testing.assert_allclose(basis_vectors[:, 0].real, expected)


def test_build_lssm_basis_vectors_uses_distinct_power_law_indices() -> None:
    basis_vectors = build_lssm_basis_vectors(
        nf=3, nq=3, npl=3, f_min=100.0, df=1.0, beta=[1.0, 2.0, 3.0]
    )

    freq_array = 100.0 + np.arange(3)
    expected = np.column_stack(
        [
            (freq_array / 100.0) ** -1.0,
            (freq_array / 100.0) ** -2.0,
            (freq_array / 100.0) ** -3.0,
        ]
    )
    np.testing.assert_allclose(basis_vectors.real, expected)


def test_idft_array_idft_1d_sh_uses_default_lssm_parameters() -> None:
    idft_array = idft_array_idft_1d_sh(
        nf=4, neta=3, nq_sh=1, npl_sh=0, fit_for_shg_amps=False
    )

    assert idft_array.shape == (4, 2)
