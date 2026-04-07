from __future__ import annotations

from typing import Any

import numpy as np

from bayeseor.analyze.maxap import MaximumAPosteriori


class FakePosterior:
    def __init__(self) -> None:
        self.T_Ninv_T = np.eye(2)
        self.dbar = np.array([1.0, 2.0], dtype=float)
        self.uprior_inds = np.array([True, False])

    def calc_SigmaI_dbar_wrapper(
        self,
        dmps: np.ndarray[Any, Any],
        T_Ninv_T: np.ndarray[Any, Any],
        dbar: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], float, np.ndarray[Any, Any], float]:
        return (
            np.array([0.5, 1.5], dtype=float),
            4.0,
            np.array([2.0, 8.0], dtype=float),
            1.0,
        )

    def calc_PowerI(self, dmps: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return dmps * 2.0


def _make_maxap() -> MaximumAPosteriori:
    maxap = object.__new__(MaximumAPosteriori)
    maxap.k_vals = np.array([0.1, 0.2], dtype=float)
    maxap.pspp = FakePosterior()
    maxap.T = np.eye(2)
    return maxap


def test_calculate_dmps_returns_array_for_scalar_ps() -> None:
    maxap = _make_maxap()

    dmps = maxap.calculate_dmps(ps=2.0)

    assert isinstance(dmps, np.ndarray)
    assert dmps.shape == maxap.k_vals.shape


def test_map_estimate_handles_uniform_prior_indices() -> None:
    maxap = _make_maxap()

    map_coeffs, map_vis, log_post = maxap.map_estimate(dmps=np.array([3.0, 4.0]))

    np.testing.assert_allclose(map_coeffs, np.array([0.5, 1.5]))
    np.testing.assert_allclose(map_vis, np.array([0.5, 1.5]))
    assert isinstance(log_post, float)


def test_calculate_prior_covariance_uses_dmps_array() -> None:
    maxap = _make_maxap()

    prior_cov = maxap.calculate_prior_covariance(dmps=np.array([1.0, 2.0]))

    np.testing.assert_allclose(prior_cov, np.array([2.0, 4.0]))
