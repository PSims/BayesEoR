import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from jsonargparse import Namespace
from matplotlib.figure import Figure

from bayeseor.analyze.analyze import DataContainer


def _make_data_container(n_dirs: int = 1, n_kbins: int = 3) -> DataContainer:
    container = object.__new__(DataContainer)
    container.Ndirs = n_dirs
    container.labels = None
    container.k_vals_identical = True
    base_k_vals = np.array([0.1, 0.2, 0.4], dtype=float)[:n_kbins]
    container.k_vals = [base_k_vals.copy() for _ in range(n_dirs)]
    container.k_vals_bins = [
        np.array([0.05, 0.15, 0.3, 0.6], dtype=float)[: n_kbins + 1]
        for _ in range(n_dirs)
    ]
    container.posteriors = [
        np.array(
            [[1.0, 2.0, 1.0], [0.8, 1.5, 0.8], [0.5, 1.0, 0.5]],
            dtype=float,
        )[:n_kbins]
        for _ in range(n_dirs)
    ]
    container.posterior_bins = [
        np.array(
            [[0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.6, 0.8], [0.4, 0.8, 1.2, 1.6]],
            dtype=float,
        )[:n_kbins]
        for _ in range(n_dirs)
    ]
    medians = np.array([1.0, 2.0, 4.0], dtype=float)[:n_kbins]
    container.avgs = [medians.copy() for _ in range(n_dirs)]
    container.medians = [medians.copy() for _ in range(n_dirs)]
    container.cred_intervals = [
        {
            68: {
                "lo": medians - 0.2,
                "hi": medians + 0.3,
            }
        }
        for _ in range(n_dirs)
    ]
    container.calc_uplims = True
    container.uplims = [medians + 0.5 for _ in range(n_dirs)]
    container.uplim_inds = None
    container.calc_kurtosis = False
    container.has_expected = np.bool_(True)
    container.expected_ps = []
    container.expected_dmps = [medians * 0.9 for _ in range(n_dirs)]
    container.ps_kind = "dmps"
    container.temp_unit = "mK"
    container.k_units = "Mpc$^{-1}$"
    container.ps_label = r"$\Delta^2(k)$"
    container.ps_units = "mK$^2$"
    container.args = [
        Namespace(priors=[[-2.0, 1.0]] * n_kbins, log_priors=True)
        for _ in range(n_dirs)
    ]
    return container


def test_get_posterior_data_returns_optional_outputs(tmp_path):
    container = object.__new__(DataContainer)
    container.sampler = "multinest"
    container.calc_uplims = True
    container.calc_kurtosis = True

    samples = np.array(
        [
            [0.25, 0.0, -1.0, 0.0],
            [0.75, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    sample_path = tmp_path / "data-.txt"
    np.savetxt(sample_path, samples)

    out = container.get_posterior_data(
        sample_path,
        2,
        posterior_weighted=True,
        return_samples=True,
    )

    assert out[0].shape == (2, 31)
    assert out[1].shape == (2, 32)
    assert out[5] is not None and out[5].shape == (2,)
    assert out[6] is not None and out[6].shape == (2,)
    assert out[7] is not None and out[7].shape == (2, 4)


def test_plot_power_spectra_defaults_uplims_and_returns_figure():
    container = _make_data_container()

    fig = container.plot_power_spectra()

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_plot_power_spectra_accepts_scalar_external_axis():
    container = _make_data_container()
    fig, ax = plt.subplots()

    returned_axes = container.plot_power_spectra(fig=fig, axs=ax)

    assert len(returned_axes) == 1
    assert returned_axes[0] is ax
    plt.close(fig)


def test_plot_posteriors_accepts_scalar_external_axis():
    container = _make_data_container(n_kbins=1)
    fig, ax = plt.subplots()

    returned_axes = container.plot_posteriors(fig=fig, axs=ax)

    assert isinstance(returned_axes, list)
    assert len(returned_axes) == 1
    assert returned_axes[0] is ax
    plt.close(fig)


def test_plot_power_spectra_and_posteriors_smoke():
    container = _make_data_container()

    fig = container.plot_power_spectra_and_posteriors(
        plot_diff=True,
        uplim_inds=np.array([[False, True, False]], dtype=bool),
    )

    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 3
    plt.close(fig)
