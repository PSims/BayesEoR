from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from astropy import units
from astropy.time import Time

import bayeseor.vis as vis_module


class FakeVectorUVData:
    last_phase_time: Time | None = None

    def __init__(
        self,
        antpairs: list[tuple[int, int]],
        data_by_antpair: dict[tuple[int, int], np.ndarray[Any, Any]],
        times: np.ndarray[Any, Any],
    ) -> None:
        self.antpairs = antpairs
        self._antpair_index = {antpair: i for i, antpair in enumerate(antpairs)}
        self.Nbls = len(antpairs)
        self.Ntimes = len(times)
        self.Nfreqs = next(iter(data_by_antpair.values())).shape[1]
        self.Nblts = self.Nbls * self.Ntimes
        self.Npols = 1
        self.freq_array = np.arange(self.Nfreqs, dtype=float)
        self.time_array = np.repeat(times, self.Nbls)
        self.polarization_array = np.array([vis_module.polstr2num("xx")])
        self.phase_calls: list[Time] = []
        self._phasor_template = np.exp(
            1j
            * np.array(
                [
                    [0.0, np.pi / 4],
                    [np.pi / 2, 3 * np.pi / 4],
                    [np.pi, 5 * np.pi / 4],
                ][: self.Ntimes]
            )
        )
        self.data_array = np.zeros((self.Nblts, self.Nfreqs, 1), dtype=complex)
        for antpair, vis in data_by_antpair.items():
            self.data_array[self.antpair2ind(*antpair), :, 0] = vis

    def copy(self) -> "FakeVectorUVData":
        return deepcopy(self)

    def antpair2ind(self, ant1: int, ant2: int) -> np.ndarray[Any, Any]:
        i_bl = self._antpair_index[(ant1, ant2)]
        return np.arange(i_bl, self.Nblts, self.Nbls)

    def get_data(
        self,
        antpairpol: tuple[int, int, str],
        force_copy: bool = False,
    ) -> np.ndarray[Any, Any]:
        ant1, ant2, _ = antpairpol
        data = self.data_array[self.antpair2ind(ant1, ant2), :, 0]
        if force_copy:
            return data.copy()
        return data

    def phase_to_time(self, phase_time: Time) -> None:
        self.phase_calls.append(phase_time)
        type(self).last_phase_time = phase_time
        for antpair in self.antpairs:
            inds = self.antpair2ind(*antpair)
            self.data_array[inds, :, 0] *= self._phasor_template


class FakePIUVData:
    def __init__(self) -> None:
        self.polarization_array = np.array(
            [vis_module.polstr2num("xx"), vis_module.polstr2num("yy")]
        )
        self.data_array = np.array([[[1.0 + 1.0j, 2.0 - 1.0j]]], dtype=complex)

    def select(self, polarizations: list[str]) -> None:
        if polarizations != ["xx"]:
            raise AssertionError("Unexpected polarization selection")
        xx_ind = np.where(self.polarization_array == vis_module.polstr2num("xx"))[0]
        self.data_array = self.data_array[..., xx_ind]
        self.polarization_array = self.polarization_array[xx_ind]


class FakePreprocessUVData:
    def __init__(self) -> None:
        self.vis_units = "Jy"
        self._antpairs = [(0, 1), (1, 2)]
        self._times = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
        self._freqs = np.array([100e6, 110e6, 120e6, 130e6], dtype=float)
        self.polarization_array = np.array([vis_module.polstr2num("xx")])
        self.Npols = 1
        self.blt_order = ("time", "baseline")
        self._unprojected = True
        self._set_data_shapes()

    def _set_data_shapes(self) -> None:
        self.Nbls = len(self._antpairs)
        self.Ntimes = len(self._times)
        self.Nfreqs = len(self._freqs)
        self.Nblts = self.Nbls * self.Ntimes
        self.freq_array = self._freqs.copy()
        self.time_array = np.repeat(self._times, self.Nbls)
        self.data_array = np.ones((self.Nblts, self.Nfreqs, 1), dtype=complex)
        self.flag_array = np.zeros((self.Nblts, self.Nfreqs, 1), dtype=bool)
        self.uvw_array = np.tile(
            np.array([[14.0, 0.0, 0.0], [28.0, 0.0, 0.0]], dtype=float),
            (self.Ntimes, 1),
        )

    def read(
        self,
        fp: Path,
        read_data: bool = True,
        times: np.ndarray[Any, Any] | None = None,
        frequencies: np.ndarray[Any, Any] | None = None,
        bls: list[tuple[int, int]] | None = None,
    ) -> None:
        self.last_read = {
            "fp": fp,
            "read_data": read_data,
            "times": None if times is None else np.asarray(times, dtype=float),
            "frequencies": (
                None if frequencies is None else np.asarray(frequencies, dtype=float)
            ),
            "bls": bls,
        }
        if times is not None:
            self._times = np.asarray(times, dtype=float)
        if frequencies is not None:
            self._freqs = np.asarray(frequencies, dtype=float)
        if bls is not None:
            self._antpairs = list(bls)
        self._set_data_shapes()

    def select(
        self,
        ant_str: str | None = None,
        blt_inds: np.ndarray[Any, Any] | None = None,
        bls: list[tuple[int, int]] | None = None,
        polarizations: list[str] | None = None,
    ) -> None:
        if bls is not None:
            self._antpairs = list(bls)
            self._set_data_shapes()
        if polarizations is not None and polarizations != ["xx"]:
            raise AssertionError("Unexpected polarization selection")

    def get_antpairs(self) -> list[tuple[int, int]]:
        return list(self._antpairs)

    def get_flags(self, antpair: tuple[int, int]) -> np.ndarray[Any, Any]:
        return np.zeros((self.Ntimes, self.Nfreqs), dtype=bool)

    def _check_for_cat_type(self, cat_type: str) -> np.ndarray[Any, Any]:
        return np.array([self._unprojected], dtype=bool)

    def unproject_phase(self) -> None:
        self._unprojected = True

    def reorder_blts(self, order: str = "time") -> None:
        self.blt_order = ("time", "baseline")

    def antpair2ind(self, antpair: tuple[int, int]) -> np.ndarray[Any, Any]:
        i_bl = self._antpairs.index(antpair)
        return np.arange(i_bl, self.Nblts, self.Nbls)

    def antnums_to_baseline(self, ant1: int, ant2: int) -> int:
        return ant1 * 1000 + ant2


def test_jy_to_ksr_preserves_shape_and_matches_quantity_conversion() -> None:
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    freqs = units.Quantity([150e6, 160e6], unit=units.Hz)

    result = vis_module.jy_to_ksr(data, freqs)

    equiv = units.brightness_temperature(freqs, beam_area=1 * units.sr)
    conv_factor = np.asarray(
        units.Quantity(1.0, unit=units.Jy).to_value(units.K, equivalencies=equiv),
        dtype=float,
    )
    expected = data * conv_factor[np.newaxis, :]
    np.testing.assert_allclose(result, expected)
    assert result.shape == data.shape


def test_form_pI_vis_combines_xx_and_yy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vis_module, "UVData", FakePIUVData)
    uvd = FakePIUVData()

    result = vis_module.form_pI_vis(uvd, norm=0.5)

    np.testing.assert_allclose(result.data_array[..., 0], np.array([[1.5 + 0.0j]]))
    np.testing.assert_array_equal(
        result.polarization_array, np.array([vis_module.polstr2num("pI")])
    )


def test_uvd_to_vector_preserves_vector_ordering() -> None:
    antpairs = [(0, 1), (1, 2)]
    uvd = FakeVectorUVData(
        antpairs=antpairs,
        data_by_antpair={
            (0, 1): np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex),
            (1, 2): np.array([[10.0, 20.0], [30.0, 40.0]], dtype=complex),
        },
        times=np.array([1.0, 2.0]),
    )

    vis_vec, phasor_vec, noise_vec = vis_module.uvd_to_vector(uvd, antpairs)

    assert vis_vec is not None
    expected = np.array(
        [
            1.0,
            10.0,
            1.0,
            10.0,
            2.0,
            20.0,
            2.0,
            20.0,
            3.0,
            30.0,
            3.0,
            30.0,
            4.0,
            40.0,
            4.0,
            40.0,
        ],
        dtype=complex,
    )
    np.testing.assert_allclose(vis_vec, expected)
    assert phasor_vec is None
    assert noise_vec is None


def test_uvd_to_vector_noise_differencing() -> None:
    antpairs = [(0, 1)]
    uvd = FakeVectorUVData(
        antpairs=antpairs,
        data_by_antpair={
            (0, 1): np.array([[1.0, 2.0], [5.0, 7.0], [9.0, 10.0]], dtype=complex)
        },
        times=np.array([1.0, 2.0, 3.0]),
    )

    _, _, noise_vec = vis_module.uvd_to_vector(uvd, antpairs, calc_noise=True)

    assert noise_vec is not None
    expected_noise = np.array(
        [
            -4.0,
            -4.0,
            -5.0,
            -5.0,
            -4.0,
            -4.0,
            -5.0,
            -5.0,
            4.0,
            4.0,
            3.0,
            3.0,
        ],
        dtype=complex,
    )
    np.testing.assert_allclose(noise_vec, expected_noise)


def test_uvd_to_vector_uses_central_time_when_phase_time_is_none() -> None:
    antpairs = [(0, 1)]
    FakeVectorUVData.last_phase_time = None
    uvd = FakeVectorUVData(
        antpairs=antpairs,
        data_by_antpair={
            (0, 1): np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=complex)
        },
        times=np.array([10.0, 20.0, 30.0]),
    )

    _, phasor_vec, _ = vis_module.uvd_to_vector(uvd, antpairs, phase=True)

    assert phasor_vec is not None
    assert uvd.phase_calls == []
    assert FakeVectorUVData.last_phase_time is not None
    assert FakeVectorUVData.last_phase_time.jd == Time(20.0, format="jd").jd
    expected_phasor = np.array(
        [
            1.0 + 0.0j,
            1.0 - 0.0j,
            np.exp(1j * np.pi / 4),
            np.exp(-1j * np.pi / 4),
            1j,
            -1j,
            np.exp(1j * 3 * np.pi / 4),
            np.exp(-1j * 3 * np.pi / 4),
            -1.0 + 0.0j,
            -1.0 - 0.0j,
            np.exp(1j * 5 * np.pi / 4),
            np.exp(-1j * 5 * np.pi / 4),
        ],
        dtype=complex,
    )
    np.testing.assert_allclose(phasor_vec, expected_phasor)


def test_preprocess_uvdata_downselects_frequency_and_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_uvd = FakePreprocessUVData()
    monkeypatch.setattr(vis_module, "UVData", lambda: fake_uvd)
    monkeypatch.setattr(vis_module, "jy_to_ksr", lambda data, freqs, mK=True: data)

    def fake_uvd_to_vector(
        uvd: FakePreprocessUVData,
        antpairs: list[tuple[int, int]],
        **_: Any,
    ) -> tuple[np.ndarray[Any, Any], None, None]:
        n_vis = 2 * len(antpairs) * uvd.Ntimes * uvd.Nfreqs
        return np.arange(n_vis, dtype=complex), None, None

    monkeypatch.setattr(vis_module, "uvd_to_vector", fake_uvd_to_vector)

    fp = tmp_path / "fake.uvh5"
    fp.touch()

    vis, antpairs, uvws, redundancy, phasor, noise, uvd = vis_module.preprocess_uvdata(
        fp,
        form_pI=False,
        pol="xx",
        freq_center=120e6,
        Nfreqs=3,
        jd_center=30.0,
        Ntimes=3,
        return_uvd=True,
    )

    assert uvd is fake_uvd
    np.testing.assert_allclose(fake_uvd.freq_array, np.array([110e6, 120e6, 130e6]))
    np.testing.assert_allclose(
        np.unique(fake_uvd.time_array), np.array([20.0, 30.0, 40.0])
    )
    assert len(antpairs) == 4
    assert uvws.shape == (3, 4, 3)
    assert redundancy.shape == (3, 4, 1)
    assert vis.shape == (36,)
    assert phasor is None
    assert noise is None


def test_generate_mock_eor_signal_instrumental_uses_current_healpix_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, Any] = {}

    class FakeHealpix:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)
            self.npix_fov = 3

    monkeypatch.setattr(vis_module, "Healpix", FakeHealpix)

    signal, white_noise_sky = vis_module.generate_mock_eor_signal_instrumental(
        Finv=np.eye(6, dtype=complex),
        nf=2,
        fov_ra_deg=10.0,
        fov_dec_deg=12.0,
        nside=8,
        telescope_latlonalt=(1.0, 2.0, 3.0),
        central_jd=2450000.0,
        nt=1,
        int_time=11.0,
        random_seed=7,
    )

    assert captured_kwargs["fov_ra_eor"] == 10.0
    assert captured_kwargs["fov_dec_eor"] == 12.0
    assert captured_kwargs["jd_center"] == 2450000.0
    assert captured_kwargs["dt"] == 11.0
    assert white_noise_sky.shape == (2, 3)
    assert signal.shape == (6,)
