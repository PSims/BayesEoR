import numpy as np
import pytest

from bayeseor.model import Healpix


def test_uniform_beam_smoke_path():
    healpix = Healpix(
        fov_ra_eor=10.0,
        jd_center=2458000.5,
        nside=8,
        beam_type="uniform",
    )

    pix = healpix.get_pixel_filter(10.0, 10.0)
    pix_radec, ra, dec = healpix.get_pixel_filter(
        10.0,
        10.0,
        return_radec=True,
    )
    ls, ms, ns = healpix.calc_lmn_from_radec(healpix.jd_center, ra, dec)
    _, _, _, az, za = healpix.calc_lmn_from_radec(
        healpix.jd_center,
        ra,
        dec,
        return_azza=True,
    )
    beam = healpix.get_beam_vals(az, za)

    assert pix.ndim == 1
    assert pix_radec.shape == ra.shape
    assert ra.shape == dec.shape
    assert ls.shape == ms.shape
    assert ms.shape == ns.shape
    assert np.all(beam == 1.0)


def test_gaussian_beam_smoke_path():
    healpix = Healpix(
        fov_ra_eor=10.0,
        jd_center=2458000.5,
        nside=8,
        beam_type="gaussian",
        fwhm_deg=12.0,
    )

    _, ra, dec = healpix.get_pixel_filter(10.0, 10.0, return_radec=True)
    _, _, _, az, za = healpix.calc_lmn_from_radec(
        healpix.jd_center,
        ra,
        dec,
        return_azza=True,
    )
    beam = healpix.get_beam_vals(az, za)

    assert beam.shape == za.shape
    assert np.all(beam >= 0.0)
    assert np.all(beam <= healpix.peak_amp)


def test_missing_required_constructor_arguments_fail_fast():
    with pytest.raises(AssertionError, match="jd_center"):
        Healpix(fov_ra_eor=10.0, nside=8)

    with pytest.raises(AssertionError, match="dt must not be None"):
        Healpix(
            fov_ra_eor=10.0,
            jd_center=2458000.5,
            nside=8,
            nt=2,
        )
