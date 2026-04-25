import math
from typing import Any, cast

import astropy.units as u
from astropy import constants as const
from astropy.cosmology import Planck18

from bayeseor.cosmology import Cosmology


def test_f2z_and_z2f_are_consistent_for_float_frequency() -> None:
    cosmology = Cosmology()
    freq_hz = 150e6

    z = cosmology.f2z(freq_hz)
    recovered_freq_hz = cosmology.z2f(z)

    assert math.isclose(recovered_freq_hz, freq_hz, rel_tol=1e-12)


def test_f2z_accepts_astropy_quantity() -> None:
    cosmology = Cosmology()
    freq = 150 * u.MHz

    z = cosmology.f2z(freq)
    f_21_hz = float(cast(Any, 1420.40575177 * u.MHz).to_value(u.Hz))
    freq_hz = float(cast(Any, freq).to_value(u.Hz))
    expected = f_21_hz / freq_hz - 1.0

    assert math.isclose(z, expected, rel_tol=1e-12)


def test_distance_conversions_match_planck18() -> None:
    cosmology = Cosmology()
    z = 8.0

    planck18 = cast(Any, Planck18)
    expected_dldth = float(
        cast(Any, planck18.comoving_transverse_distance(z)).to_value(u.Mpc)
    )

    assert math.isclose(cosmology.dL_dth(z), expected_dldth, rel_tol=1e-12)

    # Reconstruct the expected dL/df using the same physical relation.
    c_km_s = float(cast(Any, const.c).to_value(u.km / u.s))  # pyright: ignore
    h0 = float(cast(Any, planck18.H0).to_value(u.km / (u.s * u.Mpc)))
    e_z = float(planck18.efunc(z))
    f21_hz = float(cast(Any, 1420.40575177 * u.MHz).to_value(u.Hz))
    expected_dldf = c_km_s / h0 / e_z * (1 + z) ** 2 / f21_hz

    assert math.isclose(cosmology.dL_df(z), expected_dldf, rel_tol=1e-12)


def test_inst_to_cosmo_vol_is_product_of_component_conversions() -> None:
    cosmology = Cosmology()
    z = 8.0

    expected = cosmology.dL_dth(z) ** 2 * cosmology.dL_df(z)

    assert math.isclose(cosmology.inst_to_cosmo_vol(z), expected, rel_tol=1e-12)
