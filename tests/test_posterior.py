import numpy as np
import pytest

from bayeseor.posterior import PowerSpectrumPosteriorProbability, PriorC


def _build_minimal_pspp(
    *, dimensionless_PS: bool = True
) -> PowerSpectrumPosteriorProbability:
    return PowerSpectrumPosteriorProbability(
        T_Ninv_T=np.eye(2, dtype=complex),
        dbar=np.zeros(2, dtype=complex),
        k_vals=np.array([0.1]),
        k_cube_voxels_in_bin=[np.array([0, 1])],
        nuv=2,
        neta=2,
        nf=2,
        nq=0,
        Ninv=np.eye(2),
        d_Ninv_d=np.array([1.0]),
        redshift=8.0,
        ps_box_size_ra_Mpc=1.0,
        ps_box_size_dec_Mpc=1.0,
        ps_box_size_para_Mpc=1.0,
        use_gpu=False,
        dimensionless_PS=dimensionless_PS,
    )


def test_priorc_maps_unit_cube_to_prior_ranges() -> None:
    prior = PriorC([[1.0, 3.0], [10.0, 14.0]])

    theta = prior.prior_func([0.25, 0.5])

    assert theta == [1.5, 12.0]


def test_calc_poweri_returns_expected_length_for_dimensionless_path() -> None:
    pspp = _build_minimal_pspp()

    power_i = pspp.calc_PowerI(np.array([2.0]))

    assert power_i.shape == (2,)
    expected_norm = pspp.calc_physical_dimensionless_power_spectral_normalisation(0)
    assert np.allclose(power_i, expected_norm / 2.0)


def test_calc_poweri_rejects_unimplemented_physical_pk_path() -> None:
    pspp = _build_minimal_pspp(dimensionless_PS=False)

    with pytest.raises(NotImplementedError, match="dimensionless_PS=True"):
        pspp.calc_PowerI(np.array([1.0]))
