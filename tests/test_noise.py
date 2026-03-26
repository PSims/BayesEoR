import numpy as np

from bayeseor.model.noise import generate_gaussian_noise


def test_generate_gaussian_noise_sigma_zero_returns_signal_and_zero_noise() -> None:
    signal = np.array([1 + 2j, 3 + 4j])
    uvw = np.array([[1.0, 0.0, 0.0], [-1.0, -0.0, 0.0]])
    redundancy = np.ones((2, 1))

    data, noise, conj_map = generate_gaussian_noise(
        sigma=0.0,
        s=signal,
        nf=1,
        nt=1,
        uvw_array_meters=uvw,
        bl_redundancy_array=redundancy,
    )

    np.testing.assert_array_equal(data, signal)
    np.testing.assert_array_equal(noise, np.zeros_like(signal))
    assert conj_map == {0: 1}


def test_generate_gaussian_noise_is_deterministic_for_integer_seed() -> None:
    signal = np.zeros(4, dtype=complex)
    uvw = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, -0.0, 0.0],
            [2.0, 0.0, 0.0],
            [-2.0, -0.0, 0.0],
        ]
    )
    redundancy = np.ones((4, 1))

    result1 = generate_gaussian_noise(
        sigma=2.0,
        s=signal,
        nf=1,
        nt=1,
        uvw_array_meters=uvw,
        bl_redundancy_array=redundancy,
        random_seed=7,
    )
    result2 = generate_gaussian_noise(
        sigma=2.0,
        s=signal,
        nf=1,
        nt=1,
        uvw_array_meters=uvw,
        bl_redundancy_array=redundancy,
        random_seed=7,
    )

    np.testing.assert_allclose(result1[0], result2[0])
    np.testing.assert_allclose(result1[1], result2[1])
    assert result1[2] == result2[2]


def test_generate_gaussian_noise_enforces_hermitian_pairs() -> None:
    signal = np.zeros(2, dtype=complex)
    uvw = np.array([[1.0, 0.0, 0.0], [-1.0, -0.0, 0.0]])
    redundancy = np.ones((2, 1))

    _, noise, conj_map = generate_gaussian_noise(
        sigma=1.0,
        s=signal,
        nf=1,
        nt=1,
        uvw_array_meters=uvw,
        bl_redundancy_array=redundancy,
        random_seed=3,
    )

    assert conj_map == {0: 1}
    assert np.allclose(noise[1], noise[0].conjugate())
