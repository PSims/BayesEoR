import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits


def test_build_Finv(build_matrices):
    BM = build_matrices

    BM.build_Finv()
    if BM.use_sparse_matrices:
        suffix = ".npz"
    else:
        suffix = ".h5"
    Finv = BM.read_data(BM.array_save_directory, f"Finv{suffix}")

    Finv_shape = (
        # Visibility axis
        # Number of visibilities in the data vector
        BM.nbls*BM.nf*BM.nt,
        # Sky model axis
        # Number of HEALPix pixels in the model FoV across all frequencies
        BM.hpx.npix_fov*BM.nf
    )

    assert Finv.shape == Finv_shape

def test_build_Fprime(build_matrices):
    BM = build_matrices

    BM.build_Fprime()
    if BM.use_sparse_matrices:
        suffix = ".npz"
    else:
        suffix = ".h5"
    Fprime = BM.read_data(BM.array_save_directory, f"Fprime{suffix}")

    Fprime_shape = (
        # Sky model axis
        # Number of HEALPix pixels in the model FoV across all frequencies
        BM.hpx.npix_fov*BM.nf,
        # Combined EoR + FG model (u, v, f) cube axis
        # Number of EoR model parameters
        (BM.nu*BM.nv - 1)*(BM.neta - 1)
        # Number of FG model parameters
        + (BM.nu_fg*BM.nv_fg - (not BM.fit_for_monopole))*BM.neta
        # FIXME: add support for the SHG uv-plane (issue #50)
    )

    assert Fprime.shape == Fprime_shape

def test_build_T(build_matrices):
    BM = build_matrices

    # Ensure that pre-requisite matrices are built
    # FIXME: add new functionality for T = Finv_Fprime * Fz
    BM.build_Finv()
    BM.build_Fprime_Fz()

    # Test build_T with a single thread
    with threadpool_limits(limits=1):
        BM.build_T()
    # T always saved as a dense matrix
    T_single = BM.read_data(BM.array_save_directory, "T.h5")

    T_shape = (
        # Visibility axis
        # Number of visibilities in the data vector
        BM.nbls*BM.nf*BM.nt,
        # Combined EoR + FG model (u, v, eta) cube axis
        # Number of EoR model parameters
        (BM.nu*BM.nv - 1)*(BM.neta - 1)
        # Number of eta=0 FG model parameters accounting
        # for the Large Spectral Scale Model (LSSM)
        + BM.nu_fg*BM.nv_fg*(1 + BM.nq)*(BM.nq > 0)
        # Number of (u, v)=(0, 0) monopole model parameters
        + (BM.neta - 1)*BM.fit_for_monopole
    )

    assert T_single.shape == T_shape

    Nthreads = threadpool_info["num_threads"]
    if Nthreads > 1:
        BM.build_T()
        T_multi = BM.read_data(BM.array_save_directory, "T")

        assert T_multi.shape == T_shape

        # Ensure that T is identical with or without threading
        x = 1j*np.random.normal(0, 1, T_shape[1])
        x += np.random.normal(0, 1, T_shape[1])
        T_single_x = T_single @ x
        T_multi_x = T_multi @ x
        diff = T_single_x - T_multi_x
        assert np.abs(diff).max() <= 1e-10
