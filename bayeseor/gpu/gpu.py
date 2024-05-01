import numpy as np
from pathlib import Path
from sysconfig import get_paths

from ..utils import mpiprint

class GPUInterface(object):
    """
    Class to interface with GPUs.

    Parameters
    ----------
    base_dir : str
        Path to directory containing a shared library.  Defaults to the parent
        directory of `__file__`.
    rank : int
        MPI rank.
    verbose : bool
        If True (default), print status.

    """
    def __init__(self, base_dir=None, rank=0, verbose=True):
        if base_dir is None:
            # Look for MAGMA .so files in environment's lib directory
            self.base_dir = Path(get_paths()['stdlib']).parent
        else:
            self.base_dir = Path(base_dir)
        self.rank = rank
        self.verbose = verbose

        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            import ctypes
            from numpy import ctypeslib

            so_path = self.base_dir / 'libmagma.so'
            if self.verbose:
                mpiprint(
                    f'Loading shared library from {so_path}',
                    rank=self.rank
                )
            # libmagma.so contains functions from the Matrix Algebra for
            # GPU and Multicore Architectures (MAGMA) library.  We use
            # the function magma_zpotrf from include/magma_z.h which
            # computes the Cholesky decomposition of a complex, Hermitian,
            # positive-definite matrix.  We must also call magma_init
            # (magma_finalize) before (after) the magma_zpotrf call to
            # initialize (finalze) GPU communications.
            magma = ctypes.CDLL(so_path)
            self.magma_init = magma.magma_init
            self.magma_finalize = magma.magma_finalize
            self.magma_zpotrf = magma.magma_zpotrf
            # The function magma_zpotrf takes as arguments
            # 1. uplo (int): 121 or 122 to keep the upper- or lower-triangular
            #                matrix in memory from the decomposition
            # 2. n (int): The order of the matrix dA
            # 3. dA (complex pointer): pointer to the matrix dA with shape
            #                          (ldda, n)
            # 4. ldda (int): The leading dimension of dA
            # 5. info (int pointer): Info flag which is only zero if the
            #                        function executed successfully
            self.magma_zpotrf.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypeslib.ndpointer(np.complex128, ndim=2, flags='C'),
                ctypes.c_int,
                ctypeslib.ndpointer(int, ndim=1, flags='C')
            ]
            if self.verbose:
                mpiprint('Computing on GPU(s)', rank=self.rank, end='\n\n')
            self.gpu_initialized = True

        except Exception as e:
            self.gpu_initialized = False
            if self.verbose:
                mpiprint(
                    '\nException loading GPU encountered...', rank=self.rank
                )
                mpiprint(repr(e), rank=self.rank)