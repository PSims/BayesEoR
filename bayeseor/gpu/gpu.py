import numpy as np
from pathlib import Path

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
            self.base_dir = Path(__file__).parent
        else:
            self.base_dir = Path(base_dir)
        self.rank = rank
        self.verbose = verbose

        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            import ctypes
            from numpy import ctypeslib

            so_path = self.base_dir / 'wrapmzpotrf.so'
            if self.verbose:
                mpiprint(
                    f'Loading shared library from {so_path}',
                    rank=self.rank
                )
            # wrapmzpotrf contains a python wrapper function of the C routine
            # from the Matrix Algebra for GPU and Multicore Architectures
            # (MAGMA) library called mzpotrf which computes the Cholesky
            # decomposition of a complex, Hermitian, positive-definite matrix.
            wrapmzpotrf = ctypes.CDLL(so_path)
            # nrhs specifies the number of columns in the object on the right
            # hand side of the linear system Ax=b.  In our case, b is a column
            # vector and thus nrhs = 1 (1 column).
            self.nrhs = 1
            wrapmzpotrf.cpu_interface.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypeslib.ndpointer(np.complex128, ndim=2, flags='C'),
                ctypeslib.ndpointer(np.complex128, ndim=1, flags='C'),
                ctypes.c_int,
                ctypeslib.ndpointer(int, ndim=1, flags='C')]
            self.wrapmzpotrf = wrapmzpotrf
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