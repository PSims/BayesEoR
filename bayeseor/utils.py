import hashlib
import os
import pickle
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Protocol, cast

import numpy as np
from rich.console import Console


from . import __version__

cns = Console()
cns.is_jupyter = False


class MPIComm(Protocol):
    def Get_rank(self) -> int: ...

    def bcast(self, value: Any, root: int = 0) -> Any: ...

    def Barrier(self) -> None: ...


def _get_default_mpi_comm() -> MPIComm:
    from mpi4py import MPI

    return cast(MPIComm, MPI.COMM_WORLD)


def mpiprint(*args, rank=0, highlight=False, soft_wrap=True, **kwargs):
    """
    Prints only if root worker.

    """
    if rank == 0:
        cns.print(*args, highlight=highlight, soft_wrap=soft_wrap, **kwargs)


def parse_uprior_inds(upriors_str, nkbins):
    """
    Parse an array indexing string.

    `upriors_str` must follow standard array slicing syntax and include no
    spaces.  Examples of valid strings:
    * '1:4': equivalent to `slice(1, 4)`
    * '1,3,4': equivalent to indexing with `[1, 3, 4]`
    * '3' or '-3'
    * 'all': all bins use a uniform prior
    * '': no bins use a uniform prior

    Parameters
    ----------
    upriors_str : str
        String containing array indexes (follows array slicing syntax).
    nkbins : int
        Number of k-bins.

    Returns
    -------
    uprior_inds : numpy.ndarray
        Boolean array that is True for any k bins using a uniform prior.
        False entries use a log-uniform prior.

    """
    if upriors_str.lower() == "all":
        uprior_inds = np.ones(nkbins, dtype=bool)
    else:
        uprior_inds = np.zeros(nkbins, dtype=bool)
        if upriors_str != "":
            if ":" in upriors_str:
                bounds = list(map(int, upriors_str.split(":")))
                uprior_inds[slice(*bounds)] = True
            elif "," in upriors_str:
                up_inds = list(map(int, upriors_str.split(",")))
                uprior_inds[up_inds] = True
            else:
                uprior_inds[int(upriors_str)] = True

    return uprior_inds


def write_log_files(parser, args, out_dir=Path("./"), verbose=False):
    """
    Write log files containing the current version and analysis parameters.

    Parameters
    ----------
    parser : :class:`..params.BayesEoRParser`
        BayesEoRParser instance.
    args : Namespace
        Namespace object containing parsed command line arguments.
    out_dir : Path or str, optional
        Path to output directory for log files. Defaults to ``Path('./')``.
    verbose : bool, optional
        Verbose output. Defaults to False.

    """
    # Make log file directory if it doesn't exist
    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=False)

    # Write version info
    ver_file = out_dir / "version.txt"
    if not ver_file.exists():
        with open(ver_file, "w") as f:
            f.write(f"{__version__}\n")

    # Write args to disk
    args_file = out_dir / "args.json"
    if not args_file.exists():
        parser.save(args, args_file, format="json", skip_none=False)

    if verbose:
        print(f"Log files written successfully to {out_dir.absolute()}\n")


def save_numpy_dict(
    fp, arr, args, version=__version__, extra=None, clobber=False
):
    """
    Save array to disk with metadata as dictionary.

    Parameters
    ----------
    fp : :class:`pathlib.Path` or str
        File path for dictionary.
    arr : array_like
        Data to write to disk.
    args : dict
        Dictionary of associated metadata.
    version : str
        Version string. Defaults to ``__version__``.
    extra : dict
        Dictionary of extra info. Defaults to None.
    clobber : bool, optional
        Clobber file if it exists. Defaults to False.

    """
    if not isinstance(fp, Path):
        fp = Path(fp)
    if fp.exists() and not clobber:
        raise ValueError(f"clobber is false but file already exists: {fp}")
    if extra is not None:
        if not isinstance(extra, dict):
            raise ValueError("extra must be a dictionary")

    if not fp.parent.exists():
        fp.parent.mkdir(exist_ok=True, parents=True)

    out_dict = {"data": arr, "args": args, "version": version}
    if extra is not None:
        out_dict.update({"extra": extra})

    np.save(fp, np.array(out_dict, dtype=object))


def load_numpy_dict(fp):
    """
    Load data array from disk saved via :func:`.save_numpy_dict`.

    Parameters
    ----------
    fp : :class:`pathlib.Path` or str
        File path for dictionary with contents from
        `bayeseor.vis.save_numpy_dict`.

    """
    if not isinstance(fp, Path):
        fp = Path(fp)
    if not fp.exists():
        raise FileNotFoundError(f"{fp} does not exist")

    return np.load(fp, allow_pickle=True).item()["data"]


def vector_is_hermitian(data, conj_map, nt, nf, nbls, rtol=0, atol=1e-14):
    """
    Checks if the data in the vector `data` is Hermitian symmetric
    based on the mapping contained in `conj_map`.

    Parameters
    ----------
    data : array-like
        Array of values.
    conj_map : dictionary
        Dictionary object which contains the indices in the data vector
        per time and frequency corresponding to baselines and their
        conjugates.

    """
    hermitian = np.zeros(data.size)
    for i_t in range(nt):
        time_ind = i_t * nbls * nf
        for i_freq in range(nf):
            freq_ind = i_freq * nbls
            start_ind = time_ind + freq_ind
            for bl_ind in conj_map.keys():
                conj_bl_ind = conj_map[bl_ind]
                close = np.allclose(
                    data[start_ind + conj_bl_ind],
                    data[start_ind + bl_ind].conjugate(),
                    rtol=rtol,
                    atol=atol,
                )
                if close:
                    hermitian[start_ind + bl_ind] = 1
                    hermitian[start_ind + conj_bl_ind] = 1
    return np.all(hermitian)


def write_map_dict(dir, pspp, bm, n, clobber=False, fn="map-dict.npy"):
    """
    Writes a python dictionary with minimum sufficient info for maximum a
    posteriori (MAP) calculations.  Memory intensive attributes are popped
    before writing to disk since they can be easily loaded later.

    Parameters
    ----------
    dir : str
        Directory in which to save the dictionary.
    pspp : PowerSpectrumPosteriorProbability
        Class containing posterior calculation variables and functions.
    bm : BuildMatrices
        Class containing matrix creation and retrieval functions.
    n : array-like
        Noise vector.
    clobber : bool
        If True, overwrite existing dictionary on disk.
    fn : str
        Filename for dictionary.

    """
    fp = Path(dir) / fn
    if not fp.exists() or clobber:
        pspp_copy = deepcopy(pspp)
        del pspp.T_Ninv_T, pspp.dbar, pspp.Ninv
        map_dict = {"pspp": pspp_copy, "bm": bm, "n": n}
        print(f"\nWriting MAP dict to {fp}\n")
        with open(fp, "wb") as f:
            pickle.dump(map_dict, f, protocol=4)


class ShortTempPathManager:
    """
    Manages the creation and cleanup of a short symbolic link to an output directory.

    This class is designed to handle MultiNest's 100-character path length limitation
    by creating a temporary symbolic link to output_dir with a shorter path
    to pass to MultiNest.

    The symbolic link is created only on MPI rank 0, and errors during creation
    are broadcast to all ranks to prevent deadlocks. If symlink creation fails,
    all ranks will raise a RuntimeError containing the original exception details.

    Attributes
    ----------
    output_dir : pathlib.Path
        The actual output directory.
    tmp_dir : pathlib.Path
        The temporary directory to store symbolic links.
    short_dir : pathlib.Path
        The symbolic link path.
    mpi_comm : MPI.Comm
        The MPI communicator.
    mpi_rank : int
        The rank of the current process.
    """

    def __init__(
        self,
        output_dir: str | Path,
        tmp_dir: str | Path | None = None,
        mpi_comm: Optional[MPIComm] = None,
    ) -> None:
        """
        Initialise the ShortTempPathManager and create the symbolic link.

        Parameters
        ----------
        output_dir : str | pathlib.Path
            The actual output directory. This directory must already exist and be a
            directory.
        tmp_dir : str | pathlib.Path, optional
            The temporary directory to store symbolic links. If None, a relative
            directory (./.mn_chains_symlinks/) is used to avoid non-shared /tmp.
        mpi_comm : MPI.Comm, optional
            The MPI communicator. If None, defaults to MPI.COMM_WORLD.
        """
        self.output_dir: Path = Path(output_dir).absolute()
        if not self.output_dir.exists():
            raise FileNotFoundError(
                f"output_dir '{self.output_dir}' does not exist. Please create it "
                "before initialising ShortTempPathManager."
            )
        if not self.output_dir.is_dir():
            raise NotADirectoryError(
                f"output_dir '{self.output_dir}' is not a directory."
            )

        # Do not use self.tmp_dir: Path = Path(tmp_dir).absolute() otherwise we
        # risk exceeding the path length limit.
        # Instead, use relative paths for the temporary symlink directory.
        # Do not use /tmp/ or tempfile.gettempdir() despite it having a short absolute
        # path because in MPI environments running on clusters, /tmp/ may not be
        # shared across nodes.
        if tmp_dir is None:
            tmp_dir = "./.mn_chains_symlinks/"
        self.tmp_dir: Path = Path(tmp_dir)
        self.mpi_comm: MPIComm = mpi_comm or _get_default_mpi_comm()
        self.mpi_rank: int = self.mpi_comm.Get_rank()

        # Generate the short path using a stronger hash to reduce collision risk
        path_hash: str = hashlib.sha256(
            str(self.output_dir).encode("utf-8")
        ).hexdigest()[:16]

        self.short_dir: Path = self.tmp_dir / f"mn_{path_hash}"

        # Create the symbolic link (only on rank 0) with error handling
        # Initialize on all ranks - these will be overwritten by broadcast
        success = True
        error_msg = None
        exception_type = None
        if self.mpi_rank == 0:
            try:
                self._create_short_path()
            except OSError as e:
                success = False
                error_msg = str(e)
                exception_type = type(e).__name__

        # Broadcast success status to all ranks
        success = self.mpi_comm.bcast(success, root=0)
        error_msg = self.mpi_comm.bcast(error_msg, root=0)
        exception_type = self.mpi_comm.bcast(exception_type, root=0)

        # If creation failed, raise the error on all ranks
        # Note: We wrap the original OSError in a RuntimeError since
        # exception objects cannot be serialized across MPI processes
        if not success:
            raise RuntimeError(
                f"Failed to create symbolic link on rank 0 "
                f"({exception_type}): {error_msg}"
            )

        # Synchronise all ranks to ensure the symbolic link is created
        self.mpi_comm.Barrier()

    def _is_safe_to_remove(self, path: Path) -> bool:
        """
        Return True if `path` resolves inside the configured tmp_dir.
        Protects against accidental deletion outside tmp_dir.
        """
        try:
            resolved = path.resolve()
            tmp_resolved = self.tmp_dir.resolve()
            return str(resolved) == str(tmp_resolved) or str(
                resolved
            ).startswith(str(tmp_resolved) + os.sep)
        except Exception:
            return False

    def _create_short_path(self) -> None:
        """
        Create a short symbolic link to the output directory.
        """
        # Ensure the temporary directory exists
        self.tmp_dir.mkdir(exist_ok=True)

        # Check if symlink already exists and points to the correct target
        if self.short_dir.is_symlink():
            try:
                if self.short_dir.resolve() == self.output_dir.resolve():
                    # Symlink already points to the correct target, no action needed
                    return
            except OSError:
                # If we can't resolve the symlink, it may be broken, so recreate it
                pass
            # Symlink exists but points to wrong target, remove it
            self.short_dir.unlink()
        elif self.short_dir.exists():
            # Exists but is not a symlink (could be file or directory)
            if not self._is_safe_to_remove(self.short_dir):
                raise RuntimeError(
                    f"Refusing to remove {self.short_dir}: "
                    f"it is not inside {self.tmp_dir}"
                )
            if self.short_dir.is_dir():
                print(f"Removing directory: {self.short_dir}")
                shutil.rmtree(self.short_dir)
            else:
                print(f"Removing file: {self.short_dir}")
                self.short_dir.unlink()

        # Create the symbolic link
        self.short_dir.symlink_to(self.output_dir)

    def cleanup(self) -> None:
        """
        Remove the symbolic link if it exists (only on rank 0).
        """
        if self.mpi_rank == 0:
            if self.short_dir.exists() or self.short_dir.is_symlink():
                if self.short_dir.is_dir() and not self.short_dir.is_symlink():
                    print(f"Removing directory: {self.short_dir}")
                    shutil.rmtree(self.short_dir)
                else:
                    print(f"Removing symbolic link: {self.short_dir}")
                    self.short_dir.unlink()

        # Synchronise all ranks to ensure cleanup is complete
        self.mpi_comm.Barrier()

    @property
    def short_out_dir(self) -> Path:
        """
        Get the short symbolic link path.

        Returns:
            Path: The short symbolic link path.
        """
        return self.short_dir


class MultiNestPathManager:
    """
    Manages the creation of a temporary symbolic link for MultiNest output directories
    to work around the 100-character path limitation.

    Steps:
    1. Create out_dir backup for printing on completion of sampling (long_out_dir)
    2a. Create unique short path string (short_out_dir)
    2b. Create symbolic link between out_dir and short_out_dir on rank 0
    3. Reassign out_dir to short_out_dir for MultiNest use
    4. After sampling, clean up symbolic link and print long_out_dir


    Attributes
    ----------
    long_out_dir : Path
        The original long path to the output directory.
    short_out_dir : Path
        The temporary short path created as a symbolic link to the long path.
    rank : int
        The MPI rank of the current process.
    mpi_comm : MPI.Comm
        The MPI communicator to use for coordination.

    Methods
    -------
    setup_multinest_path():
        Sets up the short path for MultiNest and reassigns `out_dir`.
    cleanup():
        Cleans up the symbolic link after MultiNest sampling.
    Examples
    --------
    ```python
    from bayeseor.setup import MultiNestPathManager
    from pathlib import Path
    from mpi4py import MPI

    out_dir = Path("/very/long/path/to/output_directory")
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    path_manager = MultiNestPathManager(out_dir, rank, mpi_comm=mpi_comm)

    # Setup MultiNest path
    out_dir = path_manager.setup_multinest_path()

    # Perform MultiNest sampling...

    # Cleanup after sampling
    path_manager.cleanup()
    ```
    """

    def __init__(
        self, out_dir: Path, rank: int, mpi_comm: Optional[MPIComm] = None
    ):
        """
        Initializes the MultiNestPathManager.

        Parameters
        ----------
        out_dir : Path
            The original long path to the output directory.
        rank : int
            The MPI rank of the current process.
        mpi_comm : MPI.Comm, optional
            The MPI communicator to use for coordination.
            If None, defaults to MPI.COMM_WORLD.
        """
        self.rank = rank
        self.mpi_comm = mpi_comm
        # Step 1.
        self.long_out_dir = out_dir
        # Step 2a.
        self.short_path_manager = ShortTempPathManager(
            out_dir, mpi_comm=mpi_comm
        )
        # Step 2b.
        self.short_out_dir = self.short_path_manager.short_out_dir

    def setup_multinest_path(self) -> Path:
        """
        Sets up the short path for MultiNest and reassigns `out_dir`.
        Call this for Step 3.

        Returns
        -------
        Path
            The short path to be used as the MultiNest output directory.
        """
        if self.rank == 0:
            # Print resolved locations to help users locate the symlink and tmp dir
            try:
                tmp_resolved = self.short_path_manager.tmp_dir.resolve()
            except Exception as e:
                print(f"\nWarning: could not resolve tmp_dir path due to: {e}")
                print("Using un-resolved tmp_dir path instead.")
                tmp_resolved = self.short_path_manager.tmp_dir

            mpiprint(
                f"\nCreated short path: symlink {self.short_out_dir} pointing to {self.long_out_dir}"
            )
            mpiprint(f"\nTemporary symlink directory (tmp_dir): {tmp_resolved}")

            mpiprint(f"\nMultiNest output base: {self.short_out_dir}")

        return self.short_out_dir

    def cleanup(self) -> None:
        """
        Cleans up the symbolic link after MultiNest sampling.
        Call this for Step 4.
        """
        self.short_path_manager.cleanup()
        if self.rank == 0:
            mpiprint(
                f"Final MultiNest output base: {self.long_out_dir}",
            )
