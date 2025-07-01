import numpy as np
import pickle
from pathlib import Path
from copy import deepcopy

from rich.console import Console
cns = Console()

from ..model.healpix import Healpix
from .. import __version__

def mpiprint(*args, rank=0, highlight=False, soft_wrap=True, **kwargs):
    """
    Prints only if root worker.

    """
    if rank == 0:
        cns.print(*args, highlight=highlight, soft_wrap=soft_wrap, **kwargs)


def write_log_files(parser, args):
    """
    Write log files containing the current version and analysis parameters.

    Parameters
    ----------
    parser : BayesEoRParser
        BayesEoRParser instance.
    args : Namespace
        Namespace object containing command line and analysis parameters.

    """
    # Make log file directory if it doesn't exist
    out_dir = Path(args.output_dir) / args.file_root
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

    print(f"Log files written successfully to {out_dir.absolute()}")


def save_numpy_dict(fp, arr, args, version=__version__, clobber=False):
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
        Version string.  Defaults to ``__version__``.
    clobber : bool, optional
        Clobber file if it exists.  Defaults to False.

    """
    if not isinstance(fp, Path):
        fp = Path(fp)
    if fp.exists() and not clobber:
        raise ValueError(
            f"clobber is false but file already exists: {fp}"
        )
    if not fp.parent.exists():
        fp.parent.mkdir(exist_ok=True, parents=True)

    np.save(fp, {"data": arr, "args": args, "version": version})


def load_numpy_dict(fp):
    """
    Load array from disk saved via :func:`.save_numpy_dict`.

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
                    data[start_ind+conj_bl_ind],
                    data[start_ind+bl_ind].conjugate(),
                    rtol=rtol,
                    atol=atol
                )
                if close:
                    hermitian[start_ind+bl_ind] = 1
                    hermitian[start_ind+conj_bl_ind] = 1
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
        map_dict = {
            "pspp": pspp_copy,
            "bm": bm,
            "n": n
        }
        print(f"\nWriting MAP dict to {fp}\n")
        with open(fp, "wb") as f:
            pickle.dump(map_dict, f, protocol=4)


class ArrayIndexing(object):
    """
    Class for convenient vector slicing in various data spaces.

    """
    def __init__(
            self, nu_eor=15, nv_eor=15, nu_fg=15, nv_fg=15, nf=38,
            neta=38, nq=0, nt=34, ffm=False, nside=128,
            fov_ra_eor=12.9080728652, fov_dec_eor=None,
            fov_ra_fg=12.9080728652, fov_dec_fg=None,
            tele_latlonalt=(0, 0, 0), central_jd=2458098.3065661727
            ):
        hpx = Healpix(
            fov_ra_deg=fov_ra_eor,
            fov_dec_deg=fov_dec_eor,
            nside=nside,
            telescope_latlonalt=tele_latlonalt,
            central_jd=central_jd
        )
        self.npix_eor = hpx.npix_fov

        hpx = Healpix(
            fov_ra_deg=fov_ra_fg,
            fov_dec_deg=fov_dec_fg,
            nside=nside,
            telescope_latlonalt=tele_latlonalt,
            central_jd=central_jd
        )
        self.npix_fg = hpx.npix_fov

        # Joint params
        self.nf = nf
        self.neta = neta
        self.ffm = ffm  # fit for monopole
        self.nt = nt

        # EoR model
        self.nu_eor = nu_eor
        self.nv_eor = nv_eor
        self.nuv_eor = self.nu_eor * self.nv_eor - 1

        # FG model
        self.nu_fg = nu_fg
        self.nv_fg = nv_fg
        self.nuv_fg = self.nu_fg * self.nv_fg - (not self.ffm)
        self.nq = nq

        # uveta vector
        self.neor_uveta = (self.nu_eor*self.nv_eor - 1) * (self.neta - 1)
        self.neta0_uveta = self.nuv_fg
        self.nlssm_uveta = self.nuv_fg * nq
        self.nmp_uveta = self.fit_for_monopole * (self.neta + self.nq)
        self.nfg_uveta = self.neta0_uveta + self.nlssm_uveta + self.nmp_uveta
        self.nuveta = self.neor_uveta + self.nfg_uveta
        # EoR model masks
        self.uveta_eor_mask = np.zeros(self.nuveta, dtype=bool)
        self.uveta_eor_mask[:self.neor_uveta] = True
        # # FG model masks
        self.uveta_fg_mask = np.zeros(self.nuveta, dtype=bool)
        self.uveta_fg_mask[self.neor_uveta:] = True
        self.uveta_eta0_mask = np.zeros(self.nuveta, dtype=bool)
        inds = slice(self.neor_uveta, self.neor_uveta+self.neta0_uveta)
        self.uveta_eta0_mask[inds] = True
        self.uveta_lssm_mask = np.zeros(self.nuveta, dtype=bool)
        inds = slice(
            self.neor_uveta+self.neta0_uveta, self.nuveta-self.nmp_uveta
        )
        self.uveta_lssm_mask[inds] = True

        # uvf vector