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


def get_array_dir_name(args, prefix="./array-storage/"):
    """
    Generate the output path for BayesEoR matrices based on analysis params.

    This function constructs two strings which form two subdirectories:
      1. The `analysis_dir` string contains all the unique analysis/model
         parameters, e.g. the field(s) of view, nu, nv, neta, etc.
      2. The `inst_dir` string contains all of the instrument model specific
         parameters, e.g. the instrument model filename, beam type, beam
         center, integration time, etc.
    The final array save directory is produced via
    ```
    matrices_path = Path(prefix) / analysis_dir / inst_dir
    ```

    Parameters
    ----------
    args : Namespace
        Namespace with analysis params from `BayesEoRParser`.
    prefix : str
        Prefix for `matrices_path`.  Defaults to './'.

    Returns
    -------
    matrices_path : Path
        Path containing the uniquely identifying info for each analysis, i.e.
        model parameters and the instrument model.

    """
    if args.array_dir_prefix is not None:
        prefix = args.array_dir_prefix
    matrices_path = Path(prefix)

    # Root matrix dir
    analysis_dir = (
        f"nu-{args.nu}-nv-{args.nv}-neta-{args.neta}"
        + f"-sigma-{args.sigma:.2E}-nside-{args.nside}"
    )

    fovs_match = (
        args.fov_ra_eor == args.fov_ra_fg
        and args.fov_dec_eor == args.fov_dec_fg
    )
    fov_str = "-fov-deg"
    if not fovs_match:
        fov_str += "-eor"
    if not args.fov_ra_eor == args.fov_dec_eor and not args.simple_za_filter:
        fov_str += f"-ra-{args.fov_ra_eor:.1f}-dec-{args.fov_dec_eor:.1f}"
    else:
        fov_str += f"-{args.fov_ra_eor:.1f}"
    if not fovs_match:
        fov_str += "-fg"
        if args.fov_ra_fg != args.fov_dec_fg and not args.simple_za_filter:
            fov_str += f"-ra-{args.fov_ra_fg:.1f}-dec-{args.fov_dec_fg:.1f}"
        else:
            fov_str += f"-{args.fov_ra_fg:.1f}"
    if args.simple_za_filter:
        fov_str += "-za-filter"
    analysis_dir += fov_str

    nu_nv_match = (
        args.nu == args.nu_fg and args.nv == args.nv_fg
    )
    if not nu_nv_match:
        analysis_dir += f"-nufg-{args.nu_fg}-nvfg-{args.nv_fg}"
    
    analysis_dir += f"-nq-{args.nq}"
    if args.nq > 0:
        if args.npl == 1:
            analysis_dir += f"-beta-{args.beta[0]:.2f}"
        else:
            for i in range(args.npl):
                analysis_dir += f"-b{i+1}-{args.beta[i]:.2f}"
    if args.fit_for_monopole:
        analysis_dir += "-ffm"

    if args.use_shg:
        shg_str = "-shg"
        if args.nu_sh > 0:
            shg_str += f"-nush-{args.nu_sh}"
        if args.nv_sh > 0:
            shg_str += f"-nvsh-{args.nv_sh}"
        if args.nq_sh > 0:
            shg_str += f"-nqsh-{args.nq_sh}"
        if args.npl_sh > 0:
            shg_str += f"-nplsh-{args.npl_sh}"
        if args.fit_for_shg_amps:
            shg_str += "-ffsa"
        analysis_dir += shg_str
    
    if args.beam_center is not None:
        beam_center_signs = [
            "+" if args.beam_center[i] >= 0 else "" for i in range(2)
        ]
        beam_center_str = "-beam-center-RA0{}{:.2f}-DEC0{}{:.2f}".format(
                beam_center_signs[0],
                args.beam_center[0],
                beam_center_signs[1],
                args.beam_center[1]
        )
        analysis_dir += beam_center_str
    
    if not args.drift_scan:
        analysis_dir += "-phased"
    
    if args.taper_func:
        analysis_dir += f"-{args.taper_func}"
    
    matrices_path /= analysis_dir
    
    if args.include_instrumental_effects:
        beam_info_str = ""
        if not "." in args.beam_type:
            beam_info_str = f"{args.beam_type}-beam"
            if args.achromatic_beam:
                beam_info_str = "achromatic-" + beam_info_str
            if (not args.beam_peak_amplitude == 1
                and args.beam_type in ["uniform", "gaussian", "gausscosine"]):
                beam_info_str += f"-peak-amp-{args.beam_peak_amplitude}"
            
            if args.beam_type in ["gaussian", "gausscosine"]:
                if args.fwhm_deg is not None:
                    beam_info_str += f"-fwhm-{args.fwhm_deg:.4f}deg"
                elif args.antenna_diameter is not None:
                    beam_info_str += (
                        f"-antenna-diameter-{args.antenna_diameter}m"
                    )
                if args.beam_type == "gausscosine":
                    beam_info_str += f"-cosfreq-{args.cosfreq:.2f}wls"
            elif args.beam_type in ["airy", "taperairy"]:
                beam_info_str += f"-antenna-diameter-{args.antenna_diameter}m"
                if args.beam_type == "taperairy":
                    beam_info_str += f"-fwhm-{args.fwhm_deg}deg"
            if args.achromatic_beam:
                beam_info_str += f"-ref-freq-{args.beam_ref_freq:.2f}MHz"
        else:
            beam_info_str = Path(args.beam_type).stem

        inst_dir = "-".join((Path(args.inst_model).name, beam_info_str))
        if args.drift_scan:
            inst_dir += "-dspb"
        if args.noise_data_path is not None:
            inst_dir += "-noise-vec"

        matrices_path /= inst_dir
    
    matrices_path.mkdir(exist_ok=True, parents=True)

    return str(matrices_path) + "/", fov_str


def gen_output_file_base(
    nu,
    nv,
    neta,
    nq,
    beta,
    sigma,
    log_priors=True,
    dimensionless_PS=True,
    use_shg=False,
    fit_for_shg_amps=False,
    nu_sh=None,
    nv_sh=None,
    nq_sh=None,
    use_Multinest=True,
    output_dir="./",
    version_number=1
):
    """
    Generate a directory name for the sampler output.

    This function first generates the name of the subdirectory in which the
    sampler outputs will be written, `file_root`.  It then searches in
    `output_dir` and incriments the `version_number` in the subdirectory name
    by 1 until a unique subdirectory name is found.

    Parameters
    ----------
    nu : int
        Number of pixels on the u-axis of the model uv-plane for the EoR model.
    nv : int
        Number of pixels on the v-axis of the model uv-plane for the EoR model.
    neta : int
        Number of line-of-sight Fourier modes.
    nq : int
        Number of Large Spectral Scale Model (LSSM) quadratic basis vectors.
    beta : list of float
        Brightness temperature power law spectral index/indices used in the
        LSSM.
    sigma : float
        Standard deviation of the visibility noise.
    log_priors : bool, optional
        Priors on power spectrum coefficients are in log_10 units
        (True, default) or linear units (False).
    dimensionless_PS : bool, optional
        Sampler output is the dimensionless power spectrum \Delta^2(k)
        (True, default) or the power spectrum P(k) (False).
    use_shg : bool, optional
        Whether or not the model contains the SubHarmonic Grid (SHG).  Defaults
        to False.
    fit_for_shg_amps : bool, optional
        Whether or not the SHG amplitudes are being fit.  Defaults to False.
    nu_sh : int, optional
        Number of pixels on a side for the u axis in the SHG.
    nv_sh : int, optional
        Number of pixels on a side for the v axis in the SHG.
    nq_sh : int, optional
        Number of LSSM quadratic basis vectors.
    use_Multinest : bool, optional
        Analysis uses Multinest (True, default) or Polychord (False).
    output_dir : str or Path, optional
        Directory in which to search for files in subdirectories.
    version_number : int, optional
        Version number suffix for `file_root`.  Defaults to 1.

    Returns
    -------
    file_root : str
        Unique subdirectory name for the sampler output.

    """
    if use_Multinest:
        file_root = "MN-"
    else:
        file_root = "PC-"
    if beta is not None:
        npl = len(beta)
    else:
        npl = 0
    file_root = f"Test-{nu}-{nv}-{neta}-{nq}-{npl}"
    if beta is not None:
        beta_str = ""
        for b in beta:
            beta_str += f"-{b:.2f}"
        file_root += beta_str
    if use_shg:
        file_root += f"-SHG-{nu_sh}-{nv_sh}-{nq_sh}-{npl}"
    if fit_for_shg_amps:
        file_root += "-ffsa"
    file_root += f"-{sigma:.1E}"
    if log_priors:
        file_root += "-lp"
    if dimensionless_PS:
        file_root += "-dPS"
    file_root += f"-v{version_number}"

    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    suffixes = ["phys_live.txt", ".resume", "resume.dat"]

    def check_for_files(directory, suffixes):
        Nfiles = 0
        for suffix in suffixes:
            Nfiles += len(list(directory.glob(f"*{suffix}")))
        return Nfiles > 0

    while check_for_files(output_dir / file_root, suffixes):
        current_version = int(file_root.split("-v")[-1])
        next_version = current_version + 1
        file_root = file_root.replace(
            f"v{version_number}", f"v{next_version}"
        )
        version_number = next_version
    
    return file_root + "/"


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