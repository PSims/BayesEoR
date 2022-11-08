import numpy as np
import pickle
import subprocess
from subprocess import os
from types import ModuleType
from astropy import units
from astropy.units import Quantity
from astropy.constants import c
from astropy.cosmology import Planck18
from pathlib import Path
from copy import deepcopy

import BayesEoR.Params.params as p
from BayesEoR.Linalg import Healpix


class PriorC(object):
    def __init__(self, priors_min_max):
        self.priors_min_max = priors_min_max

    def prior_func(self, cube):
        pmm = self.priors_min_max
        theta = []
        for i_p in range(len(cube)):
            theta_i = pmm[i_p][0] + ((pmm[i_p][1] - pmm[i_p][0]) * cube[i_p])
            theta.append(theta_i)
        return theta


class Cosmology:
    """
    Class for performing cosmological distance
    calculations using `astropy.cosmology.Planck18`.

    """
    def __init__(self):
        self.cosmo = Planck18
        self.Om0 = self.cosmo.Om0
        self.Ode0 = self.cosmo.Ode0
        self.Ok0 = self.cosmo.Ok0
        self.H0 = self.cosmo.H0
        self.c = c.to('m/s')
        self.f_21 = 1420.40575177 * units.MHz

    def f2z(self, f):
        """
        Convert a frequency `f` in Hz to redshift
        relative to `self.f_21`.

        Parameters
        ----------
        f : float
            Input frequency in Hz.

        Returns
        -------
        z : float
            Redshift corresponding to frequency `f`.

        """
        if not isinstance(f, Quantity):
            f *= units.Hz
        else:
            f = f.to('Hz')
        return (self.f_21/f - 1).value

    def z2f(self, z):
        """
        Convert a redshift `z` relative to `self.f_21`
        to a frequency in Hz.

        Parameters
        ----------
        z : float
            Input redshift.

        Returns
        -------
        f : float
            Frequency corresponding to redshift `z`.

        """
        return (self.f_21 / (1 + z)).to('Hz').value

    def dL_df(self, z):
        """
        Comoving differential distance at redshift per frequency.

        Parameters
        ----------
        z : float
            Input redshift.

        Returns
        -------
        dl_df : float
            Conversion factor relating a bandwidth in Hz to a comoving size in
            Mpc at redshift `z`.

        """
        d_h = self.c.to('km/s') / self.H0  # Hubble distance
        e_z = self.cosmo.efunc(z)
        dl_df = d_h / e_z * (1 + z)**2 / self.f_21.to('Hz')
        return dl_df.value

    def dL_dth(self, z):
        """
        Comoving transverse distance per radian in Mpc.

        Parameters
        ----------
        z : float
            Input redshift.

        Returns
        -------
        dl_dth : float
            Conversion factor relating an angular size in
            radians to a comoving transverse size in Mpc
            at redshift `z`.

        """
        dl_dth = self.cosmo.comoving_transverse_distance(z)
        return dl_dth.value

    def inst_to_cosmo_vol(self, z):
        """
        Conversion factor to go from an instrumentally
        sampled volume in sr Hz to a comoving cosmological
        volume in Mpc^3.

        Parameters
        ----------
        z : float
            Input redshift.

        Returns
        -------
        i2cV : float
            Volume conversion factor for sr Hz --> Mpc^3 at redshift `z`.

        """
        i2cV = self.dL_dth(z)**2 * self.dL_df(z)
        return i2cV


def generate_output_file_base(file_root, version_number='1'):
    """
    Generate a filename for the sampler output.  The version number of the
    output file is incrimented until a new `file_root` is found to avoid
    overwriting existing sampler data.

    Parameters
    ----------
    file_root : str
        Filename root with a version number string `-v{}-` suffix.
    version_number : str
        Version number as a string.  Defaults to '1'.

    Returns
    -------
    file_root : str
        Updated filename root with a new, largest version number.

    """
    file_name_exists = (
            os.path.isfile('chains/'+file_root+'_phys_live.txt')
            or os.path.isfile('chains/'+file_root+'.resume')
            or os.path.isfile('chains/'+file_root+'resume.dat'))
    while file_name_exists:
        fr1, fr2 = file_root.split('-v')
        fr21, fr22 = fr2.split('-')
        next_version_number = str(int(fr21)+1)
        file_root = file_root.replace('v'+version_number+'-',
                                      'v'+next_version_number+'-')
        version_number = next_version_number
        file_name_exists = (
                os.path.isfile('chains/'+file_root+'_phys_live.txt')
                or os.path.isfile('chains/'+file_root+'.resume')
                or os.path.isfile('chains/'+file_root+'resume.dat'))
    return file_root


def load_inst_model(
        inst_model_dir,
        uvw_file='uvw_model.npy',
        red_file='redundancy_model.npy',
        phasor_file='phasor_vector.npy'):
    """
    Load the instrument model consisting of
    - a (u, v, w) array with shape (nt, nbls, 3)
    - baseline redundancy array with shape (nt, nbls, 1)
    - a phasor vector with shape (ndata,) (if modelling phased visibilities)

    The phasor vector takes an unphased set of visibilities and phases them
    to the central time step in the observation.

    This function first looks for an 'instrument_model.npy' pickled dictionary
    in `inst_model_dir`.  If not found, it will then load the individual numpy
    arrays specified by `uvw_file`, `red_file`, and `phasor_file`.

    Parameters
    ----------
    inst_model_dir : str
        Path to the instrument model directory.
    uvw_file : str
        File containing instrumentally sampled (u, v, w) coords.  Defaults to
        'uvw_model.npy'.
    red_file : str
        File containing baseline redundancy info.  Defaults to
        'redundancy_model.npy'.
    phasor_file : str
        File containing the phasor vector.  Defaults to 'phasor_vector.npy'.

    Returns
    -------
    uvw_array_m : np.ndarray
        Array of (u, v, w) coordinates.
    bl_red_array : np.ndarray
        Array of baseline redundancies.
    phasor_vec : np.ndarray
        Array of phasor values.

    """
    if os.path.exists(os.path.join(inst_model_dir, 'instrument_model.npy')):
        data_dict = np.load(
            os.path.join(inst_model_dir, 'instrument_model.npy'),
            allow_pickle=True
        ).item()
        uvw_array_m = data_dict['uvw_model']
        bl_red_array = data_dict['redundancy_model']
        if 'phasor_vector' in data_dict:
            phasor_vec = data_dict['phasor_vector']
        else:
            phasor_vec = None
    else:
        # Support for old instrument model formats
        uvw_array_m = np.load(os.path.join(inst_model_dir, uvw_file))
        bl_red_array = np.load(os.path.join(inst_model_dir, red_file))
        phasor_vec = np.load(os.path.join(inst_model_dir, phasor_file))

    return uvw_array_m, bl_red_array, phasor_vec


def get_git_version_info(directory=None):
    """
    Get git version info from repository in `directory`.

    Parameters
    ----------
    directory : str
        Path to GitHub repository.  If None, uses one directory up from
        __file__.

    Returns
    -------
    version_info : dict
        Dictionary containing git hash information.

    """
    cwd = os.getcwd()
    if directory is None:
        directory = Path(__file__).parent
    os.chdir(directory)

    version_info = {}
    version_info['git_origin'] = subprocess.check_output(
        ['git', 'config', '--get', 'remote.origin.url'],
        stderr=subprocess.STDOUT)
    version_info['git_hash'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        stderr=subprocess.STDOUT)
    version_info['git_description'] = subprocess.check_output(
        ['git', 'describe', '--dirty', '--tag', '--always'])
    version_info['git_branch'] = subprocess.check_output(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
        stderr=subprocess.STDOUT)
    for key in version_info.keys():
        version_info[key] = version_info[key].decode('utf8').strip('\n')
    
    os.chdir(cwd)

    return version_info


def write_log_file(array_save_directory, file_root, priors):
    """
    Write a log file containing current git hash, array save
    directory, multinest output file root, and parameters from
    BayesEoR.Params.params for a complete record of what parameters
    went into each analysis run.

    Parameters
    ----------
    array_save_directory : str
        Directory where arrays used in the analysis are saved.
    file_root : str
        Filename for sampler output files.
    priors : array-like
        Array-like containing prior ranges for each k-bin.

    """
    # Make log file directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), 'log_files/')
    if not os.path.exists(log_dir):
        print('Creating log directory at {}'.format(log_dir))
        os.mkdir(log_dir)

    # Get git version and hash info
    version_info = get_git_version_info()

    log_file = log_dir + file_root + '.log'
    dashed_line = '-'*44
    with open(log_file, 'w') as f:
        f.write('#' + dashed_line + '\n# GitHub Info\n#' + dashed_line + '\n')
        for key in version_info.keys():
            f.write('{}: {}\n'.format(key, version_info[key]))
        f.write('\n\n')
        f.write('#' + dashed_line + '\n# Directories\n#' + dashed_line + '\n')
        f.write('Array save directory:\t{}\n'.format(array_save_directory))
        f.write('Multinest output file root:\t{}\n'.format(file_root))
        f.write('\n\n')
        f.write('#' + dashed_line + '\n# Parser / Params Variables\n#'
                + dashed_line + '\n')
        for key in p.__dict__.keys():
            if (not key.startswith('_')
                    and not isinstance(p.__dict__[key], ModuleType)):
                f.write('{} = {}\n'.format(key, p.__dict__[key]))
        f.write('priors = {}\n'.format(priors))
    print('Log file written successfully to {}'.format(log_file))


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


def mpiprint(*message, rank=0, end='\n'):
    """
    Wrapper of print function.  Only prints a message if rank == 0 when
    using multiple MPI processes.

    Parameters
    ----------
    message : str or sequence of str
        Message to print.
    rank : int
        MPI rank.
    end : str
        String argument suffix for `message`.

    """
    if rank == 0:
        print(' '.join(map(str, message)), end=end)


def write_map_dict(dir, pspp, bm, n, clobber=False, fn='map-dict.npy'):
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
            'pspp': pspp_copy,
            'bm': bm,
            'n': n
        }
        print(f'\nWriting MAP dict to {fp}\n')
        with open(fp, 'wb') as f:
            pickle.dump(map_dict, f, protocol=4)


def parse_uprior_inds(upriors_str, nkbins):
    """
    Parse a string containing array indexes.

    `upriors_str` must follow standard array slicing syntax and include no
    spaces.  Examples of valid strings:
    * '1:4': equivalent to `slice(1, 4)`
    * '1,3,4': equivalent to indexing with `[1, 3, 4]`
    * '3' or '-3'
    * 'all'

    Parameters
    ----------
    upriors_str : str
        String containing array indexes (follows array slicing syntax).
    nkbins : int
        Number of k-bins.

    Returns
    -------
    uprior_inds : array
        Boolean array that is True for any k-bins using a uniform prior.
        False entries use a log-uniform prior.

    """
    if upriors_str.lower() == 'all':
        uprior_inds = np.ones(nkbins, dtype=bool)
    else:
        uprior_inds = np.zeros(nkbins, dtype=bool)
        if ':' in upriors_str:
            bounds = list(map(int, upriors_str.split(':')))
            uprior_inds[slice(*bounds)] = True
        elif ',' in upriors_str:
            up_inds = list(map(int, upriors_str.split(',')))
            uprior_inds[up_inds] = True
        else:
            uprior_inds[int(upriors_str)] = True

    return uprior_inds


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