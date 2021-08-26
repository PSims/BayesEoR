import numpy as np
import subprocess
from subprocess import os
from scipy import integrate
import pickle
from types import ModuleType
from astropy import units
from astropy.units import Quantity
from astropy.constants import c
from astropy.cosmology import Planck18

import BayesEoR.Params.params as p


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
    - a phasor vector with shape (ndata,)

    The phasor vector takes an unphased set of visibilities and phases them
    to the central time step in the observation.

    Parameters
    ----------
    inst_model_dir : str
        Path to the instrument model directory
    uvw_file : str
        File containing instrumentally sampled (u, v, w) coords.  Defaults to
        'uvw_multi_time_step_array_meters'.
    red_file : str
        File containing baseline redundancy info.  Defaults to
        'uvw_redundancy_multi_time_step_array'.
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
    uvw_array_m = np.load(os.path.join(inst_model_dir, uvw_file))
    bl_red_array = np.load(os.path.join(inst_model_dir, red_file))
    phasor_vec = np.load(os.path.join(inst_model_dir, phasor_file))

    return uvw_array_m, bl_red_array, phasor_vec


def write_log_file(array_save_directory, file_root):
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

    """
    # Make log file directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), 'log_files/')
    if not os.path.exists(log_dir):
        print('Creating log directory at {}'.format(log_dir))
        os.mkdir(log_dir)

    # Get git version and hash info
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
    print('Log file written successfully to {}'.format(log_file))


def vector_is_hermitian(data, conj_map, nt, nf, nbls):
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
                if (
                    data[start_ind+conj_bl_ind]
                    == data[start_ind+bl_ind].conjugate()
                ):
                    hermitian[start_ind+bl_ind] = 1
                    hermitian[start_ind+conj_bl_ind] = 1
    return np.all(hermitian)
