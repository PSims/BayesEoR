import numpy as np
import subprocess
from subprocess import os
from scipy import integrate
import pickle
from types import ModuleType

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
    Class for performing cosmological distance calculations
    """
    def __init__(self, **kwargs):
        # ===== Defaults =====
        self.z1 = 0.
        self.z2 = 10.
        self.Print = 0

        # ===== Inputs =====
        if 'z1' in kwargs:
            self.z1 = kwargs['z1']
        if 'z2' in kwargs:
            self.z2 = kwargs['z2']
        if 'Print' in kwargs:
            self.Print = kwargs['Print']

        self.Omega_m = 0.279
        self.Omega_lambda = 0.721
        self.Omega_k = 0.0
        self.c = p.speed_of_light # m/s
        self.c_km_per_sec = p.speed_of_light / 1.0e3 # km/s
        self.H_0 = 70.0 # km/s/Mpc
        self.f_21 = 1420.40575177 # MHz
        # Hubble Parameter at redshift z2
        self.E_z2 = np.sqrt(self.Omega_m * (1.+self.z2)**3 + self.Omega_lambda)

    def dL_df(self, z):
        """
        Comoving differential distance at redshift per frequency

        [cMpc]/Hz
        """
        # Hubble parameter
        E_z = np.sqrt(self.Omega_m * (1. + z)**3 + self.Omega_lambda)
        fac = self.c_km_per_sec / (self.H_0 * E_z)
        fac *= (1 + z)**2 / (self.f_21 * 1.0e6)
        return fac

    def dL_dth(self, z):
        """
        Comoving transverse distance per radian in Mpc

        [cMpc]/radian
        """
        Comoving_Distance_Mpc, CD_uncertainty =\
            integrate.quad(self.Comoving_Distance_Mpc_Integrand, 0, z)
        return Comoving_Distance_Mpc

    def X2Y(self, z):
        """
        Conversion factor for Mpc^3 --> sr Hz
        """
        return self.dL_dth(z)**2 * self.dL_df(z)


    def Comoving_Distance_Mpc_Integrand(self, z, **kwargs):
        # Hubble parameter
        E_z = (self.Omega_m*((1.+z)**3) + self.Omega_lambda)**0.5
        # Hubble distance in Mpc
        self.Hubble_Distance = self.c_km_per_sec/self.H_0
        return (self.Hubble_Distance/E_z)

    ###
    # Calculate 21cmFast Box size in degrees at a given redshift
    ###
    def Calculate_Comoving_Distance_Mpc_Between_Redshifts_z1_and_z2(
            self, **kwargs):
        self.Comoving_Distance_Mpc, self.Comoving_convergence_uncertainty =\
            integrate.quad(
                self.Comoving_Distance_Mpc_Integrand, self.z1, self.z2)
        return self.Comoving_Distance_Mpc,\
               self.Comoving_convergence_uncertainty

    ###
    # Calculate 21cmFast frequency depth at a given redshift
    # using Morales & Hewitt 2004 eqn.
    ###
    def Convert_from_Comoving_Mpc_to_Delta_Frequency_at_Redshift_z2(
            self, **kwargs):
        # ===== Defaults =====
        self.Box_Side_cMpc=3000.

        # ===== Inputs =====
        if 'Box_Side_cMpc' in kwargs:
            self.Box_Side_cMpc=kwargs['Box_Side_cMpc']

        if self.Print:
            print('Convert_from_Comoving_Mpc_to_Delta_Frequency_at_Redshift_z2'
                  ' at \nRedshift z =', self.z2,
                  '\nBox depth in cMpc, Box_Side_cMpc =', self.Box_Side_cMpc)


        self.Delta_f_MHz = (self.H_0*self.f_21*self.E_z2*self.Box_Side_cMpc /
                            (self.c_km_per_sec * (1.+self.z2)**2.))
        self.Delta_f_Hz = self.Delta_f_MHz * 1.e6

        if self.Print:
            print('Delta_f_MHz = ', self.Delta_f_MHz)
        return self.Delta_f_MHz

    ###
    # Calculate 21cmFast k_parallel - space values
    ###
    def Convert_from_Tau_to_Kz(self, Tau_Array, **kwargs):
        self.K_z_Array = (
                ((2.*np.pi*self.H_0*self.f_21*self.E_z2) /
                 (self.c_km_per_sec*(1.+self.z2)**2.)) *
                Tau_Array)
        return self.K_z_Array

    ###
    # Calculate 21cmFast k_perp - space values
    ###
    def Convert_from_U_to_Kx(self, U_Array, **kwargs):
        Comoving_Distance_Mpc, Comoving_convergence_uncertainty =\
            self.Calculate_Comoving_Distance_Mpc_Between_Redshifts_z1_and_z2()
        self.K_x_Array = (2.*np.pi / Comoving_Distance_Mpc) * U_Array # h*cMPc^-1
        return self.K_x_Array

    ###
    # Calculate 21cmFast k_perp - space values
    ###
    def Convert_from_V_to_Ky(self, V_Array, **kwargs):
        Comoving_Distance_Mpc, Comoving_convergence_uncertainty =\
            self.Calculate_Comoving_Distance_Mpc_Between_Redshifts_z1_and_z2()
        self.K_y_Array = (2.*np.pi / Comoving_Distance_Mpc) * V_Array # h*cMPc^-1
        return self.K_y_Array

    ###
    # Convert from Frequency to Redshift
    ###
    def Convert_from_21cmFrequency_to_Redshift(
            self, Frequency_Array_MHz, **kwargs):
        One_plus_z_Array = self.f_21 / Frequency_Array_MHz
        self.z_Array = One_plus_z_Array - 1.
        return self.z_Array

    ###
    # Convert from Redshift to Frequency
    ###
    def Convert_from_Redshift_to_21cmFrequency(self, Redshift, **kwargs):
        One_plus_z = 1. + Redshift
        self.redshifted_21cm_frequency = self.f_21 / One_plus_z
        return self.redshifted_21cm_frequency

    ###
    # Calculate angular separation of 21cmFast box at a given redshift
    ###
    def Convert_from_Comoving_Mpc_to_Delta_Angle_at_Redshift_z2(self, **kwargs):
        """
            Angular diameter distance
            An object of size x at redshift z that appears to have
            angular size \delta\theta has the angular diameter distance
            of d_A(z)=x/\delta\theta.

            Angular diameter distance:
            d_A(z)  = \frac{d_M(z)}{1+z}
            with d_M(z) = Comoving_Distance_Mpc
        """
        # ===== Defaults =====
        self.Box_Side_cMpc=3000.

        # ===== Inputs =====
        if 'Box_Side_cMpc' in kwargs:
            self.Box_Side_cMpc=kwargs['Box_Side_cMpc']

        Comoving_Distance_Mpc, Comoving_convergence_uncertainty =\
            self.Calculate_Comoving_Distance_Mpc_Between_Redshifts_z1_and_z2()
        angular_diameter_distance_Mpc = (Comoving_Distance_Mpc
                                         / (1 + (self.z2-self.z1)))
        Box_width_proper_distance_Mpc = (self.Box_Side_cMpc
                                         / (1 + (self.z2-self.z1)))
        if self.Print:
            print('Convert_from_Comoving_Mpc_to_Delta_Angle_at_Redshift_z2 at'
                  '\nRedshift z =',self.z2,
                  '\nBox depth in cMpc, Box_Side_cMpc =', self.Box_Side_cMpc,
                  '\nBox proper depth in Mpc, Box proper width =',
                  Box_width_proper_distance_Mpc,
                  '\nComoving distance between z1={} and z2={}: {}'\
                  .format(self.z1,self.z2,Comoving_Distance_Mpc),
                  '\nAngular diameter distance between z1={} and z2={}: {}'\
                  .format(self.z1,self.z2,angular_diameter_distance_Mpc))

        ###
        # tan(theta) = Box_Side_cMpc/Comoving_Distance_Mpc
        ###
        # From Hogg 1999 (although no discussion of
        # applicability for large theta there)
        self.theta_rad = (Box_width_proper_distance_Mpc
                          / angular_diameter_distance_Mpc)
        self.theta_deg = self.theta_rad*180./np.pi
        if self.Print:
            print('theta_deg = ', self.theta_deg)
        return self.theta_deg


def generate_output_file_base(file_root, **kwargs):

    # ===== Defaults =====
    default_version_number = '1'

    # ===== Inputs =====
    if 'version_number' in kwargs:
        version_number = kwargs.pop('version_number', default_version_number)

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


def load_uvw_instrument_sampling_m(instrument_model_directory):
    file_dir = instrument_model_directory
    file_name = "uvw_multi_time_step_array_meters"
    with open(file_dir + file_name, 'rb') as f:
        uvw_multi_time_step_array_meters = pickle.load(f)
    return uvw_multi_time_step_array_meters


def load_baseline_redundancy_array(instrument_model_directory):
    file_dir = instrument_model_directory
    file_name =\
        "unique_H37_baseline_hermitian_redundancy_multi_time_step_array"
    with open(file_dir+file_name, 'rb') as f:
        unique_H37_baseline_hermitian_redundancy_multi_time_step_array =\
            pickle.load(f)
    return unique_H37_baseline_hermitian_redundancy_multi_time_step_array


def write_log_file(array_save_directory, file_root):
    """
        Write a log file containing current git hash, array save
        directory, multinest output file root, and parameters from
        BayesEoR.Params.params for a complete record of what parameters
        went into each analysis run.
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

    log_file = log_dir + file_root + '.log'
    dashed_line = '-'*44
    with open(log_file, 'w') as f:
        f.write('#' + dashed_line + '\n# GitHub Info\n#' + dashed_line + '\n')
        for key in version_info.keys():
            f.write('{}: {}'.format(key, version_info[key]))
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
