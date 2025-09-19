from astropy import units
from astropy.units import Quantity
from astropy.constants import c
from astropy.cosmology import Planck18

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