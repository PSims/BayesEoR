from typing import Any, cast

from astropy import constants as const, units
from astropy.cosmology import Planck18
from astropy.units import Quantity


class Cosmology:
    """
    Class for performing cosmological distance
    calculations using `astropy.cosmology.Planck18`.

    """

    def __init__(self) -> None:
        self.cosmo: Any = Planck18
        self.Om0: float = float(cast(Any, self.cosmo).Om0)
        self.Ode0: float = float(cast(Any, self.cosmo).Ode0)
        self.Ok0: float = float(cast(Any, self.cosmo).Ok0)
        self.H0: Quantity = cast(Quantity, cast(Any, self.cosmo).H0)
        self.c: Quantity = cast(Quantity, const.c.to("m/s"))  # pyright: ignore
        self.f_21: Quantity = cast(Quantity, 1420.40575177 * units.MHz)

    def f2z(self, f: float | Quantity) -> float:
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
            freq_hz = float(f)
        else:
            freq_hz = float(cast(Any, f).to_value(units.Hz))
        f_21_hz = float(cast(Any, self.f_21).to_value(units.Hz))
        return f_21_hz / freq_hz - 1.0

    def z2f(self, z: float) -> float:
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
        f_21_hz = float(cast(Any, self.f_21).to_value(units.Hz))
        return f_21_hz / (1 + z)

    def dL_df(self, z: float) -> float:
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
        c_km_s = float(cast(Any, self.c).to_value(units.km / units.s))
        h0_km_s_mpc = float(
            cast(Any, self.H0).to_value(units.km / (units.s * units.Mpc))
        )
        d_h_mpc = c_km_s / h0_km_s_mpc
        e_z = float(cast(Any, self.cosmo).efunc(z))
        f_21_hz = float(cast(Any, self.f_21).to_value(units.Hz))
        dl_df = d_h_mpc / e_z * (1 + z) ** 2 / f_21_hz
        return dl_df

    def dL_dth(self, z: float) -> float:
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
        dl_dth = cast(Any, self.cosmo).comoving_transverse_distance(z)
        return float(cast(Any, dl_dth).to_value(units.Mpc))

    def inst_to_cosmo_vol(self, z: float) -> float:
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
        i2cV = self.dL_dth(z) ** 2 * self.dL_df(z)
        return i2cV
