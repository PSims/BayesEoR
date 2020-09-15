"""
    Create a class used to store and manipulate healpix maps using
    astropy_healpix.
"""

import numpy as np
import astropy_healpix as ahp
from astropy_healpix import HEALPix
from astropy_healpix import healpy as hp
from astropy.coordinates import\
    EarthLocation, AltAz, ICRS, Angle, SkyCoord
from astropy.time import Time
import astropy.units as u

SECS_PER_HOUR = 60 * 60
SECS_PER_DAY = SECS_PER_HOUR * 24
DAYS_PER_SEC = 1.0 / SECS_PER_DAY
DEGREES_PER_HOUR = 360.0 / 24
DEGREES_PER_SEC = DEGREES_PER_HOUR * 1 / SECS_PER_HOUR


class Healpix(HEALPix):
    """
    Class to store and manipulate HEALPix maps using astropy_healpix
    functions.

    Parameters
    ----------
    fov_deg : float
        Specifies the field of view in degrees which sets a cutoff
        at zenith angles > fov_deg / 2 for pixels that are included
        in the sky model.
    nside : int
        Sets the nside resolution of the HEALPix map.  Defaults to
        512.
    telescope_latlonalt : tuple
        Tuple containing the latitude, longitude, and altitude of the
        telescope in degrees.
    central_jd : float
        Central time step of the observation in JD2000 format.
    beam_type : str, optional
        Can be either (case insensitive) 'uniform' or 'gaussian'.
        Specifies the type of beam to use.
        Defaults to 'uniform'.
    peak_amp : float, optional
        Sets the peak amplitude of the beam.  Defaults to 1.0.
    fwhm_deg : float, optional
        Required if beam_type='gaussian'. Sets the full width half
        maximum of the beam in degrees.
    """
    def __init__(
            self,
            fov_deg=None,
            nside=512,
            telescope_latlonalt=None,
            central_jd=None,
            beam_type=None,
            peak_amp=1.0,
            fwhm_deg=None
            ):
        # Use HEALPix as parent class to get
        # useful astropy_healpix functions
        super().__init__(nside, frame=ICRS())

        self.fov_deg = fov_deg
        self.pixel_area_sr = self.pixel_area.value
        self.lat, self.lon, self.alt = telescope_latlonalt
        # Set telescope location
        self.telescope_location = EarthLocation.from_geodetic(
            self.lon * u.degree, self.lat * u.degree
            )

        # Calculate field center in (RA, DEC)
        self.central_jd = central_jd
        t = Time(self.central_jd, scale='utc', format='jd')
        zen = AltAz(alt=Angle('90d'),
                    az=Angle('0d'),
                    obstime=t,
                    location=self.telescope_location)
        north = AltAz(alt=Angle('0d'),
                      az=Angle('0d'),
                      obstime=t,
                      location=self.telescope_location)
        zen_radec = zen.transform_to(ICRS)
        north_radec = north.transform_to(ICRS)
        self.pointing_center = (zen_radec.ra.deg, zen_radec.dec.deg)
        self.north_pole = (north_radec.ra.deg, north_radec.dec.deg)

        # Beam params
        if beam_type is not None:
            assert beam_type.lower() in ['uniform', 'gaussian'], \
                "Only uniform and Gaussian beams are currently supported"
            self.beam_type = beam_type.lower()
        self.peak_amp = peak_amp

        if beam_type.lower() == 'gaussian':
            assert fwhm_deg is not None, \
                "If using a Gaussian beam, must also pass fwhm_deg."
        self.fwhm_deg = fwhm_deg

        # Extra params to be set
        self.pix = None
        self.npix_fov = None
        self.ls = None
        self.ms = None
        self.l0 = 0.0
        self.m0 = 0.0

    def calc_lm_from_radec(self, center=None, north=None, ret=True):
        """
        Return arrays of (l, m) coordinates in radians of all
        HEALPix pixels within a disc of radius self.fov_deg / 2
        relative to self.pointing_center.

        Adapted from healvis.observatory.calc_azza.

        Parameters
        ----------
        center : tuple
            Center of cone search in (RA, DEC) in units of degrees.
        north : tuple
            North pole in (RA, DEC) in units of degrees.
        ret : boolean
            If True, return arrays of l and m coordinates.  Otherwise,
            return nothing.

        Returns
        -------
        ls : np.ndarray
            Array containing EW direction cosine of each HEALPix pixel.
        ms : np.ndarray
            Array containing NS direction cosine of each HEALPix pixel.
        """
        cvec = hp.ang2vec(center[0], center[1], lonlat=True)

        if north is None:
            north = np.array([0, 90.])
        nvec = hp.ang2vec(north[0], north[1], lonlat=True)
        # Get pixel indices that lie within the FoV
        # Filter pixels that lie outside the rectangular patch of sky
        # set by the central (RA0, DEC0) and the FoV as:
        # RA0 - FoV/2 <= RA <= RA0 + Fov/2
        # DEC0 - FOV/2 <= DEC <= DEC0 + FoV/2
        lons, lats = hp.pix2ang(
            self.nside,
            np.arange(self.npix),
            lonlat=True
            )
        lons_inds = np.logical_and(
            lons >= self.pointing_center[0] - self.fov_deg / 2,
            lons <= self.pointing_center[0] + self.fov_deg / 2,
            )
        lats_inds = np.logical_and(
            lats >= self.pointing_center[1] - self.fov_deg / 2,
            lats <= self.pointing_center[1] + self.fov_deg / 2
            )
        pix = np.where(lons_inds * lats_inds)[0]
        self.pix = pix
        self.npix_fov = pix.size
        vecs = hp.pix2vec(self.nside, pix).T # Shape (npix, 3)

        # Convert from (x, y, z) to (az, za)
        colat = np.arccos(np.dot(cvec, nvec))
        xvec = np.cross(nvec, cvec) * 1 / np.sin(colat)
        yvec = np.cross(cvec, xvec)
        sdotx = np.tensordot(vecs, xvec, 1)
        sdotz = np.tensordot(vecs, cvec, 1)
        sdoty = np.tensordot(vecs, yvec, 1)
        za_arr = np.arccos(sdotz)
        # xy plane is tangent. Increasing azimuthal angle eastward,
        # zero at North (y axis). x is East.
        az_arr = np.arctan2(sdotx, sdoty) % (2 * np.pi)

        # Convert from (az, za) to (l, m)
        self.ls = np.sin(za_arr) * np.sin(az_arr) # radians
        self.ms = np.sin(za_arr) * np.cos(az_arr) # radians

        if ret:
            return self.ls, self.ms

    def set_beam(self, beam_type=None, fwhm_deg=None, peak_amp=1.0):
        """
        Set beam params if not set in __init__.

        Parameters
        ----------
        beam_type : str, optional
            Can be either (case insensitive) 'uniform' or 'gaussian'.
            Specifies the type of beam to use.
            Defaults to 'uniform'.
        peak_amp : float, optional
            Sets the peak amplitude of the beam.  Defaults to 1.0.
        fwhm_deg : float, optional
            Required if beam_type='gaussian'. Sets the full width half
            maximum of the beam in degrees.

        """
        if beam_type is not None:
            assert beam_type.lower() in ['uniform', 'gaussian'], \
                "Only uniform and Gaussian beams are currently supported"
        self.peak_amp = peak_amp

        if beam_type.lower() == 'gaussian':
            assert fwhm_deg is not None, \
                "If using a Gaussian beam, must also pass fwhm_deg."
        self.fwhm_deg = fwhm_deg

    def get_beam_vals(self,
                      beam_center=None,
                      radec=False):
        """
        Get an array of beam values from (l, m) coordinates.

        Parameters
        ----------
        beam_center : tuple, optional
            Tuple of floats containing the beam pointing center
            in (l, m) coordinates (units of radians) or
            (RA, DEC) coordinates (units of degrees) if radec=True.
        radec : boolean
            If True, beam_center will be interpreted as an offset
            from the field center in (RA, DEC) coordinates in units
            of degrees.

        Returns
        -------
        beam_vals : np.ndarray
            Array containing beam amplitude values at each (l, m).
        """
        if self.beam_type.lower() == 'uniform':
            beam_vals = np.ones(self.npix_fov)

        elif self.beam_type.lower() == 'gaussian':
            stddev_rad = np.deg2rad(self._fwhm_to_stddev(self.fwhm_deg))

            if beam_center is not None:
                # Beam offset
                if radec:
                    # print('Converting beam_center from (RA, DEC) '
                    #       'to (l, m)...')
                    # Input is in (RA, DEC)
                    l0, m0 = beam_center
                    center_radec = SkyCoord(l0, m0, unit='deg')
                    # Convert from (RA, DEC) -> (alt, az)
                    t = Time(self.central_jd, scale='utc', format='jd')
                    center_altaz = center_radec.transform_to(
                        AltAz(obstime=t, location=self.telescope_location)
                        )
                    # Convert from (alt, az) -> (l, m)
                    self.l0 = (
                            np.sin(np.pi/2 - center_altaz.alt.rad)
                            * np.sin(center_altaz.az.rad)
                        )
                    self.m0 = (
                            np.sin(np.pi/2 - center_altaz.alt.rad)
                            * np.cos(center_altaz.az.rad)
                        )
                else:
                    # Input is assumed to be in (l, m)
                    self.l0, self.m0 = beam_center

                # print('Beam center (l, m) = ({:.2f}, {:.2f}) deg'.format(
                #       np.rad2deg(self.l0), np.rad2deg(self.m0)))

            # Calculate Gaussian beam values from (l, m)
            beam_vals = self._gaussian_2d(
                self.ls, self.ms, self.l0, self.m0,
                stddev_rad, stddev_rad, self.peak_amp)

        return beam_vals

    def _gaussian_2d(self, xs, ys, x0, y0, sigmax, sigmay, amp):
        """
        Calculates the value of a 2-dimensional Gaussian function
        at the values contained in xs and ys.  It is assumed that
        xs, ys, x0, y0, sigmax, and sigmay all have the same units.

        Parameters
        ----------
        xs : np.ndarray
            x coordinates, must have the same shape as ys.
        ys : np.ndarray
            y coordinates, must have the same shape as xs.
        x0 : float
            Centroid of the Gaussian beam envelope in the x direction.
        y0 : float
            Centroid of the Gaussian beam envelope in the y direction.
        sigmax : float
            Standard deviation of the Gaussian
            beam envelope in the x direction.
        sigmay : float
            Standard deviation of the Gaussian
            beam envelope in the y direction.
        amp : float
            Amplitude of the Gaussian envelope at the centroid.
        """
        return amp * np.exp(
            - (xs - x0)**2 / (2 * sigmax**2)
            - (ys - y0)**2 / (2 * sigmay**2)
            )

    def _fwhm_to_stddev(self, fwhm):
        """
        Converts a full width half maximum to a standard deviation
        for a Gaussian beam.
        """
        return fwhm / (2 * np.sqrt(2 * np.log(2)))