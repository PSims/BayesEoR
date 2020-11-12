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
DEGREES_PER_DAY = 360.0
DEGREES_PER_HOUR = DEGREES_PER_DAY / 24
DEGREES_PER_SEC = DEGREES_PER_HOUR * 1 / SECS_PER_HOUR


class Healpix(HEALPix):
    """
    Class to store and manipulate HEALPix maps using `astropy_healpix`
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
    nt : int
        Number of time integrations. Defaults to 1.
    int_time : float, optional
        Integration time in seconds. Required if `nt > 1`.
    beam_type : str, optional
        Can be either (case insensitive) 'uniform' or 'gaussian'.
        Specifies the type of beam to use.
        Defaults to 'uniform'.
    peak_amp : float, optional
        Sets the peak amplitude of the beam.  Defaults to 1.0.
    fwhm_deg : float, optional
        Required if `beam_type='gaussian'`. Sets the full width half
        maximum of the beam in degrees.
    beam_center : tuple of floats, optional
        Sets the beam's pointing center in (RA, DEC) in units of
        degrees.
    rel : boolean
        If True, will treat `beam_center` as a tuple of offsets along
        the RA and DEC axes relative to the pointing center determined
        from `telescope_latlonalt` and `central_jd`.
    """
    def __init__(
            self,
            fov_deg=None,
            nside=512,
            telescope_latlonalt=None,
            central_jd=None,
            nt=1,
            int_time=None,
            beam_type=None,
            peak_amp=1.0,
            fwhm_deg=None,
            beam_center=None,
            rel=False
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
        zen_radec = zen.transform_to(ICRS)
        self.field_center = (zen_radec.ra.deg, zen_radec.dec.deg)

        # Set time axis params for calculating (l(t),  m(t))
        self.nt = nt
        self.int_time = int_time
        if self.nt % 2:
            self.time_inds = np.arange(-(self.nt//2), self.nt//2 + 1)
        else:
            self.time_inds = np.arange(-(self.nt//2), self.nt//2)
        # Calculate JD per integration from `central_jd`
        if self.int_time is not None:
            self.jds = (self.central_jd
                        + self.time_inds * self.int_time * DAYS_PER_SEC)
        else:
            self.jds = np.array([self.central_jd])
        # Calculate pointing center and north pole per integration
        self.pointing_centers = []
        self.north_poles = []
        for jd in self.jds:

            t = Time(jd, scale='utc', format='jd')
            # Calculate north pole in (alt, az)
            north = AltAz(alt=Angle('0d'),
                          az=Angle('0d'),
                          obstime=t,
                          location=self.telescope_location)
            north_radec = north.transform_to(ICRS)
            self.north_poles.append((north_radec.ra.deg, north_radec.dec.deg))
            # Calculate zenith angle in (alt, az)
            zen = AltAz(alt=Angle('90d'),
                        az=Angle('0d'),
                        obstime=t,
                        location=self.telescope_location)
            zen_radec = zen.transform_to(ICRS)
            self.pointing_centers.append((zen_radec.ra.deg, zen_radec.dec.deg))

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

        if beam_center is not None:
            self.set_beam_center_radec(beam_center, rel=rel)
        else:
            self.l0 = 0.0
            self.m0 = 0.0

        # Pixel params
        self.pix = None # HEALPix pixel numbers within the FoV
        self.npix_fov = None # Number of pixels within the FoV
        # Set self.pix and self.npix_fov
        self.set_pixel_filter()

    def set_pixel_filter(self):
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
            lons >= self.field_center[0] - self.fov_deg / 2,
            lons <= self.field_center[0] + self.fov_deg / 2,
            )
        lats_inds = np.logical_and(
            lats >= self.field_center[1] - self.fov_deg / 2,
            lats <= self.field_center[1] + self.fov_deg / 2
            )
        pix = np.where(lons_inds * lats_inds)[0]
        self.pix = pix
        self.npix_fov = pix.size

    def calc_lm_from_radec(self,
                           center=None,
                           north=None,
                           radec_offset=None,
                           time_index=0):
        """
        Return arrays of (l, m) coordinates in radians of all
        HEALPix pixels within a disc of radius self.fov_deg / 2
        relative to center=(RA, DEC). The pixels used in this
        calculation are set by `self.set_pixel_filter`.

        Adapted from healvis.observatory.calc_azza.

        Parameters
        ----------
        center : tuple of floats, optional
            Central (RA, DEC) in units of degrees.  Sets the center
            of the rectangular patch of sky.  Defaults to
            `self.field_center`.
        north : tuple of floats, optional
            North pole in (RA, DEC) in units of degrees.  Defaults to
            the value of the north pole at `self.central_jd` is used.
        radec_offset : tuple of floats, optional
            Offset in (RA, DEC) in units of degrees.  Shifts the center
            to `(center[0] + RA_offset, center[1] + DEC_offset)`.
            Defaults to None, i.e. no offset.
        time_index : int, optional
            If passing `radec_offset`, the north vector will be
            calculated relative to `self.jds[time_index]`.
            Defaults to the central time index.

        Returns
        -------
        ls : np.ndarray
            Array containing EW direction cosine of each HEALPix pixel.
        ms : np.ndarray
            Array containing NS direction cosine of each HEALPix pixel.
        """
        if center is None:
            center = self.field_center
            if north is None and radec_offset is None:
                north = self.north_poles[self.nt//2]

        if radec_offset is not None:
            if time_index is None:
                jd = self.central_jd
            else:
                jd = self.jds[time_index]
            jd += radec_offset[0] * 1.0 / DEGREES_PER_DAY
            t = Time(jd, scale='utc', format='jd')

            # Calculate zenith angle in (alt, az)
            zen = AltAz(alt=Angle('90d'),
                        az=Angle('0d'),
                        obstime=t,
                        location=self.telescope_location)
            zen_radec = zen.transform_to(ICRS)
            center = (zen_radec.ra.deg, zen_radec.dec.deg)

            # Calculate north pole in (alt, az)
            north = AltAz(alt=Angle('0d'),
                          az=Angle('0d'),
                          obstime=t,
                          location=self.telescope_location)
            north_radec = north.transform_to(ICRS)
            north = (north_radec.ra.deg, north_radec.dec.deg)

        cvec = hp.ang2vec(center[0], center[1], lonlat=True)

        nvec = hp.ang2vec(north[0], north[1], lonlat=True)
        vecs = hp.pix2vec(self.nside, self.pix).T # Shape (npix, 3)

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
        ls = np.sin(za_arr) * np.sin(az_arr) # radians
        ms = np.sin(za_arr) * np.cos(az_arr) # radians

        return ls, ms

    def set_beam(self, beam_type=None, fwhm_deg=None, peak_amp=1.0):
        """
        Set beam params if not set in `__init__`.

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
        self.beam_type = beam_type

    def set_beam_center_radec(self, beam_center, rel=False):
        """
        Set the beam center (l, m) coordinates from a set of (RA, DEC)
        coordinates. This function is currently only working for
        snapshot observations. It will likely need to be updated
        to be calculated relative to the pointing center, not field
        center, per integration if it is to be used per time.

        Parameters
        ----------
        beam_center : tuple of floats
            Beam center can be passed as:
              - (RA, DEC) in units of degrees if `rel=False`
              - (RA_offset, DEC_offset) in units of degrees if
                `rel=True`.  This is assumed to be tuple of offsets
                relative to `self.pointing_center`.
        rel : boolean
            If True, assume `beam_center` is being passed as a tuple
            of offsets (RA_offset, DEC_offset) in units of degrees
            relative to `self.pointing_center`.
        """
        if rel:
            beam_center = (self.field_center[0] + beam_center[0],
                           self.field_center[1] + beam_center[1])
        # Input is in (RA, DEC) in units of degrees
        l0, m0 = beam_center
        center_radec = SkyCoord(l0, m0, unit='deg')
        # Convert from (RA, DEC) -> (alt, az)
        t = Time(self.central_jd, scale='utc', format='jd')
        center_altaz = center_radec.transform_to(
            AltAz(obstime=t, location=self.telescope_location)
            )
        # Convert from (alt, az) -> (l, m)
        self.l0 = (
                np.sin(np.pi / 2 - center_altaz.alt.rad)
                * np.sin(center_altaz.az.rad)
        )
        self.m0 = (
                np.sin(np.pi / 2 - center_altaz.alt.rad)
                * np.cos(center_altaz.az.rad)
        )

    def get_beam_vals(self,
                      ls,
                      ms,
                      beam_center=None,
                      rel=False):
        """
        Get an array of beam values from (l, m) coordinates.
        If `beam_type='gaussian'`, this function assumes that the
        beam width is symmetric along the l and m axes.

        Parameters
        ----------
        ls : np.ndarray of floats
            Array of EW direction cosine coordinate values.
        ms : np.ndarray of floats
            Array of NS direction cosine coordinate values.
        beam_center : tuple of floats
            Beam center can be passed as:
              - (RA, DEC) in units of degrees if `rel=False`
              - (RA_offset, DEC_offset) in units of degrees if
                `rel=True`.  This is assumed to be tuple of offsets
                relative to `self.pointing_center`.
        rel : boolean
            If True, assume `beam_center` is being passed as a tuple
            of offsets (RA_offset, DEC_offset) in units of degrees
            relative to `self.pointing_center`.

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
                self.set_beam_center_radec(beam_center, rel=rel)

            # Calculate Gaussian beam values from (l, m)
            beam_vals = self._gaussian_2d(
                ls, ms, self.l0, self.m0,
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