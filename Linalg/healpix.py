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
from astropy.constants import c
from scipy.special import j1

c_ms = c.to('m/s').value

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
        Can be either (case insensitive) 'uniform', 'gaussian',
        or 'airy'. Specifies the type of beam to use.
        Defaults to 'uniform'.
    peak_amp : float, optional
        Sets the peak amplitude of the beam.  Defaults to 1.0.
    fwhm_deg : float, optional
        Required if `beam_type='gaussian'`. Sets the full width half
        maximum of the beam in degrees.
    diam : float, optional
        Effective diameter of the antenna in meters.  Used if
        `beam_type = 'airy'`.
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
            diam=None
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
            self.time_inds = np.arange(-(self.nt // 2), self.nt // 2 + 1)
        else:
            self.time_inds = np.arange(-(self.nt // 2), self.nt // 2)
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
            beam_type = beam_type.lower()
            assert beam_type in ['uniform', 'gaussian', 'airy'], \
                "Only uniform, Gaussian, and Airy beams are currently supported."
            self.beam_type = beam_type
        self.peak_amp = peak_amp

        if beam_type == 'gaussian':
            assert fwhm_deg is not None, \
                "If using a Gaussian beam, must also pass fwhm_deg."
        elif beam_type == 'airy':
            assert diam is not None or fwhm_deg is not None, \
                "If using an Airy beam, must also pass either " \
                "fwhm_deg or diam."
        self.fwhm_deg = fwhm_deg
        self.diam = diam

        # Pixel params
        self.pix = None  # HEALPix pixel numbers within the FoV
        self.npix_fov = None  # Number of pixels within the FoV
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
        thetas = (90 - lats) * np.pi / 180
        lons_inds = np.logical_and(
            (lons - self.field_center[0]) * np.sin(thetas) >= -self.fov_deg / 2,
            (lons - self.field_center[0]) * np.sin(thetas) <= self.fov_deg / 2,
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
                           time_index=0,
                           return_azza=False):
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
                north = self.north_poles[self.nt // 2]

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
        vecs = hp.pix2vec(self.nside, self.pix).T  # Shape (npix, 3)

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
        ls = np.sin(za_arr) * np.sin(az_arr)  # radians
        ms = np.sin(za_arr) * np.cos(az_arr)  # radians

        if return_azza:
            return ls, ms, az_arr, za_arr
        else:
            return ls, ms

    def get_beam_vals(self,
                      az,
                      za,
                      freq=None):
        """
        Get an array of beam values from (l, m) coordinates.
        If `beam_type='gaussian'`, this function assumes that the
        beam width is symmetric along the l and m axes.

        Parameters
        ----------
        az : np.ndarray
            Azimuthal angle of each pixel in units of radians.
        za : np.ndarray
            Zenith angle of each pixel in units of radians.
        freq : float, optional
            Frequency in Hz at which to calculate the beam.

        Returns
        -------
        beam_vals : np.ndarray
            Array containing beam amplitude values at each (az, za).
        """
        if self.beam_type == 'uniform':
            beam_vals = np.ones(self.npix_fov)

        elif self.beam_type == 'gaussian':
            stddev_rad = np.deg2rad(self._fwhm_to_stddev(self.fwhm_deg))

            # Calculate Gaussian beam values from za
            beam_vals = self._gaussian_za(za, stddev_rad, self.peak_amp)

        elif self.beam_type == 'airy':
            if self.diam is not None:
                beam_vals = self._airy_disk(za, self.diam, freq)
            else:
                diam_eff = self._fwhm_to_diam(self.fwhm_deg, freq)
                beam_vals = self._airy_disk(za, diam_eff, freq)

        return beam_vals

    def _gaussian_za(self, za, sigma, amp):
        """
        Calculates the value of an azimuthally symmetric
        Gaussian function from an array of zenith angles.

        Parameters
        ----------
        za : np.ndarray
            Zenith angles of each pixel in units of radians.
        sigma : float
            Standard deviation of the Gaussian function
            in units of radians.
        amp : float
            Peak amplitude of the Gaussian function.

        Returns
        -------
        beam_vals : np.ndarray
            Array of Gaussian function amplitudes for each
            zenith angle in `za`.
        """
        beam_vals = amp * np.exp(-za ** 2 / (2 * sigma ** 2))
        return beam_vals

    def _fwhm_to_stddev(self, fwhm):
        """
        Converts a full width half maximum to a standard deviation
        for a Gaussian beam.
        """
        return fwhm / (2 * np.sqrt(2 * np.log(2)))

    def _airy_disk(self, za, diam, freq):
        """
        Airy disk calculation from an array of zenith angles.

        Parameters
        ----------
        za : np.ndarray of floats
            Zenith angle of each pixel in units of radians.
        diam : float
            Antenna diameter in units of meters.
        freq : float
            Frequency in Hz.

        Returns
        -------
        beam_vals : np.ndarray
            Array of Airy disk amplitudes for each zenith
            angle in `za`.
        """
        xvals = (
                diam / 2. * np.sin(za)
                * 2. * np.pi * freq / c_ms
        )
        beam_vals = np.zeros_like(xvals)
        nz = xvals != 0.
        ze = xvals == 0.
        beam_vals[nz] = 2. * j1(xvals[nz]) / xvals[nz]
        beam_vals[ze] = 1.
        return beam_vals ** 2

    def _fwhm_to_diam(self, fwhm, freq):
        """
        Converts the FWHM [deg] of a Gaussian into an effective
        dish diameter for use in the calculation of an
        Airy disk.

        Modified from `pyuvsim.analyticbeam.diameter_to_sigma`.
        """
        scalar = 2.2150894
        wavelength = c_ms / freq
        fwhm = np.deg2rad(fwhm)
        diam = (scalar * wavelength
                / (np.pi * np.sin(fwhm / np.sqrt(2)))
                )
        return diam