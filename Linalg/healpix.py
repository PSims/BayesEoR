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
from pyuvdata import utils as uvutils

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
    fov_ra_deg : float
        Field of view in degrees of the RA axis of the sky model.
    fov_dec_deg : float
        Field of view in degrees of the DEC axis of the sky model.  
        Defaults to `fov_ra_deg`.
    nside : int
        Nside resolution of the HEALPix map.  Defaults to 256.
    telescope_latlonalt : tuple
        Tuple containing the latitude, longitude, and altitude of the
        telescope in degrees, degrees, and meters, respectively.
    central_jd : float
        Central time step of the observation in JD2000 format.
    nt : int
        Number of time integrations. Defaults to 1.
    int_time : float, optional
        Integration time in seconds. Required if `nt` > 1.
    beam_type : str, optional
        Can be either (case insensitive) 'uniform', 'gaussian',
        or 'airy'.  Defaults to 'uniform'.
    peak_amp : float, optional
        Peak amplitude of the beam.  Defaults to 1.0.
    fwhm_deg : float, optional
        Required if `beam_type` = 'gaussian'. Sets the full width at half
        maximum of the beam in degrees.
    diam : float, optional
        Effective diameter of the antenna in meters.  Used if
        `beam_type` = 'airy'.

    """
    def __init__(
            self,
            fov_ra_deg=None,
            fov_dec_deg=None,
            nside=256,
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

        self.fov_ra_deg = fov_ra_deg
        if fov_dec_deg is None:
            self.fov_dec_deg = fov_ra_deg
        else:
            self.fov_dec_deg = fov_dec_deg
        self.pixel_area_sr = self.pixel_area.value
        self.lat, self.lon, self.alt = telescope_latlonalt
        # Set telescope location
        telescope_xyz = uvutils.XYZ_from_LatLonAlt(
            self.lat * np.pi / 180,
            self.lon * np.pi / 180,
            self.alt
            )
        self.telescope_location = EarthLocation.from_geocentric(
            *telescope_xyz, unit='m'
            )

        # Calculate field center in (RA, DEC)
        self.central_jd = central_jd
        t = Time(self.central_jd, scale='utc', format='jd')
        zen = AltAz(alt=Angle('90d'),
                    az=Angle('0d'),
                    obstime=t,
                    location=self.telescope_location)
        zen_radec = zen.transform_to(ICRS())
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
        # Calculate pointing center per integration
        self.pointing_centers = []
        for jd in self.jds:
            t = Time(jd, scale='utc', format='jd')
            # Calculate zenith angle in (alt, az)
            zen = AltAz(alt=Angle('90d'),
                        az=Angle('0d'),
                        obstime=t,
                        location=self.telescope_location)
            zen_radec = zen.transform_to(ICRS())
            self.pointing_centers.append((zen_radec.ra.deg, zen_radec.dec.deg))

        # Beam params
        if beam_type is not None:
            beam_type = beam_type.lower()
            assert beam_type in ['uniform', 'gaussian', 'airy'], \
                "Only uniform, Gaussian, and Airy beams are currently supported."
            self.beam_type = beam_type
        else:
            self.beam_type = 'uniform'
        self.peak_amp = peak_amp

        if beam_type == 'gaussian':
            assert diam is not None or fwhm_deg is not None, \
                "If using a Gaussian beam, must also pass either " \
                "'fwhm_deg' or 'diam'."
        elif beam_type == 'airy':
            assert diam is not None or fwhm_deg is not None, \
                "If using an Airy beam, must also pass either " \
                "'fwhm_deg' or 'diam'."
        self.fwhm_deg = fwhm_deg
        self.diam = diam

        # Pixel params
        self.pix = None  # HEALPix pixel numbers within the FoV
        self.npix_fov = None  # Number of pixels within the FoV
        self.ra = None # Array to hold right ascension values
        self.dec = None # Array to hold declination values
        # Set self.pix and self.npix_fov
        self.set_pixel_filter()

    def set_pixel_filter(self, inverse=False):
        """
        Filter pixels that lie outside of a rectangular region
        centered on `self.field_center`.
        
        Parameters
        ----------
        inverse: boolean
            If False, return the pixels within the rectangular region.
            If True, return the pixels outside the rectangular region.

        This function sets:
        pix : array of ints
            Array of HEALPix pixel indices lying within the rectangular
            region with shape (npix_fov,).
        npix_fov : int
            Number of HEALPix pixels lying within the rectangular region
        ra : array of floats
            Array of right ascension values for each pixel in
            `self.pix`.
        dec : array of floats
            Array of declination values for each pixel in `self.pix`.
        """
        lons, lats = hp.pix2ang(
            self.nside,
            np.arange(self.npix),
            lonlat=True
            )
        thetas = (90 - lats) * np.pi / 180
        lons_inds = np.logical_and(
            (lons - self.field_center[0])*np.sin(thetas) >= -self.fov_ra_deg/2,
            (lons - self.field_center[0])*np.sin(thetas) <= self.fov_ra_deg/2,
            )
        lats_inds = np.logical_and(
            lats >= self.field_center[1] - self.fov_dec_deg / 2,
            lats <= self.field_center[1] + self.fov_dec_deg / 2
            )
        if inverse:
            pix = np.where(np.logical_not(lons_inds * lats_inds))[0]
        else:
            pix = np.where(lons_inds * lats_inds)[0]
        self.pix = pix
        self.npix_fov = pix.size
        self.ra = lons[pix]
        self.dec = lats[pix]

    def calc_lmn_from_radec(self, time, return_azza=False, radec_offset=None):
        """
        Return arrays of (l, m, n) coordinates in radians of all
        HEALPix pixels within the region set by `self.pix`. The pixels
        used in this calculation are set by `self.set_pixel_filter()`.

        Adapted from `pyradiosky.skymodel.update_positions`.

        Parameters
        ----------
        time : float
            Julian date at which to convert from
            ICRS to AltAz coordinate frames.
        return_azza : boolean
            If True, return both (l, m) and (az, za)
            coordinate arrays.  Otherwise return only
            (l, m).  Defaults to 'False'.
        radec_offset : tuple of floats
            Will likely be deprecated.

        Returns
        -------
        l : np.ndarray of floats
            Array containing the EW direction cosine
            of each HEALPix pixel.
        m : np.ndarray of floats
            Array containing the NS direction cosine
            of each HEALPix pixel.
        n : np.ndarray of floats
            Array containing the radial direction
            cosine of each HEALPix pixel.
        """
        if not isinstance(time, Time):
            time = Time(time, format='jd')

        skycoord = SkyCoord(self.ra*u.deg, self.dec*u.deg, frame='icrs')
        altaz = skycoord.transform_to(
            AltAz(obstime=time, location=self.telescope_location)
            )
        az = altaz.az.rad
        za = np.pi/2 - altaz.alt.rad

        # Convert from (az, za) to (l, m, n)
        l = np.sin(za) * np.sin(az)
        m = np.sin(za) * np.cos(az)
        n = np.cos(za)

        if return_azza:
            return l, m, n, az, za
        else:
            return l, m, n

    def get_beam_vals(self, az, za, freq=None):
        """
        Get an array of beam values from (l, m) coordinates.
        If `beam_type='gaussian'`, this function assumes that the
        beam width is symmetric along the l and m axes.

        Parameters
        ----------
        az : np.ndarray of floats
            Azimuthal angle of each pixel in units of radians.
        za : np.ndarray of floats
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
            # Calculate Gaussian beam values from za
            if self.fwhm_deg is not None:
                stddev_rad = np.deg2rad(
                    self._fwhm_to_stddev(self.fwhm_deg)
                    )
            else:
                stddev_rad = self._diam_to_stddev(self.diam, freq)
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
        return fwhm / 2.355

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

    def _diam_to_stddev(self, diam, freq):
        """
        Approximates the effective stddev [deg] of an Airy disk
        corresponding to an antenna with a diameter of `diam`
        in meters.

        Copied from `pyuvsim.analyticbeam.diameter_to_sigma`.
        """
        scalar = 2.2150894
        wavelength = c_ms / freq
        sigma = np.arcsin(scalar * wavelength / (np.pi * diam))
        sigma *= np.sqrt(2) / 2.355
        return sigma
