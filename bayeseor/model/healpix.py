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
from pyuvdata import UVBeam
from pyuvdata import utils as uvutils

c_ms = c.to('m/s').value

SECS_PER_HOUR = 60 * 60
SECS_PER_DAY = SECS_PER_HOUR * 24
DAYS_PER_SEC = 1.0 / SECS_PER_DAY
DEGREES_PER_DAY = 360.0
DEGREES_PER_HOUR = DEGREES_PER_DAY / 24
DEGREES_PER_SEC = DEGREES_PER_HOUR * 1 / SECS_PER_HOUR
HERA_LAT_LON_ALT = (
    -30.72152777777791,  # deg
    21.428305555555557,  # deg
    1073.0000000093132  # meters
)

class Healpix(HEALPix):
    """
    Class to store and manipulate HEALPix maps using `astropy_healpix`
    functions.

    Parameters
    ----------
    fov_ra_eor : float
        Field of view in degrees of the RA axis of the EoR sky model.
    fov_dec_eor : float, optional
        Field of view in degrees of the DEC axis of the EoR sky model.
        Defaults to `fov_ra_eor`.
    fov_ra_fg : float, optional
        Field of view in degrees of the RA axis of the FG sky model.
    fov_dec_fg : float, optional
        Field of view in degrees of the DEC axis of the FG sky model.
        Defaults to `fov_ra_fg`.
    simple_za_filter : boolean, optional
        If True, filter pixels in the FoV by zenith angle only.  Otherwise,
        filter pixels in a rectangular region set by the FoV values along
        RA and DEC.
    nside : int
        Nside resolution of the HEALPix map.  Defaults to 256.
    telescope_latlonalt : tuple, optional
        Tuple containing the latitude, longitude, and altitude of the
        telescope in degrees, degrees, and meters, respectively.  Defaults
        to the location of the HERA telescope, i.e. (-30.72152777777791, 
        21.428305555555557, 1073.0000000093132).
    central_jd : float
        Central time step of the observation in JD2000 format.
    nt : int, optional
        Number of time integrations. Defaults to 1.
    int_time : float, optional
        Integration time in seconds. Required if ``nt > 1``.
    beam_type : str, optional
        Can be either a path to a pyuvdata.UVBeam compatible
        file or one of {'uniform', 'gaussian', 'airy'}.  Defaults to 'uniform'.
    peak_amp : float, optional
        Peak amplitude of the beam.  Defaults to 1.0.
    fwhm_deg : float, optional
        Required if ``beam_type = 'gaussian'``. Sets the full width at half
        maximum of the beam in degrees.
    diam : float, optional
        Antenna (aperture) diameter in meters.  Used if ``beam_type = 'airy'``.
    cosfreq : float, optional
        Cosine frequency in inverse radians.  Used if
        ``beam_type = 'gausscosine'``.
    tanh_freq : float, optional
        Exponential frequency (rate parameter) in inverse radians.  Used if
        ``beam_type = 'tanhairy'``.
    tanh_sl_red : float, optional
        Airy sidelobe amplitude reduction as a fractional percent.  For
        example, passing 0.99 reduces the sidelobes by 0.01, i.e. two orders
        of magnitude.  Used if ``beam_type = 'tanhairy'``.

    """
    def __init__(
            self,
            fov_ra_eor=None,
            fov_dec_eor=None,
            fov_ra_fg=None,
            fov_dec_fg=None,
            simple_za_filter=False,
            nside=256,
            telescope_latlonalt=HERA_LAT_LON_ALT,
            central_jd=None,
            nt=1,
            int_time=None,
            beam_type=None,
            peak_amp=1.0,
            fwhm_deg=None,
            diam=None,
            cosfreq=None,
            tanh_freq=None,
            tanh_sl_red=None
            ):
        # Use HEALPix as parent class to get useful astropy_healpix functions
        super().__init__(nside, frame=ICRS())

        assert fov_ra_eor is not None, \
            "Missing required keyword argument: fov_ra_eor ."

        self.fov_ra_eor = fov_ra_eor
        if fov_dec_eor is None:
            self.fov_dec_eor = self.fov_ra_eor
        else:
            self.fov_dec_eor = fov_dec_eor
        
        if fov_ra_fg is None:
            self.fov_ra_fg = self.fov_ra_eor
            self.fov_dec_fg = self.fov_dec_eor
            self.fovs_match = True
        else:
            assert fov_ra_fg >= fov_ra_eor, \
                "fov_ra_fg must be greater than or equal to fov_ra_eor."
            self.fov_ra_fg = fov_ra_fg
            if fov_dec_fg is None:
                self.fov_dec_fg = self.fov_ra_fg
            else:
                self.fov_dec_fg = fov_dec_fg
            self.fovs_match = np.logical_and(
                self.fov_ra_eor == self.fov_ra_fg,
                self.fov_dec_eor == self.fov_dec_fg
            )

        self.pixel_area_sr = self.pixel_area.to('sr').value
        self.tele_lat, self.tele_lon, self.tele_alt = telescope_latlonalt
        # Set telescope location
        telescope_xyz = uvutils.XYZ_from_LatLonAlt(
            self.tele_lat * np.pi / 180,
            self.tele_lon * np.pi / 180,
            self.tele_alt
        )
        self.telescope_location = EarthLocation.from_geocentric(
            *telescope_xyz, unit='m'
        )

        # Calculate field center in (RA, DEC)
        self.central_jd = central_jd
        t = Time(self.central_jd, scale='utc', format='jd')
        zen = AltAz(
            alt=Angle('90d'),
            az=Angle('0d'),
            obstime=t,
            location=self.telescope_location
        )
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
            self.jds = (
                self.central_jd
                + self.time_inds * self.int_time * DAYS_PER_SEC
            )
        else:
            self.jds = np.array([self.central_jd])
        # Calculate pointing center per integration
        self.pointing_centers = []
        for jd in self.jds:
            t = Time(jd, scale='utc', format='jd')
            zen = AltAz(
                alt=Angle('90d'),
                az=Angle('0d'),
                obstime=t,
                location=self.telescope_location
            )
            zen_radec = zen.transform_to(ICRS())
            self.pointing_centers.append((zen_radec.ra.deg, zen_radec.dec.deg))

        # Beam params
        if beam_type is not None:
            if not '.' in str(beam_type):
                beam_type = beam_type.lower()
                allowed_types = [
                    'uniform', 'gaussian', 'airy', 'gausscosine', 'taperairy',
                    'tanhairy'
                ]
                assert beam_type in allowed_types, \
                    f"Only {', '.join(allowed_types)} beams are supported."
                self.beam_type = beam_type
                self.uvb = None
            else:
                # assume beam_type is a path to a UVBeam compatible file
                uvb = UVBeam()
                uvb.read_beamfits(beam_type)
                assert uvb.beam_type == 'power', (
                    "UVBeam.beam_type must be 'power', not '{}'.".format(
                        uvb.beam_type
                    )
                )
                if 1 in uvb.polarization_array:
                    uvb.select(polarizations=[1])
                elif -5 in uvb.polarization_array:
                    # this works for now, but if we're analyzing different
                    # polarizations in the future, we need to add a param
                    # specifying polarization (in params.py and as a CLA)
                    uvb.select(polarizations=[-5])
                uvb.freq_interp_kind = 'quadratic'
                if uvb.pixel_coordinate_system == 'healpix':
                    uvb.interpolation_function = 'healpix_simple'
                else:
                    uvb.interpolation_function = 'az_za_simple'
                self.beam_type = 'uvbeam'
                self.uvb = uvb
        else:
            self.beam_type = 'uniform'
            self.uvb = None
        self.peak_amp = peak_amp

        if beam_type == 'gaussian':
            required_params = [diam, fwhm_deg]
            assert self._check_required_params(required_params, all_req=False),\
                "If using a Gaussian beam, must pass either " \
                "'fwhm_deg' or 'diam'."
        elif beam_type == 'airy':
            required_params = [diam, fwhm_deg]
            assert self._check_required_params(required_params, all_req=False),\
                "If using an Airy beam, must pass either " \
                "'fwhm_deg' or 'diam'."
        elif beam_type == 'taperairy':
            required_params = [diam, fwhm_deg]
            assert self._check_required_params(required_params), \
                "If using a taperairy beam, must pass " \
                "'diam' and 'fwhm_deg'."
        elif beam_type == 'gausscosine':
            required_params = [fwhm_deg, cosfreq]
            assert self._check_required_params(required_params), \
                "If using a gausscosine beam, must pass " \
                "'fwhm_deg' and 'cosfreq'."
        elif beam_type == 'tanhairy':
            required_params = [diam, tanh_freq, tanh_sl_red]
            assert self._check_required_params(required_params), \
                "If using a tanhairy beam, must pass " \
                "'diam', 'tanh_freq', and 'tanh_sl_red'."
        self.fwhm_deg = fwhm_deg
        self.diam = diam
        self.cosfreq = cosfreq
        self.tanh_freq = tanh_freq
        self.tanh_sl_red = tanh_sl_red

        # Pixel filters
        self.simple_za_filter = simple_za_filter
        pix_eor, ra_eor, dec_eor = self.get_pixel_filter(
            self.fov_ra_eor, self.fov_dec_eor, return_radec=True,
            simple_za_filter=self.simple_za_filter
        )
        self.pix_eor = pix_eor
        self.ra_eor = ra_eor
        self.dec_eor = dec_eor
        self.npix_fov_eor = self.pix_eor.size

        if self.fovs_match:
            self.pix_fg = self.pix_eor.copy()
            self.ra_fg = self.ra_eor.copy()
            self.dec_fg = self.dec_eor.copy()
            self.npix_fov_fg = self.pix_fg.size
        else:
            pix_fg, ra_fg, dec_fg = self.get_pixel_filter(
                self.fov_ra_fg, self.fov_dec_fg, return_radec=True,
                simple_za_filter=self.simple_za_filter
            )
            self.pix_fg = pix_fg
            self.npix_fov_fg = self.pix_fg.size
            self.ra_fg = ra_fg
            self.dec_fg = dec_fg
        self.pix = self.pix_fg
        self.ra = self.ra_fg
        self.dec = self.dec_fg
        self.npix_fov = self.npix_fov_fg
        self.fov_ra = self.fov_ra_fg
        self.fov_dec = self.fov_dec_fg
        # If the FoV values of the two models are different, so to are their
        # HEALPix pixel index arrays.  This mask allows you to take a set of
        # pixel values for the EoR model and propagate them into the FG model.
        self.eor_to_fg_pix = np.in1d(self.pix_fg, self.pix_eor)

    def get_pixel_filter(
            self, fov_ra, fov_dec, return_radec=False, inverse=False,
            simple_za_filter=False):
        """
        Return HEALPix pixel indices lying inside an observed region.

        This function gets the HEALPix pixel indices for all pixel centers
        lying inside

            - a rectangle with equal arc length on all sides if
              ``simple_za_filter = True``
            - a circle with radius `fov_ra` if ``simple_za_filter = False``

        Parameters
        ----------
        fov_ra : float
            Field of view in degrees of the RA axis.
        fov_dec : float
            Field of view in degrees of the DEC axis.
        return_radec : bool, optional
            Return the (RA, DEC) coordinates associated with each pixel center.
            Defaults to False.
        inverse : boolean, optional
            If `False`, return the pixels within the observed region.
            If `True`, return the pixels outside the observed region.
        simple_za_filter : boolean, optional
            If `True`, return the pixels inside a circular region defined by
            ``za <= fov_ra/2``.  Otherwise, return the pixels inside a
            rectangular region with equal arc length on all sides.

        Returns
        -------
        pix : array
            HEALPix pixel numbers lying within the observed region set by
            `fov_ra` and `fov_dec`.
        ra : array
            Array of RA values for each pixel center.  Only returned if
            ``return_radec = True``.
        dec : array
            Array of DEC values for each pixel center.  Only returned if
            ``return_radec = True``.

        """
        lons, lats = hp.pix2ang(
            self.nside,
            np.arange(self.npix),
            lonlat=True
        )
        if simple_za_filter:
            _, _, _, _, za = self.calc_lmn_from_radec(
                self.central_jd, lons, lats, return_azza=True
            )
            max_za = np.deg2rad(fov_ra) / 2
            pix = np.where(za <= max_za)[0]
        else:
            thetas = (90 - lats) * np.pi / 180
            if self.field_center[0] - fov_ra/2 < 0:
                lons[lons > 180] -= 360  # lons in (-180, 180]
            lons_inds = np.logical_and(
                (lons - self.field_center[0])*np.sin(thetas) >= -fov_ra/2,
                (lons - self.field_center[0])*np.sin(thetas) <= fov_ra/2,
                )
            lats_inds = np.logical_and(
                lats >= self.field_center[1] - fov_dec / 2,
                lats <= self.field_center[1] + fov_dec / 2
                )
            if inverse:
                pix = np.where(np.logical_not(lons_inds * lats_inds))[0]
            else:
                pix = np.where(lons_inds * lats_inds)[0]
            lons[lons < 0] += 360  # RA in [0, 360)
        if not return_radec:
            return pix
        else:
            return pix, lons[pix], lats[pix]
    
    def get_extent_ra_dec(self, fov_ra, fov_dec, fov_fac=1.0):
        """
        Get the sampled extent of the sky in RA and DEC.

        Parameters
        ----------
        fov_ra : float
            Field of view in degrees of the RA axis.
        fov_dec : float
            Field of view in degrees of the DEC axis.
        fov_fac : float
            Scaling factor for the sampled extent.

        Returns
        -------
        range_ra : tuple
            `fov_fac` scaled (min, max) sampled RA values.
        range_dec : tuple
            `fov_fac` scaled (min, max) sampled DEC values.

        """
        range_ra = [
            self.field_center[0] - fov_fac*fov_ra/2,
            self.field_center[0] + fov_fac*fov_ra/2
        ]
        range_dec = [
            self.field_center[1] - fov_fac*fov_dec/2,
            self.field_center[1] + fov_fac*fov_dec/2
        ]
        return range_ra, range_dec

    def calc_lmn_from_radec(
            self, time, ra, dec, return_azza=False, radec_offset=None):
        """
        Return arrays of (l, m, n) coordinates in radians for all (RA, DEC).

        Parameters
        ----------
        time : float
            Julian date used in ICRS to AltAz coordinate frame conversion.
        ra : array
            Array of RA values in degrees.
        dec : array
            Array of DEC values in degrees.
        return_azza : boolean
            If True, return both (l, m, n) and (az, za) coordinate arrays.
            Otherwise return only (l, m, n).  Defaults to 'False'.
        radec_offset : tuple of floats
            Will likely be deprecated.

        Returns
        -------
        l : np.ndarray of floats
            Array containing the EW direction cosine of each HEALPix pixel.
        m : np.ndarray of floats
            Array containing the NS direction cosine of each HEALPix pixel.
        n : np.ndarray of floats
            Array containing the radial direction cosine of each HEALPix pixel.

        Notes
        -----
        * Adapted from `pyradiosky.skymodel.update_positions` 
          (https://github.com/RadioAstronomySoftwareGroup/pyradiosky).

        """
        if not isinstance(time, Time):
            time = Time(time, format='jd')

        skycoord = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
        altaz = skycoord.transform_to(
            AltAz(obstime=time, location=self.telescope_location)
        )
        az = altaz.az.rad
        za = np.pi/2 - altaz.alt.rad

        # Convert from (az, za) to (l, m, n)
        ls = np.sin(za) * np.sin(az)
        ms = np.sin(za) * np.cos(az)
        ns = np.cos(za)

        if return_azza:
            return ls, ms, ns, az, za
        else:
            return ls, ms, ns

    def get_beam_vals(self, az, za, freq=None):
        """
        Get an array of beam values from (az, za) coordinates.
        
        If ``self.beam_type = 'gaussian'``, this function assumes that the
        beam width is symmetric along the l and m axes.

        Parameters
        ----------
        az : np.ndarray of floats
            Azimuthal angle of each pixel in radians.
        za : np.ndarray of floats
            Zenith angle of each pixel in radians.
        freq : float, optional
            Frequency in Hz.

        Returns
        -------
        beam_vals : np.ndarray
            Array containing beam amplitude values at each (az, za).

        """
        if self.beam_type == 'uniform':
            beam_vals = np.ones(self.npix_fov)

        # elif self.beam_type == 'gaussian':
        elif self.beam_type in ['gaussian', 'gausscosine']:
            if self.fwhm_deg is not None:
                stddev_rad = np.deg2rad(
                    self._fwhm_to_stddev(self.fwhm_deg)
                )
            else:
                stddev_rad = self._diam_to_stddev(self.diam, freq)
            if self.beam_type == 'gaussian':
                beam_vals = self._gaussian_za(za, stddev_rad, self.peak_amp)
            else:
                beam_vals = self._gausscosine(
                    za, stddev_rad, self.peak_amp, self.cosfreq
                )

        elif self.beam_type == 'airy':
            if self.diam is not None:
                beam_vals = self._airy_disk(za, self.diam, freq)
            else:
                diam_eff = self._fwhm_to_diam(self.fwhm_deg, freq)
                beam_vals = self._airy_disk(za, diam_eff, freq)
        
        elif self.beam_type == 'taperairy':
            stddev_rad = np.deg2rad(self._fwhm_to_stddev(self.fwhm_deg))
            beam_vals = (
                self._airy_disk(za, self.diam, freq)
                * self._gaussian_za(za, stddev_rad, self.peak_amp)
            )
        
        elif self.beam_type == 'tanhairy':
            beam_vals = (
                self._airy_disk(za, self.diam, freq)
                * self._tanh_taper(za, self.tanh_freq, self.tanh_sl_red)
            )
        
        elif self.beam_type == 'uvbeam':
            beam_vals, _ = self.uvb.interp(
                az_array=az, za_array=za, freq_array=np.array([freq]),
                reuse_spline=False
            )
            beam_vals = beam_vals[0, 0, 0, 0].real

        return beam_vals

    def _gaussian_za(self, za, sigma, amp):
        """
        Calculate azimuthally symmetric Gaussian beam amplitudes.

        Parameters
        ----------
        za : np.ndarray
            Zenith angle of each pixel in radians.
        sigma : float
            Standard deviation in radians.
        amp : float
            Peak amplitude at ``za=0``.

        Returns
        -------
        beam_vals : np.ndarray
            Array of Gaussian beam amplitudes for each zenith angle in `za`.

        """
        beam_vals = amp * np.exp(-za ** 2 / (2 * sigma ** 2))
        return beam_vals
    
    def _gausscosine(self, za, sigma, amp, cosfreq):
        """
        Calculate azimuthally symmetric Gaussian * cosine^2 beam amplitudes.

        Parameters
        ----------
        za : np.ndarray
            Zenith angle of each pixel in radians.
        sigma : float
            Standard deviation in radians.
        amp : float
            Peak amplitude at ``za=0``.
        cosfreq : float
            Cosine squared frequency in inverse radians.

        Returns
        -------
        beam_vals : np.ndarray
            Array of Gaussian beam amplitudes for each zenith angle in `za`.

        """
        beam_vals = amp * np.exp(-za ** 2 / (2 * sigma ** 2))
        beam_vals *= np.cos(2 * np.pi * za * cosfreq/2)**2
        return beam_vals

    def _fwhm_to_stddev(self, fwhm):
        """
        Calculate standard deviation from full width at half maximum.

        Parameters
        ----------
        fwhm : float
            Full width half maximum in degrees.

        """
        return fwhm / 2.355

    def _airy_disk(self, za, diam, freq):
        """
        Calculate Airy disk amplitudes.

        Parameters
        ----------
        za : np.ndarray of floats
            Zenith angle of each pixel in radians.
        diam : float
            Antenna (aperture) diameter in meters.
        freq : float
            Frequency in Hz.

        Returns
        -------
        beam_vals : np.ndarray
            Array of Airy disk amplitudes for each zenith angle in `za`.

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
    
    def _tanh_taper(self, za, tanh_freq, tanh_sl_red):
        """
        Calculate a tanh tapering function.

        Parameters
        ----------
        za : np.ndarray of floats
            Zenith angle of each pixel in radians.
        tanh_freq : float, optional
            Exponential frequency (rate parameter) in inverse radians.
        tanh_sl_red : float, optional
            Airy sidelobe amplitude reduction as a fractional percent.  For
            example, passing 0.99 reduces the sidelobes by 0.01, i.e. two
            orders of magnitude.
        
        Returns
        -------
        taper_vals : np.ndarray
            Array of tanh taper amplitudes for each zenith angle in `za`.

        """
        taper_vals = 1 - tanh_sl_red * np.tanh(tanh_freq * za)
        return taper_vals

    def _fwhm_to_diam(self, fwhm, freq):
        """
        Calculates the effective diameter of an Airy disk from a FWHM.

        Parameters
        ----------
        fwhm : float
            Full width at half maximum of a Gaussian beam in degrees.
        freq : float
            Frequency in Hz.

        Returns
        -------
        diam : float
            Antenna (aperture) diameter in meters with an Airy disk beam
            pattern whose main lobe is described by a Gaussian beam with a
            FWHM of `fwhm`.

        Notes
        -----
        * Modified from `pyuvsim.analyticbeam.diameter_to_sigma`
          (https://github.com/RadioAstronomySoftwareGroup/pyuvsim).


        """
        scalar = 2.2150894
        wavelength = c_ms / freq
        fwhm = np.deg2rad(fwhm)
        diam = scalar * wavelength / (np.pi * np.sin(fwhm / np.sqrt(2)))
        return diam

    def _diam_to_stddev(self, diam, freq):
        """
        Calculate an effective standard deviation of an Airy disk.

        Parameters
        ----------
        diam : float
            Antenna (aperture) diameter in meters.
        freq : float
            Frequency in Hz.

        Returns
        -------
        sigma : float
            Standard deviation of a Gaussian envelope which describes the main
            lobe of an Airy disk with aperture `diam`.

        Notes
        -----
        * Copied from `pyuvsim.analyticbeam.diameter_to_sigma`
          (https://github.com/RadioAstronomySoftwareGroup/pyuvsim).

        """
        scalar = 2.2150894
        wavelength = c_ms / freq
        sigma = np.arcsin(scalar * wavelength / (np.pi * diam))
        sigma *= np.sqrt(2) / 2.355
        return sigma

    def _check_required_params(self, required_params, all_req=True):
        """
        Check if params in required_params are not None.

        Parameters
        ----------
        required_params : iterable
            Iterable of param values.
        all_req : bool
            If True, require all params are not None.  Otherwise, require
            that only one param is not None. Defaults to True.

        """
        if not all_req:
            return np.any([p is not None for p in required_params])
        else:
            return np.all([p is not None for p in required_params])
