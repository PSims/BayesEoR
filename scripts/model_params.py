"""
Determine the required values of N_u and N_v given an instrument model (uv 
uv sampling), field of view parameters, frequencies, and a beam model (
one of Gaussian, Airy, or Uniform).
"""
import numpy as np
from astropy import units
from astropy.constants import c
import astropy_healpix as ahp
from rich import print as rprint
from rich.panel import Panel

from bayeseor.params import BayesEoRParser
from bayeseor.model import load_inst_model


parser = BayesEoRParser()
parser.add_argument(
    "--Nsigma",
    type=float,
    default=5,
    help="Number of standard deviations to include in the aperture function "
         "if using a Gaussian beam.  Defaults to 5."
)
parser.add_argument(
    "--plot",
    action="store_true",
    help="If passed, plot the model uv planes for the EoR and FG models with "
         "the model grid and the aperture function as a circle with the "
         "diameter calculated based on the beam model in wavelengths.  The "
         "aperture function is only plotted for the baseline along the u (v) "
         "axis with the largest |u| (|v|)."
)
args = parser.parse_args(derived_params=False)

required_args = np.array([
    "fov_ra_eor",
    "nf",
    "nu_min_MHz",
    "channel_width_MHz",
    "beam_type",
    "inst_model"
])
missing_args = [args.__dict__[key] is None for key in required_args]
assert np.all(np.logical_not(missing_args)), (
    f"The following parameters are required: {', '.join(required_args)}\n"
    f"The following parameters are missing: "
    f"{', '.join(required_args[missing_args])}\n"
)

assert args.beam_type in ["airy", "gaussian", "uniform"], (
    "beam_type must be one of 'airy', 'gaussian', or 'uniform'"
)
if args.beam_type == "airy":
    assert args.antenna_diameter is not None, (
        "antenna_diameter required for Airy beam"
    )
elif args.beam_type == "gaussian":
    assert args.fwhm_deg is not None, (
        "fwhm_deg required for Gaussian beam"
    )

if args.plot:
    import matplotlib.pyplot as plt

    def plot_model_uv_grid(ax, Nu, delta_u, Nv, delta_v):
        u_centers_labels = (np.arange(Nu) - Nu//2)
        u_centers = u_centers_labels * delta_u.value
        u_edges = (np.arange(Nu + 1) - Nu//2) * delta_u.value
        u_edges -= delta_u.value/2
        ax.set_xticks(u_edges, minor=True)
        ax.set_xlim([u_edges[0], u_edges[-1]])
        v_centers_labels = (np.arange(Nv) - Nv//2)
        v_centers = v_centers_labels * delta_v.value
        v_edges = (np.arange(Nv + 1) - Nv//2) * delta_v.value
        v_edges -= delta_v.value/2
        ax.set_yticks(v_edges, minor=True)
        ax.set_ylim([v_edges[0], v_edges[-1]])
        ax.grid(which='minor')


# --- Input parameters ---
# FoV params
fov_ra_eor = args.fov_ra_eor * units.deg
fov_dec_eor = args.fov_dec_eor
fov_ra_fg = args.fov_ra_fg
fov_dec_fg = args.fov_dec_fg

# Frequency params
nf = args.nf
nu_min_MHz = args.nu_min_MHz * units.MHz
channel_width_MHz = args.channel_width_MHz * units.MHz
freqs = nu_min_MHz + np.arange(nf)*channel_width_MHz

# Beam params
beam_type = args.beam_type
if beam_type == "airy":
    antenna_diameter = args.antenna_diameter * units.m
elif beam_type == "gaussian":
    fwhm = args.fwhm_deg * units.deg

# uv sampling
uvs_m = load_inst_model(args.inst_model)[0][0, :, :2] * units.m


# --- Derived parameters ---
# FoV
if fov_dec_eor is None:
    fov_dec_eor = fov_ra_eor
else:
    fov_dec_eor = fov_dec_eor * units.deg
if fov_ra_fg is None:
    fov_ra_fg = fov_ra_eor
else:
    fov_ra_fg = fov_ra_fg * units.deg
if fov_dec_fg is None:
    fov_dec_fg = fov_ra_fg
else:
    fov_dec_fg = fov_dec_fg * units.deg

# Frequency
# The instrument appears largest in the uv plane in wavelengths
# at the highest frequency (smallest wavelength) in the data
wavelength = c.to('m/s') / freqs[-1].to('1/s')

# uv sampling
uvs = uvs_m / wavelength
uv_mags = np.sqrt(np.sum(uvs**2, axis=1))
u_max_inst_model = np.abs(uvs[:, 0]).max() / units.rad
v_max_inst_model = np.abs(uvs[:, 1]).max() / units.rad

# Beam params
beam_type = beam_type.lower()
if beam_type == "airy":
    aperture_width = antenna_diameter / wavelength / units.rad
elif beam_type in ["gauss", "gaussian"]:
    stddev = fwhm.to('rad') / 2.355
    stddev_uv = 1 / (2 * np.pi * stddev)
    aperture_width = stddev_uv * args.Nsigma
elif beam_type == "uniform":
    aperture_width = 0 / units.rad  # delta function aperture function


# --- BayesEoR model parameters ---
# Model uv plane
delta_u_eor = 1 / fov_ra_eor.to('rad')
delta_v_eor = 1 / fov_dec_eor.to('rad')
delta_u_fg = 1 / fov_ra_fg.to('rad')
delta_v_fg = 1 / fov_dec_fg.to('rad')

"""
The FoV along the RA and Dec axes set the separation between adjacent u and v
modes in the model uv plane, respectively.  We need to choose the number of
model uv plane grid points such that we Nyquist sample the image domain which
equates to having two image domain pixels per minimum fringe wavelength.  Here,
I only talk about u but the logic is identical for v.

The minimum fringe wavelength is `1 / u_max` where `u_max` is the maximum u
sampled by the instrument along the u axis.  We add a buffer of half the width
of the aperture function to this u_max, i.e.

u'_max = u_max + 0.5 * aperture_width

We thus need to choose u for the model uv plane which produces a fringe
wavelength which is smaller than the minimum fringe wavelength sampled by
the instrument, i.e. we need to solve

1 / u <= 1 / u'_max    ==>    u'_max <= u

Given that we have a rectilinear grid for the model uv plane and the spacing
between adjacent u is given by

delta_u = 1 / FoV_RA    (for delta_v we replace FoV_RA with FoV_Dec)

we must choose N_u such that

u'_max <= N_u * delta_u    ==>    N_u >= u'_max / delta_u
"""
Nu_eor = int(np.ceil((u_max_inst_model + 0.5*aperture_width) / delta_u_eor))
Nv_eor = int(np.ceil((v_max_inst_model + 0.5*aperture_width) / delta_v_eor))
Nu_fg = int(np.ceil((u_max_inst_model + 0.5*aperture_width) / delta_u_fg))
Nv_fg = int(np.ceil((v_max_inst_model + 0.5*aperture_width) / delta_v_fg))

# The calculation above determines the number of model uv plane pixels required
# for u > 0 (v > 0).  But, the model uv plane in BayesEoR is specified for all
# u (v) (positive and negative).  The model uv plane also requires an odd
# number of pixels along the u (v) axis for modelling the (u, v)=(0, 0)
# monopole.
Nu_eor = Nu_eor * 2 + 1
Nv_eor = Nv_eor * 2 + 1
Nu_fg = Nu_fg * 2 + 1
Nv_fg = Nv_fg * 2 + 1


# Sky model
"""
The sky model and model uv plane must satisfy Nyquist sampling to avoid any
spurious errors in the analysis.  In this case, Nyquist sampling requires at
least two image domain pixels per minimum fringe wavelength.  The minimum
fringe wavelength is the inverse of the maximum sampled |u|, i.e.

min_fringe_wavelength = 1 / |u| = 1 / sqrt(u^2 + v^2)

Given this wavelength, we then need to choose the Nside of the sky model
such that the pixel width, calculated as the square root of the pixel area,
is less than or equal to `min_fringe_wavelength / 2` or

2 * pixel_width(Nside) <= min_fringe_wavelength
"""
u_max_uv_model = 1 / units.rad * np.max((
    delta_u_eor.to('1/rad').value*(Nu_eor//2 - 1),
    delta_u_fg.to('1/rad').value*(Nu_fg//2 - 1)
))
v_max_uv_model = 1 / units.rad * np.max((
    delta_v_eor.to('1/rad').value*(Nv_eor//2 - 1),
    delta_v_fg.to('1/rad').value*(Nv_fg//2 - 1)
))
uv_max_uv_model = np.sqrt(u_max_uv_model**2 + v_max_uv_model**2)
min_fringe_wavelength = 1 / uv_max_uv_model

Nside = 16  # initial guess
while 2*ahp.nside_to_pixel_resolution(Nside).to('rad') > min_fringe_wavelength:
    Nside *= 2


rprint("\n", Panel("Configuration"))
print(f"{fov_ra_eor        = :f}")
print(f"{fov_dec_eor       = :f}")
print(f"{fov_ra_fg         = :f}")
print(f"{fov_dec_fg        = :f}")
print(f"{nf                = }")
print(f"{nu_min_MHz        = :f}")
print(f"{channel_width_MHz = :f}")
print(f"{beam_type         = }")
if beam_type == "airy":
    print(f"{antenna_diameter  = :f}")
elif beam_type == "gaussian":
    print(f"fwhm_deg          = {fwhm.to('deg'):f}")
    print(f"Nsigma            = {args.Nsigma:f}")
print(f"Instrument model  = {args.inst_model}", end="\n\n\n")

rprint(Panel("BayesEoR Model Parameters"))
print(f"{Nu_eor = }")
print(f"{Nv_eor = }")
print(f"{Nu_fg  = }")
print(f"{Nv_fg  = }")
print(f"{Nside  = }", end="\n\n")

if args.plot:
    fig, axs = plt.subplots(
        1, 2, figsize=(21, 10), sharey=False, gridspec_kw={'wspace': 0.1}
    )

    axs[0].set_title('EoR Model')
    axs[1].set_title('FG Model')

    for ax in axs:
        # Plot the uv sampling of the instrument
        ax.scatter(
            uvs[:, 0],
            uvs[:, 1],
            color='k',
            marker='o',
            label='UV Sampling'
        )

    # Plot aperture width as a circle around the max u (v)
    u_max_ind = np.where(uvs[:, 0] == u_max_inst_model.value)
    v_max_ind = np.where(uvs[:, 1] == v_max_inst_model.value)
    for ax in axs:
        circle_u = plt.Circle(
            *uvs[u_max_ind].value, aperture_width.value/2, ec='k', fc='none'
        )
        label = 'Aperture function width'
        if beam_type == "gaussian":
            label += fr' ($N_\sigma$ = {args.Nsigma})'
        circle_v = plt.Circle(
            *uvs[v_max_ind].value, aperture_width.value/2, ec='k', fc='none',
            label=label
        )
        ax.add_patch(circle_u)
        ax.add_patch(circle_v)

    plot_model_uv_grid(axs[0], Nu_eor, delta_u_eor, Nv_eor, delta_v_eor)
    plot_model_uv_grid(axs[1], Nu_fg, delta_u_fg, Nv_fg, delta_v_fg)

    xmin = np.min((axs[0].get_xlim()[0], axs[1].get_xlim()[0]))
    xmax = np.max((axs[0].get_xlim()[1], axs[1].get_xlim()[1]))
    ymin = np.min((axs[0].get_ylim()[0], axs[1].get_ylim()[0]))
    ymax = np.max((axs[0].get_ylim()[1], axs[1].get_ylim()[1]))
    for ax in axs:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.legend(loc='upper right')
        ax.set_xlabel(r'$u$ [$\lambda$]')
    axs[0].set_ylabel(r'$v$ [$\lambda$]')

    plt.show()