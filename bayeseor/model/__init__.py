from importlib import import_module

__all__ = [
    "Healpix",
    "load_inst_model",
    "generate_k_cube_in_physical_coordinates",
    "mask_k_cube",
    "generate_k_cube_model_spherical_binning",
    "calc_mean_binned_k_vals",
    "generate_k_cube_model_cylindrical_binning",
    "generate_gaussian_noise",
]

_ATTR_TO_MODULE = {
    "Healpix": ".healpix",
    "load_inst_model": ".instrument",
    "generate_k_cube_in_physical_coordinates": ".k_cube",
    "mask_k_cube": ".k_cube",
    "generate_k_cube_model_spherical_binning": ".k_cube",
    "calc_mean_binned_k_vals": ".k_cube",
    "generate_k_cube_model_cylindrical_binning": ".k_cube",
    "generate_gaussian_noise": ".noise",
}


def __getattr__(name):
    if name not in _ATTR_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_ATTR_TO_MODULE[name], __name__)
    return getattr(module, name)


def __dir__():
    return sorted(list(globals().keys()) + __all__)
