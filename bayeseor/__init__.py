from importlib.metadata import PackageNotFoundError, version

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # pragma: no cover
    try:
        __version__ = version("bayeseor")
    except PackageNotFoundError:
        # package is not installed
        __version__ = "unknown"

from . import matrices, model, params, posterior, utils
