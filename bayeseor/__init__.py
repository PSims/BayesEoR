
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from setuptools_scm import get_version

try:
    # get accurate version for developer installs
    version_str = get_version(Path(__file__).parent.parent)
    __version__ = version_str

except (LookupError, ImportError):
    try:
        # Set the version automatically from the package details.
        __version__ = version("pyuvsim")
    except PackageNotFoundError:  # pragma: nocover
        # package is not installed
        pass

from . import matrices, model, params, posterior, utils
