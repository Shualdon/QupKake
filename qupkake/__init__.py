"""Predict micro-pKa of organic molecules"""

# Add imports here
from . import _version
from .qupkake import *

print(_version.__file__)
# from ._version import __version__
__version__ = _version.get_versions()["version"]
