"""Predict micro-pKa of organic molecules"""

# Add imports here
from ._version import get_versions
from .qupkake import *

__version__ = get_versions()["version"]

from . import _version
__version__ = _version.get_versions()['version']
