"""Predict micro-pKa of organic molecules"""

# Add imports here
from ._version import get_versions
from .qupkake import *

__version__ = get_versions()["version"]

import os
import subprocess

try:
    check_package = subprocess.run(["conda", "list", "xtb"], stdout=subprocess.PIPE)
    check_package_decoded = check_package.stdout.decode("utf-8").split()
    assert "xtb" in check_package_decoded
    assert "6.4.1" in check_package_decoded

    XTB_LOCATION = os.environ.get("XTBPATH") or "xtb"
except AssertionError:
    XTB_LOCATION = os.environ.get("XTBPATH") or os.path.join(
        os.path.dirname(__file__), "xtb-641/bin/xtb"
    )
finally:
    if XTB_LOCATION != "xtb":
        if not os.path.exists(XTB_LOCATION):
            raise RuntimeError(f'xTB exectuable in: "{XTB_LOCATION}" does not exists.')
    else:
        raise RuntimeError(
            'Conda version of xTB is currently not supported.\nPlease compile it from source and export the path manually to "XTBPATH" environment variable.'
        )
