"""
Unit and regression test for the qupkake package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import qupkake


def test_qupkake_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "qupkake" in sys.modules
