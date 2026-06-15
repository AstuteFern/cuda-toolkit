"""Locate and load the compiled CUDA shared libraries via ctypes.

The ``.cu`` kernels export plain ``extern "C"`` entry points (``run_reduction``,
``run_autocorrelation``); they are not CPython extension modules. We therefore
load them as ordinary shared libraries with :mod:`ctypes` rather than importing
them. The library is shipped inside the package directory when CUDA was present
at build time; if it is missing (CPU-only install) loading raises ``OSError``
and the caller falls back to the NumPy implementation.
"""

import ctypes
import glob
import os

# Shared-library extensions, in preference order per platform.
_LIB_SUFFIXES = (".dll", ".pyd", ".so", ".dylib")


def load_library(package_dir, base_name):
    """Load the shared library ``base_name`` from ``package_dir``.

    Returns the loaded ``ctypes.CDLL``. Raises ``OSError`` if no matching
    library file is found or it cannot be loaded.
    """
    candidates = []
    for suffix in _LIB_SUFFIXES:
        candidates.append(os.path.join(package_dir, base_name + suffix))
    # Fall back to any file that starts with the base name (covers platform
    # tags some toolchains append, e.g. ``_reduction_cuda.cpython-310.so``).
    for suffix in _LIB_SUFFIXES:
        candidates.extend(glob.glob(os.path.join(package_dir, base_name + "*" + suffix)))

    for path in candidates:
        if os.path.exists(path):
            return ctypes.CDLL(path)

    raise OSError(
        "CUDA shared library '%s' not found in %s" % (base_name, package_dir)
    )
