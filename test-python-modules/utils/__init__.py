import sys

__version__ = "0.1.0"

if sys.version_info < (3, 10):
    raise RuntimeError(
        f"This package requires Python 3.10+. You are using Python {sys.version_info.major}.{sys.version_info.minor}"
    )
