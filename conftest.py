"""Root conftest - configure Python path for tests."""

import sys
from pathlib import Path

# Add src to Python path at import time (before pytest_configure)
_src_path = str(Path(__file__).parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)


def pytest_configure(config):
    """Ensure src is in path before collection."""
    src_path = str(Path(__file__).parent / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
