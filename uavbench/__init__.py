"""Compatibility package for src-layout development checkouts.

This makes `python -m uavbench...` work from the repository root without an
editable install by extending the package search path to include `src/uavbench`.
"""

from pathlib import Path
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]

_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "uavbench"
if _SRC_PACKAGE.is_dir():
    __path__.append(str(_SRC_PACKAGE))
