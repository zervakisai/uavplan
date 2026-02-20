"""Icon system for stakeholder visualization.

Provides resolution-independent icons rendered as matplotlib artists.
All icons are defined as path data and rendered via matplotlib.patches.PathPatch
— no external assets required.

Usage::

    from uavbench.visualization.icons import IconLibrary

    lib = IconLibrary(icon_size=24)
    artist = lib.stamp("uav", center=(50, 80), color="#0066FF", ax=ax, zorder=10)
"""

from uavbench.visualization.icons.library import IconLibrary, IconID

__all__ = ["IconLibrary", "IconID"]
