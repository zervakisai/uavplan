"""Shared minimal style for all paper figures — best-paper aesthetic."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Muted, professional planner colors
PLANNER_COLORS = {
    "astar":             "#2c2c2c",  # charcoal
    "periodic_replan":   "#4a7c59",  # forest
    "aggressive_replan": "#c75b3a",  # terracotta
    "incremental_astar": "#7b6d8e",  # lavender
    "apf":               "#5b8fa8",  # steel
}

PLANNER_LABELS = {
    "astar":             "A*",
    "periodic_replan":   "Periodic",
    "aggressive_replan": "Aggressive",
    "incremental_astar": "Incr. A*",
    "apf":               "APF",
}

PLANNER_ORDER = [
    "astar", "periodic_replan", "aggressive_replan",
    "incremental_astar", "apf",
]


def apply_style() -> None:
    """Apply minimal IEEE figure style globally."""
    plt.rcParams.update({
        # Typography
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,

        # Clean lines
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.3,
        "grid.alpha": 0.2,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.5,

        # No chartjunk
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Figure
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save(fig: plt.Figure, name: str, d: str = "outputs/paper_figures") -> None:
    """Save figure as PDF + PNG."""
    Path(d).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{d}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{d}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {d}/{name}.{{pdf,png}}")
