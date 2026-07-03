#!/usr/bin/env python3
"""Regenerate Figure 1 (FLARE benchmark architecture).

Matches the block diagram embedded in UAV_v15_final.docx but uses the
correct bibliography number for the Alexandridis 2008 reference (v16+:
[19]; the hand-drawn original in v15 still carried [18] from an earlier
renumbering).

Output: outputs/paper_figures/flare_architecture.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- layout constants ------------------------------------------------------
W, H = 13.0, 7.5  # figure inches
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 14,
})


def box(ax, x, y, w, h, edge, face, title=None, subtitle=None, fontsize=10):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.6, edgecolor=edge, facecolor=face, alpha=0.55,
    )
    ax.add_patch(patch)
    if title:
        ax.text(x + w / 2, y + h - 0.20, title, ha="center", va="top",
                fontsize=fontsize, fontweight="bold", color=edge)
    if subtitle:
        ax.text(x + w / 2, y + h / 2 - 0.18, subtitle,
                ha="center", va="center", fontsize=fontsize - 2, color="#333333")


def arrow(ax, xy_from, xy_to, colour="#333333", style="-"):
    arr = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.3, color=colour, linestyle=style,
    )
    ax.add_patch(arr)


def main() -> None:
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # palette
    BLUE = "#1f5fa8"; BLUE_F = "#d6e5f5"
    ORANGE = "#c75b3a"; ORANGE_F = "#fbe0d5"
    YELLOW = "#d4a429"; YELLOW_F = "#fdf0cc"
    RED = "#b23a2f"; RED_F = "#f5d4d0"
    GREEN = "#3a7c4f"; GREEN_F = "#d5e9d9"
    PURPLE = "#6b4c97"; PURPLE_F = "#e1d7ee"
    CYAN = "#2d7a8a"; CYAN_F = "#cfe6eb"

    # title
    ax.text(W / 2, H - 0.25, "FLARE Benchmark Architecture",
            ha="center", va="top", fontsize=16, fontweight="bold")

    # -- OSM Grid (top-left) --
    box(ax, 0.3, 5.55, 2.1, 0.95, BLUE, BLUE_F,
        title="OSM Grid", subtitle=r"$G = \{0,\dots,N\}^2$", fontsize=11)

    # -- 36 Contracts (top-right) --
    box(ax, 10.6, 5.55, 2.1, 0.95, GREEN, GREEN_F,
        title="36 Contracts", subtitle="DC, FC, EV, GC, FD, …", fontsize=11)

    # -- Coupled Hazard Dynamics container (top-centre) --
    cx, cy, cw, ch = 2.7, 5.0, 7.7, 1.7
    box(ax, cx, cy, cw, ch, ORANGE, ORANGE_F,
        title="Coupled Hazard Dynamics", subtitle=None, fontsize=12)
    # three sub-boxes
    subs = [
        (RED,    RED_F,    "Wind-Driven Fire",    "CA + Alexandridis [19]"),
        (YELLOW, YELLOW_F, "Structural Collapse", "Fire-triggered debris"),
        (CYAN,   CYAN_F,   "Dynamic Obstacles",   "Traffic + fire coupling"),
    ]
    sub_w = (cw - 0.4) / 3
    for i, (ed, fc, t, st) in enumerate(subs):
        sx = cx + 0.1 + i * sub_w
        box(ax, sx, cy + 0.10, sub_w - 0.05, 0.95, ed, fc,
            title=t, subtitle=st, fontsize=10)

    # dashed "triggers" / "closes roads"
    ax.text(cx + 1 * sub_w + 0.1, cy + 0.08, "triggers",
            fontsize=8, color="#777777", style="italic", ha="center")
    ax.text(cx + 2 * sub_w + 0.1, cy + 0.08, "closes roads",
            fontsize=8, color="#777777", style="italic", ha="center")

    # -- Risk Fusion / Cost Inflation (middle) --
    box(ax, 0.6, 3.45, 5.4, 0.95, PURPLE, PURPLE_F,
        title="Risk Fusion",
        subtitle=r"$R(x) = \max(R_f, R_s, R_t, R_b, R_{nfz})$", fontsize=11)
    box(ax, 7.0, 3.45, 5.4, 0.95, PURPLE, PURPLE_F,
        title="Cost Inflation",
        subtitle=r"$w(x) = 1 + \rho \cdot R(x)$", fontsize=11)

    # -- Planner Suite (single container with 5 sub-boxes) --
    pcx, pcy, pcw, pch = 0.6, 1.95, 11.8, 1.1
    box(ax, pcx, pcy, pcw, pch, "#4d4d4d", "#f0f0f0",
        title="Risk-Parameterised Planner Suite", subtitle=None, fontsize=11)
    planners = [
        (GREEN,  "#d5e9d9", r"A* ($\rho=0$)"),
        (BLUE,   "#d6e5f5", r"Periodic ($\rho=5.0$)"),
        (RED,    "#f5d4d0", r"Aggressive ($\rho=0.5$)"),
        (YELLOW, "#fdf0cc", r"Incr. A* ($\rho=2.0$)"),
        (PURPLE, "#e1d7ee", r"APF ($\rho=3.0$)"),
    ]
    plan_w = (pcw - 0.3) / 5
    for i, (ed, fc, lbl) in enumerate(planners):
        sx = pcx + 0.075 + i * plan_w
        box(ax, sx, pcy + 0.12, plan_w - 0.05, 0.50, ed, fc,
            title=None, subtitle=None)
        ax.text(sx + (plan_w - 0.05) / 2, pcy + 0.37, lbl,
                ha="center", va="center", fontsize=9, color=ed,
                fontweight="bold")

    # -- Bottom row: Runner | Mission Scoring | Evaluation Metrics --
    box(ax, 0.3, 0.55, 3.9, 1.0, BLUE, BLUE_F,
        title="Deterministic Runner",
        subtitle="Single RNG | Frozen mask | Step-indexed", fontsize=10)
    box(ax, 4.4, 0.55, 3.9, 1.0, RED, RED_F,
        title="Mission Scoring",
        subtitle=r"$M(t)$ strictly decreasing", fontsize=10)
    box(ax, 8.5, 0.55, 4.2, 1.0, GREEN, GREEN_F,
        title="Evaluation Metrics",
        subtitle="SR, MS, IQM, Friedman, Shapley", fontsize=10)

    # -- Ranking Inversion Detection (very bottom) --
    box(ax, 3.3, -0.55, 6.4, 0.80, YELLOW, YELLOW_F,
        title="Ranking Inversion Detection", subtitle=None, fontsize=11)

    # ---- arrows ----
    # OSM -> Hazards
    arrow(ax, (2.4, 6.02), (2.7, 6.02))
    # Hazards -> Risk Fusion
    arrow(ax, (cx + cw / 3, cy), (3.3, 4.40))
    # Risk Fusion -> Cost Inflation
    arrow(ax, (6.0, 3.92), (7.0, 3.92))
    # Cost Inflation -> Planner Suite
    arrow(ax, (9.7, 3.45), (9.7, 3.05))
    # Planner Suite -> Runner
    arrow(ax, (2.25, 1.95), (2.25, 1.55))
    # Runner -> Mission Scoring
    arrow(ax, (4.2, 1.05), (4.4, 1.05))
    # Mission -> Evaluation
    arrow(ax, (8.3, 1.05), (8.5, 1.05))
    # Evaluation -> Ranking Inversion
    arrow(ax, (10.6, 0.55), (8.5, 0.25))

    # Contracts dashed arrow
    arr = FancyArrowPatch(
        (11.65, 5.55), (11.65, 0.55),
        arrowstyle="-|>", mutation_scale=12,
        linewidth=1.0, color=GREEN, linestyle="--",
    )
    ax.add_patch(arr)
    ax.text(11.80, 3.1, "enforces", rotation=90, fontsize=8,
            color=GREEN, style="italic", ha="left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "flare_architecture.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "flare_architecture.pdf", bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'flare_architecture.png'}")


if __name__ == "__main__":
    main()
