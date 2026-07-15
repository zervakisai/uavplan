#!/usr/bin/env python3
"""FLARE architecture diagram (paper Figure 1).

The planner block shows the "Risk-Parameterised Planner Families": 8 planners
grouped into 6 risk-handling paradigms, with rho swept within each family
(algorithm fixed) and A* as the static rho=0 reference.
Output: outputs/paper_figures/flare_architecture.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 13.0, 7.5
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.titlesize": 14})


def box(ax, x, y, w, h, edge, face, title=None, subtitle=None, fontsize=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.6, edgecolor=edge, facecolor=face, alpha=0.55))
    if title:
        ax.text(x + w / 2, y + h - 0.20, title, ha="center", va="top",
                fontsize=fontsize, fontweight="bold", color=edge)
    if subtitle:
        ax.text(x + w / 2, y + h / 2 - 0.18, subtitle, ha="center", va="center",
                fontsize=fontsize - 2, color="#333333")


def arrow(ax, xy_from, xy_to, colour="#333333", style="-"):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>",
        mutation_scale=14, linewidth=1.3, color=colour, linestyle=style))


def main() -> None:
    fig, ax = plt.subplots(figsize=(W, H))
    # ylim extends below 0 so the bottom "Within-Family Ranking Inversion" box
    # (at y=-0.55) is inside the axes and not clipped at the boundary.
    ax.set_xlim(0, W); ax.set_ylim(-0.85, H); ax.axis("off")

    BLUE = "#1f5fa8"; BLUE_F = "#d6e5f5"
    ORANGE = "#c75b3a"; ORANGE_F = "#fbe0d5"
    YELLOW = "#d4a429"; YELLOW_F = "#fdf0cc"
    RED = "#b23a2f"; RED_F = "#f5d4d0"
    GREEN = "#3a7c4f"; GREEN_F = "#d5e9d9"
    PURPLE = "#6b4c97"; PURPLE_F = "#e1d7ee"
    CYAN = "#2d7a8a"; CYAN_F = "#cfe6eb"

    ax.text(W / 2, H - 0.25, "FLARE Benchmark Architecture",
            ha="center", va="top", fontsize=16, fontweight="bold")

    box(ax, 0.3, 5.55, 2.1, 0.95, BLUE, BLUE_F,
        title="OSM Grid", subtitle=r"$G = \{0,\dots,N\}^2$", fontsize=11)
    box(ax, 10.6, 5.55, 2.1, 0.95, GREEN, GREEN_F,
        title="36 Contracts", subtitle="DC, FC, EV, GC, FD, …", fontsize=11)

    cx, cy, cw, ch = 2.7, 5.0, 7.7, 1.7
    box(ax, cx, cy, cw, ch, ORANGE, ORANGE_F, title="Coupled Hazard Dynamics",
        subtitle=None, fontsize=12)
    subs = [(RED, RED_F, "Wind-Driven Fire", "CA + Alexandridis [19]"),
            (YELLOW, YELLOW_F, "Structural Collapse", "Fire-triggered debris"),
            (CYAN, CYAN_F, "Dynamic Obstacles", "Traffic + fire coupling")]
    sub_w = (cw - 0.4) / 3
    for i, (ed, fc, t, st) in enumerate(subs):
        box(ax, cx + 0.1 + i * sub_w, cy + 0.10, sub_w - 0.05, 0.95, ed, fc,
            title=t, subtitle=st, fontsize=10)
    ax.text(cx + 1 * sub_w + 0.1, cy + 0.08, "triggers", fontsize=8,
            color="#777777", style="italic", ha="center")
    ax.text(cx + 2 * sub_w + 0.1, cy + 0.08, "closes roads", fontsize=8,
            color="#777777", style="italic", ha="center")

    box(ax, 0.6, 3.45, 5.4, 0.95, PURPLE, PURPLE_F, title="Risk Fusion",
        subtitle=r"$R(x) = \max(R_f, R_s, R_t, R_b, R_{nfz})$", fontsize=11)
    box(ax, 7.0, 3.45, 5.4, 0.95, PURPLE, PURPLE_F, title="Cost Inflation",
        subtitle=r"$w(x) = 1 + \rho \cdot R(x)$", fontsize=11)

    # -- Planner FAMILIES (rho swept within each) --
    pcx, pcy, pcw, pch = 0.6, 1.85, 11.8, 1.45
    box(ax, pcx, pcy, pcw, pch, "#4d4d4d", "#f0f0f0",
        title="Risk-Parameterised Planner Families", subtitle=None, fontsize=11)
    ax.text(pcx + pcw / 2, pcy + pch - 0.44,
            r"$\rho \in \{0,1,2,5,10\}$ swept within each family (algorithm fixed)"
            r"   ·   A* static reference (risk-blind), excluded from sweep",
            ha="center", va="center", fontsize=8.5, color="#333333", style="italic")
    families = [
        ("#4a7c59", "#dbe8df", "Soft cost-inflation", "Periodic · Aggressive\nIncr. A*"),
        ("#a03d5f", "#eddbe2", "Worst-case / CVaR", "CVaR"),
        ("#d99b30", "#f7ecd2", "Exponential", "Risk-Sensitive"),
        ("#3a7d7b", "#d5e6e5", "Sampling", "RRT*"),
        ("#b08968", "#ece2d8", "Hard threshold", "Chance-Constr."),
        ("#5b8fa8", "#dbe7ee", "Potential field", "APF"),
    ]
    fam_w = (pcw - 0.4) / 6
    for i, (ed, fc, par, pls) in enumerate(families):
        sx = pcx + 0.09 + i * fam_w
        box(ax, sx, pcy + 0.12, fam_w - 0.06, 0.66, ed, fc, title=None, subtitle=None)
        cxx = sx + (fam_w - 0.06) / 2
        ax.text(cxx, pcy + 0.60, par, ha="center", va="center",
                fontsize=7.6, color=ed, fontweight="bold")
        ax.text(cxx, pcy + 0.30, pls, ha="center", va="center",
                fontsize=6.5, color="#333333", linespacing=0.95)

    box(ax, 0.3, 0.55, 3.9, 1.0, BLUE, BLUE_F, title="Deterministic Runner",
        subtitle="Single RNG | Frozen mask | Step-indexed", fontsize=10)
    box(ax, 4.4, 0.55, 3.9, 1.0, RED, RED_F, title="Mission Scoring",
        subtitle=r"$M(t)$ strictly decreasing", fontsize=10)
    box(ax, 8.5, 0.55, 4.2, 1.0, GREEN, GREEN_F, title="Evaluation Metrics",
        subtitle="SR, MS, IQM, Friedman, Shapley", fontsize=10)

    box(ax, 3.3, -0.55, 6.4, 0.80, YELLOW, YELLOW_F,
        title="Within-Family Ranking Inversion", subtitle=None, fontsize=11)

    # arrows
    arrow(ax, (2.4, 6.02), (2.7, 6.02))
    arrow(ax, (cx + cw / 3, cy), (3.3, 4.40))
    arrow(ax, (6.0, 3.92), (7.0, 3.92))
    arrow(ax, (9.7, 3.45), (9.7, 3.30))
    arrow(ax, (2.25, 1.85), (2.25, 1.55))
    arrow(ax, (4.2, 1.05), (4.4, 1.05))
    arrow(ax, (8.3, 1.05), (8.5, 1.05))
    arrow(ax, (10.6, 0.55), (8.5, 0.25))
    ax.add_patch(FancyArrowPatch((11.65, 5.55), (11.65, 0.55), arrowstyle="-|>",
        mutation_scale=12, linewidth=1.0, color=GREEN, linestyle="--"))
    ax.text(11.80, 3.1, "enforces", rotation=90, fontsize=8, color=GREEN,
            style="italic", ha="left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "flare_architecture.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "flare_architecture.pdf", bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'flare_architecture.png'}")


if __name__ == "__main__":
    main()
