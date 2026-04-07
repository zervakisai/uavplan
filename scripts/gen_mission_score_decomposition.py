#!/usr/bin/env python3
"""Mission score decomposition — minimal best-paper style.

Three vertically stacked subplots showing decay functions with
planner delivery markers.

Output: outputs/paper_figures/mission_score_decomposition.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from flare.visualization.labels import PLANNER_ORDER, PLANNER_SHORT

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import PLANNER_COLORS, apply_style, save

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "outputs" / "paper_results" / "all_episodes.csv"


def _get_delivery_steps() -> dict[str, float]:
    defaults = {
        "astar": float("inf"), "periodic_replan": 400,
        "aggressive_replan": 200, "incremental_astar": 350, "apf": 300,
    }
    if not CSV.exists():
        return defaults
    df = pd.read_csv(CSV)
    df = df[df["planner_id"] != "dstar_lite"]
    steps_col = next(
        (c for c in df.columns if "step" in c.lower() and "plan" not in c.lower()
         and "service" not in c.lower() and "fire" not in c.lower()),
        "executed_steps",
    )
    result = {}
    for pid in PLANNER_ORDER:
        succ = df[(df["planner_id"] == pid) & (df["success"] == True)]  # noqa: E712
        result[pid] = succ[steps_col].median() if len(succ) > 0 else defaults.get(pid, float("inf"))
    return result


def main() -> None:
    apply_style()

    t = np.arange(0, 801)
    delivery = _get_delivery_steps()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3.5, 4.8), sharex=True)
    fig.subplots_adjust(hspace=0.35)

    # --- Subplot 1: Pharma (quadratic) ---
    efficacy = np.maximum(0, 1.0 - (t / 800) ** 2)
    ax1.plot(t, efficacy, color="#333333", linewidth=1.0)
    ax1.fill_between(t, efficacy, alpha=0.04, color="#4a7c59")
    ax1.set_title("Pharma: insulin efficacy", fontsize=8)
    ax1.set_ylabel("Efficacy")

    # Planner markers — clean dots on curve, labels offset to avoid overlap
    finite = [(pid, delivery[pid]) for pid in PLANNER_ORDER
              if delivery[pid] != float("inf") and delivery[pid] <= 800]
    finite.sort(key=lambda x: x[1])

    # Pre-compute positions, then assign offsets to prevent overlap
    positions = []
    for pid, tp in finite:
        y = max(0, 1.0 - (tp / 800) ** 2)
        positions.append((pid, tp, y))

    # Assign label offsets: use fixed positions keyed by planner identity
    LABEL_OFFSETS = {
        "apf":               (-30, -14),
        "incremental_astar": (5,   14),
        "aggressive_replan": (30,  8),
        "periodic_replan":   (-5,  -18),
    }

    for pid, tp, y in positions:
        ax1.plot(tp, y, "o", color=PLANNER_COLORS[pid], markersize=4,
                 markeredgecolor="white", markeredgewidth=0.4, zorder=10)
        ox, oy = LABEL_OFFSETS.get(pid, (0, 10))
        ax1.annotate(
            f"{PLANNER_SHORT[pid]} {y:.2f}",
            (tp, y), textcoords="offset points", xytext=(ox, oy),
            fontsize=5, ha="center", color=PLANNER_COLORS[pid],
            arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.3),
        )

    # A* marker
    for pid in PLANNER_ORDER:
        if delivery[pid] == float("inf") or delivery[pid] > 800:
            ax1.text(780, 0.10, f"{PLANNER_SHORT[pid]}\n(never)", fontsize=5,
                     color=PLANNER_COLORS[pid], ha="right", style="italic")

    # Delta arrow — subtle
    if len(finite) >= 2:
        t_fast = finite[0][1]
        t_slow = finite[-1][1]
        y_fast = max(0, 1.0 - (t_fast / 800) ** 2)
        y_slow = max(0, 1.0 - (t_slow / 800) ** 2)
        ax1.annotate("", xy=(t_fast, 0.42), xytext=(t_slow, 0.42),
                     arrowprops=dict(arrowstyle="<->", color="#c75b3a", lw=0.8))
        ax1.text((t_fast + t_slow) / 2, 0.46,
                 rf"$\Delta$ eff. = {y_fast - y_slow:.2f}",
                 ha="center", fontsize=5.5, color="#c75b3a")

    # --- Subplot 2: Triage (exponential + fire coupling) ---
    kappa = 5.0
    lambda_base = 0.02
    for d_fire, ls, c, label in [
        (5,   "-",  "#c75b3a", r"$d_{\mathrm{fire}}=5$ (near fire)"),
        (20,  "--", "#E69F00", r"$d_{\mathrm{fire}}=20$"),
        (100, ":",  "#4a7c59", r"$d_{\mathrm{fire}}=100$ (safe)"),
    ]:
        lam = lambda_base * (1.0 + kappa / max(d_fire, 1.0))
        ax2.plot(t, np.exp(-lam * t), ls=ls, color=c, linewidth=0.9, label=label)

    ax2.set_title(
        r"Triage: survival $S(t) = e^{-\lambda_{\mathrm{eff}} \cdot t}$",
        fontsize=7)
    ax2.set_ylabel("Survival $S(t)$")
    ax2.legend(fontsize=5.5, loc="upper right", framealpha=0.8,
               edgecolor="none")

    # --- Subplot 3: Surveillance (linear) ---
    freshness = np.maximum(0, 1.0 - t / 800)
    ax3.plot(t, freshness, color="#333333", linewidth=1.0)
    ax3.fill_between(t, freshness, alpha=0.04, color="#5b8fa8")
    ax3.set_title("Surveillance: survey freshness", fontsize=8)
    ax3.set_xlabel("Delivery step")
    ax3.set_ylabel("Freshness")

    # Vertical planner lines on all subplots
    for ax in [ax1, ax2, ax3]:
        for pid in PLANNER_ORDER:
            tp = delivery[pid]
            if tp != float("inf") and tp <= 800:
                ax.axvline(tp, color=PLANNER_COLORS[pid], ls="--",
                           lw=0.5, alpha=0.5)
        ax.set_ylim(-0.02, 1.05)

    # Legend on subplot 3
    handles = []
    for pid in PLANNER_ORDER:
        tp = delivery[pid]
        if tp != float("inf") and tp <= 800:
            handles.append(Line2D([0], [0], color=PLANNER_COLORS[pid],
                                  ls="--", lw=0.5,
                                  label=f"{PLANNER_SHORT[pid]} (t={int(tp)})"))
        else:
            handles.append(Line2D([0], [0], color=PLANNER_COLORS[pid],
                                  ls="none", marker="x", markersize=4,
                                  label=f"{PLANNER_SHORT[pid]} (never)"))
    ax3.legend(handles=handles, fontsize=5, loc="upper right",
               framealpha=0.8, edgecolor="none")

    save(fig, "mission_score_decomposition")


if __name__ == "__main__":
    main()
