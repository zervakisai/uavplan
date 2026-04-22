"""Generate mission-type-aware figures.

Produces three figures that make each UAV mission type visually self-evident:

1. mission_score_decay_3panel.{pdf,png}
     Three horizontal panels (pharma / urban rescue / fire surveillance).
     Each panel plots the mission's value-decay curve and overlays vertical
     lines at the measured mean `executed_steps` of successful Aggressive and
     Periodic episodes (from the benchmark CSV). Urban rescue adds three
     fire-coupled survival curves for d_fire in {1, 3, 8}.

2. mission_portfolio_3x3.{pdf,png}
     3 rows (missions) x 3 cols. Col 1 = t=0 schematic with mission-specific
     iconography. Col 2 = mid-mission schematic (fire advance + UAV progress).
     Col 3 = mean mission_score per planner (5 bars) from the CSV.

3. trajectory_3scenario_comparison.{pdf,png}
     Three panels (one per scenario). Each has 3 bars (A*, Aggressive,
     Periodic) with height = mean mission score; bars are labelled with
     success rate and mission score.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

try:
    import paper_style as ps

    ps.apply_style()
except Exception:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

FIGDIR = REPO / "paper" / "figures"
CSV = REPO / "outputs" / "paper_results" / "all_episodes.csv"

# ASCII-safe icon tags — the Unicode medical-cross / warning-sign / bullseye
# glyphs are missing from Times, so they would render as tofu boxes.
ICON_PHARMA = "[Rx]"
ICON_RESCUE = "[SOS]"
ICON_SURVEIL = "[Cam]"

MISSIONS = {
    "pharma_delivery": {
        "label": "Pharma delivery",
        "icon": ICON_PHARMA,
        "scenario": "osm_penteli_pharma_delivery_medium",
        "color": "#d95f02",
    },
    "urban_rescue": {
        "label": "Urban search & rescue",
        "icon": ICON_RESCUE,
        "scenario": "osm_piraeus_urban_rescue_medium",
        "color": "#1f78b4",
    },
    "fire_surveillance": {
        "label": "Fire surveillance",
        "icon": ICON_SURVEIL,
        "scenario": "osm_downtown_fire_surveillance_medium",
        "color": "#6a3d9a",
    },
}

T_MAX = 900

PLANNER_LABELS = {
    "astar": "A*",
    "periodic_replan": "Periodic",
    "aggressive_replan": "Aggressive",
    "incremental_astar": "Incr. A*",
    "apf": "APF",
}
PLANNER_ORDER = [
    "astar",
    "periodic_replan",
    "aggressive_replan",
    "incremental_astar",
    "apf",
]


def decay_pharma(t: np.ndarray) -> np.ndarray:
    """Pharma: quadratic decay E(t) = max(0, 1 - (t/T)^2)."""
    return np.maximum(0.0, 1.0 - (t / T_MAX) ** 2)


def decay_surveil(t: np.ndarray) -> np.ndarray:
    """Fire surveillance: linear decay F(t) = max(0, 1 - t/T)."""
    return np.maximum(0.0, 1.0 - t / T_MAX)


def decay_rescue(
    t: np.ndarray, d_fire: float = 1.0, lam0: float = 1.0 / 600.0, kappa: float = 5.0
) -> np.ndarray:
    """Triage survival: S(t)=exp(-lam_eff*t), lam_eff=lam0*(1+kappa/max(d_fire,1))."""
    lam = lam0 * (1.0 + kappa / max(d_fire, 1.0))
    return np.exp(-lam * t)


def measured_tbar(df: pd.DataFrame, scenario: str, planner: str) -> float | None:
    """Mean executed_steps of successful episodes for (scenario, planner)."""
    sub = df[
        (df.scenario_id == scenario) & (df.planner_id == planner) & df.success
    ]
    return float(sub["executed_steps"].mean()) if len(sub) else None


# ---------------------------------------------------------------------------
# Figure 1: 3-panel mission-score decay with measured delivery markers
# ---------------------------------------------------------------------------
def fig_decay_3panel(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    t = np.linspace(0, T_MAX, 400)

    curves = {
        "pharma_delivery": decay_pharma(t),
        "urban_rescue": decay_rescue(t, 1.0),
        "fire_surveillance": decay_surveil(t),
    }

    for ax, (mkey, meta) in zip(axes, MISSIONS.items()):
        ax.plot(t, curves[mkey], color=meta["color"], lw=2.0, label="M(t)")

        if mkey == "urban_rescue":
            for d in (3, 8):
                ax.plot(
                    t,
                    decay_rescue(t, d),
                    color=meta["color"],
                    alpha=0.35,
                    ls="--",
                    lw=1.2,
                    label=f"d_fire={d}",
                )

        # Measured t-bar markers for Aggressive and Periodic.
        # Place Aggressive annotation BELOW curve to the left of its t-bar,
        # Periodic annotation ABOVE curve to the right of its t-bar, so the
        # two labels never overlap (the t-bars themselves can be close).
        planner_styles = [
            ("aggressive_replan", "Aggressive", "-", "below-left"),
            ("periodic_replan", "Periodic", ":", "above-right"),
        ]
        for planner, name, ls, placement in planner_styles:
            tbar = measured_tbar(df, meta["scenario"], planner)
            if tbar is None:
                continue
            yb = float(np.interp(tbar, t, curves[mkey]))
            ax.axvline(tbar, color="k", ls=ls, lw=0.9, alpha=0.65)
            ax.plot([tbar], [yb], "o", color="k", ms=4)

            if placement == "below-left":
                xt = max(tbar - 170, 10)
                yt = max(0.05, yb - 0.28)
                ha = "left"
            else:  # above-right
                xt = min(tbar + 30, T_MAX - 170)
                yt = min(0.98, yb + 0.28)
                ha = "left"

            ax.annotate(
                f"{name}\n$\\bar t$={tbar:.0f}\nM={yb:.2f}",
                xy=(tbar, yb),
                xytext=(xt, yt),
                fontsize=7,
                ha=ha,
                va="center",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="white",
                    ec="gray",
                    lw=0.3,
                    alpha=0.9,
                ),
            )

        ax.set_xlim(0, T_MAX)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Time t (steps)")
        if ax is axes[0]:
            ax.set_ylabel("Mission value M(t)")
        ax.set_title(f"{meta['icon']} {meta['label']}")
        ax.grid(True, alpha=0.25)
        if mkey == "urban_rescue":
            ax.legend(loc="upper right", fontsize=6, frameon=False)

    fig.suptitle(
        "Mission-score decay with measured delivery times (Aggressive vs. Periodic)",
        y=1.02,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            FIGDIR / f"mission_score_decay_3panel.{ext}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: 3x3 mission portfolio (schematics + bar chart per mission)
# ---------------------------------------------------------------------------
def _draw_schematic_t0(ax, mkey: str, color: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(
        mpatches.Rectangle(
            (0.04, 0.04), 0.92, 0.92, fill=False, ec="gray", lw=0.8
        )
    )
    # Start square (green)
    ax.plot(0.1, 0.82, "s", color="green", ms=10)
    ax.text(0.1, 0.92, "START", fontsize=6, color="green", ha="center")

    if mkey == "pharma_delivery":
        # Pharmacy POI (marker "P"), then final delivery GOAL
        ax.add_patch(
            mpatches.Circle((0.5, 0.55), 0.08, fc="white", ec=color, lw=1.5)
        )
        ax.text(
            0.5, 0.55, "Rx", color=color, fontsize=10,
            ha="center", va="center", fontweight="bold",
        )
        ax.text(0.5, 0.42, "Pharmacy POI", fontsize=6, ha="center")
        ax.plot(0.88, 0.15, "*", color="gold", ms=16, mec="black", mew=0.4)
        ax.text(0.88, 0.05, "Patient", fontsize=6, ha="center")

    elif mkey == "urban_rescue":
        casualties = [(0.32, 0.6), (0.58, 0.45), (0.75, 0.22)]
        for i, (x, y) in enumerate(casualties):
            ax.plot(x, y, "o", color=color, ms=11, mec="black", mew=0.4)
            ax.text(x, y - 0.08, f"C{i + 1}", fontsize=6, ha="center")
        ax.plot(0.9, 0.85, "*", color="gold", ms=14, mec="black", mew=0.4)
        ax.text(0.9, 0.93, "Safe zone", fontsize=6, ha="center")

    else:  # fire_surveillance
        xs, ys = [0.22, 0.4, 0.6, 0.82], [0.3, 0.7, 0.3, 0.7]
        ax.plot(xs + [xs[0]], ys + [ys[0]], "--", color=color, lw=1.3, alpha=0.8)
        for x, y in zip(xs, ys):
            ax.plot(x, y, "^", color=color, ms=9, mec="black", mew=0.3)
        ax.text(0.5, 0.08, "perimeter patrol", fontsize=6, ha="center")

    # Fire seed (common to all scenes)
    ax.add_patch(mpatches.Circle((0.7, 0.72), 0.07, color="red", alpha=0.45))
    ax.text(0.7, 0.62, "fire", fontsize=6, ha="center", color="darkred")


def _draw_schematic_mid(ax, color: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(
        mpatches.Rectangle(
            (0.04, 0.04), 0.92, 0.92, fill=False, ec="gray", lw=0.8
        )
    )
    # Fire has expanded — core + buffer
    ax.add_patch(mpatches.Circle((0.7, 0.72), 0.19, color="red", alpha=0.40))
    ax.add_patch(mpatches.Circle((0.7, 0.72), 0.28, color="gray", alpha=0.22))
    ax.text(0.7, 0.72, "fire", fontsize=6, ha="center", va="center", color="white")
    # UAV trail
    trail_x = [0.1, 0.22, 0.35, 0.48, 0.52]
    trail_y = [0.82, 0.72, 0.62, 0.55, 0.52]
    ax.plot(trail_x, trail_y, "-", color=color, lw=2.2)
    ax.plot(trail_x[-1], trail_y[-1], "o", color=color, ms=9, mec="black", mew=0.4)
    ax.text(trail_x[-1] + 0.03, trail_y[-1], "UAV", fontsize=6)


def fig_portfolio(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(11.0, 9.4))

    # Precompute per-scenario y-lim for bar charts (mission score can exceed 1
    # for multi-POI scenarios like urban_rescue which sums across casualties).
    scenario_ylim = {}
    for mkey, meta in MISSIONS.items():
        sub = df[df.scenario_id == meta["scenario"]].groupby("planner_id")[
            "mission_score"
        ].mean()
        scenario_ylim[mkey] = max(1.0, float(sub.max()) * 1.25) if len(sub) else 1.0

    for row, (mkey, meta) in enumerate(MISSIONS.items()):
        ax0, ax1, ax2 = axes[row]

        _draw_schematic_t0(ax0, mkey, meta["color"])
        ax0.set_title(
            f"{meta['icon']} {meta['label']}\n$t=0$ setup", fontsize=9
        )

        _draw_schematic_mid(ax1, meta["color"])
        ax1.set_title("$t=T/2$ (fire advance + UAV progress)", fontsize=9)

        sub = (
            df[df.scenario_id == meta["scenario"]]
            .groupby("planner_id")["mission_score"]
            .mean()
        )
        bars_x = [p for p in PLANNER_ORDER if p in sub.index]
        bars_y = [float(sub[p]) for p in bars_x]
        colors = ["#999999" if p == "astar" else meta["color"] for p in bars_x]
        alphas = [0.7 if p == "astar" else 0.9 for p in bars_x]

        x_pos = np.arange(len(bars_x))
        for xi, yi, ci, ai in zip(x_pos, bars_y, colors, alphas):
            ax2.bar(xi, yi, color=ci, alpha=ai, edgecolor="black", lw=0.4)
            ax2.text(
                xi,
                yi + scenario_ylim[mkey] * 0.02,
                f"{yi:.2f}",
                ha="center",
                fontsize=6.5,
            )
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(
            [PLANNER_LABELS.get(p, p) for p in bars_x], fontsize=7
        )
        ax2.set_ylim(0, scenario_ylim[mkey])
        ax2.set_ylabel("Mean mission score" if row == 1 else "")
        ax2.set_title(f"Mission score per planner", fontsize=9)
        ax2.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "FLARE mission portfolio: three scenarios, three decay models, five planners",
        y=1.00,
        fontsize=11,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            FIGDIR / f"mission_portfolio_3x3.{ext}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: 3-scenario MS comparison with SR labels
# ---------------------------------------------------------------------------
def fig_trajectory_comparison(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
    planners = [
        ("astar", "A*\n($\\rho=0$)"),
        ("aggressive_replan", "Aggressive\n($\\rho=0.5$)"),
        ("periodic_replan", "Periodic\n($\\rho=5$)"),
    ]

    for ax, (mkey, meta) in zip(axes, MISSIONS.items()):
        sub = df[df.scenario_id == meta["scenario"]]
        xs_labels = []
        ms_values = []
        annot_labels = []
        for pid, lbl in planners:
            g = sub[sub.planner_id == pid]
            ms = float(g["mission_score"].mean()) if len(g) else 0.0
            sr = float(g["success"].mean()) if len(g) else 0.0
            xs_labels.append(lbl)
            ms_values.append(ms)
            annot_labels.append(f"SR={sr:.0%}\nMS={ms:.2f}")

        bar_colors = ["#999999", meta["color"], meta["color"]]
        alphas = [0.7, 0.95, 0.75]

        y_max = max(1.0, max(ms_values) * 1.35)

        for i, (lbl, y, c, a, note) in enumerate(
            zip(xs_labels, ms_values, bar_colors, alphas, annot_labels)
        ):
            ax.bar(i, y, color=c, alpha=a, edgecolor="black", lw=0.4)
            ax.text(i, y + y_max * 0.02, note, ha="center", fontsize=7)

        ax.set_xticks(range(len(xs_labels)))
        ax.set_xticklabels(xs_labels, fontsize=7)
        ax.set_ylim(0, y_max)
        if ax is axes[0]:
            ax.set_ylabel("Mean mission score")
        ax.set_title(f"{meta['icon']} {meta['label']}", fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Navigation success vs. mission value across three scenarios",
        y=1.02,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            FIGDIR / f"trajectory_3scenario_comparison.{ext}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(CSV)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig_decay_3panel(df)
    fig_portfolio(df)
    fig_trajectory_comparison(df)
    print(f"[done] wrote 3 figures to {FIGDIR}")


if __name__ == "__main__":
    main()
