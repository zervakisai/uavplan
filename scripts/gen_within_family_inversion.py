#!/usr/bin/env python3
"""Within-family ranking-inversion figure (paper Figure 9).

Holds each family's algorithm and replan trigger fixed, sweeps rho, and asks
whether the rho that maximises navigation success (SR) is also the one that
maximises mission value (M). Panel (a) shows the decisive case (Incremental A*):
SR rises with rho while M falls. Panel (b) is a dumbbell of M at each family's
SR-optimal vs M-optimal rho, with the inversion region (rho*(SR) != rho*(M))
shaded. Output: outputs/paper_figures/within_family_inversion.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8.5,
    "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.6, "figure.dpi": 300,
})

LABEL = {"periodic_replan": "Periodic", "aggressive_replan": "Aggressive",
         "incremental_astar": "Incr. A*", "apf": "APF",
         "chance_constrained": "Chance-Constr.", "cvar_conservative": "CVaR",
         "risk_sensitive": "Risk-Sensitive", "risk_aware_rrt": "RRT*"}
C_SR, C_M = "#2c6fbb", "#c75b3a"
# A family inverts iff its navigation-optimal rho differs from its mission-optimal
# rho (matches Table 3's Inv. column and the caption "rho*(SR) != rho*(M)").


def feasible_sr(sub):
    f = sub[~sub.infeasible.astype(bool)]
    return float(f.success.astype(bool).mean()) if len(f) else np.nan


def mean_ms(sub):
    return float(sub.mission_score.mean()) if len(sub) else np.nan


def main():
    df = pd.read_csv(ROOT / "outputs/rho_sweep/rho_sweep.csv")
    df = pd.concat([df, pd.read_csv(ROOT / "outputs/rho_sweep/rho_sweep_new.csv")],
                   ignore_index=True)
    df = df[df.risk_rho.astype(str) != "none"].copy()
    df["rho"] = df.risk_rho.astype(float)
    rhos = sorted(df.rho.unique())

    curve, opt = {}, {}
    for pl in LABEL:
        d = df[df.planner_id == pl]
        if d.empty:
            continue
        sr = {r: feasible_sr(d[d.rho == r]) for r in rhos}
        ms = {r: mean_ms(d[d.rho == r]) for r in rhos}
        curve[pl] = ([sr[r] for r in rhos], [ms[r] for r in rhos])
        rsr, rm = max(sr, key=sr.get), max(ms, key=ms.get)
        opt[pl] = (rsr, rm, ms[rsr], ms[rm])

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.2),
                                   gridspec_kw={"width_ratios": [1.0, 1.15]})

    # ---- (a) flagship family: Incremental A* ----
    sr_i, m_i = curve["incremental_astar"]
    x = np.arange(len(rhos))
    ax2 = axa.twinx()
    l1, = axa.plot(x, np.array(sr_i) * 100, "-o", color=C_SR, lw=1.6, ms=4,
                   label="Navigation success SR")
    l2, = ax2.plot(x, m_i, "-s", color=C_M, lw=1.6, ms=4, label="Mission value M")
    axa.set_xticks(x)
    axa.set_xticklabels([f"{r:g}" for r in rhos])
    axa.set_xlabel("Risk coefficient ρ")
    axa.set_ylabel("Feasible success rate (%)", color=C_SR)
    ax2.set_ylabel("Mission value M", color=C_M)
    axa.tick_params(axis="y", labelcolor=C_SR)
    ax2.tick_params(axis="y", labelcolor=C_M)
    rsr, rm, m_at_srstar, m_at_mstar = opt["incremental_astar"]
    isr, im = rhos.index(rsr), rhos.index(rm)
    axa.axvline(isr, color=C_SR, ls=":", lw=1.0, alpha=0.7)
    ax2.axvline(im, color=C_M, ls=":", lw=1.0, alpha=0.7)
    axa.annotate(r"$\rho^\star$(SR)", (isr, np.array(sr_i).max() * 100), color=C_SR,
                 fontsize=6.5, ha="center", va="bottom")
    ax2.annotate(r"$\rho^\star$(M)", (im, max(m_i)), color=C_M, fontsize=6.5,
                 ha="left", va="bottom")
    axa.set_title("(a) Incremental A* family: the ρ that is best for\nnavigation is worst for mission",
                  loc="left", fontweight="bold", fontsize=8)
    axa.legend(handles=[l1, l2], loc="center right", framealpha=0.9)

    # ---- (b) dumbbell across all 8 families, with inversion band ----
    def is_inv(p):
        return opt[p][0] != opt[p][1]        # rho*(SR) != rho*(M)
    # inverting families first (top), each group ordered by M gap descending
    order = sorted(opt, key=lambda p: (not is_inv(p), -(opt[p][3] - opt[p][2])))
    n_inv = sum(1 for p in order if is_inv(p))
    ys = np.arange(len(order))

    # shaded inversion region: the top n_inv rows (largest M gaps) are the
    # families where rho*(SR) != rho*(M) -> a within-family ranking inversion.
    if n_inv:
        axb.axhspan(-0.5, n_inv - 0.5, color="#c75b3a", alpha=0.075, zorder=0)
        axb.axhline(n_inv - 0.5, color="#cfcfcf", lw=0.7, zorder=0)
        axb.text(0.315, (n_inv - 1) / 2.0, "ranking\ninversion\n" r"$\rho^\star$(SR)$\neq\rho^\star$(M)",
                 color="#c75b3a", fontsize=6.0, va="center", ha="left",
                 style="italic", alpha=0.95, zorder=1)
    if n_inv < len(order):
        axb.text(0.315, n_inv + (len(order) - n_inv - 1) / 2.0,
                 "optima\ncoincide", color="#8a8a8a", fontsize=6.0,
                 va="center", ha="left", style="italic", zorder=1)

    for y, pl in zip(ys, order):
        _, _, m_sr, m_m = opt[pl]
        inv = is_inv(pl)
        axb.plot([m_sr, m_m], [y, y], color=("#444" if inv else "#aaa"),
                 lw=1.5 if inv else 1.1, zorder=2)
        axb.scatter(m_sr, y, s=34, color="white", edgecolor=C_SR, linewidth=1.4, zorder=3)
        axb.scatter(m_m, y, s=34, color=C_M, edgecolor="white", linewidth=0.6, zorder=3)
    axb.set_yticks(ys)
    axb.set_yticklabels([LABEL[p] for p in order])
    axb.set_ylim(len(order) - 0.5, -0.5)  # inverted, room for band
    axb.set_xlabel("Mission value M")
    axb.set_xlim(0.30, 1.02)
    axb.set_title(r"(b) M at each family's navigation-optimal $\rho^\star$(SR)" "\n"
                  r"vs its mission-optimal $\rho^\star$(M)",
                  loc="left", fontweight="bold", fontsize=8)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor=C_SR, markeredgewidth=1.4, ms=6,
               label=r"M at $\rho^\star$(SR)  (navigation-optimal)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_M,
               markeredgecolor="white", ms=6, label=r"M at $\rho^\star$(M)  (mission-optimal)"),
    ]
    axb.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.02),
               framealpha=0.92)

    fig.tight_layout()
    out = ROOT / "outputs/paper_figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "within_family_inversion.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "within_family_inversion.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved within_family_inversion.{{png,pdf}}  (n_inv={n_inv}/{len(order)})")
    for pl in order:
        rsr, rm, m_sr, m_m = opt[pl]
        print(f"  {LABEL[pl]:15s} ρ⋆SR={rsr:g} ρ⋆M={rm:g}  M@SR*={m_sr:.2f} M@M*={m_m:.2f} gap={m_m-m_sr:+.2f}")


if __name__ == "__main__":
    main()
