#!/usr/bin/env python3
"""Generate the risk profile LaTeX table (Task 1).

Extracts planner risk coefficients programmatically from source files
and generates a booktabs-formatted LaTeX table.

Output: outputs/paper_tables/risk_profile.tex
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "flare"


def extract_constant(filepath: Path, varname: str) -> float:
    """Parse Python source to get a numeric constant."""
    source = filepath.read_text()
    match = re.search(rf"^{varname}\s*=\s*(.+)", source, re.MULTILINE)
    if not match:
        print(f"ERROR: Could not find {varname} in {filepath}")
        sys.exit(1)
    return float(ast.literal_eval(match.group(1).strip()))


def main() -> None:
    # Extract coefficients from source
    alpha = extract_constant(SRC / "planners" / "periodic_replan.py", "_RISK_ALPHA")
    beta = extract_constant(SRC / "planners" / "aggressive_replan.py", "_RISK_BETA")
    gamma = extract_constant(SRC / "planners" / "incremental_astar.py", "_RISK_GAMMA")
    delta = extract_constant(SRC / "planners" / "apf.py", "_RISK_DELTA")
    fire_r = extract_constant(SRC / "blocking.py", "_FIRE_RISK_RADIUS")
    traffic_r = extract_constant(SRC / "blocking.py", "_TRAFFIC_RISK_RADIUS")

    print("Extracted risk coefficients:")
    print(f"  alpha  (Periodic)       = {alpha}")
    print(f"  beta   (Aggressive)     = {beta}")
    print(f"  gamma  (Incr. A*)       = {gamma}")
    print(f"  delta  (APF)            = {delta}")
    print(f"  fire_risk_radius        = {fire_r}")
    print(f"  traffic_risk_radius     = {traffic_r}")

    # Generate LaTeX table
    tex = rf"""\begin{{table}}[t]
\centering
\caption{{Planner risk profiles. Each planner applies a different
  coefficient to the shared risk cost map $r(c) \in [0,1]$ computed
  by \texttt{{compute\_risk\_cost\_map()}}.
  Higher coefficient $\Rightarrow$ wider detours around hazards.}}
\label{{tab:risk_profiles}}
\footnotesize
\begin{{tabular}}{{@{{}}llcll@{{}}}}
\toprule
Planner & Family & Coeff. & Cost formula & Replanning trigger \\
\midrule
A*             & Static   & ---       & $w(c) = 1$                        & Never \\
Periodic       & Adaptive & $\alpha{{=}}{alpha}$ & $w(c) = 1 + {alpha} \cdot r(c)$ & Every 6 steps \\
Aggressive     & Adaptive & $\beta{{=}}{beta}$  & $w(c) = 1 + {beta} \cdot r(c)$ & On obstacle change \\
Incr.\ A*      & Adaptive & $\gamma{{=}}{gamma}$ & $w(c) = 1 + {gamma} \cdot r(c)$ & On path blocked \\
APF            & Reactive & $\delta{{=}}{delta}$ & $F_{{\text{{rep}}}} \propto \delta \cdot r(c)$ & Every step (gradient) \\
\bottomrule
\end{{tabular}}
\vspace{{2pt}}
\begin{{flushleft}}
\scriptsize $r(c)$ combines fire proximity (falloff over {int(fire_r)} cells), smoke ($\times 0.5$),
traffic proximity ({int(traffic_r)} cells), dynamic NFZ, and structural debris (8 cells).
\end{{flushleft}}
\end{{table}}"""

    out_dir = ROOT / "outputs" / "paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "risk_profile.tex"
    out_path.write_text(tex)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
