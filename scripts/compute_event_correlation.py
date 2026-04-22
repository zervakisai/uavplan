"""Aggregate hazard-event-to-UAV-decision correlations from the episode CSV."""
import ast, json, sys
from pathlib import Path
import pandas as pd


def compute(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    out: dict = {}

    out["terminations"] = df["termination_reason"].value_counts().to_dict()
    out["fire_caught_total"] = int((df.termination_reason == "fire_caught").sum())
    out["debris_caught_total"] = int((df.termination_reason == "debris_caught").sum())

    fc = df[df.termination_reason == "fire_caught"]
    out["fire_caught_by_planner"] = fc["planner_id"].value_counts().to_dict()
    out["fire_caught_step_stats"] = {
        "median": int(fc["executed_steps"].median()) if len(fc) else None,
        "p25": int(fc["executed_steps"].quantile(0.25)) if len(fc) else None,
        "p75": int(fc["executed_steps"].quantile(0.75)) if len(fc) else None,
        "n": int(len(fc)),
    }

    rep = df.groupby("planner_id")["replans"].agg(["mean", "median", "max"]).round(2)
    out["replan_stats_by_planner"] = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in rep.to_dict(orient="index").items()}

    rrc: dict[str, int] = {}
    for row in df["reject_reason_counts"].dropna():
        s = str(row).strip()
        if not s or s == "{}":
            continue
        try:
            d = ast.literal_eval(s) if "'" in s else json.loads(s)
        except Exception:
            continue
        for k, v in d.items():
            rrc[k] = rrc.get(k, 0) + int(v)
    out["reject_reasons_total"] = rrc

    per_scen: dict[str, dict] = {}
    for (scen, plan), g in df.groupby(["scenario_id", "planner_id"]):
        per_scen.setdefault(scen, {})[plan] = {
            "sr": float(g["success"].mean()),
            "ms": float(g["mission_score"].mean()),
            "replans": float(g["replans"].mean()),
            "fire_caught": int((g.termination_reason == "fire_caught").sum()),
            "executed_steps_mean_success": float(g[g.success]["executed_steps"].mean()) if g["success"].any() else None,
        }
    out["per_scenario"] = per_scen

    return out


def main() -> None:
    csv_path = sys.argv[1]
    out_json = sys.argv[2] if len(sys.argv) > 2 else "outputs/v8_final/event_correlation.json"
    data = compute(csv_path)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(data, indent=2))
    summary = {k: v for k, v in data.items() if k != "per_scenario"}
    print(json.dumps(summary, indent=2))
    print(f"[wrote] {out_json}")


if __name__ == "__main__":
    main()
