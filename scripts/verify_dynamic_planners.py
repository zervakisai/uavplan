#!/usr/bin/env python3
"""Diagnostic: verify all 6 planners handle dynamic scenarios correctly.

Checks for each planner on gov_civil_protection_hard:
  1. Episode runs without crash
  2. Exactly 2 forced interdictions fire (at steps ~12 and ~28)
  3. Feasibility guardrail keeps path feasible (no permanent unreachable)
  4. Replanning happens (either adaptive or harness-level)
"""
import sys, time
sys.path.insert(0, "src")

from uavbench.cli.benchmark import run_dynamic_episode
from uavbench.planners import PAPER_PLANNERS

SCENARIO = "gov_civil_protection_hard"
SEED = 42
HORIZON = 200  # enough for both interdictions at event_t1=60, event_t2=140
ADAPTIVE_PLANNERS = {"periodic_replan", "aggressive_replan", "incremental_dstar_lite"}

print(f"{'='*75}")
print(f"DYNAMIC PLANNER VERIFICATION — {SCENARIO} (seed={SEED}, horizon={HORIZON})")
print(f"{'='*75}\n")

all_ok = True
results = []

for pid in PAPER_PLANNERS:
    print(f"▶ {pid:30s} ... ", end="", flush=True)
    t0 = time.perf_counter()
    try:
        r = run_dynamic_episode(SCENARIO, pid, seed=SEED, episode_horizon_steps=HORIZON)
    except Exception as exc:
        print(f"💥 CRASH: {exc}")
        all_ok = False
        results.append({"planner": pid, "status": "CRASH", "error": str(exc)})
        continue
    elapsed = time.perf_counter() - t0

    # Extract key metrics from result dict
    n_interdictions = r.get("interdictions_triggered", 0)
    n_forced_replans = r.get("forced_replans_triggered", 0)
    total_replans = r.get("total_replans", 0)
    forced_blocks = r.get("forced_interdiction_blocks", 0)
    success = r.get("success", False)
    steps = r.get("episode_steps", 0)
    termination = r.get("termination_reason", "?")
    feasible = r.get("feasible_after_guardrail", True)
    replan_mode = r.get("replan_mode", "?")
    hit_rate = r.get("interdiction_hit_rate_reference", 0.0)

    planner_type = "adaptive" if pid in ADAPTIVE_PLANNERS else "static"

    checks = []
    # Check 1: ran without crash — already passed if we're here
    checks.append(("no_crash", True))
    # Check 2: exactly 2 interdictions fired
    ok2 = n_interdictions == 2
    checks.append(("2_interdictions", ok2))
    # Check 3: replanning happened
    ok3 = total_replans > 0
    checks.append(("replanning", ok3))
    # Check 4: feasibility maintained
    ok4 = feasible and termination != "unreachable"
    checks.append(("feasible", ok4))

    all_pass = all(c[1] for c in checks)
    status = "✅ PASS" if all_pass else "❌ FAIL"
    if not all_pass:
        all_ok = False
    fails = [c[0] for c in checks if not c[1]]
    fail_str = f"  FAILED: {', '.join(fails)}" if fails else ""

    print(f"{status}  ({elapsed:.1f}s) steps={steps} replans={total_replans} "
          f"interdictions={n_interdictions} mode={replan_mode} hit_rate={hit_rate:.1%}{fail_str}")

    results.append({
        "planner": pid,
        "type": planner_type,
        "status": "PASS" if all_pass else "FAIL",
        "steps": steps,
        "total_replans": total_replans,
        "n_interdictions": n_interdictions,
        "forced_blocks": forced_blocks,
        "success": success,
        "termination": termination,
        "replan_mode": replan_mode,
        "hit_rate": hit_rate,
        "feasible": feasible,
        "elapsed_s": round(elapsed, 2),
        "fails": fails,
    })

print(f"\n{'='*75}")
if all_ok:
    print("🎉 ALL 6 PLANNERS PASS — dynamic scenarios verified!")
else:
    print("⚠️  SOME PLANNERS FAILED — see details above")
print(f"{'='*75}\n")

# Summary table
print(f"{'Planner':<25} {'Type':<10} {'Mode':<15} {'Status':<6} {'Steps':>5} "
      f"{'Replans':>7} {'Interd.':>7} {'HitRate':>7} {'Feasible':>8} {'Time':>6}")
print("-" * 105)
for r in results:
    if r["status"] == "CRASH":
        print(f"{r['planner']:<25} {'?':<10} {'?':<15} {'CRASH':<6}")
        continue
    print(f"{r['planner']:<25} {r['type']:<10} {r['replan_mode']:<15} {r['status']:<6} "
          f"{r['steps']:>5} {r['total_replans']:>7} "
          f"{r['n_interdictions']:>7} {r['hit_rate']:>6.0%} "
          f"{'Y' if r['feasible'] else 'N':>8} {r['elapsed_s']:>5.1f}s")
