# UAVBench — Greece Government-Ready Mission Bank (9 Scenarios)

This folder contains the 9 canonical mission scenario YAMLs:
3 missions × 3 difficulty levels (easy / medium / hard).

## Missions

### Mission 1 — Civil Protection (ΓΓ Πολιτικής Προστασίας)
Wildfire Crisis Situational Awareness + Evacuation Corridor Monitoring

- `gov_civil_protection_easy.yaml` — 4 tasks, low injection, static dynamics
- `gov_civil_protection_medium.yaml` — 6 tasks, medium injection, moderate dynamics
- `gov_civil_protection_hard.yaml` — 8 tasks, high injection, severe dynamics + comms dropouts

### Mission 2 — Maritime Domain Awareness (ΛΣ-ΕΛΑΚΤ)
Coastal Search Corridor Patrol + Distress Event Injection

- `gov_maritime_domain_easy.yaml` — 4 waypoints, 1 event, static dynamics
- `gov_maritime_domain_medium.yaml` — 6 waypoints, 1–2 events, moderate dynamics
- `gov_maritime_domain_hard.yaml` — 8 waypoints, 2 events, severe dynamics + comms latency

### Mission 3 — Critical Infrastructure & Public Safety (ΥΠΕΘΑ/ISR-support)
Time-Critical Inspection Tour under Dynamic Restrictions

- `gov_critical_infrastructure_easy.yaml` — 4 sites, low restriction updates
- `gov_critical_infrastructure_medium.yaml` — 6 sites, medium updates, tighter windows
- `gov_critical_infrastructure_hard.yaml` — 8 sites, high updates + strict compliance

## Difficulty Knobs

All three difficulty levels share the SAME map; only these knobs change:

| Knob | Easy | Medium | Hard |
|------|------|--------|------|
| Number of tasks | 4 | 6 | 8 |
| Injection rate | low | medium | high |
| Dynamics intensity | static | moderate | severe |
| Time budget | relaxed | moderate | tight |
| Energy budget | 1.0 | 0.85 | 0.70 |
| Comms dropout prob | 0.0 | 0.05 | 0.15 |
| Comms latency (steps) | 0 | 2 | 4 |

## Usage

```python
from uavbench.missions import plan_mission, MissionID

result = plan_mission(
    start=(8, 8),
    heightmap=heightmap,
    no_fly=no_fly,
    mission_id=MissionID.CIVIL_PROTECTION,
    difficulty="easy",
    planner_id="astar",
    policy_id="greedy",
    seed=42,
)
print(result.metrics)
```

```bash
python -c "from uavbench.scenarios.registry import list_scenarios; print(list_scenarios())"
```
