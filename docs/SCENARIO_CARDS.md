# SCENARIO_CARDS.md — Incident-Grounded Scenario Catalogue (Selling Edition)

**Date:** 2025-02-20  
**Version:** 2.0 (integrity-hardened, realism knobs verified active)  
**Scope:** All 9 registered scenarios (3 missions × 3 difficulties)  
**Source:** `src/uavbench/scenarios/configs/*.yaml`  
**CLI:** `uavbench --mode mission --scenarios <id> --planners <id>`

---

## How to Read This Document

Each scenario card contains:

1. **Operational Brief** — The real-world context a reviewer or program officer would recognize
2. **Timeline Beats** — Second-by-second narrative of what happens during an episode
3. **Mechanics Mapping** — Which UAVBench subsystems drive each narrative element
4. **Parameter Table** — Every YAML field, verified against `scenarios/loader.py`
5. **Why It's Hard** — The specific planning challenge, in one sentence
6. **What Success Looks Like** — Observable outcome of a competent planner

---

## Mission 1 — Civil Protection: Wildfire Crisis SA + Evacuation Corridor

> *"At 14:32 on July 23, 2018, a fire ignited in Penteli, eastern Attica. By 17:00, 102 people were dead, making it the deadliest wildfire in Greek history. HCAA issued NOTAM A0412/18 establishing Temporary Flight Restrictions over the active front. Our scenario recreates the UAV mission that wasn't flown that day."*

**Operational context:** Γενική Γραμματεία Πολιτικής Προστασίας (GSCP)  
**OSM tile:** Penteli, Attica, Greece (500×500 grid, 0.35 building density)  
**Incident provenance:** 2018 Attica Wildfire (Penteli/Mati)

**References:**
- Lekkas et al., 2018, "The July 2018 Attica wildfires," *Nat. Hazards Earth Syst. Sci.*
- HCAA NOTAM A0412/18 (Penteli TFR)

---

### Card 1.1 — `gov_civil_protection_easy`

**Operational brief:** 48 hours post-fire. GSCP requests a routine damage-assessment flight over the burn perimeter. The UAV maps 4 survey points — 2 on the fire perimeter, 2 on evacuation corridors — to verify that routes are passable. No active threats; the fire is contained.

**Timeline beats:**

| T+ | Event | Mechanic |
|---|---|---|
| T+0 | UAV launches from staging area [50,50] | `fixed_start_xy` |
| T+0–T+180 | Navigate to 4 survey waypoints | MissionEngine task queue |
| T+100 | One low-priority checkpoint injected | `injection_rate: low` |
| T+180–T+350 | Complete survey, return toward [450,450] | `time_budget: 350` |

**Why it's hard:** It isn't — this is the calibration baseline. A planner that fails Easy has a bug.

**What success looks like:** 4/4 tasks completed, path length < 1.5× optimal, zero violations.

| Field | Value |
|---|---|
| Difficulty | EASY |
| Regime | Naturalistic |
| Paper Track | Static |
| Tasks | 4 (2 perimeter + 2 corridor) |
| Time / Energy Budget | 350 steps / 1.0 |
| Dynamics | None |
| Comms Dropout / Latency / GNSS σ | 0% / 0 / 0.0 |
| Forced Replans | 0 |

---

### Card 1.2 — `gov_civil_protection_medium`

**Operational brief:** Active wildfire response, Day 1. The fire front is moving northeast; GSCP has deployed ground crews along Marathonos Avenue. HCAA is issuing rolling TFRs as the perimeter shifts. The UAV must map 6 survey points while dodging fire cells and rerouting around traffic congestion from evacuating vehicles.

**Timeline beats:**

| T+ | Event | Mechanic |
|---|---|---|
| T+0 | Launch from [50,50]; initial plan to first waypoint | `fixed_start_xy`, `astar` / planner |
| T+30 | Fire cell ignites at [200,300]; 2 ignition points active | `fire_ignition_points: 2` |
| T+50 | Traffic congestion appears on evacuation corridors | `enable_traffic: true` |
| T+80 | Wind shift (medium) pushes fire east; path segment invalidated | `wind_speed: 0.4`, replan trigger |
| T+100 | **Forced replan** — new perimeter survey point injected | `force_replan_count: 1`, `injection_rate: medium` |
| T+120 | Comms dropout: planner works from 2-step-stale map | `comms_dropout_prob: 0.05`, `constraint_latency_steps: 2` |
| T+200 | Second injection: corridor checkpoint added | Mission engine |
| T+300 | Episode ends | `time_budget: 300` |

**Why it's hard:** The fire moves. The plan that was optimal at T+0 is invalid by T+80. The planner must replan mid-segment while a 5% comms dropout means it occasionally plans against stale data.

**What success looks like:** ≥5/6 tasks completed, ≤1 NFZ violation, total replans ≤ budget.

| Field | Value |
|---|---|
| Difficulty | MEDIUM |
| Regime | Naturalistic |
| Paper Track | Dynamic |
| Tasks | 6 (3 perimeter + 3 corridor) |
| Time / Energy Budget | 300 steps / 0.85 |
| Dynamics | Fire (blocking) + Traffic |
| Comms Dropout / Latency / GNSS σ | 5% / 2 steps / 1.0 |
| Wind | Medium (0.4) |
| Forced Replans | 1 |
| Incident | 2018 Attica Wildfire — TFRs over fire perimeters |

---

### Card 1.3 — `gov_civil_protection_hard`

**Operational brief:** The full 2018 Mati scenario. Five simultaneous ignition points. TFRs expanding over active perimeters. Eight emergency vehicles clogging corridors. High winds (0.7) driving the fire front south-southwest. The UAV has 8 survey points — 4 on a shifting fire perimeter, 4 on congested evacuation corridors — and only 250 steps to hit them all on 70% energy.

**Timeline beats:**

| T+ | Event | Mechanic |
|---|---|---|
| T+0 | Launch [50,50]. Initial plan computed. | `astar` / planner |
| T+20 | 5 fire ignition points activate simultaneously | `fire_ignition_points: 5`, `fire_blocks_movement: true` |
| T+30 | 4 TFR zones begin expanding (incident mode, r≤70) | `num_nfz_zones: 4`, `nfz_expansion_rate: 0.9` |
| T+40 | 8 emergency vehicles enter corridors; traffic blocks movement | `num_emergency_vehicles: 8`, `traffic_blocks_movement: true` |
| T+50 | **Event window opens** — first forced replan + injection | `event_t1: 50`, `force_replan_count: 2` |
| T+60 | Comms dropout (15%): 1-in-7 replans use 4-step-stale data | `comms_dropout_prob: 0.15`, `constraint_latency_steps: 4` |
| T+80 | GNSS noise: planner perceives position ±2 cells from truth | `gnss_noise_sigma: 2.0` |
| T+100 | Second event window — new perimeter point, NFZ expands | `event_t2: 130` |
| T+130 | Third injection: corridor checkpoint behind expanding NFZ | High injection rate |
| T+200 | Energy at ~70% capacity — must triage remaining tasks | `energy_budget: 0.70` |
| T+250 | Hard timeout | `time_budget: 250` |

**Why it's hard:** Cascading constraints. Fire blocks cells, TFRs expand, vehicles block corridors, comms drop 15% of updates, and the planner's GNSS fix is off by ±2 cells. The feasible space shrinks faster than most planners can replan. A planner that doesn't triage will attempt all 8 tasks and complete none.

**What success looks like:** ≥5/8 tasks completed with intelligent triage, 0 NFZ breaches, termination reason ≠ stuck.

| Field | Value |
|---|---|
| Difficulty | HARD |
| Regime | Stress Test |
| Paper Track | Dynamic |
| Tasks | 8 (4 perimeter + 4 corridor) |
| Time / Energy Budget | 250 steps / 0.70 |
| Dynamics | Fire (blocking) + Traffic (blocking) + NFZ (4 zones, incident-mode) |
| Comms Dropout / Latency / GNSS σ | 15% / 4 steps / 2.0 |
| Wind | High (0.7) |
| Fire Ignitions | 5 simultaneous |
| Emergency Vehicles | 8 |
| NFZ Zones | 4 (TFR, r≤70, expansion rate 0.9) |
| Forced Replans | 2 |
| Incident | 2018 Attica Wildfire (Penteli/Mati) — HCAA NOTAM A0412/18 |

---

## Mission 2 — Maritime Domain Awareness: Coastal Patrol + Distress Response

> *"On September 10, 2017, the tanker Agia Zoni II sank 1.5 NM off Salamina, spilling 2,500 tonnes of fuel oil into the Saronic Gulf. The Hellenic Coast Guard established a SAR box and port exclusion zone around Piraeus. Our scenario puts a patrol UAV into that operational picture."*

**Operational context:** Λιμενικό Σώμα – Ελληνική Ακτοφυλακή (ΛΣ-ΕΛΑΚΤ / Hellenic Coast Guard)  
**OSM tile:** Piraeus / Saronic Gulf (500×500 grid, 0.29 building density)  
**Incident provenance:** 2017 Agia Zoni II Oil Spill

**References:**
- REMPEC, 2017, "Agia Zoni II Pollution Report"
- HCG OPS-SAR Directive 2017-1192

---

### Card 2.1 — `gov_maritime_domain_easy`

**Operational brief:** Routine coastal surveillance. The UAV flies a 4-waypoint circular patrol corridor in calm conditions. One low-priority distress alert is injected mid-flight. No vessel traffic, no hazard zones.

**Timeline beats:**

| T+ | Event | Mechanic |
|---|---|---|
| T+0 | Launch [250,50], begin patrol circuit | `fixed_start_xy` |
| T+0–T+200 | Visit 4 waypoints in sequence | Task queue |
| T+150 | Low-priority distress alert injected | `injection_rate: low` |
| T+200–T+400 | Complete circuit, proceed to [250,450] | `time_budget: 400` |

**Why it's hard:** It isn't. Calibration baseline for maritime scenarios.

**What success looks like:** 4/4 waypoints visited, distress acknowledged, path efficient.

| Field | Value |
|---|---|
| Difficulty | EASY |
| Regime | Naturalistic |
| Paper Track | Static |
| Tasks | 4 patrol waypoints |
| Time / Energy Budget | 400 steps / 1.0 |
| Dynamics | None |
| Comms Dropout / Latency / GNSS σ | 0% / 0 / 0.0 |
| Maritime Current | [0.1, 0.0] |

---

### Card 2.2 — `gov_maritime_domain_medium`

**Operational brief:** Active maritime patrol during the Agia Zoni II aftermath. Vessel traffic transits the Saronic corridor. The UAV patrols 6 waypoints while a SAR coordination box establishes a temporary high-risk region. A distress event (SOS localization) is injected mid-patrol, requiring immediate diversion and 2-step loiter.

**Timeline beats:**

| T+ | Event | Mechanic |
|---|---|---|
| T+0 | Launch [250,50] | `fixed_start_xy` |
| T+30 | Vessel traffic appears on corridor | `enable_traffic: true` |
| T+60 | Maritime current [0.2, 0.05] affects drift calculations | `maritime_current_vec` |
| T+100 | **Event window opens** — SAR coordination box advisory | `event_t1: 100` |
| T+120 | Forced replan: distress event injected, divert to SOS site | `force_replan_count: 1` |
| T+140 | Comms dropout: planner uses 2-step-stale vessel positions | `comms_dropout_prob: 0.05` |
| T+200 | Second event window — hazard zone update | `event_t2: 200` |
| T+350 | Episode ends | `time_budget: 350` |

**Why it's hard:** The distress diversion breaks the optimal patrol sequence. The planner must decide: finish current leg or break off immediately? Stale vessel positions from comms dropout mean the safe corridor the planner sees may already be blocked.

**What success looks like:** ≥5/6 patrol points + distress localized, total energy ≤ 0.85.

| Field | Value |
|---|---|
| Difficulty | MEDIUM |
| Regime | Stress Test |
| Paper Track | Dynamic |
| Tasks | 6 waypoints + 1 distress event |
| Time / Energy Budget | 350 steps / 0.85 |
| Dynamics | Fire + Traffic |
| Comms Dropout / Latency / GNSS σ | 5% / 2 steps / 1.0 |
| Maritime Current | [0.2, 0.05] |
| Forced Replans | 1 |
| Incident | 2017 Agia Zoni II — SAR coordination box |

---

### Card 2.3 — `gov_maritime_domain_hard`

**Operational brief:** Full-scale Agia Zoni II response. The HCG SAR box is active and expanding. Port exclusion zone around Piraeus reroutes all vessel traffic, creating congestion. The UAV must patrol 8 waypoints while responding to two simultaneous distress events. A safety-zone policy update mid-mission expands the restricted area. Comms are 15% degraded; the planner's position fix is ±2 cells from truth.

**Timeline beats:**

| T+ | Event | Mechanic |
|---|---|---|
| T+0 | Launch [250,50]; plan first patrol leg | Planner |
| T+20 | Vessel traffic active, blocking movement | `traffic_blocks_movement: true` |
| T+30 | 4 fire ignition points (hazard zones) activate | `fire_ignition_points: 4` |
| T+40 | SAR box NFZ begins expanding (3 zones, r≤70) | `num_nfz_zones: 3`, `nfz_expansion_rate: 0.9` |
| T+60 | Maritime current intensifies [0.3, 0.1] | `maritime_current_vec` |
| T+80 | **Event T1**: first distress SOS — divert + loiter | `event_t1: 80`, `force_replan_count: 2` |
| T+100 | Comms dropout cascade: 15% of updates lost, 4-step-stale | `comms_dropout_prob: 0.15`, `constraint_latency_steps: 4` |
| T+120 | GNSS noise: position estimate ±2 cells | `gnss_noise_sigma: 2.0` |
| T+150 | Emergency corridor opens through NFZ | `emergency_corridor_enabled: true` |
| T+180 | **Event T2**: second distress SOS + NFZ expansion | `event_t2: 180` |
| T+230 | Energy critical (~70%) — must triage remaining patrol points | `energy_budget: 0.70` |
| T+280 | Hard timeout | `time_budget: 280` |

**Why it's hard:** Two competing objectives — patrol coverage and distress response — under an expanding NFZ that eats the patrol corridor. The planner that tries to do everything runs out of energy. The planner that focuses on distress abandons patrol coverage. The optimal strategy requires real-time utility-aware triage.

**What success looks like:** Both distress events localized, ≥5/8 patrol points, 0 SAR-box violations.

| Field | Value |
|---|---|
| Difficulty | HARD |
| Regime | Stress Test |
| Paper Track | Dynamic |
| Tasks | 8 waypoints + 2 distress events |
| Time / Energy Budget | 280 steps / 0.70 |
| Dynamics | Fire (blocking) + Traffic (blocking) + NFZ (3 SAR-box zones) |
| Comms Dropout / Latency / GNSS σ | 15% / 4 steps / 2.0 |
| Wind | High (0.6) |
| NFZ Zones | 3 (SAR box, r≤70, expansion 0.9) |
| Maritime Current | [0.3, 0.1] |
| Forced Replans | 2 |
| Incident | 2017 Agia Zoni II Oil Spill — HCG OPS-SAR Directive 2017-1192 |

---

## Mission 3 — Critical Infrastructure: Time-Critical Inspection Tour under Dynamic Restrictions

> *"On December 14, 2021, a bomb threat forced evacuation of Syntagma and Monastiraki metro stations in central Athens. ELAS established a 500-meter security cordon. Our scenario places a UAV inspection mission into the expanding cordon — where time windows close as the restricted zone grows."*

**Operational context:** ΥΠΕΘΑ / ISR-Support (Hellenic Ministry of Defence — ISR tasking)  
**OSM tile:** Downtown Athens (500×500 grid, 0.50 building density)  
**Incident provenance:** 2021 Athens Metro Bomb Threat (Syntagma)

**References:**
- ELAS Press Release 2021-12-14
- Athens Urban Transport Organisation (OASA) Disruption Report 2021-Q4

---

### Card 3.1 — `gov_critical_infrastructure_easy`

**Operational brief:** Routine infrastructure inspection tour. The UAV visits 4 sites in a pre-planned sequence, dwelling 3 steps at each for sensor data collection. Time windows are generous (120-step slack). One minor restriction zone appears mid-mission but doesn't materially affect routing.

**Timeline beats:**

| T+ | Event | Mechanic |
|---|---|---|
| T+0 | Launch [50,50]; plan route to first inspection site | `fixed_start_xy` |
| T+0–T+150 | Visit sites 1–3 with 3-step dwell at each | Task queue |
| T+100 | Minor restriction zone appears (cosmetic) | `injection_rate: low` |
| T+150–T+300 | Visit site 4, return toward [450,450] | `time_budget: 300` |

**Why it's hard:** It isn't. Calibration baseline.

**What success looks like:** 4/4 sites inspected within time windows, zero violations.

| Field | Value |
|---|---|
| Difficulty | EASY |
| Regime | Naturalistic |
| Paper Track | Static |
| Tasks | 4 inspection sites (3-step service each) |
| Time Windows | [earliest, earliest+120] per site |
| Time / Energy Budget | 300 steps / 1.0 |
| Dynamics | None |
| Comms Dropout / Latency / GNSS σ | 0% / 0 / 0.0 |

---

### Card 3.2 — `gov_critical_infrastructure_medium`

**Operational brief:** Security-sensitive inspection during the 2021 Syntagma bomb threat. Three dynamic security cordons appear at staggered intervals, expanding around the incident point [250,250]. The UAV must inspect 6 sites with tightened time windows (80-step slack) while reordering its sequence to avoid cordons.

**Timeline beats:**

| T+ | Event | Mechanic |
|---|---|---|
| T+0 | Launch [50,50]; plan optimal inspection sequence | Planner |
| T+30 | First security cordon activates at incident point | `num_nfz_zones: 2`, `restrictions_mode: incident` |
| T+60 | **Event T1**: second cordon + forced replan | `event_t1: 60`, `force_replan_count: 1` |
| T+80 | Comms dropout: stale NFZ boundary (2-step lag) | `comms_dropout_prob: 0.05`, `constraint_latency_steps: 2` |
| T+100 | Third restriction zone: site 4 now behind cordon | `injection_rate: medium` |
| T+120 | Time window for site 3 approaching deadline | `time_windows: [earliest, +80]` |
| T+140 | **Event T2**: cordon expands; reroute required | `event_t2: 140` |
| T+260 | Episode ends | `time_budget: 260` |

**Why it's hard:** The inspection sequence that was optimal at T+0 becomes infeasible at T+60 when a cordon blocks site 4. The planner must re-sequence on the fly — visiting site 5 before site 4 — while keeping all time windows satisfied. This is a dynamic TSPTW (Travelling Salesman with Time Windows).

**What success looks like:** ≥5/6 sites inspected within windows, 0 cordon violations.

| Field | Value |
|---|---|
| Difficulty | MEDIUM |
| Regime | Stress Test |
| Paper Track | Dynamic |
| Tasks | 6 inspection sites |
| Time Windows | [earliest, earliest+80] per site |
| Time / Energy Budget | 260 steps / 0.85 |
| Dynamics | Dynamic NFZ (security cordon, 2 zones, incident-mode) |
| Comms Dropout / Latency / GNSS σ | 5% / 2 steps / 1.0 |
| Forced Replans | 1 |
| Incident | 2021 Athens Metro Bomb Threat — security cordon |

---

### Card 3.3 — `gov_critical_infrastructure_hard`

**Operational brief:** Maximum-stress inspection under the full Syntagma bomb-threat scenario. Eight sites, 50-step time windows, five cascading security cordons, fire hazard overlays, and strict compliance mode — any NFZ breach instantly aborts the mission. The planner is solving a time-windowed orienteering problem where the constraint space contracts faster than it can replan, with 15% comms dropout and ±2-cell GNSS error.

**Timeline beats:**

| T+ | Event | Mechanic |
|---|---|---|
| T+0 | Launch [50,50]; compute inspection sequence | Planner |
| T+20 | 4 fire ignition points activate (hazard overlay) | `fire_ignition_points: 4`, `fire_blocks_movement: true` |
| T+30 | First 2 security cordons expand from [250,250] | `num_nfz_zones: 4`, `nfz_expansion_rate: 0.9` |
| T+40 | 6 emergency vehicles enter streets | `num_emergency_vehicles: 6` |
| T+50 | **Event T1**: third cordon + forced replan + injection | `event_t1: 50`, `force_replan_count: 2` |
| T+60 | **Strict compliance active**: any NFZ cell entered = ABORT | `strict_compliance: true` |
| T+70 | Comms dropout: 15% → planner sees 4-step-stale cordons | `comms_dropout_prob: 0.15`, `constraint_latency_steps: 4` |
| T+80 | GNSS error: planner thinks it's at [x±2, y±2] | `gnss_noise_sigma: 2.0` |
| T+90 | Site 3 time window about to close (50-step slack) | Tight time windows |
| T+100 | Fourth cordon: site 6 now inaccessible | High injection rate |
| T+120 | **Event T2**: fifth cordon expansion | `event_t2: 120` |
| T+160 | Energy at ~70% — remaining sites may be infeasible | `energy_budget: 0.70` |
| T+220 | Hard timeout | `time_budget: 220` |

**Why it's hard:** This is the hardest scenario in UAVBench. Five cordons cascade over 120 steps, each shrinking the feasible space. With 50-step time windows, a wrong ordering means missed deadlines even with perfect planning. Strict compliance means a single NFZ pixel entered — even due to GNSS error — kills the mission. The planner must solve a time-windowed orienteering problem under adversarial constraint injection with degraded sensing. Most planners will abort or timeout.

**What success looks like:** ≥5/8 sites inspected, 0 NFZ breaches (mission not aborted), intelligent triage of unreachable sites.

| Field | Value |
|---|---|
| Difficulty | HARD |
| Regime | Stress Test |
| Paper Track | Dynamic |
| Tasks | 8 inspection sites |
| Time Windows | [earliest, earliest+50] per site (tight) |
| Time / Energy Budget | 220 steps / 0.70 |
| Dynamics | Dynamic NFZ (4 cordon zones, incident-mode) + Fire (blocking) + Traffic (blocking) |
| Comms Dropout / Latency / GNSS σ | 15% / 4 steps / 2.0 |
| Strict Compliance | Yes (NFZ breach = mission abort) |
| Wind | High (0.6) |
| Fire Ignitions | 4 |
| Emergency Vehicles | 6 |
| NFZ Zones | 4 (cordon, r≤70, expansion rate 0.9) |
| Forced Replans | 2 |
| Incident | 2021 Athens Metro Bomb Threat — ELAS Press Release 2021-12-14 |

---

## Difficulty Scaling Summary

| Parameter | Easy | Medium | Hard |
|---|---|---|---|
| Tasks | 4 | 6 | 8 |
| Injection rate | Low | Medium | High |
| Dynamics intensity | Static | Moderate | Severe |
| Time budget | 300–400 | 250–350 | 200–280 |
| Energy budget | 1.0 | 0.85 | 0.70 |
| Comms dropout | 0% | 5% | 15% |
| Constraint latency | 0 steps | 2 steps | 4 steps |
| GNSS noise σ | 0.0 | 1.0 | 2.0 |
| Forced replans | 0 | 1 | 2 |
| Paper track | Static | Dynamic | Dynamic |

---

## Mechanics Mapping — What Drives Each Narrative Element

| Narrative Element | UAVBench Subsystem | Config Key |
|---|---|---|
| Fire front advancing | `dynamics/fire_spread.py` | `fire_ignition_points`, `wind_speed` |
| TFR / security cordon expanding | `dynamics/dynamic_nfz.py` | `num_nfz_zones`, `nfz_expansion_rate`, `nfz_max_radius` |
| Emergency vehicle congestion | `dynamics/traffic.py` | `num_emergency_vehicles`, `traffic_blocks_movement` |
| Vessel / vehicle rerouting | `updates/obstacles.py` | `enable_traffic` |
| Comms dropout (stale map) | `cli/benchmark.py` realism loop | `comms_dropout_prob` |
| Constraint update delay | `cli/benchmark.py` latency FIFO | `constraint_latency_steps` |
| GNSS position error | `cli/benchmark.py` noise injection | `gnss_noise_sigma` |
| Distress event / task injection | `missions/engine.py` | `injection_rate`, `force_replan_count` |
| Time-windowed inspection | `missions/spec.py` | `time_windows` (per-task) |
| Strict NFZ compliance | `missions/engine.py` | `strict_compliance` |
| Maritime currents | `envs/urban.py` | `maritime_current_vec` |
| Fire↔traffic causal coupling | `dynamics/interaction_engine.py` | Bidirectional: fire proximity → road closures |

---

## Incident Provenance Map

| Mission | Incident | Year | Authority | Key Document |
|---|---|---|---|---|
| M1 — Civil Protection | Attica Wildfire (Penteli/Mati) | 2018 | GSCP, HCAA | NOTAM A0412/18 |
| M2 — Maritime Domain | Agia Zoni II Oil Spill (Saronic Gulf) | 2017 | ΛΣ-ΕΛΑΚΤ (HCG) | OPS-SAR Directive 2017-1192 |
| M3 — Critical Infrastructure | Athens Metro Bomb Threat (Syntagma) | 2021 | ELAS, OASA | ELAS Press Release 2021-12-14 |

---

## CLI Quick-Start

```bash
# Run all 9 scenarios in mission mode with A* planner
uavbench --mode mission --track all --planners astar --trials 5

# Run only hard scenarios with incremental D* Lite
uavbench --mode mission --scenarios gov_civil_protection_hard,gov_maritime_domain_hard,gov_critical_infrastructure_hard \
         --planners incremental_dstar_lite --trials 10 --save-csv

# Compare planners on dynamic track
uavbench --mode mission --track dynamic --planners astar,periodic_replan,incremental_dstar_lite \
         --trials 5 --paper-protocol --save-json
```
