# UAVBench v2 — Mission Story Cards

Each mission family has 3 difficulty variants (easy/medium/hard).
All missions share the same tile and objective structure; difficulty modifies
dynamic intensity, task count, time budget, and event timing.

---

## Mission Family 1: Civil Protection

### Identity

| Field | Value |
|-------|-------|
| Mission Type | `CIVIL_PROTECTION` |
| Registry Prefix | `gov_civil_protection_{easy,medium,hard}` |
| Agency (EN) | GSCP (General Secretariat for Civil Protection) |
| Agency (EL) | Γενική Γραμματεία Πολιτικής Προστασίας (ΓΓΠΠ) |
| Tile | `penteli` (Penteli, Attica) |
| Building Density | 0.18 |
| Start | `(50, 50)` |
| Goal | `(450, 450)` |

### Incident Provenance

**2018 Attica Wildfire (Penteli/Mati)**

Deadliest Greek wildfire; temporary flight restrictions (TFRs) were established
over active fire perimeters in eastern Attica by HCAA. The scenario models
UAV operations under evolving fire dynamics and airspace restrictions.

References: [CITE] Lekkas et al., 2018, NHESS; [CITE] HCAA NOTAM A0412/18

### Mission Objective

**Wildfire Crisis Situational Awareness + Evacuation Corridor Monitoring**

The UAV must survey fire perimeter points and evacuation corridor checkpoints,
delivering a thermal map and corridor status report. As fire spreads, new
perimeter points are injected and airspace restrictions expand.

| Field | Value |
|-------|-------|
| Objective Label | "Wildfire SA + Evacuation Corridor" |
| Deliverable | `fire_perimeter.geojson`, `corridor_status.csv`, `alert_timeline.csv` |
| Reason String | "Survey active fire perimeter and verify evacuation corridor viability" |

### Task Queue

| Category | Weight | Time Decay (lambda) | Service Time | Description |
|----------|--------|-------------------|--------------|-------------|
| `perimeter_point` | 1.2 | 0.025 | 0 (fly-through) | Fire perimeter waypoint for thermal imaging |
| `corridor_checkpoint` | 1.0 | 0.015 | 0 (fly-through) | Evacuation corridor status check |
| `injected_perimeter_point` | 1.5 | 0.04 | 0 (fly-through) | Dynamically injected as fire perimeter shifts |

### Injection Events (Dynamic Track)

| Event | Description | Trigger |
|-------|-------------|---------|
| Perimeter shift | Fire perimeter shift detected — new POI injected | Step >= event_t1 |
| TFR expansion | Temporary aviation restriction zone (smoke/wind) | Step >= event_t1 |
| Corridor blockage | Corridor blocked — traffic congestion spike | Step >= event_t2 |

### Difficulty Matrix

| Difficulty | Tasks | Dynamics | Fire Blocks | Traffic Blocks | Wind | Events | Time Budget | Energy |
|------------|-------|----------|-------------|----------------|------|--------|-------------|--------|
| Easy | 4 | Static | No | No | Low (0.2) | 0 | 350 | 1.0 |
| Medium | 6 | Moderate | No | No | Medium (0.4) | 2 (t1=80, t2=160) | 300 | 0.85 |
| Hard | 8 | Severe | Yes | Yes | High (0.7) | 2 (t1=60, t2=140) | 250 | 0.70 |

### Scoring Parameters

| Parameter | Value |
|-----------|-------|
| Utility decay (lambda) | 0.025 |
| Risk penalty weight | 0.35 |
| Violation penalty weight | 1.0 |
| Strict compliance | No |

### Visual Profile

| Field | Value |
|-------|-------|
| Accent color | `#FF6B35` (orange) |
| POI icon | FIRE |
| Hazard icon | ALERT |
| Task icon | CAMERA |

---

## Mission Family 2: Maritime Domain

### Identity

| Field | Value |
|-------|-------|
| Mission Type | `MARITIME_DOMAIN` |
| Registry Prefix | `gov_maritime_domain_{easy,medium,hard}` |
| Agency (EN) | HCG (Hellenic Coast Guard) |
| Agency (EL) | Λιμενικό Σώμα – Ελληνική Ακτοφυλακή (ΛΣ-ΕΛΑΚΤ) |
| Tile | `piraeus` (Piraeus port) |
| Building Density | 0.29 |
| Start | `(250, 50)` |
| Goal | `(250, 450)` |

### Incident Provenance

**2017 Agia Zoni II Oil Spill (Saronic Gulf)**

Tanker sinking off Salamina; SAR coordination box and port exclusion zone
established by Hellenic Coast Guard. The scenario models maritime patrol
operations under evolving hazard conditions and restricted zones.

References: [CITE] REMPEC, 2017, Agia Zoni II Pollution Report; [CITE] HCG OPS-SAR Directive 2017-1192

### Mission Objective

**Coastal Search Corridor Patrol + Distress Event Response**

The UAV patrols a coastal corridor and responds to distress events,
delivering corridor coverage data and event response reports. Hazard zones
expand as conditions evolve.

| Field | Value |
|-------|-------|
| Objective Label | "Coastal Patrol + Distress Response" |
| Deliverable | `corridor_coverage.csv`, `event_response.csv`, `exposure_report.csv` |
| Reason String | "Patrol coastal corridor and respond to maritime distress events" |

### Task Queue

| Category | Weight | Time Decay (lambda) | Service Time | Description |
|----------|--------|-------------------|--------------|-------------|
| `patrol_waypoint` | 1.0 | 0.01 | 0 (fly-through) | Circular corridor patrol checkpoint |
| `distress_event` | 3.0 | 0.06 | 2 steps | Emergency response — hover + confirm |

### Injection Events (Dynamic Track)

| Event | Description | Trigger |
|-------|-------------|---------|
| Distress signal | Distress/safety event detected — immediate response required | Step >= event_t1 |
| Weather alert | Weather/hazard alert — temporary high-risk region | Step >= event_t1 |
| Zone expansion | Restricted area expansion — safety zone policy update | Step >= event_t2 |

### Difficulty Matrix

| Difficulty | Tasks | Dynamics | Fire Blocks | Traffic Blocks | Wind | Events | Time Budget | Energy |
|------------|-------|----------|-------------|----------------|------|--------|-------------|--------|
| Easy | 4 | Static | No | No | Low (0.2) | 0 | 400 | 1.0 |
| Medium | 6 | Moderate | No | No | Medium (0.4) | 2 (t1=100, t2=200) | 350 | 0.85 |
| Hard | 8 | Severe | Yes | Yes | High (0.6) | 2 (t1=80, t2=180) | 280 | 0.70 |

### Scoring Parameters

| Parameter | Value |
|-----------|-------|
| Utility decay (lambda) | 0.01 |
| Risk penalty weight | 0.25 |
| Violation penalty weight | 1.2 |
| Patrol weight alpha | 0.6 (easy), 0.5 (medium/hard) |

### Visual Profile

| Field | Value |
|-------|-------|
| Accent color | `#0088CC` (blue) |
| POI icon | SHIP |
| Hazard icon | DISTRESS |
| Task icon | ANCHOR |

---

## Mission Family 3: Critical Infrastructure

### Identity

| Field | Value |
|-------|-------|
| Mission Type | `CRITICAL_INFRASTRUCTURE` |
| Registry Prefix | `gov_critical_infrastructure_{easy,medium,hard}` |
| Agency (EN) | MoD (Ministry of National Defence) |
| Agency (EL) | Υπουργείο Εθνικής Άμυνας (ΥΠΕΘΑ) |
| Tile | `downtown` (Athens center) |
| Building Density | 0.50 |
| Start | `(50, 50)` |
| Goal | `(450, 450)` |

### Incident Provenance

**2021 Athens Metro Bomb Threat (Syntagma)**

Security cordon established around Syntagma/Monastiraki metro stations following
bomb threat. The scenario models infrastructure inspection operations under
evolving security restrictions and degraded communications.

References: [CITE] ELAS Press Release 2021-12-14; [CITE] Athens Urban Transport Organisation Disruption Report 2021-Q4

### Mission Objective

**Time-Critical Inspection Tour under Dynamic Restrictions**

The UAV must inspect critical infrastructure sites under time pressure,
delivering an inspection log and compliance report. Security restrictions
expand dynamically, and communications may be degraded.

| Field | Value |
|-------|-------|
| Objective Label | "Infrastructure Inspection Tour" |
| Deliverable | `inspection_log.csv`, `compliance_report.csv`, `resilience_curve.csv` |
| Reason String | "Inspect critical infrastructure sites under evolving security restrictions" |

### Task Queue

| Category | Weight | Time Decay (lambda) | Service Time | Description |
|----------|--------|-------------------|--------------|-------------|
| `inspection_site` | 1.0 + 0.1*i | 0.03 | 3 steps | Site inspection — hover + sensor sweep |

### Injection Events (Dynamic Track)

| Event | Description | Trigger |
|-------|-------------|---------|
| Restriction expansion | Dynamic restriction — temporary closure / topology change | Step >= event_t1 |
| Comms degradation | Degraded comms pocket — delayed map updates | Step >= event_t2 |

### Difficulty Matrix

| Difficulty | Tasks | Dynamics | Fire Blocks | Traffic Blocks | Wind | Events | Time Budget | Energy |
|------------|-------|----------|-------------|----------------|------|--------|-------------|--------|
| Easy | 4 | Static | No | No | None (0.1) | 0 | 300 | 1.0 |
| Medium | 6 | Moderate | No | No | Medium (0.3) | 2 (t1=60, t2=140) | 260 | 0.85 |
| Hard | 8 | Severe | Yes | Yes | High (0.6) | 2 (t1=50, t2=120) | 220 | 0.70 |

### Scoring Parameters

| Parameter | Value |
|-----------|-------|
| Utility decay (lambda) | 0.03 |
| Risk penalty weight | 0.30 |
| Violation penalty weight | 1.5 |
| Strict compliance | Hard only |

### Visual Profile

| Field | Value |
|-------|-------|
| Accent color | `#00CC88` (green) |
| POI icon | BUILDING |
| Hazard icon | SHIELD |
| Task icon | INSPECTION |

---

## Restriction Zone Types (Cross-Mission)

All dynamic restriction zones use mission-specific visual vocabularies:

| Zone Type | Mission | Color | Hatch | Border |
|-----------|---------|-------|-------|--------|
| `tfr` (Temporary Flight Restriction) | Civil Protection | Orange `#FF8C00` | 45-degree | Dashed |
| `sar_box` (SAR coordination box) | Maritime Domain | Blue `#1E90FF` | Horizontal | Dashed |
| `port_exclusion` | Maritime Domain | Blue `#1E90FF` | Vertical | Solid |
| `security_cordon` | Critical Infrastructure | Purple `#9B59B6` | Diagonal-back | Solid |

---

## Difficulty Knobs (All Missions)

These knobs scale identically across mission families:

| Knob | Easy | Medium | Hard |
|------|------|--------|------|
| `comms_dropout_prob` | 0.0 | 0.05 | 0.15 |
| `comms_latency_steps` | 0 | 2 | 4 |
| `risk_update_cadence` | 8 | 5 | 3 |
| `force_replan_count` | 0 | 1 | 2 |
| `replan_every_steps` | — | 4 | 4 |
| `max_replans_per_episode` | — | 2000 | 2000 |

---

## MC Contract Mapping

| Contract | Card Section |
|----------|-------------|
| MC-1 | Objective POI + Reason String (each mission) |
| MC-2 | Task Queue: Service Time column |
| MC-3 | Visual Profile + Objective Label + Deliverable |
| MC-4 | Difficulty Matrix → termination conditions |
