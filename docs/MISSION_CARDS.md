# UAVBench — Mission Story Cards

Each mission family has 3 difficulty variants (easy/medium/hard).
All missions share the same tile and objective structure; difficulty modifies
dynamic intensity, task count, time budget, and event timing.

---

## Mission Family 1: Fire Delivery

### Identity

| Field | Value |
|-------|-------|
| Mission Type | `FIRE_DELIVERY` |
| Registry Prefix | `gov_fire_delivery_{easy,medium,hard}` |
| Domain | `fire_delivery` |
| Inspired by | Evia/Εύβοια wildfires 2021 |
| Building Density | 0.18 (easy/medium), 0.22 (hard) |

### Incident Provenance

**2021 Evia Megafire (North Evia)**

The August 2021 Evia wildfire burned for over two weeks, becoming the largest
wildfire recorded in the EU. Settlements were isolated by rapidly advancing
fire fronts, and emergency supplies had to be delivered by sea and air to
cut-off villages. [CITE] EFFIS Annual Report 2021; [CITE] GSCP After-Action Report 2021-EV

### Mission Objective

**Emergency Medical Supply Delivery to Fire-Isolated Settlement**

The UAV must navigate through fire-affected terrain to deliver medical supplies
to an isolated village. Megafire dynamics with wind-biased spread create evolving
obstacles. Dynamic NFZs represent manned aircraft firefighting corridors.

| Field | Value |
|-------|-------|
| Objective Label | "Emergency Medical Supply Delivery" |
| Deliverable | `medical_supplies` |
| Reason String | "Emergency medical supply delivery to fire-isolated settlement" |
| Service Time | 0 (fly-through delivery at goal) |

### Task Queue

| Category | Weight | Time Decay | Service Time | Description |
|----------|--------|-----------|--------------|-------------|
| `delivery_point` | 1.0 | 0.02 | 0 (fly-through) | Delivery waypoint at isolated settlement |

### Difficulty Matrix

| Difficulty | Dynamics | Fire Ignitions | Wind | NFZ Zones | Forced Replans | Event Timing |
|------------|----------|---------------|------|-----------|---------------|-------------|
| Easy | Static | 0 | 0.2 | 0 | 0 | — |
| Medium | Moderate | 2 | 0.3 | 1 | 1 | t1=40, t2=120 |
| Hard | Severe | 4 | 0.5 | 2 | 2 | t1=30, t2=90 |

### Visual Profile

| Field | Value |
|-------|-------|
| Accent color | `#FF6B35` (orange) |
| POI icon | Medical cross |
| Fire rendering | Red-orange CA cells |
| HUD shows | Fire perimeter, delivery POI, distance_to_task, package status |

---

## Mission Family 2: Flood Rescue

### Identity

| Field | Value |
|-------|-------|
| Mission Type | `FLOOD_RESCUE` |
| Registry Prefix | `gov_flood_rescue_{easy,medium,hard}` |
| Domain | `flood_rescue` |
| Inspired by | Thessaly/Θεσσαλία floods 2023 |
| Building Density | 0.15 (easy/medium), 0.20 (hard) |

### Incident Provenance

**2023 Thessaly Floods (Storm Daniel)**

Storm Daniel caused catastrophic flooding across Thessaly in September 2023,
stranding thousands. Road infrastructure was destroyed, and aerial platforms
were critical for search and rescue assessment. [CITE] GSCP Storm Daniel AAR 2023;
[CITE] Copernicus EMS Activation EMSR686

### Mission Objective

**Search and Rescue Assessment of Flood-Stranded Population**

The UAV must reach stranded population areas through flood-affected terrain.
Water spread dynamics (modeled via traffic/road closures) create evolving
obstacles. The agent must perform on-site assessment (service_time hover).

| Field | Value |
|-------|-------|
| Objective Label | "Flood Search & Rescue Assessment" |
| Deliverable | `rescue_assessment` |
| Reason String | "Search and rescue assessment of flood-stranded population" |
| Service Time | 2 steps (hover assessment) |

### Task Queue

| Category | Weight | Time Decay | Service Time | Description |
|----------|--------|-----------|--------------|-------------|
| `rescue_site` | 1.0 | 0.02 | 2 steps | Rescue assessment — hover + confirm |

### Difficulty Matrix

| Difficulty | Dynamics | Fire Ignitions | Vehicles | Wind | NFZ Zones | Forced Replans | Event Timing |
|------------|----------|---------------|----------|------|-----------|---------------|-------------|
| Easy | Static | 0 | 0 | 0.1 | 0 | 0 | — |
| Medium | Moderate | 2 | 3 | 0.2 | 1 | 1 | t1=40, t2=120 |
| Hard | Severe | 3 | 5 | 0.4 | 2 | 2 | t1=30, t2=90 |

### Visual Profile

| Field | Value |
|-------|-------|
| Accent color | `#0088CC` (blue) |
| POI icon | Rescue marker |
| Flood rendering | Blue-tinted traffic zones (road closures as flood proxy) |
| HUD shows | Flood zones, rescue POI, distance_to_task, assessment progress |

---

## Mission Family 3: Fire Surveillance

### Identity

| Field | Value |
|-------|-------|
| Mission Type | `FIRE_SURVEILLANCE` |
| Registry Prefix | `gov_fire_surveillance_{easy,medium,hard}` |
| Domain | `fire_surveillance` |
| Inspired by | Evros/Έβρος megafire 2023 |
| Building Density | 0.16 (easy/medium), 0.22 (hard) |

### Incident Provenance

**2023 Evros Megafire (Alexandroupolis Region)**

The August 2023 Evros fire was the largest single wildfire in EU history,
burning over 96,000 hectares. Manned firefighting aircraft established
complex NFZ corridors, and real-time perimeter mapping was critical for
command post operations. [CITE] EFFIS Special Report Evros 2023;
[CITE] HCAA NOTAM Series A2023-EV

### Mission Objective

**Aerial Survey of Active Fire Perimeter for Command Post**

The UAV must survey fire perimeter points while avoiding active NFZ corridors
(manned aircraft). Fast-spreading fire dynamics and multiple NFZs create
a challenging environment requiring service_time hover at survey points.

| Field | Value |
|-------|-------|
| Objective Label | "Aerial Fire Perimeter Survey" |
| Deliverable | `perimeter_report` |
| Reason String | "Aerial survey of active fire perimeter for command post" |
| Service Time | 3 steps (hover survey) |

### Task Queue

| Category | Weight | Time Decay | Service Time | Description |
|----------|--------|-----------|--------------|-------------|
| `survey_point` | 1.0 | 0.02 | 3 steps | Perimeter survey — hover + sensor sweep |

### Difficulty Matrix

| Difficulty | Dynamics | Fire Ignitions | Wind | NFZ Zones | Forced Replans | Event Timing |
|------------|----------|---------------|------|-----------|---------------|-------------|
| Easy | Static | 0 | 0.2 | 0 | 0 | — |
| Medium | Moderate | 3 | 0.4 | 2 | 1 | t1=40, t2=120 |
| Hard | Severe | 5 | 0.6 | 3 | 2 | t1=25, t2=80 |

### Visual Profile

| Field | Value |
|-------|-------|
| Accent color | `#00CC88` (green) |
| POI icon | Survey camera |
| Fire rendering | Red-orange CA cells + NFZ corridor overlays |
| HUD shows | Fire front, NFZ corridors, survey POI, coverage % |

---

## Restriction Zone Types (Cross-Mission)

| Zone Type | Mission | Color | Description |
|-----------|---------|-------|-------------|
| `firefighting_corridor` | Fire Delivery | Orange `#FF8C00` | Manned aircraft corridor |
| `flood_exclusion` | Flood Rescue | Blue `#1E90FF` | Flooded area / road closure |
| `aircraft_corridor` | Fire Surveillance | Purple `#9B59B6` | Active firefighting NFZ |

---

## Difficulty Knobs (All Missions)

These knobs scale identically across mission families:

| Knob | Easy | Medium | Hard |
|------|------|--------|------|
| `enable_fire` | false | true | true |
| `enable_traffic` | false | true | true |
| `enable_dynamic_nfz` | false | true | true |
| `fire_blocks_movement` | false | true | true |
| `traffic_blocks_movement` | false | true | true |
| `force_replan_count` | 0 | 1 | 2 |
| `replan_every_steps` | 6 | 6 | 6 |

---

## MC Contract Mapping

| Contract | Card Section |
|----------|-------------|
| MC-1 | Objective POI + Reason String (each mission) |
| MC-2 | Task Queue: Service Time column |
| MC-3 | Visual Profile + Objective Label + Deliverable |
| MC-4 | Difficulty Matrix -> termination conditions |
