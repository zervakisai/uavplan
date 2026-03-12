# UAVBench — Mission Story Cards

Each mission family has a single OSM-based scenario (medium difficulty).
All missions use real Greek urban map tiles with dynamic obstacles.

---

## Mission Family 1: Fire Delivery

### Identity

| Field | Value |
|-------|-------|
| Mission Type | `PHARMA_DELIVERY` |
| Scenario ID | `osm_penteli_pharma_delivery_medium` |
| Domain | `pharma_delivery` |
| OSM Tile | Penteli, Attica |
| Inspired by | Evia/Εύβοια wildfires 2021 |
| Building Density | 0.18 |

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

### Scenario Parameters

| Field | Value |
|-------|-------|
| Dynamics | Moderate (dynamic track) |
| Fire Ignitions | 3 |
| NFZ Zones | 1 |
| Forced Replans | 1 |
| Event Timing | t1=40, t2=120 |

### Visual Profile

| Field | Value |
|-------|-------|
| Accent color | `#FF6B35` (orange) |
| POI icon | Medical cross |
| Fire rendering | Red-orange CA cells |
| HUD shows | Fire perimeter, delivery POI, distance_to_task, package status |

---

## Mission Family 2: Urban Rescue

### Identity

| Field | Value |
|-------|-------|
| Mission Type | `URBAN_RESCUE` |
| Scenario ID | `osm_piraeus_urban_rescue_medium` |
| Domain | `urban_rescue` |
| OSM Tile | Piraeus port |
| Inspired by | Rhodes/Ρόδος coastal fires 2023 |
| Building Density | 0.29 |

### Incident Provenance

**2023 Rhodes Coastal Fires**

The July 2023 Rhodes wildfires forced mass evacuations of coastal tourist
areas, stranding thousands of residents and visitors. Road closures and
active fire fronts complicated rescue operations, requiring aerial platforms
for urban search and rescue assessment. [CITE] Copernicus EMS Activation EMSR680

### Mission Objective

**Urban Search and Rescue Assessment of Fire-Stranded Casualties**

The UAV must reach stranded casualties through fire-affected urban terrain.
Dynamic obstacles (fire, traffic, road closures, structural collapse) create
evolving barriers. The agent must perform on-site triage assessment (service_time hover).

| Field | Value |
|-------|-------|
| Objective Label | "Urban Search & Rescue Assessment" |
| Deliverable | `rescue_assessment` |
| Reason String | "Urban search and rescue assessment of stranded population" |
| Service Time | 2 steps (hover assessment) |

### Task Queue

| Category | Weight | Time Decay | Service Time | Description |
|----------|--------|-----------|--------------|-------------|
| `rescue_site` | 1.0 | 0.02 | 2 steps | Rescue assessment — hover + confirm |

### Scenario Parameters

| Field | Value |
|-------|-------|
| Dynamics | Moderate (dynamic track) |
| Fire Ignitions | 2 |
| Vehicles | 3 |
| NFZ Zones | 1 |
| Forced Replans | 1 |
| Event Timing | t1=40, t2=120 |

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
| Scenario ID | `osm_downtown_fire_surveillance_medium` |
| Domain | `fire_surveillance` |
| OSM Tile | Athens center |
| Inspired by | Evros/Έβρος megafire 2023 |
| Building Density | 0.16 |

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

### Scenario Parameters

| Field | Value |
|-------|-------|
| Dynamics | Moderate (dynamic track) |
| Fire Ignitions | 3 |
| NFZ Zones | 2 |
| Forced Replans | 1 |
| Event Timing | t1=40, t2=120 |

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
| `flood_exclusion` | Urban Rescue | Blue `#1E90FF` | Flooded area / road closure |
| `aircraft_corridor` | Fire Surveillance | Purple `#9B59B6` | Active firefighting NFZ |

---

## Dynamics Configuration (All Missions — Medium Difficulty)

All scenarios use the same dynamics knobs:

| Knob | Value |
|------|-------|
| `enable_fire` | true |
| `enable_traffic` | true |
| `enable_dynamic_nfz` | true |
| `fire_blocks_movement` | true |
| `traffic_blocks_movement` | true |
| `force_replan_count` | 1 |
| `replan_every_steps` | 6 |

---

## MC Contract Mapping

| Contract | Card Section |
|----------|-------------|
| MC-1 | Objective POI + Reason String (each mission) |
| MC-2 | Task Queue: Service Time column |
| MC-3 | Visual Profile + Objective Label + Deliverable |
| MC-4 | Difficulty Matrix -> termination conditions |
