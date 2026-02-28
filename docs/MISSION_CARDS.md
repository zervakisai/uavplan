# Mission Briefing Cards

Each UAVBench episode begins with a mission briefing (logged at step 0).
Below are the briefing templates for each scenario family and difficulty.

---

## Mission 1: Civil Protection (Penteli Tile)

**Incident Context**: 2018 Attica Wildfire (Penteli/Mati) [CITE]

| Field | Easy | Medium | Hard |
|---|---|---|---|
| **Objective** | Emergency medical delivery during wildfire crisis | Emergency medical delivery during wildfire crisis | Emergency medical delivery during wildfire crisis |
| **Origin** | Penteli Fire Station | Penteli Fire Station | Penteli Fire Station |
| **Destination** | Evacuation Zone Alpha | Evacuation Zone Alpha | Evacuation Zone Alpha |
| **Deliverable** | Thermal-sealed medical kit | Thermal-sealed medical kit | Thermal-sealed medical kit |
| **Priority** | Routine | High | Critical |
| **Tasks** | 4 | 6 | 8 |
| **Dynamics** | Static | Moderate (fire, traffic, NFZ) | Severe (fire, traffic, NFZ, blocking) |
| **Constraints** | None | Avoid fire zones, respect NFZ | Avoid fire zones, respect NFZ, burning cells impassable, vehicle buffers |
| **Time Budget** | 350 steps | 300 steps | 250 steps |

---

## Mission 2: Maritime Domain (Piraeus Tile)

**Incident Context**: LS-ELAKT Coastal Patrol and Distress Response

| Field | Easy | Medium | Hard |
|---|---|---|---|
| **Objective** | Maritime search and rescue in multi-hazard zone | Maritime search and rescue in multi-hazard zone | Maritime search and rescue in multi-hazard zone |
| **Origin** | Piraeus Coast Guard Station | Piraeus Coast Guard Station | Piraeus Coast Guard Station |
| **Destination** | Maritime Distress Zone | Maritime Distress Zone | Maritime Distress Zone |
| **Deliverable** | Survivor location and status report | Survivor location and status report | Survivor location and status report |
| **Priority** | Routine | High | Critical |
| **Tasks** | 4 | 6 | 8 |
| **Dynamics** | Static | Moderate (fire, traffic, NFZ) | Severe (fire, traffic, NFZ, blocking) |
| **Constraints** | None | Avoid fire zones, respect NFZ | Avoid fire zones, respect NFZ, burning cells impassable, vehicle buffers |
| **Time Budget** | 400 steps | 300 steps | 250 steps |

---

## Mission 3: Critical Infrastructure (Downtown Athens Tile)

**Incident Context**: YPETHA/ISR-support infrastructure inspection under restricted airspace

| Field | Easy | Medium | Hard |
|---|---|---|---|
| **Objective** | Critical infrastructure inspection under restricted airspace | Critical infrastructure inspection under restricted airspace | Critical infrastructure inspection under restricted airspace |
| **Origin** | HCAA Operations Centre (Downtown Athens) | HCAA Operations Centre (Downtown Athens) | HCAA Operations Centre (Downtown Athens) |
| **Destination** | Infrastructure Inspection Site | Infrastructure Inspection Site | Infrastructure Inspection Site |
| **Deliverable** | Structural integrity assessment report | Structural integrity assessment report | Structural integrity assessment report |
| **Priority** | Routine | High | Critical |
| **Tasks** | 4 | 6 | 8 |
| **Dynamics** | Static | Moderate (fire, traffic, NFZ) | Severe (fire, traffic, NFZ, blocking) |
| **Constraints** | None | Avoid fire zones, respect NFZ | Avoid fire zones, respect NFZ, burning cells impassable, vehicle buffers |
| **Time Budget** | 300 steps | 275 steps | 250 steps |

---

## Briefing Event Schema

Every episode logs the briefing as the first event (step_idx=0):

```json
{
  "step": 0,
  "type": "mission_briefing",
  "payload": {
    "mission_type": "civil_protection",
    "domain": "urban",
    "origin_name": "Penteli Fire Station (Penteli)",
    "destination_name": "Evacuation Zone Alpha",
    "objective": "Emergency medical delivery during wildfire crisis",
    "deliverable": "Thermal-sealed medical kit",
    "constraints": ["Avoid active fire zones", "Respect dynamic no-fly restrictions"],
    "service_time_steps": 0,
    "priority": "critical",
    "max_time_steps": 250
  }
}
```

## Contract References

- **MC-1**: Every episode has a mission objective (this briefing provides it)
- **MC-4**: Results include the briefing in the event log
