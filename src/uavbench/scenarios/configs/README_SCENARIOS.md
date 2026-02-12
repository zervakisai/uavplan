# UAVBench Scenario Pack

8 Athens-centric scenarios across defense, civil protection, and emergency response domains.
All scenarios use real OpenStreetMap data rasterized to 500x500 grids (3m/pixel).

## Tiles

| Tile | Area | Character | Max Building Height |
|------|------|-----------|-------------------|
| `downtown` | Central Athens | Dense urban, heavy road network, hospitals/schools | 30m |
| `penteli` | Penteli foothills | Wilderness/urban interface, forest-dominated | 18m |
| `piraeus` | Piraeus port | Maritime/urban mix, tall port structures | 84m |

## Scenarios

### 1. Wildfire WUI (`osm_athens_wildfire_{easy,medium,hard}`)
- **Tile**: penteli
- **Dynamics**: Fire spread (cellular automaton)
- **Application**: Forest fire monitoring, evacuation routing
- **Progression**: Easy (3 ignitions, wind=0.3) → Hard (7 ignitions, wind=0.7)

### 2. Emergency Response (`osm_athens_emergency_{easy,medium,hard}`)
- **Tile**: downtown
- **Dynamics**: Emergency vehicle traffic
- **Application**: Medical emergency routing, crisis response
- **Progression**: Easy (5 vehicles) → Hard (12 vehicles, low ceiling)

### 3. Port Security (`osm_athens_port_{easy,medium,hard}`)
- **Tile**: piraeus
- **Dynamics**: None (static obstacles)
- **Application**: Port surveillance, critical infrastructure security
- **Progression**: Easy (alt=10) → Hard (alt=6 with 84m structures)

### 4. Combined Crisis (`osm_athens_crisis_hard`)
- **Tile**: downtown
- **Dynamics**: Fire + traffic simultaneously
- **Application**: Multi-agency coordination, compound emergencies
- **Difficulty**: Hard only (5 fires + 10 vehicles + low ceiling)

### 5. Search & Rescue (`osm_athens_sar_{easy,medium,hard}`)
- **Tile**: penteli
- **Dynamics**: None (static terrain)
- **Application**: Missing person search, wilderness rescue
- **Progression**: Easy (alt=10, L1=150) → Hard (alt=6, L1=250)

### 6. Infrastructure Patrol (`osm_athens_infrastructure_{easy,medium,hard}`)
- **Tile**: downtown
- **Dynamics**: None (static urban)
- **Application**: Infrastructure inspection, damage assessment
- **Progression**: Easy (alt=10) → Hard (alt=6, long route through urban canyons)

### 7. Border Surveillance (`osm_athens_border_{easy,medium,hard}`)
- **Tile**: penteli
- **Dynamics**: None (static terrain)
- **Application**: Perimeter monitoring, area control
- **Progression**: Easy (L1=200) → Hard (L1=400, maximum patrol distance)

### 8. Communications-Denied (`osm_athens_comms_denied_hard`)
- **Tile**: downtown
- **Dynamics**: None (future: dynamic NFZ zones)
- **Application**: Contested airspace, electronic warfare
- **Difficulty**: Hard only (low ceiling, long route)

## Difficulty Progression

| Parameter | Easy | Medium | Hard |
|-----------|------|--------|------|
| `max_altitude` | 10 | 8 | 6 |
| `safe_altitude` | 8 | 6 | 5 |
| `min_start_goal_l1` | 150 | 200 | 250+ |
| Fire ignitions | 3 | 5 | 7 |
| Wind speed | 0.3 | 0.5 | 0.7 |
| Vehicles | 5 | 8 | 12 |

## Usage

```bash
# Single scenario
python -m uavbench.cli.benchmark --scenarios osm_athens_wildfire_easy --planners astar --trials 3

# Multiple scenarios
python -m uavbench.cli.benchmark \
    --scenarios osm_athens_wildfire_easy,osm_athens_emergency_easy,osm_athens_port_easy \
    --planners astar --trials 5

# With visualization
python -m uavbench.cli.benchmark \
    --scenarios osm_athens_crisis_hard \
    --planners astar --trials 1 \
    --save-figures outputs/ --with-dynamics

# Full scenario pack
python tools/benchmark_scenario_pack.py --planners astar --trials 3 --output outputs/
```
