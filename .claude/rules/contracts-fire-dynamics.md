# Fire Dynamics Contract (FD) — No Wind, Isotropic

## FD-1: Fire State Machine
Cell states: UNBURNED → BURNING → BURNED
- UNBURNED → BURNING: if ≥1 neighbor BURNING AND rng < p_spread
- BURNING → BURNED: after burn_duration steps
- BURNED: permanent

## FD-2: Isotropic Spread (NO WIND)
IMPORTANT: v2 has NO wind parameter. Fire spreads with EQUAL probability
to all 8 neighbors (Moore neighborhood). This is a deliberate simplification.
- p_spread = base probability (same for all 8 directions)
- Fire buffer = binary dilation of radius fire_buffer_radius around BURNING cells
- Buffer cells are BLOCKED for UAV movement

## FD-3: Step Timing (CRITICAL — fixes BUG-1c)
Within each step_idx, execution order is:
1. Planner receives CURRENT fire state (frozen)
2. Planner computes action
3. env.step() validates action against CURRENT blocking mask
4. UAV moves (if valid)
5. THEN fire CA advances one step → new fire state
6. New blocking mask computed for next step_idx

Fire NEVER advances between steps 1-4. This is non-negotiable.

## FD-4: Determinism
fire_step(state, rng) is a PURE FUNCTION.
Same seed → same fire evolution. Fire does not interact with UAV.

## FD-5: No Wind Parameters
The fire CA module MUST NOT have wind_speed, wind_direction, or any
directional bias parameter. If found → DELETE.

## Tests (unit_test_fire_ca.py)
- Known seed → expected cell-by-cell spread (isotropic pattern)
- Two runs same seed → identical fire
- Fire timing: fire state unchanged between planner call and step validation
- Assert: NO wind parameter in fire_ca.py
