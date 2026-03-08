# Fire Dynamics Contract (FD) — Wind-Optional

## FD-1: Fire State Machine
Cell states: UNBURNED → BURNING → BURNED
- UNBURNED → BURNING: if ≥1 neighbor BURNING AND rng < p_spread * wind_factor
- BURNING → BURNED: after burn_duration steps
- BURNED: permanent

## FD-2: Spread (8-neighbor Moore)
Fire spreads to 8 Moore neighbors. When wind_speed=0, spread is isotropic
(equal probability all 8 directions). When wind_speed>0, spread is
directionally modulated per Alexandridis et al. 2008.
- p_spread = base_prob * wind_factor[direction]
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

## FD-5b: Wind Parameters (OPTIONAL)
Wind is an OPTIONAL parameter. When wind_speed=0, fire is isotropic
(backward compat, bit-identical to v2). When wind_speed>0, spread
probability is directionally modulated per Alexandridis et al. 2008.
- wind_speed: float >= 0 (0 = isotropic)
- wind_direction: float in radians (0 = East, π/2 = North)

## WD-1: Wind Determinism
Wind modulation is deterministic: same (seed, wind_speed, wind_direction)
→ same spread pattern. Wind factors are pre-computed at init.

## Tests (unit_test_fire_ca.py)
- Known seed → expected cell-by-cell spread
- Two runs same seed → identical fire
- Fire timing: fire state unchanged between planner call and step validation
- Wind=0 → bit-identical to isotropic model
- Wind>0 → downwind spreads faster than upwind
