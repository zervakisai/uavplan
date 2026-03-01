# Final Visualization & Safety Fixes Superprompt

Think hard. There are 7 remaining issues that must ALL be fixed before 
we can proceed to experiments. Fix them in order.

---

## FIX 1: FIRE SAFETY BUFFER (CRITICAL — affects planner behavior)

The drone currently flies 1 cell from active flames. This is unrealistic 
and dangerous. A real drone cannot fly near active fire — heat radiation, 
smoke, turbulence from updrafts would bring it down.

### Implementation:
In `_build_runtime_blocking_mask()` in `envs/urban.py`, AFTER computing 
the fire mask and BEFORE returning the final blocking mask:

```python
from scipy.ndimage import binary_dilation

# Fire safety buffer: block cells NEAR fire, not just fire cells
if fire_mask is not None and fire_mask.any():
    fire_buffer = binary_dilation(fire_mask, iterations=fire_buffer_cells)
    blocking_mask |= fire_buffer
```

### Buffer size varies by fire intensity:
- Newly burning cells (< 5 steps active): 3-cell buffer
- Active burning (5-20 steps): 5-cell buffer
- Intense burning (> 20 steps): 7-cell buffer
- Burned-out cells: 0 buffer (safe to fly over ash)

If intensity-varying buffer is too complex, use FIXED 5-cell buffer for 
all active fire. That's acceptable.

### Configuration:
Add to scenario YAML configs (all 6 files):
```yaml
safety_buffers:
  fire_buffer_cells: 5
  nfz_buffer_cells: 3
```

Read these values in urban.py __init__ from the scenario config.
If not present, default to fire_buffer_cells=5, nfz_buffer_cells=3.

---

## FIX 2: NFZ SAFETY BUFFER (CRITICAL — affects planner behavior)

The drone must NOT fly on the NFZ boundary. It must keep clearance.
A real drone operator maintains legal safety margin from restricted airspace.

### Implementation:
Same location as Fix 1, in `_build_runtime_blocking_mask()`:

```python
# NFZ safety buffer: keep clearance from restricted zones
if nfz_mask is not None and nfz_mask.any():
    nfz_buffer = binary_dilation(nfz_mask, iterations=nfz_buffer_cells)
    blocking_mask |= nfz_buffer
```

### Buffer logic:
- Schools, hospitals (sensitive_locations): 3-cell buffer minimum
- TFR (firefighting operations): 5-cell buffer (manned aircraft inside)
- Dynamic NFZ: 3-cell buffer
- The planner sees the BUFFERED mask — it plans routes WITH clearance

---

## FIX 3: VISUALIZE THE BUFFER ZONES

The viewer must SEE why the drone routes far from dangers.
Without visible buffers, it looks like the drone avoids empty space.

### Implementation:
In operational_renderer.py, add a new layer between fire drawing and 
path drawing:

```python
# Draw fire safety buffer as light orange halo
if fire_mask is not None and fire_mask.any():
    fire_buffer_vis = binary_dilation(fire_mask, iterations=5) & ~fire_mask
    # fire_buffer_vis = buffer zone MINUS actual fire (the halo ring)
    ax.imshow(np.where(fire_buffer_vis, 1, np.nan), 
              cmap=ListedColormap(['#FF8C00']),  # dark orange
              alpha=0.2, extent=extent, zorder=5.5,
              interpolation='nearest')

# Draw NFZ safety buffer as light purple halo  
if nfz_mask is not None and nfz_mask.any():
    nfz_buffer_vis = binary_dilation(nfz_mask, iterations=3) & ~nfz_mask
    ax.imshow(np.where(nfz_buffer_vis, 1, np.nan),
              cmap=ListedColormap(['#BB88FF']),  # light purple
              alpha=0.2, extent=extent, zorder=5.5,
              interpolation='nearest')
```

### Legend:
Add to legend:
- "Fire Buffer" — light orange swatch
- "NFZ Buffer" — light purple swatch

---

## FIX 4: STEP COUNTER BUG

The HUD shows "STEP: 869 / 300" — denominator 300 is WRONG.
The episode clearly runs more than 300 steps.

### Diagnosis:
Find where the denominator (300) comes from. It's likely:
- Hardcoded 300 somewhere in hud.py or operational_renderer.py
- OR the max_time_steps from the scenario config is 300 but the 
  episode actually runs longer (which would be a separate bug)

### Fix:
- If max_time_steps in YAML is 300 but episodes run 800+ steps, 
  INCREASE max_time_steps to match reality (e.g., 1500 or 2000)
- If HUD reads the wrong field, fix the field reference
- The HUD should show: STEP: {current} / {actual_max}
- It should NEVER show current > max

### Also check:
Why does the episode run 869 steps if max is 300? Is timeout not 
being enforced? If the episode should terminate at 300, that's a 
DIFFERENT bug (timeout not working).

Report what you find — this could indicate a deeper issue.

---

## FIX 5: DRONE MARKER INVISIBLE

I cannot find the drone in ANY of the 3 frames. This is unacceptable.
The drone must be the MOST VISIBLE element on the map.

### Fix:
1. Drone triangle marker: increase markersize from current to **20**
2. Add WHITE GLOW circle behind drone:
   - White circle, markersize=28, alpha=0.6, zorder just below drone
   - This creates a halo effect visible against any background
3. Add DIRECTION indicator:
   - Small line extending from drone in movement direction, 
     length=8 cells, white, linewidth=1, alpha=0.5
4. Drone color states:
   - Normal flight: bright blue #4488FF
   - Stuck 5+ steps: orange #FFA500
   - Stuck 10+ steps: red #FF0000
   - These colors must override any other coloring

### Test: 
In the generated frame, I should be able to find the drone in < 1 second.

---

## FIX 6: TRAIL CONTRAST

Executed trail and planned path look too similar. They're both light-colored 
thin lines that blend together.

### Fix:
- **Executed trail** (where drone HAS BEEN):
  - Color: #00FF88 (bright green)
  - linewidth: **3.5** (THICK — this is the dominant visual element)
  - Style: SOLID
  - alpha: 0.9
  
- **Planned path** (where drone INTENDS to go):
  - Color: #00BFFF (cyan)  
  - linewidth: **1.5** (thin)
  - Style: dashed, dash pattern (8, 4) — long dashes
  - alpha: 0.6

The executed trail must be OBVIOUSLY THICKER than the planned path.
A viewer should instantly distinguish "where it went" from "where it plans."

---

## FIX 7: DRONE MUST NOT ENTER NFZ — EVER

Looking at the frames carefully, the path appears to pass very close to 
or potentially through NFZ zones. With the buffer in Fix 2, this should 
be impossible, but verify:

### Verification:
After implementing Fixes 1-2, run a quick test:
```python
# For every step in an episode, verify:
assert not nfz_buffered_mask[drone_y, drone_x], \
    f"Drone entered NFZ buffer at step {step}, pos ({drone_x}, {drone_y})"
```

If the drone EVER enters the NFZ buffer zone, the blocking mask is not 
being enforced correctly. Debug and fix.

### Also verify fire buffer:
```python
assert not fire_buffered_mask[drone_y, drone_x], \
    f"Drone entered fire buffer at step {step}, pos ({drone_x}, {drone_y})"
```

---

## EXECUTION ORDER

1. Fix 1 + Fix 2: Safety buffers in urban.py (changes planner behavior)
2. Fix 4: Step counter bug (diagnose and fix)
3. Run pytest tests/ -x -q → must still pass
4. Fix 3: Visualize buffer zones in renderer
5. Fix 5: Drone marker bigger
6. Fix 6: Trail contrast
7. Fix 7: Verification that drone never enters buffer zones
8. Generate test GIF: gov_civil_protection_medium, aggressive_replan, seed=1
9. Extract 3 frames: ~step 200, ~step 500, ~step 800
10. Show me the frames

## GATE CHECKLIST

Before showing me frames, verify ALL of these:
- [ ] Fire buffer visible as orange halo around fire
- [ ] NFZ buffer visible as purple halo around NFZ  
- [ ] Path routes with VISIBLE clearance from fire (not hugging edge)
- [ ] Path routes with VISIBLE clearance from NFZ
- [ ] Drone marker easily findable (< 1 second visual search)
- [ ] STEP: X / Y where X is never > Y
- [ ] Executed trail OBVIOUSLY thicker than planned path
- [ ] Drone never enters fire buffer or NFZ buffer (verification passes)
- [ ] All tests pass
- [ ] REPLANS counter shows in HUD and increases over time

Commit: "safety buffers + visualization polish: fire/NFZ clearance, 
drone visibility, trail contrast"
