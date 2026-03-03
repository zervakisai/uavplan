# Event Semantics (EV) + Decision Record (EC)

## EV-1: Authoritative step_idx
Runner owns step_idx. Passes to env, dynamics, logger, renderer.
Components do NOT maintain own counters.

## EC-1: Rejected Moves
Every rejection logs: reject_reason(enum), reject_layer, reject_cell, step_idx.

## EC-2: Accepted Moves
Every acceptance logs: move_accepted=True, dynamics_step.

## RejectReason Enum
OUT_OF_BOUNDS, BLOCKED_STATIC, BLOCKED_DYNAMIC, NFZ, FIRE, FIRE_BUFFER, TRAFFIC
