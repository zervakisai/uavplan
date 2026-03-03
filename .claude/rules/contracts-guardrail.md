# Feasibility Guardrail (GC)

## GC-1: Multi-depth relaxations
D1: Remove forced interdiction blocks
D2: Erode NFZ by 1 cell
D3: Remove traffic closures
D4 (optional): Open corridor

## GC-2: Logging
Every relaxation: guardrail_depth, relaxations list, feasible_after

## GC-3: Infeasible Episodes
If infeasible after all depths → flag episode. Report exclusion rate.

## GC-4: Best-effort ONLY
Guardrail provides NO guarantees. It tries, logs results, and reports.
