Think hard about this. You are executing a phased build of UAVBench v2.

1. Read `@docs/v2/PHASE_GATES.md` to understand ALL phases and their gates.
2. Read `@docs/v2/CONTRACTS.md` to understand all contract requirements.
3. Determine the CURRENT phase by checking what exists:
   - If `src/uavbench2/` doesn't exist → Phase 0 or 1
   - If no tests pass → Phase 2
   - Check which contract tests exist and pass to determine current phase
4. If arguments are provided ($ARGUMENTS), execute that specific phase.
   Otherwise, report current phase status and what's needed for the gate.
5. BEFORE writing code, confirm the phase gate requirements.
6. AFTER implementing, run `pytest tests/v2/ -q` and report gate status.
7. If the gate passes, commit with message: "phase N: [description] — gate passed"
8. If the gate fails, diagnose and fix. Do NOT move to the next phase.

CRITICAL: Never skip a phase. Never implement Phase N+1 code during Phase N.
