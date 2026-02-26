Check whether the current phase gate passes.

1. Read `@docs/v2/PHASE_GATES.md` to identify the current phase and its gate criteria.
2. Run the relevant tests: `pytest tests/v2/ -q --tb=short`
3. For each gate criterion, report PASS or FAIL with evidence.
4. If ALL criteria pass: print "✅ Gate N PASSED — safe to proceed to Phase N+1"
5. If ANY criterion fails: print "❌ Gate N FAILED" and list what's missing.
6. Never auto-proceed to the next phase. Wait for explicit instruction.
