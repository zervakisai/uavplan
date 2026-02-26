Verify all implemented contracts have passing tests.

1. Read `@docs/v2/CONTRACTS.md` for the full contract list.
2. For each contract (DC, FC, EC, GC, EV, VC, MC, PC, MP, RS):
   a. Check if the test file exists in `tests/v2/`
   b. Run it: `pytest tests/v2/contract_test_*.py -q --tb=short`
   c. Report: contract ID, test file, PASS/FAIL/MISSING
3. Produce a summary table.
4. If any contract test is MISSING or FAILING, flag it clearly.
5. Write results to `outputs/v2/contract_verification.json`
