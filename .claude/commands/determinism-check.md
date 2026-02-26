Run two identical episodes and verify bit-identical outputs.

1. Run: `python -m uavbench2 run --scenario urban_fire_basic --planner astar --seed 42 --output /tmp/run_a/`
2. Run: `python -m uavbench2 run --scenario urban_fire_basic --planner astar --seed 42 --output /tmp/run_b/`
3. Compare JSON outputs byte-for-byte. Hash both with SHA-256.
4. If hashes match: write to `outputs/v2/determinism_hashes.json` and report PASS.
5. If hashes differ: diff the outputs, identify the first divergence point, and report FAIL.
6. This verifies contract DC-2.
