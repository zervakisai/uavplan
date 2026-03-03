"""Extract full metrics for a single episode."""

import json
from uavbench.cli.benchmark import run_dynamic_episode

result = run_dynamic_episode(
    "gov_civil_protection_medium",
    "aggressive_replan",
    seed=1,
)

# Print all keys
for k, v in sorted(result.items()):
    if isinstance(v, (list, dict)) and len(str(v)) > 200:
        print(f"{k}: <{type(v).__name__}, len={len(v)}>")
    elif isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")
