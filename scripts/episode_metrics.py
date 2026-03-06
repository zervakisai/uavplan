"""Extract full metrics for a single episode."""

from uavbench.benchmark.runner import run_episode

result = run_episode(
    "osm_penteli_fire_delivery_medium",
    "aggressive_replan",
    seed=1,
)

# Print all metric keys
for k, v in sorted(result.metrics.items()):
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")
