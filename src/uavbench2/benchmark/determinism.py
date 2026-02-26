"""Determinism verification helpers (DC-2).

hash_episode() produces a stable SHA-256 digest of an EpisodeResult.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_episode(result: Any) -> str:
    """Compute a stable SHA-256 hash of an EpisodeResult.

    Hashes events, trajectory, and metrics as a canonical JSON string.
    """
    payload = {
        "events": result.events,
        "trajectory": result.trajectory,
        "metrics": result.metrics,
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()
