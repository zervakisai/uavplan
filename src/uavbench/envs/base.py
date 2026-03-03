"""Base environment and shared enums.

TerminationReason and RejectReason enums live here so they can be imported
without pulling in the full UrbanEnvV2 implementation.
"""

from __future__ import annotations

from enum import Enum


class TerminationReason(str, Enum):
    """Why an episode ended (MC-4)."""

    SUCCESS = "success"
    COLLISION_BUILDING = "collision_building"
    COLLISION_NFZ = "collision_nfz"
    TIMEOUT = "timeout"
    INFEASIBLE = "infeasible"
    IN_PROGRESS = "in_progress"


class RejectReason(str, Enum):
    """Why a move was rejected (EC-1)."""

    BUILDING = "building"
    NO_FLY = "no_fly"
    FORCED_BLOCK = "forced_block"
    TRAFFIC_CLOSURE = "traffic_closure"
    FIRE = "fire"
    FIRE_BUFFER = "fire_buffer"
    SMOKE = "smoke"
    TRAFFIC_BUFFER = "traffic_buffer"
    DYNAMIC_NFZ = "dynamic_nfz"
    OUT_OF_BOUNDS = "out_of_bounds"


class BlockLifecycle(str, Enum):
    PENDING = "pending"
    TRIGGERED = "triggered"
    ACTIVE = "active"
    CLEARED = "cleared"


class TaskStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    EXPIRED = "expired"
    SKIPPED = "skipped"
