"""Unified event pipeline: UpdateBus + ConflictDetector.

Provides the backbone for dynamic environment → comms → planner integration:

  UpdateEvent   — typed event envelope (obstacle, constraint, risk, task, comms)
  UpdateBus     — publish/subscribe event bus with filtering + replay
  ConflictDetector — checks a planned path against live dynamic obstacles
  SafetyMonitor — enforces no-ghosting + violation counting
  DynamicObstacleManager — mission-specific moving obstacles
  ForcedReplanScheduler — guarantees ≥2 replans per episode

Usage::

    bus = UpdateBus()
    bus.subscribe(EventType.OBSTACLE, my_callback)
    bus.publish(UpdateEvent(event_type=EventType.OBSTACLE, ...))
"""

from uavbench.updates.bus import (
    EventType,
    UpdateEvent,
    UpdateBus,
)
from uavbench.updates.conflict import ConflictDetector, Conflict
from uavbench.updates.safety import SafetyMonitor, SafetyConfig, Violation
from uavbench.updates.obstacles import (
    DynamicObstacleManager,
    VehicleLayer,
    VesselLayer,
    WorkZoneLayer,
)
from uavbench.updates.forced_replan import ForcedReplanScheduler

__all__ = [
    "EventType",
    "UpdateEvent",
    "UpdateBus",
    "ConflictDetector",
    "Conflict",
    "SafetyMonitor",
    "SafetyConfig",
    "Violation",
    "DynamicObstacleManager",
    "VehicleLayer",
    "VesselLayer",
    "WorkZoneLayer",
    "ForcedReplanScheduler",
]
