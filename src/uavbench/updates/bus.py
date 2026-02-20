"""UpdateEvent data model and UpdateBus publish/subscribe system.

The UpdateBus is the single channel through which all dynamic environment
changes flow: obstacle movements, NFZ activations, risk-field spikes,
task injections, and comms status changes.

Every event carries a typed envelope (``UpdateEvent``) with:
  - globally unique ``event_id``
  - ``EventType`` enum tag
  - simulation ``step`` timestamp
  - geometric payload (position, mask, polygon)
  - severity score ∈ [0, 1]
  - optional causal ``parent_id`` for provenance chains

Subscribers register callbacks per event type.  The bus guarantees
in-order delivery within a single step and supports replay for
deterministic testing.
"""

from __future__ import annotations

import itertools
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Event types
# ─────────────────────────────────────────────────────────────────────────────

class EventType(str, Enum):
    """Categories of dynamic updates flowing through the bus."""
    OBSTACLE = "obstacle"          # moving vehicle, vessel, work-zone
    CONSTRAINT = "constraint"      # NFZ activation, corridor block
    RISK = "risk"                  # risk-field spike
    TASK = "task"                  # new task injection / expiration
    COMMS = "comms"                # comms dropout / restoration
    REPLAN = "replan"              # planner-initiated replan (for logging)


# ─────────────────────────────────────────────────────────────────────────────
# Update event envelope
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UpdateEvent:
    """Typed envelope for a single dynamic update.

    Attributes
    ----------
    event_type : EventType
        Semantic category of this event.
    step : int
        Simulation step at which the event was created.
    description : str
        Human-readable short description (for timeline / logging).
    severity : float
        Normalised severity ∈ [0, 1].  0 = informational, 1 = critical.
    position : tuple[int, int] | None
        (x, y) grid position of the event source, if point-like.
    mask : np.ndarray | None
        [H, W] bool mask for area-based events (fire, NFZ, risk field).
    polygon : list[tuple[int, int]] | None
        Polygon vertices for geometric constraints (future use).
    payload : dict
        Arbitrary key-value payload (obstacle velocity, task spec, etc.).
    event_id : str
        Globally unique ID (auto-generated UUID4 by default).
    parent_id : str | None
        Optional causal parent event ID for provenance chains.
    """
    event_type: EventType
    step: int
    description: str = ""
    severity: float = 0.5
    position: tuple[int, int] | None = None
    mask: np.ndarray | None = None
    polygon: list[tuple[int, int]] | None = None
    payload: dict = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Subscriber callback signature
# ─────────────────────────────────────────────────────────────────────────────

Callback = Callable[[UpdateEvent], None]


# ─────────────────────────────────────────────────────────────────────────────
# UpdateBus
# ─────────────────────────────────────────────────────────────────────────────

class UpdateBus:
    """Publish/subscribe event bus for dynamic environment updates.

    Features
    --------
    - Per-type subscription with optional severity filter
    - Wildcard ``subscribe(None, cb)`` to receive all events
    - In-order delivery within a step
    - Full event log for replay and deterministic testing
    - ``drain()`` returns all events since last drain

    Thread-safety: NOT thread-safe (single-threaded simulation assumed).
    """

    def __init__(self) -> None:
        self._subscribers: dict[EventType | None, list[tuple[Callback, float]]] = defaultdict(list)
        self._log: list[UpdateEvent] = []
        self._drain_cursor: int = 0

    # ── Subscribe ─────────────────────────────────────────────────

    def subscribe(
        self,
        event_type: EventType | None,
        callback: Callback,
        *,
        min_severity: float = 0.0,
    ) -> None:
        """Register a callback for events of *event_type*.

        Parameters
        ----------
        event_type : EventType or None
            ``None`` = wildcard (all types).
        callback : Callable[[UpdateEvent], None]
            Invoked synchronously on each matching publish.
        min_severity : float
            Only deliver events with severity ≥ this threshold.
        """
        self._subscribers[event_type].append((callback, min_severity))

    def unsubscribe(
        self,
        event_type: EventType | None,
        callback: Callback,
    ) -> None:
        """Remove a previously registered callback."""
        subs = self._subscribers.get(event_type, [])
        self._subscribers[event_type] = [
            (cb, ms) for cb, ms in subs if cb is not callback
        ]

    # ── Publish ───────────────────────────────────────────────────

    def publish(self, event: UpdateEvent) -> int:
        """Publish an event to all matching subscribers.

        Returns the number of callbacks invoked.
        """
        self._log.append(event)
        delivered = 0

        # Type-specific subscribers
        for cb, min_sev in self._subscribers.get(event.event_type, []):
            if event.severity >= min_sev:
                cb(event)
                delivered += 1

        # Wildcard subscribers
        for cb, min_sev in self._subscribers.get(None, []):
            if event.severity >= min_sev:
                cb(event)
                delivered += 1

        return delivered

    def publish_many(self, events: Sequence[UpdateEvent]) -> int:
        """Publish a batch of events.  Returns total deliveries."""
        return sum(self.publish(e) for e in events)

    # ── Query / Replay ────────────────────────────────────────────

    @property
    def event_log(self) -> list[UpdateEvent]:
        """Full chronological event log (read-only view)."""
        return list(self._log)

    def events_at_step(self, step: int) -> list[UpdateEvent]:
        """Return all events that occurred at a specific step."""
        return [e for e in self._log if e.step == step]

    def events_of_type(self, event_type: EventType) -> list[UpdateEvent]:
        """Return all events of a specific type."""
        return [e for e in self._log if e.event_type == event_type]

    def drain(self) -> list[UpdateEvent]:
        """Return events published since last drain and advance cursor."""
        new_events = self._log[self._drain_cursor:]
        self._drain_cursor = len(self._log)
        return new_events

    def replay(self, target_bus: "UpdateBus | None" = None) -> int:
        """Replay all logged events through another bus (or self).

        Useful for deterministic re-execution.
        """
        bus = target_bus or self
        count = 0
        for event in self._log:
            bus.publish(event)
            count += 1
        return count

    def clear(self) -> None:
        """Clear event log and reset drain cursor."""
        self._log.clear()
        self._drain_cursor = 0

    @property
    def total_events(self) -> int:
        return len(self._log)

    def summary(self) -> dict[str, int]:
        """Count of events by type."""
        counts: dict[str, int] = {}
        for e in self._log:
            counts[e.event_type.value] = counts.get(e.event_type.value, 0) + 1
        return counts
