"""BFS shortest path utility.

Extracted from forced_block.py for reuse in urban.py as fallback
when A* corridor computation fails.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def bfs_shortest_path(
    heightmap: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """Compute BFS shortest path on static grid.

    Uses only heightmap (static obstacles). No dynamics.
    Coordinates: (x, y). Grid access: heightmap[y, x].

    Returns list of (x, y) from start to goal inclusive, or [] if no path.
    """
    H, W = heightmap.shape
    sx, sy = start
    gx, gy = goal

    if (sx, sy) == (gx, gy):
        return [(sx, sy)]

    visited = {(sx, sy)}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {(sx, sy): None}
    queue = deque([(sx, sy)])

    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) == (gx, gy):
            path: list[tuple[int, int]] = []
            node: tuple[int, int] | None = (gx, gy)
            while node is not None:
                path.append(node)
                node = parent[node]
            return list(reversed(path))

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in visited:
                if heightmap[ny, nx] == 0.0:
                    visited.add((nx, ny))
                    parent[(nx, ny)] = (cx, cy)
                    queue.append((nx, ny))

    return []
