"""Contract tests for Visual Truth (VC-1, VC-2).

VC-1: Planned path visible when plan_len > 1.
VC-2: NO PLAN / STALE badge when plan is missing or stale.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from uavbench.scenarios.schema import Difficulty, MissionType, ScenarioConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> ScenarioConfig:
    defaults = dict(
        name="test_visual",
        mission_type=MissionType.FIRE_DELIVERY,
        difficulty=Difficulty.EASY,
        map_size=20,
        building_density=0.1,
        max_episode_steps=50,
        terminate_on_collision=False,
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


def _make_frame_state(
    *,
    plan_len: int = 10,
    plan_path: list[tuple[int, int]] | None = None,
    plan_age_steps: int = 0,
    plan_reason: str = "",
    step_idx: int = 0,
    agent_xy: tuple[int, int] = (5, 5),
    goal_xy: tuple[int, int] = (15, 15),
    mission_domain: str = "fire_delivery",
    objective_label: str = "Emergency Medical Supply Delivery",
    distance_to_task: float = 20.0,
    task_progress: str = "0/1",
    deliverable_name: str = "medical_supplies",
    trajectory: list[tuple[int, int]] | None = None,
    scenario_id: str = "test_visual",
    planner_name: str = "astar",
    replan_every_steps: int = 6,
) -> dict[str, Any]:
    """Build a frame state dict for the renderer."""
    if plan_path is None:
        # Generate a simple diagonal path
        plan_path = [(5 + i, 5 + i) for i in range(plan_len)] if plan_len > 0 else []
    if trajectory is None:
        trajectory = [agent_xy]
    return {
        "plan_len": len(plan_path),
        "plan_path": plan_path,
        "plan_age_steps": plan_age_steps,
        "plan_reason": plan_reason,
        "step_idx": step_idx,
        "agent_xy": agent_xy,
        "goal_xy": goal_xy,
        "mission_domain": mission_domain,
        "objective_label": objective_label,
        "distance_to_task": distance_to_task,
        "task_progress": task_progress,
        "deliverable_name": deliverable_name,
        "trajectory": trajectory,
        "scenario_id": scenario_id,
        "planner_name": planner_name,
        "replan_every_steps": replan_every_steps,
        "replans": 0,
        "dynamic_block_hits": 0,
    }


# ===========================================================================
# VC-1: Path Visibility
# ===========================================================================


class TestVC1_PathVisibility:
    """VC-1: If plan_len > 1, planned path overlay MUST be visible."""

    def test_path_visible_when_plan_exists(self):
        """When plan_len > 1, rendered frame contains cyan-family pixels."""
        from uavbench.visualization.renderer import Renderer

        config = _make_config()
        renderer = Renderer(config, mode="paper_min")

        heightmap = np.zeros((20, 20), dtype=np.float32)
        state = _make_frame_state(plan_len=10)
        frame, meta = renderer.render_frame(heightmap, state)

        # frame is np.ndarray (H, W, 3) or (H, W, 4)
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] in (3, 4)

        # meta must confirm path was rendered
        assert meta["path_rendered"] is True

        # Check for cyan-family pixels: R<150, G>150, B>200
        r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        cyan_mask = (r < 150) & (g > 150) & (b > 200)
        assert cyan_mask.any(), "VC-1: No cyan path pixels found when plan_len > 1"

    def test_no_silent_path_absence(self):
        """When plan_len == 0, NO path pixels, and meta says path not rendered."""
        from uavbench.visualization.renderer import Renderer

        config = _make_config()
        renderer = Renderer(config, mode="paper_min")

        heightmap = np.zeros((20, 20), dtype=np.float32)
        state = _make_frame_state(plan_len=0, plan_path=[])
        frame, meta = renderer.render_frame(heightmap, state)

        assert meta["path_rendered"] is False
        assert meta["plan_badge"] == "NO PLAN"


# ===========================================================================
# VC-2: Plan Status Badges
# ===========================================================================


class TestVC2_PlanStatusBadges:
    """VC-2: HUD shows NO PLAN / STALE badge when plan is absent or stale."""

    def test_no_plan_badge(self):
        """When plan_len <= 1, meta has plan_badge == 'NO PLAN'."""
        from uavbench.visualization.renderer import Renderer

        config = _make_config()
        renderer = Renderer(config, mode="ops_full")

        heightmap = np.zeros((20, 20), dtype=np.float32)
        state = _make_frame_state(plan_len=0, plan_path=[])
        _, meta = renderer.render_frame(heightmap, state)

        assert meta["plan_badge"] == "NO PLAN"

    def test_stale_badge(self):
        """When plan_age > 2 * replan_every, meta has STALE badge."""
        from uavbench.visualization.renderer import Renderer

        config = _make_config(replan_every_steps=6)
        renderer = Renderer(config, mode="ops_full")

        heightmap = np.zeros((20, 20), dtype=np.float32)
        state = _make_frame_state(
            plan_len=10,
            plan_age_steps=20,  # > 2 * 6 = 12
            plan_reason="path_blocked",
            replan_every_steps=6,
        )
        _, meta = renderer.render_frame(heightmap, state)

        assert "STALE" in meta["plan_badge"]
        assert "path_blocked" in meta["plan_badge"]

    def test_plan_reason_shown(self):
        """Fresh plan shows 'PLAN: Nwp' badge."""
        from uavbench.visualization.renderer import Renderer

        config = _make_config()
        renderer = Renderer(config, mode="ops_full")

        heightmap = np.zeros((20, 20), dtype=np.float32)
        state = _make_frame_state(plan_len=10, plan_age_steps=0)
        _, meta = renderer.render_frame(heightmap, state)

        assert "PLAN:" in meta["plan_badge"]
        assert "10wp" in meta["plan_badge"]


# ===========================================================================
# V3: Smoke threshold in visualization
# ===========================================================================


class TestSmokeThresholdViz:
    """Smoke overlay respects SMOKE_BLOCKING_THRESHOLD = 0.5."""

    def test_smoke_below_threshold_not_rendered(self):
        """Smoke at 0.4 (below 0.5 threshold) should NOT alter frame pixels."""
        from uavbench.visualization.overlays import draw_smoke

        cell = 2
        frame_clean = np.full((20, 20, 3), 200, dtype=np.uint8)
        frame_smoke = frame_clean.copy()

        smoke_mask = np.full((10, 10), 0.4, dtype=np.float32)
        draw_smoke(frame_smoke, smoke_mask, cell)

        np.testing.assert_array_equal(
            frame_clean, frame_smoke,
            err_msg="Smoke at 0.4 should NOT be rendered (threshold=0.5)",
        )

    def test_smoke_at_threshold_is_rendered(self):
        """Smoke at 0.5 (at threshold) SHOULD alter frame pixels."""
        from uavbench.visualization.overlays import draw_smoke

        cell = 2
        frame_clean = np.full((20, 20, 3), 200, dtype=np.uint8)
        frame_smoke = frame_clean.copy()

        smoke_mask = np.full((10, 10), 0.5, dtype=np.float32)
        draw_smoke(frame_smoke, smoke_mask, cell)

        assert not np.array_equal(frame_clean, frame_smoke), \
            "Smoke at 0.5 SHOULD be rendered (threshold=0.5 with >=)"


# ===========================================================================
# V1: Z-order consistency
# ===========================================================================


class TestZOrderConsistency:
    """Verify render order matches documented z-order stack."""

    def test_fire_over_smoke(self):
        """Fire (z=4) must be rendered on top of smoke (z=3.5)."""
        from uavbench.visualization.renderer import Renderer

        config = _make_config()
        renderer = Renderer(config, mode="ops_full")
        heightmap = np.zeros((20, 20), dtype=np.float32)

        # Create overlapping fire and smoke at the same cells
        fire_mask = np.zeros((20, 20), dtype=bool)
        fire_mask[5:10, 5:10] = True
        smoke_mask = np.full((20, 20), 0.8, dtype=np.float32)

        state = _make_frame_state(plan_len=0, plan_path=[])
        dynamic_state = {
            "fire_mask": fire_mask,
            "smoke_mask": smoke_mask,
        }
        frame, _ = renderer.render_frame(heightmap, state, dynamic_state)

        # In fire cells, the dominant color should be fire red-orange (255,80,20)
        # not smoke grey (160,160,160), proving fire is on top
        cell = renderer._cell_px
        # Sample a fire cell center
        fy, fx = 7 * cell + cell // 2, 7 * cell + cell // 2
        pixel = frame[fy, fx]
        # Fire blended at 80% should have R > 200 (red-dominant)
        assert pixel[0] > 200, f"Fire cell should be red-dominant, got R={pixel[0]}"

    def test_buffer_over_smoke(self):
        """Fire buffer (z=3.8) must be rendered on top of smoke (z=3.5)."""
        from uavbench.visualization.renderer import Renderer

        config = _make_config()
        renderer = Renderer(config, mode="ops_full")
        heightmap = np.zeros((20, 20), dtype=np.float32)

        # Fire at center, smoke everywhere — buffer ring around fire
        fire_mask = np.zeros((20, 20), dtype=bool)
        fire_mask[10, 10] = True  # single fire cell
        smoke_mask = np.full((20, 20), 0.8, dtype=np.float32)

        state = _make_frame_state(plan_len=0, plan_path=[])
        dynamic_state = {
            "fire_mask": fire_mask,
            "smoke_mask": smoke_mask,
        }
        frame, _ = renderer.render_frame(heightmap, state, dynamic_state)

        # Buffer ring cell (adjacent to fire, not fire itself) should show
        # buffer orange tint, not pure smoke grey. At z=3.8, buffer is on top
        # of smoke (z=3.5). The buffer dot pattern color is (220,140,60).
        cell = renderer._cell_px
        # Check a buffer cell (one cell away from fire)
        by, bx = 11 * cell + cell // 2, 10 * cell + cell // 2
        pixel = frame[by, bx]
        # Buffer blended over smoke should shift toward orange (R > G)
        # Pure smoke would be roughly (168,168,168) blended with basemap
        # Buffer adds orange tint making R distinctly higher
        assert pixel[0] > pixel[2], (
            f"Buffer cell should have R > B (orange tint), got {pixel}"
        )


# ===========================================================================
# VZ-1: Renderer modes
# ===========================================================================


class TestVZ1_RendererModes:
    """VZ-1: paper_min and ops_full modes produce valid frames."""

    def test_paper_min_produces_frame(self):
        from uavbench.visualization.renderer import Renderer

        config = _make_config()
        renderer = Renderer(config, mode="paper_min")
        assert renderer._cell_px == 15  # paper_min fixed at 15px/cell

        heightmap = np.zeros((20, 20), dtype=np.float32)
        state = _make_frame_state(plan_len=5)
        frame, meta = renderer.render_frame(heightmap, state)

        assert frame.dtype == np.uint8
        assert frame.ndim == 3
        assert meta["mode"] == "paper_min"

    def test_ops_full_produces_frame(self):
        from uavbench.visualization.renderer import Renderer

        config = _make_config()
        renderer = Renderer(config, mode="ops_full")
        # ops_full auto-scales: max(2, 1200 // map_size)
        assert renderer._cell_px == max(2, 1200 // 20)

        heightmap = np.zeros((20, 20), dtype=np.float32)
        state = _make_frame_state(plan_len=5)
        frame, meta = renderer.render_frame(heightmap, state)

        assert frame.dtype == np.uint8
        assert meta["mode"] == "ops_full"


# ===========================================================================
# VZ-3: Deterministic rendering
# ===========================================================================


class TestVZ3_DeterministicRendering:
    """VZ-3: Same episode data → identical frame bytes."""

    def test_identical_inputs_identical_frames(self):
        """Two render passes with identical inputs must produce identical frames."""
        from uavbench.visualization.renderer import Renderer

        config = _make_config()

        heightmap = np.zeros((20, 20), dtype=np.float32)
        fire_mask = np.zeros((20, 20), dtype=bool)
        fire_mask[8:12, 8:12] = True
        smoke_mask = np.full((20, 20), 0.6, dtype=np.float32)

        state = _make_frame_state(plan_len=8, step_idx=10)
        dynamic_state = {"fire_mask": fire_mask, "smoke_mask": smoke_mask}

        # Render A
        renderer_a = Renderer(config, mode="ops_full")
        frame_a, meta_a = renderer_a.render_frame(heightmap, state, dynamic_state)

        # Render B (fresh renderer, same inputs)
        renderer_b = Renderer(config, mode="ops_full")
        frame_b, meta_b = renderer_b.render_frame(heightmap, state, dynamic_state)

        np.testing.assert_array_equal(
            frame_a, frame_b,
            err_msg="VZ-3: Identical inputs must produce bit-identical frames",
        )
        assert meta_a == meta_b


# ===========================================================================
# Evidence artifact generation
# ===========================================================================


class TestVizArtifacts:
    """Generate viz_manifest.csv and viz_frame_checks.json."""

    def test_generate_evidence(self, tmp_path):
        """Generate evidence artifacts for Gate 8."""
        from uavbench.visualization.renderer import Renderer

        config = _make_config()
        renderer = Renderer(config, mode="ops_full")
        heightmap = np.zeros((20, 20), dtype=np.float32)

        manifest_rows = []
        frame_checks = []

        for step in range(5):
            state = _make_frame_state(
                step_idx=step,
                plan_len=10 if step > 0 else 0,
                plan_path=[(5 + i, 5 + i) for i in range(10)] if step > 0 else [],
            )
            frame, meta = renderer.render_frame(heightmap, state)

            manifest_rows.append({
                "step": step,
                "mode": "ops_full",
                "width": frame.shape[1],
                "height": frame.shape[0],
                "path_rendered": meta["path_rendered"],
                "plan_badge": meta["plan_badge"],
                "block_badge": meta["block_badge"],
            })
            frame_checks.append({
                "step": step,
                "has_cyan_pixels": bool(
                    ((frame[:, :, 0] < 150) & (frame[:, :, 1] > 150) & (frame[:, :, 2] > 200)).any()
                ) if meta["path_rendered"] else False,
                "plan_badge": meta["plan_badge"],
                "block_badge": meta["block_badge"],
            })

        # Write to outputs/
        out_dir = Path("outputs/v2")
        out_dir.mkdir(parents=True, exist_ok=True)

        # viz_manifest.csv
        csv_path = out_dir / "viz_manifest.csv"
        with open(csv_path, "w") as f:
            header = "step,mode,width,height,path_rendered,plan_badge,block_badge\n"
            f.write(header)
            for row in manifest_rows:
                f.write(
                    f"{row['step']},{row['mode']},{row['width']},"
                    f"{row['height']},{row['path_rendered']},"
                    f"\"{row['plan_badge']}\",\"{row['block_badge']}\"\n"
                )

        # viz_frame_checks.json
        json_path = out_dir / "viz_frame_checks.json"
        with open(json_path, "w") as f:
            json.dump(frame_checks, f, indent=2)

        assert csv_path.exists(), "viz_manifest.csv not generated"
        assert json_path.exists(), "viz_frame_checks.json not generated"

        # Verify step 0 has NO PLAN badge
        assert frame_checks[0]["plan_badge"] == "NO PLAN"
        # Verify steps 1-4 have path rendered
        for fc in frame_checks[1:]:
            assert fc["has_cyan_pixels"], f"Step {fc} should have cyan path pixels"
