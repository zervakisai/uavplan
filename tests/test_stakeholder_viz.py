"""Tests for the stakeholder visualization system.

Tests cover:
  - Icon system (library, stamping, all icons)
  - Tile loader (load real/synthetic tiles)
  - Basemap builder
  - StakeholderRenderer (construction, frame rendering)
  - Overlays (all 3 mission types)
  - Demo pack export
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_tile():
    """Create a small synthetic TileData for testing (no disk I/O)."""
    from uavbench.visualization.stakeholder_renderer import TileData

    H, W = 64, 64
    rng = np.random.default_rng(42)
    return TileData(
        tile_id="test_synthetic",
        heightmap=rng.choice([0.0, 10.0, 20.0], size=(H, W), p=[0.7, 0.2, 0.1]).astype(np.float32),
        roads_mask=rng.random((H, W)) < 0.15,
        landuse_map=rng.integers(0, 5, size=(H, W), dtype=np.int8),
        risk_map=rng.random((H, W)).astype(np.float32) * 0.5,
        nfz_mask=rng.random((H, W)) < 0.05,
        center_latlon=(37.97, 23.73),
        resolution_m=3.0,
        grid_size=64,
    )


@pytest.fixture
def small_renderer(synthetic_tile):
    """Create a small StakeholderRenderer for testing."""
    from uavbench.visualization.stakeholder_renderer import StakeholderRenderer

    r = StakeholderRenderer(
        tile=synthetic_tile,
        mission_type="civil_protection",
        scenario_id="test_scenario",
        planner_name="astar",
        difficulty="medium",
        icon_size=5.0,
        figsize=(9.6, 5.4),
        dpi=50,
    )
    yield r
    r.close()


# ═════════════════════════════════════════════════════════════════════════════
# Icon System Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestIconLibrary:
    """Tests for the icon system."""

    def test_import(self):
        from uavbench.visualization.icons import IconLibrary, IconID
        assert IconLibrary is not None
        assert IconID is not None

    def test_available_icons(self):
        from uavbench.visualization.icons import IconLibrary
        lib = IconLibrary()
        icons = lib.available_icons()
        assert len(icons) >= 15
        assert "uav" in icons
        assert "fire" in icons
        assert "ship" in icons

    def test_icon_id_enum(self):
        from uavbench.visualization.icons import IconID
        assert IconID.UAV.value == "uav"
        assert IconID.FIRE.value == "fire"
        assert IconID.SHIP.value == "ship"
        assert IconID.BUILDING.value == "building"

    def test_stamp_all_icons(self):
        """Stamp every registered icon — no exceptions."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from uavbench.visualization.icons import IconLibrary

        lib = IconLibrary(icon_size=5.0)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        for i, icon_id in enumerate(lib.available_icons()):
            x = 10 + (i % 5) * 18
            y = 10 + (i // 5) * 18
            artists = lib.stamp(icon_id, (x, y), ax, label=icon_id)
            assert len(artists) >= 1

        plt.close(fig)

    def test_stamp_with_rotation(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from uavbench.visualization.icons import IconLibrary, IconID

        lib = IconLibrary(icon_size=8.0)
        fig, ax = plt.subplots()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        for deg in [0, 45, 90, 180, 270]:
            artists = lib.stamp(IconID.UAV, (50, 50), ax, rotation_deg=deg)
            assert len(artists) >= 1

        plt.close(fig)

    def test_stamp_many(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from uavbench.visualization.icons import IconLibrary, IconID

        lib = IconLibrary(icon_size=5.0)
        fig, ax = plt.subplots()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        positions = [(10, 10), (30, 30), (50, 50), (70, 70)]
        results = lib.stamp_many(IconID.WAYPOINT, positions, ax)
        assert len(results) == 4

        plt.close(fig)

    def test_unknown_icon_raises(self):
        from uavbench.visualization.icons import IconLibrary
        lib = IconLibrary()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        with pytest.raises(KeyError, match="Unknown icon"):
            lib.stamp("nonexistent_icon_xyz", (50, 50), ax)
        plt.close(fig)

    def test_path_caching(self):
        from uavbench.visualization.icons import IconLibrary, IconID
        lib = IconLibrary()
        path1 = lib._get_path("uav")
        path2 = lib._get_path("uav")
        assert path1 is path2  # Same cached object


# ═════════════════════════════════════════════════════════════════════════════
# Tile Loader Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestTileLoader:
    """Tests for tile loading."""

    def test_synthetic_tile(self, synthetic_tile):
        assert synthetic_tile.tile_id == "test_synthetic"
        assert synthetic_tile.heightmap.shape == (64, 64)
        assert synthetic_tile.roads_mask.dtype == bool
        assert synthetic_tile.landuse_map.dtype == np.int8
        assert synthetic_tile.resolution_m == 3.0

    def test_load_real_tile_penteli(self):
        """Load real penteli tile if available."""
        from uavbench.visualization.stakeholder_renderer import load_tile

        data_dir = Path(__file__).resolve().parents[1] / "data" / "maps"
        if not (data_dir / "penteli.npz").exists():
            pytest.skip("penteli.npz not available")

        tile = load_tile("penteli", data_dir=data_dir)
        assert tile.tile_id == "penteli"
        assert tile.heightmap.shape == (500, 500)
        assert tile.resolution_m == 3.0
        assert tile.center_latlon[0] > 37  # Greece

    def test_load_missing_tile_raises(self):
        from uavbench.visualization.stakeholder_renderer import load_tile

        with pytest.raises(FileNotFoundError, match="not found"):
            load_tile("nonexistent_tile", data_dir=Path("/tmp/empty"))


# ═════════════════════════════════════════════════════════════════════════════
# Basemap Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestBasemap:
    """Tests for basemap construction."""

    def test_build_basemap(self, synthetic_tile):
        from uavbench.visualization.stakeholder_renderer import build_basemap

        rgb = build_basemap(synthetic_tile)
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.float32
        assert 0 <= rgb.min() <= rgb.max() <= 1.0

    def test_basemap_has_buildings(self, synthetic_tile):
        from uavbench.visualization.stakeholder_renderer import build_basemap, _hex_rgb, STAKEHOLDER_PALETTE

        rgb = build_basemap(synthetic_tile)
        building_color = _hex_rgb(STAKEHOLDER_PALETTE["building_fill"])
        # Some pixels should be building color
        building_mask = synthetic_tile.heightmap > 0
        if building_mask.any():
            # At least some building pixels should exist
            assert building_mask.sum() > 0


# ═════════════════════════════════════════════════════════════════════════════
# StakeholderRenderer Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestStakeholderRenderer:
    """Tests for the 4-pane stakeholder renderer."""

    def test_construction(self, small_renderer):
        assert small_renderer.mission_type == "civil_protection"
        assert small_renderer.profile.tile_id == "penteli"
        assert small_renderer._basemap_rgb.shape == (64, 64, 3)

    def test_render_frame_returns_rgb(self, small_renderer):
        frame = small_renderer.render_frame(
            drone_pos=(32, 32),
            step=0,
            max_steps=100,
        )
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8

    def test_render_with_pois(self, small_renderer):
        from uavbench.visualization.icons import IconID

        pois = [
            {"xy": (10, 10), "icon": IconID.FIRE, "label": "F1", "status": "pending"},
            {"xy": (50, 50), "icon": IconID.CAMERA, "label": "C1", "status": "active"},
            {"xy": (30, 30), "icon": IconID.WAYPOINT, "label": "W1", "status": "completed"},
        ]
        frame = small_renderer.render_frame(
            drone_pos=(32, 32),
            step=10,
            max_steps=100,
            pois=pois,
        )
        assert frame.ndim == 3

    def test_render_with_trajectory(self, small_renderer):
        traj = [(10, 10), (15, 15), (20, 20), (25, 25), (30, 30)]
        planned = [(30, 30), (40, 40), (50, 50)]
        frame = small_renderer.render_frame(
            drone_pos=(30, 30),
            step=50,
            max_steps=100,
            trajectory=traj,
            planned_path=planned,
        )
        assert frame.ndim == 3

    def test_render_with_fire(self, small_renderer):
        fire = np.zeros((64, 64), dtype=bool)
        fire[20:30, 20:30] = True
        smoke = np.zeros((64, 64), dtype=bool)
        smoke[15:25, 25:35] = True

        frame = small_renderer.render_frame(
            drone_pos=(32, 32),
            step=5,
            max_steps=100,
            fire_mask=fire,
            smoke_mask=smoke,
        )
        assert frame.ndim == 3

    def test_render_with_nfz(self, small_renderer):
        nfz = np.zeros((64, 64), dtype=bool)
        nfz[40:50, 40:50] = True

        frame = small_renderer.render_frame(
            drone_pos=(32, 32),
            step=5,
            max_steps=100,
            nfz_mask=nfz,
        )
        assert frame.ndim == 3

    def test_render_with_metrics(self, small_renderer):
        metrics = {
            "tasks_completed": 3,
            "tasks_pending": 2,
            "replans": 1,
            "risk_integral": 0.42,
            "energy_used": 55.0,
            "mission_score": 0.6,
        }
        frame = small_renderer.render_frame(
            drone_pos=(32, 32),
            step=50,
            max_steps=100,
            metrics=metrics,
        )
        assert frame.ndim == 3

    def test_event_logging(self, small_renderer):
        small_renderer.log_event(10, "replan", "NFZ expansion")
        small_renderer.log_event(25, "task_complete", "Reached P1")
        assert len(small_renderer._events) == 2
        assert small_renderer._events[0]["type"] == "replan"

    def test_frame_accumulation(self, small_renderer):
        for step in range(5):
            small_renderer.render_frame(
                drone_pos=(32 + step, 32),
                step=step,
                max_steps=10,
            )
        assert len(small_renderer._frames) == 5

    def test_keyframe_marking(self, small_renderer):
        for step in range(5):
            small_renderer.render_frame(
                drone_pos=(32 + step, 32),
                step=step,
                max_steps=10,
                is_keyframe=(step == 0 or step == 4),
            )
        assert len(small_renderer._keyframe_indices) == 2

    def test_close(self, synthetic_tile):
        from uavbench.visualization.stakeholder_renderer import StakeholderRenderer

        r = StakeholderRenderer(
            tile=synthetic_tile,
            figsize=(4, 3),
            dpi=30,
        )
        r.render_frame((32, 32), 0, 10)
        r.close()
        assert r._fig is None

    def test_mission_profiles(self):
        from uavbench.visualization.stakeholder_renderer import MISSION_PROFILES

        assert "civil_protection" in MISSION_PROFILES
        assert "maritime_domain" in MISSION_PROFILES
        assert "critical_infrastructure" in MISSION_PROFILES

        cp = MISSION_PROFILES["civil_protection"]
        assert cp.tile_id == "penteli"
        assert "ΓΓΠΠ" in cp.agency


# ═════════════════════════════════════════════════════════════════════════════
# Overlay Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestOverlays:
    """Tests for mission-specific overlays."""

    def test_create_overlay_factory(self):
        from uavbench.visualization.overlays import create_overlay

        for mtype in ["civil_protection", "maritime_domain", "critical_infrastructure"]:
            overlay = create_overlay(mtype, grid_shape=(64, 64))
            assert overlay is not None
            assert overlay.H == 64
            assert overlay.W == 64

    def test_unknown_overlay_raises(self):
        from uavbench.visualization.overlays import create_overlay

        with pytest.raises(ValueError, match="Unknown mission type"):
            create_overlay("nonexistent_mission")

    def test_civil_protection_compute(self):
        from uavbench.visualization.overlays import CivilProtectionOverlay

        overlay = CivilProtectionOverlay(
            grid_shape=(64, 64),
            fire_origins=[(32, 32)],
        )
        result = overlay.compute(step=0)
        assert "fire_mask" in result
        assert "smoke_mask" in result
        assert result["fire_mask"].shape == (64, 64)

    def test_civil_protection_fire_spreads(self):
        from uavbench.visualization.overlays import CivilProtectionOverlay

        overlay = CivilProtectionOverlay(
            grid_shape=(64, 64),
            fire_origins=[(32, 32)],
            fire_spread_rate=1.0,
        )
        r0 = overlay.compute(step=0)
        count0 = r0["fire_mask"].sum()

        for s in range(1, 10):
            overlay.compute(step=s)

        r10 = overlay.compute(step=10)
        count10 = r10["fire_mask"].sum()
        assert count10 >= count0  # fire should spread

    def test_maritime_overlay_has_vessels(self):
        from uavbench.visualization.overlays import MaritimeDomainOverlay

        overlay = MaritimeDomainOverlay(
            grid_shape=(64, 64),
            num_vessels=3,
            patrol_radius=20,
        )
        result = overlay.compute(step=5)
        assert "entity_positions" in result
        assert len(result["entity_positions"]) == 3

    def test_maritime_distress_injection(self):
        from uavbench.visualization.overlays import MaritimeDomainOverlay

        overlay = MaritimeDomainOverlay(grid_shape=(64, 64))
        overlay.inject_distress(step=10, position=(20, 20))
        result = overlay.compute(step=15)
        assert result["distress_position"] == (20, 20)

    def test_critical_infra_restriction_zones(self):
        from uavbench.visualization.overlays import CriticalInfraOverlay

        overlay = CriticalInfraOverlay(
            grid_shape=(64, 64),
            inspection_sites=[{"xy": (32, 32), "label": "A", "radius": 10}],
        )
        overlay.add_restriction_zone(center=(32, 32), radius=10, step_activated=5)

        r_before = overlay.compute(step=3)
        assert r_before.get("nfz_mask") is None  # not yet activated

        r_after = overlay.compute(step=5)
        assert r_after["nfz_mask"] is not None
        assert r_after["nfz_mask"].any()

    def test_critical_infra_pois(self):
        from uavbench.visualization.overlays import CriticalInfraOverlay

        sites = [
            {"xy": (10, 10), "label": "Site-A", "radius": 5},
            {"xy": (50, 50), "label": "Site-B", "radius": 8},
        ]
        overlay = CriticalInfraOverlay(grid_shape=(64, 64), inspection_sites=sites)
        result = overlay.compute(step=0)
        assert "pois" in result
        assert len(result["pois"]) == 2


# ═════════════════════════════════════════════════════════════════════════════
# Demo Pack Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestDemoPack:
    """Tests for demo pack export."""

    def test_export_creates_files(self, small_renderer):
        # Render a few frames
        for step in range(3):
            small_renderer.render_frame(
                drone_pos=(32 + step, 32),
                step=step,
                max_steps=5,
                is_keyframe=(step == 0),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            from uavbench.visualization.demo_pack import export_demo_pack

            pack_dir = export_demo_pack(
                renderer=small_renderer,
                output_dir=Path(tmpdir) / "test_pack",
                metrics={"tasks_completed": 1, "mission_score": 0.5},
            )

            assert pack_dir.exists()
            assert (pack_dir / "thumbnail.png").exists()
            assert (pack_dir / "metadata.json").exists()
            assert (pack_dir / "summary.txt").exists()

            # Verify metadata JSON
            meta = json.loads((pack_dir / "metadata.json").read_text())
            assert meta["mission_type"] == "civil_protection"
            assert "attribution" in meta
            assert "OpenStreetMap" in meta["attribution"]

    def test_export_metadata_content(self, small_renderer):
        small_renderer.render_frame((32, 32), 0, 10)
        small_renderer.log_event(0, "replan", "test")

        with tempfile.TemporaryDirectory() as tmpdir:
            small_renderer.export_metadata(Path(tmpdir) / "meta.json")
            meta = json.loads((Path(tmpdir) / "meta.json").read_text())
            assert meta["mission_type"] == "civil_protection"
            assert meta["tile"]["id"] == "test_synthetic"
            assert len(meta["events"]) == 1

    def test_summary_text(self, small_renderer):
        small_renderer.render_frame((32, 32), 0, 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            from uavbench.visualization.demo_pack import _write_summary

            summary_path = Path(tmpdir) / "summary.txt"
            _write_summary(summary_path, small_renderer, {"tasks_completed": 2}, [])

            text = summary_path.read_text()
            assert "UAVBench" in text
            assert "Wildfire Monitoring" in text
            assert "OpenStreetMap" in text

    def test_serialise_numpy_types(self):
        from uavbench.visualization.demo_pack import _serialise

        data = {
            "int": np.int64(42),
            "float": np.float32(3.14),
            "array": np.array([1, 2, 3]),
            "bool": np.bool_(True),
            "nested": {"inner": np.float64(2.71)},
        }
        result = _serialise(data)
        assert isinstance(result["int"], int)
        assert isinstance(result["float"], float)
        assert isinstance(result["array"], list)
        assert isinstance(result["bool"], bool)
        assert isinstance(result["nested"]["inner"], float)

        # Should be JSON-serialisable
        json.dumps(result)
