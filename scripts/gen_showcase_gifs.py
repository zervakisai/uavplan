#!/usr/bin/env python3
"""Generate 4 showcase GIFs for FLARE GitHub README (paper-figure style).

Each GIF tells one visual story about FLARE's capabilities:
  1. "The Fire Closes In"    — fire blocks corridor, planner reroutes
  2. "Who Gets Saved?"       — triage 3 casualties, some rescued, some not
  3. "The Collapse Trap"     — building collapse spawns debris, blocks route
  4. "Navigator vs Rescuer"  — side-by-side ranking inversion

Every frame is rendered through `PaperFrameRenderer` so the GIFs look
identical in style to the static paper figures.

Usage:
    python scripts/gen_showcase_gifs.py [--only 1] [--fps 15] [--skip 2]

Outputs:
    outputs/gifs/01_fire_closes_in.gif
    outputs/gifs/02_who_gets_saved.gif
    outputs/gifs/03_collapse_trap.gif
    outputs/gifs/04_navigator_vs_rescuer.gif
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from flare.benchmark.runner import run_episode
from flare.scenarios.loader import load_scenario
from flare.visualization.paper_frame import (
    Annotation,
    EpisodeCapture,
    PanelSpec,
    PaperFrameRenderer,
    growing_trajectory,
    make_capture_callback,
    pick_dyn_snapshot,
)

try:
    from PIL import Image
    _pil_ok = True
except ImportError:
    _pil_ok = False

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "gifs"

SCENARIO_PENTELI = "osm_penteli_pharma_delivery_medium"
SCENARIO_DOWNTOWN = "osm_downtown_fire_surveillance_medium"
SEED_PAPER = 42
SEED_INVERSION = 11


# ── Episode capture helper ─────────────────────────────────────────────────
def _capture_episode(
    scenario_id: str,
    planner_id: str,
    seed: int,
) -> tuple[EpisodeCapture, dict]:
    capture = EpisodeCapture()
    cb = make_capture_callback(capture, record_dyn=True)

    print(f"  Running {planner_id} on {scenario_id} seed={seed} ...")
    t0 = time.perf_counter()
    result = run_episode(scenario_id, planner_id, seed, frame_callback=cb)
    elapsed = time.perf_counter() - t0
    m = result.metrics
    status = "OK" if m.get("success") else m.get("termination_reason", "?")
    print(
        f"    [{status}] steps={m.get('executed_steps_len', 0)} "
        f"score={m.get('mission_score', 0):.3f} "
        f"tasks={m.get('tasks_completed', 0)}/{m.get('tasks_total', '?')} "
        f"replans={m.get('replans', 0)} ({elapsed:.1f}s)"
    )
    return capture, m


def _render_story_frames(
    capture: EpisodeCapture,
    config,
    planner_id: str,
    skip: int,
    suptitle_fmt: str,
    annotations_for_step=None,
) -> list[np.ndarray]:
    """Render paper-figure style frames from a captured episode."""
    pr = PaperFrameRenderer(config)
    total = len(capture.trajectory)
    goal_xy = tuple(capture.state0.get("goal_xy", (0, 0)))
    frames: list[np.ndarray] = []
    if total == 0:
        return frames

    for t in range(0, total, max(1, skip)):
        finished = (t >= total - 1)
        panel = PanelSpec(
            planner_id=planner_id,
            trajectory=growing_trajectory(capture.trajectory, t),
            goal_xy=goal_xy,
            success=True if finished else None,
        )
        anns = annotations_for_step(t) if annotations_for_step else None
        frames.append(
            pr.render_frame(
                capture.heightmap,
                capture.state0,
                pick_dyn_snapshot(capture.dyn_snapshots, t),
                panels=[panel],
                suptitle=suptitle_fmt.format(step=t),
                annotations=anns,
            )
        )
    return frames


def save_gif(
    frames: list[np.ndarray],
    path: Path,
    fps: int = 15,
    hold_last_s: float = 3.0,
    max_frames: int = 180,
) -> float:
    """Save frames as looping GIF, optionally optimize with gifsicle."""
    if not frames:
        print(f"  WARNING: No frames for {path}")
        return 0.0

    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in indices]

    hold_n = int(fps * hold_last_s)
    out = list(frames) + [frames[-1]] * hold_n

    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = 1000 / fps

    if _pil_ok:
        pil_frames = [Image.fromarray(f).quantize(colors=192, method=2) for f in out]
        pil_frames[0].save(
            str(path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(duration_ms),
            loop=0,
            optimize=True,
        )
    else:
        iio.imwrite(str(path), out, extension=".gif",
                    duration=duration_ms, loop=0)

    import shutil
    import subprocess

    if shutil.which("gifsicle"):
        tmp = str(path) + ".tmp"
        subprocess.run(
            ["gifsicle", "-O3", "--lossy=30", "-o", tmp, str(path)],
            capture_output=True,
        )
        if os.path.exists(tmp) and os.path.getsize(tmp) < os.path.getsize(path):
            os.replace(tmp, str(path))
        elif os.path.exists(tmp):
            os.remove(tmp)

    size_mb = os.path.getsize(path) / 1e6
    print(
        f"  Saved {path.name}: {len(out)} frames "
        f"({len(frames)} content + {hold_n} hold), {size_mb:.1f} MB"
    )
    return size_mb


# ═══════════════════════════════════════════════════════════════════════════
# GIF 1: The Fire Closes In
# ═══════════════════════════════════════════════════════════════════════════
def gif1_fire_closes_in(fps: int = 15, skip: int = 3) -> None:
    print("\n" + "=" * 60)
    print("GIF 1: THE FIRE CLOSES IN")
    print("=" * 60)

    config = load_scenario(SCENARIO_PENTELI)
    capture, _m = _capture_episode(SCENARIO_PENTELI, "periodic_replan", SEED_PAPER)
    frames = _render_story_frames(
        capture, config, "periodic_replan", skip,
        suptitle_fmt="The fire closes in · step {step}",
    )
    save_gif(frames, OUTPUT_DIR / "01_fire_closes_in.gif", fps=fps)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 2: Who Gets Saved?
# ═══════════════════════════════════════════════════════════════════════════
def gif2_who_gets_saved(fps: int = 15, skip: int = 3) -> None:
    print("\n" + "=" * 60)
    print("GIF 2: WHO GETS SAVED?")
    print("=" * 60)

    config = load_scenario(SCENARIO_DOWNTOWN)
    capture, _m = _capture_episode(
        SCENARIO_DOWNTOWN, "aggressive_replan", SEED_INVERSION,
    )
    frames = _render_story_frames(
        capture, config, "aggressive_replan", skip,
        suptitle_fmt="Who gets saved? · step {step}",
    )
    save_gif(frames, OUTPUT_DIR / "02_who_gets_saved.gif", fps=fps)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 3: The Collapse Trap
# ═══════════════════════════════════════════════════════════════════════════
def gif3_collapse_trap(fps: int = 15, skip: int = 3) -> None:
    print("\n" + "=" * 60)
    print("GIF 3: THE COLLAPSE TRAP")
    print("=" * 60)

    scenario = SCENARIO_DOWNTOWN
    seed = 44
    config = load_scenario(scenario)
    capture, _m = _capture_episode(scenario, "aggressive_replan", seed)

    # Pre-compute debris counts for annotation
    def ann_for(step):
        dyn = pick_dyn_snapshot(capture.dyn_snapshots, step)
        debris = dyn.get("debris_mask")
        n = int(np.count_nonzero(debris)) if debris is not None else 0
        return [Annotation(
            text=f"Step {step} · Debris: {n} cells",
            xy=(0.02, 0.97), fontsize=7, ha="left", va="top",
        )]

    frames = _render_story_frames(
        capture, config, "aggressive_replan", skip,
        suptitle_fmt="The collapse trap",
        annotations_for_step=ann_for,
    )
    save_gif(frames, OUTPUT_DIR / "03_collapse_trap.gif", fps=fps)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 4: Navigator vs Rescuer (2-panel figure-style frame)
# ═══════════════════════════════════════════════════════════════════════════
def gif4_navigator_vs_rescuer(fps: int = 15, skip: int = 3) -> None:
    print("\n" + "=" * 60)
    print("GIF 4: NAVIGATOR vs RESCUER")
    print("=" * 60)

    scenario = SCENARIO_DOWNTOWN
    config = load_scenario(scenario)

    cap_nav, m_nav = _capture_episode(scenario, "incremental_astar", SEED_INVERSION)
    cap_res, m_res = _capture_episode(scenario, "aggressive_replan", SEED_INVERSION)

    # Use nav's basemap/state0 (same scenario/seed → same map)
    pr = PaperFrameRenderer(config)
    goal_nav = tuple(cap_nav.state0.get("goal_xy", (0, 0)))
    goal_res = tuple(cap_res.state0.get("goal_xy", (0, 0)))

    total_nav = len(cap_nav.trajectory)
    total_res = len(cap_res.trajectory)
    total = max(total_nav, total_res)
    if total == 0:
        print("  ERROR: No frames captured")
        return

    def clamp_step(t, n):
        return min(t, max(0, n - 1))

    nav_score = m_nav.get("mission_score", 0)
    nav_done = m_nav.get("tasks_completed", 0)
    nav_total = m_nav.get("tasks_total", "?")
    res_score = m_res.get("mission_score", 0)
    res_done = m_res.get("tasks_completed", 0)
    res_total = m_res.get("tasks_total", "?")

    frames: list[np.ndarray] = []
    for t in range(0, total, max(1, skip)):
        t_nav = clamp_step(t, total_nav)
        t_res = clamp_step(t, total_res)

        nav_finished = (t_nav >= total_nav - 1)
        res_finished = (t_res >= total_res - 1)

        nav_panel = PanelSpec(
            planner_id="incremental_astar",
            trajectory=growing_trajectory(cap_nav.trajectory, t_nav),
            goal_xy=goal_nav,
            success=True if nav_finished else None,
            title_override="Incr. A* (Navigator)",
            subtitle_override=f"t={t_nav} · score {nav_score:.2f} · {nav_done}/{nav_total}",
        )
        res_panel = PanelSpec(
            planner_id="aggressive_replan",
            trajectory=growing_trajectory(cap_res.trajectory, t_res),
            goal_xy=goal_res,
            success=True if res_finished else None,
            title_override="Aggressive (Rescuer)",
            subtitle_override=f"t={t_res} · score {res_score:.2f} · {res_done}/{res_total}",
        )

        # Use nav's capture as the basemap / dyn source (FD-4: fire is
        # agent-independent, so both planners see the same fire timeline).
        frames.append(
            pr.render_frame(
                cap_nav.heightmap,
                cap_nav.state0,
                pick_dyn_snapshot(cap_nav.dyn_snapshots, t_nav),
                panels=[nav_panel, res_panel],
                suptitle=f"Best navigator ≠ best rescuer · step {t}",
            )
        )

    # Final emphasis frame
    if frames:
        final_panel_nav = PanelSpec(
            planner_id="incremental_astar",
            trajectory=list(cap_nav.trajectory),
            goal_xy=goal_nav,
            success=True,
            title_override="Incr. A* (Navigator)",
            subtitle_override=f"score {nav_score:.2f} · {nav_done}/{nav_total}",
        )
        final_panel_res = PanelSpec(
            planner_id="aggressive_replan",
            trajectory=list(cap_res.trajectory),
            goal_xy=goal_res,
            success=True,
            title_override="Aggressive (Rescuer)",
            subtitle_override=f"score {res_score:.2f} · {res_done}/{res_total}",
        )
        final = pr.render_frame(
            cap_nav.heightmap,
            cap_nav.state0,
            pick_dyn_snapshot(cap_nav.dyn_snapshots, total_nav - 1),
            panels=[final_panel_nav, final_panel_res],
            suptitle="Best navigator ≠ best rescuer",
            annotations=[
                Annotation(
                    text="Best navigator ≠ best rescuer",
                    xy=(0.5, 0.08), fontsize=10, ha="center", va="bottom",
                ),
            ],
        )
        frames.extend([final] * int(fps * 2))

    save_gif(
        frames,
        OUTPUT_DIR / "04_navigator_vs_rescuer.gif",
        fps=fps,
        hold_last_s=0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FLARE showcase GIFs")
    parser.add_argument("--only", type=int, choices=[1, 2, 3, 4],
                        help="Generate only this GIF")
    parser.add_argument("--fps", type=int, default=15, help="GIF frame rate")
    parser.add_argument("--skip", type=int, default=3,
                        help="Render every Nth step")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    generators = {
        1: gif1_fire_closes_in,
        2: gif2_who_gets_saved,
        3: gif3_collapse_trap,
        4: gif4_navigator_vs_rescuer,
    }

    targets = [args.only] if args.only else [1, 2, 3, 4]
    for n in targets:
        try:
            generators[n](fps=args.fps, skip=args.skip)
        except Exception as e:
            print(f"\n  ERROR generating GIF {n}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.0f}s  ->  {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
