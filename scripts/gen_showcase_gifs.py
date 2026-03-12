#!/usr/bin/env python3
"""Generate 4 showcase GIFs for FLARE GitHub README.

Each GIF tells one visual story about FLARE's capabilities:
  1. "The Fire Closes In"    — fire blocks corridor, planner reroutes
  2. "Who Gets Saved?"       — triage 3 casualties, some rescued, some not
  3. "The Collapse Trap"     — building collapse spawns debris, blocks route
  4. "Navigator vs Rescuer"  — side-by-side ranking inversion

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

from uavbench.benchmark.runner import run_episode
from uavbench.scenarios.loader import load_scenario
from uavbench.visualization.renderer import Renderer

# ── Try PIL for text overlays ──────────────────────────────────────────────
_pil_ok = False
try:
    from PIL import Image, ImageDraw, ImageFont

    _pil_ok = True
except ImportError:
    pass

# ── Config ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "gifs"

SCENARIO_PENTELI = "osm_penteli_pharma_delivery_medium"
SCENARIO_DOWNTOWN = "osm_downtown_fire_surveillance_medium"
SEED_PAPER = 42
SEED_INVERSION = 11

# ── Font loading ───────────────────────────────────────────────────────────
_font_cache: dict[int, object] = {}
_font_path: str | None = None

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "DejaVuSansMono.ttf",
]


def _get_font(size: int = 14):
    global _font_path
    if not _pil_ok:
        return None
    if size in _font_cache:
        return _font_cache[size]
    if _font_path is not None:
        try:
            f = ImageFont.truetype(_font_path, size)
            _font_cache[size] = f
            return f
        except (OSError, IOError):
            _font_path = None
    for cand in _FONT_CANDIDATES:
        try:
            f = ImageFont.truetype(cand, size)
            _font_path = cand
            _font_cache[size] = f
            return f
        except (OSError, IOError):
            continue
    f = ImageFont.load_default()
    _font_cache[size] = f
    return f


# ── Text overlay ───────────────────────────────────────────────────────────
def overlay_text(
    frame: np.ndarray,
    text: str,
    xy: tuple[int, int],
    font_size: int = 14,
    color: tuple[int, ...] = (255, 255, 255),
    bg_alpha: int = 180,
    center: bool = False,
) -> np.ndarray:
    """Draw text with semi-transparent dark background on frame."""
    if not _pil_ok:
        return frame
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _get_font(font_size)

    bbox = draw.multiline_textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if center:
        x = xy[0] - tw // 2
        y = xy[1] - th // 2
    else:
        x, y = xy

    pad = 6
    draw.rectangle(
        [x - pad, y - pad, x + tw + pad, y + th + pad],
        fill=(10, 15, 26, bg_alpha),
    )
    fill = color if len(color) == 4 else color + (255,)
    draw.multiline_text((x, y), text, font=font, fill=fill)

    img = Image.alpha_composite(img, overlay)
    return np.array(img.convert("RGB"))


# ── Frame utilities ────────────────────────────────────────────────────────
def downscale(frame: np.ndarray, max_w: int) -> np.ndarray:
    """Resize frame so width <= max_w, preserving aspect ratio."""
    if not _pil_ok:
        return frame
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    scale = max_w / w
    new_w, new_h = int(w * scale), int(h * scale)
    img = Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS)
    return np.array(img)


def resize_to(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize frame to exact dimensions."""
    if not _pil_ok:
        return frame
    img = Image.fromarray(frame).resize((target_w, target_h), Image.LANCZOS)
    return np.array(img)


def save_gif(
    frames: list[np.ndarray],
    path: Path,
    fps: int = 15,
    hold_last_s: float = 3.0,
    max_frames: int = 180,
) -> float:
    """Save frames as looping GIF, optimized with gifsicle. Returns size in MB."""
    if not frames:
        print(f"  WARNING: No frames for {path}")
        return 0.0

    # Subsample if too many content frames
    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in indices]

    # Hold last frame
    hold_n = int(fps * hold_last_s)
    out = list(frames) + [frames[-1]] * hold_n

    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = 1000 / fps

    # Use Pillow encoder for better palette control
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
        iio.imwrite(str(path), out, extension=".gif", duration=duration_ms, loop=0)

    # Optimize with gifsicle if available
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


# ── Episode runner ─────────────────────────────────────────────────────────
def run_and_capture(
    scenario_id: str,
    planner_id: str,
    seed: int,
    skip: int = 2,
    extra_cb=None,
) -> tuple[list[np.ndarray], object]:
    """Run episode, capture every skip-th rendered frame.

    extra_cb(heightmap, state, dyn_state, cfg, step_counter) is called
    at EVERY step for metadata tracking (even if frame is skipped).
    """
    config = load_scenario(scenario_id)
    renderer = Renderer(config, mode="ops_full")
    frames: list[np.ndarray] = []
    counter = [0]

    def cb(heightmap, state, dyn_state, cfg):
        if extra_cb:
            extra_cb(heightmap, state, dyn_state, cfg, counter[0])
        if counter[0] % skip == 0:
            frame, _ = renderer.render_frame(heightmap, state, dyn_state)
            frames.append(frame)
        counter[0] += 1

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
        f"replans={m.get('replans', 0)} ({elapsed:.1f}s) "
        f"-> {len(frames)} frames captured"
    )
    return frames, result


# ═══════════════════════════════════════════════════════════════════════════
# GIF 1: The Fire Closes In (Penteli, periodic_replan)
# ═══════════════════════════════════════════════════════════════════════════
def gif1_fire_closes_in(fps: int = 15, skip: int = 2) -> None:
    """Fire blocks corridor, planner reroutes around it."""
    print("\n" + "=" * 60)
    print("GIF 1: THE FIRE CLOSES IN")
    print("=" * 60)

    frames, result = run_and_capture(
        SCENARIO_PENTELI, "periodic_replan", SEED_PAPER, skip=skip,
    )
    frames = [downscale(f, 600) for f in frames]
    save_gif(frames, OUTPUT_DIR / "01_fire_closes_in.gif", fps=fps)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 2: Who Gets Saved? (Piraeus, aggressive_replan)
# ═══════════════════════════════════════════════════════════════════════════
def gif2_who_gets_saved(fps: int = 15, skip: int = 2) -> None:
    """Drone surveys 3 fire hotspots — completes all tasks."""
    print("\n" + "=" * 60)
    print("GIF 2: WHO GETS SAVED?")
    print("=" * 60)

    frames, result = run_and_capture(
        SCENARIO_DOWNTOWN, "aggressive_replan", SEED_INVERSION, skip=skip,
    )
    frames = [downscale(f, 600) for f in frames]
    save_gif(frames, OUTPUT_DIR / "02_who_gets_saved.gif", fps=fps)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 3: The Collapse Trap (Penteli, aggressive_replan, zoomed)
# ═══════════════════════════════════════════════════════════════════════════
def gif3_collapse_trap(fps: int = 15, skip: int = 2) -> None:
    """Fire spreads, buildings collapse into permanent debris."""
    print("\n" + "=" * 60)
    print("GIF 3: THE COLLAPSE TRAP")
    print("=" * 60)

    # Downtown seed 44: debris_caught termination — drone trapped by collapse
    scenario = SCENARIO_DOWNTOWN
    seed = 44
    config = load_scenario(scenario)
    renderer = Renderer(config, mode="ops_full")

    frames: list[np.ndarray] = []
    debris_counts: list[int] = []
    step_indices: list[int] = []
    counter = [0]

    def cb(heightmap, state, dyn_state, cfg):
        debris = dyn_state.get("debris_mask")
        n = int(np.count_nonzero(debris)) if debris is not None else 0

        if counter[0] % skip == 0:
            # Render WITHOUT risk heatmap so fire/debris are clearly distinct
            state_noheat = dict(state)
            state_noheat["cost_map"] = None
            frame, _ = renderer.render_frame(heightmap, state_noheat, dyn_state)
            frames.append(frame)
            debris_counts.append(n)
            step_indices.append(state.get("step_idx", counter[0]))
        counter[0] += 1

    print(f"  Running aggressive_replan on {scenario} seed={seed} ...")
    t0 = time.perf_counter()
    result = run_episode(
        scenario, "aggressive_replan", seed, frame_callback=cb,
    )
    elapsed = time.perf_counter() - t0
    m = result.metrics
    print(
        f"    steps={m.get('executed_steps_len', 0)} "
        f"score={m.get('mission_score', 0):.3f} ({elapsed:.1f}s) "
        f"-> {len(frames)} frames captured"
    )

    # ── Overlay debris counter ─────────────────────────────────────────────
    for i in range(len(frames)):
        count = debris_counts[i] if i < len(debris_counts) else 0
        step = step_indices[i] if i < len(step_indices) else 0
        frames[i] = overlay_text(
            frames[i],
            f"Step {step} | Debris: {count} cells",
            (8, 8),
            font_size=12,
            color=(220, 180, 120),
        )

    frames = [downscale(f, 600) for f in frames]
    save_gif(frames, OUTPUT_DIR / "03_collapse_trap.gif", fps=fps)


# ═══════════════════════════════════════════════════════════════════════════
# GIF 4: Best Navigator != Best Rescuer (side-by-side)
# ═══════════════════════════════════════════════════════════════════════════
def gif4_navigator_vs_rescuer(fps: int = 15, skip: int = 2) -> None:
    """Side-by-side: Incr.A* reaches goal fast but rescues nobody;
    Aggressive is slower but rescues 2/3 casualties."""
    print("\n" + "=" * 60)
    print("GIF 4: NAVIGATOR vs RESCUER")
    print("=" * 60)

    # Run both episodes
    frames_nav, result_nav = run_and_capture(
        SCENARIO_DOWNTOWN, "incremental_astar", SEED_INVERSION, skip=skip,
    )
    frames_res, result_res = run_and_capture(
        SCENARIO_DOWNTOWN, "aggressive_replan", SEED_INVERSION, skip=skip,
    )
    m_nav = result_nav.metrics
    m_res = result_res.metrics

    # Downscale both panels to same width
    panel_w = 480
    frames_nav = [downscale(f, panel_w) for f in frames_nav]
    frames_res = [downscale(f, panel_w) for f in frames_res]

    if not frames_nav or not frames_res:
        print("  ERROR: No frames captured")
        return

    # Ensure identical dimensions for both panels
    h_nav, w_nav = frames_nav[0].shape[:2]
    h_res, w_res = frames_res[0].shape[:2]
    target_h = max(h_nav, h_res)
    target_w = max(w_nav, w_res)
    frames_nav = [resize_to(f, target_w, target_h) for f in frames_nav]
    frames_res = [resize_to(f, target_w, target_h) for f in frames_res]

    # ── Pad shorter episode with frozen "GOAL REACHED" frame ───────────
    max_len = max(len(frames_nav), len(frames_res))

    nav_score = m_nav.get("mission_score", 0)
    nav_tasks_done = m_nav.get("tasks_completed", 0)
    nav_tasks_total = m_nav.get("tasks_total", "?")

    res_score = m_res.get("mission_score", 0)
    res_tasks_done = m_res.get("tasks_completed", 0)
    res_tasks_total = m_res.get("tasks_total", "?")

    def make_frozen(last_frame, score, tasks_done, tasks_total):
        frozen = last_frame.copy()
        frozen = overlay_text(
            frozen,
            f"GOAL REACHED\nScore: {score:.2f}\nTasks: {tasks_done}/{tasks_total}",
            (frozen.shape[1] // 2, frozen.shape[0] // 2 - 20),
            font_size=16,
            color=(80, 255, 120),
            bg_alpha=200,
            center=True,
        )
        return frozen

    if len(frames_nav) < max_len:
        frozen = make_frozen(frames_nav[-1], nav_score, nav_tasks_done, nav_tasks_total)
        frames_nav.extend([frozen] * (max_len - len(frames_nav)))

    if len(frames_res) < max_len:
        frozen = make_frozen(frames_res[-1], res_score, res_tasks_done, res_tasks_total)
        frames_res.extend([frozen] * (max_len - len(frames_res)))

    # ── Compose side-by-side ───────────────────────────────────────────────
    divider_w = 3
    composed: list[np.ndarray] = []

    for f_left, f_right in zip(frames_nav, frames_res):
        divider = np.full((target_h, divider_w, 3), 200, dtype=np.uint8)
        combined = np.hstack([f_left, divider, f_right])

        # Panel titles at top
        combined = overlay_text(
            combined,
            "Incr. A*  (Navigator)",
            (10, 4),
            font_size=11,
            color=(180, 180, 255),
            bg_alpha=200,
        )
        combined = overlay_text(
            combined,
            "Aggressive  (Rescuer)",
            (target_w + divider_w + 10, 4),
            font_size=11,
            color=(255, 180, 180),
            bg_alpha=200,
        )
        composed.append(combined)

    # ── Final comparison frame (hold extra) ────────────────────────────────
    if composed:
        final = composed[-1].copy()
        total_w = final.shape[1]

        final = overlay_text(
            final,
            "Best navigator != Best rescuer",
            (total_w // 2, final.shape[0] // 2 - 15),
            font_size=20,
            color=(255, 255, 80),
            bg_alpha=210,
            center=True,
        )
        final = overlay_text(
            final,
            f"Score {nav_score:.2f} | Tasks {nav_tasks_done}/{nav_tasks_total}",
            (target_w // 2, final.shape[0] // 2 + 25),
            font_size=12,
            color=(180, 180, 255),
            bg_alpha=190,
            center=True,
        )
        final = overlay_text(
            final,
            f"Score {res_score:.2f} | Tasks {res_tasks_done}/{res_tasks_total}",
            (target_w + divider_w + target_w // 2, final.shape[0] // 2 + 25),
            font_size=12,
            color=(255, 180, 180),
            bg_alpha=190,
            center=True,
        )
        hold_final = int(fps * 4)
        composed.extend([final] * hold_final)

    save_gif(
        composed,
        OUTPUT_DIR / "04_navigator_vs_rescuer.gif",
        fps=fps,
        hold_last_s=0,  # already added hold via final frame
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FLARE showcase GIFs")
    parser.add_argument(
        "--only", type=int, choices=[1, 2, 3, 4], help="Generate only this GIF"
    )
    parser.add_argument("--fps", type=int, default=15, help="GIF frame rate")
    parser.add_argument(
        "--skip", type=int, default=3, help="Capture every Nth frame"
    )
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

    # README snippet
    print(f"\n{'─' * 60}")
    print("README snippet:\n")
    print(
        """\
## Demos

| Fire blocks corridor | Triage: who gets saved? |
|:---:|:---:|
| ![fire](outputs/gifs/01_fire_closes_in.gif) | ![triage](outputs/gifs/02_who_gets_saved.gif) |

| Collapse trap | Navigator != Rescuer |
|:---:|:---:|
| ![collapse](outputs/gifs/03_collapse_trap.gif) | ![inversion](outputs/gifs/04_navigator_vs_rescuer.gif) |
"""
    )


if __name__ == "__main__":
    main()
