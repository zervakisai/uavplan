from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

GridPos = Tuple[int, int]  # (x, y)


def play_path_window(
    heightmap: np.ndarray,
    no_fly: np.ndarray,
    start: GridPos,
    goal: GridPos,
    path: List[GridPos],
    *,
    title: str = "UAVBench Path Player",
    fps: int = 8,
) -> None:
    """Open a window and play the path as a live animation with interactive drawing."""
    if not path:
        raise ValueError("Empty path: nothing to play.")

    H, W = heightmap.shape

    # Ensure interactive backend is set
    try:
        if sys.platform == "darwin":
            matplotlib.use('TkAgg')
        elif sys.platform.startswith("linux"):
            matplotlib.use('TkAgg')
        elif sys.platform == "win32":
            matplotlib.use('TkAgg')
    except Exception:
        pass

    print(f"\n[VIDEO] Opening animation window...")
    print(f"  Scenario size: {W}x{H}")
    print(f"  Path length: {len(path)} steps")
    print(f"  FPS: {fps}")
    print(f"  Backend: {matplotlib.get_backend()}")

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title("UAVBench Path Animation")

    # Background layers
    ax.imshow(heightmap > 0, cmap="gray", alpha=0.8)
    ax.imshow(no_fly, cmap="Reds", alpha=0.5)

    # Start/Goal markers
    ax.scatter(*start, c="green", s=120, label="Start", zorder=5, edgecolors='darkgreen', linewidth=2)
    ax.scatter(*goal, c="gold", s=150, marker="*", label="Goal", zorder=5, edgecolors='orange', linewidth=1)

    # Animated line
    (line,) = ax.plot([], [], linewidth=3, color="blue", zorder=4, marker='o', markersize=4)

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(False)

    print(f"  Close the window to continue...\n")

    # Frame-by-frame animation using interactive drawing
    interval_ms = int(1000 / max(1, fps))
    
    try:
        for i in range(len(path)):
            # Draw path up to current position
            xs = [p[0] for p in path[:i + 1]]
            ys = [p[1] for p in path[:i + 1]]
            line.set_data(xs, ys)
            
            # Update the display
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            
            # Sleep for the frame duration
            time.sleep(interval_ms / 1000.0)
            
            # Check if window was closed
            if not plt.fignum_exists(fig.number):
                break
        
        # Keep window open for final frame
        print("[VIDEO] Animation finished. Keeping window open...")
        plt.show(block=True)
        
    except KeyboardInterrupt:
        print("[VIDEO] Animation interrupted by user.")
    except Exception as e:
        print(f"[WARNING] Animation error: {e}")
    finally:
        plt.close(fig)


def save_path_video(
    heightmap: np.ndarray,
    no_fly: np.ndarray,
    start: GridPos,
    goal: GridPos,
    path: List[GridPos],
    output_path: str | Path,
    *,
    title: str = "UAVBench Path",
    fps: int = 8,
    dpi: int = 100,
) -> None:
    """Save an animated path visualization as a video or GIF file.
    
    Attempts to save as MP4 first (requires ffmpeg), falls back to GIF (Pillow).
    
    Args:
        heightmap: 2D array of building heights
        no_fly: 2D boolean array of no-fly zones
        start: (x, y) starting position
        goal: (x, y) goal position
        path: List of (x, y) positions along the path
        output_path: Path where to save the video/GIF
        title: Title for the animation
        fps: Frames per second for the video
        dpi: Resolution (dots per inch) for the output video
    """
    if not path:
        raise ValueError("Empty path: nothing to save.")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    H, W = heightmap.shape

    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)

    # Background layers
    ax.imshow(heightmap > 0, cmap="gray", alpha=0.8)
    ax.imshow(no_fly, cmap="Reds", alpha=0.5)

    # Start/Goal markers
    ax.scatter(*start, c="green", s=90, label="Start")
    ax.scatter(*goal, c="gold", s=120, marker="*", label="Goal")

    # Animated line
    (line,) = ax.plot([], [], linewidth=2, color="blue")

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc="upper right")

    interval_ms = int(1000 / max(1, fps))

    def init():
        line.set_data([], [])
        return (line,)

    def update(i: int):
        xs = [p[0] for p in path[: i + 1]]
        ys = [p[1] for p in path[: i + 1]]
        line.set_data(xs, ys)
        return (line,)

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(path),
        interval=interval_ms,
        blit=True,
        repeat=False,
    )

    # Try to save as MP4 first (if ffmpeg available)
    if str(output_path).endswith(".mp4"):
        try:
            anim.save(str(output_path), writer="ffmpeg", fps=fps, dpi=dpi)
            print(f"✓ Video saved: {output_path}")
            plt.close(fig)
            return
        except Exception as e:
            print(f"⚠ MP4 save failed (ffmpeg not installed): {e}")
            # Fall back to GIF
            gif_path = str(output_path).replace(".mp4", ".gif")
            print(f"  Falling back to GIF format: {gif_path}")
            output_path = Path(gif_path)

    # Save as GIF (works with Pillow, no ffmpeg required)
    try:
        anim.save(str(output_path), writer="pillow", fps=fps, dpi=dpi)
        print(f"✓ Animation saved: {output_path}")
    except Exception as e:
        print(f"✗ Failed to save animation: {e}")
        print("  Ensure Pillow is installed: pip install Pillow")
    finally:
        plt.close(fig)
