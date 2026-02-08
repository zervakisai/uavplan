"""Allow running as: python -m tools.osm_pipeline [subcommand].

Subcommands: fetch, rasterize.
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python -m tools.osm_pipeline <command> [options]")
        print()
        print("Commands:")
        print("  fetch      Download OSM data for Athens tiles")
        print("  rasterize  Convert .geojson layers to numpy arrays")
        print()
        print("Examples:")
        print("  python -m tools.osm_pipeline fetch --list")
        print("  python -m tools.osm_pipeline fetch --tile downtown --output data/maps/")
        print("  python -m tools.osm_pipeline rasterize --tile downtown")
        return

    cmd = sys.argv[1]
    # Remove the subcommand from argv so argparse in submodules works
    sys.argv = [f"tools.osm_pipeline.{cmd}"] + sys.argv[2:]

    if cmd == "fetch":
        from tools.osm_pipeline.fetch import main as fetch_main
        fetch_main()
    elif cmd == "rasterize":
        from tools.osm_pipeline.rasterize import main as rasterize_main
        rasterize_main()
    else:
        print(f"ERROR: Unknown command '{cmd}'. Use --help.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
