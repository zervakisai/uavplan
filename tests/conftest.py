"""Pytest configuration for UAVBench test suite.

Layout
------
tests/unit/          Pure logic, no UrbanEnv, ≤ 64×64 grids.  <5 s total.
tests/integration/   Small synthetic envs (32×32), contracts.  <15 s total.
tests/benchmark/     Full 500×500 OSM runs.  Minutes-to-hours.  Skipped by default.

Run fast tests only:   pytest tests/unit tests/integration
Run everything:        pytest tests/ --run-slow
Run benchmarks only:   pytest tests/benchmark --run-slow
"""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow benchmark tests (500×500 OSM, full episodes).",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (500×500 OSM)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return  # run everything
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
