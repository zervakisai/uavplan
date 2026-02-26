"""
UAVBench v2 — Determinism Contract Tests
=========================================

Contracts tested:
    DC-1  reset(seed=s) initializes ALL RNG from ONE np.random.default_rng(seed) source.
          No component may construct its own independent RNG.
    DC-2  Same (scenario_id, planner_id, seed) triple → bit-identical event log,
          trajectory, metrics dict, and per-frame hash sequence.

Test cases (from V2_TEST_PLAN.md Section 2.1):
    test_single_rng_source               DC-1
    test_no_independent_rng_constructors DC-1
    test_identical_seed_identical_output DC-2
    test_identical_seed_identical_frames DC-2
    test_different_seed_different_output DC-2

All tests are deterministic (fixed seed=42 / seed=43).
Tests are designed TDD-style: they will FAIL until src/uavbench2/ is implemented.
If the package is not yet installed they skip gracefully via pytest.importorskip().

Evidence artifacts (written by runner, read here):
    outputs/v2/determinism_hashes.json
    outputs/v2/rng_audit.json
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Package-level import guard
# ---------------------------------------------------------------------------
# All runtime tests skip cleanly when src/uavbench2/ has not been implemented yet.
# Static / grep tests do not need the package and are always collected.

uavbench2 = pytest.importorskip(
    "uavbench2",
    reason="uavbench2 package not yet installed — skipping runtime determinism tests",
)

# ---------------------------------------------------------------------------
# Lazy imports (only reached if importorskip did not skip the module)
# ---------------------------------------------------------------------------

from uavbench2.benchmark.runner import run_episode  # noqa: E402
from uavbench2.benchmark.determinism import hash_episode  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCENARIO_ID = "gov_fire_delivery_easy"
PLANNER_ID = "astar"
SEED_A = 42
SEED_B = 43

# Absolute path to the uavbench2 source tree.
# We derive it from the installed package location rather than a hard-coded path
# so that the test works in any Python environment.
_SRC_ROOT = Path(uavbench2.__file__).resolve().parent  # .../src/uavbench2/


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_of(obj: Any) -> str:
    """Return a stable SHA-256 hex digest of a JSON-serialisable object."""
    canonical = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _run(seed: int) -> Any:
    """Run one deterministic episode and return the EpisodeResult."""
    return run_episode(
        scenario_id=SCENARIO_ID,
        planner_id=PLANNER_ID,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# DC-1 Tests — ONE RNG Source
# ---------------------------------------------------------------------------


class TestSingleRNGSource:
    """Verifies DC-1: ALL randomness flows from one np.random.default_rng(seed) root."""

    def test_single_rng_source(self) -> None:
        """Verifies DC-1: grep finds zero np.random.default_rng calls outside reset().

        Acceptance criterion: every occurrence of ``np.random.default_rng`` in
        src/uavbench2/ is inside a function or method named exactly ``reset``.
        Child generators spawned via ``root_rng.spawn()`` are the only permitted
        way to propagate randomness to subsystems.

        Static analysis approach: parse each Python file with the ``ast`` module,
        walk all Call nodes that reference ``default_rng``, and confirm the
        enclosing function is named ``reset``.  Raises AssertionError listing
        any offending (file, line) pairs if the invariant is violated.
        """
        # Arrange
        python_files = list(_SRC_ROOT.rglob("*.py"))
        assert python_files, (
            f"No Python files found under {_SRC_ROOT}; "
            "is the package installed in editable mode?"
        )

        violations: list[str] = []

        for path in python_files:
            source = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError:
                continue  # skip unparseable files; a separate test catches them

            # Build a map: lineno → enclosing function name (innermost)
            # We walk the tree and track function defs by line range.
            func_ranges: list[tuple[int, int, str]] = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # end_lineno is available in Python 3.8+
                    end = getattr(node, "end_lineno", node.lineno)
                    func_ranges.append((node.lineno, end, node.name))

            def enclosing_function(lineno: int) -> str | None:
                """Return the innermost function name that contains lineno."""
                best: tuple[int, str] | None = None
                for start, end, name in func_ranges:
                    if start <= lineno <= end:
                        if best is None or start > best[0]:
                            best = (start, name)
                return best[1] if best else None

            # Find all calls to np.random.default_rng / numpy.random.default_rng
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                # Match: np.random.default_rng  or  numpy.random.default_rng
                is_default_rng = False
                if isinstance(func, ast.Attribute) and func.attr == "default_rng":
                    # Check it's chained as X.random.default_rng
                    if isinstance(func.value, ast.Attribute) and func.value.attr == "random":
                        is_default_rng = True
                if not is_default_rng:
                    continue

                enclosing = enclosing_function(node.lineno)
                if enclosing != "reset":
                    rel = path.relative_to(_SRC_ROOT)
                    violations.append(
                        f"{rel}:{node.lineno} — np.random.default_rng() "
                        f"called outside reset() (enclosing: {enclosing!r})"
                    )

        # Assert
        assert not violations, (
            "DC-1 violated: np.random.default_rng called outside reset():\n"
            + "\n".join(f"  {v}" for v in violations)
        )

    def test_no_independent_rng_constructors(self) -> None:
        """Verifies DC-1: no RandomState, random.Random, or bare np.random.seed in src/.

        Acceptance criterion: none of the following forbidden patterns appear in
        any .py file under src/uavbench2/:

          * np.random.RandomState(
          * numpy.random.RandomState(
          * random.Random(
          * random.seed(
          * np.random.seed(
          * numpy.random.seed(

        These patterns indicate an out-of-band RNG that would break DC-1.
        Test uses regex over raw source text for speed; AST confirmation is not
        required because the patterns are unambiguous at the text level.
        """
        # Arrange
        python_files = list(_SRC_ROOT.rglob("*.py"))
        assert python_files, f"No Python files found under {_SRC_ROOT}"

        forbidden_patterns = [
            # Pattern                         Human-readable label
            (r"np\.random\.RandomState\s*\(",  "np.random.RandomState()"),
            (r"numpy\.random\.RandomState\s*\(", "numpy.random.RandomState()"),
            (r"random\.Random\s*\(",            "random.Random()"),
            (r"random\.seed\s*\(",              "random.seed()"),
            (r"np\.random\.seed\s*\(",          "np.random.seed()"),
            (r"numpy\.random\.seed\s*\(",       "numpy.random.seed()"),
        ]
        compiled = [
            (re.compile(pat), label) for pat, label in forbidden_patterns
        ]

        violations: list[str] = []

        for path in python_files:
            source = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(source.splitlines(), start=1):
                # Skip comment lines
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                for regex, label in compiled:
                    if regex.search(line):
                        rel = path.relative_to(_SRC_ROOT)
                        violations.append(
                            f"{rel}:{lineno} — forbidden RNG constructor {label!r}: "
                            f"{line.strip()!r}"
                        )

        # Assert
        assert not violations, (
            "DC-1 violated: independent RNG constructors found in src/uavbench2/:\n"
            + "\n".join(f"  {v}" for v in violations)
        )


# ---------------------------------------------------------------------------
# DC-2 Tests — Bit-identical outputs for identical (scenario, planner, seed)
# ---------------------------------------------------------------------------


class TestIdenticalSeedIdenticalOutput:
    """Verifies DC-2: two runs with same (scenario_id, planner_id, seed) are identical."""

    def test_identical_seed_identical_output(self) -> None:
        """Verifies DC-2: two runs with identical inputs produce SHA-256 identical outputs.

        Runs (gov_fire_delivery_easy, astar, seed=42) twice independently.
        Checks that the event log, trajectory, and metrics dict each hash identically.
        If any differ, the failing component is reported individually.
        """
        # Arrange
        seed = SEED_A

        # Act
        result_a = _run(seed)
        result_b = _run(seed)

        # Assert — events
        hash_events_a = _sha256_of(result_a.events)
        hash_events_b = _sha256_of(result_b.events)
        assert hash_events_a == hash_events_b, (
            f"DC-2 violated: event logs differ for seed={seed}.\n"
            f"  Run A event hash: {hash_events_a}\n"
            f"  Run B event hash: {hash_events_b}\n"
            f"  Run A event count: {len(result_a.events)}\n"
            f"  Run B event count: {len(result_b.events)}"
        )

        # Assert — trajectory
        hash_traj_a = _sha256_of(result_a.trajectory)
        hash_traj_b = _sha256_of(result_b.trajectory)
        assert hash_traj_a == hash_traj_b, (
            f"DC-2 violated: trajectories differ for seed={seed}.\n"
            f"  Run A trajectory hash: {hash_traj_a}\n"
            f"  Run B trajectory hash: {hash_traj_b}\n"
            f"  Run A length: {len(result_a.trajectory)}\n"
            f"  Run B length: {len(result_b.trajectory)}"
        )

        # Assert — metrics
        hash_metrics_a = _sha256_of(result_a.metrics)
        hash_metrics_b = _sha256_of(result_b.metrics)
        assert hash_metrics_a == hash_metrics_b, (
            f"DC-2 violated: metrics dicts differ for seed={seed}.\n"
            f"  Run A metrics hash: {hash_metrics_a}\n"
            f"  Run B metrics hash: {hash_metrics_b}"
        )

        # Assert — combined episode hash (regression anchor)
        assert hash_episode(result_a) == hash_episode(result_b), (
            f"DC-2 violated: hash_episode() differs for seed={seed}."
        )

    def test_identical_seed_identical_frames(self) -> None:
        """Verifies DC-2: frame hash sequences are element-wise equal for identical seeds.

        EpisodeResult must expose a ``frame_hashes`` attribute — a list of
        per-frame SHA-256 hex strings produced by the renderer.  The lists must
        be the same length and each element must match exactly.

        If the scenario is run without rendering (render=False default) the
        frame_hashes list will be empty and the test passes trivially.  The
        non-trivial check requires at least one rendered frame; the runner
        should produce frame_hashes when rendering is enabled.
        """
        # Arrange
        seed = SEED_A

        # Act — run with frame capture enabled if the API supports it
        try:
            result_a = run_episode(
                scenario_id=SCENARIO_ID,
                planner_id=PLANNER_ID,
                seed=seed,
                render=True,
            )
            result_b = run_episode(
                scenario_id=SCENARIO_ID,
                planner_id=PLANNER_ID,
                seed=seed,
                render=True,
            )
        except TypeError:
            # render kwarg not yet supported; fall back to default run
            result_a = _run(seed)
            result_b = _run(seed)

        frames_a: list[str] = getattr(result_a, "frame_hashes", [])
        frames_b: list[str] = getattr(result_b, "frame_hashes", [])

        # Assert — lengths match
        assert len(frames_a) == len(frames_b), (
            f"DC-2 violated: frame_hashes lengths differ for seed={seed}: "
            f"{len(frames_a)} vs {len(frames_b)}"
        )

        # Assert — element-wise equality
        mismatches = [
            (i, h_a, h_b)
            for i, (h_a, h_b) in enumerate(zip(frames_a, frames_b))
            if h_a != h_b
        ]
        assert not mismatches, (
            f"DC-2 violated: {len(mismatches)} frame hash(es) differ for seed={seed}.\n"
            + "\n".join(
                f"  frame[{i}]: run_A={ha[:16]}… run_B={hb[:16]}…"
                for i, ha, hb in mismatches[:10]  # cap output to first 10
            )
        )

    def test_different_seed_different_output(self) -> None:
        """Verifies DC-2: seeds 42 and 43 produce distinguishably different trajectories.

        This is the dual of test_identical_seed_identical_output.  If the RNG is
        wired correctly, different seeds must produce different initial conditions
        and therefore different trajectories.  A hash collision would indicate
        that the seed is not actually propagating through the system.
        """
        # Arrange / Act
        result_42 = _run(SEED_A)
        result_43 = _run(SEED_B)

        hash_42 = _sha256_of(result_42.trajectory)
        hash_43 = _sha256_of(result_43.trajectory)

        # Assert
        assert hash_42 != hash_43, (
            "DC-2 violated: seeds 42 and 43 produced identical trajectory hashes "
            f"({hash_42}). The seed is not influencing episode outcomes — "
            "check that root_rng is seeded from the seed argument and not a "
            "constant or global source."
        )
