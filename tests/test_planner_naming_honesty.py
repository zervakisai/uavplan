"""Tests for planner naming honesty and deprecation warnings.

Validates that:
1. Every planner class has a docstring disclosing what it actually implements.
2. Renamed planners carry honest names and canonical citations.
3. Legacy aliases still resolve to the correct classes.
4. Registry contains both honest and legacy keys.
5. Deprecated keys emit DeprecationWarning on access.
6. Paper-suite planners do NOT emit DeprecationWarning.
"""

import inspect
import warnings

import pytest

from uavbench.planners import (
    PLANNERS,
    PAPER_PLANNERS,
    DEPRECATED_ALIASES,
    DEPRECATED_PLANNERS,
    # Honest names
    PeriodicReplanPlanner,
    PeriodicReplanConfig,
    AggressiveReplanPlanner,
    AggressiveReplanConfig,
    GreedyLocalPlanner,
    GreedyLocalConfig,
    GridMPPIPlanner,
    DStarLiteRealPlanner,
    # Legacy aliases
    DStarLitePlanner,
    DStarLiteConfig,
    ADStarPlanner,
    ADStarConfig,
    DWAPlanner,
    DWAConfig,
    MPPIPlanner,
    MPPIConfig,
    # Unchanged
    AStarPlanner,
    ThetaStarPlanner,
)


class TestLegacyAliases:
    """Legacy class names must resolve to the same honest class."""

    def test_dstar_lite_alias(self):
        assert DStarLitePlanner is PeriodicReplanPlanner

    def test_dstar_lite_config_alias(self):
        assert DStarLiteConfig is PeriodicReplanConfig

    def test_ad_star_alias(self):
        assert ADStarPlanner is AggressiveReplanPlanner

    def test_ad_star_config_alias(self):
        assert ADStarConfig is AggressiveReplanConfig

    def test_dwa_alias(self):
        assert DWAPlanner is GreedyLocalPlanner

    def test_dwa_config_alias(self):
        assert DWAConfig is GreedyLocalConfig

    def test_mppi_alias(self):
        assert MPPIPlanner is GridMPPIPlanner


class TestDeprecationWarnings:
    """Deprecated registry keys must emit DeprecationWarning."""

    @pytest.mark.parametrize("key", sorted(DEPRECATED_ALIASES.keys()))
    def test_legacy_alias_warns(self, key):
        with pytest.warns(DeprecationWarning, match=rf"'{key}' is deprecated"):
            PLANNERS[key]

    @pytest.mark.parametrize("key", sorted(DEPRECATED_ALIASES.keys()))
    def test_legacy_alias_suggests_replacement(self, key):
        new_key = DEPRECATED_ALIASES[key]
        with pytest.warns(DeprecationWarning, match=rf"Use '{new_key}' instead"):
            PLANNERS[key]

    @pytest.mark.parametrize("key", sorted(DEPRECATED_PLANNERS))
    def test_deprecated_planner_warns(self, key):
        with pytest.warns(DeprecationWarning, match=r"not in the paper benchmark suite"):
            PLANNERS[key]

    @pytest.mark.parametrize("key", PAPER_PLANNERS)
    def test_paper_planner_no_warning(self, key):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            PLANNERS[key]  # must NOT raise


class TestPaperSuite:
    """Paper suite must contain exactly the 6 planners for the paper."""

    def test_paper_suite_count(self):
        assert len(PAPER_PLANNERS) == 6

    def test_paper_suite_keys_in_registry(self):
        for key in PAPER_PLANNERS:
            assert key in PLANNERS

    def test_paper_suite_excludes_greedy_local(self):
        assert "greedy_local" not in PAPER_PLANNERS

    def test_paper_suite_excludes_legacy_aliases(self):
        for key in DEPRECATED_ALIASES:
            assert key not in PAPER_PLANNERS

    def test_paper_suite_expected_planners(self):
        assert set(PAPER_PLANNERS) == {
            "astar",
            "theta_star",
            "periodic_replan",
            "aggressive_replan",
            "incremental_dstar_lite",
            "grid_mppi",
        }


class TestRegistryKeys:
    """Registry must expose both paper-suite and deprecated keys."""

    @pytest.mark.parametrize(
        "key,cls",
        [
            # Paper suite
            ("astar", AStarPlanner),
            ("theta_star", ThetaStarPlanner),
            ("periodic_replan", PeriodicReplanPlanner),
            ("aggressive_replan", AggressiveReplanPlanner),
            ("grid_mppi", GridMPPIPlanner),
            ("incremental_dstar_lite", DStarLiteRealPlanner),
        ],
    )
    def test_paper_suite_key(self, key, cls):
        assert key in PLANNERS, f"Registry missing key '{key}'"
        assert PLANNERS[key] is cls

    @pytest.mark.parametrize(
        "key,cls",
        [
            ("greedy_local", GreedyLocalPlanner),
            ("dstar_lite", DStarLitePlanner),
            ("ad_star", ADStarPlanner),
            ("dwa", DWAPlanner),
            ("mppi", MPPIPlanner),
        ],
    )
    def test_deprecated_key_still_accessible(self, key, cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert key in PLANNERS
            assert PLANNERS[key] is cls


class TestDocstringHonesty:
    """Every renamed planner must disclose its simplification in the docstring."""

    _RENAMED_PLANNERS = {
        "PeriodicReplanPlanner": (PeriodicReplanPlanner, "Koenig"),
        "AggressiveReplanPlanner": (AggressiveReplanPlanner, "Likhachev"),
        "GreedyLocalPlanner": (GreedyLocalPlanner, "Fox"),
        "GridMPPIPlanner": (GridMPPIPlanner, "Williams"),
    }

    @pytest.mark.parametrize("name", _RENAMED_PLANNERS.keys())
    def test_docstring_exists(self, name):
        cls = self._RENAMED_PLANNERS[name][0]
        doc = inspect.getdoc(cls)
        assert doc is not None, f"{name} has no docstring"

    @pytest.mark.parametrize("name", _RENAMED_PLANNERS.keys())
    def test_docstring_cites_canonical(self, name):
        cls, author_fragment = self._RENAMED_PLANNERS[name]
        doc = inspect.getdoc(cls)
        assert author_fragment.lower() in doc.lower(), (
            f"{name} docstring must cite canonical author '{author_fragment}'"
        )

    @pytest.mark.parametrize("name", _RENAMED_PLANNERS.keys())
    def test_docstring_discloses_simplification(self, name):
        cls = self._RENAMED_PLANNERS[name][0]
        doc = inspect.getdoc(cls).lower()
        disclosure_keywords = [
            "simplified", "not implement", "grid-discretized",
            "greedy", "does not implement", "4-connected",
            "cardinal", "periodic", "aggressive",
        ]
        assert any(kw in doc for kw in disclosure_keywords), (
            f"{name} docstring must disclose simplification; "
            f"none of {disclosure_keywords} found"
        )


class TestAllPlannersHaveDocstrings:
    """Even unchanged planners should have docstrings."""

    @pytest.mark.parametrize("key", PAPER_PLANNERS)
    def test_paper_planner_has_docstring(self, key):
        cls = PLANNERS[key]
        doc = inspect.getdoc(cls)
        assert doc is not None, f"PLANNERS['{key}'] ({cls.__name__}) has no docstring"
