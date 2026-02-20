"""Tests for P1 realism features: constraint latency, comms dropout, GNSS noise."""

import numpy as np
import pytest
from dataclasses import replace

from uavbench.scenarios.schema import ScenarioConfig, Domain, Difficulty


def _base_config(**overrides) -> ScenarioConfig:
    """Minimal config for P1 feature testing."""
    defaults = dict(
        name="test_p1",
        domain=Domain.URBAN,
        difficulty=Difficulty.EASY,
        map_size=10,
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


class TestConstraintLatencySchema:
    def test_default_is_zero(self):
        cfg = _base_config()
        assert cfg.constraint_latency_steps == 0

    def test_valid_positive(self):
        cfg = _base_config(constraint_latency_steps=3)
        cfg.validate()
        assert cfg.constraint_latency_steps == 3

    def test_negative_raises(self):
        cfg = _base_config(constraint_latency_steps=-1)
        with pytest.raises(ValueError, match="constraint_latency_steps"):
            cfg.validate()


class TestCommsDropoutSchema:
    def test_default_is_zero(self):
        cfg = _base_config()
        assert cfg.comms_dropout_prob == 0.0

    def test_valid_range(self):
        for p in (0.0, 0.5, 1.0):
            cfg = _base_config(comms_dropout_prob=p)
            cfg.validate()

    def test_out_of_range_raises(self):
        for p in (-0.1, 1.1):
            cfg = _base_config(comms_dropout_prob=p)
            with pytest.raises(ValueError, match="comms_dropout_prob"):
                cfg.validate()


class TestGNSSNoiseSchema:
    def test_default_is_zero(self):
        cfg = _base_config()
        assert cfg.gnss_noise_sigma == 0.0

    def test_valid_positive(self):
        cfg = _base_config(gnss_noise_sigma=1.5)
        cfg.validate()
        assert cfg.gnss_noise_sigma == 1.5

    def test_negative_raises(self):
        cfg = _base_config(gnss_noise_sigma=-0.5)
        with pytest.raises(ValueError, match="gnss_noise_sigma"):
            cfg.validate()


class TestP1FieldsInBenchmarkResults:
    """Verify that P1 fields appear in benchmark result dict."""

    def test_realism_fields_present(self):
        """Quick smoke: run a trivial episode, check P1 fields exist."""
        from uavbench.cli.benchmark import run_dynamic_episode

        result = run_dynamic_episode(
            "gov_civil_protection_easy",
            "astar",
            seed=0,
            config_overrides={
                "constraint_latency_steps": 2,
                "comms_dropout_prob": 0.1,
                "gnss_noise_sigma": 0.5,
            },
        )
        assert result["constraint_latency_steps"] == 2
        assert result["comms_dropout_prob"] == 0.1
        assert result["gnss_noise_sigma"] == 0.5
        assert result["replan_mode"] in ("native", "harness_replan")
