# Artifact Evaluation — UAVBench

This document follows the standard **ACM/IEEE Artifact Evaluation** badge checklist
(ICRA 2025 / IEEE RAS compatible).

---

## Badges Sought

| Badge | Status | Evidence |
|-------|--------|----------|
| **Available** | ✅ | Public GitHub repository under MIT license |
| **Functional** | ✅ | `make test` passes 245+ unit tests on clean install |
| **Reusable** | ✅ | Modular API, documented extension points, pip-installable |
| **Reproduced** | ✅ | `make reproduce` reproduces all paper tables and figures from scratch |

---

## 1. Available

- **Repository:** <https://github.com/uavbench/uavbench>
- **License:** MIT (`LICENSE`)
- **Persistent DOI:** Zenodo archive (to be minted at camera-ready)
- **Data:** OpenStreetMap-derived tiles bundled in `data/maps/` (ODbL license)

## 2. Functional

### Quick smoke test (< 30 s)

```bash
git clone https://github.com/uavbench/uavbench.git && cd uavbench
make install
make test        # 245+ tests, all green
```

### Dependencies

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.10 | Tested on 3.10, 3.11, 3.12, 3.14 |
| OS | macOS / Linux | Windows untested but likely works |
| RAM | ≥ 4 GB | Grid maps are < 100 MB |
| GPU | Not required | Pure CPU benchmark |

All Python dependencies are pinned in `requirements-lock.txt`.

### What the artifact does

1. Loads 9 scenario YAML configs (3 missions × 3 difficulties)
2. Instantiates `UrbanEnv` with real OpenStreetMap tiles
3. Runs 6 planners (A*, θ*, D* Lite, AD*, DWA, MPPI) under identical fairness constraints
4. Collects comprehensive metrics: success, path length, replans, risk exposure, collisions
5. Exports publication-ready CSV, LaTeX tables, and reproducibility manifest

## 3. Reusable

### Extension points

| Extension | How |
|-----------|-----|
| **New planner** | Implement `plan(grid, start, goal, **kwargs) → list[tuple]` and register in `PLANNERS` dict |
| **New scenario** | Add YAML to `src/uavbench/scenarios/configs/` — auto-discovered by registry |
| **New mission** | Subclass policy in `src/uavbench/missions/` |
| **New metrics** | Extend `EpisodeMetrics` dataclass in `metrics/comprehensive.py` |
| **Custom ablation** | Pass `--protocol-variant <name>` to CLI |

### API Documentation

- `docs/API_REFERENCE.md` — module-by-module reference
- `docs/PAPER_PROTOCOL.md` — fair evaluation protocol specification
- `BENCHMARK_GUIDE.md` — end-to-end operational guide
- `notebooks/getting_started.ipynb` — interactive tutorial

### Code quality

- **Type-checked:** `mypy src/ tests/` passes
- **Test coverage:** 245+ tests across 16 test modules
- **CI-ready:** `make test` and `make lint` targets

## 4. Reproduced

### One-command reproduction

```bash
make reproduce   # install → test → full validation pipeline
```

This runs the complete `paper_best_paper_validation.py` script which produces:

| Artifact | Path | Description |
|----------|------|-------------|
| Literature matrix | `literature_positioning_matrix.csv` | Gap analysis vs related benchmarks |
| Statistical summary | `statistical_summary.csv` | Mean ± CI per planner per track |
| Effect sizes | `effect_sizes.csv` | Cohen's d, Mann-Whitney p-values |
| Stress curves | `stress_intensity_curve.csv` | Success vs α ∈ [0,1] |
| Ablation deltas | `ablation_deltas.csv` | ΔSuccess per variant |
| Feasibility proof | `feasibility_proof.csv` | Guardrail vs no-guardrail infeasibility |
| Fairness audit | `fairness_audit.json` | 5-check protocol verification |
| Failure taxonomy | `failure_mode_table.csv` | Categorized failure modes |
| Ranking stability | `ranking_stability.csv` | Kendall τ across seeds |
| Sensitivity heatmap | `sensitivity_ranking_stability.csv` | Perturbation robustness |
| Runtime profile | `runtime_profile.json` | CPU, Python version, wall-clock time |
| Validation manifest | `validation_manifest.json` | Complete file inventory |
| Figures 1–5 | `figures/` | Publication-ready PNG/PDF |
| LaTeX tables | `*.tex` | Camera-ready table fragments |

### Reproducibility statement

All benchmark runs use deterministic seeding (`numpy.random.default_rng(seed)`).
Given identical hardware, OS, and Python version, outputs are bit-for-bit reproducible.
The `validation_manifest.json` records the exact environment for cross-machine verification.

### Expected runtime

| Configuration | Seeds | Wall-clock (M1 Mac) |
|---------------|-------|---------------------|
| Smoke test | 3 | ~2 min |
| Quick validation | 10 | ~15 min |
| Full paper run | 30 | ~60 min |

---

## Checklist for AE Reviewers

- [ ] `make install` succeeds without errors
- [ ] `make test` reports 245+ tests passed
- [ ] `make reproduce` completes and generates `validation_manifest.json`
- [ ] Generated CSV files contain non-trivial numeric data
- [ ] `fairness_audit.json` shows `overall_pass: true`
- [ ] `runtime_profile.json` contains valid CPU and Python info
- [ ] Figures in `figures/` directory are readable PNG/PDF files
- [ ] Adding a new planner via PLANNERS dict works without modifying core code
