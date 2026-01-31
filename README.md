# UAVBench Documentation Index

Welcome! This is your complete guide to the UAVBench project. Below you'll find links to all documentation organized by topic.

---

## 🎯 Start Here

- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** ← **START HERE** - Complete overview of all files and what they do
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture diagrams and data flow

---

## 🚀 Getting Started

### Installation & Setup
```bash
pip install -e ".[viz]"  # Install with visualization support
```

### First Commands
```bash
# Run a simple benchmark
uavbench --trials 5

# Watch the best path animation
uavbench --trials 10 --play best

# Save animations
uavbench --trials 10 --save-videos best
```

### Quick Links
- **[QUICK_VIDEO_REFERENCE.md](QUICK_VIDEO_REFERENCE.md)** - One-page command reference
- **[ANIMATION_WORKING.md](ANIMATION_WORKING.md)** - How to use interactive visualization

---

## 📚 Documentation by Topic

### Core Concepts
| Topic | File | Purpose |
|-------|------|---------|
| **Full Project Overview** | [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Complete file listing and descriptions |
| **System Architecture** | [ARCHITECTURE.md](ARCHITECTURE.md) | Diagrams, data flows, design patterns |
| **AI Agent Instructions** | [.github/copilot-instructions.md](.github/copilot-instructions.md) | For AI coding agents working on this project |

### Using the CLI
| Topic | File | Purpose |
|-------|------|---------|
| **Video Features** | [ANIMATION_WORKING.md](ANIMATION_WORKING.md) | How animation works, real examples |
| **Visualization Guide** | [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) | Comprehensive visualization reference |
| **Quick Reference** | [QUICK_VIDEO_REFERENCE.md](QUICK_VIDEO_REFERENCE.md) | Command cheat sheet |
| **Video Feature Summary** | [VIDEO_FEATURE_SUMMARY.md](VIDEO_FEATURE_SUMMARY.md) | Technical details of video export |
| **Videos Folder Guide** | [VIDEOS_FOLDER_GUIDE.md](VIDEOS_FOLDER_GUIDE.md) | How videos are saved to `videos/` folder |

### Troubleshooting & Setup
| Topic | File | Purpose |
|-------|------|---------|
| **Matplotlib Setup** | [MATPLOTLIB_RESOLUTION.md](MATPLOTLIB_RESOLUTION.md) | Fix visualization/dependency issues |
| **Code Review** | [CODE_REVIEW.md](CODE_REVIEW.md) | Issues found and how they were fixed |

---

## 🏗️ Project Structure

```
src/uavbench/
├── envs/              # Environment implementations
│   ├── base.py        # Abstract base class
│   └── urban.py       # 2.5D urban environment
├── scenarios/         # Scenario configuration
│   ├── schema.py      # Data validation
│   ├── loader.py      # YAML loader
│   └── configs/       # Scenario YAML files
├── planners/          # Path planning algorithms
│   └── astar.py       # A* implementation
├── cli/               # Command-line interface
│   └── benchmark.py   # Main entry point
└── viz/               # Visualization
    └── player.py      # Animation & video export

tests/
├── test_scenario_basic.py     # Scenario tests
└── test_urban_env_basic.py    # Environment tests
```

---

## 🎬 Common Tasks

### Watch Animation of Best Path
```bash
uavbench --scenarios urban_easy --planners astar --trials 10 --play best --fps 8
```
See: [ANIMATION_WORKING.md](ANIMATION_WORKING.md)

### Save Animations as Videos
```bash
uavbench --scenarios urban_easy --planners astar --trials 10 --save-videos both --fps 8
```
Files saved to: `videos/`  
See: [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)

### Run Full Benchmark Suite
```bash
uavbench \
  --scenarios urban_easy,urban_medium,urban_hard \
  --planners astar \
  --trials 50 \
  --seed-base 42
```
See: [QUICK_VIDEO_REFERENCE.md](QUICK_VIDEO_REFERENCE.md)

### Compare Multiple Planners
```bash
uavbench \
  --scenarios urban_easy \
  --planners astar,rrtstar \
  --trials 20 \
  --save-videos best
```

---

## 📖 Detailed References

### Understanding the System

1. **What files exist?**
   - See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) section "Complete File Structure"

2. **How does the code work?**
   - See [ARCHITECTURE.md](ARCHITECTURE.md) for diagrams
   - See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) "Core Source Files" section

3. **How do I run benchmarks?**
   - See [QUICK_VIDEO_REFERENCE.md](QUICK_VIDEO_REFERENCE.md) for commands
   - See [ANIMATION_WORKING.md](ANIMATION_WORKING.md) for examples

4. **How do I fix errors?**
   - See [MATPLOTLIB_RESOLUTION.md](MATPLOTLIB_RESOLUTION.md) for visualization issues
   - See [CODE_REVIEW.md](CODE_REVIEW.md) for past issues

### Design Patterns

**RNG Discipline** - Always use `self._rng`, never global `np.random`
- See [.github/copilot-instructions.md](.github/copilot-instructions.md) section "Random Number Generation"

**Type Safety** - Use Pydantic for config validation
- See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) section "scenarios/schema.py"

**Immutable Collections** - Trajectory and events return copies
- See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) section "envs/base.py"

---

## 🔧 Development

### Running Tests
```bash
pytest tests/
pytest tests/test_urban_env_basic.py -v  # Single file
pytest -k "trajectory" -v                 # Pattern match
```

### Adding a New Planner
See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) section "Extension Points"

### Adding a New Scenario
See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) section "Extension Points"

### Code Quality
- See [CODE_REVIEW.md](CODE_REVIEW.md) for best practices
- See [.github/copilot-instructions.md](.github/copilot-instructions.md) for patterns

---

## 📊 Quick Stats

- **Total Python Files**: 13 (11 core + 2 tests)
- **Documentation Files**: 8 markdown files
- **Lines of Code**: ~2,500 core logic
- **Supported Scenarios**: 3 (easy, medium, hard)
- **Implemented Planners**: 1 (A*, extensible)
- **Test Coverage**: Basic environment and scenario tests

---

## ✅ Feature Status

| Feature | Status | Location |
|---------|--------|----------|
| Urban environment | ✅ Working | `envs/urban.py` |
| A* planner | ✅ Working | `planners/astar.py` |
| Scenario loading | ✅ Working | `scenarios/loader.py` |
| CLI interface | ✅ Working | `cli/benchmark.py` |
| Interactive visualization | ✅ Working | `viz/player.py` |
| Video export (GIF) | ✅ Working | `viz/player.py` |
| Video export (MP4) | ✅ With ffmpeg | `viz/player.py` |
| Metrics computation | ✅ Working | `cli/benchmark.py` |
| Tests | ✅ Passing | `tests/` |

---

## 🔗 External Dependencies

### Core
- `gymnasium` - RL environment framework
- `numpy` - Numerical computing
- `pydantic` - Data validation
- `PyYAML` - YAML parsing

### Optional [viz]
- `matplotlib` - Visualization
- `pillow` - Image/GIF support

### Optional [dev]
- `pytest` - Testing
- `mypy` - Type checking
- `sphinx` - Documentation

### Optional (system)
- `ffmpeg` - For MP4 video export (not Python package)

---

## 💡 Tips & Tricks

### Slow-Motion Replay (for debugging)
```bash
uavbench --play best --fps 2
```

### Fast Review (for quick check)
```bash
uavbench --play best --fps 20
```

### Reproducible Results
```bash
uavbench --seed-base 42 --trials 50
```

### Save Only Best Paths
```bash
uavbench --save-videos best  # Faster than saving worst too
```

### Monitor Multiple Planners
```bash
uavbench --planners astar,rrtstar,ga --save-videos best
```

---

## 🆘 Troubleshooting

### Problem: Window doesn't appear
**Solution**: Install matplotlib properly
```bash
pip install --upgrade 'matplotlib>=3.8.0'
```
See: [MATPLOTLIB_RESOLUTION.md](MATPLOTLIB_RESOLUTION.md)

### Problem: "ModuleNotFoundError: matplotlib"
**Solution**: Install visualization dependencies
```bash
pip install 'uavbench[viz]'
```

### Problem: MP4 files not created
**Solution**: ffmpeg not installed (fallback to GIF is normal). Optional:
```bash
brew install ffmpeg  # macOS
```

### Problem: Animation is static/frozen
**Solution**: You're using the latest version which fixes this. Update:
```bash
pip install --upgrade uavbench
```

For more troubleshooting, see:
- [MATPLOTLIB_RESOLUTION.md](MATPLOTLIB_RESOLUTION.md) - Visualization issues
- [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) - Troubleshooting section
- [CODE_REVIEW.md](CODE_REVIEW.md) - Past issues and fixes

---

## 📞 Quick Navigation

**"I want to..."**

| Goal | See This File |
|------|---------------|
| Understand the project | [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) |
| See system architecture | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Run a benchmark | [QUICK_VIDEO_REFERENCE.md](QUICK_VIDEO_REFERENCE.md) |
| Watch animations | [ANIMATION_WORKING.md](ANIMATION_WORKING.md) |
| Learn visualization options | [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) |
| Fix a problem | [MATPLOTLIB_RESOLUTION.md](MATPLOTLIB_RESOLUTION.md) or [CODE_REVIEW.md](CODE_REVIEW.md) |
| Add a new feature | [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) "Extension Points" |
| Understand the code | [ARCHITECTURE.md](ARCHITECTURE.md) or [.github/copilot-instructions.md](.github/copilot-instructions.md) |

---

## 🎓 Learning Path

1. **New to UAVBench?**
   - Start: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
   - Then: [QUICK_VIDEO_REFERENCE.md](QUICK_VIDEO_REFERENCE.md)
   - Try: Run your first benchmark

2. **Want to understand the code?**
   - Read: [ARCHITECTURE.md](ARCHITECTURE.md)
   - Explore: Source code in `src/uavbench/`
   - Reference: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) sections on each file

3. **Want to extend the project?**
   - Read: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) "Extension Points"
   - Learn: [.github/copilot-instructions.md](.github/copilot-instructions.md)
   - Code: Implement new planner/scenario/domain

4. **Troubleshooting?**
   - Check: [MATPLOTLIB_RESOLUTION.md](MATPLOTLIB_RESOLUTION.md)
   - Review: [CODE_REVIEW.md](CODE_REVIEW.md)
   - Search: Relevant documentation file

---

## 📝 Document Legend

| File | Purpose | Audience |
|------|---------|----------|
| PROJECT_SUMMARY.md | Complete file reference | Everyone |
| ARCHITECTURE.md | System design & diagrams | Developers |
| QUICK_VIDEO_REFERENCE.md | Command cheat sheet | Users |
| ANIMATION_WORKING.md | Visualization guide | Users |
| VISUALIZATION_GUIDE.md | Comprehensive viz docs | Users |
| VIDEO_FEATURE_SUMMARY.md | Technical details | Developers |
| MATPLOTLIB_RESOLUTION.md | Dependency fix log | Troubleshooters |
| CODE_REVIEW.md | Code fixes & review | Developers |
| .github/copilot-instructions.md | AI agent guide | AI/Developers |

---

**Last Updated**: 31 January 2026  
**Version**: 0.0.1  
**Status**: Complete and Functional ✅

---

Happy benchmarking! 🚁📊
