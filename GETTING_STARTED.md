# 🚀 Getting Started with UAVBench

## Quick Navigation: Find What You Need

This page helps you find the **right documentation for your goal**.

---

## 📖 I Want To...

### Understand the Project
**Goal**: Get a complete overview of what UAVBench does and how it's structured

👉 **See This File**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
- What is UAVBench?
- Complete file listing
- What each file does
- How files work together
- Extension points

---

### See System Architecture
**Goal**: Understand how the system is designed and how components interact

👉 **See This File**: [ARCHITECTURE.md](ARCHITECTURE.md)
- System overview diagrams
- Data flow between modules
- Environment design
- Configuration system
- Planner architecture
- CLI workflow

---

### Run a Benchmark
**Goal**: Execute experiments and benchmark path planning algorithms

👉 **See This File**: [QUICK_VIDEO_REFERENCE.md](QUICK_VIDEO_REFERENCE.md)
- One-liner commands for common tasks
- All command-line options explained
- Output files reference
- Example workflows

**Quick Start**:
```bash
# Basic benchmark
uavbench --scenarios urban_easy --planners astar --trials 5

# With visualization
uavbench --scenarios urban_easy --planners astar --trials 5 --play best

# Save videos
uavbench --scenarios urban_easy --planners astar --trials 5 --save-videos best
```

---

### Watch Animations
**Goal**: See your path plans visualized with smooth animations

👉 **See This File**: [ANIMATION_WORKING.md](ANIMATION_WORKING.md)
- What's new in animations
- Real-world examples
- How to use `--play` option
- Visualization elements
- Interactive features

**Quick Start**:
```bash
uavbench --scenarios urban_easy --planners astar --trials 10 --play best --fps 8
```

---

### Learn Visualization Options
**Goal**: Master all visualization features and advanced options

👉 **See This File**: [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)
- Complete visualization reference
- All command-line options
- Video format handling (MP4 vs GIF)
- Requirements (ffmpeg, Pillow)
- Advanced usage patterns
- Batch processing examples
- Tips & tricks

---

### Save Videos to Disk
**Goal**: Export path animations as video files to the `videos/` folder

👉 **See This File**: [VIDEOS_FOLDER_GUIDE.md](VIDEOS_FOLDER_GUIDE.md)
- How videos are saved
- Filename patterns
- Video folder structure
- Troubleshooting
- Verification steps
- Example workflows

**Quick Start**:
```bash
uavbench --scenarios urban_easy --planners astar --trials 5 --save-videos best
ls -lh videos/  # Check saved videos
```

---

### Fix a Problem
**Goal**: Troubleshoot errors or resolve issues

**If it's about visualization/matplotlib**:
👉 **See This File**: [MATPLOTLIB_RESOLUTION.md](MATPLOTLIB_RESOLUTION.md)
- matplotlib import errors
- Dependency issues
- Installation fixes
- Testing verification

**If it's about code/logic issues**:
👉 **See This File**: [CODE_REVIEW.md](CODE_REVIEW.md)
- Errors that were found
- Root cause analysis
- How they were fixed
- Recommendations

**General troubleshooting**:
👉 **See This File**: [README.md](README.md#troubleshooting)
- Common problems
- Quick solutions
- Support links

---

### Add a New Feature
**Goal**: Extend UAVBench with new domains, planners, or capabilities

**First, understand the structure**:
👉 **See This File**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
- Look for **"Extension Points"** section
- See where to add new code
- Understand existing patterns

**Then, learn the code patterns**:
👉 **See This File**: [.github/copilot-instructions.md](.github/copilot-instructions.md)
- Architecture & design patterns
- Critical conventions
- Code organization
- Common pitfalls to avoid

**Example: Adding a new planner**
1. Read: PROJECT_SUMMARY.md → "How to add a planner"
2. Reference: See existing `src/uavbench/planners/astar.py`
3. Implement: Follow the same pattern
4. Test: Add tests in `tests/`

---

### Understand the Code
**Goal**: Learn how the code is organized and written

**For high-level understanding**:
👉 **See This File**: [ARCHITECTURE.md](ARCHITECTURE.md)
- Visual diagrams
- Data flow
- Module relationships
- Design patterns

**For detailed code patterns and conventions**:
👉 **See This File**: [.github/copilot-instructions.md](.github/copilot-instructions.md)
- Critical patterns
- Code conventions
- Random number generation discipline
- Trajectory & event logging
- Type safety & validation
- Common pitfalls

---

## 📊 Documentation Overview Table

| I Want To... | Primary File | Backup File | Time |
|---|---|---|---|
| Understand the project | PROJECT_SUMMARY.md | ARCHITECTURE.md | 30 min |
| See architecture | ARCHITECTURE.md | PROJECT_SUMMARY.md | 20 min |
| Run a benchmark | QUICK_VIDEO_REFERENCE.md | README.md | 5 min |
| Watch animations | ANIMATION_WORKING.md | VISUALIZATION_GUIDE.md | 10 min |
| Learn visualization | VISUALIZATION_GUIDE.md | ANIMATION_WORKING.md | 15 min |
| Save videos | VIDEOS_FOLDER_GUIDE.md | QUICK_VIDEO_REFERENCE.md | 5 min |
| Fix problems | MATPLOTLIB_RESOLUTION.md | CODE_REVIEW.md | 10 min |
| Add features | PROJECT_SUMMARY.md | .github/copilot-instructions.md | 45 min |
| Understand code | ARCHITECTURE.md | .github/copilot-instructions.md | 30 min |

---

## 🎯 Common Workflows

### Workflow 1: New User (5 minutes)
```
1. Read: This page (GETTING_STARTED.md) ← You are here!
2. Run: uavbench --trials 3 --play best
3. Explore: QUICK_VIDEO_REFERENCE.md for more commands
```

### Workflow 2: Learning the Codebase (1-2 hours)
```
1. Read: PROJECT_SUMMARY.md (30 min)
   ↓
2. Read: ARCHITECTURE.md (20 min)
   ↓
3. Browse: Key source files mentioned in those docs (30 min)
   ↓
4. Study: .github/copilot-instructions.md for patterns (30 min)
```

### Workflow 3: Running Experiments (15 minutes)
```
1. Open: QUICK_VIDEO_REFERENCE.md
2. Choose: Command for your use case
3. Run: uavbench command
4. Check: VIDEOS_FOLDER_GUIDE.md if saving videos
5. View: Videos in videos/ folder
```

### Workflow 4: Fixing an Issue (10-30 minutes)
```
1. Check: Error message
2. Go to: MATPLOTLIB_RESOLUTION.md or CODE_REVIEW.md
3. Find: Your issue
4. Apply: Solution
5. Test: Run command again
```

### Workflow 5: Adding a New Feature (2-4 hours)
```
1. Read: PROJECT_SUMMARY.md (30 min)
2. Study: ARCHITECTURE.md (20 min)
3. Review: .github/copilot-instructions.md (30 min)
4. Examine: Relevant existing code (30 min)
5. Implement: Your feature (60+ min)
6. Test: Add tests (30+ min)
```

---

## 🔗 Quick Links

### Essential Files (Read These First)
- 📌 [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Complete overview
- 🏗️ [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- 🚀 [QUICK_VIDEO_REFERENCE.md](QUICK_VIDEO_REFERENCE.md) - Command cheat sheet

### Feature Documentation
- 🎬 [ANIMATION_WORKING.md](ANIMATION_WORKING.md) - Animation features
- 📹 [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) - Full viz reference
- 💾 [VIDEOS_FOLDER_GUIDE.md](VIDEOS_FOLDER_GUIDE.md) - Video output guide

### Technical Reference
- 🛠️ [.github/copilot-instructions.md](.github/copilot-instructions.md) - Code patterns
- 🐛 [CODE_REVIEW.md](CODE_REVIEW.md) - Issues & fixes
- 🔧 [MATPLOTLIB_RESOLUTION.md](MATPLOTLIB_RESOLUTION.md) - Setup guide

### Meta Documentation
- 📚 [README.md](README.md) - Documentation index
- 📖 [DOCUMENTATION_SUMMARY.md](DOCUMENTATION_SUMMARY.md) - Doc statistics
- 👈 [GETTING_STARTED.md](GETTING_STARTED.md) - This file!

---

## ⚡ Super Quick Reference

```bash
# Install with visualization
pip install -e ".[viz]"

# Run simple benchmark
uavbench --trials 5

# Watch best path
uavbench --trials 10 --play best --fps 8

# Save videos
uavbench --trials 10 --save-videos both --fps 8

# Check saved videos
ls -lh videos/
open videos/urban_easy_astar_best_*.gif

# Run full experiment
uavbench --scenarios urban_easy,urban_medium \
         --planners astar \
         --trials 20 \
         --save-videos best \
         --seed-base 42
```

See [QUICK_VIDEO_REFERENCE.md](QUICK_VIDEO_REFERENCE.md) for more commands.

---

## 📚 All Documentation Files

| File | Purpose | Audience |
|------|---------|----------|
| **GETTING_STARTED.md** | ← You are here! Quick navigation | Everyone |
| **PROJECT_SUMMARY.md** | Complete file reference | Developers, researchers |
| **ARCHITECTURE.md** | System design & diagrams | Developers |
| **README.md** | Documentation index | Everyone |
| **QUICK_VIDEO_REFERENCE.md** | Command cheat sheet | Users |
| **ANIMATION_WORKING.md** | Animation features | Users, researchers |
| **VISUALIZATION_GUIDE.md** | Complete viz reference | Advanced users |
| **VIDEOS_FOLDER_GUIDE.md** | Video output guide | Users |
| **.github/copilot-instructions.md** | Code patterns | Developers, AI |
| **CODE_REVIEW.md** | Issues & fixes | Developers |
| **MATPLOTLIB_RESOLUTION.md** | Setup & troubleshooting | Troubleshooters |
| **DOCUMENTATION_SUMMARY.md** | Doc statistics | Meta |

---

## ✅ Checklist: What You Can Do

After reading this page, you can now:

- ✅ Find the right documentation for your goal
- ✅ Navigate the project quickly
- ✅ Run basic benchmarks
- ✅ Watch animations
- ✅ Save videos
- ✅ Understand the architecture
- ✅ Learn code patterns
- ✅ Know where to troubleshoot
- ✅ Know where to add features

---

## 🎓 Recommended Reading Order

### For New Users
1. This page (GETTING_STARTED.md) ← You are here
2. [QUICK_VIDEO_REFERENCE.md](QUICK_VIDEO_REFERENCE.md)
3. [ANIMATION_WORKING.md](ANIMATION_WORKING.md)

### For Developers
1. [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
2. [ARCHITECTURE.md](ARCHITECTURE.md)
3. [.github/copilot-instructions.md](.github/copilot-instructions.md)

### For Troubleshooters
1. [README.md](README.md#troubleshooting)
2. [MATPLOTLIB_RESOLUTION.md](MATPLOTLIB_RESOLUTION.md)
3. [CODE_REVIEW.md](CODE_REVIEW.md)

---

## 🤔 Can't Find What You're Looking For?

1. **Check the Quick Links section** above
2. **Search README.md** for your topic
3. **Browse PROJECT_SUMMARY.md** for file locations
4. **Check ARCHITECTURE.md** for system overview
5. **See CODE_REVIEW.md** for common issues

---

## 📞 Next Steps

✨ **Ready to get started?**

- **Just want to run it?** → [QUICK_VIDEO_REFERENCE.md](QUICK_VIDEO_REFERENCE.md)
- **Want to understand it?** → [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
- **Want to extend it?** → [ARCHITECTURE.md](ARCHITECTURE.md) + [.github/copilot-instructions.md](.github/copilot-instructions.md)
- **Having issues?** → [MATPLOTLIB_RESOLUTION.md](MATPLOTLIB_RESOLUTION.md) or [CODE_REVIEW.md](CODE_REVIEW.md)

---

**Your UAVBench documentation is complete and organized!** 🎉

Everything you need is here. Pick a goal above and follow the link to the right documentation file.

Happy coding! 🚀
