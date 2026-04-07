# PhD Research Gap Analysis — FLARE Paper #2 & #3

**Author:** Konstantinos Zervakis
**Date:** 2026-04-07
**Purpose:** Documented literature gaps justifying two follow-up papers that
close the 2D research arc of the PhD, building on FLARE (Paper #1).

---

## 0. Context

Paper #1 (*Ranking Inversion in Risk-Parameterised UAV Path Planning for
Wildfire Emergency Response*) establishes FLARE: a deterministic 2D
benchmark with five classical risk-parameterised planners, full-observability
hazard dynamics, and strictly-decreasing mission-impact functions M(t). The
central finding is that navigation success rate (SR) and mission score M(t)
produce systematically divergent planner orderings (Friedman p < 0.001).

Three structural assumptions of Paper #1 remain open for investigation:

| # | Assumption of Paper #1 | Paper that challenges it |
|---|---|---|
| A1 | Full observability of the hazard field | **Paper #2** |
| A2 | Classical, non-learned planners only | **Paper #2** |
| A3 | Deterministic, point-estimate cost map | **Paper #3** |

Each gap below is documented with at least one recent (2023–2025) reference
confirming that the research community is active in the area **but has not
yet addressed the specific intersection that FLARE enables**.

---

## Paper #2 — Planning Under Perceptual Fog: Vision-Limited UAV Navigation in Unknown Hazard Fields

### Gap 2.1 — Partial observability is not systematically evaluated against mission-level metrics

**State of the art.** POMDP-based UAV navigation is an active sub-field.
Lauri et al. (2022) provide the authoritative survey of POMDPs in robotics,
documenting that active SLAM, informative path planning, and multi-robot
coordination all reduce to belief-space decision problems [1]. Dressel &
Kochenderfer and subsequent work have deployed POMCP and adaptive belief
trees (ABT) for UAV search [2]. Very recent contributions include *PyroTrack*
(Karimi et al. 2024), a belief-based DRL path planner that explicitly models
"outdated" regions via a certainty factor for wildfire tracking [3], and
*Shrinking POMCP* (Rangi & Kochenderfer 2024) for real-time UAV search and
rescue [4]. The decision-making survey of Lu et al. (2025) confirms belief-
space planning remains an open area dominated by approximations [5].

**What is missing.** None of these works evaluate their planners against a
**strictly-decreasing mission-impact function M(t)**. All report
success rate, coverage, or tracking error — exactly the navigation-centric
metrics that Paper #1 proved are insufficient. The ranking-inversion
phenomenon has therefore never been tested under partial observability, and
it is a priori unclear whether it persists, amplifies, or disappears.

**Paper #2 contribution angle.** Port the M(t) evaluation framework of Paper
#1 into a belief-space setting, and show whether classical planners that
become *myopically conservative* under partial observability suffer a
larger ranking inversion than adaptive/learning methods.

---

### Gap 2.2 — Vision-based hazard perception is decoupled from risk-aware planning

**State of the art.** UAV-based fire/smoke segmentation has matured rapidly.
The 2025 comprehensive survey of Saeed & Xia catalogues CNN, transformer, and
attention-based approaches for wildfire detection from UAV imagery [6].
Dedicated datasets such as the Boreal Forest Fire dataset (*Scientific Data*,
2025) provide pixel-level smoke/fire annotations [7]. Lightweight onboard
segmentation at 25 FPS is now demonstrated [6]. At the European scale, the
ECMWF *Probability of Fire* (PoF) model integrates vegetation dryness,
lightning, and human presence into a gridded risk forecast [6].

**What is missing.** Vision pipelines produce per-pixel fire/smoke
probabilities, yet published risk-aware planners still consume **hand-crafted
distance-decay masks** (Primatesta et al. 2019 [8]; Zhou et al. 2025 [9]).
The *Vision-based Fire Management Survey* of 2025 [6] explicitly identifies
"end-to-end integration of perception outputs into planner cost maps" as an
open problem. No reproducible benchmark couples: (a) onboard CNN-inferred
hazard probabilities, (b) belief-map propagation between observations, and
(c) a time-decreasing mission-effectiveness score.

**Paper #2 contribution angle.** Add an egocentric-sensor + CNN perception
layer to FLARE, feed its probabilistic outputs into the existing R(x) risk-
fusion stage, and quantify how perception error propagates into both
navigation and mission metrics.

---

### Gap 2.3 — DRL for UAV wildfire navigation is trained and evaluated on navigation rewards

**State of the art.** DRL for UAV wildfire response has accelerated.
Julian & Kochenderfer's *Distributed Wildfire Surveillance* (JGCD 2019) is
the canonical reference [10]; extensions include swarm reconnaissance with
DRL (Chen et al. 2024, *Electronics*) [11], DDPG-based fire-front tracking
[12], and vision-based model-free DRL for unknown complex environments
(Yousaf et al. 2025, *Drones*) [13]. The 2025 ACM survey on RL for UAVs
(Patra et al.) catalogues 200+ papers [14]. The *Quantitative Analysis of
Reward Shaping* preprint (2025) confirms that reward design dominates
training outcomes [15].

**What is missing.** Every DRL UAV planner surveyed rewards **proximity to
goal, survival, or coverage**. None rewards a strictly-decreasing M(t) that
distinguishes an *early* successful delivery from a *late* successful
delivery. As a direct consequence, learned policies inherit the same SR-
versus-M(t) misalignment that Paper #1 identified in classical planners —
but this has never been empirically demonstrated, because no benchmark
provided the M(t) channel. The *Deep RL that Matters* critique of Henderson
et al. [16] further stresses that RL evaluation is fragile without
controlled benchmarks; FLARE provides exactly the controlled substrate that
is missing.

**Paper #2 contribution angle.** Train model-free (PPO + CNN) and model-
based (Dreamer-style) agents against (i) an SR reward and (ii) an M(t)
reward, and measure whether the inversion is an artifact of objective mis-
specification rather than an algorithmic limitation.

---

### Gap 2.4 — No reproducible UAV benchmark combines partial observability, unknown hazards, and time-decreasing mission scoring

**State of the art.** A 2025 arXiv benchmark confusingly also named *UAVBench*
(Li et al., arXiv 2511.11252) uses LLM-generated scenarios annotated with
risk tags [17] — but it is scenario-descriptive, not physics-simulated, and
reports only categorical safety tags. *Shrinking POMCP* [4] provides a
POMDP testbed but no mission-impact scoring. The *Decision-Making for Path
Planning Under Uncertainty* review (Lu et al. 2025) notes that "standardised
evaluation protocols for belief-space UAV planners remain rare" [5].
Bonsignorio et al. (*Nature Machine Intelligence* 2025) reiterate the broader
reproducibility crisis in robotics research [18].

**What is missing.** No public benchmark simultaneously offers: deterministic
replay (DC-1-style contracts), configurable sensor models, unknown a-priori
hazards with ground-truth reveal, and strictly-decreasing M(t) evaluation.
FLARE already provides 1 and 4; Paper #2 adds 2 and 3.

> **Naming conflict note.** The 2025 arXiv pre-print reuses the name
> *UAVBench*. Our project has already been renamed **FLARE** — the gap
> analysis actually strengthens the case for retaining the new name.

---

## Paper #3 — Belief-Space and Quantum-Inspired Planning Under Expanding Wildfire Hazards

### Gap 3.1 — Risk-aware UAV planning uses point-estimate cost maps

**State of the art.** Risk-aware edge-cost inflation is the dominant design
pattern: Primatesta et al. (2019) [8], Hu et al. (2023, *Sci. Rep.*) [19],
Zhou et al. (*Risk Analysis* 2025) [9], Karaağaç et al. (*JIRS* 2025) [20],
and Melissaris et al. (*Drones* 2025) [21]. All these works — including
FLARE Paper #1 — treat R(x) as a **deterministic scalar field**.

**What is missing.** A principled treatment of R(x) as a *distribution*,
with chance-constrained (CC) formulations that bound collision probability.
Chance-constrained geofencing exists (Yang et al. 2025, *JGCD*) [22], but
is restricted to static convex obstacles — not expanding stochastic fire.
Belief-space planning reviews [5] confirm that CC approaches for
time-varying stochastic hazards remain computationally prohibitive and are
an open research area.

**Paper #3 contribution angle.** Formulate FLARE as a Chance-Constrained
Stochastic Shortest Path problem on a time-expanded graph, with uncertainty
derived from the CA fire model's ensemble of rollouts.

---

### Gap 3.2 — Quantum-assisted UAV path planning is nascent, static, and hazard-free

**State of the art.** 2025 saw the first dedicated QAOA-for-UAV papers.
*QUAV* (arXiv 2508.21361) formulates UAV pathfinding as a QUBO and reports
a 447m vs 662m advantage over RRT on five scenarios [23]. *Dynamic-Depth
QAOA for Constrained Shortest Path* (arXiv 2511.08657) demonstrates the
method on 10- and 16-qubit CSPP instances [24]. Quantum grid path-planning
via parallel QAOA circuits (arXiv 2510.07413) addresses unweighted graphs
[25]. The 2025 *QAOA for Multi-Objective 6G Routing* (*Computer Networks*)
shows feasibility on large-scale transport graphs [26].

**What is missing.** Every quantum UAV planner published to date assumes
**static obstacles**, **deterministic edge costs**, and **no time-varying
hazards**. None of them evaluates on a mission-impact metric. QAOA for
*time-expanded* graphs with stochastic edge weights — the natural formulation
for a spreading fire — does not exist in the literature. Furthermore, no
quantum UAV paper compares against a reproducible classical baseline under
identical conditions.

**Paper #3 contribution angle.** Provide the first QUBO encoding of UAV
path planning under expanding wildfire dynamics, evaluate QAOA (Qiskit /
D-Wave simulator) against classical belief-space baselines on the M(t)
metric, and characterise the graph-structure regimes (anisotropy,
connectivity, time-horizon depth) where quantum approaches show promise.

---

### Gap 3.3 — No unified comparison of belief-space classical, sampling-based, and quantum methods on the same mission-aware benchmark

**State of the art.** Belief-space classical methods (CC-A*, CC-RRT*),
sampling-based POMDP solvers (POMCP, DESPOT, PyroTrack [3]), and quantum-
inspired optimisers (QUAV [23], DDQAOA [24]) each publish on **their own**
testbeds. The *Decision-Making-Based Path Planning for Autonomous UAVs* 2025
survey (arXiv 2508.09304) [27] catalogues these families but finds no head-
to-head evaluation.

**What is missing.** A common benchmark that lets these three families be
compared on (i) identical hazard dynamics, (ii) identical M(t) scoring, and
(iii) deterministic replay. Without such a comparison, the community cannot
answer whether the quantum speedup claims of [23]–[25] translate to
mission-level advantages in realistic disaster scenarios.

**Paper #3 contribution angle.** Deliver the first three-family comparison
(classical belief-space vs sampling-based POMDP vs QAOA) under one
reproducible mission-impact framework, using FLARE's existing determinism
and statistical-validation contracts.

---

## Summary Table — Paper Positioning in the Literature

| Capability | Classical risk-aware planners ([8][9][19]–[21]) | DRL UAV wildfire ([10]–[13]) | POMDP/belief UAV ([1]–[5]) | Quantum UAV ([23]–[26]) | **FLARE #1** | **FLARE #2 (proposed)** | **FLARE #3 (proposed)** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Deterministic replay | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| Expanding fire dynamics | partial | partial | ✗ | ✗ | ✓ | ✓ | ✓ |
| Partial observability | ✗ | partial | ✓ | ✗ | ✗ | ✓ | ✓ |
| Vision/CNN perception in loop | ✗ | partial ([13]) | ✗ | ✗ | ✗ | ✓ | — |
| Unknown a-priori hazards | ✗ | partial | ✓ | ✗ | ✗ | ✓ | ✓ |
| Belief map / distributional cost | ✗ | ✗ | ✓ | ✗ | ✗ | partial | ✓ |
| Chance-constrained formulation | ✗ | ✗ | partial | ✗ | ✗ | ✗ | ✓ |
| Quantum/QUBO encoding | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ |
| Mission-impact M(t) evaluation | ✗ | ✗ | ✗ | ✗ | **✓** | **✓** | **✓** |

The final row is the discriminating column that no prior work provides.

---

## References

[1] Lauri, M.; Hsu, D.; Pajarinen, J. *Partially Observable Markov Decision
Processes in Robotics: A Survey.* IEEE T-RO, 2022. arXiv:2209.10342.

[2] *Partially Observable Markov Decision Processes (POMDPs) and Robotics.*
arXiv:2107.07599.

[3] Karimi et al. *PyroTrack: Belief-Based Deep Reinforcement Learning Path
Planning for Aerial Wildfire Monitoring in Partially Observable Environments.*
2024. https://par.nsf.gov/servlets/purl/10588496

[4] Rangi, A.; Kochenderfer, M.J. *Shrinking POMCP: A Framework for Real-Time
UAV Search and Rescue.* 2024. https://par.nsf.gov/servlets/purl/10569097

[5] Lu, Y. et al. *Decision-Making for Path Planning of Mobile Robots Under
Uncertainty: A Review of Belief-Space Planning Simplifications.* Robotics
14(9):127, 2025. https://www.mdpi.com/2218-6581/14/9/127

[6] Saeed, F.; Xia, Y. *Vision-based fire management system using autonomous
unmanned aerial vehicles: a comprehensive survey.* Artificial Intelligence
Review 58, 2025. https://doi.org/10.1007/s10462-025-11415-3

[7] *Boreal Forest Fire: UAV-collected Wildfire Detection and Smoke
Segmentation Dataset.* Scientific Data, 2025.
https://www.nature.com/articles/s41597-025-05634-0

[8] Primatesta, S.; Rizzo, A.; la Cour-Harbo, A. *A risk-aware path planning
strategy for UAVs in urban environments.* J. Intell. Robot. Syst. 95:629–643,
2019. (Already cited in Paper #1, ref [48].)

[9] Zhou, Y. et al. *A risk-based unmanned aerial vehicle path planning
scheme for complex air–ground environments.* Risk Analysis 45:321–342, 2025.
https://doi.org/10.1111/risa.17685 (Already in Paper #1, ref [47].)

[10] Julian, K.D.; Kochenderfer, M.J. *Distributed Wildfire Surveillance with
Autonomous Aircraft Using Deep Reinforcement Learning.* Journal of Guidance,
Control, and Dynamics, 2019. https://arc.aiaa.org/doi/10.2514/1.G004106

[11] *A Deep Reinforcement Learning Algorithm for Trajectory Planning of
Swarm UAV Fulfilling Wildfire Reconnaissance.* Electronics 13(13):2568, 2024.
https://www.mdpi.com/2079-9292/13/13/2568

[12] *Fire front path planning and tracking control of UAVs using deep
reinforcement learning.* 2024.

[13] *Model-Free UAV Navigation in Unknown Complex Environments Using
Vision-Based Reinforcement Learning.* Drones 9(8):566, 2025.
https://www.mdpi.com/2504-446X/9/8/566

[14] Patra, S.; Basak, S.; Patel, H. *A Survey on Reinforcement Learning
Methods for UAV Systems.* ACM Comput. Surv. 57:1–42, 2025. (Already in Paper
#1, ref [11].)

[15] *A Quantitative Analysis of Reinforcement Learning Reward Shaping.*
Preprint, October 2025.
https://www.preprints.org/manuscript/202510.2185

[16] Henderson, P. et al. *Deep Reinforcement Learning That Matters.* AAAI
2018. (Already in Paper #1, ref [23].)

[17] Li et al. *UAVBench: An Open Benchmark Dataset for Autonomous and
Agentic AI UAV Systems via LLM-Generated Flight Scenarios.* arXiv:2511.11252,
2025. https://arxiv.org/html/2511.11252v1 **⚠ Naming collision — our project
has been renamed FLARE.**

[18] Bonsignorio, F.; del Pobil, A.P.; Zereik, E. *Towards reproducible
robotics research.* Nature Machine Intelligence 7:15–22, 2025. (Already in
Paper #1, ref [27].)

[19] Hu, J. et al. *UAV path planning based on third-party risk modeling.*
Scientific Reports 13:22658, 2023. (Already in Paper #1, ref [49].)

[20] Karaağaç, C.; Yılmaz, N.; Hançerlioğulları, A. *Risk-Aware Enabled Path
Planning for Drones Flight in Unknown Environment.* JIRS 111:56, 2025.
(Already in Paper #1, ref [16].)

[21] Melissaris, C. et al. *Risk-aware UAV trajectory optimization using open
urban GIS data and target level of safety constraints.* Drones 9:666, 2025.
(Already in Paper #1, ref [15].)

[22] *Unmanned-Aerial-Vehicle Online Trajectory Planning Using Confidence
Bounds of Chance-Constrained Geofences.* Journal of Guidance, Control, and
Dynamics, 2025. https://arc.aiaa.org/doi/10.2514/1.G007853

[23] *QUAV: Quantum-Assisted Path Planning and Optimization for UAV
Navigation with Obstacle Avoidance.* arXiv:2508.21361, 2025.
https://arxiv.org/abs/2508.21361

[24] *Dynamic Depth Quantum Approximate Optimization Algorithm for Solving
Constrained Shortest Path Problem.* arXiv:2511.08657, 2025.
https://arxiv.org/abs/2511.08657

[25] *Quantum Grid Path Planning Using Parallel QAOA Circuits Based on
Minimum Energy Principle.* arXiv:2510.07413, 2025.
https://arxiv.org/html/2510.07413v1

[26] *Quantum Approximate Optimization Algorithm applied to multi-objective
routing for large scale 6G networks.* Computer Networks, 2025.
https://doi.org/10.1016/j.comnet.2025.111345

[27] *Decision-Making-Based Path Planning for Autonomous UAVs: A Survey.*
arXiv:2508.09304, 2025. https://arxiv.org/html/2508.09304v1
