| benchmark | dynamic_hazards | causal_coupling | deterministic_stress_instrumentation | feasibility_guarantee | fair_protocol | dual_use_operational_semantics | statistical_seed_sweeps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AirSim | Scenario-dependent | Not explicit for planner-eval hazard coupling | Not explicit benchmark primitive | Not explicit benchmark contract | Not explicit cross-planner protocol contract | Application-dependent | Possible, not standard benchmark output |
| CARLA | Scenario-dependent | Limited explicit multi-layer hazard coupling for planner-eval | Not explicit benchmark primitive | Not explicit benchmark contract | Partial protocol support, not unified planner-eval contract | Application-dependent | Possible, not standard benchmark output |
| Common Grid RL Benchmarks | Usually simplified | Usually absent | Usually absent | Usually absent | Task-dependent | Usually absent | Common |
| Sampling-based UAV Simulators | Scenario-dependent | Typically not explicit benchmark contract | Typically absent | Typically absent | Typically absent | Application-dependent | Possible |
| Dynamic Path Planning Testbeds | Yes | Often partial | Often partial | Rarely explicit | Often partial | Often not central | Varies |
| UAVBench | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

**Conclusion:** In this positioning matrix, UAVBench is the only listed benchmark that explicitly satisfies all required dimensions simultaneously as first-class benchmark contracts.