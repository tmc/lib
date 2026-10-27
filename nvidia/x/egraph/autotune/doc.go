// Package autotune provides live hardware, toolchain, and NVLink/C2C fabric
// auto-discovery together with an online CUPTI PC-sampling warp-stall and
// roofline telemetry cost model for e-graph kernel schedule selection.
//
// Hardware discovery maps a target architecture string ([TwinForTarget]) or
// runtime compute capability and fabric attributes ([AutoDiscoverTarget]) to a
// calibrated [HardwareTwin] covering Hopper (sm_90a), Blackwell datacenter
// (sm_100a, sm_103a), Vera Rubin (sm_107a, sm_107f), and Blackwell workstation
// and DGX Spark GB10 (sm_120a, sm_121a).
//
// Online profiling combines PC-sampling stall distributions ([StallProfile])
// and range execution counters ([RangeMetrics]) in [DiagnoseBottleneck] and
// [OnlineCostModel] (aliased as [ProfileCache]) to override static analytical
// roofline estimates when register spills, HBM/L2 scoreboard latency, MIO/SMEM
// throttling, or tensor pipe saturation are observed at runtime.
package autotune
