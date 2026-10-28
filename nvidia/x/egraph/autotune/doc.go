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
//
// The closed loop is [Daemon]. Each [Daemon.Step] ranks schedule candidates
// by [ScheduleWeights.Predict], builds and samples the best unmeasured ones
// through a [Builder] and a [LiveCollector], disqualifies candidates that
// spill ([ErrSpillDisqualified]) or whose outputs disagree with the serving
// kernel ([ErrNumericalDrift]), refines the weights from each sample's
// [StallHistogram] ([ScheduleWeights.Refine]), and hands a winner that beats
// the serving kernel by DaemonConfig.ImproveFloor to a [Swapper], such as
// [ServiceSwapper] over a cudahotswap service. [AttributionMap] sums per-PC
// stall samples ([HistogramsFromBatch]) into per-op histograms, [KnobPreamble]
// emits a candidate's TMA stage, cta_group and setmaxnreg knobs as CUDA C++,
// and [TuneCache] persists outcomes keyed by [HardwareTwin.Fingerprint] so
// [Daemon.Warm] can skip exploration.
//
// The loop runs on the host and is tested against a synthetic device model.
// This package provides no Builder or LiveCollector backed by NVRTC and CUPTI;
// stall thresholds and penalties are heuristics, not measured.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia.
package autotune
