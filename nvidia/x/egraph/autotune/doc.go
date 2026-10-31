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
// [NVRTCBuilder] compiles candidates with NVRTC to a cubin and counts STL
// and LDL instructions in its SASS with nvdisasm ([CountSpills]); spilling
// candidates are rejected before they run. [CUPTICollector] loads the cubin,
// times launches with CUDA events and collects stall histograms with CUPTI
// PC sampling. It does not use CUPTI range profiling; RangeMetrics carries
// only the register count. Programs using it must call driver.ProfilerStop
// before exit. It is a [Preloader]: the daemon loads every candidate of a
// step before sampling any, because loading a module into a context after
// PC sampling was configured on it made later enables fail on GB10 (see
// package cuptipcsampling). Give the first [Daemon.Step] every candidate.
//
// Ordinary tests drive the loop against a synthetic device model. With
// CUDA_EGRAPH_AUTOTUNE_NATIVE=1 under an inherited gpu-lock,
// TestAutotuneNativeQualification tunes a tiled SGEMM on the GPU (written
// for sm_121a). It checks that the chosen candidate matches a host
// reference, that a register-capped candidate is rejected for spilling, and
// that PC sampling returns samples. It does not assert a speedup. It also
// compiles [KnobPreamble] output with NVRTC for sm_121a and sm_90a.
//
// Not included: the spec's gate of at least 5% lower median latency than
// the analytical model on a 4-layer Llama block within 8 trials. It needs a
// tuning budget this package does not spend. Stall thresholds and penalties
// are heuristics, not measured.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia.
package autotune
