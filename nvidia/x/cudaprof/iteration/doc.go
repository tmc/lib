// Package iteration measures a capture iteration by iteration: the steps
// of a training loop or the calls of an inference benchmark.
//
// A whole-capture window mixes the work being measured with everything
// around it, such as weight uploads, JIT compilation and the first call's
// allocations, and two captures whose programs load differently are not
// comparable over it. [Analyze] instead cuts the device timeline at each
// launch of a marker kernel (or each NVTX range of a marker name) and
// measures every iteration between two markers: its window, the device
// busy and idle time in it, its launches, the exclusive device time of each
// op group, and the host API calls made during it. The iteration after the
// last marker has no end and is not measured, and the first [Options.Skip]
// iterations are warm-up. The rest are the steady state, which [Report]
// sums.
//
// [Report.Gate] is a completeness check: the marker fired as often as
// expected, and every steady iteration launched the same number of kernels.
// A capture that fails it holds different work in different iterations, and
// its per-iteration means describe no iteration that ran.
//
// With [Options.MinGap] set, [Report.Gaps] lists the steady iterations'
// device idle gaps and the host API calls begun in each. Host CPU samples
// in [Options.Host], such as perf records read by package perfscript,
// label each gap with the host stacks sampled while the device waited.
//
// [Report.Advise] reads the steady state for four costs a per-kernel profile
// does not show: device allocation, CUDA graph construction and host
// synchronization repeated every iteration, and a large idle share.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package iteration
