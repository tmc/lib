// Package cuptibench collects CUPTI hardware metrics for replayable CUDA workloads.
// Collect provides testing-independent results; Benchmark adds ordinary Go
// benchmark timing and reporting. Whole, Regions, and Kernels select operation,
// named-region, and kernel attribution respectively.
//
// Each subbenchmark must prepare its own inputs independently of its siblings.
// Measure a full pass directly: summing isolated kernels does not preserve cache
// state, launch gaps, or stream overlap. Counter replay and event timing are
// separate executions. Preparation must document this comparability boundary.
//
// Plans and CUDA resources have explicit Close ownership. Callbacks run with the
// retained primary context current on a locked OS thread. Callers provide device
// isolation and must not launch unrelated work during measurements. Native support
// and collection permission are determined at runtime, without configuration changes.
package cuptibench
