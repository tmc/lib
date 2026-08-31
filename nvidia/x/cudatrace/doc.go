// Package cudatrace correlates Go runtime tracing, flight recording, and CUDA driver
// kernel execution into unified timeline traces viewable in Perfetto.
//
// # Timeline Correlation
//
// cudatrace instruments kernel dispatches and stream memory transfers with GPU events
// to measure exact hardware start and end timestamps, translating them into calibrated
// host nanoseconds and exporting to [perfetto.Writer].
//
// # Go runtime/trace Integration
//
// Alongside GPU tracks, cudatrace links with Go's standard runtime/trace regions and tasks
// allowing seamless navigation between CPU goroutines, memory copies, and GPU execution.
package cudatrace
