// Package cuptiactivity collects bounded CUPTI activity records and exports
// complete records to Perfetto.
//
// Start requires exclusive ownership of process-wide CUPTI activity APIs.
// Synchronize application work before Session.Close when complete device
// timestamps are required. Close drains buffers, not application CUDA work;
// failed cleanup retains ownership for retry. Inspect Session.Result for losses,
// pending buffers, incomplete records and errors before using a trace.
//
// Record decoding accepts CUPTI 13.x versions when the loaded library reports
// matching native record sizes. Other major versions return an error, not
// guessed layouts.
//
// In practice that window is narrower than 13.x, because the Go layouts are the
// ones cupti was generated against (cupti.ApiVersion, 130200). Measured on an
// sbsa host against each toolkit installed there:
//
//	13.0 (2025.3.1), 13.1 (2025.4.1)	 no cuptiActivityGetStructSize symbol
//	13.2 (2026.1.2)			 collects
//	13.3 (2026.2.1), 13.4 (2026.3.0)	 ActivityKindConcurrentKernel is 232
//					 bytes against a 224-byte Go layout
//
// Both ends fail closed, which is the intended behavior and not a bug to route
// around: a mismatched layout decodes fields at the wrong offsets. The narrowing
// is this package's own, not a property of CUPTI: it compares whole runtime
// records against Go layouts fixed at 13.2, where records grow append-only and a
// consumer reading only the compiled prefix spans the band. Widening it means
// decoding a prefix, not pinning CUPTI.
//
// Note that this pins CUPTI independently of the compiler, so a host whose
// nvrtc and ptxas are pinned to 13.0 still needs 13.2 on the loader path for
// collection.
//
// WriteRecord produces a standalone CUPTI-relative timeline; it does not align
// CUPTI timestamps with host clocks. This package does not collect hardware
// metrics; see github.com/tmc/lib/nvidia/x/cuptirange for Range Profiling.
package cuptiactivity
