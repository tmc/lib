// Package cuptiactivity collects bounded CUPTI activity records and exports
// complete records to Perfetto.
//
// Start requires exclusive ownership of process-wide CUPTI activity APIs.
// Synchronize application work before Session.Close when complete device
// timestamps are required. Close drains buffers, not application CUDA work;
// failed cleanup retains ownership for retry. Inspect Session.Result for losses,
// pending buffers, incomplete records and errors before using a trace.
//
// Record decoding currently accepts only CUPTI version 130200 with matching
// native record sizes. Other versions return an error, not guessed layouts.
// WriteRecord produces a standalone CUPTI-relative timeline; it does not align
// CUPTI timestamps with host clocks. This package does not collect hardware
// metrics; see github.com/tmc/lib/nvidia/x/cuptirange for Range Profiling.
package cuptiactivity
