// Package cudatrace records host-side CUDA submission regions and explicit
// timestamped GPU spans in Perfetto traces.
//
// # Timeline Correlation
//
// Span and SpanSource use the host clock around the supplied function. They do
// not insert CUDA events or synchronize a stream, so they do not measure GPU
// execution. Use RecordSpan with externally measured and calibrated GPU
// timestamps when a hardware interval is available.
//
// # Go runtime/trace Integration
//
// Alongside GPU tracks, cudatrace links with Go's standard runtime/trace regions and tasks
// allowing seamless navigation between CPU goroutines, memory copies, and GPU execution.
package cudatrace
