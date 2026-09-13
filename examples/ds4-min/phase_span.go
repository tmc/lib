package main

// The fine phase names live on markCUDAPhase, whose call sites already
// bracket every region worth naming: indexer-query, indexer-score,
// attention, compression and the rest. Until now those names reached only
// the CUDA-event profiler behind the phaseMark hook, which is installed by
// a test and nil in an ordinary run, so nothing carried them into a trace.
//
// A phase is the interval between two marks, not a begin/end pair, so the
// host span for one closes when the next mark arrives or when the
// enclosing batch or token ends. That keeps the names and boundaries the
// same as the CUDA-event profiler's, with no per-phase GPU
// synchronization: nothing here touches a stream.
//
// At op level this emits on the order of 700,000 intervals for a
// 2048-token prefill, which is why it is only usable with the streaming
// recorder.

// openPhase is the phase span currently open, if any. ds4-min evaluates on
// one goroutine, and the span recorders take their own locks.
var openPhase tracedEnd

// markPhaseSpan closes the open phase span and opens one named name. It is
// a no-op when no recorder wants op-level spans, so the ordinary run pays
// one comparison per mark.
func markPhaseSpan(name string) {
	if !tracing(levelOp) {
		return
	}
	openPhase.end()
	openPhase = beginTraced(levelOp, name, nil)
}

// closePhaseSpan ends the open phase span. The last phase in a batch has no
// following mark to close it, so the batch closes it instead; without this
// that span would be counted as begun and never ended, which the streaming
// accounting reports rather than hides.
func closePhaseSpan() {
	openPhase.end()
	openPhase = tracedEnd{}
}
