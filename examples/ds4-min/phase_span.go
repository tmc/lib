package main

// The fine phase names live on markCUDAPhase, whose call sites already
// bracket every region worth naming: dense-front, qkv-normalization,
// attention, attention-output, compression, the three ffn phases and the
// group tail. Until now those names reached only the CUDA-event profiler
// behind the phaseMark hook, which a test installs and an ordinary run
// leaves nil, so nothing carried them into a trace.
//
// A phase is the interval between two marks, not a begin/end pair, so the
// host span for one closes when the next mark arrives or when the
// enclosing batch or token ends. The names and boundaries are the
// CUDA-event profiler's, and nothing here touches a stream, so there is no
// per-phase GPU synchronization.
//
// These spans are deliberately host-only: they open no NVTX range. Fine
// names in the host sidecar and fine names in NVTX are separate pieces of
// work, and a 2048-token prefill would otherwise add about 715,000 ranges
// to a capture that carries 11,356 today, changing the thing being
// measured. Opening a range here is the whole of what the NVTX half would
// need, and it should be done deliberately with its own qualification.
//
// At op level a 2048-token prefill emits 715,776 of these intervals, seven
// times the buffered recorder's cap, so they are usable only with
// -spans-stream.

// A phaseEnd closes one phase span in both recorders. The zero value is
// the no-op end, so the first mark of a run needs no guard.
type phaseEnd struct {
	span   spanEnd
	stream streamEnd
}

func (e phaseEnd) end() {
	e.span.end()
	e.stream.end()
}

// openPhase is the phase span currently open, if any. ds4-min evaluates on
// one goroutine, and the span recorders take their own locks.
var openPhase phaseEnd

// markPhaseSpan closes the open phase span and opens one named name. It is
// a no-op when no recorder wants op-level spans, so an ordinary run pays
// one comparison per mark.
func markPhaseSpan(name string) {
	if !tracing(levelOp) {
		return
	}
	openPhase.end()
	openPhase = phaseEnd{
		span:   begin(levelOp, name, nil),
		stream: beginStream(levelOp, name),
	}
}

// closePhaseSpan ends the open phase span. The last phase in a batch has
// no following mark to close it, so the batch closes it instead; without
// this that span would be counted as begun and never ended, which the
// streaming accounting reports rather than hides.
func closePhaseSpan() {
	openPhase.end()
	openPhase = phaseEnd{}
}
