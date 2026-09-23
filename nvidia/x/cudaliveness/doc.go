// Package cudaliveness asserts that an instrumentation probe was live for a run.
//
// A probe that reports no defects is reporting one of two things: the subject
// had none, or the probe never ran. Nothing in the report distinguishes them,
// and only the first is a measurement. The failure is quiet by construction --
// it produces a clean result, quickly, and a clean result is what the caller
// was hoping for.
//
// A control is a prediction made before a run about what a live probe must
// report, paired with what it did report. [Assert] returns nil only when every
// supplied control ran and held. Supplying none is itself an error: a clean
// report with no control is not a measurement.
//
// The three controls here are the ones that have caught real inert probes:
// [Planted] plants a defect the probe must find, [Counted] predicts the
// probe's own instrumentation count exactly, and [Refused] hands the probe an
// instrument it must reject. They are independent -- a probe can find a
// planted defect while instrumenting the wrong kernel, and can report a
// plausible count while loading nothing.
//
// A control that did not run is not a failing control. Its Check reports an
// error matching [cudaerrors.ErrHardwarePending], because an unrun control is
// the absence of a measurement and must never be reported as a negative one.
//
// Liveness is not repeatability. These controls establish that the probe
// observed the subject; they say nothing about whether the subject produces
// the same result twice. A probe qualified by every control here, attached to
// a subject that does not reproduce itself, still yields a different answer on
// each run.
package cudaliveness
