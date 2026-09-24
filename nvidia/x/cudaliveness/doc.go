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
// [Executed] is the control for a patched kernel image. It asks whether the
// patched bytes executed, or whether the patch went into bytes the device
// never ran, and needs three things: a prediction of the patched output with
// its derivation, a different output from the same launch with the unpatched
// image, and no copy of the kernel text the patch left stale. It cannot say
// that the patch is correct, only that it executed. A patched output can
// equal its prediction by coincidence -- the patched site may never be
// reached, or the prediction may be the unpatched output -- which is why the
// unpatched run is required and a matching prediction alone does not pass.
// Executed is the one control that can fail without running: a missing
// derivation or a stale copy is known before any launch, and no launch would
// change the answer, so it reports that failure rather than a pending run.
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
