package cudacensus

import (
	"fmt"
	"math"
	"time"
)

// Calibration compares an estimate against a kernel whose FLOP count is known
// analytically. It is the only thing in this package that can say how wrong the
// estimator is, so an estimate offered without one is offered without an error
// bar.
//
// AnalyticFLOPs is derived from the workload's shape, not from a counter: for a
// dense matrix multiply of (m, k) by (k, n) it is 2*m*n*k, and the derivation
// belongs in Derivation so a reader can check the arithmetic rather than trust
// the number.
type Calibration struct {
	Workload      string        // the named kernel, reproducible by a reader
	Device        string        // the identity it ran on
	AnalyticFLOPs uint64        // from the workload's shape
	Derivation    string        // how AnalyticFLOPs was computed, e.g. "2*m*n*k, m=n=k=4096"
	Duration      time.Duration // device execution span, from the device clock
	Estimate      Estimate      // what the estimator said
	RanAt         time.Time
}

// Validate reports whether the calibration records enough for its error to mean
// anything.
func (c Calibration) Validate() error {
	switch {
	case c.Workload == "":
		return fmt.Errorf("cudacensus: calibration has no workload")
	case c.Device == "":
		return fmt.Errorf("cudacensus: calibration of %q has no device identity", c.Workload)
	case c.AnalyticFLOPs == 0:
		return fmt.Errorf("cudacensus: calibration of %q has no analytic FLOP count", c.Workload)
	case c.Derivation == "":
		return fmt.Errorf("cudacensus: calibration of %q does not say how its analytic FLOP count was derived", c.Workload)
	case c.Duration <= 0:
		return fmt.Errorf("cudacensus: calibration of %q has no device duration", c.Workload)
	case c.RanAt.IsZero():
		return fmt.Errorf("cudacensus: calibration of %q has no timestamp", c.Workload)
	}
	return nil
}

// AnalyticTFLOPS returns the throughput the workload's shape and duration imply.
func (c Calibration) AnalyticTFLOPS() float64 {
	return float64(c.AnalyticFLOPs) / c.Duration.Seconds() / 1e12
}

// RelativeError returns the estimator's signed relative error against the
// analytic figure: positive when the estimator claimed more throughput than the
// workload could have achieved.
//
// It reports an error rather than a number when the estimate was unavailable or
// rejected, because there is nothing to compare and a zero would read as a
// perfect estimator. That case is a result to report, not a calibration failure:
// the honest outcome of calibrating an unavailable estimate is that the
// estimator declined to guess.
func (c Calibration) RelativeError() (float64, error) {
	if err := c.Validate(); err != nil {
		return 0, err
	}
	switch {
	case c.Estimate.Rejected:
		return 0, fmt.Errorf("cudacensus: calibration of %q has no error to report: the estimate was rejected above the ceiling (%s)", c.Workload, c.Estimate.RejectReason)
	case !c.Estimate.Available:
		return 0, fmt.Errorf("cudacensus: calibration of %q has no error to report: the estimate was unavailable (%s)", c.Workload, c.Estimate.Derivation.UnavailableReason)
	}
	analytic := c.AnalyticTFLOPS()
	if analytic == 0 {
		return 0, fmt.Errorf("cudacensus: calibration of %q has a zero analytic throughput", c.Workload)
	}
	return (c.Estimate.AchievedTFLOPS - analytic) / analytic, nil
}

// Receipt turns the calibration into the receipt a threshold needs. failsWhen is
// the caller's statement of where the separation is known or suspected to break,
// and it is required: a calibration that found no failure mode did not look for
// one.
func (c Calibration) Receipt(separates, failsWhen string) (CalibrationReceipt, error) {
	relErr, err := c.RelativeError()
	if err != nil {
		return CalibrationReceipt{}, err
	}
	r := CalibrationReceipt{
		Workload:   c.Workload,
		Device:     c.Device,
		Separates:  separates,
		FailsWhen:  failsWhen,
		Samples:    len(c.Estimate.Derivation.Readings),
		Error:      math.Abs(relErr),
		RecordedAt: c.RanAt,
	}
	if err := r.Validate(); err != nil {
		return CalibrationReceipt{}, err
	}
	return r, nil
}

// Report renders the calibration as a line for the handoff: the analytic figure,
// the estimate, and the error between them. It prints the disagreement rather
// than the verdict, because a calibration that hides its error is a calibration
// nobody can use.
func (c Calibration) Report() string {
	if err := c.Validate(); err != nil {
		return fmt.Sprintf("calibration invalid: %v", err)
	}
	relErr, err := c.RelativeError()
	if err != nil {
		return fmt.Sprintf("%s on %s: analytic %.4f TFLOPS over %v; no comparison: %v",
			c.Workload, c.Device, c.AnalyticTFLOPS(), c.Duration, err)
	}
	return fmt.Sprintf("%s on %s: analytic %.4f TFLOPS (%s) over %v, estimated %.4f TFLOPS by %s, relative error %+.2f%% (%s)",
		c.Workload, c.Device, c.AnalyticTFLOPS(), c.Derivation, c.Duration,
		c.Estimate.AchievedTFLOPS, c.Estimate.Derivation.Attribution.Method,
		relErr*100, c.Estimate.Confidence)
}
