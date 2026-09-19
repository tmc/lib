package cudacensus

import (
	"math"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

// calibratedRun builds a calibration whose analytic figure and estimate are both
// known, so the error between them is arithmetic rather than a fixture's opinion.
//
// The workload is a dense 10240^3 matrix multiply: 2*m*n*k = 2.147e12 FLOPs. Run
// in 100ms that is 21.47 TFLOPS analytically. The estimator sees half the FP32
// pipe active against a 40 TFLOPS peak, so it says 20 TFLOPS: an error of about
// -6.9%, which is the number the test asserts on rather than a pass/fail verdict.
func calibratedRun(t *testing.T, fp32Active float64) Calibration {
	t.Helper()
	m, err := cudaroofline.NewModel("sm_121", cudaroofline.DeviceResident)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	est := NewEstimate(m, Inputs{
		Readings: []Reading{
			{Counter: FP32Active, Value: fp32Active, Available: true},
			{Counter: DRAMBandwidthUtil, Value: 0.05, Available: true},
		},
		Attribution: fp32Attribution(),
		Interval:    10 * time.Millisecond,
	})
	return Calibration{
		Workload:      "gemm-10240-fp32",
		Device:        "GB10 sm_121a, driver 580.65.06",
		AnalyticFLOPs: 2 * 10240 * 10240 * 10240,
		Derivation:    "2*m*n*k, m=n=k=10240",
		Duration:      100 * time.Millisecond,
		Estimate:      est,
		RanAt:         time.Date(2026, 9, 18, 11, 0, 0, 0, time.UTC),
	}
}

// TestCalibrationErrorReported: the harness reports the disagreement between the
// estimate and the analytic value. Reporting it is the deliverable; a calibration
// that returned a verdict and hid the error would be worthless.
func TestCalibrationErrorReported(t *testing.T) {
	c := calibratedRun(t, 0.5)
	if err := c.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if !c.Estimate.Available {
		t.Fatalf("the calibration's own estimate is unavailable: %s", c.Estimate.Derivation.UnavailableReason)
	}

	analytic := c.AnalyticTFLOPS()
	if want := 21.47483648; math.Abs(analytic-want) > 1e-6 {
		t.Errorf("AnalyticTFLOPS = %v, want %v", analytic, want)
	}

	relErr, err := c.RelativeError()
	if err != nil {
		t.Fatalf("RelativeError: %v", err)
	}
	want := (c.Estimate.AchievedTFLOPS - analytic) / analytic
	if math.Abs(relErr-want) > 1e-12 {
		t.Errorf("RelativeError = %v, want %v", relErr, want)
	}
	if relErr >= 0 {
		t.Errorf("RelativeError = %v: this fixture underestimates, so the sign carries information and must be negative", relErr)
	}

	report := c.Report()
	for _, s := range []string{"analytic", "estimated", "relative error", "2*m*n*k"} {
		if !strings.Contains(report, s) {
			t.Errorf("Report() = %q, want it to contain %q", report, s)
		}
	}
}

// TestCalibrationDisagreementIsNotHidden: an estimator claiming more than the
// workload could have done reports a positive error, and if the figure is also
// above the roofline the estimate is rejected and there is no error to report at
// all. Both are outcomes, and neither is a zero.
func TestCalibrationDisagreementIsNotHidden(t *testing.T) {
	over := calibratedRun(t, 0.9)
	relErr, err := over.RelativeError()
	if err != nil {
		t.Fatalf("RelativeError: %v", err)
	}
	if relErr <= 0 {
		t.Errorf("RelativeError = %v, want positive: the estimator claimed 36 TFLOPS against an analytic 21.47", relErr)
	}

	rejected := calibratedRun(t, 0.5)
	rejected.Estimate.Rejected = true
	rejected.Estimate.Available = false
	rejected.Estimate.RejectReason = "above the ceiling"
	if _, err := rejected.RelativeError(); err == nil {
		t.Error("RelativeError returned a number for a rejected estimate")
	}
	if !strings.Contains(rejected.Report(), "no comparison") {
		t.Errorf("Report() = %q, want it to say there is no comparison", rejected.Report())
	}
}

// TestCalibrationRequiresItsOwnDerivation: an analytic FLOP count nobody can
// recompute is a claim, not a reference.
func TestCalibrationRequiresItsOwnDerivation(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*Calibration)
	}{
		{"no workload", func(c *Calibration) { c.Workload = "" }},
		{"no device", func(c *Calibration) { c.Device = "" }},
		{"no analytic count", func(c *Calibration) { c.AnalyticFLOPs = 0 }},
		{"no derivation", func(c *Calibration) { c.Derivation = "" }},
		{"no duration", func(c *Calibration) { c.Duration = 0 }},
		{"no timestamp", func(c *Calibration) { c.RanAt = time.Time{} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := calibratedRun(t, 0.5)
			tc.mutate(&c)
			if err := c.Validate(); err == nil {
				t.Errorf("Validate accepted a calibration with %s", tc.name)
			}
			if _, err := c.RelativeError(); err == nil {
				t.Errorf("RelativeError returned a number for a calibration with %s", tc.name)
			}
		})
	}
}

// TestReceiptComesFromACalibration closes the loop lane 35 asks for: the receipt a
// threshold needs is produced by a calibration, so a threshold cannot be
// introduced without one having run.
func TestReceiptComesFromACalibration(t *testing.T) {
	c := calibratedRun(t, 0.5)
	r, err := c.Receipt("fp32 from fp16_tensor", "a kernel whose tensor pipes dominate while fp32_active stays high")
	if err != nil {
		t.Fatalf("Receipt: %v", err)
	}
	if r.Error <= 0 {
		t.Errorf("Receipt.Error = %v, want the calibration's absolute error", r.Error)
	}
	if _, err := NewThreshold(0.8, r); err != nil {
		t.Errorf("NewThreshold with a calibration-derived receipt: %v", err)
	}

	// A receipt that does not say where it fails is refused at the point it would
	// be built, not later.
	if _, err := c.Receipt("fp32 from fp16_tensor", ""); err == nil {
		t.Error("Receipt accepted an empty failure mode")
	}
}
