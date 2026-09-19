package cudacensus

import (
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

// goodReceipt is a receipt that validates, so that tests of what a threshold
// refuses are not accidentally testing the receipt.
func goodReceipt() CalibrationReceipt {
	return CalibrationReceipt{
		Workload:   "gemm-4096-fp16",
		Device:     "GB10 sm_121a, driver 580.65.06",
		Separates:  "fp16_tensor from int8_tensor",
		FailsWhen:  "a kernel interleaves HMMA and IMMA in equal measure",
		Samples:    64,
		Error:      0.11,
		RecordedAt: time.Date(2026, 9, 18, 10, 0, 0, 0, time.UTC),
	}
}

// TestPrecisionAttributionFromInstructionMixLabeled: the preferred method says
// so in the result, and carries no threshold.
func TestPrecisionAttributionFromInstructionMixLabeled(t *testing.T) {
	mix := InstructionMix{Counts: map[cudaroofline.Precision]uint64{
		cudaroofline.FP16Tensor: 900,
		cudaroofline.FP32:       100,
	}}
	attr, err := AttributeFromMix(mix)
	if err != nil {
		t.Fatalf("AttributeFromMix: %v", err)
	}
	if attr.Method != FromInstructionMix {
		t.Errorf("Method = %q, want %q", attr.Method, FromInstructionMix)
	}
	if attr.Precision != cudaroofline.FP16Tensor {
		t.Errorf("Precision = %q, want %q", attr.Precision, cudaroofline.FP16Tensor)
	}
	if got, want := attr.Share, 0.9; got != want {
		t.Errorf("Share = %v, want %v", got, want)
	}
	if attr.Threshold != 0 {
		t.Errorf("Threshold = %v, want 0: the static mix applies none", attr.Threshold)
	}
	if !attr.Known {
		t.Error("Known = false for a dominant mix")
	}
}

// TestPrecisionAttributionFromCounterRatioLabeled: the fallback says so, and
// carries the threshold and the receipt that justified it.
func TestPrecisionAttributionFromCounterRatioLabeled(t *testing.T) {
	th, err := NewThreshold(0.8, goodReceipt())
	if err != nil {
		t.Fatalf("NewThreshold: %v", err)
	}
	readings := []Reading{
		{Counter: HMMATensorActive, Value: 0.90, Available: true},
		{Counter: IMMATensorActive, Value: 0.05, Available: true},
		{Counter: AnyTensorActive, Value: 0.95, Available: true},
	}
	attr, err := AttributeFromCounters(readings, th)
	if err != nil {
		t.Fatalf("AttributeFromCounters: %v", err)
	}
	if attr.Method != FromCounterRatio {
		t.Fatalf("Method = %q, want %q (%s)", attr.Method, FromCounterRatio, attr.Reason)
	}
	if attr.Precision != cudaroofline.FP16Tensor {
		t.Errorf("Precision = %q, want %q", attr.Precision, cudaroofline.FP16Tensor)
	}
	if attr.Threshold != 0.8 {
		t.Errorf("Threshold = %v, want 0.8", attr.Threshold)
	}
	if attr.Receipt.Workload == "" {
		t.Error("the attribution carries no calibration receipt")
	}
}

// TestThresholdRequiresCalibrationReceipt: a threshold constant cannot enter this
// package without a recorded calibration, and attribution refuses to run without
// one. This is the mechanism behind lane 35's rule, so it is checked at both ends.
func TestThresholdRequiresCalibrationReceipt(t *testing.T) {
	if _, err := NewThreshold(0.8, CalibrationReceipt{}); err == nil {
		t.Error("NewThreshold accepted an empty receipt")
	}

	// Each field is individually load-bearing: dropping any one of them leaves a
	// receipt a reader cannot check.
	for _, tc := range []struct {
		name   string
		mutate func(*CalibrationReceipt)
	}{
		{"no workload", func(r *CalibrationReceipt) { r.Workload = "" }},
		{"no device", func(r *CalibrationReceipt) { r.Device = "" }},
		{"no separation", func(r *CalibrationReceipt) { r.Separates = "" }},
		{"no failure mode", func(r *CalibrationReceipt) { r.FailsWhen = "" }},
		{"no samples", func(r *CalibrationReceipt) { r.Samples = 0 }},
		{"no timestamp", func(r *CalibrationReceipt) { r.RecordedAt = time.Time{} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := goodReceipt()
			tc.mutate(&r)
			if _, err := NewThreshold(0.8, r); err == nil {
				t.Errorf("NewThreshold accepted a receipt with %s", tc.name)
			}
		})
	}

	// And the zero Threshold cannot be applied by a caller who skipped the
	// constructor.
	readings := []Reading{{Counter: HMMATensorActive, Value: 0.9, Available: true}}
	attr, err := AttributeFromCounters(readings, Threshold{})
	if err == nil {
		t.Error("AttributeFromCounters accepted a zero Threshold")
	}
	if attr.Known {
		t.Error("an uncalibrated attribution reported a known precision")
	}
}

// TestThresholdRejectsValueOutsideShare: the value is a share of activity, so a
// figure outside (0, 1] is not a threshold of anything.
func TestThresholdRejectsValueOutsideShare(t *testing.T) {
	for _, v := range []float64{-1, 0, 1.5} {
		if _, err := NewThreshold(v, goodReceipt()); err == nil {
			t.Errorf("NewThreshold(%v) accepted a value outside (0, 1]", v)
		}
	}
	if _, err := NewThreshold(1, goodReceipt()); err != nil {
		t.Errorf("NewThreshold(1): %v, want accepted", err)
	}
}

// TestAnyTensorActiveAttributesNothing is the central honesty constraint as a
// test. A device that reports only ANY_TENSOR_UTIL has said the pipes were busy,
// and no threshold can extract a format from that.
func TestAnyTensorActiveAttributesNothing(t *testing.T) {
	th, err := NewThreshold(0.8, goodReceipt())
	if err != nil {
		t.Fatalf("NewThreshold: %v", err)
	}
	readings := []Reading{
		{Counter: AnyTensorActive, Value: 0.95, Available: true},
		{Counter: HMMATensorActive, Available: false, Reason: "GPM metric not supported"},
		{Counter: IMMATensorActive, Available: false, Reason: "GPM metric not supported"},
	}
	attr, err := AttributeFromCounters(readings, th)
	if err != nil {
		t.Fatalf("AttributeFromCounters: %v", err)
	}
	if attr.Known {
		t.Fatalf("attributed %q from ANY_TENSOR_UTIL alone", attr.Precision)
	}
	if attr.Method != Unattributed {
		t.Errorf("Method = %q, want %q", attr.Method, Unattributed)
	}
	if !strings.Contains(attr.Reason, string(AnyTensorActive)) {
		t.Errorf("Reason = %q, want it to name %s", attr.Reason, AnyTensorActive)
	}
}

// TestMixedPrecisionDefeatsRatioThresholding is the negative control lane 35
// names. The ratio method must report that it could not decide, not pick the
// larger of two comparable formats.
func TestMixedPrecisionDefeatsRatioThresholding(t *testing.T) {
	th, err := NewThreshold(0.8, goodReceipt())
	if err != nil {
		t.Fatalf("NewThreshold: %v", err)
	}
	readings := []Reading{
		{Counter: HMMATensorActive, Value: 0.45, Available: true},
		{Counter: IMMATensorActive, Value: 0.40, Available: true},
	}
	attr, err := AttributeFromCounters(readings, th)
	if err != nil {
		t.Fatalf("AttributeFromCounters: %v", err)
	}
	if attr.Known {
		t.Fatalf("attributed %q to a kernel running two formats in comparable measure", attr.Precision)
	}
	if attr.Share == 0 {
		t.Error("the unattributed result does not report the share it measured")
	}
	if !strings.Contains(attr.Reason, "threshold") {
		t.Errorf("Reason = %q, want it to name the threshold it fell below", attr.Reason)
	}

	// The same shape passes once the threshold is calibrated lower, and the
	// result then says it rests on that threshold. Attribution is a function of
	// the constant, which is why the constant needs a receipt.
	loose, err := NewThreshold(0.5, goodReceipt())
	if err != nil {
		t.Fatalf("NewThreshold: %v", err)
	}
	if attr, err := AttributeFromCounters(readings, loose); err != nil || !attr.Known {
		t.Errorf("a 0.5 threshold did not attribute the same shape: %v %+v", err, attr)
	}
}

// TestTiedInstructionMixAttributesNothing: a tie is a real kernel, and resolving
// it by map order would make the answer depend on the runtime.
func TestTiedInstructionMixAttributesNothing(t *testing.T) {
	mix := InstructionMix{Counts: map[cudaroofline.Precision]uint64{
		cudaroofline.FP16Tensor: 500,
		cudaroofline.INT8Tensor: 500,
	}}
	attr, err := AttributeFromMix(mix)
	if err == nil {
		t.Fatalf("AttributeFromMix attributed %q to a tie", attr.Precision)
	}
	if attr.Method != Unattributed {
		t.Errorf("Method = %q, want %q", attr.Method, Unattributed)
	}
}

// TestEmptyInstructionMixAttributesNothing: no instructions counted is not
// evidence of FP32.
func TestEmptyInstructionMixAttributesNothing(t *testing.T) {
	if attr, err := AttributeFromMix(InstructionMix{}); err == nil {
		t.Fatalf("AttributeFromMix attributed %q to an empty mix", attr.Precision)
	}
}
