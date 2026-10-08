package cudatrial

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

func TestHardwareStatus_ReportsPendingWithoutDevice(t *testing.T) {
	// In the absence of an active GPU lock and verified device node,
	// CheckHardwareStatus must return StatusPendingHardware.
	orig := hardwareProbeFunc
	defer func() { hardwareProbeFunc = orig }()

	// Explicit mock of no hardware present.
	hardwareProbeFunc = func() bool { return false }
	if status := CheckHardwareStatus(); status != StatusPendingHardware {
		t.Fatalf("expected StatusPendingHardware, got %v", status)
	}

	// Mock hardware present.
	hardwareProbeFunc = func() bool { return true }
	if status := CheckHardwareStatus(); status != StatusHardwareAvailable {
		t.Fatalf("expected StatusHardwareAvailable, got %v", status)
	}
}

func TestControl_FakedHardwareCoverageRejected(t *testing.T) {
	fakeCoverage := []byte{0, 1, 2, 4}

	// Attempting to validate or record hardware coverage with StatusPendingHardware must fail with ErrHardwarePending.
	err := ValidateHardwareCoverage(fakeCoverage, StatusPendingHardware)
	if err == nil || !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("expected ErrHardwarePending for StatusPendingHardware, got %v", err)
	}

	p, err := NewPersistentCoverage(32, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Notify(SignalStart); err != nil {
		t.Fatal(err)
	}

	err = p.RecordHardwareTrace(make([]byte, 32), StatusPendingHardware)
	if err == nil || !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("expected ErrHardwarePending on RecordHardwareTrace with StatusPendingHardware, got %v", err)
	}
}

func TestControl_MismatchedMapSizePanics(t *testing.T) {
	b1, err := NewEdgeBitmap(32)
	if err != nil {
		t.Fatal(err)
	}
	b2, err := NewEdgeBitmap(64)
	if err != nil {
		t.Fatal(err)
	}

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic on mismatched map size, got nil")
		}
	}()

	b1.MustMerge(b2)
}
