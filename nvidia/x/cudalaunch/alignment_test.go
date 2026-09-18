package cudalaunch

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

// TestSubEightByteDevicePointerIsAccepted pins the absence of a device-pointer
// alignment check. Argument inspection runs on the host and needs no device, so
// this is the cheap half of the evidence; the expensive half is a real launch
// through a 4-byte-aligned base pointer, which the driver accepts.
//
// An earlier version rejected any device address not a multiple of 8. That rule
// enforced nothing -- it cannot see the pointee type, so it is too strict for a
// float and too loose for a float4 -- and it refused launches the hardware runs.
// Offsetting a buffer by one float is how a caller asks whether a kernel's
// vectorized load has an alignment guard, so the check made that unaskable.
func TestSubEightByteDevicePointerIsAccepted(t *testing.T) {
	for _, addr := range []uintptr{0x32ee01204, 0x1004, 0x2, 0x1} {
		got, err := inspectArgument(driver.DevicePtr(addr))
		if err != nil {
			t.Errorf("inspectArgument(0x%x): %v", addr, err)
			continue
		}
		if !got.isPointer || got.size != 8 {
			t.Errorf("inspectArgument(0x%x) = %+v, want an 8-byte pointer", addr, got)
		}
	}
}

// TestEightByteAlignedPointerStillWorks is the positive control: if the loop
// above passed because inspectArgument stopped reporting pointers at all, this
// fails too.
func TestEightByteAlignedPointerStillWorks(t *testing.T) {
	got, err := inspectArgument(driver.DevicePtr(0x32ee01200))
	if err != nil {
		t.Fatalf("inspectArgument: %v", err)
	}
	if !got.isPointer || got.size != 8 {
		t.Fatalf("got %+v, want an 8-byte pointer", got)
	}
}

// TestNoAlignmentRejectionRemains guards the wording as well as the behaviour,
// so a reintroduced check is caught even if it is spelled differently.
func TestNoAlignmentRejectionRemains(t *testing.T) {
	if _, err := inspectArgument(driver.DevicePtr(0x1)); err != nil {
		if strings.Contains(err.Error(), "aligned") {
			t.Fatalf("alignment rejection reintroduced: %v", err)
		}
		t.Fatalf("unexpected error: %v", err)
	}
}
