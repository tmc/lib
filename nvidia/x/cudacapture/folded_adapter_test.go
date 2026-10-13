package cudacapture_test

import (
	"bytes"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

// twoDeviceJoins returns constructed, host-only correlation joins; no
// device or profiler produced them. The two "site;[GPU]K" joins overlap
// in time on different devices, so their weight depends on Device:
// per-device unions sum to 400, a single union would be 300.
func twoDeviceJoins() []cudacapture.CorrelationJoin {
	return []cudacapture.CorrelationJoin{
		{CorrelationID: 1, KernelName: "K", Callstack: []string{"main", "site"}, DeviceStart: 100, DeviceEnd: 300, DeviceDurationNs: 200, Device: 0},
		{CorrelationID: 2, KernelName: "K", Callstack: []string{"main", "site"}, DeviceStart: 200, DeviceEnd: 400, DeviceDurationNs: 200, Device: 1},
		{CorrelationID: 3, KernelName: "L", DeviceStart: 1000, DeviceEnd: 1050, DeviceDurationNs: 50, Device: 1},
	}
}

const twoDeviceFolded = "[GPU]L 50\nmain;site;[GPU]K 400\n"

func TestFoldedAdapterMatchesDirectExport(t *testing.T) {
	joins := twoDeviceJoins()

	// The direct path maps each join to a KernelExecution independently of
	// the adapter.
	execs := make([]cudaprof.KernelExecution, len(joins))
	for i, j := range joins {
		execs[i] = cudaprof.KernelExecution{
			KernelName: j.KernelName,
			Callstack:  j.Callstack,
			StartNs:    j.DeviceStart,
			EndNs:      j.DeviceEnd,
			Device:     j.Device,
		}
	}
	var direct, adapter bytes.Buffer
	if err := cudaprof.ExportFoldedStacks(&direct, execs); err != nil {
		t.Fatalf("cudaprof.ExportFoldedStacks: %v", err)
	}
	if err := cudacapture.ExportFoldedStacks(&adapter, joins); err != nil {
		t.Fatalf("cudacapture.ExportFoldedStacks: %v", err)
	}
	if direct.String() != twoDeviceFolded {
		t.Fatalf("direct output = %q, want %q", direct.String(), twoDeviceFolded)
	}
	if !bytes.Equal(adapter.Bytes(), direct.Bytes()) {
		t.Fatalf("adapter output = %q, direct output = %q", adapter.String(), direct.String())
	}
}

func TestFoldedAdapterKeepsDevice(t *testing.T) {
	// Collapsing both overlapping joins onto one device gives 300, not 400.
	// The adapter must report 400 for the two-device input.
	joins := twoDeviceJoins()
	var two bytes.Buffer
	if err := cudacapture.ExportFoldedStacks(&two, joins); err != nil {
		t.Fatalf("two devices: %v", err)
	}
	for i := range joins {
		joins[i].Device = 0
	}
	var one bytes.Buffer
	if err := cudacapture.ExportFoldedStacks(&one, joins); err != nil {
		t.Fatalf("one device: %v", err)
	}
	if got, want := two.String(), twoDeviceFolded; got != want {
		t.Fatalf("two-device output = %q, want %q", got, want)
	}
	if got, want := one.String(), "[GPU]L 50\nmain;site;[GPU]K 300\n"; got != want {
		t.Fatalf("one-device output = %q, want %q", got, want)
	}
}
