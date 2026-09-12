package cuptipcsampling

import (
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

// These are 64-bit header layouts, not evidence of runtime sampling support.
// PC and Data use sizeof including trailing padding; Params use the end of
// their final field. A collector must not confuse the two size conventions.
func TestSamplingOutputLayouts(t *testing.T) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		t.Skip("64-bit layout")
	}
	var pc cupti.PCSamplingPCData
	var data cupti.PCSamplingData
	for _, tt := range []struct {
		name      string
		got, want uintptr
	}{
		{"reason size", unsafe.Sizeof(cupti.PCSamplingStallReason{}), 8},
		{"PC size", unsafe.Sizeof(pc), 64},
		{"PC function name", unsafe.Offsetof(pc.FunctionName), 32},
		{"PC reason count", unsafe.Offsetof(pc.StallReasonCount), 40},
		{"PC reasons", unsafe.Offsetof(pc.StallReason), 48},
		{"PC correlation", unsafe.Offsetof(pc.CorrelationId), 56},
		{"data size", unsafe.Sizeof(data), 80},
		{"data PCs", unsafe.Offsetof(data.PPcData), 56},
		{"data non-user samples", unsafe.Offsetof(data.NonUsrKernelsTotalSamples), 64},
		{"data buffer full", unsafe.Offsetof(data.HardwareBufferFull), 72},
		{"get data params", unsafe.Sizeof(cupti.PCSamplingGetDataParams{}), 32},
		{"configuration", unsafe.Sizeof(cupti.PCSamplingConfigurationInfo{}), 32},
		{"configuration params", unsafe.Sizeof(cupti.PCSamplingConfigurationInfoParams{}), 40},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if tt.got != tt.want {
				t.Fatalf("got %d, want %d", tt.got, tt.want)
			}
		})
	}
}
