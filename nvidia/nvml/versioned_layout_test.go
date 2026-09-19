package nvml_test

import (
	"testing"
	"unsafe"
)

// remappedRowsInfoV2 is nvmlRemappedRowsInfo_v2_t from the pinned cuda-13.3
// header. It is written out here rather than bound because this package is
// generated from the unpinned header, which does not declare it; moving the pin
// is the decision TestPhysicalVersionedSymbols exists to inform.
type remappedRowsInfoV2 struct {
	CorrActiveRemaps   uint32
	CorrInactiveRemaps uint32
	UncActiveRemaps    uint32
	UncInactiveRemaps  uint32
	BPending           uint32
	BFailureOccurred   uint32
}

// accountingStatsV2 is nvmlAccountingStats_v2_t from the same header. Unlike
// the v1 struct it leads with the pid, which is an input, and it drops the
// reserved tail.
type accountingStatsV2 struct {
	Pid               uint32
	IsRunning         uint32
	GpuUtilization    uint32
	MemoryUtilization uint32
	MaxMemoryUsage    uint64
	SampleCount       uint32
	SumGpuUtil        uint64
	SumFbUtil         uint64
	Time              uint64
	StartTime         uint64
}

// TestVersionedStructLayout checks the two structs transcribed by hand from the
// pinned header against the layout a C compiler gives them.
//
// The physical test passes these to the driver by pointer. A transcription slip
// there does not fail: the driver writes into whatever offsets it believes in,
// Go reads the ones it believes in, and the comparison against the _v1 form
// reports a disagreement between NVML versions that is really a disagreement
// between this file and the header. That failure would be reported from a
// machine nobody can debug on, so the layout is checked here, where it runs
// without a GPU.
//
// The expected offsets are the LP64 layout of the C declarations: four
// unsigned int, then an unsigned long long forced to offset 16, then an
// unsigned int at 24 with four bytes of tail padding before the next
// unsigned long long at 32.
func TestVersionedStructLayout(t *testing.T) {
	for _, tt := range []struct {
		name   string
		got    uintptr
		want   uintptr
		detail string
	}{
		{"remappedRowsInfoV2 size", unsafe.Sizeof(remappedRowsInfoV2{}), 24, "six unsigned int"},
		{"accountingStatsV2 size", unsafe.Sizeof(accountingStatsV2{}), 64, "four unsigned int, ull, unsigned int, pad, four ull"},
		{"accountingStatsV2.MaxMemoryUsage", unsafe.Offsetof(accountingStatsV2{}.MaxMemoryUsage), 16, "first 8-byte field"},
		{"accountingStatsV2.SampleCount", unsafe.Offsetof(accountingStatsV2{}.SampleCount), 24, "unsigned int between two ull"},
		{"accountingStatsV2.SumGpuUtil", unsafe.Offsetof(accountingStatsV2{}.SumGpuUtil), 32, "after the tail padding"},
		{"accountingStatsV2.SumFbUtil", unsafe.Offsetof(accountingStatsV2{}.SumFbUtil), 40, ""},
		{"accountingStatsV2.Time", unsafe.Offsetof(accountingStatsV2{}.Time), 48, ""},
		{"accountingStatsV2.StartTime", unsafe.Offsetof(accountingStatsV2{}.StartTime), 56, ""},
	} {
		if tt.got != tt.want {
			t.Errorf("%s = %d, want %d (%s)", tt.name, tt.got, tt.want, tt.detail)
		}
	}
}
