package nvml_test

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/nvml"
)

func compareRemappedRows(pending, failure uint32, v2 remappedRowsInfoV2) error {
	if v2.BPending != pending {
		return fmt.Errorf("pending: v1=%d v2=%d", pending, v2.BPending)
	}
	if v2.BFailureOccurred != failure {
		return fmt.Errorf("failure occurred: v1=%d v2=%d", failure, v2.BFailureOccurred)
	}
	return nil
}

func compareAccountingStats(v1 nvml.AccountingStats, v2 accountingStatsV2) error {
	if v1.StartTime != v2.StartTime {
		return fmt.Errorf("start time v1=%d v2=%d", v1.StartTime, v2.StartTime)
	}
	if v1.IsRunning != v2.IsRunning || v1.IsRunning != 0 {
		return nil
	}
	for _, f := range []struct {
		name   string
		v1, v2 uint64
	}{
		{"gpu utilization", uint64(v1.GpuUtilization), uint64(v2.GpuUtilization)},
		{"memory utilization", uint64(v1.MemoryUtilization), uint64(v2.MemoryUtilization)},
		{"max memory usage", v1.MaxMemoryUsage, v2.MaxMemoryUsage},
		{"time", v1.Time, v2.Time},
	} {
		if f.v1 != f.v2 {
			return fmt.Errorf("terminated: %s v1=%d v2=%d", f.name, f.v1, f.v2)
		}
	}
	return nil
}

// TestVersionedOracleMutation checks the comparisons used by the physical
// test with distinct values. The negative cases write a v1 result layout
// into the v2 destination, modeling a wrong-version implementation.
func TestVersionedOracleMutation(t *testing.T) {
	t.Run("remapped rows", func(t *testing.T) {
		v1 := [4]uint32{11, 13, 1, 1}
		v2 := remappedRowsInfoV2{5, 6, 7, 6, 1, 1}
		if err := compareRemappedRows(v1[2], v1[3], v2); err != nil {
			t.Fatal(err)
		}
		var mutant remappedRowsInfoV2
		copy(unsafe.Slice((*byte)(unsafe.Pointer(&mutant)), unsafe.Sizeof(mutant)), unsafe.Slice((*byte)(unsafe.Pointer(&v1)), unsafe.Sizeof(v1)))
		if err := compareRemappedRows(v1[2], v1[3], mutant); err == nil {
			t.Fatal("accepted v1 layout in v2 result")
		}
	})
	t.Run("accounting stats", func(t *testing.T) {
		v1 := nvml.AccountingStats{GpuUtilization: 11, MemoryUtilization: 13, MaxMemoryUsage: 17, Time: 19, StartTime: 23}
		v2 := accountingStatsV2{Pid: 29, GpuUtilization: 11, MemoryUtilization: 13, MaxMemoryUsage: 17, Time: 19, StartTime: 23}
		if err := compareAccountingStats(v1, v2); err != nil {
			t.Fatal(err)
		}
		var mutant accountingStatsV2
		copy(unsafe.Slice((*byte)(unsafe.Pointer(&mutant)), unsafe.Sizeof(mutant)), unsafe.Slice((*byte)(unsafe.Pointer(&v1)), unsafe.Sizeof(v1)))
		if err := compareAccountingStats(v1, mutant); err == nil {
			t.Fatal("accepted v1 layout in v2 result")
		}
	})
}
