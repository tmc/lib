package cuptiactivity_test

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

func memoryEvent(ts uint64, op cupti.ActivityMemoryOperationType) cuptiactivity.ScopedRecord {
	return cuptiactivity.ScopedRecord{Session: "session", Run: "run", Process: "process-generation", Record: cuptiactivity.Record{Kind: cupti.ActivityKindMemory2, MemoryOperation: op, MemoryKind: cupti.ActivityMemoryKindDevice, Address: 1<<63 + 1, Bytes: 1<<63 + 9, Timestamp: ts}}
}

func TestMemoryLifetimes(t *testing.T) {
	a, r := cupti.ActivityMemoryOperationTypeAllocation, cupti.ActivityMemoryOperationTypeRelease
	records := []cuptiactivity.ScopedRecord{memoryEvent(40, r), memoryEvent(10, a), memoryEvent(20, r), memoryEvent(30, a), memoryEvent(50, a)}
	want := []cuptiactivity.AllocationLifetime{{AllocationIndex: 1, ReleaseIndex: 2, Generation: 1}, {AllocationIndex: 3, ReleaseIndex: 0, Generation: 2}, {AllocationIndex: 4, ReleaseIndex: -1, Generation: 3}}
	got, err := cuptiactivity.MemoryLifetimes(records)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("lifetimes=%v err=%v", got, err)
	}
	if records[0].Record.Bytes != 1<<63+9 || records[1].Record.Timestamp != 10 {
		t.Fatal("records changed")
	}
	for _, tt := range []struct {
		name   string
		change func(*cuptiactivity.ScopedRecord)
	}{
		{"process generation", func(r *cuptiactivity.ScopedRecord) { r.Process = "other" }},
		{"session", func(r *cuptiactivity.ScopedRecord) { r.Session = "other" }},
		{"run", func(r *cuptiactivity.ScopedRecord) { r.Run = "other" }},
		{"device", func(r *cuptiactivity.ScopedRecord) { r.Record.DeviceID++ }},
		{"context", func(r *cuptiactivity.ScopedRecord) { r.Record.ContextID++ }},
		{"memory kind", func(r *cuptiactivity.ScopedRecord) { r.Record.MemoryKind = cupti.ActivityMemoryKindPinned }},
		{"address", func(r *cuptiactivity.ScopedRecord) { r.Record.Address++ }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			records := []cuptiactivity.ScopedRecord{memoryEvent(10, a), memoryEvent(20, r)}
			tt.change(&records[1])
			got, err := cuptiactivity.MemoryLifetimes(records)
			if err != nil || len(got) != 2 || got[0].ReleaseIndex != -1 || got[1].AllocationIndex != -1 || got[1].Generation != 0 {
				t.Fatalf("cross-scope join=%v err=%v", got, err)
			}
		})
	}
}

func TestMemoryLifetimesRejectAmbiguity(t *testing.T) {
	a := cupti.ActivityMemoryOperationTypeAllocation
	for _, tt := range []struct {
		name   string
		change func(*cuptiactivity.ScopedRecord)
	}{
		{"overlap", func(*cuptiactivity.ScopedRecord) {}},
		{"same timestamp", func(r *cuptiactivity.ScopedRecord) { r.Record.Timestamp = 10 }},
		{"missing timestamp", func(r *cuptiactivity.ScopedRecord) { r.Record.Timestamp = 0 }},
		{"missing address", func(r *cuptiactivity.ScopedRecord) { r.Record.Address = 0 }},
		{"missing scope", func(r *cuptiactivity.ScopedRecord) { r.Process = "" }},
		{"unknown operation", func(r *cuptiactivity.ScopedRecord) { r.Record.MemoryOperation = 99 }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			records := []cuptiactivity.ScopedRecord{memoryEvent(10, a), memoryEvent(20, a)}
			tt.change(&records[1])
			if got, err := cuptiactivity.MemoryLifetimes(records); err == nil || got != nil {
				t.Fatalf("ambiguous lifetimes=%v err=%v", got, err)
			}
		})
	}
}

func ExampleMemoryLifetimes() {
	r := cuptiactivity.ScopedRecord{Session: "capture", Run: "live", Process: "generation-1", Record: cuptiactivity.Record{Kind: cupti.ActivityKindMemory2, MemoryOperation: cupti.ActivityMemoryOperationTypeAllocation, Timestamp: 10, Address: 4096}}
	lifetimes, err := cuptiactivity.MemoryLifetimes([]cuptiactivity.ScopedRecord{r})
	fmt.Println(lifetimes[0].ReleaseIndex, err)
	// Output: -1 <nil>
}

func ExampleAllocationLifetime() {
	l := cuptiactivity.AllocationLifetime{AllocationIndex: 3, ReleaseIndex: -1, Generation: 1}
	fmt.Println(l.Generation, l.ReleaseIndex)
	// Output: 1 -1
}
