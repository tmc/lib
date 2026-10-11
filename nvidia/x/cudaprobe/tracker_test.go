package cudaprobe

import "testing"

func allocationEvent(seq uint64, symbol string, addr uint64) Event {
	return Event{Identity: Identity{AddressSpace: 1, Process: Process{PID: 7, Generation: 99}, Symbol: symbol}, Sequence: seq, Decoded: true, ReturnKnown: true, Address: addr, Bytes: 64}
}

func TestAllocationLifetime(t *testing.T) {
	for _, tt := range []struct {
		name       string
		events     []Event
		count      int
		incomplete bool
	}{
		{"matched", []Event{allocationEvent(1, "cudaMalloc", 8), allocationEvent(2, "cudaFree", 8)}, 0, false},
		{"reuse", []Event{allocationEvent(1, "cudaMalloc", 8), allocationEvent(2, "cudaFree", 8), allocationEvent(3, "cudaMalloc", 8)}, 1, false},
		{"unmatched", []Event{allocationEvent(1, "cudaMalloc", 8)}, 1, false},
		{"cap", []Event{allocationEvent(1, "cudaMalloc", 8), allocationEvent(2, "cudaMalloc", 9)}, 1, true},
		{"gap", []Event{allocationEvent(2, "cudaMalloc", 8)}, 1, true},
		{"unknown-free", []Event{allocationEvent(1, "cudaFree", 8)}, 0, true},
		{"duplicate", []Event{allocationEvent(1, "cudaMalloc", 8), allocationEvent(2, "cudaMalloc", 8)}, 1, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			tracker, err := NewTracker(Process{PID: 7, Generation: 99}, 1)
			if err != nil {
				t.Fatal(err)
			}
			for _, e := range tt.events {
				if err := tracker.Observe(e); err != nil {
					t.Fatal(err)
				}
			}
			tracker.Exit()
			got, incomplete := tracker.Report()
			if len(got) != tt.count || incomplete != tt.incomplete {
				t.Fatalf("report=%v incomplete=%t", got, incomplete)
			}
			if err := tracker.Observe(allocationEvent(10, "cudaFree", 8)); err == nil {
				t.Fatal("accepted event after exit")
			}
		})
	}
}

func TestFailedFreeAndPIDReuse(t *testing.T) {
	tk, _ := NewTracker(Process{PID: 7, Generation: 99}, 2)
	if err := tk.Observe(allocationEvent(1, "cudaMalloc", 8)); err != nil {
		t.Fatal(err)
	}
	e := allocationEvent(2, "cudaFree", 8)
	e.Status = 1
	if err := tk.Observe(e); err != nil {
		t.Fatal(err)
	}
	e.Status = 0
	e.Sequence = 3
	e.Identity.Process.Generation++
	if err := tk.Observe(e); err == nil {
		t.Fatal("accepted reused PID")
	}
	e.Identity.Process.Generation--
	e.Sequence = 2
	if err := tk.Observe(e); err == nil {
		t.Fatal("accepted replay")
	}
	got, incomplete := tk.Report()
	if len(got) != 1 || incomplete {
		t.Fatalf("rejected event mutated tracking: %v %t", got, incomplete)
	}
	got[0].Address = 100
	again, _ := tk.Report()
	if again[0].Address != 8 {
		t.Fatal("report aliases tracker")
	}
	tk.Gap()
	_, incomplete = tk.Report()
	if !incomplete {
		t.Fatal("lost tail gap")
	}
}

func TestContextControlsDoNotInferAllocationLifetimes(t *testing.T) {
	for _, tt := range []struct {
		symbol, operation string
		incomplete        bool
	}{
		{"cuCtxDestroy_v2", "destroy", true},
		{"cuCtxSetCurrent", "set_current", true},
		{"cuCtxCreate_v2", "create", true},
		{"cuCtxGetCurrent", "get_current", false},
	} {
		t.Run(tt.symbol, func(t *testing.T) {
			tracker, err := NewTracker(Process{PID: 7, Generation: 99}, 2)
			if err != nil {
				t.Fatal(err)
			}
			original := allocationEvent(1, "cudaMalloc", 0x8000)
			if err := tracker.Observe(original); err != nil {
				t.Fatal(err)
			}
			e := allocationEvent(2, tt.symbol, 0x8000)
			e.Control = &Control{Scope: "context", Operation: tt.operation, Handle: 1}
			if err := tracker.Observe(e); err != nil {
				t.Fatal(err)
			}
			allocations, incomplete := tracker.Report()
			want := Allocation{AddressSpace: 1, Address: 0x8000, Bytes: 64, Sequence: 1}
			if len(allocations) != 1 || allocations[0] != want || incomplete != tt.incomplete {
				t.Fatalf("allocations=%+v incomplete=%v want=%+v incomplete=%v", allocations, incomplete, want, tt.incomplete)
			}
		})
	}
}
