package cuptibench_test

import (
	"fmt"
	"testing"
)

// validateOverlap checks device atomic tickets, not elapsed-time estimates.
// The layout is counter, A entry/exit, B entry/exit, A output, B output.
func validateOverlap(v []uint32, dependent bool) error {
	if len(v) != 7 || v[0] != 4 {
		return fmt.Errorf("incomplete ticket record: %v", v)
	}
	seen := [4]bool{}
	for _, ticket := range v[1:5] {
		if ticket >= 4 || seen[ticket] {
			return fmt.Errorf("invalid tickets: %v", v)
		}
		seen[ticket] = true
	}
	if v[1] >= v[2] || v[3] >= v[4] {
		return fmt.Errorf("reversed interval: %v", v)
	}
	overlap := v[1] < v[4] && v[3] < v[2]
	if dependent {
		if overlap || v[2] >= v[3] {
			return fmt.Errorf("invalid dependent execution: %v", v)
		}
	} else if !overlap {
		return fmt.Errorf("overlap not established: %v", v)
	}
	return nil
}

func TestOverlapTickets(t *testing.T) {
	for _, tt := range []struct {
		name             string
		record           []uint32
		dependent, valid bool
	}{
		{"overlap", []uint32{4, 0, 2, 1, 3, 17, 23}, false, true},
		{"reverse-start", []uint32{4, 1, 3, 0, 2, 17, 23}, false, true},
		{"dependent", []uint32{4, 0, 1, 2, 3, 17, 34}, true, true},
		{"serialized-rejected", []uint32{4, 0, 1, 2, 3, 17, 23}, false, false},
		{"missing", []uint32{3, 0, 2, 1, ^uint32(0), 17, 23}, false, false},
		{"duplicate", []uint32{4, 0, 2, 0, 3, 17, 23}, false, false},
		{"reversed", []uint32{4, 2, 0, 1, 3, 17, 23}, false, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if err := validateOverlap(tt.record, tt.dependent); (err == nil) != tt.valid {
				t.Fatalf("valid=%t error=%v", tt.valid, err)
			}
		})
	}
}

// A native teardown failure keeps all dependencies alive until process exit.
// This is test-only retention; it deliberately provides no retry path.
var retainedOverlapResources []any

type overlapCleanup struct {
	retain    bool
	resources []any
}

type overlapTB struct {
	testing.TB
	owner *overlapCleanup
}

func (tb overlapTB) Cleanup(f func()) {
	tb.owner.resources = append(tb.owner.resources, f)
	tb.TB.Cleanup(func() {
		if !tb.owner.retain {
			f()
		}
	})
}

func (tb overlapTB) Error(args ...any) {
	tb.owner.retain = true
	tb.TB.Error(args...)
}

// Fake only the cleanup sink; tests do not need a CUDA context.
type overlapCleanupTB struct {
	testing.TB
	callbacks []func()
	errors    int
}

func (tb *overlapCleanupTB) Cleanup(f func()) { tb.callbacks = append(tb.callbacks, f) }
func (tb *overlapCleanupTB) Error(...any)     { tb.errors++ }

func TestOverlapCleanup(t *testing.T) {
	for _, fail := range []bool{false, true} {
		t.Run(fmt.Sprint(fail), func(t *testing.T) {
			sink := &overlapCleanupTB{}
			owner := &overlapCleanup{}
			tb := overlapTB{TB: sink, owner: owner}
			released := 0
			tb.Cleanup(func() { released++ })
			tb.Cleanup(func() {
				if fail {
					tb.Error(fmt.Errorf("native close failed"))
				} else {
					released++
				}
			})
			for i := len(sink.callbacks) - 1; i >= 0; i-- {
				sink.callbacks[i]()
			}
			want := 2
			if fail {
				want = 0
			}
			if released != want || owner.retain != fail || len(owner.resources) != 2 {
				t.Fatalf("released=%d retain=%t retained dependencies=%d", released, owner.retain, len(owner.resources))
			}
		})
	}
}
