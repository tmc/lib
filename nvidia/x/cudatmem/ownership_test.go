package cudatmem_test

import (
	"errors"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatmem"
)

func newAllocator(t *testing.T) *cudatmem.Allocator {
	t.Helper()
	a, err := cudatmem.New(cudatmem.SM100A)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func acquire(t *testing.T, a *cudatmem.Allocator) *cudatmem.Permit {
	t.Helper()
	p, err := a.AcquirePermit(32)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestTMEMReleaseRequiresOwningPermit(t *testing.T) {
	a := newAllocator(t)
	p1, p2 := acquire(t, a), acquire(t, a)
	x1, err := a.Allocate(p1, 32, 64, 1)
	if err != nil {
		t.Fatal(err)
	}
	x2, err := a.Allocate(p2, 32, 64, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Release(p2, x1, 32); !errors.Is(err, cudatmem.ErrPermission) {
		t.Fatalf("Release(foreign permit) = %v, want ErrPermission", err)
	}
	if err := a.Release(p1, x2, 32); !errors.Is(err, cudatmem.ErrPermission) {
		t.Fatalf("Release(foreign permit) = %v, want ErrPermission", err)
	}
	if err := a.Release(p1, x1, 32); err != nil {
		t.Fatalf("Release(owner) = %v", err)
	}
	if err := a.Release(p2, x2, 32); err != nil {
		t.Fatalf("Release(owner) = %v", err)
	}
	if err := a.Release(p1, x1, 32); !errors.Is(err, cudatmem.ErrClosed) {
		t.Fatalf("duplicate Release = %v, want ErrClosed", err)
	}
	for _, p := range []*cudatmem.Permit{p1, p2} {
		if err := a.Relinquish(p); err != nil {
			t.Fatal(err)
		}
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestTMEMColumnOrder(t *testing.T) {
	type step struct {
		release bool // release the most recent allocation instead of allocating
		columns uint32
		ok      bool
	}
	tests := []struct {
		name  string
		steps []step
	}{
		{"equal", []step{{columns: 64, ok: true}, {columns: 64, ok: true}}},
		{"decreasing", []step{{columns: 64, ok: true}, {columns: 32, ok: true}}},
		{"increasing", []step{{columns: 32, ok: true}, {columns: 64}}},
		{"history survives release", []step{{columns: 64, ok: true}, {release: true, ok: true}, {columns: 128}}},
		{"refusal keeps history", []step{{columns: 64, ok: true}, {columns: 128}, {columns: 64, ok: true}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := newAllocator(t)
			p := acquire(t, a)
			var live []*cudatmem.Allocation
			for i, s := range tt.steps {
				if s.release {
					x := live[len(live)-1]
					live = live[:len(live)-1]
					if err := a.Release(p, x, 32); err != nil {
						t.Fatalf("step %d: Release = %v", i, err)
					}
					continue
				}
				x, err := a.Allocate(p, 32, s.columns, 1)
				if s.ok {
					if err != nil {
						t.Fatalf("step %d: Allocate(%d) = %v", i, s.columns, err)
					}
					live = append(live, x)
					continue
				}
				if !errors.Is(err, cudatmem.ErrInvalidUnit) {
					t.Fatalf("step %d: Allocate(%d) = %v, want ErrInvalidUnit", i, s.columns, err)
				}
			}
			for _, x := range live {
				if err := a.Release(p, x, 32); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}

func TestTMEMColumnOrderPerPermit(t *testing.T) {
	a := newAllocator(t)
	p1, p2 := acquire(t, a), acquire(t, a)
	if _, err := a.Allocate(p1, 32, 32, 1); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Allocate(p2, 32, 64, 1); err != nil {
		t.Fatalf("other permit's history applied: %v", err)
	}
}

func TestTMEMRelinquishKeepsDeallocation(t *testing.T) {
	a := newAllocator(t)
	p, other := acquire(t, a), acquire(t, a)
	x, err := a.Allocate(p, 32, 64, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := x.BeginUser(); err != nil {
		t.Fatal(err)
	}
	if err := a.Relinquish(p); err != nil {
		t.Fatalf("Relinquish with live allocation = %v", err)
	}
	if err := a.Relinquish(p); !errors.Is(err, cudatmem.ErrPermission) {
		t.Fatalf("duplicate Relinquish = %v, want ErrPermission", err)
	}
	if _, err := a.Allocate(p, 32, 32, 1); !errors.Is(err, cudatmem.ErrPermission) {
		t.Fatalf("Allocate after Relinquish = %v, want ErrPermission", err)
	}
	if err := a.Release(other, x, 32); !errors.Is(err, cudatmem.ErrPermission) {
		t.Fatalf("Release(foreign permit) after Relinquish = %v, want ErrPermission", err)
	}
	if err := a.Release(p, x, 31); !errors.Is(err, cudatmem.ErrParticipation) {
		t.Fatalf("Release(participants=31) = %v, want ErrParticipation", err)
	}
	if err := a.Release(p, x, 32); !errors.Is(err, cudatmem.ErrBusy) {
		t.Fatalf("Release with live user = %v, want ErrBusy", err)
	}
	if err := a.Close(); !errors.Is(err, cudatmem.ErrBusy) {
		t.Fatalf("Close with live allocation = %v, want ErrBusy", err)
	}
	if err := x.CompleteUser(); err != nil {
		t.Fatal(err)
	}
	if err := a.Release(p, x, 32); err != nil {
		t.Fatalf("Release after Relinquish = %v", err)
	}
	if err := a.Release(p, x, 32); !errors.Is(err, cudatmem.ErrClosed) {
		t.Fatalf("duplicate Release = %v, want ErrClosed", err)
	}
	if err := a.Relinquish(other); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestTMEMRelinquishIgnoresOtherPermits(t *testing.T) {
	a := newAllocator(t)
	p1, p2 := acquire(t, a), acquire(t, a)
	x, err := a.Allocate(p1, 32, 32, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Relinquish(p2); err != nil {
		t.Fatalf("Relinquish blocked by another permit's allocation: %v", err)
	}
	if err := a.Release(p1, x, 32); err != nil {
		t.Fatal(err)
	}
}

func TestTMEMCloseAfterAllRelinquished(t *testing.T) {
	a := newAllocator(t)
	p := acquire(t, a)
	x, err := a.Allocate(p, 32, 32, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Relinquish(p); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); !errors.Is(err, cudatmem.ErrBusy) {
		t.Fatalf("Close with live allocation and no permits = %v, want ErrBusy", err)
	}
	if err := a.Release(p, x, 32); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestTMEMRelinquishedForeignRelease(t *testing.T) {
	a := newAllocator(t)
	p, q := acquire(t, a), acquire(t, a)
	x, err := a.Allocate(p, 32, 32, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Relinquish(q); err != nil {
		t.Fatal(err)
	}
	if err := a.Release(q, x, 32); !errors.Is(err, cudatmem.ErrPermission) {
		t.Fatalf("Release(relinquished foreign permit) = %v, want ErrPermission", err)
	}
	if err := a.Release(p, x, 32); err != nil {
		t.Fatal(err)
	}
}

func TestTMEMCapacityFailureKeepsHistory(t *testing.T) {
	a := newAllocator(t)
	p, q := acquire(t, a), acquire(t, a)
	x, err := a.Allocate(p, 32, 256, 1)
	if err != nil {
		t.Fatal(err)
	}
	y, err := a.Allocate(q, 32, 256, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.Allocate(p, 32, 128, 1); !errors.Is(err, cudatmem.ErrInvalidUnit) {
		t.Fatalf("Allocate(full pool) = %v, want ErrInvalidUnit", err)
	}
	if err := a.Release(q, y, 32); err != nil {
		t.Fatal(err)
	}
	// The refused 128-column request is not part of p's execution order.
	z, err := a.Allocate(p, 32, 256, 1)
	if err != nil {
		t.Fatalf("Allocate(256) after capacity failure = %v", err)
	}
	for _, r := range []*cudatmem.Allocation{x, z} {
		if err := a.Release(p, r, 32); err != nil {
			t.Fatal(err)
		}
	}
}

func TestTMEMPermissionReasons(t *testing.T) {
	a := newAllocator(t)
	b := newAllocator(t)
	p, q := acquire(t, a), acquire(t, a)
	foreign := acquire(t, b)
	x, err := a.Allocate(p, 32, 32, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Relinquish(p); err != nil {
		t.Fatal(err)
	}
	_, allocNil := a.Allocate(nil, 32, 32, 1)
	_, allocForeign := a.Allocate(foreign, 32, 32, 1)
	_, allocRelinquished := a.Allocate(p, 32, 32, 1)
	tests := []struct {
		name string
		err  error
		want string
	}{
		{"allocate nil permit", allocNil, "cudatmem: allocation permission required: nil permit"},
		{"allocate foreign allocator", allocForeign, "cudatmem: allocation permission required: permit from another allocator"},
		{"allocate after relinquish", allocRelinquished, "cudatmem: allocation permission required: permit relinquished"},
		{"duplicate relinquish", a.Relinquish(p), "cudatmem: allocation permission required: permit already relinquished"},
		{"relinquish nil permit", a.Relinquish(nil), "cudatmem: allocation permission required: nil permit"},
		{"relinquish foreign allocator", a.Relinquish(foreign), "cudatmem: allocation permission required: permit from another allocator"},
		{"release nil permit", a.Release(nil, x, 32), "cudatmem: allocation permission required: nil permit"},
		{"release foreign allocator", a.Release(foreign, x, 32), "cudatmem: allocation permission required: permit from another allocator"},
		{"release foreign permit", a.Release(q, x, 32), "cudatmem: allocation permission required: allocation owned by another permit"},
	}
	for _, tt := range tests {
		if !errors.Is(tt.err, cudatmem.ErrPermission) {
			t.Errorf("%s: %v, want ErrPermission", tt.name, tt.err)
			continue
		}
		if got := tt.err.Error(); got != tt.want {
			t.Errorf("%s: %q, want %q", tt.name, got, tt.want)
		}
	}
}

// The race tests below fail only under the race detector.

func TestTMEMRaceForeignPermit(t *testing.T) {
	a, b := newAllocator(t), newAllocator(t)
	p := acquire(t, b)
	var wg sync.WaitGroup
	wg.Add(3)
	go func() { defer wg.Done(); _, _ = a.Allocate(p, 32, 32, 1) }()
	go func() { defer wg.Done(); _ = a.Relinquish(p) }()
	go func() { defer wg.Done(); _ = b.Relinquish(p) }()
	wg.Wait()
}

func TestTMEMRaceAccessors(t *testing.T) {
	a := newAllocator(t)
	p := acquire(t, a)
	x, err := a.Allocate(p, 32, 32, 1)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); _ = x.BeginUser() }()
	go func() { defer wg.Done(); _, _, _ = x.Columns(), x.Rows(), x.Offset() }()
	wg.Wait()
}
