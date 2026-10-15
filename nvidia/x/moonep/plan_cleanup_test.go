//go:build !windows

package moonep

import (
	"errors"
	"fmt"
	"slices"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func planCleanupNoDriver(t *testing.T) {
	t.Helper()
	if driver.Loaded() {
		t.Fatal("plan cleanup host control loaded the CUDA driver")
	}
	t.Cleanup(func() {
		if driver.Loaded() {
			t.Error("plan cleanup host control loaded the CUDA driver")
		}
	})
}

func planCleanupPointers(p *CommPlan) []driver.DevicePtr {
	return []driver.DevicePtr{p.Dst, p.CuSeqlens, p.ExpertsToCopy, p.ZeroFillRanges, p.RemoteStats, p.DupGroups, p.DupLoffs, p.DupCounts}
}

func planCleanupFixture() *CommPlan {
	return &CommPlan{Dst: 0x1000, CuSeqlens: 0x2000, ExpertsToCopy: 0x3000, ZeroFillRanges: 0x4000, RemoteStats: 0x5000, DupGroups: 0x6000, DupLoffs: 0x7000, DupCounts: 0x8000}
}

func planCleanupRefusesWork(t *testing.T, p *CommPlan) {
	t.Helper()
	if !p.closed {
		t.Error("cleanup did not close admission")
	}
	// Use the current pointer snapshot so pointer changes cannot mask a reopened
	// admission gate. This is the shared gate used by NewDedup and Execute.
	if release, err := dedupAcquire(p, dedupSnapshot(p)); err == nil {
		release()
		t.Error("plan admitted new work after cleanup started")
	}
}

func TestPlanCleanupRetainsFailedPointers(t *testing.T) {
	planCleanupNoDriver(t)
	for _, tt := range []struct {
		name string
		fail []int
	}{
		{"none", nil},
		{"first", []int{0}},
		{"middle", []int{4}},
		{"last", []int{7}},
		{"multiple", []int{1, 4, 7}},
		{"all", []int{0, 1, 2, 3, 4, 5, 6, 7}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			p := planCleanupFixture()
			original := planCleanupPointers(p)
			failures := make(map[driver.DevicePtr]error)
			for _, i := range tt.fail {
				failures[original[i]] = fmt.Errorf("free slot %d", i)
			}
			var calls []driver.DevicePtr
			err := p.close(func(ptr driver.DevicePtr) error {
				if !p.closed {
					t.Error("free began before admission closed")
				}
				calls = append(calls, ptr)
				return failures[ptr]
			})
			if !slices.Equal(calls, original) {
				t.Fatalf("free calls = %x, want each original slot once: %x", calls, original)
			}
			if (err != nil) != (len(failures) != 0) {
				t.Errorf("Close error = %v, failures = %d", err, len(failures))
			}
			var remaining []driver.DevicePtr
			for i, ptr := range planCleanupPointers(p) {
				failure := failures[original[i]]
				if failure != nil {
					if !errors.Is(err, failure) {
						t.Errorf("Close error lost slot %d failure: %v", i, err)
					}
					if ptr != original[i] {
						t.Errorf("failed slot %d = %#x, want retained %#x", i, ptr, original[i])
					}
					remaining = append(remaining, original[i])
				} else if ptr != 0 {
					t.Errorf("successful slot %d still owns %#x", i, ptr)
				}
			}
			planCleanupRefusesWork(t, p)
			calls = nil
			if err := p.close(func(ptr driver.DevicePtr) error {
				calls = append(calls, ptr)
				return nil
			}); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(calls, remaining) {
				t.Errorf("retry calls = %x, want only failed slots %x", calls, remaining)
			}
			for i, ptr := range planCleanupPointers(p) {
				if ptr != 0 {
					t.Fatalf("retry retained slot %d: %#x", i, ptr)
				}
			}
			planCleanupRefusesWork(t, p)
			// No fake address reaches the public native wrapper: all slots are zero.
			if err := p.Close(); err != nil {
				t.Fatal(err)
			}
			if err := p.close(func(ptr driver.DevicePtr) error {
				t.Errorf("completed cleanup freed %#x twice", ptr)
				return nil
			}); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestPlanCleanupRepeatedFailure(t *testing.T) {
	planCleanupNoDriver(t)
	p := planCleanupFixture()
	original := planCleanupPointers(p)
	first, second := errors.New("first free failure"), errors.New("retry free failure")
	var calls []driver.DevicePtr
	for phase := 0; phase < 3; phase++ {
		calls = nil
		err := p.close(func(ptr driver.DevicePtr) error {
			calls = append(calls, ptr)
			if phase == 0 && (ptr == original[1] || ptr == original[4] || ptr == original[7]) {
				return first
			}
			if phase == 1 && ptr == original[4] {
				return second
			}
			return nil
		})
		wantCalls := [][]driver.DevicePtr{original, {original[1], original[4], original[7]}, {original[4]}}[phase]
		if !slices.Equal(calls, wantCalls) {
			t.Errorf("phase %d calls = %x, want %x", phase, calls, wantCalls)
		}
		wantErr := []error{first, second, nil}[phase]
		if !errors.Is(err, wantErr) {
			t.Errorf("phase %d error = %v, want %v", phase, err, wantErr)
		}
		planCleanupRefusesWork(t, p)
	}
}

func TestPlanCleanupActiveRefusal(t *testing.T) {
	planCleanupNoDriver(t)
	p := planCleanupFixture()
	original := planCleanupPointers(p)
	release, err := dedupAcquire(p, dedupSnapshot(p))
	if err != nil {
		t.Fatal(err)
	}
	called := false
	err = p.close(func(driver.DevicePtr) error {
		called = true
		return nil
	})
	if err == nil || called || p.closed || !slices.Equal(planCleanupPointers(p), original) {
		t.Fatalf("active Close changed ownership: error=%v called=%t closed=%t pointers=%x", err, called, p.closed, planCleanupPointers(p))
	}
	release()
	// An active-work refusal does not start cleanup or permanently close admission.
	release, err = dedupAcquire(p, dedupSnapshot(p))
	if err != nil {
		t.Fatal(err)
	}
	release()
	if err := p.close(func(driver.DevicePtr) error { return nil }); err != nil {
		t.Fatal(err)
	}
	planCleanupRefusesWork(t, p)
}

func TestPlanCleanupZero(t *testing.T) {
	planCleanupNoDriver(t)
	var p CommPlan
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
	planCleanupRefusesWork(t, &p)
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestPlanCleanupConcurrentClose(t *testing.T) {
	planCleanupNoDriver(t)
	p := planCleanupFixture()
	original := planCleanupPointers(p)
	var calls []driver.DevicePtr
	var wg sync.WaitGroup
	for range 20 {
		wg.Go(func() {
			if err := p.close(func(ptr driver.DevicePtr) error {
				// The production transition holds p.mu around every free.
				calls = append(calls, ptr)
				return nil
			}); err != nil {
				t.Error(err)
			}
		})
	}
	wg.Wait()
	if !slices.Equal(calls, original) {
		t.Errorf("concurrent Close calls = %x, want each slot once %x", calls, original)
	}
	planCleanupRefusesWork(t, p)
}
