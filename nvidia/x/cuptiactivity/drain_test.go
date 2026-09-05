package cuptiactivity

import (
	"errors"
	"fmt"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestActivityOwnerRetry(t *testing.T) {
	pool, err := newBufferPool(64, 1)
	if err != nil {
		t.Fatal(err)
	}
	ptr, size := pool.request()
	failure := errors.New("native failure")
	calls := make(map[cupti.ActivityKind]int)
	flushes, releases := 0, 0
	o := activityOwner{
		pool:  pool,
		kinds: []cupti.ActivityKind{cupti.ActivityKindDriver, cupti.ActivityKindRuntime},
		disable: func(k cupti.ActivityKind) error {
			calls[k]++
			if k == cupti.ActivityKindRuntime && calls[k] == 1 {
				return failure
			}
			return nil
		},
		flush: func() error {
			flushes++
			if flushes == 1 {
				return failure
			}
			return nil
		},
		release: func() { releases++ },
	}
	if err := o.close(); !errors.Is(err, failure) {
		t.Fatalf("close = %v", err)
	}
	if releases != 0 {
		t.Fatal("released after native failure")
	}
	if err := o.close(); err == nil {
		t.Fatal("released an outstanding buffer")
	}
	if releases != 0 {
		t.Fatal("released with outstanding buffer")
	}
	if err := pool.finish(ptr, size, 0, nil); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := o.close(); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if releases != 1 || flushes != 3 {
		t.Fatalf("releases=%d flushes=%d", releases, flushes)
	}
	if calls[cupti.ActivityKindDriver] != 1 || calls[cupti.ActivityKindRuntime] != 2 {
		t.Fatalf("disable calls = %v", calls)
	}
}

func TestStartActivitiesRollback(t *testing.T) {
	for _, retry := range []bool{false, true} {
		t.Run(fmt.Sprint(retry), func(t *testing.T) {
			pool, err := newBufferPool(64, 1)
			if err != nil {
				t.Fatal(err)
			}
			startFailure, closeFailure := errors.New("enable failed"), errors.New("disable failed")
			enabled, disabled, released := 0, 0, 0
			kinds := []cupti.ActivityKind{cupti.ActivityKindDriver, cupti.ActivityKindRuntime}
			layouts := map[cupti.ActivityKind]uint{kinds[0]: 8, kinds[1]: 8}
			o, err := startActivities(kinds, layouts, pool,
				func(k cupti.ActivityKind) error {
					enabled++
					if k == kinds[1] {
						return startFailure
					}
					return nil
				},
				func(k cupti.ActivityKind) error {
					disabled++
					if k != kinds[0] {
						t.Error("disabled a kind not enabled by this owner")
					}
					if retry && disabled == 1 {
						return closeFailure
					}
					return nil
				}, func() error { return nil }, func() { released++ })
			if !errors.Is(err, startFailure) {
				t.Fatalf("start error = %v", err)
			}
			if enabled != 2 || disabled != 1 {
				t.Fatalf("enabled=%d disabled=%d", enabled, disabled)
			}
			if retry {
				if o == nil || !errors.Is(err, closeFailure) || released != 0 {
					t.Fatalf("lost rollback owner: %v", err)
				}
				if err := o.close(); err != nil {
					t.Fatal(err)
				}
			} else if o != nil {
				t.Fatal("returned already released owner")
			}
			if released != 1 {
				t.Fatalf("releases=%d", released)
			}
		})
	}
}

func TestStartActivitiesValidation(t *testing.T) {
	pool, err := newBufferPool(64, 1)
	if err != nil {
		t.Fatal(err)
	}
	k := cupti.ActivityKindDriver
	for _, kinds := range [][]cupti.ActivityKind{nil, {k, k}, {cupti.ActivityKindMemcpy}} {
		calls := 0
		native := func(cupti.ActivityKind) error { calls++; return nil }
		o, err := startActivities(kinds, map[cupti.ActivityKind]uint{k: 8}, pool, native, native,
			func() error { calls++; return nil }, func() { calls++ })
		if err == nil || o != nil || calls != 0 {
			t.Fatalf("kinds=%v owner=%v error=%v native calls=%d", kinds, o, err, calls)
		}
	}
}

func TestCloseWaitsForCallbacks(t *testing.T) {
	pool, err := newBufferPool(64, 1)
	if err != nil {
		t.Fatal(err)
	}
	busy := errors.New("callback still active")
	checks, releases := 0, 0
	o := activityOwner{pool: pool, flush: func() error { return nil },
		quiesce: func() error {
			checks++
			if checks == 1 {
				return busy
			}
			return nil
		},
		release: func() { releases++ },
	}
	if err := o.close(); !errors.Is(err, busy) {
		t.Fatalf("close = %v", err)
	}
	if o.closed || releases != 0 {
		t.Fatal("closed before callback returned")
	}
	if err := o.close(); err != nil {
		t.Fatal(err)
	}
	if !o.closed || releases != 1 {
		t.Fatal("did not release after retry")
	}
}
