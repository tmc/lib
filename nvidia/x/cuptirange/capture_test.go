package cuptirange

import (
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/internal/cuptiowner"
	"github.com/tmc/lib/nvidia/x/cuda"
)

type testCaptureContext struct {
	runs, releases int
	releaseError   error
}

func (c *testCaptureContext) Run(fn func() error) error { c.runs++; return fn() }
func (c *testCaptureContext) Release() error {
	c.releases++
	if c.releases == 1 {
		return c.releaseError
	}
	return nil
}

func TestCaptureCleanup(t *testing.T) {
	failure := errors.New("native failure")
	ctx := &testCaptureContext{releaseError: failure}
	disables := 0
	c := &Capture{context: ctx, store: &counterStore{host: 1, api: counterAPI{
		disable: func(*cupti.RangeProfilerDisableParams) error {
			disables++
			if disables == 1 {
				return failure
			}
			return nil
		},
	}}}
	if err := c.Close(); !errors.Is(err, failure) {
		t.Fatalf("disable error=%v", err)
	}
	if ctx.releases != 0 || c.store == nil || c.context == nil {
		t.Fatal("released context after failed disable")
	}
	if err := c.Close(); !errors.Is(err, failure) {
		t.Fatalf("release error=%v", err)
	}
	if c.store != nil || c.context == nil {
		t.Fatal("lost context cleanup owner")
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := c.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if disables != 2 || ctx.runs != 2 || ctx.releases != 2 || c.context != nil {
		t.Fatalf("disable=%d run=%d release=%d", disables, ctx.runs, ctx.releases)
	}
}

func TestCaptureInvalid(t *testing.T) {
	var p *Plan
	if c, err := p.Capture(nil, Limits{}, nil); c != nil || err == nil {
		t.Fatal("nil capture accepted")
	}
	var c *Capture
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestCaptureRejectsKnownInsufficientPassBudget(t *testing.T) {
	names, err := newMetricNames([]string{"metric"})
	if err != nil {
		t.Fatal(err)
	}
	defer names.close()
	p := &Plan{host: 1, names: names, passes: 3}
	// An uninitialized device ensures preflight does not need a native context.
	dev := new(cuda.Device)
	result, err := p.Capture(dev, Limits{MaxPasses: 2, MaxRanges: 1, MaxRangeTreeNodes: 1, MaxCounterBytes: 1024}, func(uint) error {
		t.Fatal("workload executed despite known insufficient budget")
		return nil
	})
	if result != nil || err == nil || !strings.Contains(err.Error(), "requires 3 passes, limit is 2") {
		t.Fatalf("capture=%v error=%v", result, err)
	}
}

func TestCaptureRetainsCollectionEvidence(t *testing.T) {
	for _, failed := range []bool{false, true} {
		store := &counterStore{image: []byte{1, 2, 3}, quality: CollectionQuality{Decoded: true}}
		var err error
		if failed {
			err = errors.New("collection incomplete")
			store.quality.DroppedRanges = 1<<63 + 7
		}
		var c Capture
		c.retainCollection(store, 3, err)
		if c.Quality != store.quality {
			t.Fatal("lost quality evidence")
		}
		if failed {
			if c.Image != nil || c.Ranges != 0 || len(c.DiagnosticImage) != 3 {
				t.Fatalf("failed capture=%+v", &c)
			}
		} else if c.DiagnosticImage != nil || c.Ranges != 3 || len(c.Image) != 3 {
			t.Fatalf("successful capture=%+v", &c)
		}
		store.image[0] = 99
		if failed && c.DiagnosticImage[0] != 1 || !failed && c.Image[0] != 1 {
			t.Fatal("capture borrows counter buffer")
		}
	}
}

func TestCaptureLeaseSurvivesCleanupFailure(t *testing.T) {
	lease, err := cuptiowner.AcquireDevice(98765)
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	c := &Capture{context: &testCaptureContext{releaseError: errors.New("release")}, lease: lease}
	if err := c.Close(); err == nil {
		t.Fatal("cleanup failure missing")
	}
	if another, err := cuptiowner.AcquireDevice(98765); err == nil {
		another.Release()
		t.Fatal("cleanup relinquished ownership")
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	another, err := cuptiowner.AcquireDevice(98765)
	if err != nil {
		t.Fatal(err)
	}
	another.Release()
}

func TestPlanRetainsCaptureCleanup(t *testing.T) {
	p := &Plan{active: true}
	if err := p.Close(); err == nil {
		t.Fatal("closed active plan")
	}
	p.active = false
	ctx := &testCaptureContext{releaseError: errors.New("release")}
	p.pending = &Capture{context: ctx}
	if err := p.Close(); err == nil || p.pending == nil {
		t.Fatal("lost cleanup owner")
	}
	if err := p.Close(); err != nil || p.pending != nil {
		t.Fatalf("retry failed: %v", err)
	}
}
