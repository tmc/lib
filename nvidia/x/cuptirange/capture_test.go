package cuptirange

import (
	"errors"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
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
