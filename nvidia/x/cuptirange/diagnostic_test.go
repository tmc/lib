package cuptirange

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestFailureTraceRetained(t *testing.T) {
	for _, partial := range []bool{false, true} {
		k := &kernelNames{limit: 2, current: []kernelLaunch{{API: "cuLaunchKernel", Symbol: "first"}}}
		if partial {
			k.failure = errors.New("trace interrupted")
		} else {
			if err := k.end(); err != nil {
				t.Fatal(err)
			}
		}
		c := Capture{}
		store := &counterStore{image: []byte{1, 2}, kernelNames: k}
		c.retainCollection(store, 0, errors.New("native Stop failure"))
		if c.DiagnosticTrace == nil || len(c.DiagnosticTrace.Launches) != 1 || c.DiagnosticTrace.Launches[0].Symbol != "first" {
			t.Fatalf("trace=%+v", c.DiagnosticTrace)
		}
		want := uint(1)
		if partial {
			want = 0
		}
		if c.DiagnosticTrace.CompletedPasses != want {
			t.Fatal("partial trace promoted to complete")
		}
		k.current[0].Symbol = "changed"
		store.image[0] = 9
		if c.DiagnosticTrace.Launches[0].Symbol != "first" || c.DiagnosticImage[0] != 1 {
			t.Fatal("borrowed diagnostic storage")
		}
		if c.Image != nil || c.Ranges != 0 {
			t.Fatal("failed evidence became metrics")
		}
	}
}

func TestCleanupFailureDemotesCapture(t *testing.T) {
	store := &counterStore{image: []byte{3}, kernelNames: &kernelNames{passes: 1, current: []kernelLaunch{{Symbol: "kernel"}}}}
	failure := errors.New("disable failed")
	store.host = 1
	store.api.disable = func(*cupti.RangeProfilerDisableParams) error { return failure }
	c := Capture{store: store, context: &testCaptureContext{}, RangeInfo: []RangeInfo{{ID: "kernel#1"}}}
	c.retainCollection(store, 1, nil)
	if err := c.finishCollection(nil); !errors.Is(err, failure) {
		t.Fatalf("cleanup error=%v", err)
	}
	if c.store == nil || c.context == nil {
		t.Fatal("cleanup owner lost")
	}
	store.api.disable = func(*cupti.RangeProfilerDisableParams) error { return nil }
	defer c.Close()
	if c.Image != nil || c.Ranges != 0 || c.RangeInfo != nil || c.DiagnosticImage[0] != 3 || c.DiagnosticTrace.CompletedPasses != 1 {
		t.Fatalf("capture=%+v", &c)
	}
}
