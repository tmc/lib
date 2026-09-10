//go:build !darwin && !tpu

package main

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
)

// TestCUDATracerDisabledByDefault checks the off state: no path means no
// tracer, and no tracer means no file and no device synchronization.
func TestCUDATracerDisabledByDefault(t *testing.T) {
	tracer, err := newCUDATracer(context.Background(), "", nil)
	if err != nil {
		t.Fatalf("newCUDATracer with no path: %v", err)
	}
	if tracer != nil {
		t.Fatalf("newCUDATracer with no path returned %#v, want nil", tracer)
	}
	if tracer.enabled() {
		t.Error("nil tracer reports enabled")
	}
}

// TestCUDATracerRequiresStream keeps a requested trace from being silently
// dropped: asking for a trace without a stream is an error, not a no-op.
func TestCUDATracerRequiresStream(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trace.pftrace")
	if _, err := newCUDATracer(context.Background(), path, nil); err == nil {
		t.Error("newCUDATracer with a path and no stream succeeded, want error")
	}
}

// TestCUDATracerRegionPassesThrough checks that a disabled tracer runs the work
// and reports its result unchanged.
func TestCUDATracerRegionPassesThrough(t *testing.T) {
	var tracer *cudaTracer
	ran := false
	if err := tracer.region(context.Background(), "decode", "test", func() error {
		ran = true
		return nil
	}); err != nil {
		t.Fatalf("region: %v", err)
	}
	if !ran {
		t.Fatal("region did not run the work")
	}

	want := errors.New("forward failed")
	if got := tracer.region(context.Background(), "decode", "test", func() error { return want }); !errors.Is(got, want) {
		t.Errorf("region error = %v, want %v", got, want)
	}
}

func TestCUDATracerCloseIsIdempotent(t *testing.T) {
	var tracer *cudaTracer
	if err := tracer.close(); err != nil {
		t.Errorf("close on a nil tracer: %v", err)
	}
	var eng *ModelEngine
	if err := eng.closeTracer(); err != nil {
		t.Errorf("closeTracer on a nil engine: %v", err)
	}
}
