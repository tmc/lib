//go:build linux

package main

import (
	"errors"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/nvidia/x/cuda"
)

func TestCUDAFoundationClosingAdmission(t *testing.T) {
	r := &cudaFoundation{closing: true, device: new(cuda.Device)}
	tests := []struct {
		name string
		run  func() error
	}{
		{"context", func() error { return r.withContext(func() error { t.Fatal("callback after Close"); return nil }) }},
		{"warm", r.warmModel},
		{"range", func() error { _, err := r.modelRange(0, 1); return err }},
		{"map", func() error { _, err := r.mapModelRange(0, 1); return err }},
		{"probe", r.probe},
		{"probe-model", func() error { return r.probeModel(nil) }},
		{"math", r.loadMathKernels},
		{"quantizer", r.loadQ80Quantizer},
		{"resident", r.loadResidentSet},
		{"bind", func() error { _, err := r.bindModel(); return err }},
		{"load", func() error { return r.load(nil, nil) }},
		{"stage", func() error { _, err := r.stage(nil, nil); return err }},
		{"direct", func() error { _, err := r.stageDirect(nil, nil); return err }},
		{"state", func() error { _, err := (&cudaRuntime{cudaFoundation: r}).newIndependentState(1); return err }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.run(); !errors.Is(err, dsv4.ErrClosed) {
				t.Fatalf("error = %v, want ErrClosed", err)
			}
		})
	}
}

func TestCUDAFoundationInitializationAdmission(t *testing.T) {
	if err := new(cudaFoundation).checkOpen(); err != nil {
		t.Fatalf("initializing foundation: %v", err)
	}
	if err := (*cudaFoundation)(nil).checkOpen(); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatalf("nil foundation: %v", err)
	}
}
