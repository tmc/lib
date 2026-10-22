//go:build linux

package main

import (
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4expertcache"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

type cudaStreamingTestLoadError struct {
	error
	close func() error
}

func TestCUDAStreamingV41Refusal(t *testing.T) {
	_, _, err := openSessionBackend(config{backend: "cuda", family: "v41", streaming: streamingOptions{enabled: true, budget: "6"}})
	if err == nil || !strings.Contains(err.Error(), "V41 execution unsupported") {
		t.Fatal("V41 ignored streaming admission", err)
	}
}

func (e *cudaStreamingTestLoadError) Close() error  { return e.close() }
func (e *cudaStreamingTestLoadError) Unwrap() error { return e.error }

func TestCUDAStreamingConstructorRetainsModuleError(t *testing.T) {
	cause := errors.New("wrapper and unload fault")
	fail := true
	calls := 0
	partial := &cudaStreamingTestLoadError{cause, func() error {
		calls++
		if fail {
			return cause
		}
		return nil
	}}
	o, err := newCUDAStreamingExpertsWithLoader(new(cuda.Device), new(dsv4expertcache.CUDASelections), func() (*cuda.Module, error) { return nil, partial })
	if o == nil || !errors.Is(err, cause) || len(o.partial) != 1 {
		t.Fatal("partial module owner discarded", err)
	}
	selectionCloses := 0
	o.closeSelections = func() error { selectionCloses++; return nil }
	if err := o.Close(); !errors.Is(err, cause) || o.selections == nil || len(o.partial) != 1 {
		t.Fatal("selection freed before partial module", err)
	}
	fail = false
	var cleanup io.Closer
	if !errors.As(err, &cleanup) {
		t.Fatal("error lost retry owner")
	}
	if err := cleanup.Close(); err != nil {
		t.Fatal(err)
	}
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}
	if calls != 2 || selectionCloses != 1 || o.selections != nil || len(o.partial) != 0 {
		t.Fatal("partial owner released twice", calls)
	}
}

// These controls exercise actual inference lifecycle methods with instance-
// local ownership faults. They perform no CUDA allocation, compile or launch.
func TestCUDAStreamingRuntimeCloseRetainsReader(t *testing.T) {
	o := &cudaStreamingCoordinator{}
	q := o.request()
	fail := true
	consumer := &cudaStreamingTestConsumer{func() error {
		if fail {
			return errors.New("consumer uncertain")
		}
		return nil
	}}
	if err := q.run(func() (io.Closer, error) { return consumer, nil }, func(io.Closer) error { return nil }); err == nil {
		t.Fatal("missing completion fault")
	}
	arenaFrees := 0
	r := &cudaRuntime{streamingRequest: q, arena: new(cudamem.Arena), cleanupOps: &cudaChildCleanupOperations{
		arena: func(*cudamem.Arena) error { arenaFrees++; return nil },
	}}
	if err := r.Close(); err == nil || r.arena == nil || q.pending != consumer || arenaFrees != 0 {
		t.Fatal("runtime released activation owner before reader")
	}
	fail = false
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if r.arena != nil || q.pending != nil || o.active != nil || arenaFrees != 1 {
		t.Fatal("runtime close retry incomplete")
	}
}

func TestCUDAStreamingRuntimeRejectsFallbacks(t *testing.T) {
	r := &cudaRuntime{cudaFoundation: &cudaFoundation{streamingExperts: &cudaStreamingExperts{}}, model: new(cudaModel)}
	if _, err := r.captureGraph(func() { t.Fatal("streaming graph body entered") }); err == nil {
		t.Fatal("streaming graph admitted")
	}
	if err := r.evalBatchExperts(nil, 0); err == nil || !strings.Contains(err.Error(), "grouped experts unsupported") {
		t.Fatal("streaming grouped weights consumed", err)
	}
	q := r.streamingExperts.coordinator.request()
	r.streamingRequest = q
	r.model.layers[0].expertGate.device = 1
	if err := r.evalStreamingExperts(new(cudaTokenState), 0); err == nil || !strings.Contains(err.Error(), "whole tensor address") {
		t.Fatal("streaming admitted whole tensor address", err)
	}
	if q.pending != nil || q.owner.active != nil {
		t.Fatal("rejected tensor reached selection")
	}
}

func TestCUDAStreamingConstructorSkipsRoutedArtifacts(t *testing.T) {
	o := &cudaStreamingExperts{}
	cause := errors.New("stop before allocation")
	ops := cudaRuntimeConstructorOperations{
		foundation: func(string, cudaRuntimeOptions) (*cudaFoundation, error) {
			return &cudaFoundation{streamingExperts: o}, nil
		},
		bind:    func(*cudaFoundation) (*cudaModel, error) { return new(cudaModel), nil },
		moe:     func(*cudaRuntime, int) error { t.Fatal("streaming reached whole routed repack"); return nil },
		dense:   func(*cudaRuntime, int) error { return nil },
		ballast: func(*cudaRuntime) error { return nil },
		allocate: func(r *cudaRuntime) error {
			if !r.disableGraphs || r.streamingRequest == nil || r.streamingRequest.owner != &o.coordinator {
				t.Fatal("streaming request not propagated")
			}
			return cause
		},
		cleanup: &cudaChildCleanupOperations{foundation: func(*cudaFoundation) error { return nil }},
	}
	if r, err := newCUDARuntimeWithOperations("not-opened", 8, cudaRuntimeOptions{}, ops); r != nil || !errors.Is(err, cause) {
		t.Fatal(r, err)
	}
}
