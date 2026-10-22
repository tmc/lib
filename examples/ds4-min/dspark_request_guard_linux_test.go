//go:build linux

package main

import (
	"errors"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// Closing owners must be refused before shape validation, compilation,
// context entry or launch. All buffers/devices here are absent deliberately.
func TestDSparkRequestClosingAdmission(t *testing.T) {
	r := &cudaRuntime{cudaFoundation: &cudaFoundation{closing: true}}
	w := &cudaDSparkWeights{closed: true}
	var empty cudamem.Slice[float32]
	for _, test := range []struct {
		name string
		call func() error
	}{
		{"dense", func() error { return w.encodeDense(r, "absent", empty, empty) }},
		{"head", func() error { return w.encodeHead(r, empty, 0, 1, 0, nil) }},
		{"launch-head", func() error { return w.launchHead(r, "absent", 1) }},
		{"attention", func() error { return w.encodeNoncausal(r, 0, empty, empty, empty, 1, 0, 1, 1) }},
		{"rope", func() error { return w.encodeRope(r, empty, 1, 1, 0, false) }},
		{"cache", func() error { return w.encodeStoreRaw(r, empty, empty, 1, 0, 1) }},
		{"qkv", func() error { return w.encodeStageQKV(r, 0, 0, empty, empty, empty, empty, empty) }},
		{"attention-output", func() error { return w.encodeAttentionOutput(r, 0, 0, empty, empty, empty, empty, empty) }},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := test.call(); !errors.Is(err, dsv4.ErrClosed) {
				t.Fatalf("closing request reached later admission: %v", err)
			}
		})
	}
	cause := errors.New("request failed")
	r.failed = cause
	if err := dsparkRequestOpen(r); !errors.Is(err, cause) {
		t.Fatalf("failed request admission lost cause: %v", err)
	}
}
