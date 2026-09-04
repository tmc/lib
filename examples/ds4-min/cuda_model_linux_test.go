//go:build linux

package main

import (
	"os"
	"testing"
)

func TestCUDABindModel(t *testing.T) {
	path := os.Getenv("DS4_MODEL")
	if path == "" {
		t.Skip("DS4_MODEL is not set")
	}
	r, err := newCUDAFoundation(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})
	if !r.hmm {
		if _, err := r.bindModel(); err == nil {
			t.Fatal("bindModel succeeded without host page tables")
		}
		t.Skip("binding the whole model requires HMM tensor views")
	}
	m, err := r.bindModel()
	if err != nil {
		t.Fatal(err)
	}

	for _, tt := range []struct {
		name       string
		tensor     cudaModelTensor
		typeID     uint32
		dimensions []uint64
	}{
		{"output norm", m.output.norm, 0, []uint64{cudaWidth}},
		{"output HC scale", m.output.hcScale, 0, []uint64{3}},
		{"output HC base", m.output.hcBase, 0, []uint64{24}},
		{"output HC projection", m.output.hcFn, 1, []uint64{cudaHCWidth, 24}},
	} {
		if tt.tensor.typeID != tt.typeID {
			t.Errorf("%s type = %d, want %d", tt.name, tt.tensor.typeID, tt.typeID)
		}
		if len(tt.tensor.dimensions) != len(tt.dimensions) {
			t.Errorf("%s dimensions = %v, want %v", tt.name, tt.tensor.dimensions, tt.dimensions)
			continue
		}
		for i, want := range tt.dimensions {
			if tt.tensor.dimensions[i] != want {
				t.Errorf("%s dimensions = %v, want %v", tt.name, tt.tensor.dimensions, tt.dimensions)
				break
			}
		}
	}

	for i := range m.layers {
		l := &m.layers[i]
		if l.attnSinks.device == 0 || l.attnSinks.typeID != 0 || len(l.attnSinks.dimensions) != 1 || l.attnSinks.dimensions[0] != 64 {
			t.Errorf("layer %d attention sinks = %+v, want an F32 vector of 64", i, l.attnSinks)
		}
		if l.attnKV.device == 0 || len(l.attnKV.dimensions) != 2 || l.attnKV.dimensions[1] != cudaKVWidth {
			t.Errorf("layer %d key-value projection = %+v, want %d output rows", i, l.attnKV, cudaKVWidth)
		}
		if l.hcAttnFn.typeID != 1 || len(l.hcAttnFn.dimensions) != 2 ||
			l.hcAttnFn.dimensions[0] != cudaHCWidth || l.hcAttnFn.dimensions[1] != 24 {
			t.Errorf("layer %d attention HC projection = %+v, want an F16 %dx24", i, l.hcAttnFn, cudaHCWidth)
		}
		if got, want := l.expertMap.device != 0, i < 3; got != want {
			t.Errorf("layer %d binds the identity map = %t, want %t", i, got, want)
		}
		if got, want := l.expertBias.device != 0, i >= 3; got != want {
			t.Errorf("layer %d binds the router bias = %t, want %t", i, got, want)
		}
		if got, want := l.attnCompressorKV.device != 0, i >= 2; got != want {
			t.Errorf("layer %d binds the attention compressor = %t, want %t", i, got, want)
		}
		if got, want := l.indexerProj.device != 0, i >= 2 && i%2 == 0; got != want {
			t.Errorf("layer %d binds the indexer = %t, want %t", i, got, want)
		}
	}
	t.Logf("bound %d layers and the output head through host page tables", len(m.layers))
}
