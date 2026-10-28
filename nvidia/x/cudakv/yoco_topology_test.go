package cudakv_test

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudakv"
)

func TestSharedLayerTopologyAndYoCoBoundary(t *testing.T) {
	// 60-layer stack where layers 0..29 are producers and layers 30..59 share producer 29.
	specs := make([]cudakv.SharedLayerSpec, 60)
	for i := 0; i < 60; i++ {
		if i < 30 {
			specs[i] = cudakv.SharedLayerSpec{
				LayerID:       i,
				SourceLayerID: i,
				Role:          cudakv.LayerKVProducer,
				IsSWA:         true,
				WindowSize:    128,
			}
		} else {
			specs[i] = cudakv.SharedLayerSpec{
				LayerID:       i,
				SourceLayerID: 29,
				Role:          cudakv.LayerKVSharedConsumer,
				IsSWA:         true,
				WindowSize:    128,
			}
		}
	}

	topo, err := cudakv.NewSharedLayerTopology(specs)
	if err != nil {
		t.Fatalf("NewSharedLayerTopology() unexpected error: %v", err)
	}
	if topo.NumLayers() != 60 {
		t.Fatalf("NumLayers() = %d, want 60", topo.NumLayers())
	}
	producers := topo.ProducerLayers()
	if len(producers) != 30 {
		t.Fatalf("len(ProducerLayers()) = %d, want 30", len(producers))
	}
	for i := 0; i < 30; i++ {
		if producers[i] != i {
			t.Fatalf("ProducerLayers()[%d] = %d, want %d", i, producers[i], i)
		}
		slot, err := topo.PhysicalSlotForLayer(i)
		if err != nil {
			t.Fatalf("PhysicalSlotForLayer(%d) unexpected error: %v", i, err)
		}
		if slot != i {
			t.Fatalf("PhysicalSlotForLayer(%d) = %d, want %d", i, slot, i)
		}
	}
	for i := 30; i < 60; i++ {
		slot, err := topo.PhysicalSlotForLayer(i)
		if err != nil {
			t.Fatalf("PhysicalSlotForLayer(%d) unexpected error: %v", i, err)
		}
		if slot != 29 {
			t.Fatalf("PhysicalSlotForLayer(%d) = %d, want 29", i, slot)
		}
	}
	if got := topo.MemorySavingsRatio(); math.Abs(got-0.5) > 1e-9 {
		t.Fatalf("MemorySavingsRatio() = %v, want 0.5", got)
	}

	// Negative topology controls.
	if _, err := cudakv.NewSharedLayerTopology(nil); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("empty specs error = %v, want ErrUnsupported", err)
	}
	badForwardRef := []cudakv.SharedLayerSpec{
		{LayerID: 0, SourceLayerID: 1, Role: cudakv.LayerKVSharedConsumer},
		{LayerID: 1, SourceLayerID: 1, Role: cudakv.LayerKVProducer},
	}
	if _, err := cudakv.NewSharedLayerTopology(badForwardRef); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("forward consumer reference error = %v, want ErrUnsupported", err)
	}
	if _, err := topo.PhysicalSlotForLayer(60); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("out-of-range PhysicalSlotForLayer error = %v, want ErrUnsupported", err)
	}

	// BuildYoCoBoundary test: 2 sequences, seq0 has seqLenQ=4, seqLenK=10 (prefix=6); seq1 has seqLenQ=2, seqLenK=5 (prefix=3).
	// With recentCap=3:
	// seq0: winLen=min(4,3)=3, winStartInSeqQ = 4-3 = 1 (KV positions 7, 8, 9).
	//   qIdx=0 (KV 6): before winStartInSeqQ -> [-1, -1, -1]
	//   qIdx=1 (KV 7): [7, -1, -1]
	//   qIdx=2 (KV 8): [7, 8, -1]
	//   qIdx=3 (KV 9): [7, 8, 9]
	// seq1 (kvBase=10, prefix=3): winLen=min(2,3)=2, winStartInSeqQ = 0 (KV positions 13, 14).
	//   qIdx=0 (KV 13): [13, -1, -1]
	//   qIdx=1 (KV 14): [13, 14, -1]
	cuSeqlensQ := []int32{0, 4, 6}
	seqlensK := []int32{10, 5}
	yoco, err := cudakv.BuildYoCoBoundary(cuSeqlensQ, seqlensK, 3)
	if err != nil {
		t.Fatalf("BuildYoCoBoundary() unexpected error: %v", err)
	}
	wantFull := []int32{6, 7, 8, 9, 13, 14}
	if !reflect.DeepEqual(yoco.FullIndices, wantFull) {
		t.Fatalf("FullIndices = %v, want %v", yoco.FullIndices, wantFull)
	}
	wantBoundary := []int32{
		-1, -1, -1,
		7, -1, -1,
		7, 8, -1,
		7, 8, 9,
		13, -1, -1,
		13, 14, -1,
	}
	if !reflect.DeepEqual(yoco.BoundaryIndices, wantBoundary) {
		t.Fatalf("BoundaryIndices = %v, want %v", yoco.BoundaryIndices, wantBoundary)
	}

	// Default recentCap = 6 when 0 is passed.
	yocoDefault, err := cudakv.BuildYoCoBoundary([]int32{0, 2}, []int32{2}, 0)
	if err != nil {
		t.Fatalf("BuildYoCoBoundary(recentCap=0) error: %v", err)
	}
	if yocoDefault.RecentCap != 6 || len(yocoDefault.BoundaryIndices) != 12 {
		t.Fatalf("default RecentCap = %d (len=%d), want 6 (len=12)", yocoDefault.RecentCap, len(yocoDefault.BoundaryIndices))
	}

	// Negative boundary controls.
	if _, err := cudakv.BuildYoCoBoundary([]int32{0}, nil, 6); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("short cuSeqlensQ error = %v, want ErrUnsupported", err)
	}
	if _, err := cudakv.BuildYoCoBoundary([]int32{0, 5}, []int32{3}, 6); !errors.Is(err, cudaerrors.ErrUnsupported) {
		t.Fatalf("seqlensK < seqLenQ error = %v, want ErrUnsupported", err)
	}
}
