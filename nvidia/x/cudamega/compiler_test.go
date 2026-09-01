package cudamega

import (
	"testing"
)

func TestCompileLayerDependencies(t *testing.T) {
	arch := ModelArchitecture{
		Dim:        128,
		HiddenDim:  256,
		NumLayers:  1,
		NumHeads:   4,
		NumKVHeads: 2,
		HeadDim:    32,
		VocabSize:  512,
		TileSize:   32,
	}

	layer := LayerPointers{}
	scratch := ScratchPointers{}

	queue, nextBarrier := CompileLayer(arch, layer, scratch, 0, 0)
	if queue.Len() == 0 {
		t.Fatalf("expected non-empty instruction queue")
	}
	if nextBarrier <= 0 {
		t.Errorf("expected positive next barrier index, got %d", nextBarrier)
	}

	// Verify that the first instruction (RMSNorm) has no consume barrier dependency
	firstInst := queue.Instructions[0]
	if firstInst.Op != OpRMSNorm {
		t.Errorf("expected first op to be RMSNorm, got %v", firstInst.Op)
	}
	if firstInst.Barrier.ConsumeBarrier != -1 {
		t.Errorf("expected root op consume barrier to be -1, got %d", firstInst.Barrier.ConsumeBarrier)
	}
}

func TestCompileModelEndToEnd(t *testing.T) {
	arch := ModelArchitecture{
		Dim:        2048,
		HiddenDim:  8192,
		NumLayers:  16,
		NumHeads:   32,
		NumKVHeads: 8,
		HeadDim:    64,
		VocabSize:  128256,
		TileSize:   64,
	}

	layers := make([]LayerPointers, arch.NumLayers)
	scratch := ScratchPointers{}

	queue, err := CompileModel(arch, layers, scratch, 42)
	if err != nil {
		t.Fatalf("unexpected compile error: %v", err)
	}

	if queue.Len() == 0 {
		t.Fatalf("compiled queue is empty")
	}

	lastInst := queue.Instructions[queue.Len()-1]
	if lastInst.Op != OpHalt {
		t.Errorf("expected last op to be Halt, got %v", lastInst.Op)
	}

	if queue.TotalBarriers <= 0 {
		t.Errorf("expected positive total barriers, got %d", queue.TotalBarriers)
	}
}
