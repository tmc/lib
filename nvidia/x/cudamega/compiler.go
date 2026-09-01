package cudamega

import (
	"fmt"
)

// ModelArchitecture describes the layer dimensions and hyperparameters for megakernel compilation.
type ModelArchitecture struct {
	Dim              int
	HiddenDim        int
	NumLayers        int
	NumHeads         int
	NumKVHeads       int
	HeadDim          int
	VocabSize        int
	TileSize         int // e.g. 32 or 64 elements per block tile
}

// LayerPointers defines the GPU memory buffer addresses for a single transformer layer.
type LayerPointers struct {
	InputNormWeight uint64
	PostNormWeight  uint64
	WQ              uint64
	WK              uint64
	WV              uint64
	WO              uint64
	WGate           uint64
	WUp             uint64
	WDown           uint64
	KVCacheK        uint64
	KVCacheV        uint64
}

// ScratchPointers defines the intermediate activation buffer addresses on the GPU.
type ScratchPointers struct {
	XIn      uint64 // Input residual stream [Dim]
	XNorm    uint64 // Normed activation [Dim]
	Q        uint64 // Query activation [NumHeads * HeadDim]
	K        uint64 // Key activation [NumKVHeads * HeadDim]
	V        uint64 // Value activation [NumKVHeads * HeadDim]
	AttnOut  uint64 // Attention projection output [Dim]
	FFNNorm  uint64 // Post-attention normed activation [Dim]
	FFNGate  uint64 // FFN Gate activation [HiddenDim]
	FFNUp    uint64 // FFN Up activation [HiddenDim]
	FFNOut   uint64 // FFN Down projection output [Dim]
	LMHead   uint64 // LM Head projection weights [VocabSize * Dim]
	Logits   uint64 // Final vocab logits [VocabSize]
	TokenOut uint64 // Next sampled token ID [1]
}

// CompileLayer compiles a single transformer layer's operations into a SymbolicWorkQueue with barrier dependencies.
func CompileLayer(arch ModelArchitecture, layer LayerPointers, scratch ScratchPointers, pos int, startBarrier int32) (*SymbolicWorkQueue, int32) {
	tileSize := arch.TileSize
	if tileSize <= 0 {
		tileSize = 32
	}

	wq := NewWorkQueue()
	currentBarrier := startBarrier

	// 1. RMSNorm (InputNorm)
	bNorm := currentBarrier
	currentBarrier++
	dimTiles := uint32((arch.Dim + tileSize - 1) / tileSize)
	wq.Add(Instruction{
		Op:         OpRMSNorm,
		TotalTiles: dimTiles,
		TileDimM:   uint32(tileSize),
		TileDimK:   uint32(arch.Dim),
		SrcA:       scratch.XIn,
		SrcB:       layer.InputNormWeight,
		Dst:        scratch.XNorm,
		Barrier: BarrierSpec{
			ConsumeBarrier: -1,
			ProduceBarrier: bNorm,
		},
	})

	// 2. Q, K, V Projections (Tiled GEMV)
	bQKV := currentBarrier
	currentBarrier++
	qDim := arch.NumHeads * arch.HeadDim
	qTiles := uint32((qDim + tileSize - 1) / tileSize)
	wq.Add(Instruction{
		Op:         OpTileGEMV,
		TotalTiles: qTiles,
		TileDimM:   uint32(tileSize),
		TileDimK:   uint32(arch.Dim),
		SrcA:       scratch.XNorm,
		SrcB:       layer.WQ,
		Dst:        scratch.Q,
		Barrier: BarrierSpec{
			ConsumeBarrier: bNorm,
			ProduceBarrier: bQKV,
		},
	})

	kvDim := arch.NumKVHeads * arch.HeadDim
	kvTiles := uint32((kvDim + tileSize - 1) / tileSize)
	wq.Add(Instruction{
		Op:         OpTileGEMV,
		TotalTiles: kvTiles,
		TileDimM:   uint32(tileSize),
		TileDimK:   uint32(arch.Dim),
		SrcA:       scratch.XNorm,
		SrcB:       layer.WK,
		Dst:        scratch.K,
		Barrier: BarrierSpec{
			ConsumeBarrier: bNorm,
			ProduceBarrier: bQKV,
		},
	})
	wq.Add(Instruction{
		Op:         OpTileGEMV,
		TotalTiles: kvTiles,
		TileDimM:   uint32(tileSize),
		TileDimK:   uint32(arch.Dim),
		SrcA:       scratch.XNorm,
		SrcB:       layer.WV,
		Dst:        scratch.V,
		Barrier: BarrierSpec{
			ConsumeBarrier: bNorm,
			ProduceBarrier: bQKV,
		},
	})

	// 3. RoPE on Q and K
	bRoPE := currentBarrier
	currentBarrier++
	wq.Add(Instruction{
		Op:         OpRoPE,
		TotalTiles: uint32(arch.NumHeads),
		TileDimM:   uint32(arch.HeadDim),
		Param0:     uint32(pos),
		Param1:     uint32(arch.HeadDim),
		SrcA:       scratch.Q,
		SrcB:       scratch.K,
		Dst:        scratch.Q,
		Barrier: BarrierSpec{
			ConsumeBarrier: bQKV,
			ProduceBarrier: bRoPE,
		},
	})

	// 4. KV Store to KV Cache
	bKVStore := currentBarrier
	currentBarrier++
	wq.Add(Instruction{
		Op:         OpKVStore,
		TotalTiles: uint32(arch.NumKVHeads),
		TileDimM:   uint32(arch.HeadDim),
		Param0:     uint32(pos),
		SrcA:       scratch.K,
		SrcB:       scratch.V,
		Dst:        layer.KVCacheK,
		SrcC:       layer.KVCacheV,
		Barrier: BarrierSpec{
			ConsumeBarrier: bRoPE,
			ProduceBarrier: bKVStore,
		},
	})

	// 5. Flash Attention Decode
	bAttn := currentBarrier
	currentBarrier++
	wq.Add(Instruction{
		Op:         OpFlashAttn,
		TotalTiles: uint32(arch.NumHeads),
		TileDimM:   uint32(arch.HeadDim),
		Param0:     uint32(pos),
		SrcA:       scratch.Q,
		SrcB:       layer.KVCacheK,
		SrcC:       layer.KVCacheV,
		Dst:        scratch.AttnOut,
		Barrier: BarrierSpec{
			ConsumeBarrier: bKVStore,
			ProduceBarrier: bAttn,
		},
	})

	// 6. Attention Out Projection (WO) + Residual Add
	bWO := currentBarrier
	currentBarrier++
	wq.Add(Instruction{
		Op:         OpTileGEMV,
		TotalTiles: dimTiles,
		TileDimM:   uint32(tileSize),
		TileDimK:   uint32(qDim),
		SrcA:       scratch.AttnOut,
		SrcB:       layer.WO,
		Dst:        scratch.FFNNorm,
		Barrier: BarrierSpec{
			ConsumeBarrier: bAttn,
			ProduceBarrier: bWO,
		},
	})
	wq.Add(Instruction{
		Op:         OpResidual,
		TotalTiles: dimTiles,
		TileDimM:   uint32(tileSize),
		SrcA:       scratch.XIn,
		SrcB:       scratch.FFNNorm,
		Dst:        scratch.XIn,
		Barrier: BarrierSpec{
			ConsumeBarrier: bWO,
			ProduceBarrier: bWO,
		},
	})

	// 7. Post-Attention RMSNorm
	bPostNorm := currentBarrier
	currentBarrier++
	wq.Add(Instruction{
		Op:         OpRMSNorm,
		TotalTiles: dimTiles,
		TileDimM:   uint32(tileSize),
		TileDimK:   uint32(arch.Dim),
		SrcA:       scratch.XIn,
		SrcB:       layer.PostNormWeight,
		Dst:        scratch.FFNNorm,
		Barrier: BarrierSpec{
			ConsumeBarrier: bWO,
			ProduceBarrier: bPostNorm,
		},
	})

	// 8. FFN Gate & Up Projections (SwiGLU)
	bFFN := currentBarrier
	currentBarrier++
	hiddenTiles := uint32((arch.HiddenDim + tileSize - 1) / tileSize)
	wq.Add(Instruction{
		Op:         OpTileGEMV,
		TotalTiles: hiddenTiles,
		TileDimM:   uint32(tileSize),
		TileDimK:   uint32(arch.Dim),
		SrcA:       scratch.FFNNorm,
		SrcB:       layer.WGate,
		Dst:        scratch.FFNGate,
		Barrier: BarrierSpec{
			ConsumeBarrier: bPostNorm,
			ProduceBarrier: bFFN,
		},
	})
	wq.Add(Instruction{
		Op:         OpTileGEMV,
		TotalTiles: hiddenTiles,
		TileDimM:   uint32(tileSize),
		TileDimK:   uint32(arch.Dim),
		SrcA:       scratch.FFNNorm,
		SrcB:       layer.WUp,
		Dst:        scratch.FFNUp,
		Barrier: BarrierSpec{
			ConsumeBarrier: bPostNorm,
			ProduceBarrier: bFFN,
		},
	})
	wq.Add(Instruction{
		Op:         OpSwiGLU,
		TotalTiles: hiddenTiles,
		TileDimM:   uint32(tileSize),
		SrcA:       scratch.FFNGate,
		SrcB:       scratch.FFNUp,
		Dst:        scratch.FFNGate,
		Barrier: BarrierSpec{
			ConsumeBarrier: bFFN,
			ProduceBarrier: bFFN,
		},
	})

	// 9. FFN Down Projection + Residual Add
	bDown := currentBarrier
	currentBarrier++
	wq.Add(Instruction{
		Op:         OpTileGEMV,
		TotalTiles: dimTiles,
		TileDimM:   uint32(tileSize),
		TileDimK:   uint32(arch.HiddenDim),
		SrcA:       scratch.FFNGate,
		SrcB:       layer.WDown,
		Dst:        scratch.FFNOut,
		Barrier: BarrierSpec{
			ConsumeBarrier: bFFN,
			ProduceBarrier: bDown,
		},
	})
	wq.Add(Instruction{
		Op:         OpResidual,
		TotalTiles: dimTiles,
		TileDimM:   uint32(tileSize),
		SrcA:       scratch.XIn,
		SrcB:       scratch.FFNOut,
		Dst:        scratch.XIn,
		Barrier: BarrierSpec{
			ConsumeBarrier: bDown,
			ProduceBarrier: bDown,
		},
	})

	return wq, currentBarrier
}

// CompileModel compiles an entire multi-layer transformer forward decode pass into a single SymbolicWorkQueue.
func CompileModel(arch ModelArchitecture, layers []LayerPointers, scratch ScratchPointers, pos int) (*SymbolicWorkQueue, error) {
	if len(layers) != arch.NumLayers {
		return nil, fmt.Errorf("cudamega: layer count mismatch (expected %d, got %d)", arch.NumLayers, len(layers))
	}

	fullQueue := NewWorkQueue()
	var currentBarrier int32 = 0

	for l := 0; l < arch.NumLayers; l++ {
		layerQueue, nextBarrier := CompileLayer(arch, layers[l], scratch, pos, currentBarrier)
		for _, inst := range layerQueue.Instructions {
			fullQueue.Add(inst)
		}
		currentBarrier = nextBarrier
	}

	// Final LM Head and Argmax Sampling
	tileSize := arch.TileSize
	if tileSize <= 0 {
		tileSize = 32
	}
	vocabTiles := uint32((arch.VocabSize + tileSize - 1) / tileSize)
	bLMHead := currentBarrier
	currentBarrier++

	fullQueue.Add(Instruction{
		Op:         OpTileGEMV,
		TotalTiles: vocabTiles,
		TileDimM:   uint32(tileSize),
		TileDimK:   uint32(arch.Dim),
		SrcA:       scratch.XIn,
		SrcB:       scratch.LMHead,
		Dst:        scratch.Logits,
		Barrier: BarrierSpec{
			ConsumeBarrier: currentBarrier - 2,
			ProduceBarrier: bLMHead,
		},
	})

	fullQueue.Add(Instruction{
		Op:         OpArgmax,
		TotalTiles: 1,
		TileDimK:   uint32(arch.VocabSize),
		SrcA:       scratch.Logits,
		Dst:        scratch.TokenOut,
		Barrier: BarrierSpec{
			ConsumeBarrier: bLMHead,
			ProduceBarrier: -1,
		},
	})

	fullQueue.Add(Instruction{
		Op:         OpHalt,
		TotalTiles: 1,
		Barrier: BarrierSpec{
			ConsumeBarrier: -1,
			ProduceBarrier: -1,
		},
	})

	return fullQueue, nil
}
