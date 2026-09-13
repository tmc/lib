package cudapipeline

import (
	"fmt"
)

// TMABox specifies the tile geometry and element layout for a 2D Tensor Memory
// Accelerator (TMA) bulk transfer (cp.async.bulk.tensor.2d).
type TMABox struct {
	DimX        int // Horizontal box dimension in elements
	DimY        int // Vertical box dimension in elements
	ElementSize int // Size of each element in bytes (e.g. 2 for fp16/bf16, 1 for fp8, 4 for tf32/fp32)
	StrideBytes int // Stride between rows in shared memory (must be >= DimX * ElementSize and multiple of 16)
}

// PayloadBytes returns the total payload bytes transferred by this TMA box.
func (b TMABox) PayloadBytes() int {
	return b.DimX * b.DimY * b.ElementSize
}

// RowBytes returns the byte width of a single row in the box.
func (b TMABox) RowBytes() int {
	return b.DimX * b.ElementSize
}

// Buffer models a shared memory allocation for a single pipeline stage.
type Buffer struct {
	Offset      int // Offset in shared memory in bytes
	SizeBytes   int // Total allocated byte capacity
	Alignment   int // Byte alignment of the buffer (must be at least 16)
	StrideBytes int // Row stride in shared memory
}

// Barrier models the PTX mbarrier state used to synchronize an async pipeline stage.
type Barrier struct {
	Offset           int // Shared memory offset in bytes (must be 8-byte aligned)
	ArrivalCount     int // Expected arriving threads/tokens per phase
	TransactionBytes int // Total transaction bytes expected via mbarrier.arrive.expect_tx
	Phase            int // Current phase counter (0, 1, 2, ...)
}

// Parity returns the current phase parity (0 or 1).
func (b Barrier) Parity() int {
	return b.Phase & 1
}

// AdvancePhase increments the barrier phase counter, toggling parity.
func (b *Barrier) AdvancePhase() {
	b.Phase++
}

// Stage models a single stage of an asynchronous multi-stage pipeline,
// binding a buffer allocation, a TMA box transfer, and an mbarrier.
type Stage struct {
	Index         int
	Buffer        Buffer
	TMABox        TMABox
	Barrier       Barrier
	ArrivalTokens int
}

// PipelineConfig specifies parameters for constructing a multi-stage pipeline.
type PipelineConfig struct {
	TargetArch            string // Target GPU architecture (e.g. ArchSM90, ArchSM121a)
	NumStages             int    // Number of pipeline stages (>= 2 for multi-stage buffering)
	Box                   TMABox // TMA 2D box configuration for each stage
	SharedMemBaseOffset   int    // Base shared memory offset for stage buffers
	BarrierBaseOffset     int    // Base shared memory offset for mbarriers
	ArrivalTokensPerStage int    // Arriving threads/tokens per stage (default: 1)
	MemoryLatencyCycles   int    // Estimated round-trip memory latency in clock cycles (e.g. 600)
	ComputeIntervalCycles int    // Estimated compute duration per stage in clock cycles (e.g. 300)
}

// Pipeline models a multi-stage producer-consumer asynchronous memory pipeline
// synchronized with PTX transaction barriers.
type Pipeline struct {
	Config PipelineConfig
	Stages []Stage
	Arch   ArchProfile
}

// alignUp aligns val to the next multiple of align (which must be a power of 2 or positive integer).
func alignUp(val, align int) int {
	if align <= 1 {
		return val
	}
	rem := val % align
	if rem == 0 {
		return val
	}
	return val + (align - rem)
}

// NewPipeline constructs a multi-stage pipeline from the given configuration,
// calculating shared memory layout and initializing stage barriers.
func NewPipeline(cfg PipelineConfig) (*Pipeline, error) {
	if cfg.TargetArch == "" {
		cfg.TargetArch = ArchSM121a
	}
	arch, err := ArchProfileFor(cfg.TargetArch)
	if err != nil {
		return nil, fmt.Errorf("new pipeline: %w", err)
	}

	if cfg.NumStages < 2 {
		return nil, fmt.Errorf("new pipeline: num stages must be at least 2 for multi-stage buffering, got %d", cfg.NumStages)
	}
	if cfg.Box.DimX <= 0 || cfg.Box.DimY <= 0 || cfg.Box.ElementSize <= 0 {
		return nil, fmt.Errorf("new pipeline: invalid TMA box dimensions (DimX=%d, DimY=%d, ElementSize=%d)",
			cfg.Box.DimX, cfg.Box.DimY, cfg.Box.ElementSize)
	}

	rowBytes := cfg.Box.RowBytes()
	if cfg.Box.StrideBytes == 0 {
		cfg.Box.StrideBytes = alignUp(rowBytes, 16)
	}

	if cfg.ArrivalTokensPerStage <= 0 {
		cfg.ArrivalTokensPerStage = 1
	}

	// Buffer sizing and layout.
	stageBufSize := cfg.Box.StrideBytes * cfg.Box.DimY
	bufAlign := 128 // 128-byte alignment for optimal TMA throughput
	curBufOffset := alignUp(cfg.SharedMemBaseOffset, bufAlign)

	stages := make([]Stage, cfg.NumStages)
	for i := 0; i < cfg.NumStages; i++ {
		stages[i].Index = i
		stages[i].Buffer = Buffer{
			Offset:      curBufOffset,
			SizeBytes:   stageBufSize,
			Alignment:   bufAlign,
			StrideBytes: cfg.Box.StrideBytes,
		}
		stages[i].TMABox = cfg.Box
		stages[i].ArrivalTokens = cfg.ArrivalTokensPerStage
		curBufOffset = alignUp(curBufOffset+stageBufSize, bufAlign)
	}

	// Barrier layout: 8 bytes per mbarrier.b64, 8-byte aligned.
	barrierBase := cfg.BarrierBaseOffset
	if barrierBase == 0 {
		barrierBase = curBufOffset
	}
	curBarrierOffset := alignUp(barrierBase, 8)

	for i := 0; i < cfg.NumStages; i++ {
		stages[i].Barrier = Barrier{
			Offset:           curBarrierOffset,
			ArrivalCount:     cfg.ArrivalTokensPerStage,
			TransactionBytes: cfg.Box.PayloadBytes(),
			Phase:            0,
		}
		curBarrierOffset += 8
	}

	return &Pipeline{
		Config: cfg,
		Stages: stages,
		Arch:   arch,
	}, nil
}

// NumStages returns the number of pipeline stages.
func (p *Pipeline) NumStages() int {
	return len(p.Stages)
}

// Stage returns a pointer to the stage at index i, or nil if out of bounds.
func (p *Pipeline) Stage(i int) *Stage {
	if i < 0 || i >= len(p.Stages) {
		return nil
	}
	return &p.Stages[i]
}

// TotalSharedBytes returns the total shared memory span in bytes occupied
// by all stage buffers and barriers in the pipeline.
func (p *Pipeline) TotalSharedBytes() int {
	if len(p.Stages) == 0 {
		return 0
	}
	maxOffset := 0
	for _, s := range p.Stages {
		bufEnd := s.Buffer.Offset + s.Buffer.SizeBytes
		if bufEnd > maxOffset {
			maxOffset = bufEnd
		}
		barEnd := s.Barrier.Offset + 8
		if barEnd > maxOffset {
			maxOffset = barEnd
		}
	}
	return maxOffset
}
