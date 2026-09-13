package cudapipeline

import (
	"fmt"
	"strings"
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

// MemorySpace identifies the physical or logical storage tier targeted by a pipeline buffer.
type MemorySpace string

const (
	MemorySpaceShared MemorySpace = "shared"
	MemorySpaceGlobal MemorySpace = "global"
	MemorySpaceTMEM   MemorySpace = "tmem"
)

// Buffer models a memory allocation (in shared memory or TMEM) for a single pipeline stage.
type Buffer struct {
	Space       MemorySpace // Storage tier (MemorySpaceShared or MemorySpaceTMEM; default: MemorySpaceShared)
	Offset      int         // Offset in bytes within the memory space
	SizeBytes   int         // Total allocated byte capacity
	Alignment   int         // Byte alignment of the buffer (must be at least 16)
	StrideBytes int         // Row stride in memory
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
	TargetArch            string      // Target GPU architecture (e.g. ArchSM90, ArchSM100, ArchSM121a)
	NumStages             int         // Number of pipeline stages (>= 2 for multi-stage buffering)
	Box                   TMABox      // TMA 2D box configuration for each stage
	MemorySpace           MemorySpace // Target memory space (MemorySpaceShared or MemorySpaceTMEM; default: MemorySpaceShared)
	SharedMemBaseOffset   int         // Base shared memory offset for stage buffers (when MemorySpace == MemorySpaceShared)
	TMEMBaseOffset        int         // Base TMEM byte offset for stage buffers (when MemorySpace == MemorySpaceTMEM)
	BarrierBaseOffset     int         // Base shared memory offset for mbarriers (always in shared memory)
	ArrivalTokensPerStage int         // Arriving threads/tokens per stage (default: 1)
	MemoryLatencyCycles   int         // Estimated round-trip memory latency in clock cycles (e.g. 600)
	ComputeIntervalCycles int         // Estimated compute duration per stage in clock cycles (e.g. 300)
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
// calculating memory layout and initializing stage barriers.
func NewPipeline(cfg PipelineConfig) (*Pipeline, error) {
	if cfg.TargetArch == "" {
		cfg.TargetArch = ArchSM121a
	}
	arch, err := ArchProfileFor(cfg.TargetArch)
	if err != nil {
		return nil, fmt.Errorf("new pipeline: %w", err)
	}

	if cfg.MemorySpace == "" {
		cfg.MemorySpace = MemorySpaceShared
	}
	if cfg.MemorySpace == MemorySpaceTMEM && !arch.SupportsTMEM {
		return nil, fmt.Errorf("new pipeline: architecture %s does not support Tensor Memory (TMEM)", arch.Name)
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
	stages := make([]Stage, cfg.NumStages)

	var curSharedBufOffset int
	if cfg.MemorySpace == MemorySpaceTMEM {
		curTMEMOffset := alignUp(cfg.TMEMBaseOffset, bufAlign)
		for i := 0; i < cfg.NumStages; i++ {
			stages[i].Index = i
			stages[i].Buffer = Buffer{
				Space:       MemorySpaceTMEM,
				Offset:      curTMEMOffset,
				SizeBytes:   stageBufSize,
				Alignment:   bufAlign,
				StrideBytes: cfg.Box.StrideBytes,
			}
			stages[i].TMABox = cfg.Box
			stages[i].ArrivalTokens = cfg.ArrivalTokensPerStage
			curTMEMOffset = alignUp(curTMEMOffset+stageBufSize, bufAlign)
		}
	} else {
		curSharedBufOffset = alignUp(cfg.SharedMemBaseOffset, bufAlign)
		for i := 0; i < cfg.NumStages; i++ {
			stages[i].Index = i
			stages[i].Buffer = Buffer{
				Space:       MemorySpaceShared,
				Offset:      curSharedBufOffset,
				SizeBytes:   stageBufSize,
				Alignment:   bufAlign,
				StrideBytes: cfg.Box.StrideBytes,
			}
			stages[i].TMABox = cfg.Box
			stages[i].ArrivalTokens = cfg.ArrivalTokensPerStage
			curSharedBufOffset = alignUp(curSharedBufOffset+stageBufSize, bufAlign)
		}
	}

	// Barrier layout: 8 bytes per mbarrier.b64, 8-byte aligned in shared memory.
	barrierBase := cfg.BarrierBaseOffset
	if barrierBase == 0 {
		if cfg.MemorySpace == MemorySpaceTMEM {
			barrierBase = cfg.SharedMemBaseOffset
		} else {
			barrierBase = curSharedBufOffset
		}
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
// by stage buffers (if residing in shared memory) and stage barriers in the pipeline.
func (p *Pipeline) TotalSharedBytes() int {
	if len(p.Stages) == 0 {
		return 0
	}
	maxOffset := 0
	for _, s := range p.Stages {
		if s.Buffer.Space == MemorySpaceShared || s.Buffer.Space == "" {
			bufEnd := s.Buffer.Offset + s.Buffer.SizeBytes
			if bufEnd > maxOffset {
				maxOffset = bufEnd
			}
		}
		barEnd := s.Barrier.Offset + 8
		if barEnd > maxOffset {
			maxOffset = barEnd
		}
	}
	return maxOffset
}

// TotalTMEMBytes returns the total TMEM span in bytes occupied by stage buffers
// in the pipeline, or 0 if no stage resides in TMEM.
func (p *Pipeline) TotalTMEMBytes() int {
	if len(p.Stages) == 0 {
		return 0
	}
	maxOffset := 0
	for _, s := range p.Stages {
		if s.Buffer.Space == MemorySpaceTMEM {
			bufEnd := s.Buffer.Offset + s.Buffer.SizeBytes
			if bufEnd > maxOffset {
				maxOffset = bufEnd
			}
		}
	}
	return maxOffset
}

// TotalTMEMColumns returns the number of 512-byte TMEM columns required
// to hold all TMEM stages in the pipeline, rounded up to the hardware allocation unit.
func (p *Pipeline) TotalTMEMColumns() int {
	bytes := p.TotalTMEMBytes()
	if bytes == 0 {
		return 0
	}
	bytesPerCol := p.Arch.TMEMBytesPerColumn
	if bytesPerCol <= 0 {
		bytesPerCol = BlackwellTMEMBytesPerCol
	}
	rawCols := (bytes + bytesPerCol - 1) / bytesPerCol
	allocUnit := p.Arch.TMEMAllocColUnit
	if allocUnit <= 0 {
		allocUnit = BlackwellTMEMAllocUnitCols
	}
	return alignUp(rawCols, allocUnit)
}

// Violation kinds identified during static pipeline analysis.
const (
	ErrUninitializedBarrier = "uninitialized barrier"
	ErrDuplicateInit        = "duplicate barrier initialization"
	ErrInvalidatedBarrier   = "barrier invalidated"
	ErrMissingArrive        = "missing barrier arrive"
	ErrMismatchedTxBytes    = "mismatched transaction byte count"
	ErrUnexpectedTx         = "unexpected transaction without expect_tx"
	ErrPhaseParity          = "phase parity mismatch"
	ErrInvalidParity        = "invalid parity value"
	ErrTMEMCapacityExceeded = "TMEM capacity exceeded"
	ErrInvalidTMEMColumns   = "invalid TMEM column count"
	ErrUnsupportedTMEM      = "unsupported TMEM operation on architecture"
)

// ValidationError records a static pipeline or mbarrier protocol violation.
type ValidationError struct {
	Line      int    // 1-based source line of the instruction
	Opcode    string // instruction opcode token (e.g. "mbarrier.arrive.expect_tx")
	Violation string // category of protocol violation
	Barrier   string // barrier operand or symbol, if applicable
	Phase     int    // pipeline or barrier phase index
	Expected  int64  // expected quantity (bytes, parity, arrival count), if applicable
	Actual    int64  // observed quantity, if applicable
	Message   string // human-readable explanation of the violation
}

func (e ValidationError) Error() string {
	var b strings.Builder
	if e.Line > 0 {
		fmt.Fprintf(&b, "line %d: ", e.Line)
	}
	if e.Opcode != "" {
		fmt.Fprintf(&b, "%s: ", e.Opcode)
	}
	if e.Barrier != "" {
		fmt.Fprintf(&b, "barrier %q: ", e.Barrier)
	}
	b.WriteString(e.Violation)
	if e.Message != "" {
		fmt.Fprintf(&b, ": %s", e.Message)
	}
	return b.String()
}

// EventKind identifies an mbarrier lifecycle event during static analysis.
type EventKind string

const (
	EventInit     EventKind = "init"
	EventInval    EventKind = "inval"
	EventExpectTx EventKind = "expect_tx"
	EventArrive   EventKind = "arrive"
	EventCopy     EventKind = "copy"
	EventWait     EventKind = "wait"
)

// BarrierEvent records an operation applied to an mbarrier during static analysis.
type BarrierEvent struct {
	Line   int       // instruction line number
	Opcode string    // instruction opcode
	Kind   EventKind // event kind
	Phase  int       // barrier phase at time of event
	Bytes  int64     // bytes expected or completed, if applicable
	Count  int       // thread arrival count, if applicable
	Parity int       // parity value, if applicable
}

// Option configures a BarrierTracker.
type Option func(*BarrierTracker)

// WithTensorSize associates a tensor-map parameter or register name with a known transfer byte size.
func WithTensorSize(tensorMap string, bytes int64) Option {
	return func(t *BarrierTracker) {
		t.SetTensorSize(tensorMap, bytes)
	}
}

// WithBarrierArrivalCount sets the expected arrival count for a barrier if not literal in PTX.
func WithBarrierArrivalCount(barrier string, count int) Option {
	return func(t *BarrierTracker) {
		t.barrierArrivals[barrier] = count
	}
}

// WithTargetArch sets the target architecture for static PTX validation.
func WithTargetArch(arch string) Option {
	return func(t *BarrierTracker) {
		t.targetArch = arch
	}
}
