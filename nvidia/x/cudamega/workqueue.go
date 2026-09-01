package cudamega

import (
	"fmt"
)

// OpCode identifies the operation to be executed by a threadblock in the megakernel.
type OpCode uint32

const (
	// OpHalt terminates the persistent megakernel execution on all SMs.
	OpHalt OpCode = iota
	// OpRMSNorm performs RMS normalization on input activation vectors.
	OpRMSNorm
	// OpTileGEMV performs 2D/1D tiled matrix-vector multiplication.
	OpTileGEMV
	// OpRoPE applies rotary position embedding to Q and K activations.
	OpRoPE
	// OpKVStore writes computed K and V projections into the KV cache.
	OpKVStore
	// OpFlashAttn performs online softmax causal self-attention decode.
	OpFlashAttn
	// OpSwiGLU performs elementwise SiLU(gate) * up projection.
	OpSwiGLU
	// OpResidual performs elementwise vector addition (x = x + residual).
	OpResidual
	// OpArgmax computes argmax across vocab logits to sample the next token.
	OpArgmax
)

func (op OpCode) String() string {
	switch op {
	case OpHalt:
		return "Halt"
	case OpRMSNorm:
		return "RMSNorm"
	case OpTileGEMV:
		return "TileGEMV"
	case OpRoPE:
		return "RoPE"
	case OpKVStore:
		return "KVStore"
	case OpFlashAttn:
		return "FlashAttn"
	case OpSwiGLU:
		return "SwiGLU"
	case OpResidual:
		return "Residual"
	case OpArgmax:
		return "Argmax"
	default:
		return fmt.Sprintf("OpCode(%d)", uint32(op))
	}
}

// BarrierSpec defines the fine-grained barrier synchronization for an instruction tile.
type BarrierSpec struct {
	// ConsumeBarrier is the index of the barrier that must reach 0 before execution begins.
	// A value of -1 indicates no barrier dependency.
	ConsumeBarrier int32
	// ProduceBarrier is the index of the barrier to increment at launch and decrement on completion.
	// A value of -1 indicates no barrier notification.
	ProduceBarrier int32
	// BarrierStride specifies the stride multiplier for mapping 2D launch index to barrier index.
	BarrierStride int32
}

// Instruction represents a symbolic operation tile in the megakernel work queue.
type Instruction struct {
	Op         OpCode
	TotalTiles uint32
	TileDimM   uint32
	TileDimK   uint32
	TileDimN   uint32
	SrcA       uint64 // GPU virtual address of source A
	SrcB       uint64 // GPU virtual address of source B
	SrcC       uint64 // GPU virtual address of source C / auxiliary
	Dst        uint64 // GPU virtual address of destination
	Param0     uint32 // Context-dependent parameter (e.g. sequence pos, epsilon bits)
	Param1     uint32 // Context-dependent parameter (e.g. head dim, num heads)
	Barrier    BarrierSpec
}

// SymbolicWorkQueue manages a sequence of instructions scheduled for persistent GPU dispatch.
type SymbolicWorkQueue struct {
	Instructions  []Instruction
	TotalBarriers int
}

// NewWorkQueue allocates an empty SymbolicWorkQueue.
func NewWorkQueue() *SymbolicWorkQueue {
	return &SymbolicWorkQueue{
		Instructions: make([]Instruction, 0, 64),
	}
}

// Add appends an instruction to the work queue.
func (q *SymbolicWorkQueue) Add(inst Instruction) {
	if int(inst.Barrier.ConsumeBarrier)+1 > q.TotalBarriers {
		q.TotalBarriers = int(inst.Barrier.ConsumeBarrier) + 1
	}
	if int(inst.Barrier.ProduceBarrier)+1 > q.TotalBarriers {
		q.TotalBarriers = int(inst.Barrier.ProduceBarrier) + 1
	}
	q.Instructions = append(q.Instructions, inst)
}

// Len returns the number of instructions in the queue.
func (q *SymbolicWorkQueue) Len() int {
	return len(q.Instructions)
}

// TotalTileCount computes the total number of individual work tiles across all instructions.
func (q *SymbolicWorkQueue) TotalTileCount() int {
	total := 0
	for _, inst := range q.Instructions {
		total += int(inst.TotalTiles)
	}
	return total
}
