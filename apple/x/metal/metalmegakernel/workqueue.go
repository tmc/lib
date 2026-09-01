package metalmegakernel

import (
	"fmt"
)

// OpCode identifies the operation tile in the Metal megakernel work queue.
type OpCode uint32

const (
	OpHalt OpCode = iota
	OpRMSNorm
	OpTileGEMV
	OpRoPE
	OpKVStore
	OpFlashAttn
	OpSwiGLU
	OpResidual
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

// BarrierSpec defines inter-threadgroup barrier synchronization for a Metal work tile.
type BarrierSpec struct {
	ConsumeBarrier int32
	ProduceBarrier int32
	BarrierStride  int32
}

// Instruction represents a scheduled operation tile in the Metal work queue.
type Instruction struct {
	Op         OpCode
	TotalTiles uint32
	TileDimM   uint32
	TileDimK   uint32
	TileDimN   uint32
	SrcAOffset uint64 // Shared memory / buffer offset
	SrcBOffset uint64
	SrcCOffset uint64
	DstOffset  uint64
	Param0     uint32
	Param1     uint32
	Barrier    BarrierSpec
}

// MetalWorkQueue manages a sequence of instructions scheduled for persistent Metal dispatch.
type MetalWorkQueue struct {
	Instructions  []Instruction
	TotalBarriers int
}

// NewWorkQueue creates an empty MetalWorkQueue.
func NewWorkQueue() *MetalWorkQueue {
	return &MetalWorkQueue{
		Instructions: make([]Instruction, 0, 64),
	}
}

// Add appends an instruction to the work queue.
func (q *MetalWorkQueue) Add(inst Instruction) {
	if int(inst.Barrier.ConsumeBarrier)+1 > q.TotalBarriers {
		q.TotalBarriers = int(inst.Barrier.ConsumeBarrier) + 1
	}
	if int(inst.Barrier.ProduceBarrier)+1 > q.TotalBarriers {
		q.TotalBarriers = int(inst.Barrier.ProduceBarrier) + 1
	}
	q.Instructions = append(q.Instructions, inst)
}

// Len returns the count of instructions in the queue.
func (q *MetalWorkQueue) Len() int {
	return len(q.Instructions)
}

// TotalTileCount computes the total number of work tiles in the queue.
func (q *MetalWorkQueue) TotalTileCount() int {
	total := 0
	for _, inst := range q.Instructions {
		total += int(inst.TotalTiles)
	}
	return total
}
