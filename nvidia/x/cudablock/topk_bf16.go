package cudablock

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// TopKBF16Buffers holds caller-owned BF16 input and FP32/int32 output storage.
// Values are exactly decoded FP32, not rounded back to BF16. Field extents,
// device ownership, alias rejection and publication rules match TopKBuffers.
// Input is two-byte aligned; all other active views are four-byte aligned.
type TopKBF16Buffers struct {
	Input          cudamem.Slice[uint16]
	Lengths        cudamem.Slice[int32]
	Offsets        cudamem.Slice[int32]
	ScratchIndices cudamem.Slice[int32]
	ScratchValues  cudamem.Slice[float32]
	ScratchCounts  cudamem.Slice[int32]
	Status         cudamem.Slice[int32]
	Indices        cudamem.Slice[int32]
	Values         cudamem.Slice[float32]
	Counts         cudamem.Slice[int32]
}

// TopKBF16 owns a BF16-input selection artifact. Rows and stride are measured
// in BF16 elements. Decoding shifts the storage bits left 16 into an FP32 word;
// no rounding or FP16 conversion is performed. Every NaN payload rejects the
// invocation, including short rows. Infinities, signed zeros and lowest-index
// ties follow TopKConfig's decoded-value policy. Its zero value is closed.
// Execute and Close are serialized. Device and stream lifetimes match TopK.
type TopKBF16 struct {
	owner *TopK
}

// CompileTopKBF16 compiles the ordinary one-thread-per-row BF16 baseline.
// Config limits and output extents match FP32; input uses two bytes per element.
// A nonnil result returned with an error retains cleanup and must be closed.
func CompileTopKBF16(dev *cuda.Device, config TopKConfig) (*TopKBF16, error) {
	owner, err := compileTopKNamedSource(dev, config, TopKBF16Source, "topk_bf16")
	if owner == nil {
		return nil, err
	}
	return &TopKBF16{owner: owner}, err
}

// Artifact returns a detached descriptor for the exact BF16-input image.
func (t *TopKBF16) Artifact() *cudakernel.Artifact {
	if t == nil {
		return nil
	}
	return t.owner.Artifact()
}

// Close drains failed work and retries cleanup through the shared TopK owner.
func (t *TopKBF16) Close() error {
	if t == nil {
		return nil
	}
	return t.owner.Close()
}

// Execute validates all rows before publishing outputs on the same stream.
// Scratch and status are diagnostic only. An error before publication leaves
// outputs untouched; publication/completion errors leave them undefined.
// Failed completion retains borrows until Close successfully drains the stream.
func (t *TopKBF16) Execute(stream *cuda.Stream, b TopKBF16Buffers) error {
	if t == nil || t.owner == nil {
		return fmt.Errorf("cudablock: closed topk bf16")
	}
	input := topkBuffer{name: "input", ptr: uint64(b.Input.Ptr()), length: b.Input.Len(), owner: b.Input.Device(), borrow: b.Input.Borrow, bytes: 2, alignment: 2}
	common := TopKBuffers{Lengths: b.Lengths, Offsets: b.Offsets, ScratchIndices: b.ScratchIndices, ScratchValues: b.ScratchValues, ScratchCounts: b.ScratchCounts, Status: b.Status, Indices: b.Indices, Values: b.Values, Counts: b.Counts}
	return t.owner.execute(stream, common, input)
}
