//go:build darwin && arm64

package main

import (
	"context"
	"fmt"

	apple "github.com/tmc/apple/metal"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// metalBatch records dependent operations in one Metal command buffer.
// Inputs and outputs remain GPU-resident until Wait returns.
type metalBatch struct {
	ops     *metalPinnedOps
	buffer  apple.MTLCommandBuffer
	encoded int
	commits int
	waits   int
	done    bool
}

type metalBatchStats struct {
	Encoded int
	Commits int
	Waits   int
}

func newMetalBatch(ops *metalPinnedOps) (*metalBatch, error) {
	if ops == nil || ops.dev == nil {
		return nil, fmt.Errorf("Metal batch: missing ops")
	}
	return &metalBatch{ops: ops, buffer: ops.dev.Queue().CommandBuffer()}, nil
}

// EncodeHCPre appends the hyper-connection input projection and normalization.
// out holds the 4096-element projection followed by its normalization.
func (b *metalBatch) EncodeHCPre(state metalmem.Slice[float32], hcFn, hcScale, hcBase, norm boundMetalTensor, out metalmem.Slice[float32]) error {
	if err := b.ready(); err != nil {
		return err
	}
	if state.Len() != 4*4096 || out.Len() != 2*4096 || hcFn.typeID != 1 || hcScale.typeID != 0 || hcBase.typeID != 0 || norm.typeID != 0 {
		return fmt.Errorf("HC pre: invalid tensor profile")
	}
	enc := b.buffer.ComputeCommandEncoder()
	enc.SetComputePipelineState(b.ops.kernels.HCPre)
	enc.SetBufferWithOffsetAtIndex(state.Buffer(), state.Offset(), 0)
	for i, tensor := range []boundMetalTensor{hcFn, hcScale, hcBase, norm} {
		enc.SetBufferWithOffsetAtIndex(tensor.data.Buffer(), tensor.data.Offset(), uint(i+1))
	}
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 5)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(1))
	enc.EndEncoding()
	b.encoded++
	return nil
}

// EncodeRMS appends an RMS normalization using caller-owned storage.
func (b *metalBatch) EncodeRMS(in metalmem.Slice[float32], weight boundMetalTensor, out metalmem.Slice[float32], groups int, weighted bool) error {
	if err := b.ready(); err != nil {
		return err
	}
	if in.Len() == 0 || out.Len() != in.Len() || groups <= 0 || in.Len()%groups != 0 || weight.typeID != 0 || len(weight.dimensions) == 0 ||
		(weighted && uint64(in.Len()/groups) != weight.dimensions[0]) {
		return fmt.Errorf("RMS norm: invalid shape")
	}
	enc := b.buffer.ComputeCommandEncoder()
	enc.SetComputePipelineState(b.ops.kernels.RMS)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(weight.data.Buffer(), weight.data.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 2)
	libmetal.SetScalar(enc, uint32(in.Len()/groups), 3)
	libmetal.SetScalar(enc, uint32(groups), 4)
	libmetal.SetScalar(enc, boolScalar(weighted), 5)
	enc.SetThreadgroupMemoryLengthAtIndex(32*4, 0)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(
		apple.MTLSize{Width: uint(groups), Height: 1, Depth: 1},
		apple.MTLSize{Width: 256, Height: 1, Depth: 1},
	)
	enc.EndEncoding()
	b.encoded++
	return nil
}

// EncodeMatvecF16 appends an F16 matrix-vector product using caller-owned storage.
func (b *metalBatch) EncodeMatvecF16(t boundMetalTensor, in, out metalmem.Slice[float32]) error {
	if err := b.ready(); err != nil {
		return err
	}
	if t.typeID != 1 || len(t.dimensions) != 2 || uint64(in.Len()) != t.dimensions[0] || uint64(out.Len()) != t.dimensions[1] {
		return fmt.Errorf("matvec %q: invalid F16 tensor or input", t.name)
	}
	enc := b.buffer.ComputeCommandEncoder()
	enc.SetComputePipelineState(b.ops.kernels.MatvecF16)
	enc.SetBufferWithOffsetAtIndex(t.data.Buffer(), t.data.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 2)
	libmetal.SetScalar(enc, uint32(in.Len()), 3)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(out.Len()), libmetal.Threads(min(out.Len(), 256)))
	enc.EndEncoding()
	b.encoded++
	return nil
}

// EncodeMatvecQ80 appends a Q8_0 matrix-vector product using caller-owned storage.
func (b *metalBatch) EncodeMatvecQ80(t boundMetalTensor, in, out metalmem.Slice[float32]) error {
	if err := b.ready(); err != nil {
		return err
	}
	if t.typeID != 8 || len(t.dimensions) != 2 || t.dimensions[0]%32 != 0 || uint64(in.Len()) != t.dimensions[0] || uint64(out.Len()) != t.dimensions[1] {
		return fmt.Errorf("matvec %q: invalid Q8_0 tensor or input", t.name)
	}
	enc := b.buffer.ComputeCommandEncoder()
	enc.SetComputePipelineState(b.ops.kernels.MatvecQ80)
	enc.SetBufferWithOffsetAtIndex(t.data.Buffer(), t.data.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 2)
	libmetal.SetScalar(enc, uint32(in.Len()), 3)
	libmetal.SetScalar(enc, uint32(out.Len()), 4)
	enc.SetThreadgroupMemoryLengthAtIndex(2*32*4, 0)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(
		apple.MTLSize{Width: uint((out.Len() + 1) / 2), Height: 1, Depth: 1},
		apple.MTLSize{Width: 32, Height: 4, Depth: 1},
	)
	enc.EndEncoding()
	b.encoded++
	return nil
}

// Wait commits the batch and waits for its single command buffer to complete.
func (b *metalBatch) Wait(ctx context.Context) error {
	if err := b.ready(); err != nil {
		return err
	}
	b.done = true
	b.buffer.Commit()
	b.commits++
	b.waits++
	if err := libmetal.WaitCommandBuffer(ctx, b.buffer); err != nil {
		return fmt.Errorf("Metal batch: %w", err)
	}
	return nil
}

func (b *metalBatch) Stats() metalBatchStats {
	if b == nil {
		return metalBatchStats{}
	}
	return metalBatchStats{Encoded: b.encoded, Commits: b.commits, Waits: b.waits}
}

func (b *metalBatch) ready() error {
	if b == nil || b.ops == nil || b.buffer == nil {
		return fmt.Errorf("Metal batch: not initialized")
	}
	if b.done {
		return fmt.Errorf("Metal batch: already committed")
	}
	return nil
}

func boolScalar(v bool) uint32 {
	if v {
		return 1
	}
	return 0
}
