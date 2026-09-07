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
	enc     apple.MTLComputeCommandEncoder
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
	if err := b.ops.hc.encodePre(b, state, hcFn, hcScale, hcBase, norm, out); err != nil {
		return err
	}
	// Encoded counts API operations, not the operation's internal dispatches.
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
	enc := b.encoder()
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
	enc := b.encoder()
	enc.SetComputePipelineState(b.ops.kernels.MatvecF16)
	enc.SetBufferWithOffsetAtIndex(t.data.Buffer(), t.data.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 2)
	libmetal.SetScalar(enc, uint32(in.Len()), 3)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(out.Len()), libmetal.Threads(min(out.Len(), 256)))
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
	enc := b.encoder()
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
	b.encoded++
	return nil
}

// EncodeRopeAttentionRaw appends rotary attention over the uncompressed cache.
// out holds the rotated query, rotated key-value vector, and inverse-rotated
// attention heads, in that order.
func (b *metalBatch) EncodeRopeAttentionRaw(q, kv metalmem.Slice[float32], sinks boundMetalTensor, cache, scores, out metalmem.Slice[float32], position int) error {
	if err := b.ready(); err != nil {
		return err
	}
	if q.Len() != 64*512 || kv.Len() != 512 || sinks.typeID != 0 || len(sinks.dimensions) != 1 || sinks.dimensions[0] != 64 ||
		cache.Len() == 0 || cache.Len()%512 != 0 || scores.Len() != 64*(cache.Len()/512) || out.Len() != 32768+512+32768 || position < 0 {
		return fmt.Errorf("raw attention: invalid shape")
	}
	enc := b.encoder()
	enc.SetComputePipelineState(b.ops.kernels.RopeAttentionRaw)
	for i, data := range []metalmem.Slice[float32]{q, kv} {
		enc.SetBufferWithOffsetAtIndex(data.Buffer(), data.Offset(), uint(i))
	}
	enc.SetBufferWithOffsetAtIndex(sinks.data.Buffer(), sinks.data.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(cache.Buffer(), cache.Offset(), 3)
	enc.SetBufferWithOffsetAtIndex(scores.Buffer(), scores.Offset(), 4)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 5)
	libmetal.SetScalar(enc, uint32(position), 6)
	libmetal.SetScalar(enc, uint32(cache.Len()/512), 7)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(64))
	b.encoded++
	return nil
}

// EncodeCompressor appends one compressor update and, at emission positions,
// writes the completed row to the compressed cache. The caller advances state.n
// only after the batch completes successfully.
func (b *metalBatch) EncodeCompressor(kv, score metalmem.Slice[float32], ape, norm boundMetalTensor, state *metalCompressedState, position int, indexer bool) (bool, error) {
	if err := b.ready(); err != nil {
		return false, err
	}
	wantHeadDim := 512
	if indexer {
		wantHeadDim = 128
	}
	if state == nil || state.headDim != wantHeadDim || (state.ratio != 4 && state.ratio != 128) ||
		kv.Len() != state.width || score.Len() != state.width || position < 0 ||
		ape.typeID != 1 || norm.typeID != 0 || len(ape.dimensions) != 2 ||
		ape.dimensions[0] != uint64(state.width) || ape.dimensions[1] != uint64(state.ratio) {
		return false, fmt.Errorf("attention compressor: invalid shape")
	}
	emit := (position+1)%state.ratio == 0
	if emit && state.n >= state.cap {
		return false, fmt.Errorf("attention compressor: cache capacity exceeded")
	}
	enc := b.encoder()
	enc.SetComputePipelineState(b.ops.kernels.CompressorPool)
	for i, data := range []metalmem.Slice[float32]{kv, score, state.stateKV, state.stateScore, state.pooled} {
		enc.SetBufferWithOffsetAtIndex(data.Buffer(), data.Offset(), uint(i))
	}
	enc.SetBufferWithOffsetAtIndex(ape.data.Buffer(), ape.data.Offset(), 5)
	libmetal.SetScalar(enc, uint32(state.headDim), 6)
	libmetal.SetScalar(enc, uint32(state.width), 7)
	libmetal.SetScalar(enc, uint32(state.ratio), 8)
	libmetal.SetScalar(enc, uint32(position), 9)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(state.headDim))
	b.encoded++
	if !emit {
		return false, nil
	}
	enc = b.encoder()
	if indexer {
		enc.SetComputePipelineState(b.ops.kernels.IndexerFinish)
	} else {
		enc.SetComputePipelineState(b.ops.kernels.CompressorFinish)
	}
	enc.SetBufferWithOffsetAtIndex(state.pooled.Buffer(), state.pooled.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(norm.data.Buffer(), norm.data.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(state.cache.Buffer(), state.cache.Offset(), 2)
	libmetal.SetScalar(enc, uint32(state.n), 3)
	libmetal.SetScalar(enc, uint32(position+1-state.ratio), 4)
	scratch := 32 * 4
	if indexer {
		scratch = 256 * 4
	}
	enc.SetThreadgroupMemoryLengthAtIndex(uint(scratch), 0)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(128))
	b.encoded++
	return true, nil
}

// EncodeRopeAttentionMixed appends attention over raw and compressed caches.
func (b *metalBatch) EncodeRopeAttentionMixed(q, kv metalmem.Slice[float32], sinks boundMetalTensor, raw, comp metalmem.Slice[float32], nComp int, allowed metalmem.Slice[uint32], scores, out metalmem.Slice[float32], position, layer int) error {
	if err := b.ready(); err != nil {
		return err
	}
	rowCap := raw.Len() / 512
	nRaw := min(position+1, rowCap)
	if q.Len() != 64*512 || kv.Len() != 512 || sinks.typeID != 0 || len(sinks.dimensions) != 1 || sinks.dimensions[0] != 64 ||
		rowCap == 0 || raw.Len()%512 != 0 || comp.Len()%512 != 0 || nComp < 0 || nComp > comp.Len()/512 ||
		allowed.Len() < max(1, nComp) || scores.Len() < 64*(nRaw+nComp) || out.Len() != 32768+512+32768 || position < 0 || layer < 2 {
		return fmt.Errorf("mixed attention: invalid shape")
	}
	enc := b.encoder()
	enc.SetComputePipelineState(b.ops.kernels.RopeAttentionMix)
	for i, data := range []metalmem.Slice[float32]{q, kv} {
		enc.SetBufferWithOffsetAtIndex(data.Buffer(), data.Offset(), uint(i))
	}
	enc.SetBufferWithOffsetAtIndex(sinks.data.Buffer(), sinks.data.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(raw.Buffer(), raw.Offset(), 3)
	enc.SetBufferWithOffsetAtIndex(comp.Buffer(), comp.Offset(), 4)
	enc.SetBufferWithOffsetAtIndex(allowed.Buffer(), allowed.Offset(), 5)
	enc.SetBufferWithOffsetAtIndex(scores.Buffer(), scores.Offset(), 6)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 7)
	libmetal.SetScalar(enc, uint32(position), 8)
	libmetal.SetScalar(enc, uint32(layer), 9)
	libmetal.SetScalar(enc, uint32(rowCap), 10)
	libmetal.SetScalar(enc, uint32(nComp), 11)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(64))
	b.encoded++
	return nil
}

// EncodeMatvecIQ2Selected appends paired IQ2_XXS projections for GPU-selected experts.
func (b *metalBatch) EncodeMatvecIQ2Selected(gate, up boundMetalTensor, expertIDs metalmem.Slice[int32], in, gateOut, upOut metalmem.Slice[float32]) error {
	if err := b.ready(); err != nil {
		return err
	}
	if gate.typeID != 16 || up.typeID != 16 || len(gate.dimensions) != 3 || len(up.dimensions) != 3 ||
		gate.dimensions[0] != up.dimensions[0] || gate.dimensions[1] != up.dimensions[1] || gate.dimensions[2] != up.dimensions[2] ||
		gate.dimensions[0]%256 != 0 || uint64(in.Len()) != gate.dimensions[0] || expertIDs.Len() == 0 ||
		uint64(gateOut.Len()) != gate.dimensions[1]*uint64(expertIDs.Len()) || upOut.Len() != gateOut.Len() {
		return fmt.Errorf("IQ2_XXS selected pair: invalid tensor profile")
	}
	enc := b.encoder()
	enc.SetComputePipelineState(b.ops.kernels.MatvecIQ2Selected)
	for i, tensor := range []boundMetalTensor{gate, up} {
		enc.SetBufferWithOffsetAtIndex(tensor.data.Buffer(), tensor.data.Offset(), uint(i))
	}
	enc.SetBufferWithOffsetAtIndex(expertIDs.Buffer(), expertIDs.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 3)
	enc.SetBufferWithOffsetAtIndex(gateOut.Buffer(), gateOut.Offset(), 4)
	enc.SetBufferWithOffsetAtIndex(upOut.Buffer(), upOut.Offset(), 5)
	libmetal.SetScalar(enc, uint32(in.Len()), 6)
	libmetal.SetScalar(enc, uint32(gate.dimensions[1]), 7)
	libmetal.SetScalar(enc, uint32(gate.dimensions[2]), 8)
	libmetal.SetScalar(enc, uint32(expertIDs.Len()), 9)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(
		apple.MTLSize{Width: uint(gate.dimensions[1]), Height: uint(expertIDs.Len()), Depth: 1},
		libmetal.Threads(32),
	)
	b.encoded++
	return nil
}

// EncodeMatvecQ2KSelected appends Q2_K projections for GPU-selected experts.
// Inputs and outputs are slot-major, with one vector for each expert ID.
func (b *metalBatch) EncodeMatvecQ2KSelected(t boundMetalTensor, expertIDs metalmem.Slice[int32], in, out metalmem.Slice[float32]) error {
	if err := b.ready(); err != nil {
		return err
	}
	if t.typeID != 10 || len(t.dimensions) != 3 || t.dimensions[0]%256 != 0 || expertIDs.Len() != 6 ||
		uint64(in.Len()) != t.dimensions[0]*uint64(expertIDs.Len()) || uint64(out.Len()) != t.dimensions[1]*uint64(expertIDs.Len()) {
		return fmt.Errorf("Q2_K selected: invalid tensor profile")
	}
	enc := b.encoder()
	enc.SetComputePipelineState(b.ops.kernels.MatvecQ2KSelected)
	enc.SetBufferWithOffsetAtIndex(t.data.Buffer(), t.data.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(expertIDs.Buffer(), expertIDs.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 3)
	libmetal.SetScalar(enc, uint32(t.dimensions[0]), 4)
	libmetal.SetScalar(enc, uint32(t.dimensions[1]), 5)
	libmetal.SetScalar(enc, uint32(t.dimensions[2]), 6)
	libmetal.SetScalar(enc, uint32(expertIDs.Len()), 7)
	enc.DispatchThreadsThreadsPerThreadgroup(
		apple.MTLSize{Width: uint(t.dimensions[1]), Height: uint(expertIDs.Len()), Depth: 1},
		apple.MTLSize{Width: uint(min(int(t.dimensions[1]), 256)), Height: 1, Depth: 1},
	)
	b.encoded++
	return nil
}

// EncodeClamp10 appends an element-wise clamp to [-10, 10].
func (b *metalBatch) EncodeClamp10(in, out metalmem.Slice[float32]) error {
	return b.encodeClamp10(in, out, false)
}

// EncodeClampGate10 appends the gated clamp used by routed experts.
func (b *metalBatch) EncodeClampGate10(in, out metalmem.Slice[float32]) error {
	return b.encodeClamp10(in, out, true)
}

func (b *metalBatch) encodeClamp10(in, out metalmem.Slice[float32], gate bool) error {
	if err := b.ready(); err != nil {
		return err
	}
	if in.Len() == 0 || out.Len() != in.Len() {
		return fmt.Errorf("clamp: invalid shape")
	}
	enc := b.encoder()
	enc.SetComputePipelineState(b.ops.kernels.Clamp10)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 1)
	libmetal.SetScalar(enc, uint32(in.Len()), 2)
	libmetal.SetScalar(enc, boolScalar(gate), 3)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(in.Len()), libmetal.Threads(min(in.Len(), 256)))
	b.encoded++
	return nil
}

// EncodeSwiGLU appends an element-wise SwiGLU operation.
func (b *metalBatch) EncodeSwiGLU(gate, up, out metalmem.Slice[float32]) error {
	return b.encodeSwiGLU(gate, up, out, 1)
}

// EncodeSwiGLUWeighted appends an element-wise weighted SwiGLU operation.
func (b *metalBatch) EncodeSwiGLUWeighted(gate, up, out metalmem.Slice[float32], weight float32) error {
	return b.encodeSwiGLU(gate, up, out, weight)
}

// EncodeSwiGLUWeightedSlots appends a slot-major SwiGLU operation whose
// weights remain GPU-resident.
func (b *metalBatch) EncodeSwiGLUWeightedSlots(gate, up, weights, out metalmem.Slice[float32]) error {
	if err := b.ready(); err != nil {
		return err
	}
	if gate.Len() == 0 || up.Len() != gate.Len() || out.Len() != gate.Len() || weights.Len() == 0 || gate.Len()%weights.Len() != 0 {
		return fmt.Errorf("weighted SwiGLU slots: invalid shape")
	}
	enc := b.encoder()
	enc.SetComputePipelineState(b.ops.kernels.SwiGLUWeights)
	enc.SetBufferWithOffsetAtIndex(gate.Buffer(), gate.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(up.Buffer(), up.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(weights.Buffer(), weights.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 3)
	libmetal.SetScalar(enc, uint32(gate.Len()/weights.Len()), 4)
	libmetal.SetScalar(enc, uint32(weights.Len()), 5)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(gate.Len()), libmetal.Threads(min(gate.Len(), 256)))
	b.encoded++
	return nil
}

// EncodeRouterHash appends hash routing into caller-owned GPU storage.
func (b *metalBatch) EncodeRouterHash(logits metalmem.Slice[float32], table boundMetalTensor, token uint32, probs metalmem.Slice[float32], expertIDs metalmem.Slice[int32], expertW metalmem.Slice[float32]) error {
	if err := b.ready(); err != nil {
		return err
	}
	if logits.Len() != 256 || probs.Len() != 256 || expertIDs.Len() != 6 || expertW.Len() != 6 || table.typeID != 26 || len(table.dimensions) != 2 || table.dimensions[0] != 6 || uint64(token) >= table.dimensions[1] {
		return fmt.Errorf("router hash: invalid tensor profile")
	}
	enc := b.encoder()
	enc.SetComputePipelineState(b.ops.kernels.RouterHash)
	enc.SetBufferWithOffsetAtIndex(logits.Buffer(), logits.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(table.data.Buffer(), table.data.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(probs.Buffer(), probs.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(expertIDs.Buffer(), expertIDs.Offset(), 3)
	enc.SetBufferWithOffsetAtIndex(expertW.Buffer(), expertW.Offset(), 4)
	libmetal.SetScalar(enc, token, 5)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(1))
	b.encoded++
	return nil
}

// EncodeRouterBiased appends biased top-six routing into caller-owned GPU storage.
func (b *metalBatch) EncodeRouterBiased(logits metalmem.Slice[float32], bias boundMetalTensor, probs metalmem.Slice[float32], expertIDs metalmem.Slice[int32], expertW metalmem.Slice[float32]) error {
	if err := b.ready(); err != nil {
		return err
	}
	if logits.Len() != 256 || probs.Len() != 256 || expertIDs.Len() != 6 || expertW.Len() != 6 || bias.typeID != 0 || len(bias.dimensions) != 1 || bias.dimensions[0] != 256 {
		return fmt.Errorf("biased router: invalid tensor profile")
	}
	enc := b.encoder()
	enc.SetComputePipelineState(b.ops.kernels.RouterBiased)
	enc.SetBufferWithOffsetAtIndex(logits.Buffer(), logits.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(bias.data.Buffer(), bias.data.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(probs.Buffer(), probs.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(expertIDs.Buffer(), expertIDs.Offset(), 3)
	enc.SetBufferWithOffsetAtIndex(expertW.Buffer(), expertW.Offset(), 4)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(1))
	b.encoded++
	return nil
}

func (b *metalBatch) encodeSwiGLU(gate, up, out metalmem.Slice[float32], weight float32) error {
	if err := b.ready(); err != nil {
		return err
	}
	if gate.Len() == 0 || up.Len() != gate.Len() || out.Len() != gate.Len() {
		return fmt.Errorf("SwiGLU: invalid shape")
	}
	enc := b.encoder()
	enc.SetComputePipelineState(b.ops.kernels.SwiGLU)
	enc.SetBufferWithOffsetAtIndex(gate.Buffer(), gate.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(up.Buffer(), up.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 2)
	libmetal.SetScalar(enc, uint32(gate.Len()), 3)
	libmetal.SetScalar(enc, weight, 4)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(gate.Len()), libmetal.Threads(min(gate.Len(), 256)))
	b.encoded++
	return nil
}

// EncodeSumVectors appends a reduction across count adjacent vectors.
func (b *metalBatch) EncodeSumVectors(in, out metalmem.Slice[float32], count int) error {
	if err := b.ready(); err != nil {
		return err
	}
	if count <= 0 || in.Len() == 0 || in.Len()%count != 0 || out.Len() != in.Len()/count {
		return fmt.Errorf("sum vectors: invalid shape")
	}
	enc := b.encoder()
	enc.SetComputePipelineState(b.ops.kernels.SumVectors)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 1)
	libmetal.SetScalar(enc, uint32(out.Len()), 2)
	libmetal.SetScalar(enc, uint32(count), 3)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(out.Len()), libmetal.Threads(min(out.Len(), 256)))
	b.encoded++
	return nil
}

// EncodeHCPost appends the hyper-connection residual update.
func (b *metalBatch) EncodeHCPost(state, block metalmem.Slice[float32], hcFn, hcScale, hcBase boundMetalTensor, out metalmem.Slice[float32]) error {
	if err := b.ready(); err != nil {
		return err
	}
	if err := b.ops.hc.encodePost(b, state, block, hcFn, hcScale, hcBase, out); err != nil {
		return err
	}
	b.encoded++
	return nil
}

// Wait commits the batch and waits for its single command buffer to complete.
// encoder returns the batch's compute encoder, opening it on first use.
// Operations share one encoder because they already share one command buffer:
// a serial compute encoder orders its dispatches and tracks their hazards, so
// the arithmetic is unchanged and only the submission structure shrinks.
func (b *metalBatch) encoder() apple.MTLComputeCommandEncoder {
	if b.enc == nil {
		b.enc = b.buffer.ComputeCommandEncoder()
	}
	return b.enc
}

// endEncoding closes the shared encoder. Committing without it drops every
// dispatch, so Wait calls it before Commit.
func (b *metalBatch) endEncoding() {
	if b.enc != nil {
		b.enc.EndEncoding()
		b.enc = nil
	}
}

func (b *metalBatch) Wait(ctx context.Context) error {
	if err := b.ready(); err != nil {
		return err
	}
	b.endEncoding()
	b.done = true
	b.buffer.Commit()
	b.commits++
	b.waits++
	// One span per command buffer: the batch's dispatches are the GPU
	// work, and everything between this span and the next is host cost.
	wait := beginOp("batch.wait")
	err := libmetal.WaitCommandBuffer(ctx, b.buffer)
	wait.end()
	b.ops.hc.complete(b)
	if err != nil {
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
