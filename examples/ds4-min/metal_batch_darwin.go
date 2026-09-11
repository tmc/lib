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

	// Set by inRegion; recorded only when ops.regions is installed.
	region   string
	layer    int
	position int
}

// inRegion names the part of the layer this batch carries, for a
// metalRegionLog. It returns b so a caller can name a batch on the line
// that creates it. Naming a batch has no effect unless a log is
// installed on the ops.
func (b *metalBatch) inRegion(name string, layer, position int) *metalBatch {
	b.region, b.layer, b.position = name, layer, position
	return b
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
	return &metalBatch{ops: ops, buffer: ops.commandBuffer()}, nil
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
//
// This is the encoder that does not widen the way the matmuls did, and the
// reason is in compressor_pool rather than here. The kernel writes row
// position%ratio of a scratch window and pools only when the window closes, so
// a token's work depends on where it sits in its window -- but the two ratios
// differ in kind, which decides what a batched prefill can do with each:
//
//	ratio 128 (indexer): rows 0..ratio, pooled, no shift. Windows are
//	independent, so this is a map over T/128 windows and batches cleanly.
//
//	ratio 4 (attention): a 2*ratio sliding buffer. The pool reads the previous
//	window's rows alongside the current one, and the kernel then shifts rows
//	[ratio,2*ratio) down into [0,ratio). That carry makes it a scan: T tokens
//	are T/4 dependent window steps, not one dispatch.
//
// So batching prefill does not collapse the attention compressor, only the
// matmuls around it. That bounds the shape of a widening but not its value:
// the scan moves state buffers, not weights, so it is not what the streaming
// cost is made of. CUDA's cudaPrefillBatchWidth of 8 is two window steps.
//
// One difference from the CUDA port, noted and deliberately not changed here.
// ds4_compressor_pool does the down-shift alone; compressor_pool follows it
// with a device barrier and a copy back up, which restores what the shift just
// read and so leaves both halves holding the old upper rows. The net values
// agree with CUDA -- the backends are not divergent -- but Metal pays 16
// redundant stores per thread per emission for the round trip. Removing it is
// a kernel edit under the logits digest, so it wants the model-loading anchor
// as its gate rather than a reviewer's reading of the loop.
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
	// Four dispatches for one logical operation, matching
	// metalPinnedOps.ropeAttentionMixed. The KV preamble is separated so the
	// attention kernel can put each head in its own threadgroup; ordering
	// within the encoder supplies the dependency the old device-scope barrier
	// enforced. b.encoded counts operations at this API's level, so it still
	// advances by one.
	enc.SetComputePipelineState(b.ops.kernels.RopeQPrepare)
	enc.SetBufferWithOffsetAtIndex(q.Buffer(), q.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(kv.Buffer(), kv.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 2)
	libmetal.SetScalar(enc, uint32(position), 3)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(64))

	enc.SetComputePipelineState(b.ops.kernels.RopeKVCommit)
	enc.SetBufferWithOffsetAtIndex(raw.Buffer(), raw.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 1)
	libmetal.SetScalar(enc, uint32(position), 2)
	libmetal.SetScalar(enc, uint32(rowCap), 3)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(mixedWideThreads))

	enc.SetComputePipelineState(b.ops.kernels.RopeAttentionWide)
	enc.SetBufferWithOffsetAtIndex(sinks.data.Buffer(), sinks.data.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(raw.Buffer(), raw.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(comp.Buffer(), comp.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(allowed.Buffer(), allowed.Offset(), 3)
	enc.SetBufferWithOffsetAtIndex(scores.Buffer(), scores.Offset(), 4)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 5)
	libmetal.SetScalar(enc, uint32(position), 6)
	libmetal.SetScalar(enc, uint32(rowCap), 7)
	libmetal.SetScalar(enc, uint32(nComp), 8)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(64), libmetal.Threads(mixedWideThreads))

	enc.SetComputePipelineState(b.ops.kernels.RopeOutUnrotate)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 0)
	libmetal.SetScalar(enc, uint32(position), 1)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(64))
	_ = layer
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

// EncodeSwiGLUWeightedGPU appends a weighted SwiGLU operation whose weight is
// read from GPU memory instead of an encode-time scalar. The kernel declares
// the weight as a constant reference, which Metal binds from a buffer as
// readily as from SetBytes, so this is the same uniform the scalar form
// produces and the same kernel; TestMetalSwiGLUSlotsBitIdentity pins that the
// two agree bit for bit. The slot-major kernel does not, because there the
// weight is a divergent device load rather than a constant.
func (b *metalBatch) EncodeSwiGLUWeightedGPU(gate, up, out, weight metalmem.Slice[float32]) error {
	if err := b.ready(); err != nil {
		return err
	}
	if gate.Len() == 0 || up.Len() != gate.Len() || out.Len() != gate.Len() || weight.Len() != 1 {
		return fmt.Errorf("weighted SwiGLU: invalid shape")
	}
	enc := b.encoder()
	enc.SetComputePipelineState(b.ops.kernels.SwiGLU)
	enc.SetBufferWithOffsetAtIndex(gate.Buffer(), gate.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(up.Buffer(), up.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 2)
	libmetal.SetScalar(enc, uint32(gate.Len()), 3)
	enc.SetBufferWithOffsetAtIndex(weight.Buffer(), weight.Offset(), 4)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(gate.Len()), libmetal.Threads(min(gate.Len(), 256)))
	b.encoded++
	return nil
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

// tokenRow returns row token of a token-major slice whose rows are width
// elements wide. Row 0 of a one-row slice is the slice itself: the same
// buffer at the same offset with the same length, so a one-token call
// through a token-major path binds exactly what the single-token form binds.
func tokenRow[T any](s metalmem.Slice[T], token, width int) (metalmem.Slice[T], error) {
	return s.Slice(token*width, (token+1)*width)
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

// EncodeRouterHashTokens appends hash routing for tokens token rows.
//
// logits and probs hold 256 elements for each token, expertIDs and expertW
// six, and tokenIDs one token id for each row, all token-major. The token id
// the kernel hashes with comes from tokenIDs rather than an encode-time
// scalar: the kernel declares it as a constant reference, which Metal binds
// from a buffer as readily as from SetBytes, so the uniform each dispatch
// sees is the one the scalar form produces. EncodeSwiGLUWeightedGPU rests on
// the same property.
//
// Each token keeps its own dispatch. router_hash is a one-thread kernel that
// reads 256 logits and writes six slots with no cross-token term, so widening
// it to a thread per token would mean a different kernel; the rows here are
// the same arithmetic the single-token form runs, moved into one command
// buffer. TestMetalRouterTokensBitIdentity pins that against N single-token
// calls.
func (b *metalBatch) EncodeRouterHashTokens(logits metalmem.Slice[float32], table boundMetalTensor, tokenIDs metalmem.Slice[uint32], probs metalmem.Slice[float32], expertIDs metalmem.Slice[int32], expertW metalmem.Slice[float32], tokens int) error {
	if err := b.ready(); err != nil {
		return err
	}
	if tokens <= 0 || logits.Len() != 256*tokens || probs.Len() != 256*tokens || expertIDs.Len() != 6*tokens || expertW.Len() != 6*tokens ||
		tokenIDs.Len() != tokens || table.typeID != 26 || len(table.dimensions) != 2 || table.dimensions[0] != 6 {
		return fmt.Errorf("router hash: invalid tensor profile")
	}
	// The single-token form rejects an out-of-range token id at encode time.
	// The ids now live in a buffer, but it is unified memory the caller
	// filled before encoding, so the same check still runs here.
	ids := tokenIDs.HostSlice()
	if len(ids) != tokens {
		return fmt.Errorf("router hash: token ids are not host-visible")
	}
	for i, id := range ids {
		if uint64(id) >= table.dimensions[1] {
			return fmt.Errorf("router hash: token %d id %d out of range", i, id)
		}
	}
	enc := b.encoder()
	enc.SetComputePipelineState(b.ops.kernels.RouterHash)
	enc.SetBufferWithOffsetAtIndex(table.data.Buffer(), table.data.Offset(), 1)
	for token := 0; token < tokens; token++ {
		rowLogits, err := tokenRow(logits, token, 256)
		if err != nil {
			return fmt.Errorf("router hash: logits row %d: %w", token, err)
		}
		rowProbs, err := tokenRow(probs, token, 256)
		if err != nil {
			return fmt.Errorf("router hash: probs row %d: %w", token, err)
		}
		rowIDs, err := tokenRow(expertIDs, token, 6)
		if err != nil {
			return fmt.Errorf("router hash: expert IDs row %d: %w", token, err)
		}
		rowW, err := tokenRow(expertW, token, 6)
		if err != nil {
			return fmt.Errorf("router hash: expert weights row %d: %w", token, err)
		}
		rowToken, err := tokenRow(tokenIDs, token, 1)
		if err != nil {
			return fmt.Errorf("router hash: token id row %d: %w", token, err)
		}
		enc.SetBufferWithOffsetAtIndex(rowLogits.Buffer(), rowLogits.Offset(), 0)
		enc.SetBufferWithOffsetAtIndex(rowProbs.Buffer(), rowProbs.Offset(), 2)
		enc.SetBufferWithOffsetAtIndex(rowIDs.Buffer(), rowIDs.Offset(), 3)
		enc.SetBufferWithOffsetAtIndex(rowW.Buffer(), rowW.Offset(), 4)
		enc.SetBufferWithOffsetAtIndex(rowToken.Buffer(), rowToken.Offset(), 5)
		enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(1))
	}
	// Encoded counts API operations, not the operation's internal dispatches.
	b.encoded++
	return nil
}

// EncodeRouterBiased appends biased top-six routing into caller-owned GPU storage.
func (b *metalBatch) EncodeRouterBiased(logits metalmem.Slice[float32], bias boundMetalTensor, probs metalmem.Slice[float32], expertIDs metalmem.Slice[int32], expertW metalmem.Slice[float32]) error {
	return b.EncodeRouterBiasedTokens(logits, bias, probs, expertIDs, expertW, 1)
}

// EncodeRouterBiasedTokens appends biased top-six routing for tokens token rows.
//
// logits and probs hold 256 elements for each token and expertIDs and expertW
// six, all token-major. Each token keeps its own dispatch of the same
// one-thread kernel the single-token form runs, so the selection scan and the
// weight normalization see one token's 256 scores in the order they always
// did. TestMetalRouterTokensBitIdentity pins that against N single-token calls.
func (b *metalBatch) EncodeRouterBiasedTokens(logits metalmem.Slice[float32], bias boundMetalTensor, probs metalmem.Slice[float32], expertIDs metalmem.Slice[int32], expertW metalmem.Slice[float32], tokens int) error {
	if err := b.ready(); err != nil {
		return err
	}
	if tokens <= 0 || logits.Len() != 256*tokens || probs.Len() != 256*tokens || expertIDs.Len() != 6*tokens || expertW.Len() != 6*tokens ||
		bias.typeID != 0 || len(bias.dimensions) != 1 || bias.dimensions[0] != 256 {
		return fmt.Errorf("biased router: invalid tensor profile")
	}
	enc := b.encoder()
	enc.SetComputePipelineState(b.ops.kernels.RouterBiased)
	enc.SetBufferWithOffsetAtIndex(bias.data.Buffer(), bias.data.Offset(), 1)
	for token := 0; token < tokens; token++ {
		rowLogits, err := tokenRow(logits, token, 256)
		if err != nil {
			return fmt.Errorf("biased router: logits row %d: %w", token, err)
		}
		rowProbs, err := tokenRow(probs, token, 256)
		if err != nil {
			return fmt.Errorf("biased router: probs row %d: %w", token, err)
		}
		rowIDs, err := tokenRow(expertIDs, token, 6)
		if err != nil {
			return fmt.Errorf("biased router: expert IDs row %d: %w", token, err)
		}
		rowW, err := tokenRow(expertW, token, 6)
		if err != nil {
			return fmt.Errorf("biased router: expert weights row %d: %w", token, err)
		}
		enc.SetBufferWithOffsetAtIndex(rowLogits.Buffer(), rowLogits.Offset(), 0)
		enc.SetBufferWithOffsetAtIndex(rowProbs.Buffer(), rowProbs.Offset(), 2)
		enc.SetBufferWithOffsetAtIndex(rowIDs.Buffer(), rowIDs.Offset(), 3)
		enc.SetBufferWithOffsetAtIndex(rowW.Buffer(), rowW.Offset(), 4)
		enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(1))
	}
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
	return b.EncodeSumVectorsTokens(in, out, count, 1)
}

// EncodeSumVectorsTokens appends, for each of tokens token rows, a reduction
// across that token's count vectors.
//
// The slices are token-major: out holds one width-element vector for each
// token, and in holds that token's count vectors of width elements each,
// grouped by token. The kernel reduces across vectors adjacent in memory, so
// the stride it must not cross is the token's, not the whole buffer's:
// deriving the width from in.Len()/count instead of out.Len()/tokens reduces
// down the token axis and sums one slot of every token together.
//
// Each token gets its own dispatch over its own count rows, which is the
// accumulation the single-token form performs, in the same slot order.
// TestMetalSumVectorsTokensBitIdentity pins that against N single-token calls,
// and its negative control pins that the two axes disagree on that data.
func (b *metalBatch) EncodeSumVectorsTokens(in, out metalmem.Slice[float32], count, tokens int) error {
	if err := b.ready(); err != nil {
		return err
	}
	if count <= 0 || tokens <= 0 || out.Len() == 0 || out.Len()%tokens != 0 {
		return fmt.Errorf("sum vectors: invalid shape")
	}
	width := out.Len() / tokens
	if in.Len() != width*count*tokens {
		return fmt.Errorf("sum vectors: invalid shape")
	}
	enc := b.encoder()
	enc.SetComputePipelineState(b.ops.kernels.SumVectors)
	for token := 0; token < tokens; token++ {
		src, err := tokenRow(in, token, width*count)
		if err != nil {
			return fmt.Errorf("sum vectors: input row %d: %w", token, err)
		}
		dst, err := tokenRow(out, token, width)
		if err != nil {
			return fmt.Errorf("sum vectors: output row %d: %w", token, err)
		}
		enc.SetBufferWithOffsetAtIndex(src.Buffer(), src.Offset(), 0)
		enc.SetBufferWithOffsetAtIndex(dst.Buffer(), dst.Offset(), 1)
		libmetal.SetScalar(enc, uint32(width), 2)
		libmetal.SetScalar(enc, uint32(count), 3)
		enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(width), libmetal.Threads(min(width, 256)))
	}
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

// Commit ends encoding and submits the batch without blocking on it.
//
// Command buffers submitted to one queue complete in submission order, so a
// caller may commit a batch, encode and commit the next while the GPU runs
// this one, and wait on both afterwards. Wait must still be called on a
// committed batch: waiting is what records the batch's span and releases its
// hyper-connection claim, and a batch that is committed but never waited on
// contributes neither.
func (b *metalBatch) Commit() error {
	if err := b.ready(); err != nil {
		return err
	}
	b.endEncoding()
	b.done = true
	b.buffer.Commit()
	b.commits++
	b.ops.submissions.Commits++
	b.ops.submissions.Encoded += b.encoded
	return nil
}

func (b *metalBatch) Wait(ctx context.Context) error {
	if b == nil || b.ops == nil || b.buffer == nil {
		return fmt.Errorf("Metal batch: not initialized")
	}
	if !b.done {
		if err := b.Commit(); err != nil {
			return err
		}
	}
	if b.waits > 0 {
		return fmt.Errorf("Metal batch: already waited")
	}
	b.waits++
	b.ops.submissions.Waits++
	// One span per command buffer: the batch's dispatches are the GPU
	// work, and everything between this span and the next is host cost.
	wait := beginOp("batch.wait")
	err := libmetal.WaitCommandBuffer(ctx, b.buffer)
	wait.end()
	b.ops.hc.complete(b)
	if err != nil {
		return fmt.Errorf("Metal batch: %w", err)
	}
	b.ops.record(b.region, b.layer, b.position, b.buffer)
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
