//go:build darwin && arm64

package main

import (
	"context"
	"fmt"
	"math"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/apple/x/metal/metalx"
)

func (o *metalPinnedOps) IndexerAllowed(qrNorm, current []float32, qWeight, headWeight boundMetalTensor, cache metalmem.Slice[float32], nComp, position, layer int) ([]uint32, error) {
	if len(qrNorm) != 1024 || len(current) != ds4Width || nComp < 0 || nComp > cache.Len()/128 || position < 0 || layer < 2 || layer%2 != 0 {
		return nil, fmt.Errorf("attention indexer: invalid shape")
	}
	allowed := make([]uint32, nComp)
	if nComp <= 512 {
		for i := range allowed {
			allowed[i] = 1
		}
		return allowed, nil
	}
	var q []float32
	var err error
	switch qWeight.typeID {
	case 1:
		q, err = o.MatvecF16(qWeight, qrNorm)
	case 8:
		q, err = o.MatvecQ80(qWeight, qrNorm)
	default:
		err = fmt.Errorf("attention indexer: unsupported query weight type %d", qWeight.typeID)
	}
	if err != nil {
		return nil, err
	}
	weights, err := o.MatvecF16(headWeight, current)
	if err != nil {
		return nil, err
	}
	if len(q) != 64*128 || len(weights) != 64 {
		return nil, fmt.Errorf("attention indexer: invalid projection result")
	}
	for h := range 64 {
		row := q[h*128 : (h+1)*128]
		ropeCompressedTail(row, position, false)
		indexerQAT(row)
		weights[h] *= 1 / float32(math.Sqrt(64*128))
	}
	scores := make([]float32, nComp)
	rows := cache.HostSlice()
	for c := range scores {
		row := rows[c*128 : (c+1)*128]
		var score float32
		for h := range 64 {
			var dot float32
			for i, value := range row {
				dot += value * q[h*128+i]
			}
			if dot > 0 {
				score += dot * weights[h]
			}
		}
		scores[c] = score
	}
	return selectIndexerRows(scores, 512), nil
}

func selectIndexerRows(scores []float32, k int) []uint32 {
	allowed := make([]uint32, len(scores))
	k = min(k, len(scores))
	for range k {
		best := 0
		bestScore := float32(-1e30)
		for i, score := range scores {
			if allowed[i] == 0 && score > bestScore {
				best = i
				bestScore = score
			}
		}
		allowed[best] = 1
	}
	return allowed
}

func ropeCompressedTail(x []float32, position int, inverse bool) {
	start := len(x) - 64
	sign := 1.0
	if inverse {
		sign = -1
	}
	for r := 0; r < 64; r += 2 {
		extrap := float64(position) * math.Pow(160000, -float64(r)/64)
		low := math.Max(0, math.Floor(64*math.Log(65536/(32*2*math.Pi))/(2*math.Log(160000))))
		high := math.Min(63, math.Ceil(64*math.Log(65536/(1*2*math.Pi))/(2*math.Log(160000))))
		y := (float64(r/2) - low) / math.Max(0.001, high-low)
		ramp := 1 - min(1, max(0, y))
		theta := extrap * (0.0625*(1-ramp) + ramp)
		c, s := float32(math.Cos(theta)), float32(sign*math.Sin(theta))
		a, b := x[start+r], x[start+r+1]
		x[start+r] = a*c - b*s
		x[start+r+1] = a*s + b*c
	}
}

func indexerQAT(x []float32) {
	for stride := 1; stride < 128; stride <<= 1 {
		for base := 0; base < 128; base += 2 * stride {
			for i := range stride {
				a, b := x[base+i], x[base+stride+i]
				x[base+i] = a + b
				x[base+stride+i] = a - b
			}
		}
	}
	for i := range x {
		x[i] *= 0.08838834764831845
	}
	for off := 0; off < len(x); off += 32 {
		var amax float32
		for _, value := range x[off : off+32] {
			amax = max(amax, abs32(value))
		}
		amax = max(amax, 7.052966104933725e-38)
		scale := float32(math.Exp2(math.Ceil(math.Log2(float64(amax / 6)))))
		for i := off; i < off+32; i++ {
			x[i] = roundE2M1(max(-6, min(6, x[i]/scale))) * scale
		}
	}
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func roundE2M1(x float32) float32 {
	values := [...]float32{0, .5, 1, 1.5, 2, 3, 4, 6}
	sign := float32(1)
	if x < 0 {
		sign, x = -1, -x
	}
	best := 0
	bestDiff := abs32(x - values[0])
	for i := 1; i < len(values); i++ {
		diff := abs32(x - values[i])
		if diff < bestDiff || diff == bestDiff && i%2 == 0 && best%2 != 0 {
			best, bestDiff = i, diff
		}
	}
	return sign * values[best]
}

type metalPinnedKernels struct {
	GetRowF16          apple.MTLComputePipelineState `metal:"get_row_f16"`
	MatvecF16          apple.MTLComputePipelineState `metal:"matvec_f16"`
	MatvecQ80          apple.MTLComputePipelineState `metal:"matvec_q8_0"`
	MatvecQ80BlockDiag apple.MTLComputePipelineState `metal:"matvec_q8_0_blockdiag"`
	Layer0Attention    apple.MTLComputePipelineState `metal:"layer0_attention_pre"`
	Layer0Post         apple.MTLComputePipelineState `metal:"layer0_attention_post"`
	RouterHash         apple.MTLComputePipelineState `metal:"router_hash"`
	RouterBiased       apple.MTLComputePipelineState `metal:"router_biased"`
	SwiGLU             apple.MTLComputePipelineState `metal:"swiglu"`
	SwiGLUWeights      apple.MTLComputePipelineState `metal:"swiglu_weighted_slots"`
	MatvecIQ2Pair      apple.MTLComputePipelineState `metal:"matvec_iq2_xxs_pair"`
	MatvecIQ2PairF32   apple.MTLComputePipelineState `metal:"matvec_iq2_xxs_pair_f32"`
	MatvecIQ2Selected  apple.MTLComputePipelineState `metal:"matvec_iq2_xxs_pair_selected_f32"`
	QuantQ8K           apple.MTLComputePipelineState `metal:"quant_q8_k"`
	Clamp10            apple.MTLComputePipelineState `metal:"clamp10"`
	MatvecQ2K          apple.MTLComputePipelineState `metal:"matvec_q2_k"`
	MatvecQ2KF32       apple.MTLComputePipelineState `metal:"matvec_q2_k_f32"`
	MatvecQ2KSelected  apple.MTLComputePipelineState `metal:"matvec_q2_k_selected_f32"`
	SumVectors         apple.MTLComputePipelineState `metal:"sum_vectors"`
	RMS                apple.MTLComputePipelineState `metal:"rms_norm"`
	RopeAttention      apple.MTLComputePipelineState `metal:"rope_attention_one"`
	RopeAttentionRaw   apple.MTLComputePipelineState `metal:"rope_attention_raw"`
	RopeAttentionMix   apple.MTLComputePipelineState `metal:"rope_attention_mixed"`
	RopeKVCommit       apple.MTLComputePipelineState `metal:"rope_kv_commit"`
	RopeAttentionWide  apple.MTLComputePipelineState `metal:"rope_attention_mixed_wide"`
	RopeQPrepare       apple.MTLComputePipelineState `metal:"rope_q_prepare"`
	RopeOutUnrotate    apple.MTLComputePipelineState `metal:"rope_out_unrotate"`
	CompressorPool     apple.MTLComputePipelineState `metal:"compressor_pool"`
	CompressorFinish   apple.MTLComputePipelineState `metal:"compressor_finish_attention"`
	IndexerFinish      apple.MTLComputePipelineState `metal:"compressor_finish_indexer"`
}

// mixedWideThreads is the threadgroup width of rope_attention_mixed_wide. It
// must be a power of two, because the kernel's reductions halve the thread
// count each round, and must not exceed the 256-element threadgroup scratch
// array those reductions use.
const mixedWideThreads = 256

func (o *metalPinnedOps) RopeAttentionMixed(q, kv []float32, sinks boundMetalTensor, raw, comp metalmem.Slice[float32], nComp int, allowed []uint32, position int, layer int) ([]float32, []float32, []float32, error) {
	return o.ropeAttentionMixed(q, kv, sinks, raw, comp, nComp, allowed, position, layer, true)
}

// ropeAttentionMixed encodes either the wide two-kernel path or the original
// single-threadgroup kernel. The narrow path is retained solely as the
// reference the wide path is checked against; production always takes wide.
func (o *metalPinnedOps) ropeAttentionMixed(q, kv []float32, sinks boundMetalTensor, raw, comp metalmem.Slice[float32], nComp int, allowed []uint32, position int, layer int, wide bool) ([]float32, []float32, []float32, error) {
	if len(q) != 64*512 || len(kv) != 512 || sinks.typeID != 0 || len(sinks.dimensions) != 1 || sinks.dimensions[0] != 64 ||
		raw.Len() == 0 || raw.Len()%512 != 0 || comp.Len()%512 != 0 || nComp < 0 || nComp > comp.Len()/512 || position < 0 || layer < 2 {
		return nil, nil, nil, fmt.Errorf("mixed attention: invalid shape")
	}
	qbuf, err := metalmem.AllocFromSlice(o.dev, q)
	if err != nil {
		return nil, nil, nil, err
	}
	defer qbuf.Free()
	kvbuf, err := metalmem.AllocFromSlice(o.dev, kv)
	if err != nil {
		return nil, nil, nil, err
	}
	defer kvbuf.Free()
	if nComp == 0 {
		allowed = []uint32{1}
	} else if allowed == nil {
		allowed = make([]uint32, max(1, nComp))
		for i := range allowed {
			allowed[i] = 1
		}
	}
	if len(allowed) < max(1, nComp) {
		return nil, nil, nil, fmt.Errorf("mixed attention: invalid allowed mask")
	}
	allowbuf, err := metalmem.AllocFromSlice(o.dev, allowed)
	if err != nil {
		return nil, nil, nil, err
	}
	defer allowbuf.Free()
	rowCap := raw.Len() / 512
	nRaw := min(position+1, rowCap)
	scores, err := metalmem.Alloc[float32](o.dev, 64*(nRaw+nComp))
	if err != nil {
		return nil, nil, nil, err
	}
	defer scores.Free()
	out, err := metalmem.Alloc[float32](o.dev, 2*len(q)+len(kv))
	if err != nil {
		return nil, nil, nil, err
	}
	defer out.Free()
	cb := o.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	// Two dispatches, not one. The KV preamble writes state every head reads,
	// which the single-threadgroup kernel ordered with a device barrier; that
	// barrier is what pinned all 64 heads to one threadgroup, and so to one GPU
	// core. Encoder ordering enforces the same dependency across threadgroups.
	if !wide {
		enc.SetComputePipelineState(o.kernels.RopeAttentionMix)
		for i, data := range []metalmem.Slice[float32]{qbuf, kvbuf} {
			enc.SetBufferWithOffsetAtIndex(data.Buffer(), data.Offset(), uint(i))
		}
		enc.SetBufferWithOffsetAtIndex(sinks.data.Buffer(), sinks.data.Offset(), 2)
		enc.SetBufferWithOffsetAtIndex(raw.Buffer(), raw.Offset(), 3)
		enc.SetBufferWithOffsetAtIndex(comp.Buffer(), comp.Offset(), 4)
		enc.SetBufferWithOffsetAtIndex(allowbuf.Buffer(), allowbuf.Offset(), 5)
		enc.SetBufferWithOffsetAtIndex(scores.Buffer(), scores.Offset(), 6)
		enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 7)
		libmetal.SetScalar(enc, uint32(position), 8)
		libmetal.SetScalar(enc, uint32(layer), 9)
		libmetal.SetScalar(enc, uint32(rowCap), 10)
		libmetal.SetScalar(enc, uint32(nComp), 11)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(64))
	} else {
		enc.SetComputePipelineState(o.kernels.RopeQPrepare)
		enc.SetBufferWithOffsetAtIndex(qbuf.Buffer(), qbuf.Offset(), 0)
		enc.SetBufferWithOffsetAtIndex(kvbuf.Buffer(), kvbuf.Offset(), 1)
		enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 2)
		libmetal.SetScalar(enc, uint32(position), 3)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(64))

		enc.SetComputePipelineState(o.kernels.RopeKVCommit)
		enc.SetBufferWithOffsetAtIndex(raw.Buffer(), raw.Offset(), 0)
		enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 1)
		libmetal.SetScalar(enc, uint32(position), 2)
		libmetal.SetScalar(enc, uint32(rowCap), 3)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(mixedWideThreads))

		enc.SetComputePipelineState(o.kernels.RopeAttentionWide)
		enc.SetBufferWithOffsetAtIndex(sinks.data.Buffer(), sinks.data.Offset(), 0)
		enc.SetBufferWithOffsetAtIndex(raw.Buffer(), raw.Offset(), 1)
		enc.SetBufferWithOffsetAtIndex(comp.Buffer(), comp.Offset(), 2)
		enc.SetBufferWithOffsetAtIndex(allowbuf.Buffer(), allowbuf.Offset(), 3)
		enc.SetBufferWithOffsetAtIndex(scores.Buffer(), scores.Offset(), 4)
		enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 5)
		libmetal.SetScalar(enc, uint32(position), 6)
		libmetal.SetScalar(enc, uint32(rowCap), 7)
		libmetal.SetScalar(enc, uint32(nComp), 8)
		// One threadgroup per head, so the head axis reaches the whole device.
		enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(64), libmetal.Threads(mixedWideThreads))

		enc.SetComputePipelineState(o.kernels.RopeOutUnrotate)
		enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 0)
		libmetal.SetScalar(enc, uint32(position), 1)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(64))
	}
	enc.EndEncoding()
	_ = layer
	cb.Commit()
	if err := libmetal.WaitCommandBuffer(context.Background(), cb); err != nil {
		return nil, nil, nil, fmt.Errorf("mixed attention: %w", err)
	}
	o.record("rope-attention-mixed", layer, position, cb)
	host := out.HostSlice()
	return append([]float32(nil), host[:len(q)]...), append([]float32(nil), host[len(q):len(q)+len(kv)]...), append([]float32(nil), host[len(q)+len(kv):]...), nil
}

func (o *metalPinnedOps) CompressorAttention(kv, score []float32, ape, norm boundMetalTensor, state *metalCompressedState, position int) ([]float32, bool, error) {
	return o.compressor(kv, score, ape, norm, state, position, false)
}

func (o *metalPinnedOps) CompressorIndexer(kv, score []float32, ape, norm boundMetalTensor, state *metalCompressedState, position int) ([]float32, bool, error) {
	return o.compressor(kv, score, ape, norm, state, position, true)
}

func (o *metalPinnedOps) compressor(kv, score []float32, ape, norm boundMetalTensor, state *metalCompressedState, position int, indexer bool) ([]float32, bool, error) {
	wantHeadDim := 512
	if indexer {
		wantHeadDim = 128
	}
	if state == nil || state.headDim != wantHeadDim || (state.ratio != 4 && state.ratio != 128) ||
		len(kv) != state.width || len(score) != state.width || position < 0 ||
		ape.typeID != 1 || norm.typeID != 0 || len(ape.dimensions) != 2 ||
		ape.dimensions[0] != uint64(state.width) || ape.dimensions[1] != uint64(state.ratio) {
		return nil, false, fmt.Errorf("attention compressor: invalid shape")
	}
	emit := (position+1)%state.ratio == 0
	if emit && state.n >= state.cap {
		return nil, false, fmt.Errorf("attention compressor: cache capacity exceeded")
	}
	kvbuf, err := metalmem.AllocFromSlice(o.dev, kv)
	if err != nil {
		return nil, false, err
	}
	defer kvbuf.Free()
	scorebuf, err := metalmem.AllocFromSlice(o.dev, score)
	if err != nil {
		return nil, false, err
	}
	defer scorebuf.Free()
	cb := o.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(o.kernels.CompressorPool)
	for i, data := range []metalmem.Slice[float32]{kvbuf, scorebuf, state.stateKV, state.stateScore, state.pooled} {
		enc.SetBufferWithOffsetAtIndex(data.Buffer(), data.Offset(), uint(i))
	}
	enc.SetBufferWithOffsetAtIndex(ape.data.Buffer(), ape.data.Offset(), 5)
	libmetal.SetScalar(enc, uint32(state.headDim), 6)
	libmetal.SetScalar(enc, uint32(state.width), 7)
	libmetal.SetScalar(enc, uint32(state.ratio), 8)
	libmetal.SetScalar(enc, uint32(position), 9)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(state.headDim))
	enc.EndEncoding()
	if emit {
		enc = cb.ComputeCommandEncoder()
		if indexer {
			enc.SetComputePipelineState(o.kernels.IndexerFinish)
		} else {
			enc.SetComputePipelineState(o.kernels.CompressorFinish)
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
		enc.EndEncoding()
	}
	if err := o.submitAndWait("compressor", cb); err != nil {
		return nil, false, fmt.Errorf("attention compressor: %w", err)
	}
	if !emit {
		return nil, false, nil
	}
	out := append([]float32(nil), state.cache.HostSlice()[state.n*state.headDim:(state.n+1)*state.headDim]...)
	state.n++
	return out, true, nil
}

func (o *metalPinnedOps) RopeAttentionRaw(q, kv []float32, sinks boundMetalTensor, cache metalmem.Slice[float32], position int) ([]float32, []float32, []float32, error) {
	if len(q) != 64*512 || len(kv) != 512 || sinks.typeID != 0 || len(sinks.dimensions) != 1 || sinks.dimensions[0] != 64 ||
		cache.Len() == 0 || cache.Len()%512 != 0 || position < 0 {
		return nil, nil, nil, fmt.Errorf("raw attention: invalid shape")
	}
	qbuf, err := metalmem.AllocFromSlice(o.dev, q)
	if err != nil {
		return nil, nil, nil, err
	}
	defer qbuf.Free()
	kvbuf, err := metalmem.AllocFromSlice(o.dev, kv)
	if err != nil {
		return nil, nil, nil, err
	}
	defer kvbuf.Free()
	cap := cache.Len() / 512
	scores, err := metalmem.Alloc[float32](o.dev, 64*cap)
	if err != nil {
		return nil, nil, nil, err
	}
	defer scores.Free()
	out, err := metalmem.Alloc[float32](o.dev, 32768+512+32768)
	if err != nil {
		return nil, nil, nil, err
	}
	defer out.Free()
	cb := o.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(o.kernels.RopeAttentionRaw)
	for i, data := range []metalmem.Slice[float32]{qbuf, kvbuf} {
		enc.SetBufferWithOffsetAtIndex(data.Buffer(), data.Offset(), uint(i))
	}
	enc.SetBufferWithOffsetAtIndex(sinks.data.Buffer(), sinks.data.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(cache.Buffer(), cache.Offset(), 3)
	enc.SetBufferWithOffsetAtIndex(scores.Buffer(), scores.Offset(), 4)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 5)
	libmetal.SetScalar(enc, uint32(position), 6)
	libmetal.SetScalar(enc, uint32(cap), 7)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(64))
	enc.EndEncoding()
	if err := o.submitAndWait("rope-attention-raw", cb); err != nil {
		return nil, nil, nil, fmt.Errorf("raw attention: %w", err)
	}
	host := out.HostSlice()
	return append([]float32(nil), host[:32768]...), append([]float32(nil), host[32768:33280]...), append([]float32(nil), host[33280:]...), nil
}

func (o *metalPinnedOps) MatvecQ2KF32(t boundMetalTensor, x []float32) ([]float32, error) {
	if t.typeID != 10 || len(t.dimensions) != 2 || t.dimensions[0]%256 != 0 || uint64(len(x)) != t.dimensions[0] {
		return nil, fmt.Errorf("matvec %q: invalid Q2_K tensor or input", t.name)
	}
	in, err := metalmem.AllocFromSlice(o.dev, x)
	if err != nil {
		return nil, err
	}
	defer in.Free()
	rows := int(t.dimensions[1])
	out, err := metalmem.Alloc[float32](o.dev, rows)
	if err != nil {
		return nil, err
	}
	defer out.Free()
	cb := o.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(o.kernels.MatvecQ2KF32)
	enc.SetBufferWithOffsetAtIndex(t.data.Buffer(), t.data.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 2)
	libmetal.SetScalar(enc, uint32(len(x)), 3)
	libmetal.SetScalar(enc, uint32(rows), 4)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(rows), libmetal.Threads(min(rows, 256)))
	enc.EndEncoding()
	if err := o.submitAndWait("matvec-q2k-f32", cb); err != nil {
		return nil, fmt.Errorf("matvec %q F32: %w", t.name, err)
	}
	return append([]float32(nil), out.HostSlice()...), nil
}

func (o *metalPinnedOps) SumVectors(vectors []float32, count int) ([]float32, error) {
	if count <= 0 || len(vectors) == 0 || len(vectors)%count != 0 {
		return nil, fmt.Errorf("sum vectors: invalid shape")
	}
	in, err := metalmem.AllocFromSlice(o.dev, vectors)
	if err != nil {
		return nil, err
	}
	defer in.Free()
	width := len(vectors) / count
	out, err := metalmem.Alloc[float32](o.dev, width)
	if err != nil {
		return nil, err
	}
	defer out.Free()
	cb := o.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(o.kernels.SumVectors)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 1)
	libmetal.SetScalar(enc, uint32(width), 2)
	libmetal.SetScalar(enc, uint32(count), 3)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(width), libmetal.Threads(min(width, 256)))
	enc.EndEncoding()
	if err := o.submitAndWait("sum-vectors", cb); err != nil {
		return nil, fmt.Errorf("sum vectors: %w", err)
	}
	return append([]float32(nil), out.HostSlice()...), nil
}

func (o *metalPinnedOps) MatvecQ2K(t boundMetalTensor, x []float32) ([]float32, error) {
	if t.typeID != 10 || len(t.dimensions) != 2 || t.dimensions[0]%256 != 0 || uint64(len(x)) != t.dimensions[0] {
		return nil, fmt.Errorf("matvec %q: invalid Q2_K tensor or input", t.name)
	}
	in, err := metalmem.AllocFromSlice(o.dev, x)
	if err != nil {
		return nil, err
	}
	defer in.Free()
	q8, err := metalmem.Alloc[byte](o.dev, len(x)/256*292)
	if err != nil {
		return nil, err
	}
	defer q8.Free()
	rows := int(t.dimensions[1])
	out, err := metalmem.Alloc[float32](o.dev, rows)
	if err != nil {
		return nil, err
	}
	defer out.Free()
	cb := o.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(o.kernels.QuantQ8K)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(q8.Buffer(), q8.Offset(), 1)
	libmetal.SetScalar(enc, uint32(len(x)), 2)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(len(x)/256), libmetal.Threads(1))
	enc.EndEncoding()
	enc = cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(o.kernels.MatvecQ2K)
	enc.SetBufferWithOffsetAtIndex(t.data.Buffer(), t.data.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(q8.Buffer(), q8.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 2)
	libmetal.SetScalar(enc, uint32(len(x)), 3)
	libmetal.SetScalar(enc, uint32(rows), 4)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(rows), libmetal.Threads(min(rows, 256)))
	enc.EndEncoding()
	if err := o.submitAndWait("matvec-q2k", cb); err != nil {
		return nil, fmt.Errorf("matvec %q: %w", t.name, err)
	}
	return append([]float32(nil), out.HostSlice()...), nil
}

func (o *metalPinnedOps) Clamp10(x []float32) ([]float32, error) {
	return o.clamp10(x, false)
}

func (o *metalPinnedOps) ClampGate10(x []float32) ([]float32, error) {
	return o.clamp10(x, true)
}

func (o *metalPinnedOps) clamp10(x []float32, gate bool) ([]float32, error) {
	if len(x) == 0 {
		return nil, fmt.Errorf("clamp: invalid shape")
	}
	in, err := metalmem.AllocFromSlice(o.dev, x)
	if err != nil {
		return nil, err
	}
	defer in.Free()
	out, err := metalmem.Alloc[float32](o.dev, len(x))
	if err != nil {
		return nil, err
	}
	defer out.Free()
	cb := o.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(o.kernels.Clamp10)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 1)
	libmetal.SetScalar(enc, uint32(len(x)), 2)
	mode := uint32(0)
	if gate {
		mode = 1
	}
	libmetal.SetScalar(enc, mode, 3)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(len(x)), libmetal.Threads(min(len(x), 256)))
	enc.EndEncoding()
	if err := o.submitAndWait("clamp10", cb); err != nil {
		return nil, fmt.Errorf("clamp: %w", err)
	}
	return append([]float32(nil), out.HostSlice()...), nil
}

func (o *metalPinnedOps) MatvecIQ2PairQ8K(gate, up boundMetalTensor, x []float32) ([]float32, []float32, error) {
	if gate.typeID != 16 || up.typeID != 16 || len(gate.dimensions) != 2 || len(up.dimensions) != 2 ||
		gate.dimensions[0] != up.dimensions[0] || gate.dimensions[1] != up.dimensions[1] ||
		gate.dimensions[0]%256 != 0 || uint64(len(x)) != gate.dimensions[0] {
		return nil, nil, fmt.Errorf("IQ2_XXS pair: invalid tensor profile")
	}
	in, err := metalmem.AllocFromSlice(o.dev, x)
	if err != nil {
		return nil, nil, err
	}
	defer in.Free()
	q8, err := metalmem.Alloc[byte](o.dev, len(x)/256*292)
	if err != nil {
		return nil, nil, err
	}
	defer q8.Free()
	rows := int(gate.dimensions[1])
	out, err := metalmem.Alloc[float32](o.dev, 2*rows)
	if err != nil {
		return nil, nil, err
	}
	defer out.Free()
	cb := o.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(o.kernels.QuantQ8K)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(q8.Buffer(), q8.Offset(), 1)
	libmetal.SetScalar(enc, uint32(len(x)), 2)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(len(x)/256), libmetal.Threads(1))
	enc.EndEncoding()
	enc = cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(o.kernels.MatvecIQ2Pair)
	enc.SetBufferWithOffsetAtIndex(gate.data.Buffer(), gate.data.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(up.data.Buffer(), up.data.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(q8.Buffer(), q8.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 3)
	libmetal.SetScalar(enc, uint32(len(x)), 4)
	libmetal.SetScalar(enc, uint32(rows), 5)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(rows), libmetal.Threads(32))
	enc.EndEncoding()
	if err := o.submitAndWait("matvec-iq2-pair-q8k", cb); err != nil {
		return nil, nil, fmt.Errorf("IQ2_XXS pair: %w", err)
	}
	host := out.HostSlice()
	return append([]float32(nil), host[:rows]...), append([]float32(nil), host[rows:]...), nil
}

func (o *metalPinnedOps) MatvecIQ2PairF32(gate, up boundMetalTensor, x []float32) ([]float32, []float32, error) {
	if gate.typeID != 16 || up.typeID != 16 || len(gate.dimensions) != 2 || len(up.dimensions) != 2 ||
		gate.dimensions[0] != up.dimensions[0] || gate.dimensions[1] != up.dimensions[1] ||
		gate.dimensions[0]%256 != 0 || uint64(len(x)) != gate.dimensions[0] {
		return nil, nil, fmt.Errorf("IQ2_XXS pair F32: invalid tensor profile")
	}
	in, err := metalmem.AllocFromSlice(o.dev, x)
	if err != nil {
		return nil, nil, err
	}
	defer in.Free()
	rows := int(gate.dimensions[1])
	out, err := metalmem.Alloc[float32](o.dev, 2*rows)
	if err != nil {
		return nil, nil, err
	}
	defer out.Free()
	cb := o.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(o.kernels.MatvecIQ2PairF32)
	enc.SetBufferWithOffsetAtIndex(gate.data.Buffer(), gate.data.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(up.data.Buffer(), up.data.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 3)
	libmetal.SetScalar(enc, uint32(len(x)), 4)
	libmetal.SetScalar(enc, uint32(rows), 5)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(rows), libmetal.Threads(32))
	enc.EndEncoding()
	if err := o.submitAndWait("matvec-iq2-pair-f32", cb); err != nil {
		return nil, nil, fmt.Errorf("IQ2_XXS pair F32: %w", err)
	}
	host := out.HostSlice()
	return append([]float32(nil), host[:rows]...), append([]float32(nil), host[rows:]...), nil
}

func (o *metalPinnedOps) SwiGLU(gate, up []float32) ([]float32, error) {
	return o.swiglu(gate, up, 1)
}

func (o *metalPinnedOps) SwiGLUWeighted(gate, up []float32, weight float32) ([]float32, error) {
	return o.swiglu(gate, up, weight)
}

func (o *metalPinnedOps) swiglu(gate, up []float32, weight float32) ([]float32, error) {
	if len(gate) == 0 || len(gate) != len(up) {
		return nil, fmt.Errorf("SwiGLU: invalid shape")
	}
	g, err := metalmem.AllocFromSlice(o.dev, gate)
	if err != nil {
		return nil, err
	}
	defer g.Free()
	u, err := metalmem.AllocFromSlice(o.dev, up)
	if err != nil {
		return nil, err
	}
	defer u.Free()
	out, err := metalmem.Alloc[float32](o.dev, len(gate))
	if err != nil {
		return nil, err
	}
	defer out.Free()
	cb := o.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(o.kernels.SwiGLU)
	enc.SetBufferWithOffsetAtIndex(g.Buffer(), g.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(u.Buffer(), u.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 2)
	libmetal.SetScalar(enc, uint32(len(gate)), 3)
	libmetal.SetScalar(enc, weight, 4)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(len(gate)), libmetal.Threads(min(len(gate), 256)))
	enc.EndEncoding()
	if err := o.submitAndWait("swiglu", cb); err != nil {
		return nil, fmt.Errorf("SwiGLU: %w", err)
	}
	return append([]float32(nil), out.HostSlice()...), nil
}

func (o *metalPinnedOps) RouterHash(logits []float32, table boundMetalTensor, token uint32) ([]float32, []int32, []float32, error) {
	if len(logits) != 256 || table.typeID != 26 || len(table.dimensions) != 2 || table.dimensions[0] != 6 || uint64(token) >= table.dimensions[1] {
		return nil, nil, nil, fmt.Errorf("router hash: invalid tensor profile")
	}
	in, err := metalmem.AllocFromSlice(o.dev, logits)
	if err != nil {
		return nil, nil, nil, err
	}
	defer in.Free()
	out, err := metalmem.Alloc[float32](o.dev, 256)
	if err != nil {
		return nil, nil, nil, err
	}
	defer out.Free()
	selected, err := metalmem.Alloc[int32](o.dev, 6)
	if err != nil {
		return nil, nil, nil, err
	}
	defer selected.Free()
	weights, err := metalmem.Alloc[float32](o.dev, 6)
	if err != nil {
		return nil, nil, nil, err
	}
	defer weights.Free()
	cb := o.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(o.kernels.RouterHash)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(table.data.Buffer(), table.data.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(selected.Buffer(), selected.Offset(), 3)
	enc.SetBufferWithOffsetAtIndex(weights.Buffer(), weights.Offset(), 4)
	libmetal.SetScalar(enc, token, 5)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(1))
	enc.EndEncoding()
	if err := o.submitAndWait("router-hash", cb); err != nil {
		return nil, nil, nil, fmt.Errorf("router hash: %w", err)
	}
	return append([]float32(nil), out.HostSlice()...), append([]int32(nil), selected.HostSlice()...), append([]float32(nil), weights.HostSlice()...), nil
}

func (o *metalPinnedOps) RouterBiased(logits []float32, bias boundMetalTensor) ([]float32, []int32, []float32, error) {
	if len(logits) != 256 || bias.typeID != 0 || len(bias.dimensions) != 1 || bias.dimensions[0] != 256 {
		return nil, nil, nil, fmt.Errorf("biased router: invalid tensor profile")
	}
	in, err := metalmem.AllocFromSlice(o.dev, logits)
	if err != nil {
		return nil, nil, nil, err
	}
	defer in.Free()
	out, err := metalmem.Alloc[float32](o.dev, 256)
	if err != nil {
		return nil, nil, nil, err
	}
	defer out.Free()
	selected, err := metalmem.Alloc[int32](o.dev, 6)
	if err != nil {
		return nil, nil, nil, err
	}
	defer selected.Free()
	weights, err := metalmem.Alloc[float32](o.dev, 6)
	if err != nil {
		return nil, nil, nil, err
	}
	defer weights.Free()
	cb := o.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(o.kernels.RouterBiased)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(bias.data.Buffer(), bias.data.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(selected.Buffer(), selected.Offset(), 3)
	enc.SetBufferWithOffsetAtIndex(weights.Buffer(), weights.Offset(), 4)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(1))
	enc.EndEncoding()
	if err := o.submitAndWait("router-biased", cb); err != nil {
		return nil, nil, nil, fmt.Errorf("biased router: %w", err)
	}
	return append([]float32(nil), out.HostSlice()...), append([]int32(nil), selected.HostSlice()...), append([]float32(nil), weights.HostSlice()...), nil
}

func (o *metalPinnedOps) Layer0AttentionPost(embedding boundMetalTensor, block []float32, hcFn, hcScale, hcBase boundMetalTensor, token uint32) ([]float32, error) {
	if len(block) != 4096 || embedding.typeID != 1 || len(embedding.dimensions) != 2 || token >= uint32(embedding.dimensions[1]) ||
		hcFn.typeID != 1 || hcScale.typeID != 0 || hcBase.typeID != 0 {
		return nil, fmt.Errorf("layer 0 attention post: invalid tensor profile")
	}
	in, err := metalmem.AllocFromSlice(o.dev, block)
	if err != nil {
		return nil, err
	}
	defer in.Free()
	out, err := metalmem.Alloc[float32](o.dev, 4*4096)
	if err != nil {
		return nil, err
	}
	defer out.Free()
	cb := o.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(o.kernels.Layer0Post)
	for i, tensor := range []boundMetalTensor{embedding, hcFn, hcScale, hcBase} {
		enc.SetBufferWithOffsetAtIndex(tensor.data.Buffer(), tensor.data.Offset(), uint(i))
	}
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 4)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 5)
	libmetal.SetScalar(enc, token, 6)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(1))
	enc.EndEncoding()
	if err := o.submitAndWait("layer0-attention-post", cb); err != nil {
		return nil, fmt.Errorf("layer 0 attention post: %w", err)
	}
	return append([]float32(nil), out.HostSlice()...), nil
}

func (o *metalPinnedOps) RopeAttentionOne(q, kv []float32, sinks boundMetalTensor, position uint32) ([]float32, []float32, []float32, error) {
	if len(q) != 64*512 || len(kv) != 512 || sinks.typeID != 0 || len(sinks.dimensions) != 1 || sinks.dimensions[0] != 64 {
		return nil, nil, nil, fmt.Errorf("rope attention: invalid shape")
	}
	qbuf, err := metalmem.AllocFromSlice(o.dev, q)
	if err != nil {
		return nil, nil, nil, err
	}
	defer qbuf.Free()
	kvbuf, err := metalmem.AllocFromSlice(o.dev, kv)
	if err != nil {
		return nil, nil, nil, err
	}
	defer kvbuf.Free()
	out, err := metalmem.Alloc[float32](o.dev, 2*len(q)+len(kv))
	if err != nil {
		return nil, nil, nil, err
	}
	defer out.Free()
	cb := o.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(o.kernels.RopeAttention)
	enc.SetBufferWithOffsetAtIndex(qbuf.Buffer(), qbuf.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(kvbuf.Buffer(), kvbuf.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(sinks.data.Buffer(), sinks.data.Offset(), 2)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 3)
	libmetal.SetScalar(enc, position, 4)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(64), libmetal.Threads(64))
	enc.EndEncoding()
	if err := o.submitAndWait("rope-attention-one", cb); err != nil {
		return nil, nil, nil, fmt.Errorf("rope attention: %w", err)
	}
	host := out.HostSlice()
	qcur := append([]float32(nil), host[:len(q)]...)
	kvcur := append([]float32(nil), host[len(q):len(q)+len(kv)]...)
	heads := append([]float32(nil), host[len(q)+len(kv):]...)
	return qcur, kvcur, heads, nil
}

func (o *metalPinnedOps) RMS(x []float32, weight boundMetalTensor, groups int, weighted bool) ([]float32, error) {
	if len(x) == 0 || groups <= 0 || len(x)%groups != 0 || weight.typeID != 0 || len(weight.dimensions) == 0 ||
		(weighted && uint64(len(x)/groups) != weight.dimensions[0]) {
		return nil, fmt.Errorf("RMS norm: invalid shape")
	}
	in, err := metalmem.AllocFromSlice(o.dev, x)
	if err != nil {
		return nil, err
	}
	defer in.Free()
	out, err := metalmem.Alloc[float32](o.dev, len(x))
	if err != nil {
		return nil, err
	}
	defer out.Free()
	cb := o.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(o.kernels.RMS)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(weight.data.Buffer(), weight.data.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 2)
	libmetal.SetScalar(enc, uint32(len(x)/groups), 3)
	libmetal.SetScalar(enc, uint32(groups), 4)
	if weighted {
		libmetal.SetScalar(enc, uint32(1), 5)
	} else {
		libmetal.SetScalar(enc, uint32(0), 5)
	}
	enc.SetThreadgroupMemoryLengthAtIndex(32*4, 0)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(
		apple.MTLSize{Width: uint(groups), Height: 1, Depth: 1},
		apple.MTLSize{Width: 256, Height: 1, Depth: 1},
	)
	enc.EndEncoding()
	if err := o.submitAndWait("rms", cb); err != nil {
		return nil, fmt.Errorf("RMS norm: %w", err)
	}
	return append([]float32(nil), out.HostSlice()...), nil
}

func (o *metalPinnedOps) Layer0AttentionPre(embedding, hcFn, hcScale, hcBase, norm boundMetalTensor, token uint32) ([]float32, []float32, error) {
	if embedding.typeID != 1 || hcFn.typeID != 1 || hcScale.typeID != 0 || hcBase.typeID != 0 || norm.typeID != 0 ||
		len(embedding.dimensions) != 2 || embedding.dimensions[0] != 4096 || token >= uint32(embedding.dimensions[1]) ||
		len(hcFn.dimensions) != 2 || hcFn.dimensions[0] != 16384 || hcFn.dimensions[1] != 24 {
		return nil, nil, fmt.Errorf("layer 0 attention pre: invalid tensor profile")
	}
	out, err := metalmem.Alloc[float32](o.dev, 8192)
	if err != nil {
		return nil, nil, err
	}
	defer out.Free()
	cb := o.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(o.kernels.Layer0Attention)
	for i, tensor := range []boundMetalTensor{embedding, hcFn, hcScale, hcBase, norm} {
		enc.SetBufferWithOffsetAtIndex(tensor.data.Buffer(), tensor.data.Offset(), uint(i))
	}
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 5)
	libmetal.SetScalar(enc, token, 6)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(1))
	enc.EndEncoding()
	if err := o.submitAndWait("layer0-attention-pre", cb); err != nil {
		return nil, nil, fmt.Errorf("layer 0 attention pre: %w", err)
	}
	host := out.HostSlice()
	return append([]float32(nil), host[:4096]...), append([]float32(nil), host[4096:]...), nil
}

type metalPinnedOps struct {
	dev     *libmetal.Device
	library apple.MTLLibrary
	kernels metalPinnedKernels
	hc      *metalHC
	head    *metalHeadProjection

	matmulF16    *metalMatmulPreservedF16
	matmulQ80    *metalMatmulPreservedQ80
	matmulQ2K    *metalMatmulPreservedQ2K
	matmulExpert *metalMatmulPreservedExpert

	// regions, when installed, collects each command buffer's GPU-time
	// span as it completes. Nil in production.
	regions *metalRegionLog

	// submissions counts command buffer traffic for the life of these ops.
	// Always on: it costs four increments per batch and it is the only way to
	// tell a submission-structure change from a rearrangement.
	submissions metalSubmissions
}

func newMetalPinnedOps(dev *libmetal.Device) (*metalPinnedOps, error) {
	if dev == nil {
		return nil, fmt.Errorf("Metal ops: missing device")
	}
	o := &metalPinnedOps{dev: dev}
	lib, err := dev.CompileLibrary(metalPinnedMSL)
	if err != nil {
		return nil, err
	}
	o.library = lib
	if err := metalx.Bind(dev, lib, &o.kernels); err != nil {
		o.Close()
		return nil, err
	}
	o.hc, err = newMetalHC(dev)
	if err != nil {
		o.Close()
		return nil, err
	}
	o.head, err = newMetalHeadProjection(dev)
	if err != nil {
		o.Close()
		return nil, err
	}
	o.head.ops = o
	return o, nil
}

func (o *metalPinnedOps) GetRowF16(t boundMetalTensor, row uint32) ([]float32, error) {
	if t.typeID != 1 || len(t.dimensions) != 2 || row >= uint32(t.dimensions[1]) {
		return nil, fmt.Errorf("get row %q: invalid F16 tensor or row", t.name)
	}
	cols := int(t.dimensions[0])
	out, err := metalmem.Alloc[float32](o.dev, cols)
	if err != nil {
		return nil, err
	}
	defer out.Free()
	if err := o.dispatch("get-row-f16", o.kernels.GetRowF16, t.data, out, row, uint32(cols), cols); err != nil {
		return nil, err
	}
	return append([]float32(nil), out.HostSlice()...), nil
}

func (o *metalPinnedOps) MatvecQ80(t boundMetalTensor, x []float32) ([]float32, error) {
	if t.typeID != 8 || len(t.dimensions) != 2 || t.dimensions[0]%32 != 0 || uint64(len(x)) != t.dimensions[0] {
		return nil, fmt.Errorf("matvec %q: invalid Q8_0 tensor or input", t.name)
	}
	in, err := metalmem.AllocFromSlice(o.dev, x)
	if err != nil {
		return nil, err
	}
	defer in.Free()
	rows := int(t.dimensions[1])
	out, err := metalmem.Alloc[float32](o.dev, rows)
	if err != nil {
		return nil, err
	}
	defer out.Free()
	cb := o.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(o.kernels.MatvecQ80)
	enc.SetBufferWithOffsetAtIndex(t.data.Buffer(), t.data.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 2)
	libmetal.SetScalar(enc, uint32(len(x)), 3)
	libmetal.SetScalar(enc, uint32(rows), 4)
	enc.SetThreadgroupMemoryLengthAtIndex(2*32*4, 0)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(
		apple.MTLSize{Width: uint((rows + 1) / 2), Height: 1, Depth: 1},
		apple.MTLSize{Width: 32, Height: 4, Depth: 1},
	)
	enc.EndEncoding()
	if err := o.submitAndWait("matvec-q80", cb); err != nil {
		return nil, fmt.Errorf("matvec %q: %w", t.name, err)
	}
	return append([]float32(nil), out.HostSlice()...), nil
}

// MatvecIQ2Selected runs one token's routed gate and up projections and
// returns their outputs, slot-major.
//
// It is the per-token reference a batched selected kernel has to reproduce
// bit for bit, so it does not reimplement the dispatch: it drives the same
// metalBatch encoder the runtime uses at metal_runtime_darwin.go:825. A
// reference that hand-copies the bindings can drift from the path it claims
// to stand for, and a bit-identity test against a drifted reference proves
// nothing about production.
func (o *metalPinnedOps) MatvecIQ2Selected(gate, up boundMetalTensor, expertIDs []int32, x []float32) (gateOut, upOut []float32, err error) {
	if len(gate.dimensions) != 3 {
		return nil, nil, fmt.Errorf("matvec %q: IQ2_XXS selected tensor is not 3-D", gate.name)
	}
	rows := int(gate.dimensions[1])
	ids, err := metalmem.AllocFromSlice(o.dev, expertIDs)
	if err != nil {
		return nil, nil, err
	}
	defer ids.Free()
	in, err := metalmem.AllocFromSlice(o.dev, x)
	if err != nil {
		return nil, nil, err
	}
	defer in.Free()
	gateBuf, err := metalmem.Alloc[float32](o.dev, rows*len(expertIDs))
	if err != nil {
		return nil, nil, err
	}
	defer gateBuf.Free()
	upBuf, err := metalmem.Alloc[float32](o.dev, rows*len(expertIDs))
	if err != nil {
		return nil, nil, err
	}
	defer upBuf.Free()
	batch, err := newMetalBatch(o)
	if err != nil {
		return nil, nil, err
	}
	if err := batch.EncodeMatvecIQ2Selected(gate, up, ids, in, gateBuf, upBuf); err != nil {
		return nil, nil, err
	}
	if err := batch.Wait(context.Background()); err != nil {
		return nil, nil, fmt.Errorf("matvec %q: %w", gate.name, err)
	}
	return append([]float32(nil), gateBuf.HostSlice()...), append([]float32(nil), upBuf.HostSlice()...), nil
}

// MatvecQ2KSelected runs one token's routed down projection over its already
// slot-major input and returns the six slot outputs. Like MatvecIQ2Selected it
// drives the production encoder rather than restating its dispatch.
func (o *metalPinnedOps) MatvecQ2KSelected(t boundMetalTensor, expertIDs []int32, x []float32) ([]float32, error) {
	if len(t.dimensions) != 3 {
		return nil, fmt.Errorf("matvec %q: Q2_K selected tensor is not 3-D", t.name)
	}
	rows := int(t.dimensions[1])
	ids, err := metalmem.AllocFromSlice(o.dev, expertIDs)
	if err != nil {
		return nil, err
	}
	defer ids.Free()
	in, err := metalmem.AllocFromSlice(o.dev, x)
	if err != nil {
		return nil, err
	}
	defer in.Free()
	out, err := metalmem.Alloc[float32](o.dev, rows*len(expertIDs))
	if err != nil {
		return nil, err
	}
	defer out.Free()
	batch, err := newMetalBatch(o)
	if err != nil {
		return nil, err
	}
	if err := batch.EncodeMatvecQ2KSelected(t, ids, in, out); err != nil {
		return nil, err
	}
	if err := batch.Wait(context.Background()); err != nil {
		return nil, fmt.Errorf("matvec %q: %w", t.name, err)
	}
	return append([]float32(nil), out.HostSlice()...), nil
}

func (o *metalPinnedOps) MatvecF16(t boundMetalTensor, x []float32) ([]float32, error) {
	if t.typeID != 1 || len(t.dimensions) != 2 || uint64(len(x)) != t.dimensions[0] {
		return nil, fmt.Errorf("matvec %q: invalid F16 tensor or input", t.name)
	}
	if t.dimensions[0] == 16384 && t.dimensions[1] == 4 {
		return o.head.matvec(t, x)
	}
	in, err := metalmem.AllocFromSlice(o.dev, x)
	if err != nil {
		return nil, err
	}
	defer in.Free()
	rows := int(t.dimensions[1])
	out, err := metalmem.Alloc[float32](o.dev, rows)
	if err != nil {
		return nil, err
	}
	defer out.Free()
	cb := o.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(o.kernels.MatvecF16)
	enc.SetBufferWithOffsetAtIndex(t.data.Buffer(), t.data.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 1)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 2)
	libmetal.SetScalar(enc, uint32(len(x)), 3)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(rows), libmetal.Threads(min(rows, 256)))
	enc.EndEncoding()
	if err := o.submitAndWait("matvec-f16", cb); err != nil {
		return nil, fmt.Errorf("matvec %q: %w", t.name, err)
	}
	return append([]float32(nil), out.HostSlice()...), nil
}

func (o *metalPinnedOps) dispatch(region string, pipeline apple.MTLComputePipelineState, in metalmem.Slice[byte], out metalmem.Slice[float32], a, b uint32, threads int) error {
	cb := o.commandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(pipeline)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 0)
	enc.SetBufferWithOffsetAtIndex(out.Buffer(), out.Offset(), 1)
	libmetal.SetScalar(enc, a, 2)
	libmetal.SetScalar(enc, b, 3)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(threads), libmetal.Threads(min(threads, 256)))
	enc.EndEncoding()
	if err := o.submitAndWait(region, cb); err != nil {
		return fmt.Errorf("Metal dispatch: %w", err)
	}
	return nil
}

func (o *metalPinnedOps) Close() {
	if o == nil {
		return
	}
	if o.head != nil {
		o.head.Close()
		o.head = nil
	}
	if o.hc != nil {
		o.hc.Close()
		o.hc = nil
	}
	if o.matmulF16 != nil {
		o.matmulF16.Close()
		o.matmulF16 = nil
	}
	if o.matmulQ80 != nil {
		o.matmulQ80.Close()
		o.matmulQ80 = nil
	}
	if o.matmulQ2K != nil {
		o.matmulQ2K.Close()
		o.matmulQ2K = nil
	}
	if o.matmulExpert != nil {
		o.matmulExpert.Close()
		o.matmulExpert = nil
	}
	for _, pipeline := range []apple.MTLComputePipelineState{o.kernels.GetRowF16, o.kernels.MatvecF16, o.kernels.MatvecQ80, o.kernels.MatvecQ80BlockDiag, o.kernels.MatvecIQ2Pair, o.kernels.MatvecIQ2PairF32, o.kernels.MatvecIQ2Selected, o.kernels.QuantQ8K, o.kernels.MatvecQ2K, o.kernels.MatvecQ2KF32, o.kernels.MatvecQ2KSelected, o.kernels.Clamp10, o.kernels.SumVectors, o.kernels.Layer0Attention, o.kernels.Layer0Post, o.kernels.RouterHash, o.kernels.RouterBiased, o.kernels.SwiGLU, o.kernels.SwiGLUWeights, o.kernels.RMS, o.kernels.RopeAttention, o.kernels.RopeAttentionMix, o.kernels.RopeKVCommit, o.kernels.RopeAttentionWide, o.kernels.RopeQPrepare, o.kernels.RopeOutUnrotate} {
		if pipeline != nil && pipeline.GetID() != 0 {
			objectivec.ObjectFromID(pipeline.GetID()).Release()
		}
	}
	if o.library != nil && o.library.GetID() != 0 {
		objectivec.ObjectFromID(o.library.GetID()).Release()
	}
	o.library = nil
}

func (o *metalPinnedOps) ensureMatmulF16() (*metalMatmulPreservedF16, error) {
	if o == nil {
		return nil, fmt.Errorf("Metal ops: missing ops")
	}
	if o.matmulF16 != nil {
		return o.matmulF16, nil
	}
	return newMetalMatmulPreservedF16(o)
}

func (o *metalPinnedOps) ensureMatmulQ80() (*metalMatmulPreservedQ80, error) {
	if o == nil {
		return nil, fmt.Errorf("Metal ops: missing ops")
	}
	if o.matmulQ80 != nil {
		return o.matmulQ80, nil
	}
	return newMetalMatmulPreservedQ80(o)
}

func (o *metalPinnedOps) ensureMatmulQ2K() (*metalMatmulPreservedQ2K, error) {
	if o == nil {
		return nil, fmt.Errorf("Metal ops: missing ops")
	}
	if o.matmulQ2K != nil {
		return o.matmulQ2K, nil
	}
	return newMetalMatmulPreservedQ2K(o)
}

func (o *metalPinnedOps) ensureMatmulExpert() (*metalMatmulPreservedExpert, error) {
	if o == nil {
		return nil, fmt.Errorf("Metal ops: missing ops")
	}
	if o.matmulExpert != nil {
		return o.matmulExpert, nil
	}
	return newMetalMatmulPreservedExpert(o)
}

const metalPinnedMSL = `
#include <metal_stdlib>
using namespace metal;

` + metalIQ2CodebooksMSL + `
kernel void get_row_f16(
    device const half *weights [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &row [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i < cols) out[i] = float(weights[uint64_t(row)*cols+i]);
}

kernel void clamp10(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                   constant uint &n [[buffer(2)]], constant uint &gate [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
  if(i<n)out[i]=gate?min(x[i],10.0f):clamp(x[i],-10.0f,10.0f);
}

kernel void sum_vectors(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                        constant uint &width [[buffer(2)]], constant uint &count [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
  if(i<width){float v=0.0f;for(uint row=0;row<count;row++)v+=x[uint64_t(row)*width+i];out[i]=v;}
}

kernel void matvec_f16(device const half *weights [[buffer(0)]], device const float *x [[buffer(1)]],
                       device float *out [[buffer(2)]], constant uint &cols [[buffer(3)]],
                       uint row [[thread_position_in_grid]]) {
  float sum=0.0f;device const half *w=weights+uint64_t(row)*cols;
  for(uint i=0;i<cols;i++)sum+=float(w[i])*x[i];
  out[row]=sum;
}

kernel void router_hash(device const float *logits [[buffer(0)]], device const int *table [[buffer(1)]],
                        device float *out [[buffer(2)]], device int *selected [[buffer(3)]],
                        device float *weights [[buffer(4)]], constant uint &token [[buffer(5)]]) {
  float sum=0.0f;
  for(uint i=0;i<256;i++){
    float x=logits[i];
    float sp=x>20.0f?x:log(1.0f+exp(x));
    out[i]=sqrt(sp);
  }
  for(uint i=0;i<6;i++){int id=table[uint64_t(token)*6+i];selected[i]=id;sum+=out[id];}
  sum=max(sum,6.103515625e-5f);
  for(uint i=0;i<6;i++)weights[i]=out[selected[i]]/sum*1.5f;
}

kernel void router_biased(device const float *logits [[buffer(0)]], device const float *bias [[buffer(1)]],
                          device float *out [[buffer(2)]], device int *selected [[buffer(3)]],
                          device float *weights [[buffer(4)]]) {
  float selection[256];
  for(uint i=0;i<256;i++){float x=logits[i];float sp=x>20.0f?x:log(1.0f+exp(x));out[i]=sqrt(sp);selection[i]=out[i]+bias[i];}
  for(uint k=0;k<6;k++){int best=-1;for(uint i=0;i<256;i++){bool used=false;for(uint j=0;j<k;j++)used=used||selected[j]==int(i);if(!used&&(best<0||selection[i]>selection[best]))best=int(i);}selected[k]=best;}
  float sum=0.0f;for(uint k=0;k<6;k++)sum+=out[selected[k]];sum=max(sum,6.103515625e-5f);
  for(uint k=0;k<6;k++)weights[k]=out[selected[k]]/sum*1.5f;
}

kernel void swiglu(device const float *gate [[buffer(0)]], device const float *up [[buffer(1)]],
                   device float *out [[buffer(2)]], constant uint &n [[buffer(3)]],
                   constant float &weight [[buffer(4)]],
                   uint i [[thread_position_in_grid]]) {
  if(i<n){float x=min(gate[i],10.0f);float y=clamp(up[i],-10.0f,10.0f);out[i]=(x/(1.0f+exp(-x)))*y*weight;}
}

kernel void swiglu_weighted_slots(
    device const float *gate [[buffer(0)]], device const float *up [[buffer(1)]],
    device const float *weights [[buffer(2)]], device float *out [[buffer(3)]],
    constant uint &width [[buffer(4)]], constant uint &slots [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
  uint n=width*slots;
  if(i<n){float x=min(gate[i],10.0f);float y=clamp(up[i],-10.0f,10.0f);out[i]=(x/(1.0f+exp(-x)))*y*weights[i/width];}
}

kernel void matvec_q8_0(
    device const uchar *weights [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr uint rows_per_group = 2;
  constexpr uint simdgroups = 4;
  const uint blocks = cols/32;
  const uint first_row = group.x*rows_per_group;
  const ushort block_part = lane/4;
  const ushort element_part = lane%4;
  const uint first_block = simdgroup*8+block_part;
  float sums[rows_per_group] = {0.0f};
  for (uint b=first_block; b<blocks; b+=simdgroups*8) {
    float xv[8];
    for (ushort i=0; i<8; i++) xv[i]=x[b*32+element_part*8+i];
    for (ushort r=0; r<rows_per_group; r++) {
      if (first_row+r>=rows) continue;
      const device uchar *p=weights+uint64_t(first_row+r)*blocks*34+uint64_t(b)*34;
      float dot=0.0f;
      for (ushort i=0; i<8; i++) dot+=float(as_type<char>(p[2+element_part*8+i]))*xv[i];
      sums[r]+=dot*float(*reinterpret_cast<device const half *>(p));
    }
  }
  for (ushort r=0; r<rows_per_group; r++) {
    if(simdgroup==0)scratch[r*32+lane]=0.0f;
    sums[r]=simd_sum(sums[r]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (ushort r=0; r<rows_per_group; r++) if (lane==0) scratch[r*32+simdgroup]=sums[r];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (ushort r=0; r<rows_per_group; r++) {
    float total=simd_sum(scratch[r*32+lane]);
    if (lane==0 && simdgroup==0 && first_row+r<rows) out[first_row+r]=total;
  }
}

// matvec_q8_0_blockdiag runs the block-diagonal form of matvec_q8_0: blocks
// independent (rows x cols) matvecs over disjoint slices of x and out, laid out
// contiguously in all three buffers. The attention output projection is eight of
// these -- one per group of eight heads -- and the runtime used to encode them as
// eight dispatches of 1024 rows each, which on a 48-SM part is well under one
// wave apiece.
//
// The per-output arithmetic is untouched: blocks, first_block, the strided
// accumulation loop and the two-stage simd_sum below are character for character
// what matvec_q8_0 does, and only the three base pointers move. Each output
// element is therefore the same sequence of float operations in the same order,
// so the fused dispatch is bit-identical to the loop it replaces rather than
// merely close. metal_matvec_blockdiag_darwin_test.go asserts that against
// Float32bits, and any future edit here has to keep the two bodies in step.
kernel void matvec_q8_0_blockdiag(
    device const uchar *weights [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr uint rows_per_group = 2;
  constexpr uint simdgroups = 4;
  const uint blocks = cols/32;
  const uint diag = group.y;
  device const uchar *w = weights+uint64_t(diag)*uint64_t(rows)*blocks*34;
  device const float *xb = x+uint64_t(diag)*cols;
  device float *ob = out+uint64_t(diag)*rows;
  const uint first_row = group.x*rows_per_group;
  const ushort block_part = lane/4;
  const ushort element_part = lane%4;
  const uint first_block = simdgroup*8+block_part;
  float sums[rows_per_group] = {0.0f};
  for (uint b=first_block; b<blocks; b+=simdgroups*8) {
    float xv[8];
    for (ushort i=0; i<8; i++) xv[i]=xb[b*32+element_part*8+i];
    for (ushort r=0; r<rows_per_group; r++) {
      if (first_row+r>=rows) continue;
      const device uchar *p=w+uint64_t(first_row+r)*blocks*34+uint64_t(b)*34;
      float dot=0.0f;
      for (ushort i=0; i<8; i++) dot+=float(as_type<char>(p[2+element_part*8+i]))*xv[i];
      sums[r]+=dot*float(*reinterpret_cast<device const half *>(p));
    }
  }
  for (ushort r=0; r<rows_per_group; r++) {
    if(simdgroup==0)scratch[r*32+lane]=0.0f;
    sums[r]=simd_sum(sums[r]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (ushort r=0; r<rows_per_group; r++) if (lane==0) scratch[r*32+simdgroup]=sums[r];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (ushort r=0; r<rows_per_group; r++) {
    float total=simd_sum(scratch[r*32+lane]);
    if (lane==0 && simdgroup==0 && first_row+r<rows) ob[first_row+r]=total;
  }
}

kernel void quant_q8_k(device const float *x [[buffer(0)]], device uchar *out [[buffer(1)]],
                       constant uint &cols [[buffer(2)]], uint block [[thread_position_in_grid]]) {
  if(block>=cols/256)return;
  device uchar *p=out+uint64_t(block)*292;device char *q=reinterpret_cast<device char *>(p+4);
  float signed_max=0.0f,amax=0.0f;
  for(uint i=0;i<256;i++){float v=x[block*256+i],a=abs(v);if(a>amax){amax=a;signed_max=v;}}
  if(amax==0.0f){*reinterpret_cast<device float *>(p)=0.0f;for(uint i=0;i<288;i++)p[4+i]=0;return;}
  float iscale=-127.0f/signed_max;
  for(uint i=0;i<256;i++){int v=int(rint(iscale*x[block*256+i]));v=clamp(v,-128,127);q[i]=char(v);}
  device short *s=reinterpret_cast<device short *>(p+260);
  for(uint j=0;j<16;j++){int v=0;for(uint i=0;i<16;i++)v+=int(q[j*16+i]);s[j]=short(v);}
  *reinterpret_cast<device float *>(p)=1.0f/iscale;
}

kernel void matvec_iq2_xxs_pair(
    device const uchar *gate [[buffer(0)]], device const uchar *up [[buffer(1)]],
    device const uchar *q8 [[buffer(2)]], device float *out [[buffer(3)]],
    constant uint &cols [[buffer(4)]], constant uint &rows [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
  if(row>=rows)return;
  const uint blocks=cols/256,parts=cols/32;
  float gate_sum=0.0f,up_sum=0.0f;
  for(uint part=lane;part<parts;part+=32){
    uint qblock=part/8,qpart=part%8;const device uchar *qy=q8+uint64_t(qblock)*292;
    const device char *xv=reinterpret_cast<device const char *>(qy+4)+qpart*32;
    uint block=part/8,sub=part%8;
    const device uchar *pg=gate+uint64_t(row)*blocks*66+uint64_t(block)*66;
    const device uchar *pu=up+uint64_t(row)*blocks*66+uint64_t(block)*66;
    const device ushort *qg=reinterpret_cast<device const ushort *>(pg+2)+4*sub;
    const device ushort *qu=reinterpret_cast<device const ushort *>(pu+2)+4*sub;
    device const uchar *ag=reinterpret_cast<device const uchar *>(qg);
    device const uchar *au=reinterpret_cast<device const uchar *>(qu);
    uint sg=uint(qg[2])|(uint(qg[3])<<16),su=uint(qu[2])|(uint(qu[3])<<16);
    float dg=float(*reinterpret_cast<device const half *>(pg))*(0.5f+float(sg>>28));
    float du=float(*reinterpret_cast<device const half *>(pu))*(0.5f+float(su>>28));
    float vg=0.0f,vu=0.0f;
    for(ushort l=0;l<4;l++){
      constant const uchar *gg=reinterpret_cast<constant const uchar *>(ds4_metal_iq2xxs_grid+ag[l]);
      constant const uchar *gu=reinterpret_cast<constant const uchar *>(ds4_metal_iq2xxs_grid+au[l]);
      uchar signg=ds4_metal_ksigns_iq2xs[(sg>>(7*l))&127],signu=ds4_metal_ksigns_iq2xs[(su>>(7*l))&127];
      for(ushort j=0;j<8;j++){
        float v=float(xv[8*l+j]);uchar mask=uchar(1u<<j);
        vg+=v*float(gg[j])*((signg&mask)?-1.0f:1.0f);
        vu+=v*float(gu[j])*((signu&mask)?-1.0f:1.0f);
      }
    }
    float qscale=*reinterpret_cast<device const float *>(qy);
    gate_sum+=qscale*dg*vg;up_sum+=qscale*du*vu;
  }
  gate_sum=simd_sum(gate_sum)*0.25f;up_sum=simd_sum(up_sum)*0.25f;
  if(lane==0){out[row]=gate_sum;out[rows+row]=up_sum;}
}

kernel void matvec_iq2_xxs_pair_f32(
    device const uchar *gate [[buffer(0)]], device const uchar *up [[buffer(1)]],
    device const float *x [[buffer(2)]], device float *out [[buffer(3)]],
    constant uint &cols [[buffer(4)]], constant uint &rows [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
  if(row>=rows)return;const uint blocks=cols/256,parts=cols/32;float gate_sum=0.0f,up_sum=0.0f;
  for(uint part=lane;part<parts;part+=32){
    float xv[32];for(ushort i=0;i<32;i++)xv[i]=x[part*32+i];uint block=part/8,sub=part%8;
    const device uchar *pg=gate+uint64_t(row)*blocks*66+uint64_t(block)*66;
    const device uchar *pu=up+uint64_t(row)*blocks*66+uint64_t(block)*66;
    const device ushort *qg=reinterpret_cast<device const ushort *>(pg+2)+4*sub;
    const device ushort *qu=reinterpret_cast<device const ushort *>(pu+2)+4*sub;
    device const uchar *ag=reinterpret_cast<device const uchar *>(qg);device const uchar *au=reinterpret_cast<device const uchar *>(qu);
    uint sg=uint(qg[2])|(uint(qg[3])<<16),su=uint(qu[2])|(uint(qu[3])<<16);float vg=0.0f,vu=0.0f;
    for(ushort l=0;l<4;l++){
      constant const uchar *gg=reinterpret_cast<constant const uchar *>(ds4_metal_iq2xxs_grid+ag[l]);
      constant const uchar *gu=reinterpret_cast<constant const uchar *>(ds4_metal_iq2xxs_grid+au[l]);
      uchar signg=ds4_metal_ksigns_iq2xs[(sg>>(7*l))&127],signu=ds4_metal_ksigns_iq2xs[(su>>(7*l))&127];
      for(ushort j=0;j<8;j++){float v=xv[8*l+j];uchar mask=uchar(1u<<j);vg+=v*float(gg[j])*((signg&mask)?-1.0f:1.0f);vu+=v*float(gu[j])*((signu&mask)?-1.0f:1.0f);}
    }
    gate_sum+=float(*reinterpret_cast<device const half *>(pg))*(0.5f+float(sg>>28))*vg;
    up_sum+=float(*reinterpret_cast<device const half *>(pu))*(0.5f+float(su>>28))*vu;
  }
  gate_sum=simd_sum(gate_sum)*0.25f;up_sum=simd_sum(up_sum)*0.25f;if(lane==0){out[row]=gate_sum;out[rows+row]=up_sum;}
}

kernel void matvec_iq2_xxs_pair_selected_f32(
    device const uchar *gate [[buffer(0)]], device const uchar *up [[buffer(1)]],
    device const int *expert_ids [[buffer(2)]], device const float *x [[buffer(3)]],
    device float *gate_out [[buffer(4)]], device float *up_out [[buffer(5)]],
    constant uint &cols [[buffer(6)]], constant uint &rows [[buffer(7)]],
    constant uint &experts [[buffer(8)]], constant uint &slots [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
  uint row=group.x,slot=group.y;if(row>=rows||slot>=slots)return;
  int expert=expert_ids[slot];if(expert<0||uint(expert)>=experts)return;
  const uint blocks=cols/256,parts=cols/32;
  const uint64_t expert_stride=uint64_t(rows)*blocks*66;
  const device uchar *selected_gate=gate+uint64_t(expert)*expert_stride;
  const device uchar *selected_up=up+uint64_t(expert)*expert_stride;
  float gate_sum=0.0f,up_sum=0.0f;
  for(uint part=lane;part<parts;part+=32){
    float xv[32];for(ushort i=0;i<32;i++)xv[i]=x[part*32+i];uint block=part/8,sub=part%8;
    const device uchar *pg=selected_gate+uint64_t(row)*blocks*66+uint64_t(block)*66;
    const device uchar *pu=selected_up+uint64_t(row)*blocks*66+uint64_t(block)*66;
    const device ushort *qg=reinterpret_cast<device const ushort *>(pg+2)+4*sub;
    const device ushort *qu=reinterpret_cast<device const ushort *>(pu+2)+4*sub;
    device const uchar *ag=reinterpret_cast<device const uchar *>(qg);device const uchar *au=reinterpret_cast<device const uchar *>(qu);
    uint sg=uint(qg[2])|(uint(qg[3])<<16),su=uint(qu[2])|(uint(qu[3])<<16);float vg=0.0f,vu=0.0f;
    for(ushort l=0;l<4;l++){
      constant const uchar *gg=reinterpret_cast<constant const uchar *>(ds4_metal_iq2xxs_grid+ag[l]);
      constant const uchar *gu=reinterpret_cast<constant const uchar *>(ds4_metal_iq2xxs_grid+au[l]);
      uchar signg=ds4_metal_ksigns_iq2xs[(sg>>(7*l))&127],signu=ds4_metal_ksigns_iq2xs[(su>>(7*l))&127];
      for(ushort j=0;j<8;j++){float v=xv[8*l+j];uchar mask=uchar(1u<<j);vg+=v*float(gg[j])*((signg&mask)?-1.0f:1.0f);vu+=v*float(gu[j])*((signu&mask)?-1.0f:1.0f);}
    }
    gate_sum+=float(*reinterpret_cast<device const half *>(pg))*(0.5f+float(sg>>28))*vg;
    up_sum+=float(*reinterpret_cast<device const half *>(pu))*(0.5f+float(su>>28))*vu;
  }
  gate_sum=simd_sum(gate_sum)*0.25f;up_sum=simd_sum(up_sum)*0.25f;
  if(lane==0){uint64_t i=uint64_t(slot)*rows+row;gate_out[i]=gate_sum;up_out[i]=up_sum;}
}

kernel void matvec_q2_k(device const uchar *weights [[buffer(0)]], device const uchar *q8 [[buffer(1)]],
                        device float *out [[buffer(2)]], constant uint &cols [[buffer(3)]],
                        constant uint &rows [[buffer(4)]], uint row [[thread_position_in_grid]]) {
  if(row>=rows)return;uint nb=cols/256;float total=0.0f;
  for(uint block=0;block<nb;block++){
    const device uchar *p=weights+uint64_t(row)*nb*84+uint64_t(block)*84;
    const device uchar *qy=q8+uint64_t(block)*292;const device char *v=reinterpret_cast<device const char *>(qy+4);
    const device short *bs=reinterpret_cast<device const short *>(qy+260);
    int isum=0,summs=0;
    for(uint group=0;group<16;group++){
      uchar sm=p[group];uint qbase=32*(group/8)+16*(group&1),shift=((group/2)&3)*2;
      int dotv=0;for(uint i=0;i<16;i++)dotv+=int((p[16+qbase+i]>>shift)&3)*int(v[group*16+i]);
      isum+=int(sm&15)*dotv;summs+=int(bs[group])*int(sm>>4);
    }
    float yd=*reinterpret_cast<device const float *>(qy);
    total+=yd*float(*reinterpret_cast<device const half *>(p+80))*float(isum)-yd*float(*reinterpret_cast<device const half *>(p+82))*float(summs);
  }
  out[row]=total;
}

kernel void matvec_q2_k_f32(device const uchar *weights [[buffer(0)]], device const float *x [[buffer(1)]],
                            device float *out [[buffer(2)]], constant uint &cols [[buffer(3)]],
                            constant uint &rows [[buffer(4)]], uint row [[thread_position_in_grid]]) {
  if(row>=rows)return;uint nb=cols/256;float total=0.0f;
  for(uint block=0;block<nb;block++){
    const device uchar *p=weights+uint64_t(row)*nb*84+uint64_t(block)*84;
    float d=float(*reinterpret_cast<device const half *>(p+80)),dm=float(*reinterpret_cast<device const half *>(p+82));
    for(uint i=0;i<256;i++){uint group=i/16,l=i-group*16,qbase=32*(group/8)+16*(group&1),shift=((group/2)&3)*2;uchar sm=p[group];float w=d*float(sm&15)*float((p[16+qbase+l]>>shift)&3)-dm*float(sm>>4);total+=w*x[block*256+i];}
  }
  out[row]=total;
}

kernel void matvec_q2_k_selected_f32(
    device const uchar *weights [[buffer(0)]], device const int *expert_ids [[buffer(1)]],
    device const float *x [[buffer(2)]], device float *out [[buffer(3)]],
    constant uint &cols [[buffer(4)]], constant uint &rows [[buffer(5)]],
    constant uint &experts [[buffer(6)]], constant uint &slots [[buffer(7)]],
    uint2 pos [[thread_position_in_grid]]) {
  uint row=pos.x,slot=pos.y;if(row>=rows||slot>=slots)return;
  int expert=expert_ids[slot];if(expert<0||uint(expert)>=experts)return;
  uint nb=cols/256;uint64_t expert_stride=uint64_t(rows)*nb*84;
  device const uchar *selected=weights+uint64_t(expert)*expert_stride;
  device const float *selected_x=x+uint64_t(slot)*cols;
  float total=0.0f;
  for(uint block=0;block<nb;block++){
    const device uchar *p=selected+uint64_t(row)*nb*84+uint64_t(block)*84;
    float d=float(*reinterpret_cast<device const half *>(p+80)),dm=float(*reinterpret_cast<device const half *>(p+82));
    for(uint i=0;i<256;i++){uint group=i/16,l=i-group*16,qbase=32*(group/8)+16*(group&1),shift=((group/2)&3)*2;uchar sm=p[group];float w=d*float(sm&15)*float((p[16+qbase+l]>>shift)&3)-dm*float(sm>>4);total+=w*selected_x[block*256+i];}
  }
  out[uint64_t(slot)*rows+row]=total;
}

kernel void layer0_attention_pre(
    device const half *embedding [[buffer(0)]],
    device const half *hc_fn [[buffer(1)]],
    device const float *scale [[buffer(2)]],
    device const float *base [[buffer(3)]],
    device const float *norm_weight [[buffer(4)]],
    device float *out [[buffer(5)]],
    constant uint &token [[buffer(6)]]) {
  const device half *emb = embedding + uint64_t(token)*4096;
  float ss = 0.0f;
  for (uint d=0; d<4096; d++) { float v=float(emb[d]); ss += v*v; }
  const float inv = rsqrt(ss/4096.0f + 1.0e-6f);
  float mix[24];
  for (uint row=0; row<24; row++) {
    float sum=0.0f;
    const device half *wr=hc_fn+uint64_t(row)*16384;
    for (uint h=0; h<4; h++) for (uint d=0; d<4096; d++) sum += float(wr[h*4096+d])*float(emb[d])*inv;
    mix[row]=sum;
  }
  float pre[4], post[4], comb[16];
  for (uint h=0; h<4; h++) {
    pre[h]=1.0f/(1.0f+exp(-(mix[h]*scale[0]+base[h])))+1.0e-6f;
    post[h]=2.0f/(1.0f+exp(-(mix[4+h]*scale[1]+base[4+h])));
  }
  for (uint dst=0; dst<4; dst++) {
    float mx=-3.402823466e+38f;
    for (uint src=0; src<4; src++) { uint i=src+dst*4; comb[i]=mix[8+i]*scale[2]+base[8+i]; mx=max(mx,comb[i]); }
    float sum=0.0f;
    for (uint src=0; src<4; src++) { uint i=src+dst*4; comb[i]=exp(comb[i]-mx); sum+=comb[i]; }
    for (uint src=0; src<4; src++) { uint i=src+dst*4; comb[i]=comb[i]/sum+1.0e-6f; }
  }
  for (uint src=0; src<4; src++) { float sum=0.0f; for(uint dst=0;dst<4;dst++)sum+=comb[src+dst*4]; for(uint dst=0;dst<4;dst++)comb[src+dst*4]/=sum+1.0e-6f; }
  for (uint iter=1; iter<20; iter++) {
    for (uint dst=0; dst<4; dst++) { float sum=0.0f; for(uint src=0;src<4;src++)sum+=comb[src+dst*4]; for(uint src=0;src<4;src++)comb[src+dst*4]/=sum+1.0e-6f; }
    for (uint src=0; src<4; src++) { float sum=0.0f; for(uint dst=0;dst<4;dst++)sum+=comb[src+dst*4]; for(uint dst=0;dst<4;dst++)comb[src+dst*4]/=sum+1.0e-6f; }
  }
  const float pre_sum=pre[0]+pre[1]+pre[2]+pre[3];
  float cur_ss=0.0f;
  for (uint d=0; d<4096; d++) { float v=float(emb[d])*pre_sum; out[d]=v; cur_ss+=v*v; }
  const float cur_inv=rsqrt(cur_ss/4096.0f+1.0e-6f);
  for (uint d=0; d<4096; d++) out[4096+d]=out[d]*cur_inv*norm_weight[d];
}

static void hc_split4(thread float mix[24], device const float *scale, device const float *base,
                      thread float pre[4], thread float post[4], thread float comb[16]) {
  for(uint h=0;h<4;h++){
    pre[h]=1.0f/(1.0f+exp(-(mix[h]*scale[0]+base[h])))+1.0e-6f;
    post[h]=2.0f/(1.0f+exp(-(mix[4+h]*scale[1]+base[4+h])));
  }
  for(uint dst=0;dst<4;dst++){
    float mx=-3.402823466e+38f;
    for(uint src=0;src<4;src++){uint i=src+dst*4;comb[i]=mix[8+i]*scale[2]+base[8+i];mx=max(mx,comb[i]);}
    float sum=0.0f;
    for(uint src=0;src<4;src++){uint i=src+dst*4;comb[i]=exp(comb[i]-mx);sum+=comb[i];}
    for(uint src=0;src<4;src++){uint i=src+dst*4;comb[i]=comb[i]/sum+1.0e-6f;}
  }
  for(uint src=0;src<4;src++){float sum=0.0f;for(uint dst=0;dst<4;dst++)sum+=comb[src+dst*4];for(uint dst=0;dst<4;dst++)comb[src+dst*4]/=sum+1.0e-6f;}
  for(uint iter=1;iter<20;iter++){
    for(uint dst=0;dst<4;dst++){float sum=0.0f;for(uint src=0;src<4;src++)sum+=comb[src+dst*4];for(uint src=0;src<4;src++)comb[src+dst*4]/=sum+1.0e-6f;}
    for(uint src=0;src<4;src++){float sum=0.0f;for(uint dst=0;dst<4;dst++)sum+=comb[src+dst*4];for(uint dst=0;dst<4;dst++)comb[src+dst*4]/=sum+1.0e-6f;}
  }
}

kernel void layer0_attention_post(
    device const half *embedding [[buffer(0)]], device const half *hc_fn [[buffer(1)]],
    device const float *scale [[buffer(2)]], device const float *base [[buffer(3)]],
    device const float *block [[buffer(4)]], device float *out [[buffer(5)]],
    constant uint &token [[buffer(6)]]) {
  const device half *emb=embedding+uint64_t(token)*4096;
  float ss=0.0f;for(uint d=0;d<4096;d++){float v=float(emb[d]);ss+=v*v;}
  float inv=rsqrt(ss/4096.0f+1.0e-6f),mix[24];
  for(uint row=0;row<24;row++){float sum=0.0f;const device half *wr=hc_fn+uint64_t(row)*16384;for(uint h=0;h<4;h++)for(uint d=0;d<4096;d++)sum+=float(wr[h*4096+d])*float(emb[d])*inv;mix[row]=sum;}
  float pre[4],post[4],comb[16];hc_split4(mix,scale,base,pre,post,comb);
  for(uint dst=0;dst<4;dst++)for(uint d=0;d<4096;d++){float v=block[d]*post[dst];for(uint src=0;src<4;src++)v+=comb[dst+src*4]*float(emb[d]);out[dst*4096+d]=v;}
}

kernel void rms_norm(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    constant uint &groups [[buffer(4)]],
    constant uint &weighted [[buffer(5)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
  if (group >= groups) return;
  const uint64_t off=uint64_t(group)*width;
  float ss=0.0f;
  device const float4 *x4=reinterpret_cast<device const float4 *>(x+off);
  for(uint i=tid;i<width/4;i+=256)ss+=dot(x4[i],x4[i]);
  ss=simd_sum(ss);
  if(simdgroup==0)scratch[lane]=0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(lane==0)scratch[simdgroup]=ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ss=simd_sum(scratch[lane]);
  const float inv=1.0f/sqrt(ss/float(width)+1.0e-6f);
  for(uint i=tid;i<width;i+=256)out[off+i]=x[off+i]*inv*(weighted?weight[i]:1.0f);
}

inline float e4m3fn_value(int code) {
  int e=(code>>3)&15,m=code&7;
  return e==0?float(m)*0.001953125f:(1.0f+float(m)*0.125f)*exp2(float(e-7));
}

inline float e4m3fn_round(float value) {
  float sign=value<0?-1.0f:1.0f,ax=min(abs(value),448.0f);
  int lo=0,hi=126;
  while(lo<hi){int mid=(lo+hi+1)>>1;if(e4m3fn_value(mid)<=ax)lo=mid;else hi=mid-1;}
  int best=lo;
  if(best<126){float bv=e4m3fn_value(best),nv=e4m3fn_value(best+1),bd=abs(ax-bv),nd=abs(ax-nv);if(nd<bd||(nd==bd&&((best+1)&1)==0&&(best&1)!=0))best++;}
  return sign*e4m3fn_value(best);
}

kernel void compressor_pool(
    device const float *kv_cur [[buffer(0)]],
    device const float *score_cur [[buffer(1)]],
    device float *state_kv [[buffer(2)]],
    device float *state_score [[buffer(3)]],
    device float *pooled [[buffer(4)]],
    device const half *ape [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &width [[buffer(7)]],
    constant uint &ratio [[buffer(8)]],
    constant uint &position [[buffer(9)]],
    uint j [[thread_position_in_threadgroup]]) {
  if(j>=head_dim)return;
  const uint lanes=ratio==4?2:1,pos_mod=position%ratio,row=ratio==4?ratio+pos_mod:pos_mod;
  for(uint lane=0;lane<lanes;lane++){
    const uint i=lane*head_dim+j;
    state_kv[row*width+i]=kv_cur[i];
    state_score[row*width+i]=score_cur[i]+float(ape[pos_mod*width+i]);
  }
  threadgroup_barrier(mem_flags::mem_device);
  if((position+1)%ratio!=0)return;
  float maximum=-INFINITY;
  if(ratio==4){
    for(uint r=0;r<ratio;r++){
      maximum=max(maximum,state_score[r*width+j]);
      maximum=max(maximum,state_score[(ratio+r)*width+head_dim+j]);
    }
  }else{
    for(uint r=0;r<ratio;r++)maximum=max(maximum,state_score[r*width+j]);
  }
  float denom=0.0f,sum=0.0f;
  if(ratio==4){
    for(uint r=0;r<ratio;r++){
      float wp=exp(state_score[r*width+j]-maximum),wc=exp(state_score[(ratio+r)*width+head_dim+j]-maximum);
      denom+=wp+wc;sum+=wp*state_kv[r*width+j]+wc*state_kv[(ratio+r)*width+head_dim+j];
    }
  }else{
    for(uint r=0;r<ratio;r++){float w=exp(state_score[r*width+j]-maximum);denom+=w;sum+=w*state_kv[r*width+j];}
  }
  pooled[j]=denom>0.0f?sum/denom:0.0f;
  if(ratio==4){
    for(uint r=0;r<ratio;r++)for(uint lane=0;lane<2;lane++){
      const uint i=lane*head_dim+j;
      state_kv[r*width+i]=state_kv[(ratio+r)*width+i];
      state_score[r*width+i]=state_score[(ratio+r)*width+i];
    }
    threadgroup_barrier(mem_flags::mem_device);
    for(uint r=0;r<ratio;r++)for(uint lane=0;lane<2;lane++){
      const uint i=lane*head_dim+j;
      state_kv[(ratio+r)*width+i]=state_kv[r*width+i];
      state_score[(ratio+r)*width+i]=state_score[r*width+i];
    }
  }
}

inline float yarn_corr_factor(float rotations) {
  return 64.0f*log(65536.0f/(rotations*2.0f*M_PI_F))/(2.0f*log(160000.0f));
}

inline float compressed_rope_theta(uint position,uint r) {
  const float extrap=float(position)*pow(160000.0f,-float(r)/64.0f);
  const float low=max(0.0f,floor(yarn_corr_factor(32.0f))),high=min(63.0f,ceil(yarn_corr_factor(1.0f)));
  const float y=(float(r/2)-low)/max(0.001f,high-low);
  const float ramp=1.0f-clamp(y,0.0f,1.0f);
  return extrap*(0.0625f*(1.0f-ramp)+ramp);
}

kernel void compressor_finish_attention(
    device float *pooled [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *cache [[buffer(2)]],
    constant uint &cache_row [[buffer(3)]],
    constant uint &position [[buffer(4)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
  if(simdgroup==0)scratch[lane]=0.0f;
  float ss=0.0f;device float4 *p4=reinterpret_cast<device float4 *>(pooled);
  for(uint i=tid;i<128;i+=128)ss+=dot(p4[i],p4[i]);
  ss=simd_sum(ss);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(lane==0)scratch[simdgroup]=ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ss=simd_sum(scratch[lane]);
  const float inv=1.0f/sqrt(ss/512.0f+1.0e-6f);
  for(uint i=tid;i<512;i+=128)pooled[i]=pooled[i]*inv*weight[i];
  threadgroup_barrier(mem_flags::mem_device);
  if(tid<32){
    const uint r=2*tid,i=448+r;float theta=compressed_rope_theta(position,r),c=cos(theta),s=sin(theta),a=pooled[i],b=pooled[i+1];
    pooled[i]=a*c-b*s;pooled[i+1]=a*s+b*c;
  }
  threadgroup_barrier(mem_flags::mem_device);
  if(tid<7){
    const uint off=tid*64;float amax=1.0e-4f;for(uint i=0;i<64;i++)amax=max(amax,abs(pooled[off+i]));
    const float scale=exp2(ceil(log2(amax/448.0f)));
    for(uint i=0;i<64;i++)pooled[off+i]=e4m3fn_round(clamp(pooled[off+i]/scale,-448.0f,448.0f))*scale;
  }
  threadgroup_barrier(mem_flags::mem_device);
  for(uint i=tid;i<512;i+=128)cache[cache_row*512+i]=float(half(pooled[i]));
}

inline float e2m1fn_round(float value) {
  constexpr float values[8]={0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
  float sign=value<0?-1.0f:1.0f,ax=min(abs(value),6.0f),best_diff=abs(ax-values[0]);int best=0;
  for(int i=1;i<8;i++){float d=abs(ax-values[i]);if(d<best_diff||(d==best_diff&&(i&1)==0&&(best&1)!=0)){best=i;best_diff=d;}}
  return sign*values[best];
}

kernel void compressor_finish_indexer(
    device float *pooled [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *cache [[buffer(2)]],
    constant uint &cache_row [[buffer(3)]],
    constant uint &position [[buffer(4)]],
    threadgroup float *scratch [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
  threadgroup float *vals=scratch,*absbuf=scratch+128;
  if(simdgroup==0)absbuf[lane]=0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float ss=pooled[tid]*pooled[tid];ss=simd_sum(ss);
  if(lane==0)absbuf[simdgroup]=ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ss=simd_sum(absbuf[lane]);
  vals[tid]=pooled[tid]*(1.0f/sqrt(ss/128.0f+1.0e-6f))*weight[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(tid<32){
    const uint r=2*tid,i=64+r;float theta=compressed_rope_theta(position,r),c=cos(theta),s=sin(theta),a=vals[i],b=vals[i+1];
    vals[i]=a*c-b*s;vals[i+1]=a*s+b*c;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint stride=1;stride<128;stride<<=1){
    if((tid&stride)==0){uint base=(tid&~(2*stride-1))+(tid&(stride-1));float a=vals[base],b=vals[base+stride];vals[base]=a+b;vals[base+stride]=a-b;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float v=vals[tid]*0.08838834764831845f;uint block=tid>>5,lane32=tid&31,base=block*32;absbuf[tid]=abs(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint stride=16;stride>0;stride>>=1){if(lane32<stride)absbuf[base+lane32]=max(absbuf[base+lane32],absbuf[base+lane32+stride]);threadgroup_barrier(mem_flags::mem_threadgroup);}
  float amax=max(absbuf[base],7.052966104933725e-38f),scale=exp2(ceil(log2(amax/6.0f)));
  float out=e2m1fn_round(clamp(v/scale,-6.0f,6.0f))*scale;
  pooled[tid]=out;cache[cache_row*128+tid]=float(half(out));
}

kernel void rope_attention_mixed(
    device const float *q [[buffer(0)]],
    device const float *kv [[buffer(1)]],
    device const float *sinks [[buffer(2)]],
    device float *raw [[buffer(3)]],
    device const float *comp [[buffer(4)]],
    device const uint *allowed [[buffer(5)]],
    device float *scores [[buffer(6)]],
    device float *out [[buffer(7)]],
    constant uint &position [[buffer(8)]],
    constant uint &layer [[buffer(9)]],
    constant uint &raw_cap [[buffer(10)]],
    constant uint &n_comp [[buffer(11)]],
    uint head [[thread_position_in_threadgroup]]) {
  const uint qoff=head*512;
  for(uint i=0;i<512;i++)out[qoff+i]=q[qoff+i];
  if(head==0)for(uint i=0;i<512;i++)out[32768+i]=kv[i];
  for(uint r=0;r<64;r+=2){
    const uint qi=qoff+448+r;float theta=compressed_rope_theta(position,r),c=cos(theta),s=sin(theta),a=out[qi],b=out[qi+1];
    out[qi]=a*c-b*s;out[qi+1]=a*s+b*c;
    if(head==0){uint ki=32768+448+r;a=out[ki];b=out[ki+1];out[ki]=a*c-b*s;out[ki+1]=a*s+b*c;}
  }
  threadgroup_barrier(mem_flags::mem_device);
  if(head==0){
    for(uint off=0;off<448;off+=64){
      float amax=1.0e-4f;for(uint i=0;i<64;i++)amax=max(amax,abs(out[32768+off+i]));
      const float scale=exp2(ceil(log2(amax/448.0f)));
      for(uint i=0;i<64;i++)out[32768+off+i]=e4m3fn_round(clamp(out[32768+off+i]/scale,-448.0f,448.0f))*scale;
    }
    const uint row=position%raw_cap;
    for(uint i=0;i<512;i++)raw[row*512+i]=float(half(out[32768+i]));
  }
  threadgroup_barrier(mem_flags::mem_device);
  const uint n_raw=min(position+1,raw_cap),first=position+1-n_raw,total=n_raw+n_comp;
  float maximum=sinks[head];
  for(uint t=0;t<n_raw;t++){
    const uint row=(first+t)%raw_cap;float score=0.0f;
    for(uint i=0;i<512;i++)score+=float(half(out[qoff+i]))*raw[row*512+i];
    score*=0.04419417382415922f;scores[head*total+t]=score;maximum=max(maximum,score);
  }
  for(uint cidx=0;cidx<n_comp;cidx++){
    float score=-1.0e30f;
    if(allowed[cidx]){score=0.0f;for(uint i=0;i<512;i++)score+=float(half(out[qoff+i]))*comp[cidx*512+i];score*=0.04419417382415922f;maximum=max(maximum,score);}
    scores[head*total+n_raw+cidx]=score;
  }
  float denom=exp(sinks[head]-maximum);
  for(uint t=0;t<n_raw;t++)denom+=exp(scores[head*total+t]-maximum);
  for(uint cidx=0;cidx<n_comp;cidx++)if(allowed[cidx])denom+=exp(scores[head*total+n_raw+cidx]-maximum);
  for(uint i=0;i<512;i++){
    float value=0.0f;
    for(uint t=0;t<n_raw;t++){const uint row=(first+t)%raw_cap;value+=exp(scores[head*total+t]-maximum)*raw[row*512+i];}
    for(uint cidx=0;cidx<n_comp;cidx++)if(allowed[cidx])value+=exp(scores[head*total+n_raw+cidx]-maximum)*comp[cidx*512+i];
    out[33280+qoff+i]=value/denom;
  }
  for(uint r=0;r<64;r+=2){
    const uint oi=33280+qoff+448+r;float theta=compressed_rope_theta(position,r),c=cos(theta),s=-sin(theta),a=out[oi],b=out[oi+1];
    out[oi]=a*c-b*s;out[oi+1]=a*s+b*c;
  }
  (void)layer;
}

// rope_kv_commit is the head==0 quantisation pass of rope_attention_mixed,
// lifted into its own dispatch. rope_attention_mixed synchronised this work
// against the other heads with a device-scope threadgroup barrier, which
// confines every head to one threadgroup and so to one GPU core. Separating it
// lets the attention kernel spread heads across the device; ordering within the
// command buffer supplies the synchronisation the barrier used to.
kernel void rope_kv_commit(
    device float *raw [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint &position [[buffer(2)]],
    constant uint &raw_cap [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint nthreads [[threads_per_threadgroup]]) {
  // Quantisation only. rope_q_prepare has already written and rotated
  // out[32768,33280); encoder ordering supplies the dependency. Quantisation
  // covers [0,448) of that range and the rotation covered [448,512), so the two
  // do not overlap.
  for(uint blk=tid;blk<7;blk+=nthreads){
    const uint off=blk*64;
    float amax=1.0e-4f;for(uint i=0;i<64;i++)amax=max(amax,abs(out[32768+off+i]));
    const float scale=exp2(ceil(log2(amax/448.0f)));
    for(uint i=0;i<64;i++)out[32768+off+i]=e4m3fn_round(clamp(out[32768+off+i]/scale,-448.0f,448.0f))*scale;
  }
  threadgroup_barrier(mem_flags::mem_device);
  const uint row=position%raw_cap;
  for(uint i=tid;i<512;i+=nthreads)raw[row*512+i]=float(half(out[32768+i]));
}

// rope_q_prepare and rope_out_unrotate carry the two rotations of
// rope_attention_mixed unchanged: one thread per head, a constant-bound r loop,
// and the same index arithmetic. Keeping that shape keeps the compiler's
// constant folding of compressed_rope_theta, which is what a strided p=tid loop
// loses. They bracket rope_attention_mixed_wide, which is then left with only
// the score and value work -- the part that actually benefits from width.
kernel void rope_q_prepare(
    device const float *q [[buffer(0)]],
    device const float *kv [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &position [[buffer(3)]],
    uint head [[thread_position_in_threadgroup]]) {
  const uint qoff=head*512;
  for(uint i=0;i<512;i++)out[qoff+i]=q[qoff+i];
  if(head==0)for(uint i=0;i<512;i++)out[32768+i]=kv[i];
  // Byte-for-byte the preamble of rope_attention_mixed, KV block included. An
  // earlier version rotated only Q here, on the theory that a constant-bound r
  // loop was what mattered. It was exact at eight of ten test positions and
  // 1-3 ULP off at r=34 for the other two: the reference's loop body carries
  // this if(head==0) block, and the compiler unrolls -- and so constant-folds
  // compressed_rope_theta's pow -- differently for a body without it. Textual
  // identity of the loop is the property being relied on, not any weaker
  // paraphrase of it, so the KV rotation stays here rather than in
  // rope_kv_commit and the copy above feeds it.
  for(uint r=0;r<64;r+=2){
    const uint qi=qoff+448+r;float theta=compressed_rope_theta(position,r),c=cos(theta),s=sin(theta),a=out[qi],b=out[qi+1];
    out[qi]=a*c-b*s;out[qi+1]=a*s+b*c;
    if(head==0){uint ki=32768+448+r;a=out[ki];b=out[ki+1];out[ki]=a*c-b*s;out[ki+1]=a*s+b*c;}
  }
}

kernel void rope_out_unrotate(
    device float *out [[buffer(0)]],
    constant uint &position [[buffer(1)]],
    uint head [[thread_position_in_threadgroup]]) {
  const uint qoff=head*512;
  for(uint r=0;r<64;r+=2){
    const uint oi=33280+qoff+448+r;float theta=compressed_rope_theta(position,r),c=cos(theta),s=-sin(theta),a=out[oi],b=out[oi+1];
    out[oi]=a*c-b*s;out[oi+1]=a*s+b*c;
  }
}

// tg_reduce_max and tg_reduce_sum reduce one value per thread down to slot 0.
// Plain tree reductions rather than simd_ intrinsics, so the reduction does not
// depend on the SIMD width of the part it runs on.
static inline float tg_reduce_max(threadgroup float *scratch, uint tid, uint nthreads, float value) {
  scratch[tid]=value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint half_n=nthreads/2;half_n>0;half_n/=2){
    if(tid<half_n)scratch[tid]=max(scratch[tid],scratch[tid+half_n]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return scratch[0];
}


// rope_attention_mixed_wide is rope_attention_mixed with the head axis promoted
// from a thread index to a threadgroup index. One threadgroup owns one head;
// within it, threads divide the timestep axis for the score pass and the
// head_dim axis for the value pass, so neither inner loop is serial in one
// thread. It expects rope_kv_commit and rope_q_prepare to have run first in the
// same command buffer, and rope_out_unrotate to run after: the two rotations
// stay in those head-mapped kernels, in the reference's exact code shape, so
// this kernel holds only score and value work.
kernel void rope_attention_mixed_wide(
    device const float *sinks [[buffer(0)]],
    device const float *raw [[buffer(1)]],
    device const float *comp [[buffer(2)]],
    device const uint *allowed [[buffer(3)]],
    device float *scores [[buffer(4)]],
    device float *out [[buffer(5)]],
    constant uint &position [[buffer(6)]],
    constant uint &raw_cap [[buffer(7)]],
    constant uint &n_comp [[buffer(8)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint nthreads [[threads_per_threadgroup]]) {
  threadgroup float scratch[256];
  const uint qoff=head*512;
  const uint n_raw=min(position+1,raw_cap),first=position+1-n_raw,total=n_raw+n_comp;
  float local_max=-1.0e30f;
  for(uint t=tid;t<n_raw;t+=nthreads){
    const uint row=(first+t)%raw_cap;float score=0.0f;
    for(uint i=0;i<512;i++)score+=float(half(out[qoff+i]))*raw[row*512+i];
    score*=0.04419417382415922f;scores[head*total+t]=score;local_max=max(local_max,score);
  }
  for(uint cidx=tid;cidx<n_comp;cidx+=nthreads){
    float score=-1.0e30f;
    if(allowed[cidx]){score=0.0f;for(uint i=0;i<512;i++)score+=float(half(out[qoff+i]))*comp[cidx*512+i];score*=0.04419417382415922f;local_max=max(local_max,score);}
    scores[head*total+n_raw+cidx]=score;
  }
  threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
  // sinks[head] seeds the maximum exactly as the single-threadgroup kernel did.
  const float maximum=max(sinks[head],tg_reduce_max(scratch,tid,nthreads,local_max));

  // One thread sums the denominator in the original order -- sink first, then
  // raw timesteps ascending, then compressed entries -- and broadcasts it. A
  // tree reduction over strided partial sums computes the same value under
  // exact arithmetic but a different association, which perturbs the low bits;
  // summing in order keeps the wide kernel bit-identical to the narrow one, so
  // the two can be compared for equality rather than within a tolerance. The
  // serial pass is over at most raw_cap+n_comp terms, against the value pass's
  // 512 coordinates times the same count, so it is not the cost here.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(tid==0){
    float d=exp(sinks[head]-maximum);
    for(uint t=0;t<n_raw;t++)d+=exp(scores[head*total+t]-maximum);
    for(uint cidx=0;cidx<n_comp;cidx++)if(allowed[cidx])d+=exp(scores[head*total+n_raw+cidx]-maximum);
    scratch[0]=d;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float denom=scratch[0];

  // The value pass partitions head_dim, so each thread accumulates its own
  // slice over every timestep and no reduction is needed here.
  for(uint i=tid;i<512;i+=nthreads){
    float value=0.0f;
    for(uint t=0;t<n_raw;t++){const uint row=(first+t)%raw_cap;value+=exp(scores[head*total+t]-maximum)*raw[row*512+i];}
    for(uint cidx=0;cidx<n_comp;cidx++)if(allowed[cidx])value+=exp(scores[head*total+n_raw+cidx]-maximum)*comp[cidx*512+i];
    out[33280+qoff+i]=value/denom;
  }
}

kernel void rope_attention_raw(
    device const float *q [[buffer(0)]],
    device const float *kv [[buffer(1)]],
    device const float *sinks [[buffer(2)]],
    device float *cache [[buffer(3)]],
    device float *scores [[buffer(4)]],
    device float *out [[buffer(5)]],
    constant uint &position [[buffer(6)]],
    constant uint &cache_cap [[buffer(7)]],
    uint head [[thread_position_in_threadgroup]]) {
  const uint qoff=head*512;
  for(uint i=0;i<512;i++)out[qoff+i]=q[qoff+i];
  if(head==0)for(uint i=0;i<512;i++)out[32768+i]=kv[i];
  float theta=float(position);
  const float theta_scale=pow(10000.0f,-2.0f/64.0f);
  for(uint i=0;i<64;i+=2){
    const uint qi=qoff+448+i;
    float c=cos(theta),s=sin(theta),a=out[qi],b=out[qi+1];
    out[qi]=a*c-b*s;out[qi+1]=a*s+b*c;
    if(head==0){uint ki=32768+448+i;a=out[ki];b=out[ki+1];out[ki]=a*c-b*s;out[ki+1]=a*s+b*c;}
    theta*=theta_scale;
  }
  threadgroup_barrier(mem_flags::mem_device);
  if(head==0){
    for(uint off=0;off<448;off+=64){
      float amax=1.0e-4f;for(uint i=0;i<64;i++)amax=max(amax,abs(out[32768+off+i]));
      const float scale=exp2(ceil(log2(amax/448.0f)));
      for(uint i=0;i<64;i++)out[32768+off+i]=e4m3fn_round(clamp(out[32768+off+i]/scale,-448.0f,448.0f))*scale;
    }
    const uint row=position%cache_cap;
    for(uint i=0;i<512;i++)cache[row*512+i]=float(half(out[32768+i]));
  }
  threadgroup_barrier(mem_flags::mem_device);
  const uint count=min(position+1,cache_cap);
  const uint first=position+1-count;
  float maximum=sinks[head];
  for(uint t=0;t<count;t++){
    const uint row=(first+t)%cache_cap;
    float score=0.0f;for(uint i=0;i<512;i++)score+=float(half(out[qoff+i]))*cache[row*512+i];
    score/=sqrt(512.0f);scores[head*cache_cap+t]=score;maximum=max(maximum,score);
  }
  float denom=exp(sinks[head]-maximum);
  for(uint t=0;t<count;t++)denom+=exp(scores[head*cache_cap+t]-maximum);
  for(uint i=0;i<512;i++){
    float value=0.0f;
    for(uint t=0;t<count;t++){
      const uint row=(first+t)%cache_cap;
      value+=exp(scores[head*cache_cap+t]-maximum)*cache[row*512+i];
    }
    out[33280+qoff+i]=value/denom;
  }
  theta=float(position);
  for(uint i=0;i<64;i+=2){
    const uint oi=33280+qoff+448+i;
    float c=cos(theta),s=-sin(theta),a=out[oi],b=out[oi+1];
    out[oi]=a*c-b*s;out[oi+1]=a*s+b*c;
    theta*=theta_scale;
  }
}

kernel void rope_attention_one(
    device const float *q [[buffer(0)]],
    device const float *kv [[buffer(1)]],
    device const float *sinks [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &position [[buffer(4)]],
    uint head [[thread_position_in_grid]]) {
  if(head>=64)return;
  const uint qoff=head*512;
  for(uint i=0;i<512;i++)out[qoff+i]=q[qoff+i];
  if(head==0)for(uint i=0;i<512;i++)out[32768+i]=kv[i];
  float theta=float(position);
  const float theta_scale=pow(10000.0f,-2.0f/64.0f);
  for(uint i=0;i<64;i+=2){
    const uint qi=qoff+448+i;
    float c=cos(theta),s=sin(theta),a=out[qi],b=out[qi+1];
    out[qi]=a*c-b*s;out[qi+1]=a*s+b*c;
    if(head==0){uint ki=32768+448+i;a=out[ki];b=out[ki+1];out[ki]=a*c-b*s;out[ki+1]=a*s+b*c;}
    theta*=theta_scale;
  }
  threadgroup_barrier(mem_flags::mem_device);
  if(head==0){
    for(uint off=0;off<448;off+=64){
      float amax=1.0e-4f;for(uint i=0;i<64;i++)amax=max(amax,abs(out[32768+off+i]));
      const float fp8_scale=exp2(ceil(log2(amax/448.0f)));
      for(uint i=0;i<64;i++){
        float x=clamp(out[32768+off+i]/fp8_scale,-448.0f,448.0f),sign=x<0?-1.0f:1.0f,ax=min(abs(x),448.0f);
        int lo=0,hi=126;while(lo<hi){int mid=(lo+hi+1)>>1;int e=(mid>>3)&15,m=mid&7;float v=e==0?float(m)*0.001953125f:(1.0f+float(m)*0.125f)*exp2(float(e-7));if(v<=ax)lo=mid;else hi=mid-1;}
        int best=lo;if(best<126){int e=(best>>3)&15,m=best&7;float bv=e==0?float(m)*0.001953125f:(1.0f+float(m)*0.125f)*exp2(float(e-7));int ni=best+1;e=(ni>>3)&15;m=ni&7;float nv=e==0?float(m)*0.001953125f:(1.0f+float(m)*0.125f)*exp2(float(e-7));float bd=abs(ax-bv),nd=abs(ax-nv);if(nd<bd||(nd==bd&&((best+1)&1)==0&&(best&1)!=0))best++;}
        int e=(best>>3)&15,m=best&7;float v=e==0?float(m)*0.001953125f:(1.0f+float(m)*0.125f)*exp2(float(e-7));out[32768+off+i]=sign*v*fp8_scale;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  float score=0.0f;
  // The oracle rounds BOTH operands to f16 before the dot: it loads Q from f32
  // and casts it to half (dsv4_misc.metal:5748-5751) alongside the half KV, then
  // widens both for the accumulation (dsv4_attend_shared_h4_row). Rounding only
  // KV leaves Q at full f32 precision and diverges from the capture at f16 ULP
  // scale, which is the whole of the kqv_out residual.
  for(uint i=0;i<512;i++)score+=float(half(out[qoff+i]))*float(half(out[32768+i]));
  score/=sqrt(512.0f);
  const float mx=max(score,sinks[head]);
  const float weight=exp(score-mx)/(exp(score-mx)+exp(sinks[head]-mx));
  for(uint i=0;i<512;i++)out[32768+512+qoff+i]=weight*float(half(out[32768+i]));
}
`

// commandBuffer creates a command buffer and counts it.
//
// Every command buffer the inference runtime creates comes from here. That is
// what makes submissions.Buffers the whole count rather than a subset, and it
// is enforced at the source level by TestCommandBufferCreationIsCentralized:
// before centralization the package held 28 creation sites, cbufs/tok read
// 86.00 against a gputrace count of 87.0, and the missing term was the output
// head's buffer -- one per token, invisible to a metric with no variance.
func (o *metalPinnedOps) commandBuffer() apple.MTLCommandBuffer {
	o.submissions.Buffers++
	return o.dev.Queue().CommandBuffer()
}

// record files a completed command buffer's GPU-time span, if a log is
// installed. It is a no-op in production.
//
// Call it only after the buffer has completed: it reads two properties of a
// finished object and encodes nothing, so a run with a log installed submits
// the same dispatches in the same order as a run without one. That is what
// makes bit-identity the right gate for the profiling build.
func (o *metalPinnedOps) record(region string, layer, position int, cb apple.MTLCommandBuffer) {
	if o == nil || o.regions == nil {
		return
	}
	o.regions.add(metalRegion{
		Region:   region,
		Layer:    layer,
		Position: position,
		Start:    float64(cb.GPUStartTime()),
		End:      float64(cb.GPUEndTime()),
	})
}

// submitAndWait commits cb, waits for it, and files its span.
//
// Every immediate-submit op goes through here so the span log covers the same
// buffers the creation counter does. Before it existed, 18 of the 20 sites
// committed and waited inline and filed nothing, so each was a buffer the
// counter saw and the log did not -- which is why the workload benchmarks,
// whose reports require the two counts to agree, could not run at all: a
// 512-token prefill created 44*512+4 buffers and recorded 43*512+1 spans.
//
// The span carries no layer or position because an immediate op is not told
// either. rope-attention-mixed keeps its own record call rather than using
// this, since it does know both and a span that names them is worth more.
func (o *metalPinnedOps) submitAndWait(region string, cb apple.MTLCommandBuffer) error {
	cb.Commit()
	if err := libmetal.WaitCommandBuffer(context.Background(), cb); err != nil {
		return err
	}
	o.record(region, -1, 0, cb)
	return nil
}
