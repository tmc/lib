//go:build darwin

package metalattn

import (
	"math"
	"strconv"
	"testing"
	"unsafe"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// referenceDecodeAttention computes single-query grouped-query attention over the
// KV cache the straightforward way, for comparison against the Metal kernel.
func referenceDecodeAttention(q, kCache, vCache []float32, cfg Config, curPos int) []float32 {
	groupSize := cfg.NumQHeads / cfg.NumKVHeads
	out := make([]float32, cfg.NumQHeads*cfg.HeadDim)

	for h := 0; h < cfg.NumQHeads; h++ {
		kvHead := h / groupSize
		qOff := h * cfg.HeadDim

		scores := make([]float64, curPos+1)
		maxScore := math.Inf(-1)
		for t := 0; t <= curPos; t++ {
			kOff := (t*cfg.NumKVHeads + kvHead) * cfg.HeadDim
			var dot float64
			for d := 0; d < cfg.HeadDim; d++ {
				dot += float64(q[qOff+d]) * float64(kCache[kOff+d])
			}
			dot *= float64(cfg.Scale)
			scores[t] = dot
			if dot > maxScore {
				maxScore = dot
			}
		}

		var sum float64
		for t := range scores {
			scores[t] = math.Exp(scores[t] - maxScore)
			sum += scores[t]
		}

		for d := 0; d < cfg.HeadDim; d++ {
			var acc float64
			for t := 0; t <= curPos; t++ {
				vOff := (t*cfg.NumKVHeads + kvHead) * cfg.HeadDim
				acc += scores[t] * float64(vCache[vOff+d])
			}
			out[qOff+d] = float32(acc / sum)
		}
	}
	return out
}

// fillPseudoRandom fills dst with a deterministic low-amplitude sequence. The
// values must stay small enough that the reference softmax and the kernel's
// online softmax agree in float32.
func fillPseudoRandom(dst []float32, seed uint32) {
	state := seed | 1
	for i := range dst {
		state = state*1664525 + 1013904223
		dst[i] = (float32(state>>8&0xffff)/65535.0 - 0.5)
	}
}

func TestCausalFlashAttentionDecodeMatchesReference(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}

	// Llama 3.2 1B uses 32 query heads and 8 key/value heads. Keep the
	// production geometry here so every GQA group and KV head participates in
	// the oracle, not just a reduced lookalike.
	cfg := DefaultConfig(32, 8, 64)
	const maxSeqLen = 1024

	lib, err := dev.CompileLibrary(CausalFlashAttentionMSL)
	if err != nil {
		t.Fatalf("CompileLibrary: %v", err)
	}
	pso, err := dev.NewComputePipeline(lib, "CausalFlashAttentionDecode")
	if err != nil {
		t.Fatalf("NewComputePipeline: %v", err)
	}

	qBuf, err := metalmem.Alloc[float32](dev, cfg.NumQHeads*cfg.HeadDim)
	if err != nil {
		t.Fatalf("alloc q: %v", err)
	}
	kBuf, err := metalmem.Alloc[float32](dev, maxSeqLen*cfg.NumKVHeads*cfg.HeadDim)
	if err != nil {
		t.Fatalf("alloc k: %v", err)
	}
	vBuf, err := metalmem.Alloc[float32](dev, maxSeqLen*cfg.NumKVHeads*cfg.HeadDim)
	if err != nil {
		t.Fatalf("alloc v: %v", err)
	}
	outBuf, err := metalmem.Alloc[float32](dev, cfg.NumQHeads*cfg.HeadDim)
	if err != nil {
		t.Fatalf("alloc out: %v", err)
	}

	fillPseudoRandom(qBuf.HostSlice(), 7)
	fillPseudoRandom(kBuf.HostSlice(), 11)
	fillPseudoRandom(vBuf.HostSlice(), 13)

	maxSeqVal := int32(maxSeqLen)
	qHeadsVal := int32(cfg.NumQHeads)
	kvHeadsVal := int32(cfg.NumKVHeads)
	headDimVal := int32(cfg.HeadDim)
	scaleVal := cfg.Scale

	newScalar := func(p unsafe.Pointer) apple.MTLBuffer {
		return dev.Raw().NewBufferWithBytesLengthOptions(p, 4, apple.MTLResourceStorageModeShared)
	}
	maxSeqB := newScalar(unsafe.Pointer(&maxSeqVal))
	qHeadsB := newScalar(unsafe.Pointer(&qHeadsVal))
	kvHeadsB := newScalar(unsafe.Pointer(&kvHeadsVal))
	headDimB := newScalar(unsafe.Pointer(&headDimVal))
	scaleB := newScalar(unsafe.Pointer(&scaleVal))

	// Cover single-tile, exact-threadgroup-boundary, multi-tile, and the
	// sequence lengths that overflowed the old fixed 512-entry score array.
	for _, curPos := range []int{0, 1, 127, 128, 255, 256, 511, 512, 600, 1023} {
		t.Run("curPos="+strconv.Itoa(curPos), func(t *testing.T) {
			for i := range outBuf.HostSlice() {
				outBuf.HostSlice()[i] = float32(math.NaN())
			}

			curPosVal := int32(curPos)
			curPosB := newScalar(unsafe.Pointer(&curPosVal))

			cb := dev.Queue().CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			enc.SetComputePipelineState(pso)
			enc.SetBufferWithOffsetAtIndex(qBuf.Buffer(), 0, 0)
			enc.SetBufferWithOffsetAtIndex(kBuf.Buffer(), 0, 1)
			enc.SetBufferWithOffsetAtIndex(vBuf.Buffer(), 0, 2)
			enc.SetBufferWithOffsetAtIndex(outBuf.Buffer(), 0, 3)
			enc.SetBufferWithOffsetAtIndex(curPosB, 0, 4)
			enc.SetBufferWithOffsetAtIndex(maxSeqB, 0, 5)
			enc.SetBufferWithOffsetAtIndex(qHeadsB, 0, 6)
			enc.SetBufferWithOffsetAtIndex(kvHeadsB, 0, 7)
			enc.SetBufferWithOffsetAtIndex(headDimB, 0, 8)
			enc.SetBufferWithOffsetAtIndex(scaleB, 0, 9)
			enc.DispatchThreadgroupsThreadsPerThreadgroup(
				apple.MTLSize{Width: uint(cfg.NumQHeads), Height: 1, Depth: 1},
				apple.MTLSize{Width: DecodeThreadgroupSize, Height: 1, Depth: 1},
			)
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()

			got := outBuf.HostSlice()
			want := referenceDecodeAttention(qBuf.HostSlice(), kBuf.HostSlice(), vBuf.HostSlice(), cfg, curPos)

			const tol = 1e-5
			for i := range want {
				diff := math.Abs(float64(got[i] - want[i]))
				if math.IsNaN(float64(got[i])) || diff > tol {
					t.Fatalf("out[%d] (head %d, dim %d) = %v, want %v (diff %v)",
						i, i/cfg.HeadDim, i%cfg.HeadDim, got[i], want[i], diff)
				}
			}
		})
	}
}

func TestCausalFlashAttentionDecodeMutationBoundaries(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}

	cfg := DefaultConfig(32, 8, 64)
	const (
		maxSeqLen = 512
		curPos    = 256
	)

	lib, err := dev.CompileLibrary(CausalFlashAttentionMSL)
	if err != nil {
		t.Fatalf("CompileLibrary: %v", err)
	}
	pso, err := dev.NewComputePipeline(lib, "CausalFlashAttentionDecode")
	if err != nil {
		t.Fatalf("NewComputePipeline: %v", err)
	}
	qBuf, err := metalmem.Alloc[float32](dev, cfg.NumQHeads*cfg.HeadDim)
	if err != nil {
		t.Fatalf("alloc q: %v", err)
	}
	kBuf, err := metalmem.Alloc[float32](dev, maxSeqLen*cfg.NumKVHeads*cfg.HeadDim)
	if err != nil {
		t.Fatalf("alloc k: %v", err)
	}
	vBuf, err := metalmem.Alloc[float32](dev, maxSeqLen*cfg.NumKVHeads*cfg.HeadDim)
	if err != nil {
		t.Fatalf("alloc v: %v", err)
	}
	outBuf, err := metalmem.Alloc[float32](dev, cfg.NumQHeads*cfg.HeadDim)
	if err != nil {
		t.Fatalf("alloc out: %v", err)
	}
	t.Cleanup(func() {
		qBuf.Free()
		kBuf.Free()
		vBuf.Free()
		outBuf.Free()
	})
	fillPseudoRandom(qBuf.HostSlice(), 23)
	fillPseudoRandom(kBuf.HostSlice(), 29)
	fillPseudoRandom(vBuf.HostSlice(), 31)

	run := func() []float32 {
		curPosVal := int32(curPos)
		maxSeqVal := int32(maxSeqLen)
		qHeadsVal := int32(cfg.NumQHeads)
		kvHeadsVal := int32(cfg.NumKVHeads)
		headDimVal := int32(cfg.HeadDim)
		scaleVal := cfg.Scale
		newScalar := func(p unsafe.Pointer) apple.MTLBuffer {
			return dev.Raw().NewBufferWithBytesLengthOptions(p, 4, apple.MTLResourceStorageModeShared)
		}
		cb := dev.Queue().CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(qBuf.Buffer(), 0, 0)
		enc.SetBufferWithOffsetAtIndex(kBuf.Buffer(), 0, 1)
		enc.SetBufferWithOffsetAtIndex(vBuf.Buffer(), 0, 2)
		enc.SetBufferWithOffsetAtIndex(outBuf.Buffer(), 0, 3)
		enc.SetBufferWithOffsetAtIndex(newScalar(unsafe.Pointer(&curPosVal)), 0, 4)
		enc.SetBufferWithOffsetAtIndex(newScalar(unsafe.Pointer(&maxSeqVal)), 0, 5)
		enc.SetBufferWithOffsetAtIndex(newScalar(unsafe.Pointer(&qHeadsVal)), 0, 6)
		enc.SetBufferWithOffsetAtIndex(newScalar(unsafe.Pointer(&kvHeadsVal)), 0, 7)
		enc.SetBufferWithOffsetAtIndex(newScalar(unsafe.Pointer(&headDimVal)), 0, 8)
		enc.SetBufferWithOffsetAtIndex(newScalar(unsafe.Pointer(&scaleVal)), 0, 9)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			apple.MTLSize{Width: uint(cfg.NumQHeads), Height: 1, Depth: 1},
			apple.MTLSize{Width: DecodeThreadgroupSize, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		return append([]float32(nil), outBuf.HostSlice()...)
	}

	baseline := run()
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	future := (curPos+1)*kvDim + 3
	vBuf.HostSlice()[future] += 1000
	if got := run(); !equalFloat32(got, baseline) {
		t.Fatal("future value mutation changed causal output")
	}
	vBuf.HostSlice()[future] -= 1000
	active := curPos*kvDim + 3
	vBuf.HostSlice()[active] += 1000
	if got := run(); equalFloat32(got, baseline) {
		t.Fatal("active value mutation did not change causal output")
	}
}

func equalFloat32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestCausalFlashAttentionDecodeUniformValues checks the normalization directly:
// when every cached value vector is identical, attention must reproduce it
// exactly regardless of the scores, because the softmax weights sum to one.
func TestCausalFlashAttentionDecodeUniformValues(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}

	cfg := DefaultConfig(4, 4, 64)
	const maxSeqLen = 1024
	const curPos = 700

	lib, err := dev.CompileLibrary(CausalFlashAttentionMSL)
	if err != nil {
		t.Fatalf("CompileLibrary: %v", err)
	}
	pso, err := dev.NewComputePipeline(lib, "CausalFlashAttentionDecode")
	if err != nil {
		t.Fatalf("NewComputePipeline: %v", err)
	}

	qBuf, _ := metalmem.Alloc[float32](dev, cfg.NumQHeads*cfg.HeadDim)
	kBuf, _ := metalmem.Alloc[float32](dev, maxSeqLen*cfg.NumKVHeads*cfg.HeadDim)
	vBuf, _ := metalmem.Alloc[float32](dev, maxSeqLen*cfg.NumKVHeads*cfg.HeadDim)
	outBuf, _ := metalmem.Alloc[float32](dev, cfg.NumQHeads*cfg.HeadDim)

	fillPseudoRandom(qBuf.HostSlice(), 3)
	fillPseudoRandom(kBuf.HostSlice(), 5)

	// Every timestep stores the same value vector.
	pattern := make([]float32, cfg.HeadDim)
	fillPseudoRandom(pattern, 17)
	vHost := vBuf.HostSlice()
	for t := 0; t < maxSeqLen; t++ {
		for h := 0; h < cfg.NumKVHeads; h++ {
			copy(vHost[(t*cfg.NumKVHeads+h)*cfg.HeadDim:], pattern)
		}
	}

	curPosVal := int32(curPos)
	maxSeqVal := int32(maxSeqLen)
	qHeadsVal := int32(cfg.NumQHeads)
	kvHeadsVal := int32(cfg.NumKVHeads)
	headDimVal := int32(cfg.HeadDim)
	scaleVal := cfg.Scale
	newScalar := func(p unsafe.Pointer) apple.MTLBuffer {
		return dev.Raw().NewBufferWithBytesLengthOptions(p, 4, apple.MTLResourceStorageModeShared)
	}

	cb := dev.Queue().CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(qBuf.Buffer(), 0, 0)
	enc.SetBufferWithOffsetAtIndex(kBuf.Buffer(), 0, 1)
	enc.SetBufferWithOffsetAtIndex(vBuf.Buffer(), 0, 2)
	enc.SetBufferWithOffsetAtIndex(outBuf.Buffer(), 0, 3)
	enc.SetBufferWithOffsetAtIndex(newScalar(unsafe.Pointer(&curPosVal)), 0, 4)
	enc.SetBufferWithOffsetAtIndex(newScalar(unsafe.Pointer(&maxSeqVal)), 0, 5)
	enc.SetBufferWithOffsetAtIndex(newScalar(unsafe.Pointer(&qHeadsVal)), 0, 6)
	enc.SetBufferWithOffsetAtIndex(newScalar(unsafe.Pointer(&kvHeadsVal)), 0, 7)
	enc.SetBufferWithOffsetAtIndex(newScalar(unsafe.Pointer(&headDimVal)), 0, 8)
	enc.SetBufferWithOffsetAtIndex(newScalar(unsafe.Pointer(&scaleVal)), 0, 9)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(
		apple.MTLSize{Width: uint(cfg.NumQHeads), Height: 1, Depth: 1},
		apple.MTLSize{Width: DecodeThreadgroupSize, Height: 1, Depth: 1},
	)
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()

	got := outBuf.HostSlice()
	const tol = 1e-5
	for h := 0; h < cfg.NumQHeads; h++ {
		for d := 0; d < cfg.HeadDim; d++ {
			if diff := math.Abs(float64(got[h*cfg.HeadDim+d] - pattern[d])); diff > tol {
				t.Fatalf("head %d dim %d = %v, want %v (diff %v): softmax weights do not sum to 1",
					h, d, got[h*cfg.HeadDim+d], pattern[d], diff)
			}
		}
	}
}
