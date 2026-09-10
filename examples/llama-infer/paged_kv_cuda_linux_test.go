//go:build linux && !tpu

package main

// Qualification for the paged KV path in paged_kv_cuda.go.
//
// x/cudaattn already qualifies the two kernels themselves against a reference
// (TestPagedRoPEStoreAndAttention). What is unqualified here is the wiring this
// example adds on top of them: that pagedKVRuntime hands the kernels the pool
// base and extent that cudakv.PagedCache actually allocated, that it addresses
// the right layer, and that a block table produced by PagedCache.AllocSequence
// -- not one hand-written for a kernel test -- decodes to the same values the
// dense layout does.
//
// These tests skip without a CUDA device. A skip is not a pass: until this
// reports a real pass on real hardware, the paged path stays behind
// LLAMA_INFER_PAGED_KV and must not be described as working.

import (
	"context"
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func TestPagedKVMatchesDense(t *testing.T) {
	const (
		layers    = 2
		qHeads    = 4
		kvHeads   = 2
		headDim   = 6 // deliberately not a vector-width multiple
		blockSize = 3
		maxSeqs   = 3
		maxSeqLen = 6
		ropeTheta = float32(10000)
	)
	cfg := pagedKVConfig{
		BlockSize:       blockSize,
		TotalBlocks:     8,
		MaxBlocksPerSeq: (maxSeqLen + blockSize - 1) / blockSize,
	}

	dev := cudatest.Require(t)
	defer dev.Close()
	rt, err := newPagedKVRuntime(dev, nil, cfg, layers, qHeads, kvHeads, headDim, maxSeqLen, maxSeqs)
	if err != nil {
		t.Fatalf("newPagedKVRuntime: %v", err)
	}
	defer rt.Close()
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()

	// Block tables come from the allocator, so a mismatch between how cudakv
	// hands out slots and how the kernels index them shows up here.
	seqLens := []int32{4, 6, 3}
	tables := make([][]int, len(seqLens))
	for seq, seqLen := range seqLens {
		id := int64(seq + 1)
		if err := rt.cache.AllocSequence(id, int(seqLen)); err != nil {
			t.Fatalf("AllocSequence(%d, %d): %v", id, seqLen, err)
		}
		defer rt.cache.FreeSequence(id)
		tables[seq] = rt.cache.BlockTable(id)
		if len(tables[seq]) == 0 {
			t.Fatalf("sequence %d has an empty block table", id)
		}
	}
	blockTable := make([]int32, len(seqLens)*cfg.MaxBlocksPerSeq)
	for seq, table := range tables {
		for i, slot := range table {
			blockTable[seq*cfg.MaxBlocksPerSeq+i] = int32(slot)
		}
	}

	var seqIndex, positions []int32
	for seq, seqLen := range seqLens {
		for pos := 0; pos < int(seqLen); pos++ {
			seqIndex = append(seqIndex, int32(seq))
			positions = append(positions, int32(pos))
		}
	}
	tokens := len(seqIndex)
	if err := rt.UploadBatch(seqIndex, positions, seqLens, blockTable); err != nil {
		t.Fatalf("UploadBatch: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	for layer := 0; layer < layers; layer++ {
		// Distinct values per layer: a runtime that ignored the layer index and
		// always addressed layer zero would still pass with identical inputs.
		q := pagedKVSeries(tokens*qHeads*headDim, 17+layer, 29)
		k := pagedKVSeries(tokens*kvHeads*headDim, 11+layer, 31)
		v := pagedKVSeries(tokens*kvHeads*headDim, 7+layer, 23)

		dq := pagedKVUpload(t, dev, q)
		dk := pagedKVUpload(t, dev, k)
		dv := pagedKVUpload(t, dev, v)
		dqr := pagedKVZeros[float32](t, dev, len(q))
		dout := pagedKVZeros[float32](t, dev, len(seqLens)*qHeads*headDim)
		for _, free := range []func() error{dq.Free, dk.Free, dv.Free, dqr.Free, dout.Free} {
			defer free()
		}

		if err := rt.StoreKV(stream, layer, dq, dk, dv, dqr, tokens, len(seqLens), ropeTheta); err != nil {
			t.Fatalf("layer %d StoreKV: %v", layer, err)
		}
		if err := stream.Synchronize(ctx); err != nil {
			t.Fatalf("layer %d store synchronize: %v", layer, err)
		}
		if err := rt.CheckStatus(); err != nil {
			t.Fatalf("layer %d StoreKV status: %v", layer, err)
		}

		// The rotated queries are the kernel's own output; take the current
		// token's row per sequence from them so the comparison isolates the
		// paged read, not the rotation.
		gotQR := pagedKVDownload(t, dev, dqr)
		currentQ := make([]float32, len(seqLens)*qHeads*headDim)
		base := 0
		for seq, seqLen := range seqLens {
			last := base + int(seqLen) - 1
			copy(currentQ[seq*qHeads*headDim:(seq+1)*qHeads*headDim],
				gotQR[last*qHeads*headDim:(last+1)*qHeads*headDim])
			base += int(seqLen)
		}
		dcurrent := pagedKVUpload(t, dev, currentQ)
		defer dcurrent.Free()

		if err := rt.UploadBatch(seqIndex, positions, seqLens, blockTable); err != nil {
			t.Fatalf("layer %d re-upload: %v", layer, err)
		}
		if err := rt.Decode(stream, layer, dcurrent, dout, len(seqLens)); err != nil {
			t.Fatalf("layer %d Decode: %v", layer, err)
		}
		if err := stream.Synchronize(ctx); err != nil {
			t.Fatalf("layer %d decode synchronize: %v", layer, err)
		}
		if err := rt.CheckStatus(); err != nil {
			t.Fatalf("layer %d Decode status: %v", layer, err)
		}
		got := pagedKVDownload(t, dev, dout)

		// The reference is dense: contiguous per-sequence K and V in arrival
		// order, with no block table at all.
		wantK := pagedKVRoPE(k, positions, kvHeads, headDim, float64(ropeTheta))
		want := pagedKVDenseAttention(currentQ, wantK, v, seqLens, qHeads, kvHeads, headDim)
		pagedKVCompare(t, fmt.Sprintf("layer %d", layer), got, want, 3e-6)

		// The comparison above is only meaningful if the reference actually
		// depends on which sequence's query it reads. Attributing each
		// sequence's query to the next sequence's cache must move the result
		// well outside the tolerance. This control is host-side: it qualifies
		// the tolerance, not the kernel.
		rotated := make([]float32, len(currentQ))
		row := qHeads * headDim
		for seq := range seqLens {
			src := ((seq + 1) % len(seqLens)) * row
			copy(rotated[seq*row:(seq+1)*row], currentQ[src:src+row])
		}
		control := pagedKVDenseAttention(rotated, wantK, v, seqLens, qHeads, kvHeads, headDim)
		if pagedKVMaxDiff(control, want) < 1e-4 {
			t.Fatalf("layer %d null control is not discriminating: max difference %g", layer, pagedKVMaxDiff(control, want))
		}
	}
}

// TestPagedKVRejectsBadBlockTable is the device control for the qualification
// above. An out-of-range slot must be reported through the kernels' status word
// rather than read past the pool and return plausible numbers, and
// pagedKVRuntime.CheckStatus must surface it. Without this, a passing
// differential says nothing about what happens to a stale block table.
func TestPagedKVRejectsBadBlockTable(t *testing.T) {
	const (
		layers    = 1
		qHeads    = 2
		kvHeads   = 1
		headDim   = 4
		blockSize = 2
		maxSeqs   = 1
		maxSeqLen = 4
	)
	cfg := pagedKVConfig{BlockSize: blockSize, TotalBlocks: 4, MaxBlocksPerSeq: 2}

	dev := cudatest.Require(t)
	defer dev.Close()
	rt, err := newPagedKVRuntime(dev, nil, cfg, layers, qHeads, kvHeads, headDim, maxSeqLen, maxSeqs)
	if err != nil {
		t.Fatalf("newPagedKVRuntime: %v", err)
	}
	defer rt.Close()
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()

	seqLens := []int32{4}
	seqIndex := []int32{0, 0, 0, 0}
	positions := []int32{0, 1, 2, 3}
	// cfg.TotalBlocks is one past the last valid slot.
	bad := []int32{int32(cfg.TotalBlocks), 1}
	if err := rt.UploadBatch(seqIndex, positions, seqLens, bad); err != nil {
		t.Fatalf("UploadBatch: %v", err)
	}
	currentQ := pagedKVZeros[float32](t, dev, qHeads*headDim)
	out := pagedKVZeros[float32](t, dev, qHeads*headDim)
	defer currentQ.Free()
	defer out.Free()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := rt.Decode(stream, 0, currentQ, out, len(seqLens)); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if err := stream.Synchronize(ctx); err != nil {
		t.Fatalf("synchronize: %v", err)
	}
	if err := rt.CheckStatus(); err == nil {
		t.Fatal("CheckStatus accepted a block table entry past the end of the pool")
	}
}

// TestPagedKVUntrackedPoolBytes checks the accounting the paged path reports for
// memory the engine's tracker cannot see. It needs no device.
func TestPagedKVUntrackedPoolBytes(t *testing.T) {
	rt := &pagedKVRuntime{
		cfg:     pagedKVConfig{BlockSize: 16, TotalBlocks: 8, MaxBlocksPerSeq: 4},
		layers:  2,
		kvHeads: 2,
		headDim: 6,
	}
	const want = 8 * 16 * 2 * 6 * 2 * 2 * 4
	if got := rt.pagedKVUntrackedPoolBytes(); got != want {
		t.Errorf("pagedKVUntrackedPoolBytes() = %d, want %d", got, want)
	}
	if got := (*pagedKVRuntime)(nil).pagedKVUntrackedPoolBytes(); got != 0 {
		t.Errorf("nil runtime reported %d bytes", got)
	}
}

func TestDefaultPagedKVConfig(t *testing.T) {
	tests := []struct {
		name      string
		maxSeqLen int
		capacity  int
		want      pagedKVConfig
	}{
		{"exact", 32, 2, pagedKVConfig{BlockSize: 16, TotalBlocks: 4, MaxBlocksPerSeq: 2}},
		{"rounds up", 33, 1, pagedKVConfig{BlockSize: 16, TotalBlocks: 3, MaxBlocksPerSeq: 3}},
		{"zero capacity", 16, 0, pagedKVConfig{BlockSize: 16, TotalBlocks: 1, MaxBlocksPerSeq: 1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := defaultPagedKVConfig(tt.maxSeqLen, tt.capacity); got != tt.want {
				t.Errorf("defaultPagedKVConfig(%d, %d) = %+v, want %+v", tt.maxSeqLen, tt.capacity, got, tt.want)
			}
		})
	}
}

// pagedKVSeries builds a deterministic spread of small values.
func pagedKVSeries(n, stride, modulus int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i*stride)%modulus-modulus/2) / float32(modulus-stride%7)
	}
	return out
}

func pagedKVUpload[T any](t *testing.T, dev *cuda.Device, values []T) cudamem.Slice[T] {
	t.Helper()
	s, err := cudamem.Alloc[T](dev, len(values))
	if err != nil {
		t.Fatal(err)
	}
	if err := dev.WithContext(func() error { return s.CopyFromHost(values) }); err != nil {
		_ = s.Free()
		t.Fatal(err)
	}
	return s
}

func pagedKVZeros[T any](t *testing.T, dev *cuda.Device, n int) cudamem.Slice[T] {
	t.Helper()
	return pagedKVUpload(t, dev, make([]T, n))
}

func pagedKVDownload(t *testing.T, dev *cuda.Device, s cudamem.Slice[float32]) []float32 {
	t.Helper()
	out := make([]float32, s.Len())
	if err := dev.WithContext(func() error { return s.CopyToHost(out) }); err != nil {
		t.Fatal(err)
	}
	return out
}

func pagedKVRoPE(src []float32, positions []int32, heads, headDim int, theta float64) []float32 {
	dst := make([]float32, len(src))
	half := headDim / 2
	width := heads * headDim
	for token, pos := range positions {
		for h := 0; h < heads; h++ {
			base := token*width + h*headDim
			for d := 0; d < half; d++ {
				angle := float64(pos) / math.Pow(theta, 2*float64(d)/float64(headDim))
				a, b := src[base+d], src[base+d+half]
				dst[base+d] = a*float32(math.Cos(angle)) - b*float32(math.Sin(angle))
				dst[base+d+half] = a*float32(math.Sin(angle)) + b*float32(math.Cos(angle))
			}
		}
	}
	return dst
}

// pagedKVDenseAttention is the dense reference: k and v are laid out
// contiguously per sequence in arrival order, the layout the CUDA engine uses
// today. It knows nothing about blocks.
func pagedKVDenseAttention(currentQ, k, v []float32, lens []int32, qHeads, kvHeads, headDim int) []float32 {
	out := make([]float32, len(currentQ))
	group := qHeads / kvHeads
	tokenBase := 0
	for seq, seqLen32 := range lens {
		seqLen := int(seqLen32)
		for qh := 0; qh < qHeads; qh++ {
			kh := qh / group
			qbase := (seq*qHeads + qh) * headDim
			scores := make([]float64, seqLen)
			maxScore := math.Inf(-1)
			for pos := 0; pos < seqLen; pos++ {
				kbase := ((tokenBase+pos)*kvHeads + kh) * headDim
				for d := 0; d < headDim; d++ {
					scores[pos] += float64(currentQ[qbase+d] * k[kbase+d])
				}
				scores[pos] /= math.Sqrt(float64(headDim))
				maxScore = math.Max(maxScore, scores[pos])
			}
			sum := 0.0
			for pos := range scores {
				scores[pos] = math.Exp(scores[pos] - maxScore)
				sum += scores[pos]
			}
			for pos, score := range scores {
				vbase := ((tokenBase+pos)*kvHeads + kh) * headDim
				for d := 0; d < headDim; d++ {
					out[qbase+d] += float32(score/sum) * v[vbase+d]
				}
			}
		}
		tokenBase += seqLen
	}
	return out
}

func pagedKVMaxDiff(got, want []float32) float64 {
	worst := 0.0
	for i := range got {
		if d := math.Abs(float64(got[i] - want[i])); d > worst {
			worst = d
		}
	}
	return worst
}

func pagedKVCompare(t *testing.T, what string, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d values, want %d", what, len(got), len(want))
	}
	for i := range got {
		if d := math.Abs(float64(got[i] - want[i])); d > tolerance {
			t.Fatalf("%s: value %d is %g, want %g (difference %g exceeds %g)", what, i, got[i], want[i], d, tolerance)
		}
	}
}
