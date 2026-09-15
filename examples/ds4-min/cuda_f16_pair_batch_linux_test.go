//go:build linux

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/lm/quant"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaF16PairBatchFixture builds one weight pair and batch token-major
// activation rows. Weight bits straddle the ordered-chunk boundaries the
// single-token kernel is pinned to, and the activations differ per token so a
// batched kernel that reused one token's row would fail the bit comparison.
func cudaF16PairBatchFixture(cols, rows, batch int) (a, b []byte, x []float32) {
	a, b = make([]byte, cols*rows*2), make([]byte, cols*rows*2)
	for row := range rows {
		for col := range cols {
			wa := uint16(0x2000 + (row*37+col*17)%1024)
			wb := uint16(0xa000 + (row*19+col*31)%1024)
			binary.LittleEndian.PutUint16(a[(row*cols+col)*2:], wa)
			binary.LittleEndian.PutUint16(b[(row*cols+col)*2:], wb)
		}
	}
	x = make([]float32, batch*cols)
	for token := range batch {
		for col := range cols {
			x[token*cols+col] = float32((col*29+token*97)%251-125) / 128
		}
	}
	return a, b, x
}

// TestCUDAQualificationF16PairOrderedBatch holds the batched pair kernel to
// byte equality with one single-token launch per token at every supported
// width, across all three column regimes of the kernel.
func TestCUDAQualificationF16PairOrderedBatch(t *testing.T) {
	for _, tc := range []struct {
		name       string
		cols, rows int
	}{
		{"vectorized_4096", 4096, 6},
		{"partial_chunk", 263, 5},
		{"short_rows", 37, 4},
		{"wide", 8192, 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := cudaSynthFoundation(t)
			for batch := 2; batch <= cudaF16PairMaxBatch; batch++ {
				t.Run(fmt.Sprintf("batch-%d", batch), func(t *testing.T) {
					testCUDAQualificationF16PairOrderedBatch(t, r, tc.cols, tc.rows, batch)
				})
			}
		})
	}
}

func testCUDAQualificationF16PairOrderedBatch(t *testing.T, r *cudaFoundation, cols, rows, batch int) {
	t.Helper()
	a, b, x := cudaF16PairBatchFixture(cols, rows, batch)
	dims := []uint64{uint64(cols), uint64(rows)}
	ta := cudaSynthTensor(t, r, "batched ordered F16 A", 1, dims, a)
	tb := cudaSynthTensor(t, r, "batched ordered F16 B", 1, dims, b)
	in := cudaSynthUpload(t, r, x)

	batchedA, batchedB := cudaTestAlloc(t, r, batch*rows), cudaTestAlloc(t, r, batch*rows)
	sequentialA, sequentialB := cudaTestAlloc(t, r, batch*rows), cudaTestAlloc(t, r, batch*rows)
	for _, out := range []cudamem.Slice[float32]{batchedA, batchedB, sequentialA, sequentialB} {
		if err := out.CopyFromHost(makeNaNs(batch * rows)); err != nil {
			t.Fatal(err)
		}
	}

	cudaQualificationMeasure(t, r, func() error {
		return r.encodeF16PairOrderedBatch(ta, tb, in, batchedA, batchedB, batch)
	})
	cudaQualificationMeasure(t, r, func() error {
		for token := range batch {
			if err := r.encodeF16PairOrdered(ta, tb,
				in.Slice(token*cols, (token+1)*cols),
				sequentialA.Slice(token*rows, (token+1)*rows),
				sequentialB.Slice(token*rows, (token+1)*rows)); err != nil {
				return err
			}
		}
		return nil
	})

	gotA, gotB := cudaTestRead(t, batchedA), cudaTestRead(t, batchedB)
	wantA, wantB := cudaTestRead(t, sequentialA), cudaTestRead(t, sequentialB)
	if err := exactFloat32Bits(gotA, wantA); err != nil {
		t.Fatalf("batched A differs from single-token launches: %v", err)
	}
	if err := exactFloat32Bits(gotB, wantB); err != nil {
		t.Fatalf("batched B differs from single-token launches: %v", err)
	}

	// Positive control: the sequential side is itself checked against an
	// independent float64 dot, so byte equality with a wrong pair of kernels
	// cannot pass the gate.
	refA, refB := make([]float32, batch*rows), make([]float32, batch*rows)
	for token := range batch {
		for row := range rows {
			var sumA, sumB float64
			for col := range cols {
				wa := binary.LittleEndian.Uint16(a[(row*cols+col)*2:])
				wb := binary.LittleEndian.Uint16(b[(row*cols+col)*2:])
				sumA += float64(quant.Float16ToFloat32(wa)) * float64(x[token*cols+col])
				sumB += float64(quant.Float16ToFloat32(wb)) * float64(x[token*cols+col])
			}
			refA[token*rows+row] = float32(sumA)
			refB[token*rows+row] = float32(sumB)
		}
	}
	if err := cudaQualificationClose(gotA, refA, cudaF16DefaultATol, cudaF16DefaultRTol); err != nil {
		t.Fatalf("batched A differs from reference: %v", err)
	}
	if err := cudaQualificationClose(gotB, refB, cudaF16DefaultATol, cudaF16DefaultRTol); err != nil {
		t.Fatalf("batched B differs from reference: %v", err)
	}

	// Cross-token control: the tokens must not be permuted or aliased.
	if batch >= 2 {
		if err := exactFloat32Bits(gotA[:rows], gotA[rows:2*rows]); err == nil {
			t.Fatal("token 0 and token 1 outputs are identical; fixture cannot see aliasing")
		}
	}

	for _, bad := range []struct {
		name  string
		check func() error
	}{
		{"zero batch", func() error { return r.encodeF16PairOrderedBatch(ta, tb, in, batchedA, batchedB, 0) }},
		{"negative batch", func() error { return r.encodeF16PairOrderedBatch(ta, tb, in, batchedA, batchedB, -1) }},
		{"short input", func() error {
			return r.encodeF16PairOrderedBatch(ta, tb, in.Slice(0, in.Len()-1), batchedA, batchedB, batch)
		}},
		{"short output A", func() error {
			return r.encodeF16PairOrderedBatch(ta, tb, in, batchedA.Slice(0, batchedA.Len()-1), batchedB, batch)
		}},
		{"short output B", func() error {
			return r.encodeF16PairOrderedBatch(ta, tb, in, batchedA, batchedB.Slice(0, batchedB.Len()-1), batch)
		}},
		{"truncated B", func() error {
			short := tb
			short.bytes--
			return r.encodeF16PairOrderedBatch(ta, short, in, batchedA, batchedB, batch)
		}},
		{"wrong weight type", func() error {
			wrong := ta
			wrong.typeID = 0
			return r.encodeF16PairOrderedBatch(wrong, tb, in, batchedA, batchedB, batch)
		}},
	} {
		if err := bad.check(); err == nil {
			t.Errorf("%s: accepted invalid request", bad.name)
		}
	}
}

// TestCUDAQualificationF16PairOrderedBatchOversized fixes the class of bug
// where a width past the compile-time maximum is silently truncated to the
// widths the kernel unrolls instead of rejected.
func TestCUDAQualificationF16PairOrderedBatchOversized(t *testing.T) {
	const (
		cols = 256
		rows = 4
	)
	r := cudaSynthFoundation(t)
	over := cudaF16PairMaxBatch + 1
	a, b, x := cudaF16PairBatchFixture(cols, rows, over)
	dims := []uint64{uint64(cols), uint64(rows)}
	ta := cudaSynthTensor(t, r, "oversized ordered F16 A", 1, dims, a)
	tb := cudaSynthTensor(t, r, "oversized ordered F16 B", 1, dims, b)
	in := cudaSynthUpload(t, r, x)
	outA, outB := cudaTestAlloc(t, r, over*rows), cudaTestAlloc(t, r, over*rows)
	poison := makeNaNs(over * rows)
	if err := outA.CopyFromHost(poison); err != nil {
		t.Fatal(err)
	}
	if err := outB.CopyFromHost(poison); err != nil {
		t.Fatal(err)
	}
	err := r.encodeF16PairOrderedBatch(ta, tb, in, outA, outB, over)
	want := fmt.Sprintf("batch %d (max %d)", over, cudaF16PairMaxBatch)
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("encodeF16PairOrderedBatch(batch=%d) error = %v, want one containing %q", over, err, want)
	}
	// Rejection must be total: nothing may have been written.
	for i, v := range cudaTestRead(t, outA) {
		if !math.IsNaN(float64(v)) {
			t.Fatalf("rejected launch wrote out_a[%d] = %v", i, v)
		}
	}
	for i, v := range cudaTestRead(t, outB) {
		if !math.IsNaN(float64(v)) {
			t.Fatalf("rejected launch wrote out_b[%d] = %v", i, v)
		}
	}
}

// TestCUDAF16PairMaxBatchConstantSync keeps the Go bound and the CUDA unroll
// bound from drifting apart.
func TestCUDAF16PairMaxBatchConstantSync(t *testing.T) {
	needle := fmt.Sprintf("#define DS4_F16_PAIR_MAX_BATCH %d", cudaF16PairMaxBatch)
	if !strings.Contains(cudaF16DefaultSource, needle) {
		t.Fatalf("cuda_f16_default.cu does not contain %q", needle)
	}
}
