//go:build linux

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"testing"
)

// TestCUDAQualificationQ80BlockDiagBatchMatvec pins the batched block-diagonal
// attention output projection to bit identity with the per-token kernel.
//
// The runtime issues encodeQ80MatvecQuantizedBlockDiag once per layer per token
// (cuda_session_linux.go, encodeQ80NativeBlockDiag with diag=8), 43 launches per
// token of 196.3 us each. Every launch streams the whole attn_output_a weight
// tensor -- 8*1024 rows of 128 blocks of 34 bytes, 34 MiB -- because the weight
// byte a warp reads is addressed by (diag, row) alone and carries no token index.
// A batch of tokens therefore reads exactly the same weight bytes, and
// ds4_q8_0_matvec_quantized_blockdiag_batch reads them once for the whole batch.
//
// The claim this test defends is that amortizing the weight load changes nothing
// numerically: each token keeps its own accumulator, its own lane-strided walk
// over the blocks, and its own shfl_down reduction, so the float32 results must
// match Float32bits exactly, at every width from 2 to cudaQ80BlockDiagMaxBatch.
//
// Model free: synthetic Q8_0 weights with swept binary16 scales and
// full-mantissa activations, so float addition is non-associative on the test
// data and any reassociation or permutation of a token's sum is caught.
func TestCUDAQualificationQ80BlockDiagBatchMatvec(t *testing.T) {
	const (
		diag = 8
		cols = 1024
		rows = 128
	)
	r := cudaSynthFoundation(t)

	blocks := cols / 32
	rawWeights := make([]byte, diag*rows*blocks*34)
	for totalRow := range diag * rows {
		for block := range blocks {
			p := rawWeights[(totalRow*blocks+block)*34:]
			binary.LittleEndian.PutUint16(p, uint16(0x2c00+(totalRow*31+block*17)%2048))
			for i := range 32 {
				p[2+i] = byte(int8((totalRow*13+block*7+i*5)%251%127 - 63))
			}
		}
	}
	tensor := cudaSynthTensor(t, r, "attn_output_a.weight", 8,
		[]uint64{uint64(cols), uint64(diag * rows)}, rawWeights)

	// One token occupies diag*cols activations and diag*rows outputs.
	const tokenCols = diag * cols
	const tokenRows = diag * rows
	maxBatch := cudaQ80BlockDiagMaxBatch
	host := make([]float32, maxBatch*tokenCols)
	for i := range host {
		host[i] = float32(math.Sin(float64(i)*0.37)) * float32(1+i%7)
	}
	in := cudaSynthUpload(t, r, host)
	q := cudaExpertAllocBytes(t, r, maxBatch*tokenCols/32*36)
	if err := r.encodeQuantQ80(in, q, tokenCols); err != nil {
		t.Fatal(err)
	}

	for batch := 2; batch <= maxBatch; batch++ {
		t.Run(fmt.Sprintf("batch-%d", batch), func(t *testing.T) {
			qBatch := q.Slice(0, batch*tokenCols/32*36)
			out := cudaTestAlloc(t, r, batch*tokenRows)
			sequential := cudaTestAlloc(t, r, batch*tokenRows)
			if err := out.CopyFromHost(makeNaNs(batch * tokenRows)); err != nil {
				t.Fatal(err)
			}
			if err := sequential.CopyFromHost(makeNaNs(batch * tokenRows)); err != nil {
				t.Fatal(err)
			}

			cudaQualificationMeasure(t, r, func() error {
				return r.encodeQ80MatvecQuantizedBlockDiagBatch(tensor, qBatch, out, diag, batch)
			})
			got := cudaTestRead(t, out)

			cudaQualificationMeasure(t, r, func() error {
				for token := range batch {
					qStart := token * tokenCols / 32 * 36
					if err := r.encodeQ80MatvecQuantizedBlockDiag(tensor,
						q.Slice(qStart, qStart+tokenCols/32*36),
						sequential.Slice(token*tokenRows, (token+1)*tokenRows), diag); err != nil {
						return err
					}
				}
				return nil
			})
			want := cudaTestRead(t, sequential)

			nonzero := 0
			for i := range want {
				if math.IsNaN(float64(want[i])) {
					t.Fatalf("output %d unwritten by the per-token launches", i)
				}
				if want[i] != 0 {
					nonzero++
				}
			}
			if nonzero < len(want)/2 {
				t.Fatalf("only %d of %d outputs nonzero; comparison not exercising the kernel", nonzero, len(want))
			}
			if err := exactFloat32Bits(got, want); err != nil {
				t.Fatalf("batch %d differs from per-token launches: %v", batch, err)
			}
		})
	}

	t.Run("reject-oversized-batch", func(t *testing.T) {
		over := maxBatch + 1
		oversizedIn := cudaExpertAllocBytes(t, r, over*tokenCols/32*36)
		oversizedOut := cudaTestAlloc(t, r, over*tokenRows)
		poison := makeNaNs(over * tokenRows)
		if err := oversizedOut.CopyFromHost(poison); err != nil {
			t.Fatal(err)
		}
		err := r.encodeQ80MatvecQuantizedBlockDiagBatch(tensor, oversizedIn, oversizedOut, diag, over)
		wantSubstr := fmt.Sprintf("batch %d (max %d)", over, maxBatch)
		if err == nil || !strings.Contains(err.Error(), wantSubstr) {
			t.Fatalf("expected error containing %q, got %v", wantSubstr, err)
		}
		// Rejection must be total: nothing was launched, so every output is
		// still poison. A kernel that silently truncated to maxBatch would have
		// written the first maxBatch tokens.
		after := cudaTestRead(t, oversizedOut)
		for i := range after {
			if !math.IsNaN(float64(after[i])) {
				t.Fatalf("output %d written despite rejection: %v", i, after[i])
			}
		}
	})

	t.Run("reject-bad-shapes", func(t *testing.T) {
		out := cudaTestAlloc(t, r, maxBatch*tokenRows)
		for _, bad := range []struct {
			name        string
			diag, batch int
			trimIn      int
			trimOut     int
		}{
			{name: "zero batch", diag: diag, batch: 0},
			{name: "negative batch", diag: diag, batch: -1},
			{name: "zero diag", diag: 0, batch: maxBatch},
			{name: "short input", diag: diag, batch: maxBatch, trimIn: 1},
			{name: "short output", diag: diag, batch: maxBatch, trimOut: 1},
		} {
			t.Run(bad.name, func(t *testing.T) {
				if err := r.encodeQ80MatvecQuantizedBlockDiagBatch(tensor,
					q.Slice(0, q.Len()-bad.trimIn), out.Slice(0, out.Len()-bad.trimOut),
					bad.diag, bad.batch); err == nil {
					t.Fatal("accepted invalid shape")
				}
			})
		}
	})
}

// TestCUDAQualificationQ80BlockDiagMaxBatchConstantSync keeps the Go batch bound
// and the compile-time register array in the kernel at the same width. A Go
// constant larger than the CUDA one would index past sum[] in the kernel.
func TestCUDAQualificationQ80BlockDiagMaxBatchConstantSync(t *testing.T) {
	needle := fmt.Sprintf("#define DS4_Q80_BLOCKDIAG_MAX_BATCH %d", cudaQ80BlockDiagMaxBatch)
	if !strings.Contains(cudaQ80MatvecBody, needle) {
		t.Fatalf("cuda_q80_matvec.cu does not contain expected constant definition %q", needle)
	}
}
