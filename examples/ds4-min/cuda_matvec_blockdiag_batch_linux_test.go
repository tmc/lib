//go:build linux

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

// TestCUDAQualificationQ80BlockDiagBatchMatvec pins the batched block-diagonal
// attention output projection to bit identity with the per-token kernel.
//
// Each tile shares weight loads across up to eight tokens. Comparisons cross
// tile boundaries and partial row blocks; NaN-filled outputs catch omitted
// tiles as well as incorrect token offsets. Each token must retain the same
// accumulation and reduction order as the per-token kernel.
//
// Model free: synthetic Q8_0 weights with swept binary16 scales and
// full-mantissa activations, so float addition is non-associative on the test
// data and any reassociation or permutation of a token's sum is caught.
func TestCUDAQualificationQ80BlockDiagBatchMatvec(t *testing.T) {
	const (
		diag = 8
		cols = 1056
		rows = 131
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
	const maxBatch = 256
	host := make([]float32, maxBatch*tokenCols)
	for i := range host {
		host[i] = float32(math.Sin(float64(i)*0.37)) * float32(1+i%7)
	}
	in := cudaSynthUpload(t, r, host)
	q := cudaExpertAllocBytes(t, r, maxBatch*tokenCols/32*36)
	if err := r.encodeQuantQ80(in, q, tokenCols); err != nil {
		t.Fatal(err)
	}

	for _, batch := range []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 255, 256} {
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

	t.Run("reject-bad-shapes", func(t *testing.T) {
		out := cudaTestAlloc(t, r, maxBatch*tokenRows)
		for _, bad := range []struct {
			name        string
			diag, batch int
			trimIn      int
			trimOut     int
			trimWeights int
		}{
			{name: "zero batch", diag: diag, batch: 0},
			{name: "negative batch", diag: diag, batch: -1},
			{name: "zero diag", diag: 0, batch: maxBatch},
			{name: "negative diag", diag: -1, batch: maxBatch},
			{name: "grid y overflow", diag: math.MaxUint16 + 1, batch: maxBatch},
			{name: "grid z overflow", diag: diag, batch: math.MaxUint16*cudaQ80BlockDiagMaxBatch + 1},
			{name: "nondivisible rows", diag: 7, batch: maxBatch},
			{name: "short input", diag: diag, batch: maxBatch, trimIn: 1},
			{name: "short output", diag: diag, batch: maxBatch, trimOut: 1},
			{name: "short weights", diag: diag, batch: maxBatch, trimWeights: 1},
		} {
			t.Run(bad.name, func(t *testing.T) {
				if err := out.CopyFromHost(makeNaNs(out.Len())); err != nil {
					t.Fatal(err)
				}
				weights := tensor
				weights.bytes -= uint64(bad.trimWeights)
				if err := r.encodeQ80MatvecQuantizedBlockDiagBatch(weights,
					q.Slice(0, q.Len()-bad.trimIn), out.Slice(0, out.Len()-bad.trimOut),
					bad.diag, bad.batch); err == nil {
					t.Fatal("accepted invalid shape")
				}
				for i, v := range cudaTestRead(t, out) {
					if !math.IsNaN(float64(v)) {
						t.Fatalf("output %d written despite rejection: %v", i, v)
					}
				}
			})
		}
	})
}

// TestCUDAQualificationQ80BlockDiagMaxBatchConstantSync keeps the Go tile width
// and the compile-time register array in the kernel at the same width.
func TestCUDAQualificationQ80BlockDiagMaxBatchConstantSync(t *testing.T) {
	needle := fmt.Sprintf("#define DS4_Q80_BLOCKDIAG_MAX_BATCH %d", cudaQ80BlockDiagMaxBatch)
	if !strings.Contains(cudaQ80MatvecBody, needle) {
		t.Fatalf("cuda_q80_matvec.cu does not contain expected constant definition %q", needle)
	}
}

// Rejection must happen before accessing the CUDA context, even when extents
// would overflow a batch-size multiplication.
func TestQ80BlockDiagBatchInvalidExtents(t *testing.T) {
	for _, tt := range []struct {
		name              string
		cols, rows, bytes uint64
		diag, batch       int
	}{
		{"zero batch", 32, 8, 272, 8, 0},
		{"negative batch", 32, 8, 272, 8, -1},
		{"grid y overflow", 32, 65536, math.MaxUint64, 65536, 1},
		{"grid z overflow", 32, 8, 272, 8, math.MaxUint16*cudaQ80BlockDiagMaxBatch + 1},
		{"zero columns", 0, 8, 272, 8, 1},
		{"unaligned columns", 33, 8, 272, 8, 1},
		{"nondivisible rows", 32, 9, 306, 8, 1},
		{"row bound", 32, uint64(math.MaxInt32) + 1, math.MaxUint64, 1, 1},
		{"input multiplication overflow", math.MaxUint32 - 31, 65535, math.MaxUint64, 65535, math.MaxUint16 * cudaQ80BlockDiagMaxBatch},
		{"output multiplication overflow", 32, uint64(math.MaxInt32) * 65535, math.MaxUint64, 65535, math.MaxUint16 * cudaQ80BlockDiagMaxBatch},
	} {
		t.Run(tt.name, func(t *testing.T) {
			tensor := cudaModelTensor{typeID: 8, device: 1, dimensions: []uint64{tt.cols, tt.rows}, bytes: tt.bytes}
			var r cudaFoundation
			if err := r.encodeQ80MatvecQuantizedBlockDiagBatch(tensor, cudamem.Slice[byte]{}, cudamem.Slice[float32]{}, tt.diag, tt.batch); err == nil {
				t.Fatal("accepted invalid extent")
			}
		})
	}
}
