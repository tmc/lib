//go:build linux

package main

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"
)

// TestCUDAQualificationQ80BatchMatvec is the differential gate on
// ds4_q8_0_matvec_quantized_batch. The batched launch must reproduce the
// unbatched kernel's float32 output bit for bit at every width from 2 to
// cudaQ80MaxBatch: batching amortizes the weight read, it does not reassociate
// any token's accumulation. The comparison is on bits, not on a tolerance.
func TestCUDAQualificationQ80BatchMatvec(t *testing.T) {
	r := cudaSynthFoundation(t)
	const (
		cols = 1024
		rows = 128
	)
	weights := cudaSynthQ80(rand.New(rand.NewSource(80)), rows, cols, -8)
	tensor := cudaSynthTensor(t, r, "batched quantized Q8_0 fixture", 8, []uint64{cols, rows}, weights)
	for batch := 2; batch <= cudaQ80MaxBatch; batch++ {
		t.Run(fmt.Sprintf("batch-%d", batch), func(t *testing.T) {
			testCUDAQualificationQ80BatchMatvec(t, r, tensor, weights, batch, cols, rows)
		})
	}
	// An over-wide batch must be named and refused. Truncating it to
	// cudaQ80MaxBatch would silently drop the trailing tokens' outputs.
	t.Run("reject-oversized-batch", func(t *testing.T) {
		oversized := cudaQ80MaxBatch + 1
		in := cudaExpertAllocBytes(t, r, oversized*cols/32*36)
		out := cudaTestAlloc(t, r, oversized*rows)
		err := r.encodeQ80MatvecQuantizedBatch(tensor, in, out, oversized)
		want := fmt.Sprintf("batch %d (max %d)", oversized, cudaQ80MaxBatch)
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("encodeQ80MatvecQuantizedBatch(batch=%d) error = %v, want one containing %q", oversized, err, want)
		}
	})
}

func testCUDAQualificationQ80BatchMatvec(t *testing.T, r *cudaFoundation, tensor cudaModelTensor, weights []byte, batch, cols, rows int) {
	x := make([]float32, batch*cols)
	for i := range x {
		x[i] = float32((i*37+i/17+i/cols*29)%257-128) / 128
	}
	raw := cudaQ80ActivationReference(x)
	in := cudaSynthUpload(t, r, x)
	q := cudaExpertAllocBytes(t, r, len(raw))
	if err := r.encodeQuantQ80(in, q, cols); err != nil {
		t.Fatal(err)
	}
	cudaQ80CheckBytes(t, q, raw)

	out := cudaTestAlloc(t, r, batch*rows)
	sequential := cudaTestAlloc(t, r, batch*rows)
	if err := out.CopyFromHost(makeNaNs(batch * rows)); err != nil {
		t.Fatal(err)
	}
	if err := sequential.CopyFromHost(makeNaNs(batch * rows)); err != nil {
		t.Fatal(err)
	}

	cudaQualificationMeasure(t, r, func() error {
		return r.encodeQ80MatvecQuantizedBatch(tensor, q, out, batch)
	})
	got := cudaTestRead(t, out)

	blockBytes := cols / 32 * 36
	cudaQualificationMeasure(t, r, func() error {
		for token := range batch {
			if err := r.encodeQ80MatvecQuantized(tensor,
				q.Slice(token*blockBytes, (token+1)*blockBytes),
				sequential.Slice(token*rows, (token+1)*rows)); err != nil {
				return err
			}
		}
		return nil
	})
	want := cudaTestRead(t, sequential)
	if err := exactFloat32Bits(got, want); err != nil {
		t.Fatalf("batched launch differs from %d single-row launches: %v", batch, err)
	}

	// The unbatched kernel is the bit-exactness oracle. The host reference is a
	// second, independent witness that neither kernel is uniformly wrong.
	for token := range batch {
		reference := cudaQ80MatvecReference(t, weights, raw[token*blockBytes:(token+1)*blockBytes], cols, rows)
		if err := cudaQualificationClose(got[token*rows:(token+1)*rows], reference, cudaQ80ATol, cudaQ80RTol); err != nil {
			t.Fatalf("token %d differs from host reference: %v", token, err)
		}
	}

	for name, err := range map[string]error{
		"zero batch":   r.encodeQ80MatvecQuantizedBatch(tensor, q, out, 0),
		"short input":  r.encodeQ80MatvecQuantizedBatch(tensor, q.Slice(0, q.Len()-1), out, batch),
		"short output": r.encodeQ80MatvecQuantizedBatch(tensor, q, out.Slice(0, out.Len()-1), batch),
	} {
		if err == nil {
			t.Errorf("accepted %s", name)
		}
	}
}

// TestCUDAQualificationQ80MaxBatchConstantSync ties cudaQ80MaxBatch to the
// compile-time width the kernel unrolls over. A Go constant larger than the
// CUDA one would launch tokens the kernel never accumulates.
func TestCUDAQualificationQ80MaxBatchConstantSync(t *testing.T) {
	needle := fmt.Sprintf("#define DS4_Q80_MAX_BATCH %d", cudaQ80MaxBatch)
	if !strings.Contains(cudaQ80MatvecBody, needle) {
		t.Fatalf("cuda_q80_matvec.cu does not contain %q", needle)
	}
}
