//go:build linux

package main

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

// TestCUDAHeadRMSRoPEBatch checks that the fused query path, headRMSRoPEBatch
// then mixedAttentionPrepareKVBatch, is bit-identical to encodeRMS followed by
// mixedAttentionPrepareBatch: the query rows it rotates in place against the
// query region of the old scratch, and the key/value region against the old
// one. The query region of the new scratch must stay untouched.
func TestCUDAHeadRMSRoPEBatch(t *testing.T) {
	r := cudaSynthFoundation(t)
	const n = 5
	alloc := func(count int) cudamem.Slice[float32] {
		d, err := cudamem.Alloc[float32](r.device, count)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := d.Free(); err != nil {
				t.Error(err)
			}
		})
		return d
	}
	download := func(d cudamem.Slice[float32]) []float32 {
		if err := r.device.WithContext(func() error {
			return r.device.DefaultStream().Synchronize(context.Background())
		}); err != nil {
			t.Fatal(err)
		}
		v := make([]float32, d.Len())
		if err := d.CopyToHost(v); err != nil {
			t.Fatal(err)
		}
		return v
	}
	rng := rand.New(rand.NewPCG(1, 2))
	q := make([]float32, n*cudaAttentionHeads)
	for i := range q {
		// Spread head scales over several octaves so the norm is not trivial.
		q[i] = float32(rng.NormFloat64() * math.Exp2(float64(i/512%7)-3))
	}
	kv := make([]float32, n*cudaKVWidth)
	for i := range kv {
		kv[i] = float32(rng.NormFloat64())
	}
	const sentinel = float32(-12345.5)
	fill := make([]float32, n*cudaAttentionStride)
	for i := range fill {
		fill[i] = sentinel
	}

	tests := []struct {
		name      string
		position  int
		denseRoPE bool
	}{
		{"compressed/0", 0, false},
		{"compressed/4093", 4093, false},
		{"dense/0", 0, true},
		{"dense/2047", 2047, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			qOld, qNew, kvDev := alloc(len(q)), alloc(len(q)), alloc(len(kv))
			outOld, outNew := alloc(len(fill)), alloc(len(fill))
			cudaCompressedUpload(t, qOld, q)
			cudaCompressedUpload(t, qNew, q)
			cudaCompressedUpload(t, kvDev, kv)
			cudaCompressedUpload(t, outOld, fill)
			cudaCompressedUpload(t, outNew, fill)

			if err := r.encodeRMS(qOld, cudaModelTensor{}, qOld, n*64, false); err != nil {
				t.Fatal(err)
			}
			if err := r.mixedAttentionPrepareBatch(qOld, kvDev, outOld, tt.position, 128, n, tt.denseRoPE); err != nil {
				t.Fatal(err)
			}
			if err := r.headRMSRoPEBatch(qNew, tt.position, n, tt.denseRoPE); err != nil {
				t.Fatal(err)
			}
			if err := r.mixedAttentionPrepareKVBatch(kvDev, outNew, tt.position, n, tt.denseRoPE); err != nil {
				t.Fatal(err)
			}

			old, gotQ, gotOut := download(outOld), download(qNew), download(outNew)
			for tok := range n {
				for i := range cudaAttentionHeads {
					want, got := old[tok*cudaAttentionStride+i], gotQ[tok*cudaAttentionHeads+i]
					if math.Float32bits(got) != math.Float32bits(want) {
						t.Fatalf("token %d query %d (head %d dim %d) = %v, want %v", tok, i, i/512, i%512, got, want)
					}
					if v := gotOut[tok*cudaAttentionStride+i]; v != sentinel {
						t.Fatalf("token %d query scratch %d written: %v", tok, i, v)
					}
				}
				for i := cudaAttentionHeads; i < cudaAttentionScratch; i++ {
					want, got := old[tok*cudaAttentionStride+i], gotOut[tok*cudaAttentionStride+i]
					if math.Float32bits(got) != math.Float32bits(want) {
						t.Fatalf("token %d key/value %d = %v, want %v", tok, i-cudaAttentionHeads, got, want)
					}
				}
			}
		})
	}

	short := alloc(cudaAttentionHeads)
	if err := r.headRMSRoPEBatch(short, 0, 2, false); err == nil {
		t.Error("headRMSRoPEBatch accepted a short query buffer")
	}
	if err := r.mixedAttentionPrepareKVBatch(short, short, 0, 2, false); err == nil {
		t.Error("mixedAttentionPrepareKVBatch accepted a short scratch buffer")
	}
}
