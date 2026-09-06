//go:build linux

package main

import (
	"encoding/binary"
	"math"
	"testing"
)

// Attention reads the half-rounded cache at every position, including the
// zero prefix: the oracle pushes each row through f16 in kv_cache_push_raw
// (ds4.c:13199) before layer_attention_rows_one reads cache->raw_kv. The
// query value 1.0007 is not representable in f16, so this fails if the
// kernel ever attends over the working float KV instead.
func TestCUDARawZeroPrefixPrecision(t *testing.T) {
	for _, tc := range []struct {
		name  string
		query float32
	}{{"value", 0}, {"score", 512}} {
		t.Run(tc.name, func(t *testing.T) {
			r := cudaSynthFoundation(t)
			q, kv := make([]float32, 32768), make([]float32, 512)
			for i := 448; i < 512; i++ {
				kv[i] = 1.0007
			}
			for h := range 64 {
				q[h*512+448] = tc.query
			}
			rounded := func(v float32) float32 { return cudaAttentionQualificationRound(v, 10, -14) }
			score := float64(tc.query) * float64(rounded(kv[448])) / math.Sqrt(512)
			sinkValue := float32(score)
			sinkBytes := make([]byte, 64*4)
			for h := range 64 {
				binary.LittleEndian.PutUint32(sinkBytes[h*4:], math.Float32bits(sinkValue))
			}
			dq, dk := cudaSynthUpload(t, r, q), cudaSynthUpload(t, r, kv)
			sink := cudaSynthTensor(t, r, "sink", 0, []uint64{64}, sinkBytes)
			cache, scores, out := cudaTestAlloc(t, r, 128*512), cudaTestAlloc(t, r, 64*128), cudaTestAlloc(t, r, 66048)
			cudaQualificationMeasure(t, r, func() error { return r.encodeRawAttention(dq, dk, sink, cache, scores, out, 0) })
			got := cudaTestRead(t, out)[33280:]
			weight := 1 / (1 + math.Exp(float64(sinkValue)-score))
			want := make([]float32, 32768)
			for h := range 64 {
				for i := range 512 {
					want[h*512+i] = float32(float64(rounded(kv[i])) * weight)
				}
			}
			if err := cudaQualificationClose(got, want, 2e-5, 2e-5); err != nil {
				t.Fatal(err)
			}
			stored := cudaTestRead(t, cache)[:512]
			for i, v := range kv {
				wantv := rounded(v)
				if stored[i] != wantv {
					t.Fatalf("cache[%d]=%g want %g", i, stored[i], wantv)
				}
			}
		})
	}
}
