//go:build darwin

package main

import (
	"math"
	"testing"
)

func TestBatchedLayerMatchesCPU(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	cfg.MaxSeqLen = 1024
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int32{7, 91, 3, 200, 17, 42, 11, 255}
	positions := []int{1, 127, 128, 511, 512, 9, 33, 255}
	rows := []int{7, 2, 6, 0, 5, 1, 4, 3}
	for _, b := range []int{1, 2, 4, 8} {
		for repeat := 0; repeat < 3; repeat++ {
			cache, err := eng.NewBatchedAttentionCache(cfg, 8)
			if err != nil {
				t.Fatal(err)
			}
			fillAttentionCache(cache, cfg)
			want := cpuFullLayer(eng, cfg, cache, tokens[:b], positions[:b], rows[:b])
			got, err := RunBatchedLayer(eng, cfg, 0, cache, tokens[:b], positions[:b], rows[:b])
			if err != nil {
				t.Fatal(err)
			}
			if got.Dispatches != 13 {
				t.Fatalf("B=%d dispatches=%d, want 13", b, got.Dispatches)
			}
			if !equalInts(got.Positions, positions[:b]) || !equalInts(got.Rows, rows[:b]) {
				t.Fatalf("mapping changed: positions=%v rows=%v", got.Positions, got.Rows)
			}
			checkClose(t, "layer hidden", got.Hidden, want, 6e-3)
			cache.Free()
		}
	}
}

func cpuFullLayer(eng *ModelEngine, cfg Config, cache *BatchedAttentionCache, tokens []int32, positions, rows []int) []float32 {
	b := len(tokens)
	d, h := cfg.Dim, cfg.HiddenDim
	qkv := cpuBatchedQKV(eng, cfg, tokens)
	_, _, attn := cpuRowAttention(cfg, cache, qkv.Q, qkv.K, qkv.V, positions, rows)
	x := append([]float32(nil), qkv.Embedding...)
	l := &eng.layers[0]
	for r := 0; r < b; r++ {
		tmp := make([]float32, d)
		projectCPU(attn[r*d:(r+1)*d], l.WO.HostSlice(), tmp, d)
		for i := range d {
			x[r*d+i] += tmp[i]
		}
	}
	norm := make([]float32, b*d)
	for r := 0; r < b; r++ {
		var ss float64
		for _, v := range x[r*d : (r+1)*d] {
			ss += float64(v * v)
		}
		s := float32(1 / math.Sqrt(ss/float64(d)+float64(cfg.Eps)))
		for i := range d {
			norm[r*d+i] = x[r*d+i] * s * l.PostNorm.HostSlice()[i]
		}
	}
	for r := 0; r < b; r++ {
		gate, up := make([]float32, h), make([]float32, h)
		projectCPU(norm[r*d:(r+1)*d], l.WGate.HostSlice(), gate, d)
		projectCPU(norm[r*d:(r+1)*d], l.WUp.HostSlice(), up, d)
		for i := range h {
			gate[i] = (gate[i] / (1 + float32(math.Exp(float64(-gate[i]))))) * up[i]
		}
		down := make([]float32, d)
		projectCPU(gate, l.WDown.HostSlice(), down, h)
		for i := range d {
			x[r*d+i] += down[i]
		}
	}
	return x
}
