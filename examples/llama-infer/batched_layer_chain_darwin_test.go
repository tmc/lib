//go:build darwin

package main

import (
	"math"
	"testing"

	"github.com/tmc/lib/apple/x/metal/mempool"
)

func TestBatchedLayerChainMatchesCPU(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	cfg.MaxSeqLen = 513
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int32{7, 91, 3, 200}
	positions := []int{1, 127, 512, 128}
	rows := []int{3, 0, 2, 1}
	for _, b := range []int{1, 2, 4} {
		for repeat := 0; repeat < 3; repeat++ {
			cache, err := eng.NewBatchedAttentionCache(cfg, 4)
			if err != nil {
				t.Fatal(err)
			}
			fillAttentionCache(cache, cfg)
			want := cpuLayerChain(eng, cfg, cache, tokens[:b], positions[:b], rows[:b])
			got, err := RunBatchedLayerChain(eng, cfg, cache, tokens[:b], positions[:b], rows[:b])
			if err != nil {
				t.Fatal(err)
			}
			if got.Dispatches != 1+12*cfg.NumLayers {
				t.Fatalf("B=%d dispatches=%d", b, got.Dispatches)
			}
			wantBytes := int64(2 * cfg.NumLayers * 4 * cfg.MaxSeqLen * cfg.NumKVHeads * cfg.HeadDim * 4)
			if got.DeviceBytes != wantBytes {
				t.Fatalf("device bytes=%d want %d", got.DeviceBytes, wantBytes)
			}
			checkClose(t, "layer chain", got.Hidden, want, 9e-3)
			cache.Free()
		}
	}
}

func cpuLayerChain(eng *ModelEngine, cfg Config, cache *BatchedAttentionCache, tokens []int32, positions, rows []int) []float32 {
	b, d, h := len(tokens), cfg.Dim, cfg.HiddenDim
	qd, kd := cfg.NumHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
	x := make([]float32, b*d)
	embed := eng.embedTable.HostSlice()
	for r, tok := range tokens {
		copy(x[r*d:(r+1)*d], embed[int(tok)*d:(int(tok)+1)*d])
	}
	for li := range cfg.NumLayers {
		l := &eng.layers[li]
		norm := cpuNormRows(x, l.InputNorm.HostSlice(), b, d, cfg.Eps)
		q, k, v := make([]float32, b*qd), make([]float32, b*kd), make([]float32, b*kd)
		for r := 0; r < b; r++ {
			projectCPU(norm[r*d:(r+1)*d], l.WQ.HostSlice(), q[r*qd:(r+1)*qd], d)
			projectCPU(norm[r*d:(r+1)*d], l.WK.HostSlice(), k[r*kd:(r+1)*kd], d)
			projectCPU(norm[r*d:(r+1)*d], l.WV.HostSlice(), v[r*kd:(r+1)*kd], d)
		}
		view := BatchedAttentionCache{
			K: []mempool.Slice[float32]{cache.K[li]}, V: []mempool.Slice[float32]{cache.V[li]},
			rowCapacity: cache.rowCapacity, maxSeqLen: cache.maxSeqLen, kvDim: cache.kvDim,
		}
		_, _, attn := cpuRowAttention(cfg, &view, q, k, v, positions, rows)
		for r := 0; r < b; r++ {
			z := make([]float32, d)
			projectCPU(attn[r*d:(r+1)*d], l.WO.HostSlice(), z, d)
			for i := range d {
				x[r*d+i] += z[i]
			}
		}
		norm = cpuNormRows(x, l.PostNorm.HostSlice(), b, d, cfg.Eps)
		for r := 0; r < b; r++ {
			gate, up := make([]float32, h), make([]float32, h)
			projectCPU(norm[r*d:(r+1)*d], l.WGate.HostSlice(), gate, d)
			projectCPU(norm[r*d:(r+1)*d], l.WUp.HostSlice(), up, d)
			for i := range h {
				gate[i] = gate[i] / (1 + float32(math.Exp(float64(-gate[i])))) * up[i]
			}
			z := make([]float32, d)
			projectCPU(gate, l.WDown.HostSlice(), z, h)
			for i := range d {
				x[r*d+i] += z[i]
			}
		}
	}
	return x
}

func cpuNormRows(x, w []float32, b, d int, eps float32) []float32 {
	out := make([]float32, len(x))
	for r := 0; r < b; r++ {
		var ss float64
		for _, v := range x[r*d : (r+1)*d] {
			ss += float64(v * v)
		}
		s := float32(1 / math.Sqrt(ss/float64(d)+float64(eps)))
		for i := range d {
			out[r*d+i] = x[r*d+i] * s * w[i]
		}
	}
	return out
}
