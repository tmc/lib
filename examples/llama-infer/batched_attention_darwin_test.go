//go:build darwin

package main

import (
	"math"
	"testing"
)

func TestBatchedAttentionMatchesCPU(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	cfg.MaxSeqLen = 1024
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int32{7, 91, 3, 200, 17, 42, 11, 255}
	allPos := []int{1, 127, 128, 511, 512, 9, 33, 255}
	allRows := []int{7, 2, 6, 0, 5, 1, 4, 3}
	for _, b := range []int{1, 2, 4, 8} {
		for repeat := 0; repeat < 3; repeat++ {
			cache, err := eng.NewBatchedAttentionCache(cfg, 8)
			if err != nil {
				t.Fatal(err)
			}
			fillAttentionCache(cache, cfg)
			qkv, err := RunBatchedQKV(eng, cfg, 0, tokens[:b], allPos[:b])
			if err != nil {
				t.Fatal(err)
			}
			wantQ, wantK, wantA := cpuRowAttention(cfg, cache, qkv.Q, qkv.K, qkv.V, allPos[:b], allRows[:b])
			got, err := RunBatchedAttention(eng, cfg, 0, cache, qkv.Q, qkv.K, qkv.V, allPos[:b], allRows[:b])
			if err != nil {
				t.Fatal(err)
			}
			if got.Dispatches != 2 {
				t.Fatalf("B=%d dispatches=%d, want 2", b, got.Dispatches)
			}
			checkClose(t, "rope q", got.Q, wantQ, 5e-4)
			checkClose(t, "rope k", got.K, wantK, 5e-4)
			checkClose(t, "attention", got.Attention, wantA, 2e-3)
			cache.Free()
			cache.Free()
		}
	}
}

func TestBatchedAttentionRejectsInvalidState(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	c, err := eng.NewBatchedAttentionCache(cfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Free()
	qd, kd := cfg.NumHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
	q, k, v := make([]float32, 2*qd), make([]float32, 2*kd), make([]float32, 2*kd)
	for _, tc := range []struct {
		name      string
		pos, rows []int
	}{
		{"capacity", []int{0, 1, 2}, []int{0, 1, 0}}, {"position", []int{0, cfg.MaxSeqLen}, []int{0, 1}}, {"duplicate row", []int{0, 1}, []int{0, 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := RunBatchedAttention(eng, cfg, 0, c, q, k, v, tc.pos, tc.rows)
			if err == nil {
				t.Fatal("expected error")
			}
		})
	}
	c.Free()
	_, err = RunBatchedAttention(eng, cfg, 0, c, q, k, v, []int{0, 1}, []int{0, 1})
	if err == nil {
		t.Fatal("freed cache accepted")
	}
}

func fillAttentionCache(c *BatchedAttentionCache, cfg Config) {
	for l := range c.K {
		kh, vh := c.K[l].HostSlice(), c.V[l].HostSlice()
		for i := range kh {
			kh[i] = float32((i%29)-14) / 97
			vh[i] = float32((i%31)-15) / 89
		}
	}
}

func cpuRowAttention(cfg Config, c *BatchedAttentionCache, q, k, v []float32, pos, rows []int) ([]float32, []float32, []float32) {
	q = append([]float32(nil), q...)
	k = append([]float32(nil), k...)
	qd, kd := cfg.NumHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
	kc := append([]float32(nil), c.K[0].HostSlice()...)
	vc := append([]float32(nil), c.V[0].HostSlice()...)
	out := make([]float32, len(q))
	scale := float32(1 / math.Sqrt(float64(cfg.HeadDim)))
	for b, p := range pos {
		applyRoPE(q[b*qd:(b+1)*qd], p, cfg.NumHeads, cfg.HeadDim, cfg.RopeTheta)
		applyRoPE(k[b*kd:(b+1)*kd], p, cfg.NumKVHeads, cfg.HeadDim, cfg.RopeTheta)
		base := (rows[b]*cfg.MaxSeqLen + p) * kd
		copy(kc[base:base+kd], k[b*kd:(b+1)*kd])
		copy(vc[base:base+kd], v[b*kd:(b+1)*kd])
		for h := 0; h < cfg.NumHeads; h++ {
			kh := h / (cfg.NumHeads / cfg.NumKVHeads)
			scores := make([]float64, p+1)
			max := -math.MaxFloat64
			for x := 0; x <= p; x++ {
				z := float32(0)
				kb := (rows[b]*cfg.MaxSeqLen+x)*kd + kh*cfg.HeadDim
				for d := 0; d < cfg.HeadDim; d++ {
					z += q[b*qd+h*cfg.HeadDim+d] * kc[kb+d]
				}
				scores[x] = float64(z * scale)
				if scores[x] > max {
					max = scores[x]
				}
			}
			sum := 0.0
			for x := range scores {
				scores[x] = math.Exp(scores[x] - max)
				sum += scores[x]
			}
			for d := 0; d < cfg.HeadDim; d++ {
				z := 0.0
				for x := 0; x <= p; x++ {
					vb := (rows[b]*cfg.MaxSeqLen+x)*kd + kh*cfg.HeadDim
					z += scores[x] / sum * float64(vc[vb+d])
				}
				out[b*qd+h*cfg.HeadDim+d] = float32(z)
			}
		}
	}
	return q, k, out
}
