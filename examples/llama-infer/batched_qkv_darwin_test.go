//go:build darwin

package main

import (
	"math"
	"testing"
)

func TestBatchedQKVMatchesCPU(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int32{7, 91, 3, 200, 17, 42, 11, 255}
	positions := []int{0, 9, 2, 17, 4, 23, 6, 31}
	for _, batch := range []int{1, 2, 4, 8} {
		for repeat := 0; repeat < 3; repeat++ {
			got, err := RunBatchedQKV(eng, cfg, 0, tokens[:batch], positions[:batch])
			if err != nil {
				t.Fatal(err)
			}
			if got.Dispatches != 5 {
				t.Fatalf("B=%d dispatches=%d, want 5", batch, got.Dispatches)
			}
			if !equalInts(got.Positions, positions[:batch]) {
				t.Fatalf("B=%d positions=%v", batch, got.Positions)
			}
			want := cpuBatchedQKV(eng, cfg, tokens[:batch])
			checkClose(t, "embedding", got.Embedding, want.Embedding, 1e-6)
			checkClose(t, "normalized", got.Normalized, want.Normalized, 2e-4)
			checkClose(t, "q", got.Q, want.Q, 3e-4)
			checkClose(t, "k", got.K, want.K, 3e-4)
			checkClose(t, "v", got.V, want.V, 3e-4)
		}
	}

	// Finished-row compaction is represented by dense active-row slices; row
	// state and output order must follow the compacted caller order exactly.
	activeTokens := []int32{tokens[1], tokens[4], tokens[7]}
	activePositions := []int{positions[1], positions[4], positions[7]}
	got, err := RunBatchedQKV(eng, cfg, 0, activeTokens, activePositions)
	if err != nil {
		t.Fatal(err)
	}
	want := cpuBatchedQKV(eng, cfg, activeTokens)
	if !equalInts(got.Positions, activePositions) {
		t.Fatalf("compacted positions=%v", got.Positions)
	}
	checkClose(t, "compacted q", got.Q, want.Q, 3e-4)
}

func cpuBatchedQKV(eng *ModelEngine, cfg Config, tokens []int32) BatchedQKVResult {
	b := len(tokens)
	qd := cfg.NumHeads * cfg.HeadDim
	kd := cfg.NumKVHeads * cfg.HeadDim
	r := BatchedQKVResult{Embedding: make([]float32, b*cfg.Dim), Normalized: make([]float32, b*cfg.Dim), Q: make([]float32, b*qd), K: make([]float32, b*kd), V: make([]float32, b*kd)}
	embed := eng.embedTable.HostSlice()
	norm := eng.layers[0].InputNorm.HostSlice()
	for row, tok := range tokens {
		x := r.Embedding[row*cfg.Dim : (row+1)*cfg.Dim]
		copy(x, embed[int(tok)*cfg.Dim:(int(tok)+1)*cfg.Dim])
		var sum float64
		for _, v := range x {
			sum += float64(v * v)
		}
		scale := float32(1 / math.Sqrt(sum/float64(cfg.Dim)+float64(cfg.Eps)))
		n := r.Normalized[row*cfg.Dim : (row+1)*cfg.Dim]
		for i := range n {
			n[i] = x[i] * scale * norm[i]
		}
		projectCPU(n, eng.layers[0].WQ.HostSlice(), r.Q[row*qd:(row+1)*qd], cfg.Dim)
		projectCPU(n, eng.layers[0].WK.HostSlice(), r.K[row*kd:(row+1)*kd], cfg.Dim)
		projectCPU(n, eng.layers[0].WV.HostSlice(), r.V[row*kd:(row+1)*kd], cfg.Dim)
	}
	return r
}

func projectCPU(x, w, out []float32, d int) {
	for j := range out {
		var v float32
		for i, xv := range x {
			v += xv * w[j*d+i]
		}
		out[j] = v
	}
}
func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
func checkClose(t *testing.T, name string, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s lengths %d %d", name, len(got), len(want))
	}
	for i := range got {
		d := float32(math.Abs(float64(got[i] - want[i])))
		if d > tol {
			t.Fatalf("%s[%d]=%g want %g diff %g", name, i, got[i], want[i], d)
		}
	}
}
