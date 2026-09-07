//go:build linux && !tpu

package main

import (
	"math"
	"os"
	"testing"
)

// The physical gate deliberately requires the pinned checkpoint. Tiny random
// weights alone cannot establish that prompt batching preserves model state.
func TestCUDABatchedPrefill(t *testing.T) {
	path := os.Getenv("LLAMA_MODEL_PATH")
	if os.Getenv("LLAMA_CUDA_PREFILL_TEST") != "1" || path == "" {
		t.Skip("set LLAMA_CUDA_PREFILL_TEST=1 and LLAMA_MODEL_PATH for CUDA validation")
	}
	cfg := Llama3_2_1BConfig(path)
	cfg.Precision = "f32"
	cfg.MaxSeqLen = 80
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	requireCUDAAttentionCheckpoint(t, path, &eng.mCfg)
	seed := []int32{128000, 2028, 374, 264, 1296, 315, 18328, 13}
	closeFloats := func(t *testing.T, name string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s length differs", name)
		}
		var maxDiff float64
		for i, x := range got {
			y := want[i]
			d := math.Abs(float64(x) - float64(y))
			maxDiff = max(maxDiff, d)
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || math.IsNaN(float64(y)) || math.IsInf(float64(y), 0) || d > 1e-4+1e-4*math.Abs(float64(y)) {
				t.Fatalf("%s[%d]=%g want %g, diff=%g", name, i, x, y, d)
			}
		}
		t.Logf("%s count=%d max_abs=%g", name, len(got), maxDiff)
	}
	for _, tc := range []struct {
		name         string
		prefix, rows int
	}{
		{"single", 0, 1}, {"short", 0, 8}, {"tile", 0, 32}, {"tail", 0, 33}, {"prefix", 7, 33},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, err := eng.NewSequenceKV(cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer a.Free()
			b, err := eng.NewSequenceKV(cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer b.Free()
			tokens := make([]int32, tc.prefix+tc.rows)
			for i := range tokens {
				tokens[i] = seed[i%len(seed)]
			}
			wantToken, scalarDispatches, err := gpuSerialPrefillPassKVMeasured(eng, cfg, a, tokens)
			if err != nil {
				t.Fatal(err)
			}
			wantLogits, err := eng.ReadLogits()
			if err != nil {
				t.Fatal(err)
			}
			for i, token := range tokens[:tc.prefix] {
				_, err = GPUForwardPassKV(eng, cfg, b, token, i)
				if err != nil {
					t.Fatal(err)
				}
			}
			gotToken, dispatches, err := gpuBatchedPrefillPassKVMeasured(eng, b, tokens[tc.prefix:], tc.prefix)
			if err != nil {
				t.Fatal(err)
			}
			if gotToken != wantToken {
				t.Fatalf("token=%d want%d", gotToken, wantToken)
			}
			gotLogits, err := eng.ReadLogits()
			if err != nil {
				t.Fatal(err)
			}
			closeFloats(t, "logits", gotLogits, wantLogits)
			// Compare all populated positions of every layer, including the prefix.
			dim := eng.mCfg.NumKeyValueHeads * eng.mCfg.HeadDim()
			for layer := 0; layer < eng.mCfg.NumHiddenLayers; layer++ {
				x, err := a.Cache.Layer(layer)
				if err != nil {
					t.Fatal(err)
				}
				y, err := b.Cache.Layer(layer)
				if err != nil {
					t.Fatal(err)
				}
				n := len(tokens) * dim
				left := make([]float32, n)
				right := make([]float32, n)
				if err := x.K.Slice(0, n).CopyToHost(left); err != nil {
					t.Fatal(err)
				}
				if err := y.K.Slice(0, n).CopyToHost(right); err != nil {
					t.Fatal(err)
				}
				closeFloats(t, "keys", right, left)
				if err := x.V.Slice(0, n).CopyToHost(left); err != nil {
					t.Fatal(err)
				}
				if err := y.V.Slice(0, n).CopyToHost(right); err != nil {
					t.Fatal(err)
				}
				closeFloats(t, "values", right, left)
			}
			wantNext, err := GPUForwardPassKV(eng, cfg, a, wantToken, len(tokens))
			if err != nil {
				t.Fatal(err)
			}
			wantNextLogits, err := eng.ReadLogits()
			if err != nil {
				t.Fatal(err)
			}
			gotNext, err := GPUForwardPassKV(eng, cfg, b, gotToken, len(tokens))
			if err != nil {
				t.Fatal(err)
			}
			gotNextLogits, err := eng.ReadLogits()
			if err != nil {
				t.Fatal(err)
			}
			if gotNext != wantNext {
				t.Fatalf("decode token=%d want%d", gotNext, wantNext)
			}
			closeFloats(t, "decode logits", gotNextLogits, wantNextLogits)
			if dispatches == 0 {
				t.Fatal("no physical launches recorded")
			}
			if tc.prefix == 0 && tc.rows > 1 && dispatches >= scalarDispatches {
				t.Fatalf("batched dispatches=%d, scalar=%d", dispatches, scalarDispatches)
			}
			t.Logf("prefill rows=%d prefix=%d dispatches=%d scalar=%d", tc.rows, tc.prefix, dispatches, scalarDispatches)
		})
	}
	// Invalid input is rejected before changing a valid cache or allocating work.
	for _, tc := range []struct {
		name   string
		tokens []int32
		start  int
	}{
		{"negative offset", []int32{1}, -1}, {"overflow", []int32{1, 2}, 79}, {"negative token", []int32{-1}, 0}, {"invalid token", []int32{int32(eng.mCfg.VocabSize)}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := GPUBatchedPrefillPassKV(eng, cfg, nil, tc.tokens, tc.start); err == nil {
				t.Fatal("invalid input accepted")
			}
		})
	}
}
