//go:build linux && !tpu

package main

import (
	"context"
	"fmt"
	"math"
	"os"
	"testing"
)

// TestRealCheckpointCUDAGQADifferential compares the complete attention output
// from the scalar CUDA forward seam with a CPU reference computed from the same
// query and dense key/value cache. The seam uses the CausalFlashAttentionDecode
// kernel also used by batched serving, but this test does not exercise the
// batched serving address or scratch path. The null control maps query heads to
// key/value heads the wrong way and must not match.
func TestRealCheckpointCUDAGQADifferential(t *testing.T) {
	path := os.Getenv("LLAMA_MODEL_PATH")
	if path == "" {
		t.Skip("LLAMA_MODEL_PATH is empty")
	}
	modelConfig, err := LoadModelConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	requireCUDAAttentionCheckpoint(t, path, modelConfig)
	cfg := Llama3_2_1BConfig(path)
	cfg.Precision = "f32"
	cfg.MaxSeqLen = 8
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	model := &eng.mCfg
	if got := eng.CheckpointReceipt().Verdict; got != "accepted" {
		t.Fatalf("checkpoint receipt verdict = %q", got)
	}
	qHeads := model.NumAttentionHeads
	kvHeads := model.NumKeyValueHeads
	headDim := model.HeadDim()

	if err := eng.ResetKV(); err != nil {
		t.Fatal(err)
	}
	tokens := []int32{41, 42, 43, 44}
	for pos, token := range tokens {
		if _, err := GPUForwardPassKV(eng, cfg, nil, token, pos); err != nil {
			t.Fatal(err)
		}
	}
	if err := eng.stream.Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}
	curPos := len(tokens) - 1

	// The scratchpad holds the last layer's post-rotary query from the last
	// step, so read that layer's cache to match it.
	layer, err := eng.kvCache.Layer(model.NumHiddenLayers - 1)
	if err != nil {
		t.Fatal(err)
	}
	q := make([]float32, qHeads*headDim)
	if err := eng.scratch.Q.CopyToHost(q); err != nil {
		t.Fatal(err)
	}
	kvLen := cfg.MaxSeqLen * kvHeads * headDim
	k := make([]float32, kvLen)
	v := make([]float32, kvLen)
	if err := layer.K.Slice(0, kvLen).CopyToHost(k); err != nil {
		t.Fatal(err)
	}
	if err := layer.V.Slice(0, kvLen).CopyToHost(v); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, qHeads*headDim)
	if err := eng.scratch.AttnOut.Slice(0, len(got)).CopyToHost(got); err != nil {
		t.Fatal(err)
	}

	scale := float32(1 / math.Sqrt(float64(headDim)))
	want := referenceGQAAttention(q, k, v, curPos, kvHeads, qHeads, headDim, scale, false)
	maxAbs, err := maxFiniteAbsDifference(want, got)
	if err != nil {
		t.Fatal(err)
	}
	if maxAbs > 1e-4 {
		t.Fatalf("attention max absolute difference %g exceeds 1e-4", maxAbs)
	}

	swapped := referenceGQAAttention(q, k, v, curPos, kvHeads, qHeads, headDim, scale, true)
	nullAbs, err := maxFiniteAbsDifference(swapped, got)
	if err != nil {
		t.Fatalf("mis-grouped null: %v", err)
	}
	if nullAbs <= 1e-4 {
		t.Fatalf("mis-grouped reference still matched within %g; the differential cannot fail", nullAbs)
	}
	t.Logf("q_heads=%d kv_heads=%d head_dim=%d cur_pos=%d attention_elements=%d max_abs=%g mis_grouped_max_abs=%g",
		qHeads, kvHeads, headDim, curPos, len(got), maxAbs, nullAbs)
}

func maxFiniteAbsDifference(a, b []float32) (float64, error) {
	if len(a) == 0 || len(a) != len(b) {
		return 0, fmt.Errorf("difference length %d against %d", len(a), len(b))
	}
	var maximum float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		if math.IsNaN(x) || math.IsInf(x, 0) || math.IsNaN(y) || math.IsInf(y, 0) {
			return 0, fmt.Errorf("difference element %d is non-finite", i)
		}
		maximum = math.Max(maximum, math.Abs(x-y))
	}
	return maximum, nil
}

func TestMaxFiniteAbsDifferenceRejectsInvalid(t *testing.T) {
	for _, test := range []struct {
		name string
		a, b []float32
	}{
		{"nan reference", []float32{float32(math.NaN())}, []float32{0}},
		{"nan device", []float32{0}, []float32{float32(math.NaN())}},
		{"infinite null", []float32{float32(math.Inf(1))}, []float32{0}},
		{"length", []float32{0}, []float32{0, 1}},
		{"empty", nil, nil},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := maxFiniteAbsDifference(test.a, test.b); err == nil {
				t.Fatal("invalid vectors accepted")
			}
		})
	}
}

// referenceGQAAttention computes causal decode attention on the host. With
// misgroup set it maps query heads to key/value heads by remainder instead of
// by group, the mistake a grouped-query kernel is most likely to make.
func referenceGQAAttention(q, k, v []float32, curPos, kvHeads, qHeads, headDim int, scale float32, misgroup bool) []float32 {
	group := qHeads / kvHeads
	out := make([]float32, qHeads*headDim)
	scores := make([]float64, curPos+1)
	for h := range qHeads {
		kvHead := h / group
		if misgroup {
			kvHead = h % kvHeads
		}
		maximum := math.Inf(-1)
		for t := 0; t <= curPos; t++ {
			var dot float64
			base := (t*kvHeads + kvHead) * headDim
			for d := range headDim {
				dot += float64(q[h*headDim+d]) * float64(k[base+d])
			}
			scores[t] = dot * float64(scale)
			maximum = math.Max(maximum, scores[t])
		}
		var sum float64
		for t := range scores {
			scores[t] = math.Exp(scores[t] - maximum)
			sum += scores[t]
		}
		for d := range headDim {
			var acc float64
			for t := 0; t <= curPos; t++ {
				acc += scores[t] * float64(v[(t*kvHeads+kvHead)*headDim+d])
			}
			if sum > 0 {
				acc /= sum
			}
			out[h*headDim+d] = float32(acc)
		}
	}
	return out
}
