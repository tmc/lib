//go:build linux

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"
)

type cudaPrefillGatherState struct {
	Tokens int                `json:"tokens"`
	Step   int                `json:"step"`
	Digest string             `json:"logits_sha256"`
	Counts [cudaLayers][2]int `json:"compressed_counts"`
}

func cudaPrefillGatherSnapshot(t *testing.T, r *cudaRuntime, tokens, step int) cudaPrefillGatherState {
	t.Helper()
	s := cudaPrefillGatherState{Tokens: tokens, Step: step}
	logits := r.Logits()
	if len(logits) != 129280 {
		t.Fatalf("logits length %d, want 129280", len(logits))
	}
	raw := make([]byte, len(logits)*4)
	if e := nonFiniteExtent(logits); e.bad() {
		t.Fatalf("non-finite logits %v", e)
	}
	for i, v := range logits {
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
	}
	s.Digest = fmt.Sprintf("%x", sha256.Sum256(raw))
	for i := range r.state.layers {
		s.Counts[i] = [2]int{r.state.layers[i].attn.n, r.state.layers[i].indexer.n}
	}
	return s
}

// TestCUDAPrefillGather compares gather and per-token expert projections using
// one model mapping. Each prefix starts from Reset and includes continuation.
// The optional receipt permits a separate comparison across group-width builds.
func TestCUDAPrefillGather(t *testing.T) {
	gate := os.Getenv("DS4_CUDA_PREFILL_GATHER_GATE")
	if gate == "" {
		t.Skip("DS4_CUDA_PREFILL_GATHER_GATE is not set")
	}
	if gate != "1" || os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("DS4_CUDA_PREFILL_GATHER_GATE=1 and a GPU lock are required")
	}
	model, input := os.Getenv("DS4_MODEL"), os.Getenv("DS4_CUDA_PREFILL_TOKENS")
	if model == "" || input == "" {
		t.Fatal("DS4_MODEL and DS4_CUDA_PREFILL_TOKENS are required")
	}
	data, err := os.ReadFile(input)
	if err != nil {
		t.Fatal(err)
	}
	tokens, err := parsePrefillTokens(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(tokens) != 2048 {
		t.Fatalf("input contains %d tokens, want 2048", len(tokens))
	}
	r, err := newCUDARuntime(model, 4096)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	}()
	var receipt []cudaPrefillGatherState
	lengths := []int{2, 7, 8, 9, 63, 64, 65, 127, 128, 129, 255, 256, 257, 258, 511, 2048}
	for _, n := range lengths {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			var reference [3]cudaPrefillGatherState
			for arm := range 2 {
				r.disableGatherExperts = arm == 0
				if err := r.Reset(); err != nil {
					t.Fatal(err)
				}
				var hcCalls uint64
				if r.batch != nil && r.batch.hcBatch != nil {
					hcCalls = r.batch.hcBatch.preCalls
				}
				if err := r.EvalBatch(tokens[:n], 0); err != nil {
					t.Fatalf("arm %d prefill: %v", arm, err)
				}
				if os.Getenv("DS4_CUDA_HC_BATCH") == "1" {
					if r.batch == nil || r.batch.hcBatch == nil {
						t.Fatal("HC batch qualification did not allocate the candidate")
					}
					calls := r.batch.hcBatch.preCalls - hcCalls
					if arm == 0 && calls != 0 {
						t.Fatalf("scalar reference dispatched %d HC batch calls", calls)
					}
					if arm == 1 && calls == 0 {
						t.Fatal("candidate arm did not dispatch HC batch")
					}
					t.Logf("arm %d: %d HC batch pre calls", arm, calls)
				}
				for step := range 3 {
					if step != 0 {
						if err := r.Eval([]int{15342, 201}[step-1], n+step-1); err != nil {
							t.Fatalf("arm %d continuation %d: %v", arm, step, err)
						}
					}
					got := cudaPrefillGatherSnapshot(t, r, n, step)
					if n == 2048 && step == 0 && got.Digest != "3c190c73110cec7aed9f22916a1c84216eb7338071040846cc1d0bd992456aa5" {
						t.Fatalf("arm %d registered prefill digest: %s", arm, got.Digest)
					}
					if arm == 0 {
						reference[step] = got
					} else if got != reference[step] {
						t.Fatalf("step %d gather mismatch:\ngot  %+v\nwant %+v", step, got, reference[step])
					}
				}
			}
			receipt = append(receipt, reference[:]...)
		})
	}
	if t.Failed() {
		return
	}
	if path := os.Getenv("DS4_CUDA_PREFILL_GATHER_RECEIPT"); path != "" {
		if len(receipt) != 3*len(lengths) {
			t.Fatalf("refusing partial receipt: %d snapshots, want %d", len(receipt), 3*len(lengths))
		}
		data, err := json.MarshalIndent(receipt, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, append(data, '\n'), 0600); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("group width %d: %d/%d prefix cases matched logits and all layer counts across gather arms", cudaPrefillGroupWidth, len(receipt)/3, len(lengths))
}
