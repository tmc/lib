//go:build linux

package main

import (
	"errors"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func TestCUDAStateAdapterSpeculationRequiresOwnedState(t *testing.T) {
	a := &cudaStateAdapter{runtime: &cudaRuntime{}}
	for name, state := range map[string]dsv4.State{
		"nil":   nil,
		"other": &cudaRequestState{},
	} {
		t.Run(name, func(t *testing.T) {
			verifier, err := a.NewSpeculativeVerifier(state)
			if verifier != nil {
				t.Fatal("NewSpeculativeVerifier returned a verifier for an invalid state")
			}
			if !errors.Is(err, dsv4.ErrUnimplemented) {
				t.Fatalf("NewSpeculativeVerifier error = %v, want ErrUnimplemented", err)
			}
		})
	}
}

func TestCUDASpeculativeArgmaxUsesLowestIndexOnTies(t *testing.T) {
	if got := cudaSpeculativeArgmax([]float32{3, 9, 9, 1}); got != 1 {
		t.Fatalf("argmax = %d, want 1", got)
	}
}

func TestCUDASpeculativeRowTops(t *testing.T) {
	rows := [][]float32{{0, 5, 1}, {9, 0, 1}, {0, 1, 7}}
	tops, err := cudaSpeculativeRowTops(2, rows, 3)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []int{2, 1, 0} {
		if tops[i] != want {
			t.Fatalf("top[%d] = %d, want %d", i, tops[i], want)
		}
	}
	// Dropping a terminal row or returning an empty row must not silently
	// accept an incomplete vocabulary projection.
	for _, bad := range [][][]float32{rows[:2], {rows[0], nil, rows[2]}, {rows[0], {1}, rows[2]}, {rows[0], {0, float32(math.NaN()), 1}, rows[2]}} {
		if _, err := cudaSpeculativeRowTops(2, bad, 3); err == nil {
			t.Fatal("accepted malformed row readout")
		}
	}
}

func TestCUDASpeculativeGroupedAdmission(t *testing.T) {
	for _, value := range []string{"", "0", "true", "1"} {
		t.Setenv("DS4_CUDA_SPECULATIVE_GROUPED", value)
		if got := cudaSpeculativeGroupedEnabled(); got != (value == "1") {
			t.Fatalf("admission %q = %v", value, got)
		}
	}
}

// TestCUDASpeculativeGroupedGate is separate from the scalar rollback gate:
// it actually runs the multi-row transformer and compares every predicted
// token with ordered decode, then checks each accepted-prefix replay.
func TestCUDASpeculativeGroupedGate(t *testing.T) {
	if os.Getenv("DS4_SPECULATIVE_GROUPED_GATE") != "1" {
		t.Skip("DS4_SPECULATIVE_GROUPED_GATE is not set")
	}
	model := os.Getenv("DS4_MODEL")
	if model == "" {
		t.Fatal("DS4_MODEL is required")
	}
	r, err := newCUDARuntime(model, 2048)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	a := &cudaStateAdapter{runtime: r}
	state, err := a.NewState(2048)
	if err != nil {
		t.Fatal(err)
	}
	defer state.Close()
	s := state.(*cudaRequestState)
	if err := a.Eval(s, 0, 0); err != nil {
		t.Fatal(err)
	}
	frontier, err := s.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer frontier.Close()
	for _, count := range []int{2, 8} {
		if err := s.Restore(frontier); err != nil {
			t.Fatal(err)
		}
		drafts := make([]int, count)
		prefixLogits := make([][]float32, count+1)
		prefixLogits[0] = append([]float32(nil), a.Logits(s)...)
		for i := range drafts {
			drafts[i] = cudaSpeculativeArgmax(a.Logits(s))
			if err := a.Eval(s, drafts[i], 1+i); err != nil {
				t.Fatal(err)
			}
			prefixLogits[i+1] = append([]float32(nil), a.Logits(s)...)
		}
		for accepted := 0; accepted <= count; accepted++ {
			if err := s.Restore(frontier); err != nil {
				t.Fatal(err)
			}
			v := &cudaSpeculativeVerifier{adapter: a, state: s, batched: true}
			tops, err := v.VerifyBatch(0, drafts, 1)
			if err != nil {
				t.Fatal(err)
			}
			for i, want := range drafts {
				if tops[i] != want {
					t.Fatalf("batch=%d row=%d top=%d, ordered=%d", count, i, tops[i], want)
				}
			}
			if err := v.CommitPrefix(accepted); err != nil {
				t.Fatal(err)
			}
			for i, want := range prefixLogits[accepted] {
				if got := a.Logits(s)[i]; got != want {
					t.Fatalf("batch=%d prefix=%d logit=%d got=%g want=%g", count, accepted, i, got, want)
				}
			}
			t.Logf("batch=%d accepted_prefix=%d rows=%d replay_logits=%d PASS", count, accepted, len(tops), len(prefixLogits[accepted]))
		}
	}
}

func TestValidateCUDASpeculativeBatch(t *testing.T) {
	tests := []struct {
		name   string
		seed   int
		drafts []int
		pos    int
		want   string
	}{
		{name: "valid", seed: 7, drafts: []int{8, 9}, pos: 12},
		{name: "negative seed", seed: -1, drafts: []int{8}, pos: 12, want: "negative seed"},
		{name: "negative position", seed: 7, drafts: []int{8}, pos: -1, want: "negative position"},
		{name: "negative draft", seed: 7, drafts: []int{8, -1}, pos: 12, want: "negative draft token"},
		{name: "empty", seed: 7, drafts: nil, pos: 12, want: "invalid draft count"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateCUDASpeculativeBatch(test.seed, test.drafts, test.pos)
			if test.want == "" {
				if err != nil {
					t.Fatalf("validate batch: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("validate batch error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestValidateCUDASpeculativeBatchVocab(t *testing.T) {
	tests := []struct {
		name   string
		seed   int
		drafts []int
		vocab  uint64
		want   string
	}{
		{name: "in range", seed: 7, drafts: []int{8, 9}, vocab: 10},
		{name: "seed out of range", seed: 10, drafts: []int{8}, vocab: 10, want: "seed token 10 outside vocabulary"},
		{name: "draft out of range", seed: 7, drafts: []int{8, 10}, vocab: 10, want: "draft token 10 at row 1 outside vocabulary"},
		{name: "empty vocabulary", seed: 0, drafts: []int{1}, want: "empty vocabulary"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateCUDASpeculativeBatchVocab(test.seed, test.drafts, test.vocab)
			if test.want == "" {
				if err != nil {
					t.Fatalf("validate vocabulary: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("validate vocabulary error = %v, want %q", err, test.want)
			}
		})
	}
}

type cudaFixedDrafter []int

func (d cudaFixedDrafter) Draft(int, int, int) ([]int, error) {
	return append([]int(nil), d...), nil
}

// TestCUDAStateAdapterSpeculatorGate exercises the public dsv4 speculator,
// including a partially rejected prefix. It is opt-in because it loads the
// registered model and must run under gpu-lock.
func TestCUDAStateAdapterSpeculatorGate(t *testing.T) {
	if os.Getenv("DS4_STATE_SNAPSHOT_GATE") != "1" {
		t.Skip("DS4_STATE_SNAPSHOT_GATE is not set")
	}
	model := os.Getenv("DS4_MODEL")
	if model == "" {
		t.Fatal("DS4_MODEL is required")
	}
	r, err := newCUDARuntime(model, 2048)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	backend := r.stateBackend()
	state, err := backend.NewState(2048)
	if err != nil {
		t.Fatal(err)
	}
	defer state.Close()
	if err := backend.Eval(state, 0, 0); err != nil {
		t.Fatalf("seed evaluation: %v", err)
	}
	seed, err := state.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer seed.Close()
	first := cudaSpeculativeArgmax(backend.Logits(state))
	if err := backend.Eval(state, first, 1); err != nil {
		t.Fatal(err)
	}
	second := cudaSpeculativeArgmax(backend.Logits(state))
	if err := state.Restore(seed); err != nil {
		t.Fatal(err)
	}
	wrong := (second + 1) % len(backend.Logits(state))
	if wrong == second {
		t.Fatal("could not construct rejected token")
	}
	speculator, err := dsv4.NewSpeculatorForBackend(backend, state, cudaFixedDrafter{first, wrong})
	if err != nil {
		t.Fatal(err)
	}
	result, err := speculator.Verify(0, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if result.Accepted != 1 || result.Committed != 1 || len(result.Tops) != 2 {
		t.Fatalf("speculative result = %+v, want one accepted of two", result)
	}
	if got := len(backend.Logits(state)); got == 0 {
		t.Fatal("partial-prefix commit returned no logits")
	}
	t.Logf("accepted=%d committed=%d tops=%v first=%d second=%d wrong=%d", result.Accepted, result.Committed, result.Tops, first, second, wrong)
}
