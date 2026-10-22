//go:build darwin && arm64

package main

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"math"
	"os"
	"os/exec"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

// This exercises the actual scalar verifier and accepted-prefix replay. It is
// not a grouped Metal forward or a speculative performance qualification.
func TestMetalSpeculativeStateGate(t *testing.T) {
	if os.Getenv("DS4_METAL_SPECULATIVE_GATE") != "1" {
		t.Skip("opt-in admitted-model gate required")
	}
	t.Setenv("DS4_METAL_SPECULATIVE_GROUPED", "0")
	metalSpeculativeStateGate(t, []int{2}, false)
}

func TestMetalSpeculativeGroupedStateGate(t *testing.T) {
	if os.Getenv("DS4_METAL_SPECULATIVE_GROUPED_GATE") != "1" {
		t.Skip("opt-in admitted-model gate required")
	}
	t.Setenv("DS4_METAL_SPECULATIVE_GROUPED", "1")
	metalSpeculativeStateGate(t, []int{2, 8}, true)
}

func metalSpeculativeStateGate(t *testing.T, counts []int, grouped bool) {
	t.Helper()
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	path := os.Getenv("DS4_GGUF_PROFILE_MODEL")
	if path == "" {
		t.Fatal("registered model required")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	info, err := f.Stat()
	if err != nil {
		f.Close()
		t.Fatal(err)
	}
	_, err = validatePinnedModel(f, info.Size())
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	metrics, err := exec.Command("/usr/bin/vm_stat").Output()
	if err != nil {
		t.Fatal(err)
	}
	available, err := metalStateGateAvailableMemory(string(metrics))
	if err != nil {
		t.Fatal(err)
	}
	const reserve = uint64(16 << 30)
	if uint64(info.Size()) > available || available-uint64(info.Size()) < reserve {
		t.Fatalf("memory admission: available=%d model=%d reserve=%d", available, info.Size(), reserve)
	}
	r, err := newMetalRuntime(path, 128)
	if err != nil {
		t.Fatal(err)
	}
	a := &metalStateAdapter{runtime: r}
	t.Logf("metal_device=%q recommended_working_set_bytes=%d memory_admission_available_bytes=%d", r.dev.Name(), r.dev.Raw().RecommendedMaxWorkingSetSize(), available)
	t.Cleanup(func() {
		if err := a.Close(); err != nil {
			t.Errorf("backend close: %v", err)
		}
	})
	newState := func() *metalRequestState {
		s, err := a.NewState(128)
		if err != nil {
			t.Fatal(err)
		}
		state := s.(*metalRequestState)
		t.Cleanup(func() {
			if err := state.Close(); err != nil {
				t.Errorf("request close: %v", err)
			}
		})
		return state
	}
	baseline, candidate := newState(), newState()
	save := func(state *metalRequestState) []byte {
		var b bytes.Buffer
		if err := state.Save(&b); err != nil {
			t.Fatal(err)
		}
		return b.Bytes()
	}
	prefixChecks, topChecks := 0, 0
	for _, count := range counts {
		drafts := make([]int, count)
		for i := range drafts {
			drafts[i] = 3 + i
		}
		for accepted := 0; accepted <= len(drafts); accepted++ {
			for _, state := range []*metalRequestState{baseline, candidate} {
				if err := state.Reset(); err != nil {
					t.Fatal(err)
				}
				for pos, token := range []int{1, 2} {
					if err := a.Eval(state, token, pos); err != nil {
						t.Fatal(err)
					}
				}
			}
			prefix := save(candidate)
			if !bytes.Equal(prefix, save(baseline)) {
				t.Fatal("independent prefix mismatch")
			}
			wantTops := make([]int, len(drafts))
			wantRows := make([][]float32, len(drafts))
			for i, token := range drafts {
				values := a.Logits(baseline)
				if len(values) != Flash.Vocab {
					t.Fatal("baseline logits shape")
				}
				if err := checkFiniteLogits("metal scalar verifier baseline", values); err != nil {
					t.Fatal(err)
				}
				best := 0
				for j := 1; j < len(values); j++ {
					if values[j] > values[best] {
						best = j
					}
				}
				wantTops[i] = best
				if err := a.Eval(baseline, token, 2+i); err != nil {
					t.Fatal(err)
				}
				wantRows[i] = append([]float32(nil), a.Logits(baseline)...)
			}
			if grouped && accepted == 0 {
				rows, err := candidate.runtime.evalGrouped(drafts, 2)
				if err != nil {
					t.Fatal(err)
				}
				if len(rows) != len(wantRows) || candidate.runtime.lastBatch.Commits != 2 || candidate.runtime.lastBatch.Waits != 2 {
					t.Fatal("grouped route row count or submission contract")
				}
				for row := range rows {
					if len(rows[row]) != Flash.Vocab || len(wantRows[row]) != Flash.Vocab {
						t.Fatal("grouped readout logits shape")
					}
					for column := range rows[row] {
						if math.Float32bits(rows[row][column]) != math.Float32bits(wantRows[row][column]) {
							t.Fatalf("grouped row %d column %d bit mismatch", row, column)
						}
					}
				}
				if !bytes.Equal(save(candidate), save(baseline)) {
					t.Fatal("grouped terminal serialized state mismatch")
				}
				if err := candidate.Load(bytes.NewReader(prefix)); err != nil {
					t.Fatal(err)
				}
				if count == 2 {
					for phase := range 2 {
						fault := errors.New("grouped post-encoding fault")
						verifier := &metalSpeculativeVerifier{adapter: a, state: candidate, groupedHooks: &metalGroupedExecution{afterEncode: func(i int) error {
							if i == phase {
								return fault
							}
							return nil
						}}}
						if _, err := verifier.VerifyBatch(2, drafts, 2); !errors.Is(err, fault) {
							t.Fatalf("phase %d fault lost: %v", phase, err)
						}
						if !bytes.Equal(prefix, save(candidate)) {
							t.Fatalf("phase %d failure changed frontier", phase)
						}
					}
				}
			}
			if err := baseline.Load(bytes.NewReader(prefix)); err != nil {
				t.Fatal(err)
			}
			for i := 0; i < accepted; i++ {
				if err := a.Eval(baseline, drafts[i], 2+i); err != nil {
					t.Fatal(err)
				}
			}
			v, err := a.NewSpeculativeVerifier(candidate)
			if err != nil {
				t.Fatal(err)
			}
			tops, err := v.VerifyBatch(2, drafts, 2)
			if err != nil {
				t.Fatal(err)
			}
			if len(tops) != len(wantTops) {
				t.Fatal("verification frontier count")
			}
			for i := range tops {
				if tops[i] != wantTops[i] {
					t.Fatalf("frontier %d top1=%d want=%d", i, tops[i], wantTops[i])
				}
			}
			if err := v.CommitPrefix(accepted); err != nil {
				t.Fatal(err)
			}
			prefixChecks++
			topChecks += len(tops)
			want, got := save(baseline), save(candidate)
			if !bytes.Equal(want, got) {
				t.Fatalf("accepted prefix %d serialized state mismatch", accepted)
			}
			if candidate.runtime.pos != 2+accepted {
				t.Fatal("accepted frontier mismatch")
			}
			if _, err := v.VerifyBatch(2, nil, candidate.runtime.pos); err == nil {
				t.Fatal("empty draft admitted")
			}
			if !bytes.Equal(got, save(candidate)) {
				t.Fatal("failed verification changed state")
			}
			t.Logf("drafts=%d accepted=%d serialized_bytes=%d state_sha256=%x", count, accepted, len(got), sha256.Sum256(got))
			if err := a.Eval(candidate, 5, 2+accepted); err != nil {
				t.Fatal(err)
			}
			if err := a.Eval(baseline, 5, 2+accepted); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(save(baseline), save(candidate)) {
				t.Fatalf("accepted prefix %d continuation mismatch", accepted)
			}
		}
	}
	var _ dsv4.SpeculativeVerifier = (*metalSpeculativeVerifier)(nil)
	t.Logf("model_sha256=%s context=128 accepted_prefixes=%d frontier_top1_checks=%d continued_states=%d grouped=%t", pinnedModelSHA256, prefixChecks, topChecks, prefixChecks, grouped)
}
