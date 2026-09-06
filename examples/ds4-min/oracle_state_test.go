package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// The observer captures, for a handful of decode steps, the full logit
// vector the C oracle produced at that step. Those files are the only
// record of what the reference computed rather than what it chose, so
// they are what separates a port that drifts from one that is wrong: a
// selection mismatch says only that two argmaxes differ, while a logit
// comparison says by how much and where the ranking moved.
//
// The capture that ships with the prompt-A trace is
// evidence/oracle-provenance/capture-prompt-a-01/capture-prompt-a-01,
// which holds state-00, -01, -02, -04 and -08. Point
// DS4_ORACLE_STATE_DIR at a directory of state-NN.json files; NN is the
// number of tokens accepted before that step, so state-05.json is the
// state whose argmax is selection 5.

type oracleCandidate struct {
	ID    int     `json:"id"`
	Logit float32 `json:"logit"`
}

type oracleState struct {
	Accepted int               `json:"accepted_decode_count"`
	Position int               `json:"session_position"`
	Vocab    int               `json:"vocab"`
	Argmax   int               `json:"argmax"`
	Top20    []oracleCandidate `json:"top20"`
	Logits   []float32         `json:"logits"`
}

// loadOracleStates reads every state-NN.json in dir, keyed by NN. A
// capture is allowed to be sparse: the observer writes the steps it was
// asked for, not every step.
func loadOracleStates(dir string) (map[int]oracleState, error) {
	names, err := filepath.Glob(filepath.Join(dir, "state-*.json"))
	if err != nil {
		return nil, err
	}
	states := make(map[int]oracleState, len(names))
	for _, name := range names {
		data, err := os.ReadFile(name)
		if err != nil {
			return nil, err
		}
		var s oracleState
		if err := json.Unmarshal(data, &s); err != nil {
			return nil, fmt.Errorf("%s: %w", filepath.Base(name), err)
		}
		if len(s.Logits) != s.Vocab {
			return nil, fmt.Errorf("%s: %d logits for a vocabulary of %d", filepath.Base(name), len(s.Logits), s.Vocab)
		}
		states[s.Accepted] = s
	}
	if len(states) == 0 {
		return nil, fmt.Errorf("no state-NN.json under %s", dir)
	}
	return states, nil
}

// rankOf reports the position of id in logits, counting from 0, and the
// gap between the leader and id. A rank of 0 with a gap of 0 is a match.
func rankOf(logits []float32, id int) (rank int, gap float64) {
	best := 0
	rank = 0
	for i, v := range logits {
		if v > logits[best] {
			best = i
		}
		if i != id && v > logits[id] {
			rank++
		}
	}
	return rank, float64(logits[best]) - float64(logits[id])
}

// compareOracleTrace drives s through the oracle's own token sequence
// rather than the port's, so a step's logits are computed from the same
// history the oracle had. Without that teacher forcing every step after
// the first mismatch compares two different conversations, and the trace
// reports a cascade where there is one defect.
//
// It reports every step and fails at the end, so one run names all the
// steps that differ instead of only the earliest.
func compareOracleTrace(t *testing.T, s *session, dir string, want []int) {
	t.Helper()
	states, err := loadOracleStates(dir)
	if err != nil {
		t.Fatal(err)
	}
	steps := make([]int, 0, len(states))
	for k := range states {
		steps = append(steps, k)
	}
	sort.Ints(steps)
	t.Logf("oracle states captured at accepted counts %v", steps)

	bad := 0
	worst := 0.0
	margins := make([]float64, 0, len(want))
	for step, oracle := range want {
		got := greedyToken(s.logits)
		rank, gap := rankOf(s.logits, oracle)
		margin := secondGap(s.logits, got)
		margins = append(margins, margin)
		if got == oracle {
			t.Logf("step %d: %d agrees; runner-up trails by %.6g", step, got, margin)
		} else {
			bad++
			t.Errorf("step %d: chose %d, oracle chose %d, which this port ranks %d and scores %.6g below its own leader",
				step, got, oracle, rank+1, gap)
		}
		if state, ok := states[step]; ok {
			if d := compareOracleLogits(t, step, state, s.logits); d > worst {
				worst = d
			}
		}
		if step == len(want)-1 {
			break
		}
		if _, err := s.Accept(oracle); err != nil {
			t.Fatalf("step %d: accept oracle token %d: %v", step, oracle, err)
		}
	}
	// A step where the leader clears the field by more than the port's
	// own logit error agrees for a reason that has nothing to do with
	// parity: the error was there too, and the margin absorbed it.
	// Counting those steps is what stops a mostly-passing trace from
	// reading as mostly-correct.
	exposed := 0
	for _, m := range margins {
		if m < worst {
			exposed++
		}
	}
	t.Logf("logit error reaches %.6g; %d of %d steps have a leader margin below that, so the rest agree by margin rather than by accuracy",
		worst, exposed, len(margins))
	if bad != 0 {
		t.Errorf("%d of %d selections disagree with the oracle under teacher forcing", bad, len(want))
	}
}

// secondGap reports how far the leader is clear of the next candidate,
// which is the margin a drift has to cross to change the selection.
func secondGap(logits []float32, leader int) float64 {
	second := math.Inf(-1)
	for i, v := range logits {
		if i != leader && float64(v) > second {
			second = float64(v)
		}
	}
	return float64(logits[leader]) - second
}

// compareOracleLogits reports the shape of the difference rather than
// asserting a tolerance. No tolerance for a whole-vocabulary logit
// vector has been registered, and inventing one here after seeing the
// numbers would make the comparison unfalsifiable.
func compareOracleLogits(t *testing.T, step int, state oracleState, got []float32) float64 {
	t.Helper()
	if len(got) != len(state.Logits) {
		t.Errorf("step %d: %d logits, oracle recorded %d", step, len(got), len(state.Logits))
		return 0
	}
	worst, at := 0.0, 0
	sum := 0.0
	for i, v := range state.Logits {
		d := math.Abs(float64(got[i]) - float64(v))
		sum += d
		if d > worst {
			worst, at = d, i
		}
	}
	t.Logf("step %d: logits vs oracle: max |diff| %.6g at id %d, mean |diff| %.6g", step, worst, at, sum/float64(len(got)))
	for i, c := range state.Top20[:min(5, len(state.Top20))] {
		rank, gap := rankOf(got, c.ID)
		t.Logf("  oracle rank %d: id %6d logit %.6f; here rank %d, %.6g below the leader", i+1, c.ID, c.Logit, rank+1, gap)
	}
	return worst
}
