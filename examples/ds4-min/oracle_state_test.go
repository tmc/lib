package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
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

const (
	fullLogitATol = 1e-4
	fullLogitRTol = 1e-4
)

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
	// A port that can write the observer's own format compares against
	// any capture in either direction, which is how one backend's trace
	// gets held against another's rather than only against the oracle's.
	out := os.Getenv("DS4_ORACLE_STATE_OUT")
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
		if out != "" {
			if err := writeOracleState(out, step, len(s.tokens), s.logits); err != nil {
				t.Fatal(err)
			}
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

// compareOracleLogits applies the registered full-vocabulary acceptance
// rule and reports the shape of any disagreement. The rule was registered
// before the final acceptance captures: every finite value must satisfy
// 1e-4 + 1e-4*abs(oracle).
func compareOracleLogits(t *testing.T, step int, state oracleState, got []float32) float64 {
	t.Helper()
	if len(got) != len(state.Logits) {
		t.Errorf("step %d: %d logits, oracle recorded %d", step, len(got), len(state.Logits))
		return 0
	}
	d := compareLogits(state.Logits, got)
	t.Logf("step %d: logits vs oracle: max |diff| %.6g at id %d, mean |diff| %.6g, violations %d/%d",
		step, d.max, d.at, d.mean, d.violations, len(got))
	if d.nonfinite >= 0 {
		t.Errorf("step %d: nonfinite logit at id %d", step, d.nonfinite)
	} else if d.violations != 0 {
		t.Errorf("step %d: %d/%d logits exceed 1e-4 + 1e-4*abs(oracle); first id %d: got %.9g, oracle %.9g, delta %.9g, limit %.9g",
			step, d.violations, len(got), d.first, got[d.first], state.Logits[d.first],
			math.Abs(float64(got[d.first])-float64(state.Logits[d.first])),
			fullLogitATol+fullLogitRTol*math.Abs(float64(state.Logits[d.first])))
	}
	for i, c := range state.Top20[:min(5, len(state.Top20))] {
		rank, gap := rankOf(got, c.ID)
		t.Logf("  oracle rank %d: id %6d logit %.6f; here rank %d, %.6g below the leader", i+1, c.ID, c.Logit, rank+1, gap)
	}
	return d.max
}

type logitDifference struct {
	max        float64
	mean       float64
	at         int
	first      int
	violations int
	nonfinite  int
}

func compareLogits(want, got []float32) logitDifference {
	d := logitDifference{first: -1, nonfinite: -1}
	var sum float64
	for i := range want {
		if !isFinite(want[i]) || !isFinite(got[i]) {
			if d.nonfinite < 0 {
				d.nonfinite = i
			}
			continue
		}
		delta := math.Abs(float64(got[i]) - float64(want[i]))
		sum += delta
		if delta > d.max {
			d.max, d.at = delta, i
		}
		limit := fullLogitATol + fullLogitRTol*math.Abs(float64(want[i]))
		if delta > limit {
			d.violations++
			if d.first < 0 {
				d.first = i
			}
		}
	}
	if len(want) != 0 {
		d.mean = sum / float64(len(want))
	}
	return d
}

func isFinite(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

// writeOracleState records one step in the observer's state-NN.json
// schema. It writes every step the trace visits, not only the steps a
// given capture happens to hold, since the reader keys on the accepted
// count and ignores the rest.
func writeOracleState(dir string, step, position int, logits []float32) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	order := make([]int, len(logits))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return logits[order[a]] > logits[order[b]] })
	top := make([]oracleCandidate, 0, 20)
	for _, id := range order[:min(20, len(order))] {
		top = append(top, oracleCandidate{ID: id, Logit: logits[id]})
	}
	state := oracleState{
		Accepted: step,
		Position: position,
		Vocab:    len(logits),
		Argmax:   order[0],
		Top20:    top,
		Logits:   logits,
	}
	data, err := json.Marshal(state)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, fmt.Sprintf("state-%02d.json", step)), data, 0o644)
}

// TestOracleStateDivergence compares logit captures pairwise. It needs
// no model and no GPU: every input is a directory of state-NN.json,
// which either the observer or DS4_ORACLE_STATE_OUT produced. Set
// DS4_ORACLE_STATE_COMPARE to a comma-separated list of label=dir.
//
// Two implementations of the same deterministic arithmetic should agree
// to rounding. What separates a port that drifts from one that is wrong
// is not the size of the disagreement but its shape, so this reports
// four numbers per pair rather than one:
//
//   - max and mean |diff| say how large the disagreement is.
//   - bias says whether one side runs systematically hot or cold, which
//     a scale or normalization error produces and rounding does not.
//   - correlation, when three or more captures are compared, says
//     whether two ports deviate from a third *together*. A shared
//     defect correlates; independent kernel-level rounding does not.
func TestOracleStateDivergence(t *testing.T) {
	spec := os.Getenv("DS4_ORACLE_STATE_COMPARE")
	if spec == "" {
		t.Skip("no capture set to compare")
	}
	type capture struct {
		label  string
		states map[int]oracleState
	}
	var captures []capture
	for _, entry := range strings.Split(spec, ",") {
		label, dir, ok := strings.Cut(strings.TrimSpace(entry), "=")
		if !ok {
			t.Fatalf("%q is not label=dir", entry)
		}
		states, err := loadOracleStates(dir)
		if err != nil {
			t.Fatal(err)
		}
		captures = append(captures, capture{label, states})
	}
	if len(captures) < 2 {
		t.Fatalf("comparing %d captures needs at least 2", len(captures))
	}
	// Only steps every capture holds can be compared; the observer
	// writes a sparse set and saying so beats silently comparing fewer.
	var steps []int
	for step := range captures[0].states {
		shared := true
		for _, c := range captures[1:] {
			if _, ok := c.states[step]; !ok {
				shared = false
			}
		}
		if shared {
			steps = append(steps, step)
		}
	}
	sort.Ints(steps)
	if len(steps) == 0 {
		t.Fatal("the captures share no step")
	}
	t.Logf("comparing %d captures over shared steps %v", len(captures), steps)
	for _, step := range steps {
		for i, a := range captures {
			for _, b := range captures[i+1:] {
				x, y := a.states[step].Logits, b.states[step].Logits
				if len(x) != len(y) {
					t.Errorf("step %d: %s has %d logits, %s has %d", step, a.label, len(x), b.label, len(y))
					continue
				}
				max, mean, bias, sd := divergence(x, y)
				d := compareLogits(y, x)
				t.Logf("step %d: %s vs %s: max %.4f, mean %.4f, bias %+.4f, sd %.4f, violations %d/%d",
					step, a.label, b.label, max, mean, bias, sd, d.violations, len(x))
				if d.nonfinite >= 0 {
					t.Errorf("step %d: %s vs %s: nonfinite logit at id %d", step, a.label, b.label, d.nonfinite)
				} else if d.violations != 0 {
					t.Errorf("step %d: %s vs %s: %d/%d logits exceed 1e-4 + 1e-4*abs(reference)",
						step, a.label, b.label, d.violations, len(x))
				}
			}
		}
		// With three captures the last one is the reference and the
		// question is whether the other two miss it the same way.
		if len(captures) == 3 {
			ref := captures[2].states[step].Logits
			da := residual(captures[0].states[step].Logits, ref)
			db := residual(captures[1].states[step].Logits, ref)
			t.Logf("step %d: corr(%s-%s, %s-%s) = %+.4f",
				step, captures[0].label, captures[2].label, captures[1].label, captures[2].label, correlation(da, db))
		}
	}
}

// divergence reports the largest and mean absolute difference, the
// signed mean (bias), and the root-mean-square difference.
func divergence(x, y []float32) (max, mean, bias, sd float64) {
	var sumAbs, sumSigned, sumSquare float64
	for i := range x {
		d := float64(x[i]) - float64(y[i])
		if a := math.Abs(d); a > max {
			max = a
		}
		sumAbs += math.Abs(d)
		sumSigned += d
		sumSquare += d * d
	}
	n := float64(len(x))
	return max, sumAbs / n, sumSigned / n, math.Sqrt(sumSquare / n)
}

func residual(x, ref []float32) []float64 {
	d := make([]float64, len(x))
	for i := range x {
		d[i] = float64(x[i]) - float64(ref[i])
	}
	return d
}

func correlation(a, b []float64) float64 {
	n := float64(len(a))
	var ma, mb float64
	for i := range a {
		ma += a[i]
		mb += b[i]
	}
	ma, mb = ma/n, mb/n
	var cov, va, vb float64
	for i := range a {
		x, y := a[i]-ma, b[i]-mb
		cov += x * y
		va += x * x
		vb += y * y
	}
	if va == 0 || vb == 0 {
		return 0
	}
	return cov / math.Sqrt(va*vb)
}

func TestCompareLogits(t *testing.T) {
	tests := []struct {
		name       string
		want       []float32
		got        []float32
		violations int
		nonfinite  int
	}{
		{"equal", []float32{0, 1}, []float32{0, 1}, 0, -1},
		{"near zero within limit", []float32{0}, []float32{0.00009}, 0, -1},
		{"relative term within limit", []float32{1}, []float32{1.00019}, 0, -1},
		{"single violation", []float32{0, 1}, []float32{0.00011, 1}, 1, -1},
		{"nonfinite candidate", []float32{0}, []float32{float32(math.NaN())}, 0, 0},
		{"nonfinite oracle", []float32{float32(math.Inf(1))}, []float32{0}, 0, 0},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			d := compareLogits(test.want, test.got)
			if d.violations != test.violations || d.nonfinite != test.nonfinite {
				t.Fatalf("compareLogits = violations %d, nonfinite %d; want %d, %d",
					d.violations, d.nonfinite, test.violations, test.nonfinite)
			}
		})
	}
}
