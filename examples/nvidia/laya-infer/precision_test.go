package main

// Cross-precision agreement. These need a real model, so they skip unless
// LAYA_MODEL names a directory holding one:
//
//	LAYA_MODEL=$HOME/laya go test -run Precision
//
// The point of the comparison is the continuous output, not the choice. A
// choice that agrees proves nothing on its own: these questions are decided by
// wide margins, and no rounding of a GEMM operand moves an argmax that is not
// close. So the margin is reported alongside every verdict, and the threshold
// is on the probabilities.

import (
	"math"
	"os"
	"testing"
)

// scored is what one precision produced for one request: the rendered reply,
// which is the decision as a caller sees it, and the per-question option
// probabilities behind it at full width.
type scored struct {
	reply string
	probs [][]float64
}

// runPrecision loads the model at prec, answers testdata/bench.json, and
// closes it again before returning. The model is closed eagerly rather than
// through t.Cleanup so that two precisions never hold a context at once.
func runPrecision(t *testing.T, prec precision) scored {
	t.Helper()
	dir := os.Getenv("LAYA_MODEL")
	if dir == "" {
		t.Skip("set LAYA_MODEL to a directory holding model.safetensors")
	}
	m, err := newModel(dir, prec)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := m.Close(); err != nil {
			t.Error(err)
		}
	}()

	req := benchRequest(t)
	reply, err := m.Predict(req)
	if err != nil {
		t.Fatal(err)
	}

	r, err := parseRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	rows, err := m.forward(benchBatch(t, m, req))
	if err != nil {
		t.Fatal(err)
	}
	probs := make([][]float64, len(r.questions))
	for i, q := range r.questions {
		logits, _ := m.score(rows[i], len(q.opts))
		probs[i] = softmax64(logits)
	}
	return scored{reply: string(reply), probs: probs}
}

// compare reports how far want and got drifted apart, and fails if any option
// probability moved by more than tol or if a choice flipped.
func compare(t *testing.T, name string, want, got scored, tol float64) {
	t.Helper()
	if len(want.probs) != len(got.probs) {
		t.Fatalf("%s: %d questions, want %d", name, len(got.probs), len(want.probs))
	}
	var worst, tightest float64 = 0, 1
	for i := range want.probs {
		w, g := want.probs[i], got.probs[i]
		if len(w) != len(g) {
			t.Fatalf("%s: question %d has %d options, want %d", name, i, len(g), len(w))
		}
		top, second, am, gm := 0.0, 0.0, 0, 0
		for j := range w {
			worst = math.Max(worst, math.Abs(float64(w[j]-g[j])))
			if w[j] > top {
				top, second, am = w[j], top, j
			} else if w[j] > second {
				second = w[j]
			}
			if g[j] > g[gm] {
				gm = j
			}
		}
		tightest = math.Min(tightest, top-second)
		if am != gm {
			t.Errorf("%s: question %d chose option %d, want %d", name, i, gm, am)
		}
	}
	t.Logf("%s: largest probability shift %.2e, tightest top-1 margin %.3f, reply %s",
		name, worst, tightest, sameOrNot(want.reply == got.reply))
	if tightest > 0.5 {
		t.Logf("%s: every question is decided by a margin wider than 0.5, so the "+
			"agreeing choices above are not evidence -- the probability shift is", name)
	}
	if worst > tol {
		t.Errorf("%s: largest probability shift %.2e exceeds %.2e", name, worst, tol)
	}
}

func sameOrNot(same bool) string {
	if same {
		return "identical"
	}
	return "differs"
}

// TestPrecisionFastAgreesWithExact measures the drift the default path
// carries, so that a later change to the compute type has a recorded figure to
// be judged against rather than an absolute threshold pulled from nowhere.
func TestPrecisionFastAgreesWithExact(t *testing.T) {
	want := runPrecision(t, exact)
	compare(t, "fast vs exact", want, runPrecision(t, fast), 1e-3)
}
