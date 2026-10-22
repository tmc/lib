package kernelagent

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasearch"
)

// TestEachOwnerAttemptArchivedOnce checks that the owner evaluator's observer
// is the archive's only attempt path: one attempt and one outcome per owner
// attempt, for every attempt Search and Confirm admit, and nothing else.
func TestEachOwnerAttemptArchivedOnce(t *testing.T) {
	for _, strategy := range []string{"beam", "mcts"} {
		t.Run(strategy, func(t *testing.T) {
			f, l := setup(t)
			l.cfg.Strategy = strategy
			r := run(t, l, f)
			type key struct {
				phase string
				owner uint64
			}
			attempts := make(map[key]Record)
			outcomes := make(map[key]Record)
			phases := make(map[string]int)
			for _, rec := range f.archive.Records() {
				k := key{rec.Phase, rec.OwnerAttempt}
				switch rec.Kind {
				case "attempt":
					if _, dup := attempts[k]; dup {
						t.Fatalf("duplicate attempt %+v", k)
					}
					attempts[k] = rec
					phases[rec.Phase]++
				case "outcome":
					a, ok := attempts[k]
					if _, dup := outcomes[k]; dup || !ok || a.Sequence != rec.Attempt || a.Evaluation != rec.Evaluation {
						t.Fatalf("outcome %+v does not match one attempt %+v", rec, a)
					}
					outcomes[k] = rec
				}
			}
			if len(outcomes) != len(attempts) {
				t.Fatalf("attempts=%d outcomes=%d", len(attempts), len(outcomes))
			}
			if phases["screen"] != r.Search.Evaluations || phases["held-out"] != 1 || len(phases) != 2 {
				t.Fatalf("phases=%v owner evaluations=%d", phases, r.Search.Evaluations)
			}
			for k, a := range attempts {
				want := cudasearch.EvaluationSearch
				if k.phase == "held-out" {
					want = cudasearch.EvaluationConfirmation
				}
				if a.Evaluation != string(want) || k.owner < 1 || int(k.owner) > phases[k.phase] {
					t.Fatalf("owner attempt %+v kind %q", k, a.Evaluation)
				}
				// Nothing is compiled when the attempt record is written.
				if a.CandidateID != "" || a.StructuralID != "" || a.Compile != "" || a.BaselineID != f.cfg.Baseline.Identity() || a.WorkloadKey == "" {
					t.Fatalf("attempt written after work or without owner facts: %+v", a)
				}
			}
			winner := r.Search.Winner
			a, o := attempts[key{"held-out", 1}], outcomes[key{"held-out", 1}]
			if a.ConfirmationOf != winner.ID || o.ConfirmationOf != winner.ID || o.CandidateID != r.Confirmation.ID || o.Outcome != "qualified" || len(a.Derivation) != len(winner.Derivation) {
				t.Fatalf("confirmation attempt=%+v outcome=%+v", a, o)
			}
		})
	}
}

func TestArchiveStartFailurePreventsCompile(t *testing.T) {
	for _, tc := range []struct {
		name       string
		maxRecords int // epoch and proposal records precede the first attempt
		refused    bool
	}{
		{"refused", 2, true},
		{"admitted", 1000, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f, l := setup(t)
			f.archive.maxRecords = tc.maxRecords
			r, err := l.RunEpoch(context.Background(), "start", f.state, nil, nil)
			if tc.refused {
				if err == nil || r != nil || len(f.compiled) != 0 || len(f.events) != 0 || len(f.archive.Records()) != 2 {
					t.Fatalf("refused start reached work: err=%v compiled=%d events=%v", err, len(f.compiled), f.events)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()
			if len(f.compiled) == 0 || len(f.events) == 0 {
				t.Fatal("control fixture never reached compilation")
			}
		})
	}
}

func TestArchiveOutcomeFailureWithholdsCandidate(t *testing.T) {
	dir := t.TempDir()
	f, err := newFixture(dir)
	if err != nil {
		t.Fatal(err)
	}
	l, err := NewLoop(f.archive, f.cfg)
	if err != nil {
		t.Fatal(err)
	}
	// Epoch, proposal and the first attempt fit; its outcome does not.
	f.archive.maxRecords = 3
	r, err := l.RunEpoch(context.Background(), "outcome", f.state, nil, nil)
	if err == nil || r != nil {
		t.Fatal("unrecorded outcome admitted")
	}
	// The refusal canceled the epoch: one compile, closed; baseline borrowed.
	if len(f.compiled) != 1 || f.openOwners() != 0 || f.cfg.Baseline.(*stubExecutable).closed {
		t.Fatalf("compiled=%d open=%d", len(f.compiled), f.openOwners())
	}
	records := f.archive.Records()
	if last := records[len(records)-1]; len(records) != 3 || last.Kind != "attempt" {
		t.Fatalf("records=%+v", records)
	}
	// Resume quarantines the attempt whose outcome was never durable.
	if err := f.archive.Close(); err != nil {
		t.Fatal(err)
	}
	a, err := OpenArchive(dir, 1<<20, 1000)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	records = a.Records()
	if last := records[len(records)-1]; last.Kind != "outcome" || last.Attempt != 3 || last.Outcome != "quarantined" || !last.FaultCauseUnknown || last.Evaluation != string(cudasearch.EvaluationSearch) {
		t.Fatalf("recovered=%+v", last)
	}
}

func TestResultRetainsWinnerThroughConfirm(t *testing.T) {
	f, l := setup(t)
	r, err := l.RunEpoch(context.Background(), "lifetime", f.state, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Evaluators are closed when RunEpoch returns; their compiled owners are
	// gone, but the winner's source belongs to the search result.
	winner := r.Search.Winner
	if len(l.pending) != 0 || f.openOwners() != 0 || winner.Expr == nil || winner.Expr.Op != "fast" || r.Confirmation == nil || r.Confirmation.ConfirmationOf != winner.ID {
		t.Fatalf("pending=%d open=%d winner=%+v confirmation=%+v", len(l.pending), f.openOwners(), winner, r.Confirmation)
	}
	r.Close()
	r.Close()
	if winner.Expr != nil || r.Search != nil || r.Confirmation != nil || r.Promotion == nil {
		t.Fatalf("closed result=%+v winner=%+v", r, winner)
	}
	resident, err := NewResident(HostStub, 1, PlanBinding{f.cfg.Baseline.Identity(), hostPlan(t)})
	if err != nil {
		t.Fatal(err)
	}
	n := len(f.compiled)
	if c, err := l.Remeasure(context.Background(), "after-close", resident, r); err == nil || c != nil || len(f.compiled) != n {
		t.Fatalf("closed result remeasured: %v", err)
	}
}

func TestFailedEpochClosesOwners(t *testing.T) {
	f, l := setup(t)
	f.heldOutWrong = true
	r, err := l.RunEpoch(context.Background(), "held-out-wrong", f.state, nil, nil)
	if r != nil || !errors.Is(err, cudasearch.ErrVerificationFailed) || len(l.pending) != 0 || f.openOwners() != 0 || f.cfg.Baseline.(*stubExecutable).closed {
		t.Fatalf("result=%v err=%v pending=%d open=%d", r, err, len(l.pending), f.openOwners())
	}
	found := false
	for _, rec := range f.archive.Records() {
		if rec.Kind == "outcome" && rec.Phase == "held-out" {
			found = true
			if rec.Evaluation != string(cudasearch.EvaluationConfirmation) || rec.Verification != "failed" || rec.Outcome != "quarantined" || len(rec.Trials) != 0 {
				t.Fatalf("held-out failure=%+v", rec)
			}
		}
	}
	if !found {
		t.Fatal("missing held-out outcome")
	}

	// The failure follows a recorded search; runEpoch closes the search
	// result it had already filled.
	g, m := setup(t)
	g.heldOutWrong = true
	res := new(Result)
	err = m.runEpoch(context.Background(), "held-out-wrong", g.state, nil, nil, res)
	searched := false
	for _, rec := range g.archive.Records() {
		searched = searched || (rec.Kind == "search" && rec.CandidateID != "")
	}
	if !errors.Is(err, cudasearch.ErrVerificationFailed) || !searched || res.Search != nil || res.Confirmation != nil || res.Promotion != nil {
		t.Fatalf("partial result left open: %+v err=%v searched=%v", res, err, searched)
	}
}

// TestOutcomeRefusalIsOwnerRejection drives the loop's evaluator through the
// owner's public Evaluate: the refused outcome is reported by the owner with
// its wrapped error, and the compiled owner is closed.
func TestOutcomeRefusalIsOwnerRejection(t *testing.T) {
	f, l := setup(t)
	f.archive.maxRecords = 1 // the attempt fits; its outcome does not
	_, cancel := context.WithCancel(context.Background())
	defer cancel()
	e := l.evaluator("direct", "screen", f.cfg.Screen, nil, f.cfg.Generation, cancel)
	candidate, err := e.Evaluate(context.Background(), f.state, f.state.Roots["root"])
	if candidate != nil || err == nil || !strings.Contains(err.Error(), "cudasearch: record evaluation outcome") || !strings.Contains(err.Error(), "record limit") {
		t.Fatalf("candidate=%v err=%v", candidate, err)
	}
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
	if len(f.compiled) != 1 || f.openOwners() != 0 {
		t.Fatalf("compiled=%d open=%d", len(f.compiled), f.openOwners())
	}
}

func TestRunEpochUnderCappedSearchMemory(t *testing.T) {
	for _, tc := range []struct {
		name  string
		limit int64
		ok    bool
	}{
		{"admitted", 64 << 20, true},
		{"refused", 1 << 10, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f, l := setup(t)
			l.cfg.Budget.MaxMemoryBytes = tc.limit
			l.cfg.Budget.CallbackMemoryBytes = 1 << 10
			r, err := l.RunEpoch(context.Background(), "capped", f.state, nil, nil)
			if tc.ok {
				if err != nil {
					t.Fatal(err)
				}
				if r.Search.AccountedBytes <= 0 || r.Promotion == nil {
					t.Fatalf("capped search=%+v", r.Search)
				}
				r.Close()
			} else if err == nil || r != nil {
				t.Fatalf("over-budget epoch admitted: %v", err)
			}
			if len(l.pending) != 0 || f.openOwners() != 0 || f.cfg.Baseline.(*stubExecutable).closed {
				t.Fatalf("pending=%d open=%d", len(l.pending), f.openOwners())
			}
		})
	}
}
