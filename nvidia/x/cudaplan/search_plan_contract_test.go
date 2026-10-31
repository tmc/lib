package cudaplan_test

import (
	"errors"
	"github.com/tmc/lib/nvidia/driver"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaref"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func TestSelectedPlanPreservesExpression(t *testing.T) {
	expr := affineSelection()
	alternative := &egraph.Expr{Op: "Mul", Children: []*egraph.Expr{{Op: "Add", Children: []*egraph.Expr{{Op: "Input", Data: "x"}, {Op: "Input", Data: "b"}}}, {Op: "Input", Data: "a"}}}
	data := replayInputs(2)
	ids := map[string]bool{}
	for _, tc := range []struct {
		name        string
		expr        *egraph.Expr
		alternative bool
	}{{"multiply_then_add", expr, false}, {"add_then_multiply", alternative, true}} {
		t.Run(tc.name, func(t *testing.T) {
			snap, _, output, id, err := selectedPlan(tc.expr, "sm_121a")
			if err != nil {
				t.Fatal(err)
			}
			if ids[id] {
				t.Fatal("different selected expressions reused identity")
			}
			ids[id] = true
			got, err := cudaref.EvaluateSnapshot(snap, data)
			if err != nil {
				t.Fatal(err)
			}
			want := make([]float32, replayElements)
			for i := range want {
				if tc.alternative {
					want[i] = float32(float32(data["x"][i]+data["b"][i]) * data["a"][i])
				} else {
					want[i] = float32(float32(data["x"][i]*data["a"][i]) + data["b"][i])
				}
			}
			if !sameReplayOutput(got[output], want) {
				t.Fatal("IR ignores selected expression semantics")
			}
			_, _, _, again, err := selectedPlan(tc.expr, "sm_121a")
			if err != nil || again != id {
				t.Fatal("unstable identity")
			}
			_, _, _, other, err := selectedPlan(tc.expr, "sm_100a")
			if err != nil || other == id {
				t.Fatal("target missing from identity")
			}
		})
	}
}

func TestSelectedPlanDomain(t *testing.T) {
	cycle := &egraph.Expr{Op: "Add"}
	cycle.Children = []*egraph.Expr{cycle, {Op: "Input", Data: "x"}}
	wide := affineSelection()
	for range 8 {
		wide = &egraph.Expr{Op: "Add", Children: []*egraph.Expr{wide, {Op: "Input", Data: "b"}}}
	}
	for _, tc := range []struct {
		name string
		expr *egraph.Expr
	}{{"nil", nil}, {"cycle", cycle}, {"too_many", wide}, {"input_only", &egraph.Expr{Op: "Input", Data: "x"}}, {"wrong_input", &egraph.Expr{Op: "Input", Data: "unknown"}}, {"scalar", &egraph.Expr{Op: "Const", Data: float32(1)}}, {"unary", &egraph.Expr{Op: "Add", Children: []*egraph.Expr{{Op: "Input", Data: "x"}}}}, {"payload", &egraph.Expr{Op: "Mul", Data: 1, Children: []*egraph.Expr{{Op: "Input", Data: "x"}, {Op: "Input", Data: "a"}}}}} {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, _, _, err := selectedPlan(tc.expr, "sm_121a"); err == nil {
				t.Fatal("unsupported expression accepted")
			}
		})
	}
	if _, _, _, _, err := selectedPlan(affineSelection(), "unknown"); err == nil {
		t.Fatal("unknown target accepted")
	}
	if x, err := newSelectedExecutable(nil, affineSelection(), "sm_121a", replayInputs(0), nil); err == nil || x != nil {
		t.Fatal("nil device admitted")
	}
}

func TestPendingWitnessClassification(t *testing.T) {
	query := errors.New("query failed")
	for _, tc := range []struct {
		name  string
		ready bool
		err   error
		want  bool
	}{{"pending", false, nil, true}, {"ready", true, nil, false}, {"query_error", false, query, false}, {"ready_error", true, query, false}} {
		t.Run(tc.name, func(t *testing.T) {
			err := requirePendingWitness(tc.ready, tc.err)
			if (err == nil) != tc.want {
				t.Fatalf("pending classification = %v", err)
			}
			if tc.err != nil && !errors.Is(err, tc.err) {
				t.Fatal("query identity lost")
			}
		})
	}
	if pendingDelayTicks > 1<<27 {
		t.Fatal("finite delay bound exceeded")
	}
}

func TestFiniteDelayArtifactContract(t *testing.T) {
	for _, target := range []string{"sm_100a", "sm_121a"} {
		art, err := pendingDelayArtifact(target)
		if err != nil {
			t.Fatal(err)
		}
		if art.EntryPoint() != "delay" || art.Target() != target {
			t.Fatal("delay artifact identity mismatch")
		}
		if _, err := art.Pack(pendingDelayTicks); err != nil {
			t.Fatal(err)
		}
	}
}

func TestSelectedReadbackValidation(t *testing.T) {
	want := make([]float32, replayElements)
	for i := range want {
		want[i] = float32(i-20) / 4
	}
	digest, err := validateSelectedReadback(append([]float32(nil), want...), want)
	if err != nil || len(digest) != 64 {
		t.Fatalf("positive readback: %q %v", digest, err)
	}
	for _, tc := range []struct {
		name   string
		mutate func([]float32) []float32
	}{
		{"short", func(g []float32) []float32 { return g[:len(g)-1] }},
		{"poison", func(g []float32) []float32 {
			for i := range g {
				g[i] = float32(math.NaN())
			}
			return g
		}},
		{"corrupt", func(g []float32) []float32 { g[31]++; return g }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.mutate(append([]float32(nil), want...))
			if _, err := validateSelectedReadback(got, want); err == nil {
				t.Fatal("invalid executed output accepted")
			}
		})
	}
	changed := append([]float32(nil), want...)
	changed[0]++
	next, err := validateSelectedReadback(changed, changed)
	if err != nil || next == digest {
		t.Fatal("digest did not bind actual readback")
	}
}
func TestFixtureRawModuleRestoreFailure(t *testing.T) {
	restore := errors.New("restore failed")
	unloadFailure := errors.New("unload failed")
	for _, tc := range []struct {
		name         string
		inner, outer error
		want         driver.Module
	}{{"success", nil, nil, 0}, {"restore_after_unload", nil, restore, 0}, {"unload_failed", unloadFailure, nil, 7}, {"both_failed", unloadFailure, restore, 7}} {
		t.Run(tc.name, func(t *testing.T) {
			raw := driver.Module(7)
			calls, contexts := 0, 0
			withContext := func(fn func() error) error { contexts++; return errors.Join(fn(), tc.outer) }
			unload := func(m driver.Module) error {
				calls++
				if m != 7 {
					t.Fatalf("dead/foreign handle %v", m)
				}
				return tc.inner
			}
			err := closeFixtureRawModule(&raw, withContext, unload)
			if (tc.inner == nil && tc.outer == nil && err != nil) || raw != tc.want || calls != 1 || !errors.Is(err, tc.inner) && tc.inner != nil || !errors.Is(err, tc.outer) && tc.outer != nil {
				t.Fatalf("state raw=%v calls=%d err=%v", raw, calls, err)
			}
			if tc.want == 0 {
				if err := closeFixtureRawModule(&raw, withContext, unload); err != nil {
					t.Fatal(err)
				}
				if calls != 1 || contexts != 1 {
					t.Fatal("retried already unloaded raw module")
				}
			}
		})
	}
}
