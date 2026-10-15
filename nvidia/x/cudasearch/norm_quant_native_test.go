package cudasearch

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strconv"
	"syscall"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// These oracles use analytically chosen binary values, not the implementation's
// half conversion, normalization, quantization or reference-export helpers.
func normQuantSearchFixture(round int) (NormQuantWorkload, map[string][]float32) {
	w := NormQuantWorkload{Rows: 2, Cols: 64, Input: make([]float32, 128), Weight: make([]float32, 64), Weighted: true}
	weights := []float32{127, 0.5, -0.5, 1.5, -1.5, 0, 0, 0}
	quants := []int{127, 1, -1, 2, -2, 0, 0, 0}
	input, norm := float32(1), float32(1)
	header := []float32{0, 60, 240, 95} // d=1, s=508: 0x3c00, 0x5ff0.
	shift := 0
	if round == 1 {
		input, norm, w.Epsilon, shift = 2, 0.5, 12, 1
		header = []float32{0, 56, 240, 91} // d=1/2, s=254: 0x3800, 0x5bf0.
	}
	if round == 2 {
		w.Weighted, w.MaterializeNorm, w.Epsilon = false, true, 3
		norm = 0.5
		header = []float32{8, 28, 0, 76} // d=half(0.5/127), s=16.
	}
	for i := range w.Input {
		w.Input[i] = input
	}
	for i := range w.Weight {
		w.Weight[i] = weights[(i+shift)%len(weights)]
	}
	packed := make([]float32, 144)
	for b := 0; b < 4; b++ {
		copy(packed[b*36:], header)
		for i := 0; i < 32; i++ {
			q := 127
			if w.Weighted {
				q = quants[(i+shift)%len(quants)]
			}
			if q < 0 {
				q += 256
			}
			packed[b*36+4+i] = float32(q)
		}
	}
	ref := map[string][]float32{"q8_bytes": packed}
	if w.MaterializeNorm {
		v := make([]float32, 128)
		for i := range v {
			v[i] = norm
		}
		ref["norm"] = v
	}
	return w, ref
}

func TestNormQuantSearchFixtures(t *testing.T) {
	keys := make(map[string]bool)
	for round := 0; round < 3; round++ {
		w, ref := normQuantSearchFixture(round)
		c, err := NewNormQuantCompiler(&cuda.Device{}, "sm_121a", w, func() bool { return false })
		if err != nil {
			t.Fatal(err)
		}
		if _, err := (OutputPolicy{}).compare(c.References(), ref); err != nil {
			t.Fatalf("round %d: %v", round, err)
		}
		if keys[c.Bindings().Key()] {
			t.Fatal("fixture identities collide")
		}
		keys[c.Bindings().Key()] = true
		bad := c.References()
		bad["q8_bytes"][143]++
		if _, err := (OutputPolicy{}).compare(bad, ref); err == nil {
			t.Fatal("accepted corrupt last packed byte")
		}
		if w.MaterializeNorm {
			bad = c.References()
			delete(bad, "norm")
			if _, err := (OutputPolicy{}).compare(bad, ref); err == nil {
				t.Fatal("accepted missing secondary output")
			}
		}
	}
}

type normQuantExtractionCost struct{}

func (normQuantExtractionCost) NodeCost(n egraph.ENode, _ []float64) float64 {
	if n.Op == "norm_quant_fused" {
		return 1
	}
	return 2
}

func TestNormQuantSearchEnumeration(t *testing.T) {
	e, _, prototype, order := executableFixture(t)
	e.CostModel = normQuantExtractionCost{}
	e.Compile = func(_ context.Context, expr *egraph.Expr, _ WorkloadBindings) (Executable, error) {
		x := *prototype
		x.id, x.order = expr.Op, order
		if expr.Op == "norm_quant_unfused" {
			x.duration = 10 * time.Microsecond
		}
		return &x, nil
	}
	g := egraph.New()
	root := g.Add(egraph.ENode{Op: "norm_quant_unfused"})
	s := egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
	beam := NewBeamSearch(BeamConfig{BeamWidth: 1, Budget: SearchBudget{MaxStates: 2, MaxDepth: 1, MaxActions: 1, MaxEvaluations: 2, MaxMemoryBytes: 1 << 20, CallbackMemoryBytes: fixtureEvaluatorMemoryBytes(e)}})
	result, err := beam.Search(context.Background(), s, []*egraph.Rewrite{egraph.MustRewrite("bound-norm-quant", "norm_quant_unfused", "norm_quant_fused")}, e)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.AllCandidates) != 2 || result.Winner == nil || result.Winner.Expr.Op != "norm_quant_fused" {
		t.Fatalf("enumeration failed: %+v", result)
	}
}

// This gate performs 42 or 45 physical kernel launches: unfused A/A, fused
// screening, held-out winner confirmation and an unfused secondary-consumer
// case. Event durations are observations; this test does not promote a winner
// or assert speedup. All kernels and buffers belong to the concrete adapter.
func TestNativeNormQuantSearch(t *testing.T) {
	if os.Getenv("CUDASEARCH_NORM_QUANT_NATIVE") != "1" {
		t.Skip("CUDASEARCH_NORM_QUANT_NATIVE=1 requires scoped hardware authorization")
	}
	target := os.Getenv("CUDASEARCH_TARGET")
	if target != "sm_100a" && target != "sm_121a" {
		t.Fatal("CUDASEARCH_TARGET must be sm_100a or sm_121a")
	}
	ordinal, err := strconv.Atoi(os.Getenv("CUDASEARCH_DEVICE"))
	if err != nil || ordinal < 0 {
		t.Fatal("CUDASEARCH_DEVICE must name an explicit ordinal")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	lease, err := cudalock.Acquire(ctx, cudalock.WithLabel(t.Name()))
	if err != nil {
		t.Fatal(err)
	}
	owner := lease.Owner()
	check := func() bool {
		data, err := os.ReadFile(filepath.Join(lease.Dir(), "owner"))
		if err != nil || !bytes.Equal(data, []byte(owner.String())) {
			return false
		}
		p, err := os.FindProcess(owner.PID)
		return err == nil && p.Signal(syscall.Signal(0)) == nil
	}
	var dev *cuda.Device
	var owners []Executable
	var evaluators []*ExecutableEvaluator
	t.Cleanup(func() {
		for _, e := range evaluators {
			if err := e.Close(); err != nil {
				t.Errorf("retained evaluator: %v", err)
				return
			}
		}
		for _, x := range owners {
			if err := x.Close(); err != nil {
				t.Errorf("retained executable: %v", err)
				return
			}
		}
		if dev != nil {
			if err := dev.Close(); err != nil {
				t.Error(err)
				return
			}
		}
		if err := lease.Release(); err != nil {
			t.Error(err)
		}
	})
	dev, err = cuda.NewDevice(ordinal)
	if err != nil {
		t.Fatal(err)
	}
	newEvaluator := func(round int) *ExecutableEvaluator {
		w, reference := normQuantSearchFixture(round)
		compiler, err := NewNormQuantCompiler(dev, target, w, check)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := (OutputPolicy{}).compare(compiler.References(), reference); err != nil {
			t.Fatal(err)
		}
		baseline, err := compiler.Compile(ctx, &egraph.Expr{Op: "norm_quant_unfused"}, compiler.Bindings())
		if baseline != nil {
			owners = append(owners, baseline)
		}
		if err != nil {
			t.Fatal(err)
		}
		e := &ExecutableEvaluator{Generation: 1, Workload: compiler.Bindings(), Compile: compiler.Compile, Baseline: baseline, References: reference, Policy: OutputPolicy{}, Rounds: 2, TimingKind: cudatrial.TimingGPUBusy, HardwareLockChecker: check, CostModel: normQuantExtractionCost{}}
		evaluators = append(evaluators, e)
		return e
	}
	e := newEvaluator(0)
	g := egraph.New()
	root := g.Add(egraph.ENode{Op: "norm_quant_unfused"})
	s := egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
	beam := NewBeamSearch(BeamConfig{EvaluatorID: "norm-quant-native-v1", BeamWidth: 1, Budget: SearchBudget{MaxStates: 2, MaxDepth: 1, MaxActions: 1, MaxEvaluations: 2, MaxMemoryBytes: 1 << 20, CallbackMemoryBytes: fixtureEvaluatorMemoryBytes(e)}})
	result, err := beam.Search(ctx, s, []*egraph.Rewrite{egraph.MustRewrite("bound-norm-quant", "norm_quant_unfused", "norm_quant_fused")}, e)
	if err != nil {
		t.Fatal(err)
	}
	if result.Winner == nil || len(result.AllCandidates) != 2 || !result.WorkAccounted || result.Compilations != 2 {
		t.Fatalf("incomplete search: %+v", result)
	}
	for _, c := range result.AllCandidates {
		t.Logf("candidate=%s preparation=%+v", c.Expr.Op, c.Preparation)
		if !c.Validated || c.Comparison["q8_bytes"].Elements != 144 {
			t.Fatal("candidate lacks packed-output coverage")
		}
	}
	for i, c := range result.AllCandidates {
		t.Logf("screen attempt=%d receipts=%+v", i, c.Trials)
	}
	for round := 1; round <= 2; round++ {
		held := newEvaluator(round)
		var candidate *CandidatePlan
		if round == 2 {
			// This changes the output contract and expression. It qualifies
			// secondary-output preservation, not the selected winner.
			hg := egraph.New()
			hr := hg.Add(egraph.ENode{Op: "norm_quant_unfused"})
			hs := egraph.NewSearchState(hg, map[string]egraph.Id{"root": hr})
			candidate, err = held.Evaluate(ctx, hs, hr)
		} else {
			candidate, err = held.Confirm(ctx, result.Winner)
		}
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("qualification=%d preparation=%+v", round, candidate.Preparation)
		if !candidate.Validated || candidate.ID == result.Winner.ID || candidate.Comparison["q8_bytes"].Elements != 144 {
			t.Fatal("qualification lacks independent identity or coverage")
		}
		if round == 2 && candidate.Comparison["norm"].Elements != 128 {
			t.Fatal("secondary output not checked")
		}
		if round == 1 {
			proposal, err := result.ProposePromotion(candidate, 1)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("proposal only: %+v", proposal)
		} else if candidate.ConfirmationOf != "" {
			t.Fatal("secondary-output case mislabeled as winner confirmation")
		}
		t.Logf("qualification=%d op=%s comparison=%+v receipts=%+v", round, candidate.Expr.Op, candidate.Comparison, held.Receipts)
	}
	version, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatal(err)
	}
	major, minor, err := nvrtc.Version()
	if err != nil {
		t.Fatal(err)
	}
	uuid, err := dev.Raw().Uuid()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("device=%d uuid=%v target=%s driver=%d nvrtc=%d.%d winner=%s lease=%+v", ordinal, uuid, target, version, major, minor, result.Winner.Expr.Op, owner)
}
