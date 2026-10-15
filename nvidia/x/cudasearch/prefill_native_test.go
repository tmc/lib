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
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudaprepack"
	"github.com/tmc/lib/nvidia/x/cudaref"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

func prefillSearchInputs(t *testing.T, round int) ([]uint16, []uint16, map[string][]float32) {
	t.Helper()
	encode := func(v int) uint16 {
		sign := uint16(0)
		if v < 0 {
			sign = 0x8000
			v = -v
		}
		return sign | [...]uint16{0, 0x3c00, 0x4000, 0x4200}[v]
	}
	a, b := make([]float32, 256*32), make([]float32, 32*32)
	ah, bh := make([]uint16, len(a)), make([]uint16, len(b))
	for i := range a {
		v := (i+round)%7 - 3
		a[i] = float32(v)
		ah[i] = encode(v)
	}
	for i := range b {
		v := (3*i+round)%7 - 3
		b[i] = float32(v)
		bh[i] = encode(v)
	}
	g := cudair.NewGraph()
	x := g.AddInput("a", []int{256, 32}, cudair.Float32)
	y := g.AddInput("b", []int{32, 32}, cudair.Float32)
	z := g.AddMatMul(x, y)
	if err := g.SetOutputs(z); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	r, err := cudaref.EvaluateSnapshot(snap, map[string][]float32{"a": a, "b": b})
	if err != nil {
		t.Fatal(err)
	}
	want := r[z.Name]
	if len(want) != 256*32 {
		t.Fatal("missing reference output")
	}
	for row := range 256 {
		for col := range 32 {
			var sum float64
			for p := range 32 {
				sum += float64(a[row*32+p]) * float64(b[p*32+col])
			}
			if want[row*32+col] != float32(sum) {
				t.Fatal("reference mismatch")
			}
		}
	}
	return ah, bh, map[string][]float32{"out": want}
}

type prefillExtractionCost struct{}

func (prefillExtractionCost) NodeCost(n egraph.ENode, _ []float64) float64 {
	if n.Op == "prefill_wmma" {
		return 1
	}
	return 2
}

func TestPrefillSearchFixture(t *testing.T) {
	a, b, ref := prefillSearchInputs(t, 0)
	compiler, err := prefillFixtureCompiler(&cuda.Device{}, "sm_121a", a, b, func() bool { return false })
	if err != nil {
		t.Fatal(err)
	}
	if halfDigest(compiler.a) != halfDigest(a) || halfDigest(compiler.b) != halfDigest(b) || len(compiler.Preparation()) != 4 {
		t.Fatal("preparation changed fixture or lost receipts")
	}
	ha, hb, held := prefillSearchInputs(t, 1)
	if halfDigest(a) == halfDigest(ha) || halfDigest(b) == halfDigest(hb) {
		t.Fatal("held-out inputs unchanged")
	}
	if ref["out"][0] == held["out"][0] {
		t.Fatal("held-out reference unchanged")
	}
	p := OutputPolicy{IgnoreZeroSign: true}
	if _, err := p.compare(ref, ref); err != nil {
		t.Fatal(err)
	}
	ref["out"][len(ref["out"])-1]++
	_, _, fresh := prefillSearchInputs(t, 0)
	if _, err := p.compare(ref, fresh); err == nil {
		t.Fatal("accepted corrupted output")
	}
}

func prefillFixtureCompiler(dev *cuda.Device, target string, a, b []uint16, check func() bool) (*PrefillCompiler, error) {
	af, bf := make([]float32, len(a)), make([]float32, len(b))
	for i, v := range a {
		af[i] = cudaprepack.Float16ToFloat32(v)
	}
	for row := 0; row < 32; row++ {
		for col := 0; col < 32; col++ {
			bf[col*32+row] = cudaprepack.Float16ToFloat32(b[row*32+col])
		}
	}
	return NewPrefillCompilerFloat32(dev, target, 32, 32, af, bf, true, check)
}

func TestPrefillSearchEnumeration(t *testing.T) {
	e, _, prototype, order := executableFixture(t)
	e.CostModel = prefillExtractionCost{}
	e.Compile = func(_ context.Context, expr *egraph.Expr, _ WorkloadBindings) (Executable, error) {
		x := *prototype
		x.id = expr.Op
		x.order = order
		if expr.Op == "prefill_scalar" {
			x.duration = 10 * time.Microsecond
		}
		return &x, nil
	}
	g := egraph.New()
	root := g.Add(egraph.ENode{Op: "prefill_scalar"})
	s := egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
	beam := NewBeamSearch(BeamConfig{BeamWidth: 1, Budget: SearchBudget{MaxStates: 2, MaxDepth: 1, MaxActions: 1, MaxEvaluations: 2, MaxMemoryBytes: 1 << 20, CallbackMemoryBytes: fixtureEvaluatorMemoryBytes(e)}})
	r, err := beam.Search(context.Background(), s, []*egraph.Rewrite{egraph.MustRewrite("bound-prefill-wmma", "prefill_scalar", "prefill_wmma")}, e)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.AllCandidates) != 2 || r.Winner.Expr.Op != "prefill_wmma" {
		t.Fatalf("result=%+v", r)
	}
}

// TestNativePrefillSearch performs 18 launches: six for scalar A/A, six for
// scalar/WMMA ABBA, and six for independent-input confirmation. No promotion
// or maximum-occupancy claim follows from this fixture.
func TestNativePrefillSearch(t *testing.T) {
	if os.Getenv("CUDASEARCH_PREFILL_NATIVE") != "1" {
		t.Skip("CUDASEARCH_PREFILL_NATIVE=1 requires scoped hardware authorization")
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
		process, err := os.FindProcess(owner.PID)
		return err == nil && process.Signal(syscall.Signal(0)) == nil
	}
	var dev *cuda.Device
	var owners []Executable
	var evaluators []*ExecutableEvaluator
	t.Cleanup(func() {
		for _, e := range evaluators {
			if err := e.Close(); err != nil {
				t.Errorf("retained evaluator owner: %v", err)
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
	if !check() {
		t.Fatal("live lease verification failed")
	}
	a, b, reference := prefillSearchInputs(t, 0)
	compiler, err := prefillFixtureCompiler(dev, target, a, b, check)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := compiler.Compile(ctx, &egraph.Expr{Op: "prefill_scalar"}, compiler.Bindings())
	if baseline != nil {
		owners = append(owners, baseline)
	}
	if err != nil {
		t.Fatal(err)
	}
	e := &ExecutableEvaluator{Workload: compiler.Bindings(), Compile: compiler.Compile, Baseline: baseline, References: reference, Policy: OutputPolicy{IgnoreZeroSign: true}, Rounds: 2, TimingKind: cudatrial.TimingSummedKernel, HardwareLockChecker: check, CostModel: prefillExtractionCost{}}
	evaluators = append(evaluators, e)
	var attempts [][]cudatrial.Receipt
	recording := recordingEvaluator{inner: e, receipts: &attempts}
	g := egraph.New()
	root := g.Add(egraph.ENode{Op: "prefill_scalar"})
	s := egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
	rule := egraph.MustRewrite("bound-prefill-wmma", "prefill_scalar", "prefill_wmma")
	beam := NewBeamSearch(BeamConfig{BeamWidth: 1, Budget: SearchBudget{MaxStates: 2, MaxDepth: 1, MaxActions: 1, MaxEvaluations: 2, MaxMemoryBytes: 1 << 20, CallbackMemoryBytes: fixtureEvaluatorMemoryBytes(e)}})
	result, err := beam.Search(ctx, s, []*egraph.Rewrite{rule}, recording)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.AllCandidates) != 2 || len(attempts) != 2 || result.Winner == nil {
		t.Fatalf("incomplete search: %+v", result)
	}
	for _, c := range result.AllCandidates {
		if !c.Validated || c.Comparison["out"].Elements != 256*32 {
			t.Fatal("candidate lacks numerical coverage")
		}
	}
	for i, r := range attempts {
		t.Logf("screening attempt=%d receipts=%+v", i, r)
	}
	ha, hb, heldReference := prefillSearchInputs(t, 1)
	heldCompiler, err := prefillFixtureCompiler(dev, target, ha, hb, check)
	if err != nil {
		t.Fatal(err)
	}
	heldBase, err := heldCompiler.Compile(ctx, &egraph.Expr{Op: "prefill_scalar"}, heldCompiler.Bindings())
	if heldBase != nil {
		owners = append(owners, heldBase)
	}
	if err != nil {
		t.Fatal(err)
	}
	heldEval := &ExecutableEvaluator{Workload: heldCompiler.Bindings(), Compile: heldCompiler.Compile, Baseline: heldBase, References: heldReference, Policy: e.Policy, Rounds: 2, TimingKind: e.TimingKind, HardwareLockChecker: check}
	evaluators = append(evaluators, heldEval)
	held, err := heldEval.Confirm(ctx, result.Winner)
	if err != nil {
		t.Fatal(err)
	}
	if !held.Validated || held.ID == result.Winner.ID {
		t.Fatal("held-out confirmation lacks independent identity")
	}
	if len(result.Winner.Preparation) != 4 || len(held.Preparation) != 4 {
		t.Fatal("missing preparation receipts")
	}
	t.Logf("screen_preparation=%+v held_preparation=%+v", result.Winner.Preparation, held.Preparation)
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
	t.Logf("device=%d uuid=%v target=%s driver=%d nvrtc=%d.%d winner=%s screen_ns=%g held_ns=%g held_comparison=%+v held_receipts=%+v lease=%+v", ordinal, uuid, target, version, major, minor, result.Winner.Expr.Op, result.Winner.Cost, held.Cost, held.Comparison, heldEval.Receipts, owner)
}

type recordingEvaluator struct {
	inner    *ExecutableEvaluator
	receipts *[][]cudatrial.Receipt
}

func (r recordingEvaluator) Evaluate(ctx context.Context, s *egraph.SearchState, id egraph.Id) (*CandidatePlan, error) {
	c, err := r.inner.Evaluate(ctx, s, id)
	*r.receipts = append(*r.receipts, append([]cudatrial.Receipt(nil), r.inner.Receipts...))
	return c, err
}
