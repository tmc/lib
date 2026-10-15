package cudasearch

import (
	"bytes"
	"context"
	"math"
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

func gatherSearchFixture(width, round int) ([]float32, []int32, map[string][]float32) {
	data := make([]float32, 3*width)
	for e := 0; e < 3; e++ {
		for c := 0; c < width; c++ {
			data[e*width+c] = float32(e*1000+c) + float32(round)*0.25
		}
	}
	data[0] = math.Float32frombits(0x80000000)
	data[1] = math.Float32frombits(0x7fc01234 + uint32(round))
	routes := []int32{2, 0, 2, 1, 2}
	if round != 0 {
		routes = []int32{0, 1, 0, 2, 0}
	}
	// Derive each expected word from the expert/column specification, without
	// indexing the source matrix or using the production gather/byte helpers.
	want := make([]float32, len(routes)*width*4)
	for t, e := range routes {
		for c := 0; c < width; c++ {
			bits := math.Float32bits(float32(int(e)*1000+c) + float32(round)*0.25)
			if e == 0 && c == 0 {
				bits = 0x80000000
			}
			if e == 0 && c == 1 {
				bits = 0x7fc01234 + uint32(round)
			}
			for b := 0; b < 4; b++ {
				want[(t*width+c)*4+b] = float32((bits >> uint(b*8)) & 255)
			}
		}
	}
	return data, routes, map[string][]float32{"gather_bytes": want}
}

func TestGatherSearchFixtures(t *testing.T) {
	keys := make(map[string]bool)
	for _, width := range []int{32, 33} {
		for round := 0; round < 2; round++ {
			data, routes, ref := gatherSearchFixture(width, round)
			c, err := NewGatherCompiler(&cuda.Device{}, "sm_121a", width, 3, data, routes, func() bool { return false })
			if err != nil {
				t.Fatal(err)
			}
			if _, err := (OutputPolicy{}).compare(c.References(), ref); err != nil {
				t.Fatal(err)
			}
			if keys[c.Bindings().Key()] {
				t.Fatal("held-out identity collision")
			}
			keys[c.Bindings().Key()] = true
			bad := c.References()
			v := bad["gather_bytes"]
			v[len(v)-1]++
			if _, err := (OutputPolicy{}).compare(bad, ref); err == nil {
				t.Fatal("accepted last-byte corruption")
			}
			routes[0], routes[1] = routes[1], routes[0]
			permuted, err := NewGatherCompiler(&cuda.Device{}, "sm_121a", width, 3, data, routes, func() bool { return false })
			if err != nil {
				t.Fatal(err)
			}
			if _, err := (OutputPolicy{}).compare(permuted.References(), ref); err == nil {
				t.Fatal("accepted permuted routes")
			}
		}
	}
}

type gatherExtractionCost struct{}

func (gatherExtractionCost) NodeCost(n egraph.ENode, _ []float64) float64 {
	switch n.Op {
	case "gather32":
		return 1
	case "gather16":
		return 2
	default:
		return 3
	}
}

func gatherSearch(ctx context.Context, eval Evaluator) (*SearchResult, error) {
	g := egraph.New()
	root := g.Add(egraph.ENode{Op: "gather_scalar"})
	s := egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
	beam := NewBeamSearch(BeamConfig{BeamWidth: 2, Budget: SearchBudget{MaxStates: 3, MaxDepth: 1, MaxActions: 2, MaxEvaluations: 3, MaxMemoryBytes: 1 << 20, CallbackMemoryBytes: fixtureEvaluatorMemoryBytes(eval)}})
	return beam.Search(ctx, s, []*egraph.Rewrite{egraph.MustRewrite("bound-gather16", "gather_scalar", "gather16"), egraph.MustRewrite("bound-gather32", "gather_scalar", "gather32")}, eval)
}

func TestGatherSearchEnumeration(t *testing.T) {
	e, _, prototype, order := executableFixture(t)
	e.CostModel = gatherExtractionCost{}
	e.Compile = func(_ context.Context, expr *egraph.Expr, _ WorkloadBindings) (Executable, error) {
		x := *prototype
		x.id, x.order = expr.Op, order
		switch expr.Op {
		case "gather_scalar":
			x.duration = 15 * time.Microsecond
		case "gather16":
			x.duration = 10 * time.Microsecond
		case "gather32":
			x.duration = 5 * time.Microsecond
		}
		return &x, nil
	}
	r, err := gatherSearch(context.Background(), e)
	if err != nil {
		t.Fatal(err)
	}
	if r.Winner == nil || len(r.AllCandidates) != 3 || r.Winner.Expr.Op != "gather32" {
		t.Fatalf("incomplete enumeration: %+v", r)
	}
	seen := make(map[string]bool)
	for _, c := range r.AllCandidates {
		seen[c.Expr.Op] = true
	}
	for _, name := range []string{"gather_scalar", "gather16", "gather32"} {
		if !seen[name] {
			t.Fatalf("missing %s", name)
		}
	}
}

// TestNativeGatherSearch performs 48 kernel launches: for each width32/33,
// scalar A/A, scalar/tile16 and scalar/tile32 ABBA, then held-out confirmation.
// The extraction cost only exposes alternatives; actual event receipts select
// the winner. This fixture does not promote a winner or assert a speedup.
func TestNativeGatherSearch(t *testing.T) {
	if os.Getenv("CUDASEARCH_GATHER_NATIVE") != "1" {
		t.Skip("CUDASEARCH_GATHER_NATIVE=1 requires scoped hardware authorization")
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
	newEvaluator := func(width, round int) *ExecutableEvaluator {
		data, routes, ref := gatherSearchFixture(width, round)
		compiler, err := NewGatherCompiler(dev, target, width, 3, data, routes, check)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := (OutputPolicy{}).compare(compiler.References(), ref); err != nil {
			t.Fatal(err)
		}
		baseline, err := compiler.Compile(ctx, &egraph.Expr{Op: "gather_scalar"}, compiler.Bindings())
		if baseline != nil {
			owners = append(owners, baseline)
		}
		if err != nil {
			t.Fatal(err)
		}
		e := &ExecutableEvaluator{Workload: compiler.Bindings(), Compile: compiler.Compile, Baseline: baseline, References: ref, Policy: OutputPolicy{}, Rounds: 2, TimingKind: cudatrial.TimingGPUBusy, HardwareLockChecker: check, CostModel: gatherExtractionCost{}}
		evaluators = append(evaluators, e)
		return e
	}
	for _, width := range []int{32, 33} {
		e := newEvaluator(width, 0)
		var attempts [][]cudatrial.Receipt
		result, err := gatherSearch(ctx, recordingEvaluator{inner: e, receipts: &attempts})
		if err != nil {
			t.Fatal(err)
		}
		if result.Winner == nil || len(result.AllCandidates) != 3 || len(attempts) != 3 {
			t.Fatalf("incomplete search: %+v", result)
		}
		seen := make(map[string]bool)
		for _, c := range result.AllCandidates {
			t.Logf("width=%d candidate=%s preparation=%+v", width, c.Expr.Op, c.Preparation)
			seen[c.Expr.Op] = true
			if !c.Validated || c.Comparison["gather_bytes"].Elements != 5*width*4 {
				t.Fatal("candidate lacks output coverage")
			}
		}
		for _, name := range []string{"gather_scalar", "gather16", "gather32"} {
			if !seen[name] {
				t.Fatalf("missing %s", name)
			}
		}
		for i, r := range attempts {
			t.Logf("width=%d attempt=%d receipts=%+v", width, i, r)
		}
		held := newEvaluator(width, 1)
		candidate, err := held.Confirm(ctx, result.Winner)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("width=%d held_preparation=%+v", width, candidate.Preparation)
		if !candidate.Validated || candidate.ID == result.Winner.ID || candidate.Comparison["gather_bytes"].Elements != 5*width*4 {
			t.Fatal("held-out confirmation lacks independent identity or coverage")
		}
		t.Logf("width=%d winner=%s screen_ns=%g held_ns=%g held_comparison=%+v receipts=%+v", width, result.Winner.Expr.Op, result.Winner.Cost, candidate.Cost, candidate.Comparison, held.Receipts)
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
	t.Logf("device=%d uuid=%v target=%s driver=%d nvrtc=%d.%d lease=%+v", ordinal, uuid, target, version, major, minor, owner)
}
