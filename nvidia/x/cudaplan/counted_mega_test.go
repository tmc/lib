package cudaplan

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

func countedFixture(t *testing.T) *cudair.Snapshot {
	t.Helper()
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{2, 513}, cudair.Float32)
	a := g.AddUnary("neg", x)
	b := g.AddReduce("sum", a, 1)
	y := g.AddUnary("relu", b)
	if err := g.SetOutputs(y); err != nil {
		t.Fatal(err)
	}
	s, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestCountedMegaCompilesActualFP32Graph(t *testing.T) {
	p, err := Compile(countedFixture(t), WithCountedMega(2))
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	source := string(p.megaArtifact.Raw)
	for _, want := range []string{".param .u64 state", "mov.u32 i, %ctaid.x", "add.u32 i, i, 512", "neg.f32 z, x", "add.rn.f64 total, total, widened", "max.NaN.f32 z, x", "membar.gl;\nbar.sync 0;\nmov.u32 thread, %tid.x;", "atom.global.add.u32 old, [counter], 1", "ld.acquire.gpu.u32 count, [counter]", "setp.gt.u32 pred, count, 2", "setp.ge.u32 pred, spins, 16777216", "st.shared.u32 [phase_address], phase", "ld.shared.u32 phase, [phase_address]"} {
		if !strings.Contains(source, want) {
			t.Fatalf("missing actual graph or counted protocol %q", want)
		}
	}
	if len(p.megaLowering.Graph().Tasks()) != 3 || p.steps[0].Backend != BackendMega || len(p.megaArtifact.Schema.Params) != len(p.megaNames)+1 {
		t.Fatal("counted graph lost execution bindings")
	}
	if dir := os.Getenv("CUDAPLAN_COUNTED_PTX_DIR"); dir != "" {
		for _, target := range []string{"sm_80", "sm_90a", "sm_121a"} {
			artifact, _, err := planMegaArtifact(p.snap, target, 2)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(dir, target+".ptx"), artifact.Raw, 0600); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestCountedMegaOutcomeCountsAndAbort(t *testing.T) {
	p, err := Compile(countedFixture(t), WithCountedMega(2))
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	valid := make([]byte, (len(p.megaNames)+1)*4)
	for i, node := range p.snap.Order() {
		if node.Kind != cudair.KindInput {
			binary.LittleEndian.PutUint32(valid[(i+1)*4:], 2)
		}
	}
	for _, tc := range []struct {
		name    string
		word    int
		value   uint32
		wantErr bool
	}{
		{"exact", 0, 0, false}, {"abort", 0, 4, true}, {"poison_abort", 0, 5, true}, {"input_completion", 1, 2, true}, {"missing_producer", 2, 1, true}, {"overpublished", 2, 3, true}, {"overflow_count", 2, math.MaxUint32, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			data := append([]byte(nil), valid...)
			binary.LittleEndian.PutUint32(data[tc.word*4:], tc.value)
			err := p.validateCountedOutcome(data)
			if (err != nil) != tc.wantErr {
				t.Fatalf("outcome %v wantErr %v", err, tc.wantErr)
			}
		})
	}
	if err := p.validateCountedOutcome(valid[:len(valid)-1]); err == nil {
		t.Fatal("truncated outcome accepted")
	}
}

func TestCountedMegaRefusesMissingResidency(t *testing.T) {
	for _, tc := range []struct {
		name        string
		ctas        uint32
		blocks, sm  int32
		cooperative int
		bad         bool
	}{
		{"measured", 2, 1, 2, 1, false}, {"sm_only", 2, 0, 100, 1, true}, {"no_cooperative", 2, 2, 100, 0, true}, {"not_resident", 3, 1, 2, 1, true}, {"overflow", 2, math.MaxInt32, math.MaxInt32, 1, true}, {"zero_sm", 2, 2, 0, 1, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := countedResidency(tc.ctas, tc.blocks, tc.sm, tc.cooperative)
			if (err != nil) != tc.bad {
				t.Fatalf("admission %v bad=%v", err, tc.bad)
			}
		})
	}
}

func TestCountedMegaWorkspaceRefusal(t *testing.T) {
	offset, total, err := countedWorkspace(5, 4, 28)
	if err != nil || offset != 8 || total != 28 {
		t.Fatalf("aligned counter workspace %d %d %v", offset, total, err)
	}
	for _, tc := range []struct{ scratch, names, limit int }{{5, 4, 27}, {math.MaxInt, 4, math.MaxInt}, {0, 257, 4096}, {-1, 4, 100}} {
		if _, _, err := countedWorkspace(tc.scratch, tc.names, tc.limit); !errors.Is(err, ErrWorkspaceBudgetExceeded) {
			t.Fatalf("invalid workspace accepted %+v: %v", tc, err)
		}
	}
}

func TestCountedMegaModeAndTargetRefusal(t *testing.T) {
	for _, opts := range [][]PlanOption{{WithCountedMega(0)}, {WithCountedMega(1)}, {WithCountedMega(1025)}, {WithFiniteMega(true), WithCountedMega(2)}, {WithCountedMega(2), WithFiniteMega(true)}, {WithCountedMega(2), WithGraphReplay(true)}} {
		if p, err := Compile(countedFixture(t), opts...); p != nil || !errors.Is(err, ErrUnsupportedBackend) {
			t.Fatalf("invalid mode accepted %v", err)
		}
	}
	p, err := Compile(countedFixture(t), WithCountedMega(2))
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	for _, target := range []string{"sm_70", "invalid"} {
		if _, _, err := planMegaArtifact(p.snap, target, 2); !errors.Is(err, ErrUnsupportedBackend) {
			t.Fatalf("target %s accepted", target)
		}
	}
}

func TestCountedBackendRequest(t *testing.T) {
	p, err := Compile(countedFixture(t), WithCountedMega(2))
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	r := BackendRequest{Kind: cudair.KindUnary, FiniteMegaRequested: true, FiniteMegaAdmitted: true, Artifact: p.megaArtifact, MegaCTAs: 2}
	d, err := SelectBackend(r)
	if err != nil || d.Backend != BackendMega || !strings.Contains(d.Reason, "2 cooperative CTAs") {
		t.Fatalf("decision %v, %v", d, err)
	}
	for _, n := range []uint32{1, 1025, math.MaxUint32} {
		r.MegaCTAs = n
		if _, err := SelectBackend(r); err == nil {
			t.Fatalf("accepted count %d", n)
		}
	}
	r.MegaCTAs = 2
	r.FiniteMegaRequested = false
	if _, err := SelectBackend(r); err == nil {
		t.Fatal("count without worker request admitted")
	}
}

func ExampleWithCountedMega() {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{513}, cudair.Float32)
	y := g.AddUnary("neg", x)
	if err := g.SetOutputs(y); err != nil {
		fmt.Println(err)
		return
	}
	s, err := g.Snapshot()
	if err != nil {
		fmt.Println(err)
		return
	}
	p, err := Compile(s, WithCountedMega(2))
	if err != nil {
		fmt.Println(err)
		return
	}
	defer p.Close()
	fmt.Println(p.steps[0].Backend)
	// Output: Mega
}

func TestCountedMegaRefusesStripeOverflow(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{int(uint64(math.MaxUint32) - 511 + 1)}, cudair.Float32)
	y := g.AddUnary("neg", x)
	if err := g.SetOutputs(y); err != nil {
		t.Fatal(err)
	}
	s, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := planMegaArtifact(s, "sm_80", 2); err == nil {
		t.Fatal("accepted wrapping striped extent")
	}
}
