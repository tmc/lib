package cudamega_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// The emitted text for each scope, written out here rather than obtained from
// the generator. Asking the generator what it emits for a scope and then
// checking that it emitted it is an identity, and it passes whatever the
// generator does.
var (
	wantPoll = map[cudataskgraph.Scope]string{
		cudataskgraph.ScopeCTA:    "ld.acquire.cta.u32",
		cudataskgraph.ScopeDevice: "ld.acquire.gpu.u32",
		cudataskgraph.ScopeSystem: "ld.acquire.sys.u32",
	}
	wantAcquire = map[cudataskgraph.Scope]string{
		cudataskgraph.ScopeCTA:    "mbarrier.try_wait.parity.acquire.cta",
		cudataskgraph.ScopeDevice: `asm volatile("fence.sc.gpu;" ::: "memory");`,
		cudataskgraph.ScopeSystem: `asm volatile("fence.sc.sys;" ::: "memory");`,
	}
	wantRelease = map[cudataskgraph.Scope]string{
		cudataskgraph.ScopeCTA:    "mbarrier.arrive.release.cta",
		cudataskgraph.ScopeDevice: `asm volatile("fence.sc.gpu;" ::: "memory");`,
		cudataskgraph.ScopeSystem: `asm volatile("fence.sc.sys;" ::: "memory");`,
	}
	orderableScopes = []cudataskgraph.Scope{
		cudataskgraph.ScopeCTA,
		cudataskgraph.ScopeDevice,
		cudataskgraph.ScopeSystem,
	}
)

// scopePair builds the smallest graph with one edge: one producer at ps, one
// consumer at cs.
func scopePair(t *testing.T, ps, cs cudataskgraph.Scope) (cudataskgraph.Graph, map[string]cudamega.Operation) {
	t.Helper()
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "produce", Rank: 0, Bytes: 16, Scope: ps},
		{ID: "consume", Deps: []string{"produce"}, Rank: 1, Bytes: 16, Scope: cs},
	})
	if err != nil {
		t.Fatal(err)
	}
	return g, map[string]cudamega.Operation{
		"produce": {Kind: cudamega.Add, Scalar: 1},
		"consume": {Kind: cudamega.Multiply, Scalar: 2},
	}
}

// TestFenceScopeIsTheMaxOfItsEdge is the S1.3 gate.
//
// Before per-edge fence sites there was one acquire and one release per task,
// both at the task's own declared scope, and so nowhere to put a rule that
// depends on the other end of an edge. The rule is that an edge is ordered at
// the higher of the two scopes it joins: the weaker one is by construction
// insufficient, since a device-scope release has not published to a
// system-scope reader and a device-scope acquire has not ordered a
// system-scope producer's stores. A task's release is then at the highest
// scope any of its consumers needs, which for a task with no consumer is its
// own. A CTA producer cannot be widened for a non-CTA consumer; that pair
// is refused by [TestCTAProducerWiderConsumerRefused].
//
// The grid is one CTA so that CTA scope is admissible on both sides; a
// CTA-scope task with a dependency in a wider grid is refused for a different
// reason, which [TestCTAScopeIsRefusedAcrossCTAs] covers.
func TestFenceScopeIsTheMaxOfItsEdge(t *testing.T) {
	for _, ps := range orderableScopes {
		for _, cs := range orderableScopes {
			if ps == cudataskgraph.ScopeCTA && cs != cudataskgraph.ScopeCTA {
				continue // The separate refusal test covers this pair.
			}
			t.Run(scopeLabel(ps)+"-"+scopeLabel(cs), func(t *testing.T) {
				want := ps
				if cs > want {
					want = cs
				}
				g, ops := scopePair(t, ps, cs)
				grid := cudamega.Grid{CTAs: 1, ThreadsPerCTA: 128}
				p, err := cudamega.NewProgram(g, ops, grid, goldenTarget)
				if err != nil {
					t.Fatal(err)
				}
				steps := p.Steps()
				if got := steps[1].EdgeScope[0]; got != want {
					t.Errorf("edge produce->consume resolved at %s scope, want %s: the max of %s and %s",
						scopeLabel(got), scopeLabel(want), scopeLabel(ps), scopeLabel(cs))
				}
				if got := steps[0].ReleaseScope; got != want {
					t.Errorf("produce releases at %s scope, want %s: its only consumer needs %s",
						scopeLabel(got), scopeLabel(want), scopeLabel(want))
				}
				if got := steps[1].ReleaseScope; got != cs {
					t.Errorf("consume releases at %s scope, want %s: it has no consumer to raise it",
						scopeLabel(got), scopeLabel(cs))
				}

				src, err := cudamega.ProgramSource(g, ops, grid, goldenTarget)
				if err != nil {
					t.Fatal(err)
				}
				if !strings.Contains(src, wantPoll[want]) {
					t.Errorf("the edge is at %s scope but no %s appears in the emitted worker:\n%s",
						scopeLabel(want), wantPoll[want], src)
				}
				if !strings.Contains(src, wantAcquire[want]) {
					t.Errorf("the edge is at %s scope but the consumer does not acquire with %q:\n%s",
						scopeLabel(want), wantAcquire[want], src)
				}
				if !strings.Contains(src, wantRelease[want]) {
					t.Errorf("the producer must release at %s scope, and %q does not appear:\n%s",
						scopeLabel(want), wantRelease[want], src)
				}
				// Exclusivity: nothing weaker than the edge may appear, or
				// the max was computed and then not used.
				for _, s := range orderableScopes {
					if s >= want {
						continue
					}
					if strings.Contains(src, wantPoll[s]) {
						t.Errorf("the edge is at %s scope but the worker polls with %s, which orders less:\n%s",
							scopeLabel(want), wantPoll[s], src)
					}
				}
			})
		}
	}
}

// TestRefusedScopePairIsRefusedAtBothEnds is the other half of S1.3: the
// scope pairs this package cannot order must be refused at construction, with
// text naming the task at fault and the reason the target cannot serve it.
//
// Every pair containing an unsupported scope is refused from whichever end
// declares it. These target refusals are separate from CTA producer widening,
// which is rejected even when both endpoint scopes are individually supported.
func TestRefusedScopePairIsRefusedAtBothEnds(t *testing.T) {
	reasons := map[cudamega.Target]string{
		goldenTarget:  "cluster scope on target sm_121a needs lane 09's participation contract",
		cudamega.SM80: "cluster scope needs barrier.cluster, which target sm_80 does not have",
	}
	// Cluster ranks below device in this ordering, so a cluster end is not
	// the max of its edge and the pair is still refused. A gate that assumed
	// the offending scope was the stronger one would pass here by accident.
	pairs := []struct{ producer, consumer cudataskgraph.Scope }{
		{cudataskgraph.ScopeCluster, cudataskgraph.ScopeDevice},
		{cudataskgraph.ScopeDevice, cudataskgraph.ScopeCluster},
		{cudataskgraph.ScopeCluster, cudataskgraph.ScopeCluster},
		{cudataskgraph.ScopeCTA, cudataskgraph.ScopeCluster},
		{cudataskgraph.ScopeCluster, cudataskgraph.ScopeSystem},
	}
	for target, reason := range reasons {
		for _, pair := range pairs {
			name := fmt.Sprintf("%s/%s-%s", target, scopeLabel(pair.producer), scopeLabel(pair.consumer))
			t.Run(name, func(t *testing.T) {
				g, ops := scopePair(t, pair.producer, pair.consumer)
				_, err := cudamega.NewProgram(g, ops, cudamega.Grid{CTAs: 1, ThreadsPerCTA: 128}, target)
				if err == nil {
					t.Fatalf("accepted an edge this package cannot order: %s over %s",
						scopeLabel(pair.consumer), scopeLabel(pair.producer))
				}
				at := "produce"
				if pair.producer != cudataskgraph.ScopeCluster {
					at = "consume"
				}
				want := fmt.Sprintf("task %q: %s", at, reason)
				if err.Error() != want {
					t.Errorf("refusal is\n\t%v\nwant\n\t%s", err, want)
				}
			})
		}
	}
	// The control: individually supported scopes form admitted pairs except
	// for a CTA producer with a wider consumer, covered by its refusal test.
	for _, ps := range orderableScopes {
		for _, cs := range orderableScopes {
			if ps == cudataskgraph.ScopeCTA && cs != cudataskgraph.ScopeCTA {
				continue
			}
			g, ops := scopePair(t, ps, cs)
			if _, err := cudamega.NewProgram(g, ops, cudamega.Grid{CTAs: 1, ThreadsPerCTA: 128}, goldenTarget); err != nil {
				t.Errorf("refused an orderable pair %s over %s: %v", scopeLabel(cs), scopeLabel(ps), err)
			}
		}
	}
}

// TestPerEdgeWaitCountsTheProducersOwners is the S1.1 gate on emitted text.
//
// The wait that was there before compared against grid_ctas, which is the
// right number only when every CTA performs every task. Under a policy that
// deals a task to a subrange, that task publishes Count completions and never
// grid_ctas of them, so the old comparison is not a slower wait but an
// unsatisfiable one: the consumer would decline the task on every round until
// the spin bound fired.
func TestPerEdgeWaitCountsTheProducersOwners(t *testing.T) {
	f := crossFixture(t)
	src, err := cudamega.ProgramSource(f.graph, f.ops, f.grid, f.target)
	if err != nil {
		t.Fatal(err)
	}
	p, err := cudamega.NewProgram(f.graph, f.ops, f.grid, f.target)
	if err != nil {
		t.Fatal(err)
	}
	steps := p.Steps()

	for i, s := range steps {
		for _, d := range s.Deps {
			want := fmt.Sprintf("if (published < producers_%d)", d)
			if !strings.Contains(src, want) {
				t.Errorf("task %q waits on %q but the worker contains no %q:\n%s",
					s.Task.ID, steps[d].Task.ID, want, src)
			}
			decl := fmt.Sprintf("constexpr unsigned producers_%d = %du;", d, steps[d].Owner.Count())
			if !strings.Contains(src, decl) {
				t.Errorf("task %q owns %s, so %q must appear and does not:\n%s",
					steps[d].Task.ID, steps[d].Owner, decl, src)
			}
			_ = i
		}
	}
	// The count a subrange owner publishes is not the grid, so no wait may be
	// written against the grid.
	for _, ln := range strings.Split(src, "\n") {
		if strings.Contains(ln, "published") && strings.Contains(ln, "grid_ctas") {
			t.Errorf("a readiness test is written against the grid, which a subrange owner never reaches: %s", strings.TrimSpace(ln))
		}
	}
}

// TestSelectionIgnoresBytes is the S1.4 gate.
//
// Task.Bytes is a dynamic shared-memory claim -- see its doc comment in
// cudataskgraph -- and not an output size, a duration, or a cost. A selection
// policy that ranked tasks by it would be dealing CTAs by a number that says
// nothing about the work. So the assignment must be invariant under any change
// to Bytes, which is what this checks: the same graph shape with Bytes scaled
// by a thousand must produce byte-identical owners and byte-identical source.
func TestSelectionIgnoresBytes(t *testing.T) {
	base := []cudataskgraph.Task{
		{ID: "lower", Rank: 0, Bytes: 16, Scope: cudataskgraph.ScopeDevice},
		{ID: "upper", Rank: 0, Bytes: 16, Scope: cudataskgraph.ScopeDevice},
		{ID: "cross", Deps: []string{"upper"}, Rank: 1, Bytes: 16, Scope: cudataskgraph.ScopeDevice},
	}
	ops := map[string]cudamega.Operation{
		"lower": {Kind: cudamega.Add, Scalar: 3},
		"upper": {Kind: cudamega.Add, Scalar: 5},
		"cross": {Kind: cudamega.Multiply, Scalar: 2},
	}
	grid := cudamega.Grid{CTAs: 4, ThreadsPerCTA: 128, Policy: cudamega.CountedPolicy}

	owners := func(tasks []cudataskgraph.Task) ([]cudamega.Owner, string) {
		t.Helper()
		g, err := cudataskgraph.New(tasks)
		if err != nil {
			t.Fatal(err)
		}
		p, err := cudamega.NewProgram(g, ops, grid, goldenTarget)
		if err != nil {
			t.Fatal(err)
		}
		var out []cudamega.Owner
		for _, s := range p.Steps() {
			out = append(out, s.Owner)
		}
		src, err := cudamega.ProgramSource(g, ops, grid, goldenTarget)
		if err != nil {
			t.Fatal(err)
		}
		return out, src
	}

	want, wantSrc := owners(base)
	for _, scale := range []uint64{2, 64, 1000} {
		skewed := append([]cudataskgraph.Task(nil), base...)
		// Only one task's claim moves, so a policy that ordered by Bytes
		// would reorder this rank rather than merely rescale it.
		skewed[1].Bytes = base[1].Bytes * scale
		got, gotSrc := owners(skewed)
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("Bytes of %q scaled by %d moved %q from %s to %s; selection reads the graph's shape, not its shared-memory claims",
					base[1].ID, scale, base[i].ID, want[i], got[i])
			}
		}
		if gotSrc != wantSrc {
			t.Errorf("Bytes of %q scaled by %d changed the emitted worker", base[1].ID, scale)
		}
	}
}
