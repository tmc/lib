package cudamega_test

import (
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

func TestTargetCapability(t *testing.T) {
	for _, tc := range []struct {
		target               cudamega.Target
		major, minor         int
		suffix               string
		wantErr              string
		atLeast80, atLeast90 bool
	}{
		{target: cudamega.SM80, major: 8, minor: 0, atLeast80: true},
		{target: cudamega.SM90, major: 9, minor: 0, atLeast80: true, atLeast90: true},
		{target: cudamega.SM121a, major: 12, minor: 1, suffix: "a", atLeast80: true, atLeast90: true},
		{target: "sm_70", major: 7, minor: 0},
		{target: "", wantErr: "not spelled"},
		{target: "sm_", wantErr: "no capability digits"},
		{target: "sm_8", wantErr: "no capability digits"},
		{target: "compute_80", wantErr: "not spelled"},
		{target: "sm_90A", wantErr: "unrecognized suffix"},
	} {
		major, minor, suffix, err := tc.target.Capability()
		if tc.wantErr != "" {
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("%q: error %v, want one naming %q", tc.target, err, tc.wantErr)
			}
			continue
		}
		if err != nil {
			t.Errorf("%q: %v", tc.target, err)
			continue
		}
		if major != tc.major || minor != tc.minor || suffix != tc.suffix {
			t.Errorf("%q: %d.%d suffix %q, want %d.%d suffix %q", tc.target, major, minor, suffix, tc.major, tc.minor, tc.suffix)
		}
	}
}

// TestTargetConditionalScopeRefusal is the refusal that the target input
// bought.
//
// Before the target was an input, cluster scope was refused unconditionally
// and CTA scope was accepted for every target. Both were right only while the
// target was unknown. Now the refusal says which of two different things is
// wrong, and only one of them is a thing lane 09 can lift: below sm_90 there
// is no barrier.cluster to order a cluster with at all, and at sm_90 and
// above there is one and what is missing is the participation contract.
//
// The outcomes are the same -- a cluster-scope task is refused on every
// target this package can emit for -- and the test asserts the reasons,
// because a message that gave the lane-09 reason on sm_80 would be describing
// a program that could never run there as one that is merely unimplemented.
func TestTargetConditionalScopeRefusal(t *testing.T) {
	cluster, err := cudataskgraph.New([]cudataskgraph.Task{{ID: "x", Rank: 0, Scope: cudataskgraph.ScopeCluster}})
	if err != nil {
		t.Fatal(err)
	}
	cta, err := cudataskgraph.New([]cudataskgraph.Task{{ID: "x", Rank: 0, Scope: cudataskgraph.ScopeCTA}})
	if err != nil {
		t.Fatal(err)
	}
	device, err := cudataskgraph.New([]cudataskgraph.Task{{ID: "x", Rank: 0, Scope: cudataskgraph.ScopeDevice}})
	if err != nil {
		t.Fatal(err)
	}
	ops := map[string]cudamega.Operation{"x": {Kind: cudamega.Add, Scalar: 1}}
	grid := cudamega.Grid{CTAs: 1, ThreadsPerCTA: 32}

	for _, tc := range []struct {
		name    string
		graph   cudataskgraph.Graph
		target  cudamega.Target
		want    string // "" means the pair must be accepted
		notWant string
	}{
		{name: "cluster below sm_90", graph: cluster, target: cudamega.SM80,
			want: "needs barrier.cluster, which target sm_80 does not have", notWant: "lane 09"},
		{name: "cluster at sm_90", graph: cluster, target: cudamega.SM90,
			want: "needs lane 09's participation contract", notWant: "does not have"},
		{name: "cluster at sm_121a", graph: cluster, target: cudamega.SM121a,
			want: "needs lane 09's participation contract"},
		{name: "cta below sm_80", graph: cta, target: "sm_70",
			want: "needs mbarrier, which target sm_70 predates"},
		{name: "cta at sm_80", graph: cta, target: cudamega.SM80},
		{name: "device below sm_80", graph: device, target: "sm_70",
			want: "target sm_70 predates emitter floor sm_80"},
		{name: "unparseable target", graph: device, target: "turing",
			want: "not spelled sm_"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := cudamega.NewProgram(tc.graph, ops, grid, tc.target)
			if tc.want == "" {
				if err != nil {
					t.Fatalf("refused a pair the target supports: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("accepted a pair the target cannot serve")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("refusal %q does not say %q", err, tc.want)
			}
			if tc.notWant != "" && strings.Contains(err.Error(), tc.notWant) {
				t.Fatalf("refusal %q gives the other target's reason %q", err, tc.notWant)
			}
		})
	}
}

// TestEmittedTextIsTargetInvariant records the claim that made one committed
// .cu serve three targets: above the sm_80 floor the emitted text does not
// depend on the target at all, and only the compiled artifacts do.
//
// This is a claim that can become false. mbarrier is an sm_80 feature and
// barrier.cluster an sm_90 one, so the first program that emits a
// capability-conditional instruction breaks this test, which is the signal
// that the goldens must become per-target text as well as per-target PTX.
func TestEmittedTextIsTargetInvariant(t *testing.T) {
	for _, f := range goldenFixtures(t) {
		want := ""
		for i, target := range ptxTargets {
			got, err := cudamega.ProgramSource(f.graph, f.ops, f.grid, target)
			if err != nil {
				t.Fatalf("%s for %s: %v", f.name, target, err)
			}
			if i == 0 {
				want = got
				continue
			}
			if got != want {
				t.Errorf("%s: the text emitted for %s differs from the text emitted for %s", f.name, target, ptxTargets[0])
			}
		}
	}
}

// refusing is a backend that declares no forward-progress guarantee. Every
// other primitive it has is a working one, so the only reason it can be
// refused is the declaration.
type refusing struct {
	cudamega.Backend
	progress  cudamega.ForwardProgress
	admission cudamega.Admission
}

func (r refusing) ForwardProgress() cudamega.ForwardProgress { return r.progress }
func (r refusing) Residency() cudamega.Admission             { return r.admission }

// TestBackendWithoutForwardProgressGuaranteeRefused is the construction-time
// refusal the boundary rests on.
//
// A backend that cannot say its workers are guaranteed to run concurrently
// cannot run this model: a worker spinning on a predecessor's completion word
// is waiting only if that predecessor is running, and is deadlocked
// otherwise. The refusal has to happen at binding, because the alternative is
// discovering it from a device that has stopped answering.
func TestBackendWithoutForwardProgressGuaranteeRefused(t *testing.T) {
	f := ctaFixture(t)
	p, err := cudamega.NewProgram(f.graph, f.ops, f.grid, f.target)
	if err != nil {
		t.Fatal(err)
	}
	admission := cudamega.Admission{ResidentCTAs: f.grid.CTAs, WorkerThreads: f.grid.ThreadsPerCTA, SharedBytes: 64}

	for _, tc := range []struct {
		name string
		b    cudamega.Backend
		want string
	}{
		{"zero value", refusing{admission: admission}, "declares no forward-progress guarantee"},
		{"out of range", refusing{progress: 99, admission: admission}, "which is not one this package defines"},
		{"nil backend", nil, "nil backend"},
		{"too few workers", refusing{progress: cudamega.IndependentThreads,
			admission: cudamega.Admission{ResidentCTAs: 0, WorkerThreads: 128, SharedBytes: 64}}, "admission is empty"},
		{"narrower grid", refusing{progress: cudamega.CoResident,
			admission: cudamega.Admission{ResidentCTAs: 1, WorkerThreads: 64, SharedBytes: 64}}, "threads per worker"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r, err := cudamega.NewRunner(p, tc.b)
			if err == nil {
				t.Fatalf("bound a backend that may not run this program; runner %v", r)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("refusal %q does not say %q", err, tc.want)
			}
		})
	}

	// The control: the same backend with a guarantee declared is bound.
	// Without this the test above would pass on a NewRunner that refused
	// everything.
	if _, err := cudamega.NewRunner(p, refusing{progress: cudamega.CoResident, admission: admission}); err != nil {
		t.Fatalf("refused a backend that declares a guarantee and admits the grid: %v", err)
	}
}

// recording wraps a backend and notes which primitives the driver invoked.
type recording struct {
	cudamega.Backend
	seen map[string]bool
}

func (r *recording) note(name string) {
	if r.seen == nil {
		r.seen = map[string]bool{}
	}
	r.seen[name] = true
}

func (r *recording) ForwardProgress() cudamega.ForwardProgress {
	r.note("ForwardProgress")
	return r.Backend.ForwardProgress()
}
func (r *recording) Residency() cudamega.Admission {
	r.note("Residency")
	return r.Backend.Residency()
}
func (r *recording) Begin(p *cudamega.Program) error { r.note("Begin"); return r.Backend.Begin(p) }
func (r *recording) BeginTask(i int) (bool, error) {
	r.note("BeginTask")
	return r.Backend.BeginTask(i)
}
func (r *recording) BeginSelection() error {
	r.note("BeginSelection")
	return r.Backend.BeginSelection()
}
func (r *recording) Ready(i, dep int, s cudataskgraph.Scope) error {
	r.note("Ready")
	return r.Backend.Ready(i, dep, s)
}
func (r *recording) Barrier() error { r.note("Barrier"); return r.Backend.Barrier() }
func (r *recording) Take(i int) (bool, error) {
	r.note("Take")
	return r.Backend.Take(i)
}
func (r *recording) NextRound() (bool, error) {
	r.note("NextRound")
	return r.Backend.NextRound()
}
func (r *recording) EndSelection() error { r.note("EndSelection"); return r.Backend.EndSelection() }
func (r *recording) Fence(i, dep int, s cudataskgraph.Scope, k cudamega.Fence) error {
	r.note("Fence")
	return r.Backend.Fence(i, dep, s, k)
}
func (r *recording) Compute(i, src int, op cudamega.Operation) error {
	r.note("Compute")
	return r.Backend.Compute(i, src, op)
}
func (r *recording) Publish(i int) error { r.note("Publish"); return r.Backend.Publish(i) }
func (r *recording) EndTask(i int) error { r.note("EndTask"); return r.Backend.EndTask(i) }
func (r *recording) End() error          { r.note("End"); return r.Backend.End() }

// BackendMethods is every method of cudamega.Backend, by name.
func BackendMethods() []string {
	iface := reflect.TypeOf((*cudamega.Backend)(nil)).Elem()
	names := make([]string, iface.NumMethod())
	for i := range names {
		names[i] = iface.Method(i).Name
	}
	sort.Strings(names)
	return names
}

// TestDriverExercisesEveryPrimitive checks that the driver calls every method
// the interface has.
//
// A primitive the driver never invokes is a primitive the interface should
// not have: each backend would still be obliged to implement it, and nothing
// would ever find out whether it did. This is the interface half of the
// falsifier; the other half -- that no primitive has only one implementation
// -- belongs with a second backend, because it takes two backends to state.
func TestDriverExercisesEveryPrimitive(t *testing.T) {
	f := deviceFixture(t)
	p, err := cudamega.NewProgram(f.graph, f.ops, f.grid, f.target)
	if err != nil {
		t.Fatal(err)
	}
	rec := &recording{Backend: cudamega.CUDASourceBackend(p)}
	r, err := cudamega.NewRunner(p, rec)
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Run(); err != nil {
		t.Fatal(err)
	}
	var missing []string
	for _, name := range BackendMethods() {
		if !rec.seen[name] {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		t.Fatalf("the driver never invokes %v; the interface has primitives nothing exercises", missing)
	}
}
