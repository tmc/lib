package cudaagreement_test

import (
	"encoding/binary"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaagreement"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaplan"
)

// megaRig holds the unary chain's buffers on the harness device.
type megaRig struct {
	h      *harness
	plan   *cudaagreement.Plan
	ops    map[string]cudamega.Operation
	bufs   map[string]cudaagreement.Buffer
	slices map[string]cudamem.Slice[uint32]
}

func newMegaRig(t *testing.T, h *harness) *megaRig {
	t.Helper()
	r := &megaRig{h: h, bufs: map[string]cudaagreement.Buffer{}, slices: map[string]cudamem.Slice[uint32]{}}
	r.plan, r.ops = megaPlan(t, os.Getenv("CUDAAGREEMENT_NATIVE_TARGET"))
	for _, name := range []string{"x", "t1", "t2", "y"} {
		s, err := cudamem.Alloc[uint32](h.dev, megaElements)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := s.Free(); err != nil {
				t.Error(err)
			}
		})
		r.slices[name] = s
		r.bufs[name] = cudaagreement.BufferFromSlice(s)
	}
	return r
}

// run executes one arm over x with every other buffer poisoned, and returns
// the arm, its readback and the stream launches it issued.
func (r *megaRig) run(t *testing.T, e cudaagreement.Executor, x []uint32) (cudaagreement.Arm, []uint32, uint64) {
	t.Helper()
	if err := r.slices["x"].CopyFromHost(x); err != nil {
		t.Fatal(err)
	}
	dirty := make([]uint32, megaElements)
	for i := range dirty {
		dirty[i] = uint32(poison)
	}
	for _, name := range []string{"t1", "t2", "y"} {
		if err := r.slices[name].CopyFromHost(dirty); err != nil {
			t.Fatal(err)
		}
	}
	before := cudalaunch.TotalLaunches()
	rec, err := e.Execute(r.h.stream, r.plan, r.bufs)
	if err != nil {
		t.Fatal(err)
	}
	synchronize(t, r.h.stream)
	if mega, ok := e.(*cudaagreement.MegaExecutor); ok {
		if err := mega.Drain(nil); err != nil {
			t.Fatal(err)
		}
	}
	launches := cudalaunch.TotalLaunches() - before
	got := make([]uint32, megaElements)
	if err := r.slices["y"].CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	b := make([]byte, 4*len(got))
	for i, v := range got {
		binary.LittleEndian.PutUint32(b[4*i:], v)
	}
	return cudaagreement.Arm{Record: rec, Output: b}, got, launches
}

func (r *megaRig) mega(t *testing.T, ops map[string]cudamega.Operation) *cudaagreement.MegaExecutor {
	t.Helper()
	e := &cudaagreement.MegaExecutor{
		Device:     r.h.dev,
		Modules:    r.h.mods,
		Target:     cudamega.Target(os.Getenv("CUDAAGREEMENT_NATIVE_TARGET")),
		Grid:       megaGrid,
		Operations: ops,
		Compile:    megaCompile(t),
	}
	t.Cleanup(func() {
		if err := e.Close(); err != nil {
			t.Error(err)
		}
	})
	return e
}

func megaInputs() []uint32 {
	x := make([]uint32, megaElements)
	for i := range x {
		// Values near the top of the range, so the chain wraps and a
		// signed or narrowed implementation would differ.
		x[i] = uint32(i)*2654435761 ^ 0xfffff000
	}
	return x
}

func sameUint32(got, want []uint32) (int, bool) {
	for i := range want {
		if got[i] != want[i] {
			return i, false
		}
	}
	return 0, len(got) == len(want)
}

// TestMegaBackendAgreesOrDeclinesExplicitly runs the unary chain through the
// launch, graph and mega arms. The mega arm must record BackendMega for every
// step from the worker's own start-order and abort words, issue exactly one
// stream launch, and agree bitwise. A plan the worker cannot take in full must
// record the step it declined, with the reason, and fail the gate even though
// the bytes still agree.
func TestMegaBackendAgreesOrDeclinesExplicitly(t *testing.T) {
	h := newHarness(t)
	r := newMegaRig(t, h)
	graph := &cudaagreement.GraphExecutor{Device: h.dev, Modules: h.mods}
	t.Cleanup(func() {
		if err := graph.Close(); err != nil {
			t.Error(err)
		}
	})
	x := megaInputs()
	want := megaReference(x)

	launchArm, launchGot, launchN := r.run(t, &cudaagreement.LaunchExecutor{Modules: h.mods}, x)
	graphArm, graphGot, graphN := r.run(t, graph, x)
	megaArm, megaGot, megaN := r.run(t, r.mega(t, r.ops), x)
	t.Logf("launch record:\n%s", launchArm.Record)
	t.Logf("graph record:\n%s", graphArm.Record)
	t.Logf("mega record:\n%s", megaArm.Record)
	t.Logf("stream launches: launch=%d graph=%d mega=%d", launchN, graphN, megaN)

	for name, got := range map[string][]uint32{"launch": launchGot, "graph": graphGot, "mega": megaGot} {
		if i, ok := sameUint32(got, want); !ok {
			t.Fatalf("%s arm disagrees with the host reference at %d: got %#x, want %#x", name, i, got[i], want[i])
		}
	}
	if err := megaArm.Record.ExecutedOn(cudaplan.BackendMega); err != nil {
		t.Fatal(err)
	}
	for _, s := range megaArm.Record.Steps {
		if s.Executed != cudaplan.BackendMega || s.Program == "" || s.EntryPoint != "dag_worker" || s.Grid != megaGrid.CTAs || s.Block != megaGrid.ThreadsPerCTA {
			t.Fatalf("mega step record %s is not a persistent-worker record", s)
		}
	}
	if megaN != 1 {
		t.Fatalf("mega arm issued %d stream launches, want 1 persistent launch", megaN)
	}
	arms := []cudaagreement.Arm{launchArm, graphArm, megaArm}
	if err := cudaagreement.Verify(arms); err != nil {
		t.Fatalf("Verify: %v", err)
	}
	t.Logf("Verify: agreement over %d bytes, three arms on their requested backends", len(megaArm.Output))

	mutant := append([]uint32(nil), megaGot...)
	mutant[megaElements/3]++
	b := make([]byte, 4*len(mutant))
	for i, v := range mutant {
		binary.LittleEndian.PutUint32(b[4*i:], v)
	}
	corrupt := cudaagreement.Arm{Record: megaArm.Record, Output: b}
	verifyErr := cudaagreement.Verify([]cudaagreement.Arm{launchArm, graphArm, corrupt})
	if verifyErr == nil {
		t.Fatal("Verify accepted a mega output corrupted by one")
	}
	t.Logf("one-ULP control: Verify: %v", verifyErr)

	t.Run("declined", func(t *testing.T) {
		// The worker has no operation for mul5, so the executor must run it
		// ordinarily between two persistent runs, and say so.
		ops := map[string]cudamega.Operation{"add3": r.ops["add3"], "add7": r.ops["add7"]}
		arm, got, n := r.run(t, r.mega(t, ops), x)
		t.Logf("mega record with a declined step:\n%s", arm.Record)
		t.Logf("stream launches: %d", n)
		if i, ok := sameUint32(got, want); !ok {
			t.Fatalf("declining arm disagrees with the host reference at %d", i)
		}
		d := arm.Record.Declined()
		if len(d) != 1 || d[0].Step != "mul5" || d[0].Executed != cudaplan.BackendLaunch || d[0].Declined == "" {
			t.Fatalf("declined steps = %v, want mul5 on Launch with a reason", d)
		}
		if n != 3 {
			t.Fatalf("declining arm issued %d stream launches, want 3 (worker, mul5, worker)", n)
		}
		if err := cudaagreement.Agree([]cudaagreement.Arm{launchArm, arm}); err != nil {
			t.Fatalf("outputs should still match: %v", err)
		}
		err := cudaagreement.Verify([]cudaagreement.Arm{launchArm, graphArm, arm})
		if err == nil {
			t.Fatal("Verify accepted a mega arm that declined a step")
		}
		t.Logf("declined-step control: Agree: outputs identical; Verify: %v", err)
	})
}
