package cudaagreement_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaagreement"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// poison is what every intermediate and output buffer holds before an arm
// runs. An arm that launches nothing reads it back and fails against the host
// reference, so two arms cannot agree by both doing nothing.
const poison = int32(0x5a5a5a5a)

// harness owns the device resources one native agreement run needs.
type harness struct {
	dev    *cuda.Device
	stream *cuda.Stream
	mods   *cudaagreement.Modules
	plan   *cudaagreement.Plan
	bufs   map[string]cudaagreement.Buffer
	slices map[string]cudamem.Slice[int32]
}

// newHarness allocates the fixture on the device named by the environment.
// The gate is deliberately a Fatal rather than a Skip once the opt-in is set:
// a skipped native run reports as a pass and is not evidence of anything.
func newHarness(t *testing.T) *harness {
	t.Helper()
	if os.Getenv("CUDAAGREEMENT_NATIVE") != "1" {
		t.Skip("native agreement run requires CUDAAGREEMENT_NATIVE=1")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("native agreement run requires a held GPU lock")
	}
	target := os.Getenv("CUDAAGREEMENT_NATIVE_TARGET")
	if target != "sm_100a" && target != "sm_121a" {
		t.Fatal("set CUDAAGREEMENT_NATIVE_TARGET to sm_100a or sm_121a")
	}
	ordinal, err := strconv.Atoi(os.Getenv("CUDAAGREEMENT_NATIVE_DEVICE"))
	if err != nil || ordinal < 0 {
		t.Fatal("set CUDAAGREEMENT_NATIVE_DEVICE to an explicit ordinal")
	}
	dev, err := cuda.NewDevice(ordinal)
	if err != nil {
		t.Fatal(err)
	}
	h := &harness{dev: dev, plan: fixturePlan(t, target), bufs: map[string]cudaagreement.Buffer{}, slices: map[string]cudamem.Slice[int32]{}}
	h.mods = cudaagreement.NewModules(dev)
	t.Cleanup(func() {
		if err := h.mods.Close(); err != nil {
			t.Error(err)
		}
		if h.stream != nil {
			if err := h.stream.Destroy(); err != nil {
				t.Error(err)
			}
		}
		for _, s := range h.slices {
			if err := s.Free(); err != nil {
				t.Error(err)
			}
		}
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	})
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("sm_%d%da", major, minor); got != target {
		t.Fatalf("device reports %s, target is %s", got, target)
	}
	version, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("device=%d target=%s driver_cuda_version=%d elements=%d", ordinal, target, version, fixtureElements)
	h.stream, err = dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"x", "w", "t1", "t2", "y"} {
		s, err := cudamem.Alloc[int32](dev, fixtureElements)
		if err != nil {
			t.Fatal(err)
		}
		h.slices[name] = s
		h.bufs[name] = cudaagreement.Buffer{Ptr: s.Ptr(), Bytes: int(s.SizeBytes())}
	}
	return h
}

// run executes one arm over x and w and reads back its output. It returns the
// arm, the readback, and the number of stream launches the arm issued, which
// is an oracle independent of the record: a step the record calls BackendGraph
// must not appear in the launch counter.
func (h *harness) run(t *testing.T, e cudaagreement.Executor, x, w []int32) (cudaagreement.Arm, []int32, uint64) {
	t.Helper()
	if err := h.slices["x"].CopyFromHost(x); err != nil {
		t.Fatal(err)
	}
	if err := h.slices["w"].CopyFromHost(w); err != nil {
		t.Fatal(err)
	}
	dirty := make([]int32, fixtureElements)
	for i := range dirty {
		dirty[i] = poison
	}
	for _, name := range []string{"t1", "t2", "y"} {
		if err := h.slices[name].CopyFromHost(dirty); err != nil {
			t.Fatal(err)
		}
	}
	before := cudalaunch.TotalLaunches()
	rec, err := e.Execute(h.stream, h.plan, h.bufs)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	if err := h.stream.Synchronize(ctx); err != nil {
		t.Fatal(err)
	}
	launches := cudalaunch.TotalLaunches() - before
	got := make([]int32, fixtureElements)
	if err := h.slices["y"].CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	return cudaagreement.Arm{Record: rec, Output: bytesOf(got)}, got, launches
}

func sameInt32(got, want []int32) error {
	if len(got) != len(want) {
		return fmt.Errorf("readback has %d elements, reference has %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			return fmt.Errorf("element %d: got %d, want %d", i, got[i], want[i])
		}
	}
	return nil
}

// TestExecutedBackendIsRecorded runs both arms and asserts on the backend each
// step recorded after the fact, cross-checked against the process launch
// counter.
func TestExecutedBackendIsRecorded(t *testing.T) {
	h := newHarness(t)
	graph := &cudaagreement.GraphExecutor{Device: h.dev, Modules: h.mods}
	t.Cleanup(func() {
		if err := graph.Close(); err != nil {
			t.Error(err)
		}
	})
	x, w := fixtureInputs(0)
	want := fixtureReference(x, w)

	for _, tc := range []struct {
		name         string
		exec         cudaagreement.Executor
		wantLaunches uint64
	}{
		{"launch", &cudaagreement.LaunchExecutor{Modules: h.mods}, uint64(len(h.plan.Steps))},
		{"graph", graph, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			arm, got, launches := h.run(t, tc.exec, x, w)
			t.Logf("record:\n%s", arm.Record)
			if err := sameInt32(got, want); err != nil {
				t.Fatalf("%s arm disagrees with the host reference: %v", tc.name, err)
			}
			if err := arm.Record.ExecutedOn(tc.exec.Requested()); err != nil {
				t.Fatal(err)
			}
			if d := arm.Record.Declined(); len(d) != 0 {
				t.Fatalf("%s arm declined %d steps: %v", tc.name, len(d), d)
			}
			if launches != tc.wantLaunches {
				t.Fatalf("%s arm issued %d stream launches, want %d", tc.name, launches, tc.wantLaunches)
			}
			for _, s := range arm.Record.Steps {
				if s.Digest == "" || s.EntryPoint == "" || s.Grid == 0 || s.Block == 0 || s.Output.Ptr == 0 || s.Output.Bytes == 0 {
					t.Fatalf("incomplete record: %s", s)
				}
			}
			t.Logf("%s arm: launches=%d exact_reference=pass", tc.name, launches)
		})
	}
}

// TestLaunchAndGraphBackendsAgree is the stage gate, with its mutation control
// attached: the same comparison that passes on the real readbacks must fail
// when one arm's output moves by one on an integer path.
func TestLaunchAndGraphBackendsAgree(t *testing.T) {
	h := newHarness(t)
	graph := &cudaagreement.GraphExecutor{Device: h.dev, Modules: h.mods}
	t.Cleanup(func() {
		if err := graph.Close(); err != nil {
			t.Error(err)
		}
	})
	x, w := fixtureInputs(1)
	want := fixtureReference(x, w)

	launchArm, launchGot, launchN := h.run(t, &cudaagreement.LaunchExecutor{Modules: h.mods}, x, w)
	graphArm, graphGot, graphN := h.run(t, graph, x, w)
	t.Logf("launch record:\n%s", launchArm.Record)
	t.Logf("graph record:\n%s", graphArm.Record)
	t.Logf("stream launches: launch=%d graph=%d", launchN, graphN)

	for name, got := range map[string][]int32{"launch": launchGot, "graph": graphGot} {
		if err := sameInt32(got, want); err != nil {
			t.Fatalf("%s arm disagrees with the host reference: %v", name, err)
		}
	}
	arms := []cudaagreement.Arm{launchArm, graphArm}
	if err := cudaagreement.Verify(arms); err != nil {
		t.Fatalf("Verify: %v", err)
	}
	t.Logf("Verify: agreement over %d bytes, both arms on their requested backend", len(launchArm.Output))

	// Control: one ULP on an integer path is one, and the comparison must see
	// it. Without this the pass above is a comparison never shown to fail.
	mutant := append([]int32(nil), graphGot...)
	mutant[fixtureElements/2]++
	corrupt := cudaagreement.Arm{Record: graphArm.Record, Output: bytesOf(mutant)}
	agreeErr := cudaagreement.Agree([]cudaagreement.Arm{launchArm, corrupt})
	if agreeErr == nil {
		t.Fatal("Agree accepted an output corrupted by one")
	}
	verifyErr := cudaagreement.Verify([]cudaagreement.Arm{launchArm, corrupt})
	if verifyErr == nil {
		t.Fatal("Verify accepted an output corrupted by one")
	}
	t.Logf("one-ULP control: Agree: %v", agreeErr)
	t.Logf("one-ULP control: Verify: %v", verifyErr)
}

// TestDeclinedStepDoesNotCountAsAgreement forces one step of the graph arm to
// fall back. The outputs still agree, because the fallback computes the same
// thing; the gate must fail anyway, because the graph arm did not run that
// step.
func TestDeclinedStepDoesNotCountAsAgreement(t *testing.T) {
	h := newHarness(t)
	declined := errors.New("forced decline for the harness control")
	graph := &cudaagreement.GraphExecutor{
		Device:  h.dev,
		Modules: h.mods,
		Decline: func(s cudaagreement.Step) error {
			if s.Name == "add" {
				return declined
			}
			return nil
		},
	}
	t.Cleanup(func() {
		if err := graph.Close(); err != nil {
			t.Error(err)
		}
	})
	x, w := fixtureInputs(2)
	want := fixtureReference(x, w)

	launchArm, launchGot, _ := h.run(t, &cudaagreement.LaunchExecutor{Modules: h.mods}, x, w)
	graphArm, graphGot, graphN := h.run(t, graph, x, w)
	t.Logf("graph record with a forced decline:\n%s", graphArm.Record)
	for name, got := range map[string][]int32{"launch": launchGot, "graph": graphGot} {
		if err := sameInt32(got, want); err != nil {
			t.Fatalf("%s arm disagrees with the host reference: %v", name, err)
		}
	}
	if graphN != 1 {
		t.Fatalf("declining arm issued %d stream launches, want 1", graphN)
	}
	d := graphArm.Record.Declined()
	if len(d) != 1 || d[0].Step != "add" || d[0].Executed != (&cudaagreement.LaunchExecutor{}).Requested() {
		t.Fatalf("declined steps = %v, want add on Launch", d)
	}

	arms := []cudaagreement.Arm{launchArm, graphArm}
	if err := cudaagreement.Agree(arms); err != nil {
		t.Fatalf("outputs should still match: %v", err)
	}
	err := cudaagreement.Verify(arms)
	if err == nil {
		t.Fatal("Verify accepted an arm that declined a step")
	}
	t.Logf("declined-step control: Agree: outputs identical; Verify: %v", err)
}

// TestAgreementFailsOnADifferentKernel is the device-side control for the
// comparison itself. The one-ULP control above mutates a readback on the host;
// this one mutates the work, so the bytes that disagree are bytes the device
// actually produced.
func TestAgreementFailsOnADifferentKernel(t *testing.T) {
	h := newHarness(t)
	graph := &cudaagreement.GraphExecutor{Device: h.dev, Modules: h.mods}
	t.Cleanup(func() {
		if err := graph.Close(); err != nil {
			t.Error(err)
		}
	})
	x, w := fixtureInputs(3)
	launchArm, _, _ := h.run(t, &cudaagreement.LaunchExecutor{Modules: h.mods}, x, w)

	// Replace the final xor with or, which differs wherever both operands
	// have a bit set, and run the graph arm over the mutated chain.
	h.plan = fixturePlanWithOp(t, os.Getenv("CUDAAGREEMENT_NATIVE_TARGET"), "or.b32")
	graphArm, graphGot, _ := h.run(t, graph, x, w)
	if err := sameInt32(graphGot, fixtureReference(x, w)); err == nil {
		t.Fatal("mutated kernel reproduced the reference; the control cannot fire")
	}
	err := cudaagreement.Agree([]cudaagreement.Arm{launchArm, graphArm})
	if err == nil {
		t.Fatal("Agree accepted output from a different kernel")
	}
	t.Logf("mutant-kernel control: Agree: %v", err)
	if err := cudaagreement.Verify([]cudaagreement.Arm{launchArm, graphArm}); err == nil {
		t.Fatal("Verify accepted output from a different kernel")
	}
}
