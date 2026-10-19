package cudaplan_test

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaagreement"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaplan"
	"github.com/tmc/lib/nvidia/x/cudaref"
)

func float32SliceBytes(values []float32) []byte {
	out := make([]byte, 4*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(v))
	}
	return out
}

func cloneAgreementRecord(r *cudaagreement.Record) *cudaagreement.Record {
	if r == nil {
		return nil
	}
	cp := *r
	cp.Steps = append([]cudaagreement.StepRecord(nil), r.Steps...)
	return &cp
}

// workflowNativeOwners keeps every dependent alive until a successful drain.
// A failed cleanup is retained for the lifetime of the test process; it must not
// release the GPU lease or destroy storage whose quiescence is unknown.
type workflowNativeOwners struct {
	drain     func() error
	quiesced  bool
	resources [8][]workflowNativeResource
}

type workflowNativeResource struct {
	name  string
	close func() error
}

const (
	workflowCompletion = iota
	workflowGraph
	workflowPlan
	workflowBuffer
	workflowModule
	workflowStream
	workflowDevice
	workflowLease
)

var workflowRetainedOwners struct {
	sync.Mutex
	owners []*workflowNativeOwners
}

func (o *workflowNativeOwners) add(kind int, name string, close func() error) {
	o.resources[kind] = append(o.resources[kind], workflowNativeResource{name, close})
}

func (o *workflowNativeOwners) retainCompletion(c *cudaplan.Completion) {
	if c == nil {
		return
	}
	o.add(workflowCompletion, "completion", func() error {
		ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
		defer cancel()
		return c.Wait(ctx)
	})
}

func (o *workflowNativeOwners) close() error {
	if !o.quiesced {
		if err := o.drain(); err != nil {
			return fmt.Errorf("drain stream, retain native owners: %w", err)
		}
		// Cleanup submits no more work. Keep this observation on retry,
		// including after the stream itself has been destroyed.
		o.quiesced = true
	}
	for kind := range o.resources {
		for len(o.resources[kind]) > 0 {
			i := len(o.resources[kind]) - 1
			r := o.resources[kind][i]
			if err := r.close(); err != nil {
				return fmt.Errorf("close %s, retain remaining native owners: %w", r.name, err)
			}
			o.resources[kind] = o.resources[kind][:i]
		}
	}
	return nil
}

func admitWorkflowNativeDevice(t *testing.T, gate string) (*cuda.Device, *cuda.Stream, string, *workflowNativeOwners) {
	t.Helper()
	if os.Getenv(gate) != "1" {
		t.Skipf("native workflow qualification requires %s=1", gate)
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("native workflow test requires an inherited GPU lease: %v", err)
	}
	target := os.Getenv("CUDAPLAN_NATIVE_TARGET")
	if target != "sm_100a" && target != "sm_121a" {
		t.Fatal("set CUDAPLAN_NATIVE_TARGET to sm_100a or sm_121a")
	}
	ordinal, err := strconv.Atoi(os.Getenv("CUDAPLAN_NATIVE_DEVICE"))
	if err != nil || ordinal < 0 {
		t.Fatal("set CUDAPLAN_NATIVE_DEVICE to an explicit ordinal")
	}
	lockCtx, cancelLock := context.WithTimeout(context.Background(), time.Minute)
	defer cancelLock()
	lease, err := cudalock.Acquire(lockCtx, cudalock.WithLabel(t.Name()))
	if err != nil {
		t.Fatal(err)
	}
	dev, err := cuda.NewDevice(ordinal)
	if err != nil {
		_ = lease.Release()
		t.Fatal(err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		_ = dev.Close()
		_ = lease.Release()
		t.Fatal(err)
	}
	owners := &workflowNativeOwners{drain: func() error {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		return stream.Synchronize(ctx)
	}}
	owners.add(workflowStream, "stream", stream.Close)
	owners.add(workflowDevice, "device", dev.Close)
	owners.add(workflowLease, "GPU lease", lease.Release)
	t.Cleanup(func() {
		if err := owners.close(); err != nil {
			workflowRetainedOwners.Lock()
			workflowRetainedOwners.owners = append(workflowRetainedOwners.owners, owners)
			workflowRetainedOwners.Unlock()
			t.Error(err)
			return
		}
	})
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("sm_%d%da", major, minor); got != target {
		t.Fatalf("device SM%d%d (%s) does not match %s", major, minor, got, target)
	}
	version, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("device=%d target=%s driver_cuda_version=%d lease=%+v", ordinal, target, version, lease.Owner())
	return dev, stream, target, owners
}

// TestWorkflowPublicBackendAgreement feeds the shared three-compute-node FP32
// chain (p = x * a; q = p + b; y = q * a) into public cudaagreement.LaunchExecutor
// and cudaagreement.GraphExecutor, poisons intermediate/output buffers before each
// arm, drains the stream, reads back actual output bytes, compares against both the
// independent FP32 oracle and cudaref.EvaluateSnapshot, and gates on
// cudaagreement.Verify over the real returned executor records. Separately, it
// executes compiled cudaplan ordinary and WithGraphReplay(true) plans and compares
// their drained readbacks and Completion.ExecutedSteps receipts without fabricating
// a cudaagreement.Record from planned metadata.
func TestWorkflowPublicBackendAgreement(t *testing.T) {
	dev, stream, target, owners := admitWorkflowNativeDevice(t, "CUDAPLAN_GRAPH_NATIVE")
	fx := buildWorkflowChainFixture(t, target)

	agreePlan := &cudaagreement.Plan{
		Name: "workflow_fp32_three_node",
		Steps: []cudaagreement.Step{
			{
				Name:     fx.NodeNames[0],
				Artifact: fx.Artifacts["mul"],
				Inputs:   []string{"x", "a"},
				Output:   fx.NodeNames[0],
				Grid:     1,
				Block:    256,
			},
			{
				Name:     fx.NodeNames[1],
				Artifact: fx.Artifacts["add"],
				Inputs:   []string{fx.NodeNames[0], "b"},
				Output:   fx.NodeNames[1],
				Grid:     1,
				Block:    256,
			},
			{
				Name:     fx.NodeNames[2],
				Artifact: fx.Artifacts["mul"],
				Inputs:   []string{fx.NodeNames[1], "a"},
				Output:   fx.Output,
				Grid:     1,
				Block:    256,
			},
		},
	}
	if err := agreePlan.Validate(); err != nil {
		t.Fatalf("agreePlan.Validate: %v", err)
	}

	mods := cudaagreement.NewModules(dev)
	owners.add(workflowModule, "modules", mods.Close)

	bufferNames := []string{"x", "a", "b", fx.NodeNames[0], fx.NodeNames[1], fx.Output}
	slices := make(map[string]cudamem.Slice[float32], len(bufferNames))
	bufs := make(map[string]cudaagreement.Buffer, len(bufferNames))
	for _, name := range bufferNames {
		s, err := cudamem.Alloc[float32](dev, replayElements)
		// Alloc can return an owned allocation with a context restoration error.
		if s.Ptr() != 0 {
			owners.add(workflowBuffer, "buffer", s.Free)
		}
		if err != nil {
			t.Fatal(err)
		}
		u32Slice, err := cudamem.Reinterpret[uint32](s)
		if err != nil {
			t.Fatal(err)
		}
		slices[name] = s
		bufs[name] = cudaagreement.BufferFromSlice(u32Slice)
	}

	data := replayInputs(0)
	_, _, wantOracle := workflowChainOracle(data)
	refMap, err := cudaref.EvaluateSnapshot(fx.Snap, data)
	if err != nil {
		t.Fatal(err)
	}
	if !sameReplayOutput(refMap[fx.Output], wantOracle) {
		t.Fatal("cudaref and independent FP32 oracle disagree")
	}

	poison := make([]float32, replayElements)
	for i := range poison {
		poison[i] = float32(math.NaN())
	}

	runAgreementArm := func(exec cudaagreement.Executor) (cudaagreement.Arm, []float32) {
		for _, inName := range []string{"x", "a", "b"} {
			if err := slices[inName].CopyFromHost(data[inName]); err != nil {
				t.Fatal(err)
			}
		}
		for _, scratchName := range []string{fx.NodeNames[0], fx.NodeNames[1], fx.Output} {
			if err := slices[scratchName].CopyFromHost(poison); err != nil {
				t.Fatal(err)
			}
		}
		rec, err := exec.Execute(stream, agreePlan, bufs)
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
		defer cancel()
		if err := stream.Synchronize(ctx); err != nil {
			t.Fatal(err)
		}
		got := make([]float32, replayElements)
		if err := slices[fx.Output].CopyToHost(got); err != nil {
			t.Fatal(err)
		}
		return cudaagreement.Arm{Record: rec, Output: float32SliceBytes(got)}, got
	}

	launchExec := &cudaagreement.LaunchExecutor{Modules: mods}
	graphExec := &cudaagreement.GraphExecutor{Device: dev, Modules: mods}
	owners.add(workflowGraph, "graph executor", graphExec.Close)

	launchArm, launchGot := runAgreementArm(launchExec)
	graphArm, graphGot := runAgreementArm(graphExec)

	if !sameReplayOutput(launchGot, wantOracle) {
		t.Fatalf("LaunchExecutor output disagrees with FP32 oracle: got=%v want=%v", launchGot, wantOracle)
	}
	if !sameReplayOutput(graphGot, wantOracle) {
		t.Fatalf("GraphExecutor output disagrees with FP32 oracle: got=%v want=%v", graphGot, wantOracle)
	}
	for _, arm := range []cudaagreement.Arm{launchArm, graphArm} {
		if arm.Record == nil || len(arm.Record.Steps) != workflowComputeNodes {
			t.Fatalf("arm %v recorded %d steps, want %d", arm.Record, len(arm.Record.Steps), workflowComputeNodes)
		}
		for i, sr := range arm.Record.Steps {
			wantOut := bufs[agreePlan.Steps[i].Output]
			if sr.Step != fx.NodeNames[i] || sr.Requested != arm.Record.Requested || sr.Executed != arm.Record.Requested || sr.Digest != agreePlan.Steps[i].Artifact.Digest || sr.Grid != 1 || sr.Block != 256 || sr.Output.Ptr != wantOut.Ptr || sr.Output.Bytes != wantOut.Bytes {
				t.Fatalf("step record %d mismatch: got=%+v", i, sr)
			}
		}
	}
	if err := cudaagreement.Verify([]cudaagreement.Arm{launchArm, graphArm}); err != nil {
		t.Fatalf("cudaagreement.Verify(launch, graph): %v", err)
	}

	// Negative control 1: GraphExecutor.Decline on one named step falls back to
	// BackendLaunch; outputs still match, but Verify must fail.
	declineExec := &cudaagreement.GraphExecutor{
		Device:  dev,
		Modules: mods,
		Decline: func(s cudaagreement.Step) error {
			if s.Name == fx.NodeNames[1] {
				return errors.New("forced step decline for agreement control")
			}
			return nil
		},
	}
	owners.add(workflowGraph, "declined graph executor", declineExec.Close)
	declinedArm, declinedGot := runAgreementArm(declineExec)
	if !sameReplayOutput(declinedGot, wantOracle) {
		t.Fatal("declined GraphExecutor fallback disagreed with FP32 oracle")
	}
	if err := cudaagreement.Agree([]cudaagreement.Arm{launchArm, declinedArm}); err != nil {
		t.Fatalf("Agree on declined fallback output: %v", err)
	}
	if err := cudaagreement.Verify([]cudaagreement.Arm{launchArm, declinedArm}); err == nil {
		t.Fatal("Verify accepted GraphExecutor arm with a declined step")
	}

	// Negative control 2: flipping one byte of a detached successful readback
	// must fail Verify.
	flippedBytes := append([]byte(nil), graphArm.Output...)
	flippedBytes[len(flippedBytes)/2] ^= 0x01
	flippedArm := cudaagreement.Arm{Record: cloneAgreementRecord(graphArm.Record), Output: flippedBytes}
	if err := cudaagreement.Verify([]cudaagreement.Arm{launchArm, flippedArm}); err == nil {
		t.Fatal("Verify accepted detached readback with a flipped byte")
	}

	// Negative control 3: mutating one copied record digest must fail Verify.
	mutatedRec := cloneAgreementRecord(graphArm.Record)
	mutatedRec.Steps[1].Digest = fx.Artifacts["mul"].Digest
	mutatedDigestArm := cudaagreement.Arm{Record: mutatedRec, Output: append([]byte(nil), graphArm.Output...)}
	if err := cudaagreement.Verify([]cudaagreement.Arm{launchArm, mutatedDigestArm}); err == nil {
		t.Fatal("Verify accepted copied record with mutated step artifact digest")
	}

	// Separate cudaplan receipt scope: execute compiled cudaplan ordinary and
	// WithGraphReplay(true) modes over the same 3-node fixture and compare their
	// separately drained readbacks and Completion.ExecutedSteps receipts.
	reg := cudaplan.NewKernelRegistry()
	if err := reg.Register("mul", fx.Artifacts["mul"]); err != nil {
		t.Fatal(err)
	}
	if err := reg.Register("add", fx.Artifacts["add"]); err != nil {
		t.Fatal(err)
	}
	for _, replay := range []bool{false, true} {
		cpPlan, err := cudaplan.Compile(fx.Snap, cudaplan.WithDevice(dev), cudaplan.WithKernelRegistry(reg), cudaplan.WithGraphReplay(replay))
		if err != nil {
			t.Fatal(err)
		}
		owners.add(workflowPlan, "plan", cpPlan.Close)
		binding := cudaplan.NewBindings()
		for _, inName := range []string{"x", "a", "b"} {
			if err := slices[inName].CopyFromHost(data[inName]); err != nil {
				t.Fatal(err)
			}
			if err := cudaplan.SetInputSlice(binding, inName, slices[inName], dev); err != nil {
				t.Fatal(err)
			}
		}
		if err := slices[fx.Output].CopyFromHost(poison); err != nil {
			t.Fatal(err)
		}
		if err := cudaplan.SetOutputSlice(binding, fx.Output, slices[fx.Output], dev); err != nil {
			t.Fatal(err)
		}
		comp, err := cpPlan.Execute(stream, binding)
		owners.retainCompletion(comp)
		if err != nil {
			t.Fatal(err)
		}
		waitCtx, cancel := context.WithTimeout(context.Background(), time.Minute)
		waitErr := comp.Wait(waitCtx)
		cancel()
		if waitErr != nil {
			t.Fatal(waitErr)
		}
		if err := checkIntegrationReceipt(cpPlan.Steps(), comp.ExecutedSteps(), comp.Done(), replay); err != nil {
			t.Fatalf("cudaplan replay=%v receipt: %v", replay, err)
		}
		cpGot := make([]float32, replayElements)
		if err := slices[fx.Output].CopyToHost(cpGot); err != nil {
			t.Fatal(err)
		}
		if !sameReplayOutput(cpGot, wantOracle) || !sameReplayOutput(cpGot, launchGot) || !sameReplayOutput(cpGot, graphGot) {
			t.Fatalf("cudaplan replay=%v readback disagreed with executor arms or oracle", replay)
		}
	}
}

// TestWorkflowFinitePlanReference exercises cudaplan.WithFiniteMega(true) on the
// shared three-compute-node FP32 chain after native device admission, verifies
// BackendMega completion and exact agreement with the independent FP32 oracle,
// and checks negative controls for unsupported operations and fallback non-promotion.
func TestWorkflowFinitePlanReference(t *testing.T) {
	dev, stream, _, owners := admitWorkflowNativeDevice(t, "CUDAPLAN_MEGA_NATIVE")
	fx := buildWorkflowChainFixture(t, os.Getenv("CUDAPLAN_NATIVE_TARGET"))

	// Negative control 1: unsupported operation ("gelu") with WithFiniteMega(true)
	// is refused at Compile with ErrUnsupportedBackend.
	badGraph := cudair.NewGraph()
	bx := badGraph.AddInput("x", []int{replayElements}, cudair.Float32)
	by := badGraph.AddUnary("gelu", bx)
	if err := badGraph.SetOutputs(by); err != nil {
		t.Fatal(err)
	}
	badSnap, err := badGraph.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if p, err := cudaplan.Compile(badSnap, cudaplan.WithDevice(dev), cudaplan.WithFiniteMega(true)); p != nil || !errors.Is(err, cudaplan.ErrUnsupportedBackend) {
		if p != nil {
			_ = p.Close()
		}
		t.Fatalf("unsupported finite Mega operation admitted: p=%v err=%v", p, err)
	}

	// Negative control 2: combining WithFiniteMega(true) and WithGraphReplay(true)
	// is refused at Compile with ErrUnsupportedBackend.
	if p, err := cudaplan.Compile(fx.Snap, cudaplan.WithDevice(dev), cudaplan.WithFiniteMega(true), cudaplan.WithGraphReplay(true)); p != nil || !errors.Is(err, cudaplan.ErrUnsupportedBackend) {
		if p != nil {
			_ = p.Close()
		}
		t.Fatalf("finite Mega + graph replay combination admitted: p=%v err=%v", p, err)
	}

	megaPlan, err := cudaplan.Compile(fx.Snap, cudaplan.WithDevice(dev), cudaplan.WithFiniteMega(true))
	if err != nil {
		t.Fatalf("Compile WithFiniteMega: %v", err)
	}
	owners.add(workflowPlan, "finite plan", megaPlan.Close)

	planned := megaPlan.Steps()
	if len(planned) != workflowComputeNodes {
		t.Fatalf("finite Mega steps = %d, want %d", len(planned), workflowComputeNodes)
	}
	for i, s := range planned {
		if s.Backend != cudaplan.BackendMega || s.Artifact == nil || s.Artifact.EntryPoint != "finite_plan" {
			t.Fatalf("step %d not planned on finite_plan BackendMega: %+v", i, s)
		}
	}

	inputs := make(map[string]cudamem.Slice[float32], workflowInputNodes)
	for _, name := range []string{"x", "a", "b"} {
		s, err := cudamem.Alloc[float32](dev, replayElements)
		// Alloc can return an owned allocation with a context restoration error.
		if s.Ptr() != 0 {
			owners.add(workflowBuffer, "buffer", s.Free)
		}
		if err != nil {
			t.Fatal(err)
		}
		inputs[name] = s
	}
	outSlice, err := cudamem.Alloc[float32](dev, replayElements)
	if outSlice.Ptr() != 0 {
		owners.add(workflowBuffer, "output buffer", outSlice.Free)
	}
	if err != nil {
		t.Fatal(err)
	}

	data := replayInputs(1)
	_, _, wantOracle := workflowChainOracle(data)
	refMap, err := cudaref.EvaluateSnapshot(fx.Snap, data)
	if err != nil {
		t.Fatal(err)
	}
	if !sameReplayOutput(refMap[fx.Output], wantOracle) {
		t.Fatal("cudaref and independent FP32 oracle disagree")
	}

	binding := cudaplan.NewBindings()
	for name, s := range inputs {
		if err := s.CopyFromHost(data[name]); err != nil {
			t.Fatal(err)
		}
		if err := cudaplan.SetInputSlice(binding, name, s, dev); err != nil {
			t.Fatal(err)
		}
	}
	poison := make([]float32, replayElements)
	for i := range poison {
		poison[i] = float32(math.NaN())
	}
	if err := outSlice.CopyFromHost(poison); err != nil {
		t.Fatal(err)
	}
	if err := cudaplan.SetOutputSlice(binding, fx.Output, outSlice, dev); err != nil {
		t.Fatal(err)
	}

	comp, err := megaPlan.Execute(stream, binding)
	owners.retainCompletion(comp)
	if err != nil {
		t.Fatal(err)
	}
	waitCtx, cancel := context.WithTimeout(context.Background(), time.Minute)
	waitErr := comp.Wait(waitCtx)
	cancel()
	if waitErr != nil {
		t.Fatal(waitErr)
	}
	if !comp.Done() {
		t.Fatal("Completion.Done() is false after Wait")
	}
	actual := comp.ExecutedSteps()
	if len(actual) != workflowComputeNodes {
		t.Fatalf("ExecutedSteps = %d, want %d", len(actual), workflowComputeNodes)
	}
	for i, step := range actual {
		if step.Index != i || step.NodeName != fx.NodeNames[i] || step.Backend != cudaplan.BackendMega {
			t.Fatalf("executed step %d = %+v, want NodeName=%q Backend=BackendMega", i, step, fx.NodeNames[i])
		}
	}
	if err := checkIntegrationReceipt(planned, actual, comp.Done(), false); err != nil {
		t.Fatalf("checkIntegrationReceipt on finite Mega: %v", err)
	}

	got := make([]float32, replayElements)
	if err := outSlice.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	if !sameReplayOutput(got, wantOracle) {
		t.Fatalf("finite Mega output disagrees with FP32 oracle: got=%v want=%v", got, wantOracle)
	}

	// Negative control 3: matching fallback output on BackendLaunch must never
	// be accepted as BackendMega evidence by the receipt check.
	fallbackReceipt := append([]cudaplan.ExecutedStep(nil), actual...)
	fallbackReceipt[1].Backend = cudaplan.BackendLaunch
	if err := checkIntegrationReceipt(planned, fallbackReceipt, true, false); err == nil {
		t.Fatal("checkIntegrationReceipt accepted BackendLaunch fallback step for a BackendMega plan")
	}
}
