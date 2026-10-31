package cudaplan_test

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaplan"
	"github.com/tmc/lib/nvidia/x/cudasearch"
	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// This integration fixture admits only 64-element float32 Input/Add/Mul trees,
// with at most seven binary nodes. It is not a general expression compiler.
func selectedPlan(expr *egraph.Expr, target string) (*cudair.Snapshot, *cudaplan.KernelRegistry, string, string, error) {
	if target != "sm_100a" && target != "sm_121a" {
		return nil, nil, "", "", fmt.Errorf("unsupported fixture target %q", target)
	}
	g := cudair.NewGraph()
	inputs := map[string]*cudair.Node{}
	active := map[*egraph.Expr]bool{}
	count := 0
	var translate func(*egraph.Expr, int) (*cudair.Node, error)
	translate = func(e *egraph.Expr, depth int) (*cudair.Node, error) {
		if e == nil || depth > 8 || active[e] {
			return nil, fmt.Errorf("invalid finite expression")
		}
		active[e] = true
		defer delete(active, e)
		if e.Op == "Input" {
			name, ok := e.Data.(string)
			if !ok || (name != "x" && name != "a" && name != "b") || len(e.Children) != 0 {
				return nil, fmt.Errorf("unsupported input")
			}
			if inputs[name] == nil {
				inputs[name] = g.AddInput(name, []int{replayElements}, cudair.Float32)
			}
			return inputs[name], nil
		}
		if (e.Op != "Add" && e.Op != "Mul") || e.Data != nil || len(e.Children) != 2 {
			return nil, fmt.Errorf("unsupported affine node %q", e.Op)
		}
		count++
		if count > 7 {
			return nil, fmt.Errorf("expression exceeds seven binary nodes")
		}
		a, err := translate(e.Children[0], depth+1)
		if err != nil {
			return nil, err
		}
		b, err := translate(e.Children[1], depth+1)
		if err != nil {
			return nil, err
		}
		op := "add"
		if e.Op == "Mul" {
			op = "mul"
		}
		return g.AddBinary(op, a, b), nil
	}
	output, err := translate(expr, 0)
	if err != nil {
		return nil, nil, "", "", err
	}
	if count == 0 {
		return nil, nil, "", "", fmt.Errorf("input-only expression")
	}
	if err = g.SetOutputs(output); err != nil {
		return nil, nil, "", "", err
	}
	snap, err := g.Snapshot()
	if err != nil {
		return nil, nil, "", "", err
	}
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "a", Type: "ptr", IsPointer: true}, cudakernel.Param{Name: "b", Type: "ptr", IsPointer: true}, cudakernel.Param{Name: "out", Type: "ptr", IsPointer: true})
	if err != nil {
		return nil, nil, "", "", err
	}
	registry := cudaplan.NewKernelRegistry()
	identity := expr.Fingerprint() + "/float32/64/grid=1/block=256/" + target
	for _, op := range []string{"mul", "add"} {
		art, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(workflowBinaryPTX(target, op)), "apply", schema, target)
		if err != nil {
			return nil, nil, "", "", err
		}
		if err = registry.Register(op, art); err != nil {
			return nil, nil, "", "", err
		}
		identity += "/" + art.Digest()
	}
	return snap, registry, output.Name, fmt.Sprintf("%x", sha256.Sum256([]byte(identity))), nil
}

func affineSelection() *egraph.Expr {
	input := func(name string) *egraph.Expr { return &egraph.Expr{Op: "Input", Data: name} }
	return &egraph.Expr{Op: "Add", Children: []*egraph.Expr{{Op: "Mul", Children: []*egraph.Expr{input("x"), input("a")}}, input("b")}}
}

// Each executable owns private buffers/stream/events and its compiled plan.
// dev and the inherited hardware lease are borrowed from the fixture scope.
// Non-nil error results remain on that scope's owner list before error handling.
type selectedExecutable struct {
	dev              *cuda.Device
	stream           *cuda.Stream
	plan             *cudaplan.ExecutablePlan
	completion       *cudaplan.Completion
	start, end       *cuda.Event
	inputs           map[string]cudamem.Slice[float32]
	output           cudamem.Slice[float32]
	outputName, id   string
	want, lastOutput []float32
	closed           bool
}

func newSelectedExecutable(dev *cuda.Device, expr *egraph.Expr, target string, data map[string][]float32, want []float32) (x *selectedExecutable, err error) {
	snap, registry, output, id, err := selectedPlan(expr, target)
	if err != nil {
		return nil, err
	}
	if dev == nil {
		return nil, fmt.Errorf("selected plan requires a device")
	}
	if len(want) != replayElements {
		return nil, fmt.Errorf("expected output extent mismatch")
	}
	if _, err = cudalock.Inherited(os.Getenv); err != nil {
		return nil, err
	}
	x = &selectedExecutable{dev: dev, inputs: map[string]cudamem.Slice[float32]{}, outputName: output, id: id, want: append([]float32(nil), want...)}
	x.stream, err = dev.NewStream()
	if err != nil {
		return x, err
	}
	x.start, err = dev.NewEvent()
	if err != nil {
		return x, err
	}
	x.end, err = dev.NewEvent()
	if err != nil {
		return x, err
	}
	for _, n := range snap.Inputs() {
		values, ok := data[n.Name]
		if !ok || len(values) != replayElements {
			return x, fmt.Errorf("missing fixture input %q", n.Name)
		}
		b, e := cudamem.Alloc[float32](dev, replayElements)
		x.inputs[n.Name] = b
		if e != nil {
			return x, e
		}
		if e = b.CopyFromHost(values); e != nil {
			return x, e
		}
	}
	x.output, err = cudamem.Alloc[float32](dev, replayElements)
	if err != nil {
		return x, err
	}
	x.plan, err = cudaplan.Compile(snap, cudaplan.WithDevice(dev), cudaplan.WithKernelRegistry(registry))
	return x, err
}
func (x *selectedExecutable) Identity() string { return x.id }
func (x *selectedExecutable) bindings() (*cudaplan.Bindings, error) {
	b := cudaplan.NewBindings()
	for name, input := range x.inputs {
		if err := cudaplan.SetInputSlice(b, name, input, x.dev); err != nil {
			return nil, err
		}
	}
	if err := cudaplan.SetOutputSlice(b, x.outputName, x.output, x.dev); err != nil {
		return nil, err
	}
	return b, nil
}
func (x *selectedExecutable) poison() error {
	values := make([]float32, replayElements)
	for i := range values {
		values[i] = float32(math.NaN())
	}
	return x.output.CopyFromHost(values)
}
func (x *selectedExecutable) submit() error {
	b, err := x.bindings()
	if err != nil {
		return err
	}
	x.completion, err = x.plan.Execute(x.stream, b)
	return err
}
func (x *selectedExecutable) drain(ctx context.Context) error {
	if x.completion != nil {
		if err := x.completion.Wait(ctx); err != nil {
			return err
		}
		x.completion = nil
	}
	return nil
}
func (x *selectedExecutable) execute(ctx context.Context, measure bool) (cudatrial.Receipt, error) {
	if x.closed {
		return cudatrial.Receipt{}, fmt.Errorf("fixture executable closed")
	}
	if err := ctx.Err(); err != nil {
		return cudatrial.Receipt{}, err
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		return cudatrial.Receipt{}, err
	}
	if err := x.drain(ctx); err != nil {
		return cudatrial.Receipt{}, err
	}
	x.lastOutput = nil
	if err := x.poison(); err != nil {
		return cudatrial.Receipt{}, err
	}
	if measure {
		if err := x.start.Record(x.stream); err != nil {
			return cudatrial.Receipt{}, err
		}
	}
	if err := x.submit(); err != nil {
		return cudatrial.Receipt{}, err
	}
	for i, step := range x.completion.ExecutedSteps() {
		if step.Index != i || step.Backend != cudaplan.BackendLaunch {
			return cudatrial.Receipt{}, fmt.Errorf("actual ordinary dispatch mismatch: %+v", step)
		}
	}
	if len(x.completion.ExecutedSteps()) != len(x.plan.Steps()) {
		return cudatrial.Receipt{}, fmt.Errorf("missing submitted steps")
	}
	if measure {
		if err := x.end.Record(x.stream); err != nil {
			return cudatrial.Receipt{}, err
		}
	}
	if err := x.drain(ctx); err != nil {
		return cudatrial.Receipt{}, err
	}
	// Readback is after the stop event and actual drain in both modes. Thus
	// every timed arm validates its own poisoned output, without enlarging the
	// recorded event interval or relying on a separate Outputs execution.
	out := make([]float32, replayElements)
	if err := x.output.CopyToHost(out); err != nil {
		return cudatrial.Receipt{}, err
	}
	digest, err := validateSelectedReadback(out, x.want)
	if err != nil {
		return cudatrial.Receipt{}, err
	}
	x.lastOutput = out
	if !measure {
		return cudatrial.Receipt{Complete: true, Provenance: map[string]string{"output_digest": digest}}, nil
	}
	elapsed, err := x.end.ElapsedSince(x.start)
	if err != nil {
		return cudatrial.Receipt{}, err
	}
	if elapsed <= 0 {
		return cudatrial.Receipt{}, fmt.Errorf("unmeasured event interval")
	}
	r := cudatrial.Receipt{Complete: true, TimingKind: cudatrial.TimingCUDAEvent, Provenance: map[string]string{"timer": "cuda_event", "scope": "single-stream interval around plan Execute; includes submission gaps; not summed kernel or SM-active time", "executable_identity": x.id, "output_digest": digest}}
	err = r.RecordTiming(cudatrial.TimingCUDAEvent, elapsed)
	return r, err
}
func (x *selectedExecutable) Outputs(ctx context.Context) (map[string][]float32, error) {
	if _, err := x.execute(ctx, false); err != nil {
		return nil, err
	}
	return map[string][]float32{"out": append([]float32(nil), x.lastOutput...)}, nil
}
func (x *selectedExecutable) Run(ctx context.Context) (cudatrial.Receipt, error) {
	return x.execute(ctx, true)
}
func (x *selectedExecutable) Close() error {
	if x.closed {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := x.drain(ctx); err != nil {
		return err
	}
	// A failed post-submission event/launch can leave work without a completion.
	if x.stream != nil {
		if err := x.stream.Synchronize(ctx); err != nil {
			return err
		}
	}
	if x.plan != nil {
		if err := x.plan.Close(); err != nil {
			return err
		}
		x.plan = nil
	}
	for _, e := range []**cuda.Event{&x.end, &x.start} {
		if *e != nil {
			if err := (*e).Close(); err != nil {
				return err
			}
			*e = nil
		}
	}
	if err := x.output.Free(); err != nil {
		return err
	}
	for name, b := range x.inputs {
		if err := b.Free(); err != nil {
			return err
		}
		delete(x.inputs, name)
	}
	if x.stream != nil {
		if err := x.stream.Close(); err != nil {
			return err
		}
		x.stream = nil
	}
	x.closed = true
	return nil
}

// Canonical inherited admission precedes the first device/driver operation.
// The fixture never acquires/releases the ancestor's lease.
func selectedDevice(t *testing.T) (*cuda.Device, string) {
	t.Helper()
	if os.Getenv("CUDAPLAN_CONNECTED_NATIVE") != "1" {
		t.Skip("requires separately authorized CUDAPLAN_CONNECTED_NATIVE=1")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatal(err)
	}
	target := os.Getenv("CUDAPLAN_NATIVE_TARGET")
	if target != "sm_100a" && target != "sm_121a" {
		t.Fatal("explicit sm_100a or sm_121a target required")
	}
	ordinal, err := strconv.Atoi(os.Getenv("CUDAPLAN_NATIVE_DEVICE"))
	if err != nil || ordinal < 0 {
		t.Fatal("explicit device ordinal required")
	}
	dev, err := cuda.NewDevice(ordinal)
	if err != nil {
		if dev != nil {
			t.Cleanup(func() {
				if e := dev.Close(); e != nil {
					t.Errorf("retain failed device: %v", e)
				}
			})
		}
		t.Fatal(err)
	}
	return dev, target
}

func TestNativeSearchUsesSelectedPlan(t *testing.T) {
	dev, target := selectedDevice(t)
	var owners []*selectedExecutable
	var eval *cudasearch.ExecutableEvaluator
	var result *cudasearch.SearchResult
	t.Cleanup(func() {
		if eval != nil {
			if err := eval.Close(); err != nil {
				t.Error(err)
				return
			}
		}
		if result != nil {
			result.Close()
		}
		for _, x := range owners {
			if err := x.Close(); err != nil {
				t.Errorf("retain device and borrowed lease after executable cleanup failure: %v", err)
				return
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
	if fmt.Sprintf("sm_%d%da", major, minor) != target {
		t.Fatal("target/device mismatch")
	}
	data := replayInputs(2)
	want := make([]float32, replayElements)
	for i := range want {
		want[i] = float32(float32(data["x"][i]*data["a"][i]) + data["b"][i])
	}
	base, err := newSelectedExecutable(dev, affineSelection(), target, data, want)
	if base != nil {
		owners = append(owners, base)
	}
	if err != nil {
		t.Fatal(err)
	}
	workload := cudasearch.WorkloadBindings{Name: "connected-affine-64", InputNames: []string{"x", "a", "b"}, InputIdentities: map[string]string{"x": "replay-round2-x", "a": "replay-round2-a", "b": "replay-round2-b"}, NumericalPolicy: cudasearch.NumericalPolicyExact}
	compiled := 0
	eval = &cudasearch.ExecutableEvaluator{Generation: 1, Workload: workload, Baseline: base, References: map[string][]float32{"out": want}, Rounds: 2, TimingKind: cudatrial.TimingCUDAEvent, CostModel: egraph.AstSizeCost{}, HardwareLockChecker: func() bool { _, err := cudalock.Inherited(os.Getenv); return err == nil }}
	eval.Compile = func(ctx context.Context, expr *egraph.Expr, b cudasearch.WorkloadBindings) (cudasearch.Executable, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if b.Key() != workload.Key() {
			return nil, cudasearch.ErrMissingBinding
		}
		x, err := newSelectedExecutable(dev, expr, target, data, want)
		if x != nil {
			owners = append(owners, x)
		}
		compiled++
		return x, err
	}
	g := egraph.New()
	x := g.Add(egraph.ENode{Op: "Input", Data: "x"})
	a := g.Add(egraph.ENode{Op: "Input", Data: "a"})
	b := g.Add(egraph.ENode{Op: "Input", Data: "b"})
	m := g.Add(egraph.ENode{Op: "Mul", Children: []egraph.Id{x, a}})
	root := g.Add(egraph.ENode{Op: "Add", Children: []egraph.Id{m, b}})
	state := egraph.NewSearchState(g, map[string]egraph.Id{"root": root})
	search := cudasearch.NewBeamSearch(cudasearch.BeamConfig{BeamWidth: 1, Budget: cudasearch.SearchBudget{MaxStates: 1, MaxEvaluations: 1, MaxCompilations: 1, MaxExtractions: 1, MaxDepth: 1, MaxActions: 1, MaxTotalActions: 1, Timeout: 20 * time.Second}})
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()
	result, err = search.Search(ctx, state, nil, eval)
	if err != nil && !errors.Is(err, cudasearch.ErrBudgetExhausted) {
		t.Fatal(err)
	}
	if result == nil || result.Winner == nil || !result.Winner.Validated || compiled != 1 {
		t.Fatalf("missing real plan-backed winner: compiled=%d result=%+v", compiled, result)
	}
	t.Logf("actual selected plan identity=%s; CUDA event stream interval only; all other timing channels unmeasured", result.Winner.ID)
}

func validateSelectedReadback(got, want []float32) (string, error) {
	if len(got) != replayElements || len(want) != replayElements {
		return "", fmt.Errorf("readback extent mismatch")
	}
	bytes := make([]byte, 4*len(got))
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Float32bits(v) != math.Float32bits(want[i]) {
			return "", fmt.Errorf("readback differs from independent reference at %d", i)
		}
		binary.LittleEndian.PutUint32(bytes[4*i:], math.Float32bits(v))
	}
	return fmt.Sprintf("%x", sha256.Sum256(bytes)), nil
}
