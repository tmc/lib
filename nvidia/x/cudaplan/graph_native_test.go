package cudaplan_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaplan"
	"github.com/tmc/lib/nvidia/x/cudaref"
)

const replayElements = 64

// replayFixture uses separate multiply/add kernels to match the reference's
// rounding boundaries. All three inputs are tensors; no scalar ABI is inferred.
func replayFixture(t *testing.T, target string) (*cudair.Snapshot, *cudaplan.KernelRegistry, string) {
	return graphFixture(t, target, false)
}

func graphFixture(t *testing.T, target string, matmul bool) (*cudair.Snapshot, *cudaplan.KernelRegistry, string) {
	t.Helper()
	g := cudair.NewGraph()
	shape := []int{replayElements}
	if matmul {
		shape = []int{8, 8}
	}
	x := g.AddInput("x", shape, cudair.Float32)
	a := g.AddInput("a", shape, cudair.Float32)
	b := g.AddInput("b", shape, cudair.Float32)
	var product *cudair.Node
	if matmul {
		product = g.AddMatMul(x, a)
	} else {
		product = g.AddBinary("mul", x, a)
	}
	y := g.AddBinary("add", product, b)
	if err := g.SetOutputs(y); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	schema, err := cudakernel.NewSchema(
		cudakernel.Param{Name: "a", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "b", Type: "ptr", IsPointer: true},
		cudakernel.Param{Name: "out", Type: "ptr", IsPointer: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	registry := cudaplan.NewKernelRegistry()
	for _, op := range []string{"mul", "add"} {
		ptx := fmt.Sprintf(`.version 9.0
.target %s
.address_size 64
.visible .entry apply(.param .u64 a, .param .u64 b, .param .u64 out) {
 .reg .pred p;
 .reg .b32 tid;
 .reg .b64 pa, pb, po, offset;
 .reg .f32 va, vb, result;
 mov.u32 tid, %%tid.x;
 setp.ge.u32 p, tid, 64;
 @p bra done;
 ld.param.u64 pa, [a];
 ld.param.u64 pb, [b];
 ld.param.u64 po, [out];
 mul.wide.u32 offset, tid, 4;
 add.u64 pa, pa, offset;
 add.u64 pb, pb, offset;
 add.u64 po, po, offset;
 ld.global.f32 va, [pa];
 ld.global.f32 vb, [pb];
 %s.rn.f32 result, va, vb;
 st.global.f32 [po], result;
done:
 ret;
}
`, target, op)
		art, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(ptx), "apply", schema, target)
		if err != nil {
			t.Fatal(err)
		}
		if err := registry.Register(op, art); err != nil {
			t.Fatal(err)
		}
	}
	return snap, registry, y.Name
}

func replayInputs(round int) map[string][]float32 {
	inputs := map[string][]float32{"x": make([]float32, replayElements), "a": make([]float32, replayElements), "b": make([]float32, replayElements)}
	for i := range replayElements {
		inputs["x"][i] = float32(i-31+round) / 8
		inputs["a"][i] = float32(i%5-2) / 2
		inputs["b"][i] = float32(round-i%7) / 4
	}
	return inputs
}

func sameReplayOutput(got, want []float32) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			return false
		}
	}
	return true
}

func TestReplayFixtureOracle(t *testing.T) {
	for _, target := range []string{"sm_100a", "sm_121a"} {
		snap, _, output := replayFixture(t, target)
		result, err := cudaref.EvaluateSnapshot(snap, replayInputs(0))
		if err != nil {
			t.Fatal(err)
		}
		want := result[output]
		if len(want) != replayElements {
			t.Fatal("missing reference output")
		}
		mutant := append([]float32(nil), want...)
		mutant[0]++
		if sameReplayOutput(mutant, want) {
			t.Fatal("comparison accepted corrupted output")
		}
	}
}

func TestMatmulReplayFixtureOracle(t *testing.T) {
	snap, _, output := graphFixture(t, "sm_121a", true)
	data := replayInputs(0)
	result, err := cudaref.EvaluateSnapshot(snap, data)
	if err != nil {
		t.Fatal(err)
	}
	want := result[output]
	if len(want) != replayElements {
		t.Fatal("missing matmul reference output")
	}
	// Check a known row/column independently of the snapshot interpreter.
	var first float32
	for k := range 8 {
		first += data["x"][k] * data["a"][k*8]
	}
	first += data["b"][0]
	if want[0] != first {
		t.Fatalf("first result = %v, want %v", want[0], first)
	}
	mutant := append([]float32(nil), want...)
	mutant[0]++
	if sameReplayOutput(mutant, want) {
		t.Fatal("corrupted matmul output accepted")
	}
}

// TestNativeKernelGraphReplayReference first qualifies ordinary launches, then
// replays changed data at stable addresses and explicitly rebinds a new output.
func TestNativeKernelGraphReplayReference(t *testing.T) {
	runNativeGraphReference(t, "kernel")
}

func TestNativeMatmulGraphReplayReference(t *testing.T) {
	runNativeGraphReference(t, "matmul")
}

func TestNativeReductionGraphReplayReference(t *testing.T) {
	runNativeGraphReference(t, "reduction")
}

// TestNativeIntermediateReductionReference exercises a parallel reduction
// whose input is scratch, rather than caller storage. Reusing that scratch at
// the reduction step races writes against other warps' still-live reads.
func TestNativeIntermediateReductionReference(t *testing.T) {
	runNativePlanReference(t, "intermediate", true)
}

func runNativeGraphReference(t *testing.T, kind string) {
	runNativePlanReference(t, kind, false)
}

func TestNativeFiniteMegaReference(t *testing.T) {
	for _, kind := range []string{"kernel", "reduction", "branch"} {
		t.Run(kind, func(t *testing.T) { runNativePlanReference(t, kind, true) })
	}
}

func runNativePlanReference(t *testing.T, kind string, mega bool) {
	runNativePlanReferenceChecked(t, kind, mega, nil)
}

// inspect runs only after stream completion and numerical reference agreement.
func runNativePlanReferenceChecked(t *testing.T, kind string, mega bool, inspect func(*cudaplan.ExecutablePlan, *cudaplan.Completion, *cuda.Device, map[string][]float32, bool)) {
	t.Helper()
	gate := "CUDAPLAN_GRAPH_NATIVE"
	if kind == "matmul" {
		gate = "CUDAPLAN_MATMUL_GRAPH_NATIVE"
	}
	if kind == "reduction" {
		gate = "CUDAPLAN_REDUCTION_GRAPH_NATIVE"
	}
	if mega {
		gate = "CUDAPLAN_MEGA_NATIVE"
	}
	if os.Getenv(gate) != "1" {
		t.Skipf("native graph qualification requires %s=1", gate)
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("native graph test requires a verified GPU lease")
	}
	target := os.Getenv("CUDAPLAN_NATIVE_TARGET")
	if target != "sm_100a" && target != "sm_121a" {
		t.Fatal("set CUDAPLAN_NATIVE_TARGET to sm_100a or sm_121a")
	}
	snap, registry, output := graphFixture(t, target, kind == "matmul")
	if kind == "reduction" {
		snap, registry, output = reductionFixture(t, target)
	}
	if kind == "intermediate" {
		snap, output = intermediateReductionFixture(t)
	}
	if kind == "branch" {
		g := cudair.NewGraph()
		x := g.AddInput("x", []int{513}, cudair.Float32)
		a := g.AddInput("a", []int{513}, cudair.Float32)
		b := g.AddInput("b", []int{513}, cudair.Float32)
		y := g.AddBinary("add", g.AddBinary("mul", x, a), g.AddUnary("neg", b))
		if err := g.SetOutputs(y); err != nil {
			t.Fatal(err)
		}
		var err error
		snap, err = g.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		output = y.Name
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
	var stream *cuda.Stream
	var plans []*cudaplan.ExecutablePlan
	var completions []*cudaplan.Completion
	var buffers []cudamem.Slice[float32]
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()
		for _, c := range completions {
			if err := c.Wait(ctx); err != nil {
				t.Errorf("retain native resources after drain failure: %v", err)
				return
			}
		}
		for _, p := range plans {
			if err := p.Close(); err != nil {
				t.Errorf("retain dependencies after plan close failure: %v", err)
				return
			}
		}
		if stream != nil {
			if err := stream.Close(); err != nil {
				t.Error(err)
				return
			}
		}
		for _, b := range buffers {
			if err := b.Free(); err != nil {
				t.Error(err)
				return
			}
		}
		if err := dev.Close(); err != nil {
			t.Error(err)
			return
		}
		if err := lease.Release(); err != nil {
			t.Error(err)
		}
	})
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("sm_%d%da", major, minor) != target {
		t.Fatalf("device SM%d%d does not match %s", major, minor, target)
	}
	version, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("device=%d target=%s driver_cuda_version=%d lease=%+v", ordinal, target, version, lease.Owner())
	stream, err = dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	alloc := func(elements int) cudamem.Slice[float32] {
		b, err := cudamem.Alloc[float32](dev, elements)
		buffers = append(buffers, b)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	inputs := make(map[string]cudamem.Slice[float32])
	for _, input := range snap.Inputs() {
		inputs[input.Name] = alloc(input.NumElements())
	}
	outputElements := snap.Outputs()[0].NumElements()
	out, rebound := alloc(outputElements), alloc(outputElements)
	modes := []bool{false, true}
	if mega {
		modes = []bool{false}
	}
	for _, replay := range modes {
		p, err := cudaplan.Compile(snap, cudaplan.WithDevice(dev), cudaplan.WithKernelRegistry(registry), cudaplan.WithGraphReplay(replay), cudaplan.WithFiniteMega(mega))
		if p != nil {
			plans = append(plans, p)
		}
		if err != nil {
			t.Fatal(err)
		}
		if mega {
			artifact := p.Steps()[0].Artifact
			t.Logf("finite_worker entry=%s target=%s digest=%s ptx=\n%s", artifact.EntryPoint, artifact.Target, artifact.Digest, artifact.Raw)
		}
		for round := range 3 {
			data := replayInputs(round)
			if kind == "intermediate" {
				data = intermediateReductionInputs(round)
			}
			if kind == "reduction" {
				data = reductionInputs(round)
				if mega {
					// A float32 accumulator loses the middle value. This pins
					// the actual reference's float64 accumulation boundary.
					for i := range 8 {
						data["x"][i] = 0
					}
					data["x"][0], data["x"][1], data["x"][2] = 16777216, float32(round+1), -16777216
				}
			}
			if kind == "branch" {
				data = make(map[string][]float32)
				for name, input := range inputs {
					values := make([]float32, input.Len())
					for i := range values {
						values[i] = float32(i%17-8+round) / 4
						if name == "a" {
							values[i] = float32(i%5-2-round) / 2
						}
						if name == "b" {
							values[i] = float32(i%7+round) / 8
						}
					}
					data[name] = values
				}
			}
			binding := cudaplan.NewBindings()
			for name, b := range inputs {
				if err := b.CopyFromHost(data[name]); err != nil {
					t.Fatal(err)
				}
				if err := cudaplan.SetInputSlice(binding, name, b, dev); err != nil {
					t.Fatal(err)
				}
			}
			dest := out
			if round == 2 {
				dest = rebound
			}
			poison := make([]float32, outputElements)
			for i := range poison {
				poison[i] = float32(math.NaN())
			}
			if err := dest.CopyFromHost(poison); err != nil {
				t.Fatal(err)
			}
			if err := cudaplan.SetOutputSlice(binding, output, dest, dev); err != nil {
				t.Fatal(err)
			}
			if replay && round == 2 {
				c, err := p.Execute(stream, binding)
				if c != nil {
					completions = append(completions, c)
				}
				if !errors.Is(err, cudaplan.ErrStaleBinding) {
					t.Fatalf("changed graph pointer accepted: %v", err)
				}
				p.RebindGraph()
			}
			c, err := p.Execute(stream, binding)
			if c != nil {
				completions = append(completions, c)
			}
			if err != nil {
				t.Fatal(err)
			}
			receipt := c.ExecutedSteps()
			if len(receipt) != len(p.Steps()) {
				t.Fatalf("submitted %d of %d steps", len(receipt), len(p.Steps()))
			}
			for i, step := range p.Steps() {
				wantBackend := step.Backend
				if replay {
					wantBackend = cudaplan.BackendGraph
				}
				if receipt[i].Index != i || receipt[i].NodeName != step.NodeName || receipt[i].Backend != wantBackend {
					t.Fatalf("actual backend receipt=%+v planned=%+v", receipt[i], step)
				}
			}
			canceled, cancelWait := context.WithCancel(context.Background())
			cancelWait()
			if err := c.Wait(canceled); !errors.Is(err, context.Canceled) {
				t.Fatalf("canceled native wait: %v", err)
			}
			if err := dest.Free(); !errors.Is(err, cudamem.ErrInUse) {
				t.Fatalf("queued output owner not retained: %v", err)
			}
			for name, input := range inputs {
				if err := input.Free(); !errors.Is(err, cudamem.ErrInUse) {
					t.Fatalf("canceled input %s owner not retained: %v", name, err)
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
			err = c.Wait(ctx)
			cancel()
			if err != nil {
				t.Fatal(err)
			}
			got := make([]float32, outputElements)
			if err := dest.CopyToHost(got); err != nil {
				t.Fatal(err)
			}
			want, err := cudaref.EvaluateSnapshot(snap, data)
			if err != nil {
				t.Fatal(err)
			}
			if !sameReplayOutput(got, want[output]) {
				t.Fatalf("replay=%v round=%d output differs from cudaref: got=%v want=%v", replay, round, got, want[output])
			}
			if kind == "intermediate" && !sameReplayOutput(got, intermediateReductionReference(data["x"])) {
				t.Fatal("intermediate reduction differs from independent row sums")
			}
			if inspect != nil {
				inspect(p, c, dev, data, replay)
			}
			t.Logf("raw_output=%v reference=%v", got, want[output])
			got[0]++
			if sameReplayOutput(got, want[output]) {
				t.Fatal("corrupted readback accepted")
			}
			t.Logf("target=%s device=%d replay=%v mega=%v round=%d submitted=%+v exact_reference=pass", target, ordinal, replay, mega, round, receipt)
		}
	}
}

func intermediateReductionFixture(t *testing.T) (*cudair.Snapshot, string) {
	t.Helper()
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{1024, 4}, cudair.Float32)
	y := g.AddUnary("neg", g.AddReduce("sum", g.AddUnary("neg", x), 1))
	if err := g.SetOutputs(y); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	return snap, y.Name
}

func intermediateReductionInputs(round int) map[string][]float32 {
	x := make([]float32, 1024*4)
	for i := range x {
		x[i] = float32(1+i+round*4096) / 4
	}
	return map[string][]float32{"x": x}
}

func intermediateReductionReference(x []float32) []float32 {
	y := make([]float32, 1024)
	for row := range y {
		var sum float64
		for col := range 4 {
			sum += float64(x[row*4+col])
		}
		y[row] = float32(sum)
	}
	return y
}

func TestIntermediateReductionFixtureOracle(t *testing.T) {
	snap, output := intermediateReductionFixture(t)
	for round := range 3 {
		data := intermediateReductionInputs(round)
		result, err := cudaref.EvaluateSnapshot(snap, data)
		if err != nil {
			t.Fatal(err)
		}
		want := intermediateReductionReference(data["x"])
		if !sameReplayOutput(result[output], want) {
			t.Fatal("neg-sum-neg disagrees with independent row sums")
		}
		want[0]++
		if sameReplayOutput(result[output], want) {
			t.Fatal("corrupted intermediate output accepted")
		}
	}
	p, err := cudaplan.Compile(snap, cudaplan.WithFiniteMega(true))
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	steps := p.Steps()
	if len(steps) != 3 {
		t.Fatal("missing intermediate reduction steps")
	}
	a, b := steps[0], steps[1]
	if a.WorkspaceOffset < b.WorkspaceOffset+b.WorkspaceBytes && b.WorkspaceOffset < a.WorkspaceOffset+a.WorkspaceBytes {
		t.Fatal("intermediate reduction aliases its live input")
	}
}
