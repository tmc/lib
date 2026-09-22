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

func runNativeGraphReference(t *testing.T, kind string) {
	t.Helper()
	gate := "CUDAPLAN_GRAPH_NATIVE"
	if kind == "matmul" {
		gate = "CUDAPLAN_MATMUL_GRAPH_NATIVE"
	}
	if kind == "reduction" {
		gate = "CUDAPLAN_REDUCTION_GRAPH_NATIVE"
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
			if err := stream.Destroy(); err != nil {
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
	out, rebound := alloc(replayElements), alloc(replayElements)
	for _, replay := range []bool{false, true} {
		p, err := cudaplan.Compile(snap, cudaplan.WithDevice(dev), cudaplan.WithKernelRegistry(registry), cudaplan.WithGraphReplay(replay))
		if p != nil {
			plans = append(plans, p)
		}
		if err != nil {
			t.Fatal(err)
		}
		for round := range 3 {
			data := replayInputs(round)
			if kind == "reduction" {
				data = reductionInputs(round)
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
			poison := make([]float32, replayElements)
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
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
			err = c.Wait(ctx)
			cancel()
			if err != nil {
				t.Fatal(err)
			}
			got := make([]float32, replayElements)
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
			got[0]++
			if sameReplayOutput(got, want[output]) {
				t.Fatal("corrupted readback accepted")
			}
			t.Logf("target=%s device=%d replay=%v round=%d exact_reference=pass", target, ordinal, replay, round)
		}
	}
}
