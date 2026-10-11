package cudabatch

import (
	"context"
	"fmt"
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
)

// TestPlanEngineNative exercises ordinary GPU addition through the public
// compiler, binder, ExecuteForward, completion and sampler. This is not a model
// or physical KV-cache qualification. It requires explicit device admission.
func TestPlanEngineNative(t *testing.T) {
	if os.Getenv("CUDABATCH_PLAN_NATIVE") != "1" {
		t.Skip("requires CUDABATCH_PLAN_NATIVE=1 and an admitted CUDA machine")
	}
	owner, err := cudalock.Inherited(os.Getenv)
	if err != nil {
		t.Fatal(err)
	}
	ordinal, err := strconv.Atoi(os.Getenv("CUDABATCH_PLAN_DEVICE"))
	if err != nil || ordinal < 0 {
		t.Fatal("requires explicit CUDABATCH_PLAN_DEVICE ordinal")
	}
	dev, err := cuda.NewDevice(ordinal)
	if err != nil {
		t.Fatal(err)
	}
	var engine *PlanEngine
	var stream *cuda.Stream
	var buffers []cudamem.Slice[float32]
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if engine != nil {
			if err := engine.Drain(ctx); err != nil {
				t.Errorf("retain resources after drain failure: %v", err)
				return
			}
			if err := engine.Close(); err != nil {
				t.Errorf("retain resources after plan cleanup failure: %v", err)
				return
			}
		}
		for _, b := range buffers {
			if err := b.Free(); err != nil {
				t.Error(err)
				return
			}
		}
		if stream != nil {
			if err := stream.Close(); err != nil {
				t.Error(err)
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
	target := fmt.Sprintf("sm_%d%d", major, minor)
	stream, err = dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	graph := cudair.NewGraph()
	a := graph.AddInput("a", []int{1}, cudair.Float32)
	b := graph.AddInput("b", []int{1}, cudair.Float32)
	out := graph.AddBinary("add", a, b)
	if err := graph.SetOutputs(out); err != nil {
		t.Fatal(err)
	}
	snapshot, err := graph.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	schema, err := cudakernel.NewSchema(cudakernel.Param{Name: "a", Type: "ptr", IsPointer: true}, cudakernel.Param{Name: "b", Type: "ptr", IsPointer: true}, cudakernel.Param{Name: "out", Type: "ptr", IsPointer: true})
	if err != nil {
		t.Fatal(err)
	}
	ptx := fmt.Sprintf(`.version 9.0
.target %s
.address_size 64
.visible .entry apply(.param .u64 a, .param .u64 b, .param .u64 out) {
 .reg .pred p;
 .reg .b32 tid;
 .reg .b64 pa,pb,po;
 .reg .f32 va,vb,result;
 mov.u32 tid, %%tid.x;
 setp.ne.u32 p, tid, 0;
 @p bra done;
 ld.param.u64 pa,[a];
 ld.param.u64 pb,[b];
 ld.param.u64 po,[out];
 ld.global.f32 va,[pa];
 ld.global.f32 vb,[pb];
 add.rn.f32 result,va,vb;
 st.global.f32 [po],result;
done:
 ret;
}
`, target)
	artifact, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(ptx), "apply", schema, target)
	if err != nil {
		t.Fatal(err)
	}
	registry := cudaplan.NewKernelRegistry()
	if err := registry.Register("add", artifact); err != nil {
		t.Fatal(err)
	}
	alloc := func() cudamem.Slice[float32] {
		s, err := cudamem.Alloc[float32](dev, 1)
		buffers = append(buffers, s)
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	inA, inB, dest := alloc(), alloc(), alloc()
	if err := inA.CopyFromHost([]float32{40}); err != nil {
		t.Fatal(err)
	}
	if err := inB.CopyFromHost([]float32{2}); err != nil {
		t.Fatal(err)
	}
	releases := 0
	bind := func(context.Context, *ExecutionStep) (*cudaplan.Bindings, func(), error) {
		bindings := cudaplan.NewBindings()
		if err := cudaplan.SetInputSlice(bindings, "a", inA, dev); err != nil {
			return nil, nil, err
		}
		if err := cudaplan.SetInputSlice(bindings, "b", inB, dev); err != nil {
			return nil, nil, err
		}
		if err := cudaplan.SetOutputSlice(bindings, out.Name, dest, dev); err != nil {
			return nil, nil, err
		}
		// These allocations stay owned by the fixture until drain and plan close.
		return bindings, func() { releases++ }, nil
	}
	sample := func(_ context.Context, step *ExecutionStep) (map[string]int32, map[string]error, error) {
		values := make([]float32, 1)
		if err := dest.CopyToHost(values); err != nil {
			return nil, nil, err
		}
		if values[0] != 42 {
			return nil, nil, fmt.Errorf("native addition: got %v want 42", values[0])
		}
		return map[string]int32{step.Metadata.RowBindings[0].SeqID: int32(values[0])}, nil, nil
	}
	engine, err = CompilePlanEngine(snapshot, stream, bind, sample, cudaplan.WithDevice(dev), cudaplan.WithKernelRegistry(registry))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	tokens, _, err := engine.ExecuteForward(ctx, planTestStep(t, 1))
	if err != nil {
		t.Fatal(err)
	}
	if tokens["a"] != 42 || releases != 1 || !engine.StepDone(1) {
		t.Fatalf("tokens=%v releases=%d done=%v", tokens, releases, engine.StepDone(1))
	}
	// Poison the output; a second invocation must execute and overwrite it.
	if err := dest.CopyFromHost([]float32{-999}); err != nil {
		t.Fatal(err)
	}
	tokens, _, err = engine.ExecuteForward(ctx, planTestStep(t, 2))
	if err != nil || tokens["a"] != 42 || releases != 2 {
		t.Fatalf("fresh execution tokens=%v releases=%d err=%v", tokens, releases, err)
	}
	t.Logf("ordinary-plan addition: device=%d target=%s lease=%v output=42 runs=2; model/KV qualification excluded", ordinal, target, owner)
}
