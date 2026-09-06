package egraph_test

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// swigluReference computes fused_swiglu_gemv on the CPU in float64:
// silu(x*W_gate) * (x*W_up), projected back through W_down.
func swigluReference(x, wGate, wUp, wDown []float32, hidden, inter int) []float32 {
	act := make([]float64, inter)
	for j := 0; j < inter; j++ {
		var gate, up float64
		for k := 0; k < hidden; k++ {
			gate += float64(x[k]) * float64(wGate[j*hidden+k])
			up += float64(x[k]) * float64(wUp[j*hidden+k])
		}
		act[j] = gate / (1 + math.Exp(-gate)) * up
	}
	out := make([]float32, hidden)
	for i := 0; i < hidden; i++ {
		var acc float64
		for j := 0; j < inter; j++ {
			acc += act[j] * float64(wDown[i*inter+j])
		}
		out[i] = float32(acc)
	}
	return out
}

// TestGeneratedSwiGLUMatchesReference is the value oracle for the SwiGLU
// kernel. It compiled clean under the Priority A gate, which says only that it
// is valid CUDA; after fused_rmsnorm_gemv turned out to compute a different
// function entirely, an unoracled kernel is assumed wrong until checked.
func TestGeneratedSwiGLUMatchesReference(t *testing.T) {
	dev := cudatest.Require(t)
	defer dev.Close()
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}

	expr := &egraph.Expr{Op: "fused_swiglu_gemv", Children: []*egraph.Expr{{Op: "x"}, {Op: "w"}}}
	src, err := egraph.GenerateCUDA(expr, egraph.CodegenOptions{KernelName: "SwiGLU"})
	if err != nil {
		t.Fatalf("GenerateCUDA: %v", err)
	}
	compiled, err := nvrtcx.Compile(src,
		nvrtcx.WithName("swiglu.cu"),
		nvrtcx.WithArch(int(major), int(minor)),
		nvrtcx.WithNoCache(),
	)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	mod, err := compiled.Load(dev)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer mod.Unload()
	fn, err := mod.GetFunction("SwiGLU")
	if err != nil {
		t.Fatal(err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()

	const (
		hidden = 32
		inter  = 48 // not a multiple of the block width, to exercise the strides
	)
	x := make([]float32, hidden)
	wGate := make([]float32, inter*hidden)
	wUp := make([]float32, inter*hidden)
	wDown := make([]float32, hidden*inter)
	for i := range x {
		x[i] = float32((i*13)%17-8) / 16
	}
	for i := range wGate {
		wGate[i] = float32((i*11)%31-15) / 64
		wUp[i] = float32((i*7)%23-11) / 64
	}
	for i := range wDown {
		wDown[i] = float32((i*5)%19-9) / 64
	}

	dx := deviceSlice(t, dev, x)
	dg := deviceSlice(t, dev, wGate)
	du := deviceSlice(t, dev, wUp)
	dd := deviceSlice(t, dev, wDown)
	dout := deviceSlice(t, dev, make([]float32, hidden))
	for _, free := range []func() error{dx.Free, dg.Free, du.Free, dd.Free, dout.Free} {
		defer free()
	}

	// The kernel stages the activations in dynamic shared memory, one float
	// per intermediate element, and every block fills the whole array, so a
	// single block is the only launch shape its s_act indexing supports.
	if err := cudalaunch.Kernel(fn).Grid(1, 1).Block(64).SharedMem(uint32(inter*4)).On(stream).Launch(
		dx, dg, du, dd, dout, int32(hidden), int32(inter),
	); err != nil {
		t.Fatalf("launch: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := stream.Synchronize(ctx); err != nil {
		t.Fatalf("synchronize: %v", err)
	}
	got := make([]float32, hidden)
	if err := dev.WithContext(func() error { return dout.CopyToHost(got) }); err != nil {
		t.Fatal(err)
	}

	want := swigluReference(x, wGate, wUp, wDown, hidden, inter)
	for i := range want {
		tol := 1e-5 * math.Max(1, math.Abs(float64(want[i])))
		if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
			t.Errorf("out[%d] = %v, want %v (diff %v)", i, got[i], want[i], diff)
		}
	}
}
