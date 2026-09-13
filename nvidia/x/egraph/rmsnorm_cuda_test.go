package egraph_test

import (
	"context"
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// rmsNormGEMVReference computes fused_rmsnorm_gemv on the CPU in float64:
// x is RMS-normalised over its whole length, scaled elementwise by weight,
// then multiplied by W_proj. This is what the operator name promises and what
// every caller of a "fused_rmsnorm_gemv" kernel is entitled to assume.
func rmsNormGEMVReference(x, weight, wProj []float32, inDim, outDim int, eps float64) []float32 {
	var sq float64
	for _, v := range x {
		sq += float64(v) * float64(v)
	}
	scale := 1 / math.Sqrt(sq/float64(inDim)+eps)
	out := make([]float32, outDim)
	for row := 0; row < outDim; row++ {
		var acc float64
		for i := 0; i < inDim; i++ {
			acc += float64(x[i]) * scale * float64(weight[i]) * float64(wProj[row*inDim+i])
		}
		out[row] = float32(acc)
	}
	return out
}

func deviceSlice[T any](t *testing.T, dev *cuda.Device, values []T) cudamem.Slice[T] {
	t.Helper()
	s, err := cudamem.Alloc[T](dev, len(values))
	if err != nil {
		t.Fatal(err)
	}
	if err := dev.WithContext(func() error { return s.CopyFromHost(values) }); err != nil {
		_ = s.Free()
		t.Fatal(err)
	}
	return s
}

// TestGeneratedRMSNormGEMVMatchesReference runs the generated
// fused_rmsnorm_gemv kernel on the device and compares it elementwise against
// rmsNormGEMVReference.
//
// The generator emits source that compiles and returns finite numbers for any
// input, so a compile gate or a status-code check cannot tell a correct kernel
// from one that omits the normalisation entirely. Only a value oracle can.
func TestGeneratedRMSNormGEMVMatchesReference(t *testing.T) {
	for _, threads := range []uint32{32, 64, 128} {
		t.Run(fmt.Sprintf("block%d", threads), func(t *testing.T) {
			testRMSNormGEMV(t, threads)
		})
	}
}

func testRMSNormGEMV(t *testing.T, threads uint32) {
	dev := cudatest.Require(t)
	defer dev.Close()
	src, err := egraph.GenerateCUDA(
		&egraph.Expr{Op: "fused_rmsnorm_gemv", Children: []*egraph.Expr{{Op: "x"}, {Op: "w"}}},
		egraph.CodegenOptions{KernelName: "RMSNormGEMV"},
	)
	if err != nil {
		t.Fatalf("GenerateCUDA: %v", err)
	}
	compiled, err := nvrtcx.Compile(src,
		nvrtcx.WithName("rmsnorm.cu"),
		nvrtcx.WithDeviceArch(dev),
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
	fn, err := mod.GetFunction("RMSNormGEMV")
	if err != nil {
		t.Fatal(err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()

	const (
		inDim  = 128
		outDim = 8
		eps    = 1e-5
	)
	// x is scaled well away from unit RMS so that a kernel which skips the
	// normalisation is off by a large factor rather than by rounding.
	x := make([]float32, inDim)
	weight := make([]float32, inDim)
	wProj := make([]float32, outDim*inDim)
	for i := range x {
		x[i] = float32((i*13)%17-8) * 0.75
		weight[i] = float32((i*5)%9-4) / 8
	}
	for i := range wProj {
		wProj[i] = float32((i*7)%11-5) / 16
	}

	dx := deviceSlice(t, dev, x)
	dw := deviceSlice(t, dev, weight)
	dp := deviceSlice(t, dev, wProj)
	dout := deviceSlice(t, dev, make([]float32, outDim))
	for _, free := range []func() error{dx.Free, dw.Free, dp.Free, dout.Free} {
		defer free()
	}

	if err := cudalaunch.Kernel(fn).Grid(outDim, 1).Block(threads).On(stream).Launch(
		dx, dw, dp, dout, int32(inDim), int32(outDim), float32(eps),
	); err != nil {
		t.Fatalf("launch: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := stream.Synchronize(ctx); err != nil {
		t.Fatalf("synchronize: %v", err)
	}
	got := make([]float32, outDim)
	if err := dev.WithContext(func() error { return dout.CopyToHost(got) }); err != nil {
		t.Fatal(err)
	}

	want := rmsNormGEMVReference(x, weight, wProj, inDim, outDim, eps)
	for i := range want {
		tol := 1e-4 * math.Max(1, math.Abs(float64(want[i])))
		if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
			t.Errorf("out[%d] = %v, want %v (diff %v)", i, got[i], want[i], diff)
		}
	}
}
