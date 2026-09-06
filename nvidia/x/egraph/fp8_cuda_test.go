package egraph_test

import (
	"context"
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// fp8E4M3ToFloat decodes one E4M3 byte: 1 sign bit, 4 exponent bits with a
// bias of 7, 3 mantissa bits, with exponent 0 denoting subnormals.
func fp8E4M3ToFloat(b byte) float64 {
	if b == 0 {
		return 0
	}
	sign := 1.0
	if b>>7&1 == 1 {
		sign = -1
	}
	exp := int(b>>3) & 0x0F
	mant := float64(b & 0x07)
	if exp == 0 {
		return sign * mant / 8 * math.Pow(2, -6)
	}
	return sign * (1 + mant/8) * math.Pow(2, float64(exp-7))
}

func fp8GEMVReference(x []float32, w []byte, scale []float32, inDim, outDim int) []float32 {
	out := make([]float32, outDim)
	for row := 0; row < outDim; row++ {
		var acc float64
		for i := 0; i < inDim; i++ {
			acc += float64(x[i]) * fp8E4M3ToFloat(w[row*inDim+i]) * float64(scale[row])
		}
		out[row] = float32(acc)
	}
	return out
}

// TestScaledFP8GEMVMatchesReference is the value oracle for the hand-written
// FP8 template. It was compile-gated only, and a dequantise-and-GEMV kernel
// that returns finite numbers for any input is exactly the shape that survives
// a compile gate while computing something else.
//
// The block widths include one that is not a whole number of warps, because
// the kernel reduces per-warp partials into shared memory and that is where
// the equivalent code in fused_rmsnorm_gemv was wrong.
func TestScaledFP8GEMVMatchesReference(t *testing.T) {
	for _, threads := range []uint32{32, 64, 128, 96, 48} {
		t.Run(fmt.Sprintf("block%d", threads), func(t *testing.T) {
			testFP8GEMV(t, threads)
		})
	}
}

func testFP8GEMV(t *testing.T, threads uint32) {
	dev := cudatest.Require(t)
	defer dev.Close()
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := nvrtcx.Compile(egraph.ScaledFP8GEMVTemplate,
		nvrtcx.WithName("fp8.cu"),
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
	fn, err := mod.GetFunction("ScaledFP8GEMV")
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
	)
	x := make([]float32, inDim)
	for i := range x {
		x[i] = float32((i*13)%17-8) / 8
	}
	// Sweep byte values so the test covers normals, subnormals and both signs
	// rather than one narrow band of the encoding.
	w := make([]byte, outDim*inDim)
	for i := range w {
		w[i] = byte((i * 7) % 251)
	}
	scale := make([]float32, outDim)
	for i := range scale {
		scale[i] = float32(i+1) / 4
	}

	dx := deviceSlice(t, dev, x)
	dw := deviceSlice(t, dev, w)
	ds := deviceSlice(t, dev, scale)
	dout := deviceSlice(t, dev, make([]float32, outDim))
	for _, free := range []func() error{dx.Free, dw.Free, ds.Free, dout.Free} {
		defer free()
	}

	if err := cudalaunch.Kernel(fn).Grid(outDim, 1).Block(threads).On(stream).Launch(
		dx, dw, ds, dout, int32(inDim), int32(outDim),
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

	want := fp8GEMVReference(x, w, scale, inDim, outDim)
	for i := range want {
		tol := 1e-4 * math.Max(1, math.Abs(float64(want[i])))
		if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
			t.Errorf("out[%d] = %v, want %v (diff %v)", i, got[i], want[i], diff)
		}
	}
}
