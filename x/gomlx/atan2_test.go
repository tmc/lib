package gomlx

import (
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
)

// checkFloat compares got and want floating point values, enforcing:
// 1. NaN equality (if want is NaN, got must be NaN; if want is not NaN, got must not be NaN)
// 2. Infinity equality with sign matching
// 3. Exact zero sign-bit matching (+0.0 vs -0.0)
// 4. Finite tolerance check and sign-bit matching
func checkFloat(got, want, tol float64) error {
	if math.IsNaN(want) {
		if !math.IsNaN(got) {
			return fmt.Errorf("got %v, want NaN", got)
		}
		return nil
	}
	if math.IsNaN(got) {
		return fmt.Errorf("got NaN, want %v", want)
	}
	if math.IsInf(want, 0) {
		if !math.IsInf(got, int(math.Copysign(1, want))) {
			return fmt.Errorf("got %v, want %v", got, want)
		}
		return nil
	}
	if math.IsInf(got, 0) {
		return fmt.Errorf("got %v, want %v", got, want)
	}
	if want == 0.0 {
		if got != 0.0 {
			return fmt.Errorf("got %v, want %v", got, want)
		}
		if math.Signbit(got) != math.Signbit(want) {
			return fmt.Errorf("got %v (signbit %v), want %v (signbit %v)", got, math.Signbit(got), want, math.Signbit(want))
		}
		return nil
	}
	if math.Signbit(got) != math.Signbit(want) {
		return fmt.Errorf("signbit mismatch: got %v (signbit %v), want %v (signbit %v)", got, math.Signbit(got), want, math.Signbit(want))
	}
	if math.Abs(got-want) > tol {
		return fmt.Errorf("got %v, want %v (diff %e > tol %e)", got, want, math.Abs(got-want), tol)
	}
	return nil
}

func TestCheckFloatNegativeControl(t *testing.T) {
	// Verify that checkFloat rejects NaN when a finite number is expected
	if err := checkFloat(math.NaN(), 1.0, 1e-6); err == nil {
		t.Fatal("expected checkFloat to reject got=NaN for want=1.0, but got nil error")
	}
	// Verify that checkFloat rejects sign-mismatched zeros
	if err := checkFloat(0.0, math.Copysign(0.0, -1.0), 1e-6); err == nil {
		t.Fatal("expected checkFloat to reject got=+0.0 for want=-0.0, but got nil error")
	}
	// Verify that checkFloat rejects finite value when NaN is expected
	if err := checkFloat(1.0, math.NaN(), 1e-6); err == nil {
		t.Fatal("expected checkFloat to reject got=1.0 for want=NaN, but got nil error")
	}
}

func TestNativeAtan2(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	}()

	pzero := 0.0
	nzero := math.Copysign(0.0, -1.0)
	pinf := math.Inf(1)
	ninf := math.Inf(-1)
	nan := math.NaN()

	baseY := []float64{
		// Quadrants
		0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, -1.0,
		// Signed zeros pairs: (+0, +0), (-0, +0), (+0, -0), (-0, -0)
		pzero, nzero, pzero, nzero,
		// Zero vs finite positive/negative
		pzero, nzero, pzero, nzero,
		1.0, -1.0, 1.0, -1.0,
		// Zero vs infinities
		pzero, nzero, pzero, nzero,
		// Infinities pairs
		pinf, pinf, ninf, ninf,
		// Infinities vs finite
		pinf, ninf, 1.0, 1.0,
		// NaNs
		nan, 1.0, nan,
	}
	baseX := []float64{
		// Quadrants
		1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 0.0, 1.0,
		// Signed zeros pairs
		pzero, pzero, nzero, nzero,
		// Zero vs finite positive/negative
		2.0, 2.0, -2.0, -2.0,
		pzero, pzero, nzero, nzero,
		// Zero vs infinities
		pinf, pinf, ninf, ninf,
		// Infinities pairs
		pinf, ninf, pinf, ninf,
		// Infinities vs finite
		2.0, 2.0, pinf, ninf,
		// NaNs
		1.0, nan, nan,
	}

	t.Run("Float64QuadrantsAndEdgeCases", func(t *testing.T) {
		e, err := graph.NewExec(b, func(y, x *graph.Node) *graph.Node {
			return graph.Atan2(y, x)
		})
		if err != nil {
			t.Fatal(err)
		}
		defer e.Finalize()

		out, err := e.Call(baseY, baseX)
		if err != nil {
			t.Fatal(err)
		}
		defer out[0].FinalizeAll()

		got := out[0].Value().([]float64)
		for i := range baseY {
			y := baseY[i]
			x := baseX[i]
			want := math.Atan2(y, x)
			if err := checkFloat(got[i], want, 1e-12); err != nil {
				t.Fatalf("[%d] y=%v, x=%v: %v", i, y, x, err)
			}
		}
	})

	t.Run("Float32AndBroadcasting", func(t *testing.T) {
		// Test full edge fixtures in Float32
		f32Y := make([]float32, len(baseY))
		f32X := make([]float32, len(baseX))
		for i := range baseY {
			f32Y[i] = float32(baseY[i])
			f32X[i] = float32(baseX[i])
		}

		eF32, err := graph.NewExec(b, func(y, x *graph.Node) *graph.Node {
			return graph.Atan2(y, x)
		})
		if err != nil {
			t.Fatal(err)
		}
		defer eF32.Finalize()

		outF32, err := eF32.Call(f32Y, f32X)
		if err != nil {
			t.Fatal(err)
		}
		defer outF32[0].FinalizeAll()

		gotF32 := outF32[0].Value().([]float32)
		for i := range f32Y {
			want := math.Atan2(float64(f32Y[i]), float64(f32X[i]))
			if err := checkFloat(float64(gotF32[i]), want, 1e-6); err != nil {
				t.Fatalf("[%d] Float32 y=%v, x=%v: %v", i, f32Y[i], f32X[i], err)
			}
		}

		// Broadcasting: [3, 1] vs [1, 2] -> [3, 2]
		ys := [][]float32{{0.0}, {1.0}, {-1.0}}
		xs := [][]float32{{1.0, -1.0}}

		eBcast, err := graph.NewExec(b, func(y, x *graph.Node) *graph.Node {
			return graph.Atan2(y, x)
		})
		if err != nil {
			t.Fatal(err)
		}
		defer eBcast.Finalize()

		outBcast, err := eBcast.Call(ys, xs)
		if err != nil {
			t.Fatal(err)
		}
		defer outBcast[0].FinalizeAll()

		gotBcast := outBcast[0].Value().([][]float32)
		for r := range ys {
			for c := range xs[0] {
				y := float64(ys[r][0])
				x := float64(xs[0][c])
				want := math.Atan2(y, x)
				if err := checkFloat(float64(gotBcast[r][c]), want, 1e-6); err != nil {
					t.Fatalf("broadcast atan2 [%d, %d]: %v", r, c, err)
				}
			}
		}
	})

	t.Run("F16AndBF16EdgeFixtures", func(t *testing.T) {
		// Edge fixtures for half precision types: quadrants, signed zeros, infinities, NaNs
		f16Y := make([]float16.Float16, len(baseY))
		f16X := make([]float16.Float16, len(baseX))
		for i := range baseY {
			f16Y[i] = float16.FromFloat32(float32(baseY[i]))
			f16X[i] = float16.FromFloat32(float32(baseX[i]))
		}

		eF16, err := graph.NewExec(b, func(y, x *graph.Node) *graph.Node {
			return graph.Atan2(y, x)
		})
		if err != nil {
			t.Fatal(err)
		}
		defer eF16.Finalize()

		outF16, err := eF16.Call(f16Y, f16X)
		if err != nil {
			t.Fatal(err)
		}
		defer outF16[0].FinalizeAll()

		gotF16 := outF16[0].Value().([]float16.Float16)
		for i := range f16Y {
			y := float64(f16Y[i].Float32())
			x := float64(f16X[i].Float32())
			want := math.Atan2(y, x)
			if err := checkFloat(float64(gotF16[i].Float32()), want, 1e-3); err != nil {
				t.Fatalf("F16 [%d] y=%v, x=%v: %v", i, y, x, err)
			}
		}

		// BF16
		bf16Y := make([]bfloat16.BFloat16, len(baseY))
		bf16X := make([]bfloat16.BFloat16, len(baseX))
		for i := range baseY {
			bf16Y[i] = bfloat16.FromFloat32(float32(baseY[i]))
			bf16X[i] = bfloat16.FromFloat32(float32(baseX[i]))
		}

		eBF16, err := graph.NewExec(b, func(y, x *graph.Node) *graph.Node {
			return graph.Atan2(y, x)
		})
		if err != nil {
			t.Fatal(err)
		}
		defer eBF16.Finalize()

		outBF16, err := eBF16.Call(bf16Y, bf16X)
		if err != nil {
			t.Fatal(err)
		}
		defer outBF16[0].FinalizeAll()

		gotBF16 := outBF16[0].Value().([]bfloat16.BFloat16)
		for i := range bf16Y {
			y := float64(bf16Y[i].Float32())
			x := float64(bf16X[i].Float32())
			want := math.Atan2(y, x)
			if err := checkFloat(float64(gotBF16[i].Float32()), want, 1e-2); err != nil {
				t.Fatalf("BF16 [%d] y=%v, x=%v: %v", i, y, x, err)
			}
		}
	})

	t.Run("EmptyShapeNative", func(t *testing.T) {
		bEmptyY, err := b.BufferFromFlatData(0, []float64{}, shapes.Make(dtypes.Float64, 0))
		if err != nil {
			t.Fatal(err)
		}
		defer bEmptyY.Finalize()

		bEmptyX, err := b.BufferFromFlatData(0, []float64{}, shapes.Make(dtypes.Float64, 0))
		if err != nil {
			t.Fatal(err)
		}
		defer bEmptyX.Finalize()

		builder := b.Builder("empty_atan2")
		f := builder.Main()
		pY, err := f.Parameter("y", shapes.Make(dtypes.Float64, 0), nil)
		if err != nil {
			t.Fatal(err)
		}
		pX, err := f.Parameter("x", shapes.Make(dtypes.Float64, 0), nil)
		if err != nil {
			t.Fatal(err)
		}
		out, err := f.Atan2(pY, pX)
		if err != nil {
			t.Fatal(err)
		}
		if err := f.Return([]compute.Value{out}, nil); err != nil {
			t.Fatal(err)
		}
		exe, err := builder.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer exe.Finalize()

		res, err := exe.Execute([]compute.Buffer{bEmptyY, bEmptyX}, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer res[0].Finalize()

		shape, err := res[0].Shape()
		if err != nil {
			t.Fatal(err)
		}
		if shape.Size() != 0 {
			t.Fatalf("expected empty shape size 0, got %d", shape.Size())
		}
		got := make([]float64, 0)
		if err := res[0].ToFlatData(got); err != nil {
			t.Fatal(err)
		}
		if len(got) != 0 {
			t.Fatalf("expected 0 elements in empty output, got %d", len(got))
		}
	})

	t.Run("PinnedUpstreamVJPGapReproduced", func(t *testing.T) {
		// Confirm unmodified pinned GoMLX core/graph behavior:
		// GoMLX graph.Gradient(z, y, x) fails on NodeTypeAtan2 because graph.VJPRegistration
		// does not register a VJP rule for NodeTypeAtan2 upstream.
		eGrad, err := graph.NewExec(b, func(y, x *graph.Node) []*graph.Node {
			z := graph.Atan2(y, x)
			grads := graph.Gradient(z, y, x)
			return []*graph.Node{z, grads[0], grads[1]}
		})
		if err != nil {
			t.Fatal(err)
		}
		defer eGrad.Finalize()
		_, err = eGrad.Call(float32(3.0), float32(4.0))
		if err == nil {
			t.Fatal("expected error from upstream GoMLX core/graph due to missing VJP registration for NodeTypeAtan2, but Call succeeded")
		}
		errStr := err.Error()
		if !strings.Contains(errStr, "Atan2") || !strings.Contains(errStr, "no gradient is defined yet") {
			t.Fatalf("expected specific upstream missing gradient error mentioning Atan2, got: %v", err)
		}
		t.Logf("reproduced specific upstream GoMLX autodiff omission for NodeTypeAtan2: %v", err)
	})
}

func TestAtan2Validation(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()

	builder := b.Builder("atan2_val")
	f := builder.Main()

	pFloat, err := f.Parameter("pFloat", shapes.Make(dtypes.Float32, 2), nil)
	if err != nil {
		t.Fatal(err)
	}
	pInt, err := f.Parameter("pInt", shapes.Make(dtypes.Int32, 2), nil)
	if err != nil {
		t.Fatal(err)
	}
	pBadShape, err := f.Parameter("pBadShape", shapes.Make(dtypes.Float32, 3), nil)
	if err != nil {
		t.Fatal(err)
	}

	// 1. Integer rejected
	if _, err := f.Atan2(pInt, pInt); !errors.Is(err, compute.ErrNotImplemented) {
		t.Fatalf("expected unsupported error on integer dtype, got %v", err)
	}

	// 2. Mismatched dtype rejected
	if _, err := f.Atan2(pFloat, pInt); err == nil {
		t.Fatal("expected error on mismatched dtype, got nil")
	}

	// 3. Incompatible broadcast shapes rejected
	if _, err := f.Atan2(pFloat, pBadShape); err == nil {
		t.Fatal("expected error on incompatible broadcast shapes, got nil")
	}

	// 4. Closed function rejected
	if err := f.Return([]compute.Value{pFloat}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Atan2(pFloat, pFloat); err == nil {
		t.Fatal("expected error on closed function, got nil")
	}
}
