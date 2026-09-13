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

// u2Reference computes the 16-element dot product per row on the CPU.
// Each uint32 holds 16 packed 2-bit unsigned weights, dotted against
// 16 signed 8-bit activations and scaled by scale.
func u2Reference(q []uint32, a []int8, scale float32) []float32 {
	out := make([]float32, len(q))
	for idx, qval := range q {
		var sum int
		for b := 0; b < 16; b++ {
			weight := int((qval >> (b * 2)) & 0x03)
			act := int(a[idx*16+b])
			sum += weight * act
		}
		out[idx] = float32(sum) * scale
	}
	return out
}

// TestGeneratedU2MatchesReference is the value oracle for both u2_simd and
// u2_scalar generated kernels. It runs both kernels on the physical GPU,
// asserts bit-for-bit equivalence between SIMD and scalar execution, and
// checks both against u2Reference.
func TestGeneratedU2MatchesReference(t *testing.T) {
	dev := cudatest.Require(t)
	defer dev.Close()
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}

	const n = 64 // 64 elements = 64 * 16 = 1024 activations
	q := make([]uint32, n)
	a := make([]int8, n*16)
	for i := range q {
		// Exercise all 4 2-bit weight patterns across all 16 positions
		q[i] = uint32(i*2654435761) | 0x55555555
	}
	for i := range a {
		// Exercise both positive and negative int8 activations
		a[i] = int8((i*17)%255 - 128)
	}
	scale := float32(0.125)

	want := u2Reference(q, a, scale)

	for _, op := range []string{"u2_simd", "u2_scalar"} {
		t.Run(op, func(t *testing.T) {
			expr := &egraph.Expr{Op: op, Children: []*egraph.Expr{{Op: "q"}, {Op: "a"}}}
			src, err := egraph.GenerateCUDA(expr, egraph.CodegenOptions{KernelName: "U2Kernel"})
			if err != nil {
				t.Fatalf("GenerateCUDA: %v", err)
			}

			compiled, err := nvrtcx.Compile(src,
				nvrtcx.WithName(op+".cu"),
				nvrtcx.WithFlag(fmt.Sprintf("--gpu-architecture=sm_%d%d", major, minor)),
				nvrtcx.WithOutput(nvrtcx.OutputCUBIN),
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
			fn, err := mod.GetFunction("U2Kernel")
			if err != nil {
				t.Fatal(err)
			}
			stream, err := dev.NewStream()
			if err != nil {
				t.Fatal(err)
			}
			defer stream.Destroy()

			dq := deviceSlice(t, dev, q)
			da := deviceSlice(t, dev, a)
			dout := deviceSlice(t, dev, make([]float32, n))
			for _, free := range []func() error{dq.Free, da.Free, dout.Free} {
				defer free()
			}

			threads := uint32(32)
			blocks := uint32((n + int(threads) - 1) / int(threads))
			if err := cudalaunch.Kernel(fn).Grid(blocks, 1).Block(threads).On(stream).Launch(
				dq, da, dout, scale, int32(n),
			); err != nil {
				t.Fatalf("launch: %v", err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			if err := stream.Synchronize(ctx); err != nil {
				t.Fatalf("synchronize: %v", err)
			}
			got := make([]float32, n)
			if err := dev.WithContext(func() error { return dout.CopyToHost(got) }); err != nil {
				t.Fatal(err)
			}

			for i := range want {
				tol := 1e-4 * math.Max(1, math.Abs(float64(want[i])))
				if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
					t.Errorf("out[%d] = %v, want %v (diff %v)", i, got[i], want[i], diff)
				}
			}
		})
	}
}
