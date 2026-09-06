//go:build linux

package main

import (
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// Native acceptance tests for half weight storage.
//
// These require a CUDA device and are gated on LIB_CUDA_HARDWARE_TEST=1, like
// the other hardware tests in this package. Nothing here has been executed by
// the author: there is no GPU on the machine these were written on, so treat
// every expectation below as a claim awaiting its first run.
//
// What these establish that the host tests cannot: that the device conversion
// instructions agree with the Go conversions, that the vectorized and scalar
// load paths agree with each other and with a CPU oracle, and that a row whose
// stride leaves the base unaligned is handled rather than faulted.
//
// The oracle is deliberately built from the once-rounded weights, so a failure
// here means the kernel implements something other than what it should, not
// that half precision is imprecise. Quantization error is measured separately
// in halfref_test.go and its permissive budget is NOT reused here.

func requireCUDAHardware(t *testing.T) *cuda.Device {
	t.Helper()
	if os.Getenv("LIB_CUDA_HARDWARE_TEST") != "1" {
		t.Skip("set LIB_CUDA_HARDWARE_TEST=1 on a CUDA host")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { dev.Close() })
	return dev
}

// implTolerance is the budget for a kernel against its once-rounded CPU oracle.
//
// Both sides multiply identical float32 values; they differ only in the order
// the products are accumulated, since the kernel reduces across threads in a
// tree and the oracle sums in index order. That is float32 reassociation error
// over at most a few thousand terms, so the bound is tight and does not depend
// on the weight dtype. It is three orders of magnitude below the quantization
// budgets in halfref_test.go, and reusing those here would let a genuinely
// wrong kernel pass.
const (
	implRelTol = 2e-6
	implAbsTol = 1e-5
)

// checkImpl compares a kernel result against its oracle at implementation
// tolerance, rejecting NaN and Inf on either side.
func checkImpl(t *testing.T, name string, got, want, scale []float32) {
	t.Helper()
	for i := range want {
		g, w := float64(got[i]), float64(want[i])
		if math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatalf("%s[%d] is %v", name, i, g)
		}
		if math.IsNaN(w) || math.IsInf(w, 0) {
			t.Fatalf("%s[%d] oracle is %v", name, i, w)
		}
		d := math.Abs(g - w)
		s := math.Abs(float64(scale[i]))
		if d > implAbsTol && (s == 0 || d/s > implRelTol) {
			t.Errorf("%s[%d] = %v, want %v (abs %.3g, rel to term magnitude %.3g)", name, i, g, w, d, d/s)
		}
	}
}

// TestCUDAWeightDTypeGEMVMatchesRoundedOracle runs LinearGEMV in all three
// dtypes against a CPU oracle built from the same once-rounded weights.
//
// The dimensions cover the aligned vector path, all three tail widths, and the
// case with no vector group at all. The odd strides are the point: a row base
// w + row*in_dim is aligned to a 4-element vector only when in_dim is a
// multiple of four, so 129/130/131 exercise the alignment-aware fallback that
// the aligned cases never reach.
func TestCUDAWeightDTypeGEMVMatchesRoundedOracle(t *testing.T) {
	dev := requireCUDAHardware(t)
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()

	dims := []struct{ in, out int }{
		{128, 64}, // fully vectorized, every row aligned
		{129, 64}, // one-element tail; odd rows unaligned
		{130, 33}, // two-element tail
		{131, 17}, // three-element tail
		{3, 5},    // shorter than one vector group
		{5, 4},    // the review's counterexample: row 1 starts 10 bytes in
	}
	for _, d := range []WeightDType{WeightF32, WeightF16, WeightBF16} {
		modules, err := CompileGPUModulesAs(dev, d)
		if err != nil {
			t.Fatalf("%s: compile: %v", d, err)
		}
		if got := modules.WeightDType(); got != d {
			modules.Close()
			t.Fatalf("compiled for %s but module reports %s", d, got)
		}
		for _, dim := range dims {
			t.Run(d.String()+"/"+itoa(dim.in)+"x"+itoa(dim.out), func(t *testing.T) {
				x := deterministicFloats(dim.in, 1)
				w := deterministicFloats(dim.in*dim.out, 2)
				rw := d.RoundTrip(w) // what the device will hold, exactly

				arena := cudamem.NewArena(dev)
				defer arena.Free()
				dx := cudamem.AllocIn[float32](arena, "x", len(x))
				dout := cudamem.AllocIn[float32](arena, "out", dim.out)
				if err := arena.Err(); err != nil {
					t.Fatal(err)
				}
				if err := dx.CopyFromHost(x); err != nil {
					t.Fatal(err)
				}
				dw, err := uploadWeights(arena, d, w)
				if err != nil {
					t.Fatal(err)
				}
				if got, want := dw.Bytes(), dim.in*dim.out*d.Bytes(); got != want {
					t.Fatalf("weights occupy %d bytes, want %d", got, want)
				}

				var noBias WeightSlice
				if err := cudalaunch.OwnedKernel(modules.GEMV).Grid(uint32(dim.out)).Block(256).On(stream).
					Launch(dx, dw, noBias, dout, dim.in, dim.out); err != nil {
					t.Fatalf("launch: %v", err)
				}
				if err := stream.Synchronize(t.Context()); err != nil {
					t.Fatalf("sync: %v", err)
				}
				got := make([]float32, dim.out)
				if err := dout.CopyToHost(got); err != nil {
					t.Fatal(err)
				}

				want := refLinearGEMV(x, rw, nil, dim.in, dim.out)
				checkImpl(t, "gemv", got, want, dotScale(x, rw, dim.in, dim.out))
			})
		}
		modules.Close()
	}
}

// TestCUDAWeightDTypeGEMVNegativeControl checks that the comparison above can
// actually fail.
//
// A single weight is perturbed by an amount far above implRelTol before upload.
// If the test still passes, the oracle is not reading what the kernel computed
// and every other result in this file is worthless.
func TestCUDAWeightDTypeGEMVNegativeControl(t *testing.T) {
	dev := requireCUDAHardware(t)
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()

	const in, out = 129, 8
	for _, d := range []WeightDType{WeightF32, WeightF16, WeightBF16} {
		t.Run(d.String(), func(t *testing.T) {
			modules, err := CompileGPUModulesAs(dev, d)
			if err != nil {
				t.Fatal(err)
			}
			defer modules.Close()

			x := deterministicFloats(in, 1)
			w := deterministicFloats(in*out, 2)
			corrupt := append([]float32(nil), w...)
			corrupt[in+7] += 4 // a term the oracle will not have

			arena := cudamem.NewArena(dev)
			defer arena.Free()
			dx := cudamem.AllocIn[float32](arena, "x", len(x))
			dout := cudamem.AllocIn[float32](arena, "out", out)
			if err := arena.Err(); err != nil {
				t.Fatal(err)
			}
			if err := dx.CopyFromHost(x); err != nil {
				t.Fatal(err)
			}
			dw, err := uploadWeights(arena, d, corrupt)
			if err != nil {
				t.Fatal(err)
			}
			var noBias WeightSlice
			if err := cudalaunch.OwnedKernel(modules.GEMV).Grid(uint32(out)).Block(256).On(stream).
				Launch(dx, dw, noBias, dout, in, out); err != nil {
				t.Fatal(err)
			}
			if err := stream.Synchronize(t.Context()); err != nil {
				t.Fatal(err)
			}
			got := make([]float32, out)
			if err := dout.CopyToHost(got); err != nil {
				t.Fatal(err)
			}

			rw := d.RoundTrip(w) // the UNCORRUPTED oracle
			want := refLinearGEMV(x, rw, nil, in, out)
			scale := dotScale(x, rw, in, out)
			var flagged bool
			for i := range want {
				diff := math.Abs(float64(got[i] - want[i]))
				s := math.Abs(float64(scale[i]))
				if diff > implAbsTol && (s == 0 || diff/s > implRelTol) {
					flagged = true
				}
			}
			if !flagged {
				t.Fatal("corrupted weight did not move the result past tolerance; the comparison proves nothing")
			}
		})
	}
}

// TestCUDADeviceConversionMatchesGo checks the device's widening against Go's.
//
// The Go conversions are verified exhaustively and against CPython on the host;
// this carries that verification onto the device by widening every one of the
// 65536 bit patterns through the kernels' own conversion path and requiring
// bit-identical results. It is what makes the inline PTX cvt.f32.f16 and the
// bfloat16 shift evidence rather than assumption. NaN payloads are excluded,
// since narrowing is not required to preserve which NaN a value was.
func TestCUDADeviceConversionMatchesGo(t *testing.T) {
	dev := requireCUDAHardware(t)
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()

	for _, d := range []WeightDType{WeightF16, WeightBF16} {
		t.Run(d.String(), func(t *testing.T) {
			modules, err := CompileGPUModulesAs(dev, d)
			if err != nil {
				t.Fatal(err)
			}
			defer modules.Close()

			// Widen every bit pattern by using it as a 1x1 identity projection:
			// out[row] = 1.0 * w[row]. RMSNorm's weight path would fold in a
			// scale, so GEMV with a unit input is the direct readout.
			const n = 1 << 16
			packed := make([]uint16, n)
			for i := range packed {
				packed[i] = uint16(i)
			}
			var want []float32
			var idx []uint16
			for i, h := range packed {
				if d == WeightF16 && h&0x7c00 == 0x7c00 && h&0x3ff != 0 {
					continue
				}
				if d == WeightBF16 && h&0x7f80 == 0x7f80 && h&0x7f != 0 {
					continue
				}
				var v float32
				if d == WeightF16 {
					v = float16ToFloat32(h)
				} else {
					v = bfloat16ToFloat32(h)
				}
				if math.IsInf(float64(v), 0) {
					continue // an infinite weight would make the product meaningless
				}
				idx = append(idx, packed[i])
				want = append(want, v)
			}

			arena := cudamem.NewArena(dev)
			defer arena.Free()
			rows := len(idx)
			dx := cudamem.AllocIn[float32](arena, "x", 1)
			dw := cudamem.AllocIn[uint16](arena, "w", rows)
			dout := cudamem.AllocIn[float32](arena, "out", rows)
			if err := arena.Err(); err != nil {
				t.Fatal(err)
			}
			if err := dx.CopyFromHost([]float32{1}); err != nil {
				t.Fatal(err)
			}
			if err := dw.CopyFromHost(idx); err != nil {
				t.Fatal(err)
			}
			weights := WeightSlice{dtype: d, elems: rows, half: dw}
			var noBias WeightSlice
			if err := cudalaunch.OwnedKernel(modules.GEMV).Grid(uint32(rows)).Block(32).On(stream).
				Launch(dx, weights, noBias, dout, 1, rows); err != nil {
				t.Fatal(err)
			}
			if err := stream.Synchronize(t.Context()); err != nil {
				t.Fatal(err)
			}
			got := make([]float32, rows)
			if err := dout.CopyToHost(got); err != nil {
				t.Fatal(err)
			}

			var bad int
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					if bad < 10 {
						t.Errorf("pattern 0x%04x widened to %v on device, %v in Go", idx[i], got[i], want[i])
					}
					bad++
				}
			}
			if bad > 0 {
				t.Fatalf("%d of %d bit patterns disagree", bad, len(want))
			}
			t.Logf("%d bit patterns widen identically on device and in Go", len(want))
		})
	}
}

// TestCUDASingleAndBatchFusedAgreeOnOddDimensions compares the single-token
// fused SwiGLU-down kernel against its batch counterpart at batch size one.
//
// The batch kernel always summed every element of the reduction; the
// single-token kernel dropped the last intermediate_dim % 4 terms until this
// branch gave it a tail loop. The two must now agree exactly, in every dtype:
// they multiply identical values in a different thread layout, so only float32
// reassociation separates them. A dimension that is a multiple of four is
// included as the control that passed even before the fix.
func TestCUDASingleAndBatchFusedAgreeOnOddDimensions(t *testing.T) {
	dev := requireCUDAHardware(t)
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()

	dims := []struct{ hidden, inter int }{
		{16, 128}, // control: no remainder, no misaligned row
		{16, 129}, // one dropped term before the fix
		{16, 130},
		{16, 131},
		{4, 5}, // the review's counterexample stride
	}
	for _, d := range []WeightDType{WeightF32, WeightF16, WeightBF16} {
		modules, err := CompileGPUModulesAs(dev, d)
		if err != nil {
			t.Fatalf("%s: compile: %v", d, err)
		}
		for _, dim := range dims {
			t.Run(d.String()+"/"+itoa(dim.hidden)+"x"+itoa(dim.inter), func(t *testing.T) {
				gate := deterministicFloats(dim.inter, 3)
				up := deterministicFloats(dim.inter, 4)
				down := deterministicFloats(dim.hidden*dim.inter, 5)

				arena := cudamem.NewArena(dev)
				defer arena.Free()
				dg := cudamem.AllocIn[float32](arena, "gate", len(gate))
				du := cudamem.AllocIn[float32](arena, "up", len(up))
				single := cudamem.AllocIn[float32](arena, "single", dim.hidden)
				batched := cudamem.AllocIn[float32](arena, "batched", dim.hidden)
				if err := arena.Err(); err != nil {
					t.Fatal(err)
				}
				if err := dg.CopyFromHost(gate); err != nil {
					t.Fatal(err)
				}
				if err := du.CopyFromHost(up); err != nil {
					t.Fatal(err)
				}
				dw, err := uploadWeights(arena, d, down)
				if err != nil {
					t.Fatal(err)
				}

				if err := cudalaunch.OwnedKernel(modules.FusedSwiGLU).
					Grid(uint32(dim.hidden)).Block(256).On(stream).
					Launch(dg, du, dw, single, dim.hidden, dim.inter); err != nil {
					t.Fatalf("single launch: %v", err)
				}
				if err := cudalaunch.OwnedKernel(modules.BatchFusedSwiGLU).
					Grid(uint32(dim.hidden), 1).Block(256).On(stream).
					Launch(dg, du, dw, batched, dim.hidden, dim.inter, 1); err != nil {
					t.Fatalf("batch launch: %v", err)
				}
				if err := stream.Synchronize(t.Context()); err != nil {
					t.Fatal(err)
				}
				gotSingle := make([]float32, dim.hidden)
				gotBatch := make([]float32, dim.hidden)
				if err := single.CopyToHost(gotSingle); err != nil {
					t.Fatal(err)
				}
				if err := batched.CopyToHost(gotBatch); err != nil {
					t.Fatal(err)
				}

				// Oracle from the once-rounded weights, so a shared bug in both
				// kernels cannot pass by agreeing with itself.
				rd := d.RoundTrip(down)
				act := make([]float32, dim.inter)
				for i := range act {
					g := gate[i]
					act[i] = (g / (1 + float32(math.Exp(float64(-g))))) * up[i]
				}
				want := refSwiGLUDown(gate, up, rd, dim.hidden, dim.inter)
				scale := dotScale(act, rd, dim.inter, dim.hidden)
				checkImpl(t, "single", gotSingle, want, scale)
				checkImpl(t, "batch", gotBatch, want, scale)
			})
		}
		modules.Close()
	}
}

// deterministicFloats returns reproducible values spanning several magnitudes
// without depending on math/rand's sequence across Go versions.
func deterministicFloats(n int, seed uint32) []float32 {
	out := make([]float32, n)
	s := seed*2654435761 + 1
	for i := range out {
		s = s*1664525 + 1013904223
		switch i % 16 {
		case 0:
			out[i] = 0
		case 1:
			out[i] = float32(math.Copysign(0, -1))
		case 2:
			out[i] = float32(math.Ldexp(float64(int32(s))/float64(1<<31), -12))
		case 3:
			out[i] = float32(math.Ldexp(float64(int32(s))/float64(1<<31), 5))
		default:
			out[i] = float32(int32(s)) / float32(1<<31) * 0.1
		}
	}
	return out
}
