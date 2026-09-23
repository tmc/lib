//go:build linux && !tpu

package main

import (
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
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

// requireCUDAHardware returns a device satisfying reqs, or skips.
//
// The LIB_CUDA_HARDWARE_TEST opt-in stays: these tests compile kernels and
// allocate tens of MiB on whatever device they find, which is not something a
// shared host should get from an ordinary `go test`. Everything after the
// opt-in is delegated to cudatest.Require, which distinguishes an absent
// driver from a broken one: a machine with no libcuda skips, while a libcuda
// that loads and then fails to initialize is a failure. The previous guard
// called cuda.NewDevice directly and reported both as the same t.Fatal, so a
// host without a GPU failed the suite instead of skipping it.
//
// A skip establishes nothing. None of these tests has been executed by the
// author; see the package comment above.
func requireCUDAHardware(t *testing.T, reqs ...cudatest.Requirement) *cuda.Device {
	t.Helper()
	if os.Getenv("LIB_CUDA_HARDWARE_TEST") != "1" {
		t.Skip("set LIB_CUDA_HARDWARE_TEST=1 on a CUDA host")
	}
	dev := cudatest.Require(t, reqs...)
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
// No compute-capability requirement is declared for the half weight tests.
//
// sm_53 would be the floor for __half arithmetic, but forward.cu does none:
// f16_to_f32 is a single inline cvt.f32.f16, a conversion instruction far
// older than half arithmetic, and bf16_to_f32 is an integer shift with no
// half support at all. Requiring sm_53 would skip devices these kernels run
// on. cudatest.MinCompute and MinVRAM are available through the variadic
// above for a test that has a real floor to declare.

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

				var tracker cudamem.Tracker
				arena := cudamem.NewTrackedArena(dev, &tracker)
				defer arena.Free()
				dx := cudamem.AllocIn[float32](arena, "x", len(x))
				dout := cudamem.AllocIn[float32](arena, "out", dim.out)
				if err := arena.Err(); err != nil {
					t.Fatal(err)
				}
				if err := dx.CopyFromHost(x); err != nil {
					t.Fatal(err)
				}
				before, _ := tracker.Snapshot()
				dw, err := uploadWeights(arena, d, w)
				if err != nil {
					t.Fatal(err)
				}
				// Measured at the allocator, not recomputed from the same
				// bookkeeping the claim is about: WeightSlice.Bytes() is
				// defined as elems*dtype.Bytes(), so comparing it against
				// elems*dtype.Bytes() would hold even if the device allocation
				// were four bytes per weight.
				after, _ := tracker.Snapshot()
				if got, want := after-before, uint64(dim.in*dim.out*d.Bytes()); got != want {
					t.Fatalf("weight upload allocated %d device bytes, want %d", got, want)
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

// TestCUDADeviceConversionMatchesGo checks the device's widening against Go's,
// over every one of the 65536 bit patterns of each format.
//
// The comparison runs through WeightConvertProbe, which loads a weight and
// stores it with no arithmetic. An earlier version of this test read the
// conversion out through GEMV, which was wrong: a correctly widened negative
// zero becomes a positive zero as soon as it enters an accumulator, so the
// dot product could not distinguish a working sign bit from a broken one. That
// is precisely one of the cases this test exists to cover.
//
// The three classes of input are treated differently, on purpose:
//
//   - Finite values are compared bit for bit, negative zero included. 63488 of
//     the f16 patterns and 65280 of the bf16 patterns are finite.
//   - Infinities are compared bit for bit too. Both formats encode them
//     uniquely, so there is nothing to qualify: 2 patterns per format.
//   - NaNs are required to widen to some NaN, and nothing is claimed about the
//     payload. Neither PTX cvt.f32.f16 nor a bfloat16 shift guarantees which
//     NaN comes out, and the Go conversions do not either. 2046 f16 patterns
//     and 254 bf16 patterns are NaN.
//
// The three counts sum to 65536 in each format, and the test asserts that, so a
// classification that silently dropped a class could not pass.
func TestCUDADeviceConversionMatchesGo(t *testing.T) {
	dev := requireCUDAHardware(t)
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()

	for _, tt := range []struct {
		d                     WeightDType
		finite, infinite, nan int
	}{
		{WeightF16, 63488, 2, 2046},
		{WeightBF16, 65280, 2, 254},
	} {
		t.Run(tt.d.String(), func(t *testing.T) {
			d := tt.d
			if tt.finite+tt.infinite+tt.nan != 1<<16 {
				t.Fatalf("declared counts sum to %d, not 65536", tt.finite+tt.infinite+tt.nan)
			}
			modules, err := CompileGPUModulesAs(dev, d)
			if err != nil {
				t.Fatal(err)
			}
			defer modules.Close()
			probe, err := modules.mod.Function("WeightConvertProbe" + d.Suffix())
			if err != nil {
				t.Fatalf("the conversion probe is missing from the module: %v", err)
			}

			const n = 1 << 16
			patterns := make([]uint16, n)
			for i := range patterns {
				patterns[i] = uint16(i)
			}

			arena := cudamem.NewArena(dev)
			defer arena.Free()
			dw := cudamem.AllocIn[uint16](arena, "patterns", n)
			dout := cudamem.AllocIn[float32](arena, "out", n)
			if err := arena.Err(); err != nil {
				t.Fatal(err)
			}
			if err := dw.CopyFromHost(patterns); err != nil {
				t.Fatal(err)
			}
			if err := cudalaunch.OwnedKernel(probe).Grid(256).Block(256).On(stream).
				Launch(dw, dout, n); err != nil {
				t.Fatalf("launch probe: %v", err)
			}
			if err := stream.Synchronize(t.Context()); err != nil {
				t.Fatal(err)
			}
			got := make([]float32, n)
			if err := dout.CopyToHost(got); err != nil {
				t.Fatal(err)
			}

			widen := float16ToFloat32
			isNaNPattern := func(h uint16) bool { return h&0x7c00 == 0x7c00 && h&0x3ff != 0 }
			isInfPattern := func(h uint16) bool { return h&0x7c00 == 0x7c00 && h&0x3ff == 0 }
			if d == WeightBF16 {
				widen = bfloat16ToFloat32
				isNaNPattern = func(b uint16) bool { return b&0x7f80 == 0x7f80 && b&0x7f != 0 }
				isInfPattern = func(b uint16) bool { return b&0x7f80 == 0x7f80 && b&0x7f == 0 }
			}

			var finite, infinite, nan, bad int
			for i, h := range patterns {
				g := got[i]
				if isNaNPattern(h) {
					nan++
					// Payload is not claimed; only that it stays a NaN.
					if !math.IsNaN(float64(g)) {
						if bad < 10 {
							t.Errorf("NaN pattern 0x%04x widened to %v, want some NaN", h, g)
						}
						bad++
					}
					continue
				}
				if isInfPattern(h) {
					infinite++
				} else {
					finite++
				}
				want := widen(h)
				if math.Float32bits(g) != math.Float32bits(want) {
					if bad < 10 {
						t.Errorf("pattern 0x%04x widened to %v (bits 0x%08x) on device, %v (bits 0x%08x) in Go",
							h, g, math.Float32bits(g), want, math.Float32bits(want))
					}
					bad++
				}
			}
			if finite != tt.finite || infinite != tt.infinite || nan != tt.nan {
				t.Errorf("classified %d finite, %d infinite, %d NaN; declared %d, %d, %d",
					finite, infinite, nan, tt.finite, tt.infinite, tt.nan)
			}
			if bad > 0 {
				t.Fatalf("%d of 65536 patterns disagree", bad)
			}

			// Negative zero specifically, since it is the case the previous
			// version of this test could not see.
			var negZero uint16 = 0x8000
			if g := got[negZero]; math.Float32bits(g) != 0x80000000 {
				t.Errorf("negative zero widened to bits 0x%08x, want 0x80000000", math.Float32bits(g))
			}
			t.Logf("%d finite, %d infinite bit patterns widen identically; %d NaN patterns stay NaN",
				finite, infinite, nan)
		})
	}
}

// TestCUDASingleAndBatchFusedAgreeOnOddDimensions compares the single-token
// fused SwiGLU-down kernel against its batch counterpart at batch size one.
//
// The batch kernel always summed every element of the reduction; the
// single-token kernel dropped the last intermediate_dim % 4 terms until this
// branch gave it a tail loop. The two kernels multiply identical values in a
// different thread layout, so only float32 reassociation separates them from
// each other. Both are compared against a Go oracle, which introduces a second
// source of difference the reassociation argument does not cover: the oracle
// evaluates the SwiGLU sigmoid with math.Exp in float64 while the device uses
// expf. expf is accurate to a couple of ulps in float32, so that contributes a
// relative error of a few times 1e-7 to each term, still several times inside
// implRelTol. A dimension that is a multiple of four is included as the control
// that passed even before the fix.
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

// TestCUDAEmbeddingLookupOddStride checks the embedding row load at strides
// that leave a row base unaligned.
//
// The row for token tok starts at embed_table + tok*hidden_dim, so token 0 is
// always aligned and any later token is not when hidden_dim is odd. An
// unguarded vector load therefore faults on the second token and not the first,
// which is why every token index below is exercised rather than just one.
//
// The comparison is bit-exact with no tolerance: the kernel widens and stores,
// performing no arithmetic, so its output must equal the once-rounded weights
// exactly. Negative zeros in the table are preserved by that path and are
// included deliberately.
func TestCUDAEmbeddingLookupOddStride(t *testing.T) {
	dev := requireCUDAHardware(t)
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()

	for _, d := range []WeightDType{WeightF32, WeightF16, WeightBF16} {
		modules, err := CompileGPUModulesAs(dev, d)
		if err != nil {
			t.Fatalf("%s: compile: %v", d, err)
		}
		for _, hidden := range []int{5, 128, 129, 130, 131} {
			for _, tok := range []int{0, 1, 2, 3} {
				name := d.String() + "/hidden" + itoa(hidden) + "/tok" + itoa(tok)
				t.Run(name, func(t *testing.T) {
					const vocab = 5
					table := deterministicFloats(vocab*hidden, 7)
					rt := d.RoundTrip(table)

					arena := cudamem.NewArena(dev)
					defer arena.Free()
					dtok := cudamem.AllocIn[int32](arena, "tok", 1)
					dout := cudamem.AllocIn[float32](arena, "out", hidden)
					if err := arena.Err(); err != nil {
						t.Fatal(err)
					}
					if err := dtok.CopyFromHost([]int32{int32(tok)}); err != nil {
						t.Fatal(err)
					}
					dw, err := uploadWeights(arena, d, table)
					if err != nil {
						t.Fatal(err)
					}
					if err := cudalaunch.OwnedKernel(modules.Embed).Grid(4).Block(64).On(stream).
						Launch(dw, dtok, dout, hidden); err != nil {
						t.Fatalf("launch: %v", err)
					}
					if err := stream.Synchronize(t.Context()); err != nil {
						t.Fatal(err)
					}
					got := make([]float32, hidden)
					if err := dout.CopyToHost(got); err != nil {
						t.Fatal(err)
					}

					want := rt[tok*hidden : (tok+1)*hidden]
					for i := range want {
						if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
							t.Fatalf("element %d = %v (bits 0x%08x), want %v (bits 0x%08x)",
								i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
						}
					}
				})
			}
		}
		modules.Close()
	}
}
