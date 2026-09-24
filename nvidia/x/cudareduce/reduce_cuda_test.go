package cudareduce_test

import (
	"context"
	"errors"
	"math"
	"math/rand/v2"
	"os"
	"testing"
	"time"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudareduce"
	"github.com/tmc/lib/nvidia/x/cudasanitizer"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// shapes covers a single element, partial column blocks, one split and many.
var shapes = []struct{ rows, n int }{
	{1, 1}, {7, 300}, {33, 256}, {1000, 513}, {4096, 768}, {20000, 3},
}

func upload[T any](t *testing.T, dev *cuda.Device, host []T) cudamem.Slice[T] {
	t.Helper()
	s, err := cudamem.Alloc[T](dev, max(len(host), 1))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Free() })
	if len(host) > 0 {
		if err := s.CopyFromHost(host); err != nil {
			t.Fatal(err)
		}
	}
	return s.Slice(0, len(host))
}

func download(t *testing.T, s cudamem.Slice[float32]) []float32 {
	t.Helper()
	out := make([]float32, s.Len())
	if err := s.CopyToHost(out); err != nil {
		t.Fatal(err)
	}
	return out
}

func randn(r *rand.Rand, n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	return x
}

// near requires got to match the float64 reference want to 1e-6 relative,
// with an absolute floor for sums that cancel, scaled by the sum of the
// terms' magnitudes.
func near(t *testing.T, what string, got []float32, want, mag []float64) {
	t.Helper()
	for i := range want {
		d := math.Abs(float64(got[i]) - want[i])
		if d > 1e-6*math.Abs(want[i])+1e-12*mag[i] {
			t.Fatalf("%s[%d] = %v, want %v", what, i, got[i], want[i])
		}
	}
}

func setup(t *testing.T) (*cuda.Device, *cuda.Stream, *cudareduce.Reducer) {
	dev := cudatest.Require(t)
	red, err := cudareduce.New(dev)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { red.Close() })
	return dev, dev.DefaultStream(), red
}

func sync(t *testing.T, s *cuda.Stream) {
	t.Helper()
	if err := s.Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestColumnSum(t *testing.T) {
	dev, stream, red := setup(t)
	r := rand.New(rand.NewPCG(1, 1))
	for _, sh := range shapes {
		x := randn(r, sh.rows*sh.n)
		want, mag := make([]float64, sh.n), make([]float64, sh.n)
		for i, v := range x {
			want[i%sh.n] += float64(v)
			mag[i%sh.n] += math.Abs(float64(v))
		}
		prior := randn(r, sh.n)
		xs := upload(t, dev, x)
		out := upload(t, dev, prior)
		scratch := upload(t, dev, make([]float64, cudareduce.ScratchLen(sh.rows, sh.n, 1)))

		if err := red.ColumnSum(stream, out, xs, sh.rows, sh.n, scratch, false); err != nil {
			t.Fatal(err)
		}
		sync(t, stream)
		first := download(t, out)
		near(t, "sum", first, want, mag)

		// Deterministic: a second run gives the same bits.
		if err := red.ColumnSum(stream, out, xs, sh.rows, sh.n, scratch, false); err != nil {
			t.Fatal(err)
		}
		sync(t, stream)
		for i, v := range download(t, out) {
			if math.Float32bits(v) != math.Float32bits(first[i]) {
				t.Fatalf("%dx%d: element %d differs between runs", sh.rows, sh.n, i)
			}
		}

		// Accumulate adds to what is there.
		if err := out.CopyFromHost(prior); err != nil {
			t.Fatal(err)
		}
		if err := red.ColumnSum(stream, out, xs, sh.rows, sh.n, scratch, true); err != nil {
			t.Fatal(err)
		}
		sync(t, stream)
		acc := make([]float64, sh.n)
		for i := range acc {
			acc[i] = want[i] + float64(prior[i])
			mag[i] += math.Abs(float64(prior[i]))
		}
		near(t, "accumulated sum", download(t, out), acc, mag)
	}
}

func TestSegmentColumnSum(t *testing.T) {
	dev, stream, red := setup(t)
	r := rand.New(rand.NewPCG(2, 2))
	for _, groups := range []int{1, 5, cudareduce.MaxGroups} {
		for _, sh := range shapes {
			x := randn(r, sh.rows*sh.n)
			seg := make([]int32, sh.rows)
			for i := range seg {
				seg[i] = int32(r.IntN(groups+1)) - 1 // -1 is padding
			}
			want, mag := make([]float64, groups*sh.n), make([]float64, groups*sh.n)
			for i, v := range x {
				if g := seg[i/sh.n]; g >= 0 {
					want[int(g)*sh.n+i%sh.n] += float64(v)
					mag[int(g)*sh.n+i%sh.n] += math.Abs(float64(v))
				}
			}
			out := upload(t, dev, make([]float32, groups*sh.n))
			scratch := upload(t, dev, make([]float64, cudareduce.ScratchLen(sh.rows, sh.n, groups)))
			if err := red.SegmentColumnSum(stream, out, upload(t, dev, x), upload(t, dev, seg), sh.rows, sh.n, groups, scratch, false); err != nil {
				t.Fatal(err)
			}
			sync(t, stream)
			near(t, "segment sum", download(t, out), want, mag)
		}
	}
}

func TestLayerNormGrad(t *testing.T) {
	dev, stream, red := setup(t)
	r := rand.New(rand.NewPCG(3, 3))
	for _, sh := range shapes {
		for _, stats := range []bool{true, false} {
			dy, x := randn(r, sh.rows*sh.n), randn(r, sh.rows*sh.n)
			mean, rstd := randn(r, sh.rows), make([]float32, sh.rows)
			for i := range rstd {
				rstd[i] = float32(0.5 + r.Float64())
			}
			dw, db := make([]float64, sh.n), make([]float64, sh.n)
			mw, mb := make([]float64, sh.n), make([]float64, sh.n)
			for i := range x {
				row, c := i/sh.n, i%sh.n
				xh := float64(x[i])
				if stats {
					xh = (xh - float64(mean[row])) * float64(rstd[row])
				}
				dw[c] += float64(dy[i]) * xh
				mw[c] += math.Abs(float64(dy[i]) * xh)
				db[c] += float64(dy[i])
				mb[c] += math.Abs(float64(dy[i]))
			}
			dws := upload(t, dev, make([]float32, sh.n))
			dbs := upload(t, dev, make([]float32, sh.n))
			ms, rs := upload(t, dev, mean), upload(t, dev, rstd)
			if !stats {
				ms, rs = cudamem.Slice[float32]{}, cudamem.Slice[float32]{}
			}
			scratch := upload(t, dev, make([]float64, cudareduce.ScratchLen(sh.rows, sh.n, 2)))
			if err := red.LayerNormGrad(stream, dws, dbs, upload(t, dev, dy), upload(t, dev, x), ms, rs, sh.rows, sh.n, scratch, false); err != nil {
				t.Fatal(err)
			}
			sync(t, stream)
			near(t, "dw", download(t, dws), dw, mw)
			near(t, "db", download(t, dbs), db, mb)
		}
	}
}

// TestFusedFinish writes a stage 1 kernel of its own around the header, as
// a trainer fusing a reduction would, and finishes it with Finish.
func TestFusedFinish(t *testing.T) {
	dev, stream, red := setup(t)
	const src = `#include "cudareduce.cuh"
extern "C" __global__ void sumsq(const float* x, long long rows, long long n, long long rps, double* partials) {
    cudareduce_column_partials<1>(
        [=](long long r, long long c, double* acc) { double v = x[r * n + c]; acc[0] += v * v; },
        rows, n, rps, partials);
}`
	cm, err := nvrtcx.Compile(src, nvrtcx.WithDeviceArch(dev), nvrtcx.WithHeader("cudareduce.cuh", cudareduce.Header))
	if err != nil {
		t.Fatal(err)
	}
	mod, err := cm.LoadModule(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer mod.Close()
	k, err := mod.Function("sumsq")
	if err != nil {
		t.Fatal(err)
	}
	const rows, n = 1000, 300
	x := randn(rand.New(rand.NewPCG(4, 4)), rows*n)
	want, mag := make([]float64, n), make([]float64, n)
	for i, v := range x {
		want[i%n] += float64(v) * float64(v)
		mag[i%n] = want[i%n]
	}
	xs := upload(t, dev, x)
	scratch := upload(t, dev, make([]float64, cudareduce.ScratchLen(rows, n, 1)))
	out := upload(t, dev, make([]float32, n))
	splits, rps := cudareduce.Split(rows, n)
	xp, sp := xs.Ptr(), scratch.Ptr()
	r64, n64, rps64 := int64(rows), int64(n), int64(rps)
	err = stream.WithHandle(func(h driver.Stream) error {
		return k.Launch([3]uint32{(n + cudareduce.Threads - 1) / cudareduce.Threads, uint32(splits), 1},
			[3]uint32{cudareduce.Threads, 1, 1}, 0, h,
			[]unsafe.Pointer{unsafe.Pointer(&xp), unsafe.Pointer(&r64), unsafe.Pointer(&n64), unsafe.Pointer(&rps64), unsafe.Pointer(&sp)})
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := red.Finish(stream, out, scratch, rows, n, 1, false); err != nil {
		t.Fatal(err)
	}
	sync(t, stream)
	near(t, "sum of squares", download(t, out), want, mag)
}

// TestRacecheck reruns the reduction tests under compute-sanitizer
// racecheck, and memcheck, and requires a clean report. It is skipped when
// compute-sanitizer is not installed.
func TestRacecheck(t *testing.T) {
	cudatest.Require(t) // the parent holds the GPU lock for the child
	if testing.Short() {
		t.Skip("skipping sanitizer runs in short mode")
	}
	for _, tool := range []cudasanitizer.Tool{cudasanitizer.Racecheck, cudasanitizer.Memcheck} {
		t.Run(string(tool), func(t *testing.T) {
			res, err := cudasanitizer.Run(context.Background(), cudasanitizer.Config{
				Tool:          tool,
				MissingTool:   cudasanitizer.MissingToolSkip,
				Env:           append(os.Environ(), "CUDATEST_NO_LOCK=1"),
				Timeout:       10 * time.Minute,
				RequireOutput: []string{"--- PASS: TestColumnSum", "--- PASS: TestSegmentColumnSum", "--- PASS: TestLayerNormGrad", "--- PASS: TestFusedFinish"},
			}, os.Args[0], "-test.run=^Test(ColumnSum|SegmentColumnSum|LayerNormGrad|FusedFinish)$", "-test.v")
			if err != nil {
				t.Fatal(err)
			}
			if res.Outcome != cudasanitizer.OutcomeClean {
				if errors.Is(res.Err, cudasanitizer.ErrToolNotFound) {
					t.Skip("compute-sanitizer not installed")
				}
				t.Fatalf("%s: outcome %v, err %v, missing %q\n%s\n%s", tool, res.Outcome, res.Err, res.MissingEvidence, res.Log.Text, res.Stdout)
			}
			t.Logf("%s clean: %v", tool, res.Args)
		})
	}
}
