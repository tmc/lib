package main

// Correctness of the fused attention kernel against cudaref's online-attention
// oracle. The kernel is the only thing under test, so these run without a model
// and need only a CUDA device.

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaref"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

// attnCase is one attention geometry: B sequences of T padded positions with
// real lengths lens, nh heads of flashHeadDim columns, and a symmetric window
// of win (negative for full attention).
type attnCase struct {
	name      string
	B, T, nh  int
	lens      []int32
	win       int
	allMasked bool // at least one query position past its sequence length
}

// tf32 rounds x the way cvt.rna.tf32.f32 does, so the reference sees the same
// operands the tensor cores do and the comparison measures the algorithm rather
// than the operand precision.
func tf32(x float32) float32 {
	b := math.Float32bits(x)
	if b&0x7f800000 == 0x7f800000 { // inf or NaN round to themselves
		return x
	}
	return math.Float32frombits((b + 0x1000 + ((b >> 13) & 1)) &^ 0x1fff)
}

// attnRef computes the expected output with cudaref's oracle, row by row, using
// attend for the per-row attention. It returns [B*T, nh*headDim] row-major, the
// layout the kernel writes.
func attnRef(t *testing.T, c attnCase, q, k, v []float32,
	attend func(scores, values []float32, mask []bool, seqLen, headDim, tileSize int) ([]float32, error),
) []float32 {
	t.Helper()
	hd, d := flashHeadDim, c.nh*flashHeadDim
	scale := float32(1 / math.Sqrt(float64(hd)))
	out := make([]float32, c.B*c.T*d)
	scores := make([]float32, c.T)
	mask := make([]bool, c.T)
	for b := range c.B {
		length := int(c.lens[b])
		for h := range c.nh {
			base := ((b * c.nh) + h) * c.T * hd
			for i := range c.T {
				lo, hi := 0, -1
				if i < length {
					lo, hi = 0, length-1
					if c.win >= 0 {
						if i-c.win > 0 {
							lo = i - c.win
						}
						if i+c.win < length-1 {
							hi = i + c.win
						}
					}
				}
				for j := range c.T {
					mask[j] = j < lo || j > hi
					var dot float64
					for x := range hd {
						dot += float64(tf32(q[base+i*hd+x])) * float64(tf32(k[base+j*hd+x]))
					}
					scores[j] = float32(dot) * scale
				}
				row, err := attend(scores, v[base:base+c.T*hd], mask, c.T, hd, flashTile)
				if err != nil {
					t.Fatalf("oracle: %v", err)
				}
				copy(out[(b*c.T+i)*d+h*hd:], row)
			}
		}
	}
	return out
}

// attnRun launches FlashAttention over one case and returns its output.
func attnRun(t *testing.T, dev *cuda.Device, k *kernels, c attnCase, q, kk, v []float32) []float32 {
	t.Helper()
	hd := flashHeadDim
	got := make([]float32, c.B*c.T*c.nh*hd)
	if err := dev.WithContext(func() error {
		stream, err := dev.NewStream()
		if err != nil {
			return err
		}
		defer stream.Destroy()

		var free []func() error
		defer func() {
			for _, f := range free {
				f()
			}
		}()
		up := func(src []float32) (cudamem.Slice[float32], error) {
			s, err := cudamem.Alloc[float32](dev, len(src))
			if err != nil {
				return s, err
			}
			free = append(free, s.Free)
			return s, s.CopyFromHost(src)
		}
		dq, err := up(q)
		if err != nil {
			return err
		}
		dk, err := up(kk)
		if err != nil {
			return err
		}
		dv, err := up(v)
		if err != nil {
			return err
		}
		do, err := up(got)
		if err != nil {
			return err
		}
		dl, err := cudamem.Alloc[int32](dev, c.B)
		if err != nil {
			return err
		}
		free = append(free, dl.Free)
		if err := dl.CopyFromHost(c.lens); err != nil {
			return err
		}
		if err := cudalaunch.Kernel(k.flashAttn).
			Grid(uint32((c.T+flashTile-1)/flashTile), uint32(c.nh), uint32(c.B)).
			Block(128).On(stream).
			Launch(dq, dk, dv, dl, do,
				int32(c.B), int32(c.T), int32(c.nh), int32(c.win),
				float32(1/math.Sqrt(float64(hd)))); err != nil {
			return err
		}
		if err := stream.Synchronize(nil); err != nil {
			return err
		}
		return do.CopyToHost(got)
	}); err != nil {
		t.Fatal(err)
	}
	return got
}

// maxDelta returns the largest absolute difference between a and b.
func maxDelta(a, b []float32) float64 {
	var mx float64
	for i := range a {
		if v := math.Abs(float64(a[i] - b[i])); v > mx {
			mx = v
		}
	}
	return mx
}

func attnInputs(c attnCase) (q, k, v []float32) {
	r := rand.New(rand.NewPCG(1, 2))
	n := c.B * c.nh * c.T * flashHeadDim
	q, k, v = make([]float32, n), make([]float32, n), make([]float32, n)
	for i := range n {
		q[i] = float32(r.NormFloat64())
		k[i] = float32(r.NormFloat64())
		v[i] = float32(r.NormFloat64())
	}
	return q, k, v
}

var attnCases = []attnCase{
	{name: "full_one_tile", B: 1, T: 37, nh: 2, lens: []int32{37}, win: -1},
	{name: "full_ragged", B: 2, T: 100, nh: 2, lens: []int32{100, 61}, win: -1, allMasked: true},
	{name: "full_padded_tail", B: 2, T: 200, nh: 1, lens: []int32{200, 3}, win: -1, allMasked: true},
	{name: "window_64", B: 2, T: 200, nh: 2, lens: []int32{200, 137}, win: 64, allMasked: true},
	{name: "window_8", B: 1, T: 130, nh: 1, lens: []int32{130}, win: 8},
}

func TestFlashAttention(t *testing.T) {
	dev := cudatest.Require(t)
	var k *kernels
	if err := dev.WithContext(func() error {
		var err error
		k, err = compileKernels(dev)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := dev.WithContext(k.Close); err != nil {
			t.Error(err)
		}
	})

	// tol covers tf32 rounding of the probabilities and of V, which the oracle
	// keeps in float64; it is far below the spread the mutants produce.
	const tol = 5e-3
	for _, c := range attnCases {
		t.Run(c.name, func(t *testing.T) {
			q, kk, v := attnInputs(c)
			got := attnRun(t, dev, k, c, q, kk, v)
			want := attnRef(t, c, q, kk, v, cudaref.OnlineAttentionRow)
			if d := maxDelta(got, want); d > tol {
				t.Fatalf("max |delta| = %g, want <= %g", d, tol)
			} else {
				t.Logf("max |delta| = %g", d)
			}
			if c.allMasked {
				for i, x := range got {
					if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
						t.Fatalf("out[%d] = %g, want finite", i, x)
					}
				}
			}
		})
	}
}

// TestFlashAttentionMutants shows the tolerance above is tight enough to reject
// the two deliberate oracle mutants, so a rescale or normalizer bug in the
// kernel could not pass TestFlashAttention.
func TestFlashAttentionMutants(t *testing.T) {
	dev := cudatest.Require(t)
	var k *kernels
	if err := dev.WithContext(func() error {
		var err error
		k, err = compileKernels(dev)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := dev.WithContext(k.Close); err != nil {
			t.Error(err)
		}
	})

	const tol = 5e-3
	for _, mut := range []struct {
		name string
		fn   func(scores, values []float32, mask []bool, seqLen, headDim, tileSize int) ([]float32, error)
	}{
		{"rescale", cudaref.OnlineAttentionRowMutantRescale},
		{"no_normalizer", cudaref.OnlineAttentionRowMutantNoNormalizer},
	} {
		t.Run(mut.name, func(t *testing.T) {
			for _, c := range attnCases {
				if c.T <= flashTile {
					continue // a single tile leaves the mutants nothing to corrupt
				}
				q, kk, v := attnInputs(c)
				got := attnRun(t, dev, k, c, q, kk, v)
				bad := attnRef(t, c, q, kk, v, mut.fn)
				if d := maxDelta(got, bad); d <= tol {
					t.Fatalf("%s: max |delta| to the mutant = %g, want > %g", c.name, d, tol)
				}
			}
		})
	}
}
