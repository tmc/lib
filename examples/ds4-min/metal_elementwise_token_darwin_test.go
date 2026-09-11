//go:build darwin && arm64

package main

import (
	"encoding/binary"
	"math"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// TestMetalElementwiseTokenIndependence checks that the normalization and
// element-wise encoders produce bit-identical rows whether a token's row is
// dispatched alone or as one row of a wider buffer.
//
// This is the property the token-major widening depends on. The matmul
// encoders needed a tokens parameter because their kernels index a weight
// matrix per row; these five do not, and reading the kernels says why:
//
//	rms_norm            offsets by group*width and reduces within one group
//	swiglu              indexes a flat i with no row concept at all
//	swiglu_weighted_slots reads weights[i/width] off the same flat i
//	clamp10             indexes a flat i
//
// So a T-row call is already T independent row computations, and no row's
// arithmetic or reduction order changes with T. That makes the widening free
// at these call sites -- but "free" is a claim about kernels that someone will
// later be tempted to optimize, and a cross-row reduction in rms_norm or a
// threadgroup-wide shuffle in swiglu would be invisible at the call site and
// would move the registered logits digest. This test is what makes that
// attempt fail loudly instead.
//
// Bit identity is the bar, not closeness: these rows feed matmuls whose
// outputs are pinned by digest, so a single changed ULP here is a changed
// digest downstream.
func TestMetalElementwiseTokenIndependence(t *testing.T) {
	const (
		tokens = 5
		width  = 512
		slots  = 6
	)

	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()

	// Row r element i, a pure function of both so a row carries the same
	// values whether it is dispatched alone or inside the wide buffer.
	value := func(r, i int) float32 {
		return float32((r*37+i*13)%29-14) / 8
	}
	fill := func(rows, per int, off int) []float32 {
		v := make([]float32, rows*per)
		for r := 0; r < rows; r++ {
			for i := 0; i < per; i++ {
				v[r*per+i] = value(r+off, i)
			}
		}
		return v
	}

	for _, tc := range []struct {
		name string
		// run encodes the operation over rows consecutive rows, the first of
		// which is row base, and returns the output.
		run func(t *testing.T, base, rows int) []float32
	}{
		{"rms", func(t *testing.T, base, rows int) []float32 {
			arena := metalmem.NewArena(dev)
			defer arena.Free()
			in := metalmem.UploadIn(arena, "in", fill(rows, width, base))
			out := metalmem.AllocIn[float32](arena, "out", rows*width)
			w := metalmem.AllocIn[byte](arena, "w", 4*width)
			for i := 0; i < width; i++ {
				binary.LittleEndian.PutUint32(w.HostSlice()[4*i:], math.Float32bits(1+float32(i%7)/8))
			}
			weight := boundMetalTensor{name: "norm", typeID: 0, dimensions: []uint64{width}, data: w}
			batch, err := newMetalBatch(ops)
			if err != nil {
				t.Fatal(err)
			}
			if err := batch.EncodeRMS(in, weight, out, rows, true); err != nil {
				t.Fatal(err)
			}
			if err := batch.Wait(t.Context()); err != nil {
				t.Fatal(err)
			}
			return append([]float32(nil), out.HostSlice()...)
		}},
		{"clamp10", func(t *testing.T, base, rows int) []float32 {
			arena := metalmem.NewArena(dev)
			defer arena.Free()
			in := metalmem.UploadIn(arena, "in", fill(rows, width, base))
			out := metalmem.AllocIn[float32](arena, "out", rows*width)
			batch, err := newMetalBatch(ops)
			if err != nil {
				t.Fatal(err)
			}
			if err := batch.EncodeClamp10(in, out); err != nil {
				t.Fatal(err)
			}
			if err := batch.Wait(t.Context()); err != nil {
				t.Fatal(err)
			}
			return append([]float32(nil), out.HostSlice()...)
		}},
		{"swiglu", func(t *testing.T, base, rows int) []float32 {
			arena := metalmem.NewArena(dev)
			defer arena.Free()
			gate := metalmem.UploadIn(arena, "gate", fill(rows, width, base))
			up := metalmem.UploadIn(arena, "up", fill(rows, width, base+3))
			out := metalmem.AllocIn[float32](arena, "out", rows*width)
			batch, err := newMetalBatch(ops)
			if err != nil {
				t.Fatal(err)
			}
			if err := batch.EncodeSwiGLU(gate, up, out); err != nil {
				t.Fatal(err)
			}
			if err := batch.Wait(t.Context()); err != nil {
				t.Fatal(err)
			}
			return append([]float32(nil), out.HostSlice()...)
		}},
		{"swiglu-slots", func(t *testing.T, base, rows int) []float32 {
			// The one case with a within-row structure: rows of slots*width
			// against weights of slots, so the kernel's weights[i/width] has
			// to land on the right slot of the right row.
			//
			// Note what the encoder passes as the kernel's slots argument:
			// weights.Len(), which at T rows is T*6, not 6. The kernel has no
			// per-row slot count and does not need one -- i/width already
			// walks the whole flattened [row][slot] axis. Mutating the index
			// to (i/width)%slots is therefore the identity here and the test
			// stays green, which is correct rather than a gap: that mutation
			// does not change the mapping. Breaking the row apart takes a
			// mutation that stops distinguishing rows at all, such as
			// weights[0], and that one does go red.
			arena := metalmem.NewArena(dev)
			defer arena.Free()
			gate := metalmem.UploadIn(arena, "gate", fill(rows, slots*width, base))
			up := metalmem.UploadIn(arena, "up", fill(rows, slots*width, base+3))
			weights := metalmem.UploadIn(arena, "weights", fill(rows, slots, base+11))
			out := metalmem.AllocIn[float32](arena, "out", rows*slots*width)
			batch, err := newMetalBatch(ops)
			if err != nil {
				t.Fatal(err)
			}
			if err := batch.EncodeSwiGLUWeightedSlots(gate, up, weights, out); err != nil {
				t.Fatal(err)
			}
			if err := batch.Wait(t.Context()); err != nil {
				t.Fatal(err)
			}
			return append([]float32(nil), out.HostSlice()...)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			wide := tc.run(t, 0, tokens)
			per := len(wide) / tokens
			for token := 0; token < tokens; token++ {
				alone := tc.run(t, token, 1)
				if len(alone) != per {
					t.Fatalf("token %d alone produced %d values, want %d", token, len(alone), per)
				}
				row := wide[token*per : (token+1)*per]
				for i := range alone {
					if math.Float32bits(row[i]) != math.Float32bits(alone[i]) {
						t.Fatalf("token %d element %d: batched %08x (%v), alone %08x (%v)",
							token, i, math.Float32bits(row[i]), row[i],
							math.Float32bits(alone[i]), alone[i])
					}
				}
			}
		})
	}
}
