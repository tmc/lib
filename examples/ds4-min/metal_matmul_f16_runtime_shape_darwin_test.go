//go:build darwin && arm64

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// The runtime's F16 projections do not use the kernel EncodeMatmulF16 batches.
//
// MatvecF16 routes 16384x4 -- the output head -- to o.head.matvec, a SIMD kernel
// with a two-level simd_sum reduction, and everything else to matvec_f16, which
// walks a single fp32 accumulator over the k loop. EncodeMatmulF16 is a clone of
// the first. TestMetalMatmulPreservedBitIdentity covers it at exactly 16384x4,
// which is the one shape that takes the head route, so the batched kernel has
// never been compared against the serial one the runtime actually runs.
//
// The five call sites that matter are the attention compressor KV and gate, the
// indexer compressor KV and gate, and the router. They are every
// EncodeMatvecF16 in metal_runtime_darwin.go and nothing else, so
//
//	grep -n EncodeMatvecF16 metal_runtime_darwin.go
//
// names them; an earlier revision of this comment carried line numbers instead
// and they were four commits stale by the time anyone read them. Their weights
// are 4096x1024 and 4096x256 -- neither is 16384x4, so all five take
// matvec_f16.
//
// Measured at both shapes: they disagree on ~97% of outputs, at T=1 as much as
// at T=8. The difference is not a batching artifact -- the two kernels are not
// the same function, and one token is enough to show it.
//
// The divergence is not at rounding scale. Measured worst relative difference
// per case: 8.9e-4, 2.6e-3 and 1.6e-2 at 4096x1024 for T of 1, 3 and 8, and
// 8.4e-5, 7.4e-4 and 7.3e-4 at 4096x256. All but one case exceed the registered
// 1e-4, so the frozen-tolerance gates would catch this swap as readily as the
// digest -- which is worth knowing, because it means the constraint does not
// depend on the digest alone.
//
// So this asserts the disagreement. Making the two agree is a legitimate goal --
// it is what would let the five projections batch -- but it changes the logits
// digest, and this test going red is the signal to re-register that digest
// deliberately rather than discover the change downstream.
//
// It is model-free: it builds its own weights and needs no GGUF and no model
// slot.
func TestMetalMatmulF16AtRuntimeShape(t *testing.T) {
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

	mm, err := newMetalMatmulPreservedF16(ops)
	if err != nil {
		t.Fatal(err)
	}
	defer mm.Close()

	arena := metalmem.NewArena(dev)
	defer arena.Free()

	// The attention compressor KV projection (ds4Width x 1024) and the router
	// and indexer compressors (ds4Width x 256).
	const k = ds4Width
	shapes := []int{1024, 256}

	rng := uint32(0x9e3779b9)
	next := func() uint32 {
		rng = rng*1664525 + 1013904223
		return rng
	}

	for _, n := range shapes {
		// Same construction as TestMetalMatmulPreservedBitIdentity: exponents
		// spread around the bias so partial sums do not land exactly and the
		// reduction order is observable. requireOrderSensitive holds that
		// property to account, so a measured agreement would mean something.
		weightBytes := metalmem.AllocIn[byte](arena, fmt.Sprintf("f16-weight-%d", n), 2*k*n)
		for i := 0; i < k*n; i++ {
			r := next()
			bits := uint16(r&0x8000) | uint16(9+(r>>16)%13)<<10 | uint16((r>>4)&0x03ff)
			binary.LittleEndian.PutUint16(weightBytes.HostSlice()[2*i:], bits)
		}
		weightTensor := boundMetalTensor{
			name:       fmt.Sprintf("f16.%dx%d.weight", k, n),
			typeID:     1,
			dimensions: []uint64{k, uint64(n)},
			data:       weightBytes,
		}

		for _, T := range []int{1, 3, 8} {
			t.Run(fmt.Sprintf("n=%d/T=%d", n, T), func(t *testing.T) {
				activations := make([]float32, k*T)
				for i := range activations {
					r := next()
					activations[i] = float32(int32(r>>12)-524288) / float32(int32(1)<<(4+r%9))
				}
				requireOrderSensitive(t, weightBytes.HostSlice(), activations, k, n, T)

				serial := make([]float32, n*T)
				for tok := 0; tok < T; tok++ {
					res, err := ops.MatvecF16(weightTensor, activations[tok*k:(tok+1)*k])
					if err != nil {
						t.Fatalf("ops.MatvecF16 token %d: %v", tok, err)
					}
					copy(serial[tok*n:(tok+1)*n], res)
				}

				batched, err := mm.Mul(weightTensor, activations, T)
				if err != nil {
					t.Fatalf("mm.Mul(T=%d): %v", T, err)
				}

				differing, worst := 0, 0.0
				for i := range serial {
					if math.Float32bits(batched[i]) != math.Float32bits(serial[i]) {
						differing++
						if d := relative(batched[i], serial[i]); d > worst {
							worst = d
						}
					}
				}
				if differing == 0 {
					t.Fatalf("EncodeMatmulF16 now agrees with matvec_f16 at %dx%d, T=%d. "+
						"That is a real change to the five runtime F16 projections and it moves the "+
						"logits digest; re-register metalRegionProfileLogitsSHA256 from a measured "+
						"anchor run and update this test deliberately.", k, n, T)
				}
				t.Logf("%dx%d T=%d: %d of %d outputs differ, worst relative %g", k, n, T, differing, n*T, worst)
			})
		}
	}
}

// relative reports the relative difference between a and b, falling back to the
// absolute difference when b is zero.
func relative(a, b float32) float64 {
	d := math.Abs(float64(a) - float64(b))
	if b == 0 {
		return d
	}
	return d / math.Abs(float64(b))
}
