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

// floatToF16Bits encodes a float32 to IEEE 754 binary16 uint16.
func floatToF16Bits(val float32) uint16 {
	bits := math.Float32bits(val)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits >> 23) & 0xff) - 127 + 15
	mant := bits & 0x007fffff

	if exp <= 0 {
		return sign
	} else if exp >= 31 {
		return sign | 0x7c00
	}
	return sign | uint16(exp<<10) | uint16(mant>>13)
}

func TestMetalMatmulPreservedBitIdentity(t *testing.T) {
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

	const k = 16384
	const n = 4 // Matches output head projection

	// F16 weights spanning 2^-6 to 2^6, built as bit patterns so every value
	// is exactly representable by construction: sign, an exponent drawn from
	// [9,21] around the bias of 15, and a full random mantissa.
	//
	// The dynamic range is the point. An earlier version of this test drew
	// weights from {-2,-1.75,..,2} and activations from {-0.75,..,0.75}, so
	// every product was a multiple of 2^-5 with magnitude below 1.5 and every
	// partial sum over 16384 terms fit float32's 24-bit mantissa exactly.
	// Exact partial sums are order-independent, so the test passed with the
	// k-loop reversed -- it could not see the one property it exists to check.
	weightBytes := metalmem.AllocIn[byte](arena, "head-weight", 2*k*n)
	rng := uint32(0x9e3779b9)
	next := func() uint32 {
		rng = rng*1664525 + 1013904223
		return rng
	}
	for i := 0; i < k*n; i++ {
		r := next()
		bits := uint16(r&0x8000) | uint16(9+(r>>16)%13)<<10 | uint16((r>>4)&0x03ff)
		binary.LittleEndian.PutUint16(weightBytes.HostSlice()[2*i:], bits)
	}
	weightTensor := boundMetalTensor{
		name:       "output.weight",
		typeID:     1,
		dimensions: []uint64{k, n},
		data:       weightBytes,
	}

	// Test multiple token counts: T=1 (must equal matvec), T=3 (odd/tail), T=8, T=13 (non-power of 2), T=64
	testTokens := []int{1, 3, 7, 8, 13, 64}

	for _, T := range testTokens {
		t.Run(fmt.Sprintf("T=%d", T), func(t *testing.T) {
			activations := make([]float32, k*T)
			for i := range activations {
				r := next()
				activations[i] = float32(int32(r>>12)-524288) / float32(int32(1)<<(4+r%9))
			}
			requireOrderSensitive(t, weightBytes.HostSlice(), activations, k, n, T)

			// 1. Run sequential reference using ops.MatvecF16 (which routes to ops.head) for each token
			wantOut := make([]float32, n*T)
			for tok := 0; tok < T; tok++ {
				tokIn := activations[tok*k : (tok+1)*k]
				res, err := ops.MatvecF16(weightTensor, tokIn)
				if err != nil {
					t.Fatalf("ops.MatvecF16 token %d failed: %v", tok, err)
				}
				if len(res) != n {
					t.Fatalf("ops.MatvecF16 token %d returned %d rows, want %d", tok, len(res), n)
				}
				copy(wantOut[tok*n:(tok+1)*n], res)
			}

			// 2. Run batched matmul for all T tokens
			gotOut, err := mm.Mul(weightTensor, activations, T)
			if err != nil {
				t.Fatalf("mm.Mul(T=%d) failed: %v", T, err)
			}
			if len(gotOut) != n*T {
				t.Fatalf("mm.Mul returned %d elements, want %d", len(gotOut), n*T)
			}

			// 3. Bit-exact float32 comparison across all elements using math.Float32bits
			for tok := 0; tok < T; tok++ {
				for row := 0; row < n; row++ {
					idx := tok*n + row
					gotBits := math.Float32bits(gotOut[idx])
					wantBits := math.Float32bits(wantOut[idx])
					if gotBits != wantBits {
						t.Fatalf("bit mismatch at token %d row %d: got 0x%08x (%g), want 0x%08x (%g)",
							tok, row, gotBits, gotOut[idx], wantBits, wantOut[idx])
					}
				}
			}
		})
	}
}

func BenchmarkMetalMatmulPreservedF16(b *testing.B) {
	dev, err := libmetal.NewDevice()
	if err != nil {
		b.Fatal(err)
	}
	defer dev.Close()

	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		b.Fatal(err)
	}
	defer ops.Close()

	mm, err := newMetalMatmulPreservedF16(ops)
	if err != nil {
		b.Fatal(err)
	}
	defer mm.Close()

	arena := metalmem.NewArena(dev)
	defer arena.Free()

	const k = 16384
	const n = 4

	weightBytes := metalmem.AllocIn[byte](arena, "head-weight", 2*k*n)
	weightTensor := boundMetalTensor{
		name:       "output.weight",
		typeID:     1,
		dimensions: []uint64{k, n},
		data:       weightBytes,
	}

	for _, T := range []int{1, 8, 64, 256, 512} {
		activations := make([]float32, k*T)
		b.Run(fmt.Sprintf("T=%d", T), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := mm.Mul(weightTensor, activations, T); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(b.N*T)/b.Elapsed().Seconds(), "tokens/s")
			b.ReportMetric(1, "cbufs/batch")
		})
	}
}

// requireOrderSensitive fails unless summing this test's own products in
// reverse changes the float32 result for at least one output element.
//
// Without it the bit-identity assertion is fail-open: inputs whose partial
// sums are all exact make every association agree, so a kernel that
// reassociates passes. The guard runs on the host, over the same F16 weights
// the GPU reads, so it measures the conditioning of the actual test data
// rather than a claim about it.
func requireOrderSensitive(t *testing.T, weight []byte, in []float32, k, n, tokens int) {
	t.Helper()
	for tok := 0; tok < tokens; tok++ {
		for row := 0; row < n; row++ {
			terms := make([]float32, k)
			for i := 0; i < k; i++ {
				w := f16BitsToFloat(binary.LittleEndian.Uint16(weight[2*(row*k+i):]))
				terms[i] = w * in[tok*k+i]
			}
			var fwd, rev float32
			for i := 0; i < k; i++ {
				fwd += terms[i]
			}
			for i := k - 1; i >= 0; i-- {
				rev += terms[i]
			}
			if math.Float32bits(fwd) != math.Float32bits(rev) {
				return
			}
		}
	}
	t.Fatalf("test inputs are order-insensitive: forward and reverse summation agree bit-for-bit on every one of %d outputs, so the bit-identity assertion cannot detect reassociation", n*tokens)
}

// f16BitsToFloat decodes IEEE 754 binary16. Normals only, which is all this
// test generates.
func f16BitsToFloat(bits uint16) float32 {
	sign := uint32(bits&0x8000) << 16
	exp := uint32(bits>>10) & 0x1f
	mant := uint32(bits & 0x03ff)
	if exp == 0 {
		return math.Float32frombits(sign)
	}
	return math.Float32frombits(sign | (exp+112)<<23 | mant<<13)
}
