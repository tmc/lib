//go:build linux && arm64

package sampler

import "math"

// Range-reduction data and polynomial from Arm optimized-routines expf and
// exp2f_data, commit 59afb6fed4275e2fc249df49fce43ee5e75af178 (MIT).
// See LICENSE.arm-expf. This is the 32-interval approximation, not a table of
// observed libm disagreements. Arithmetic follows the actual pinned Linux
// ARM64 glibc 2.39 scalar binary: round-away reduction and three double FMAs.
var expTable = [...]uint64{
	0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51,
	0x3fef72b83c7d517b, 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1,
	0x3fef06fe0a31b715, 0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d,
	0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429, 0x3feea47eb03a5585,
	0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
	0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
	0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069,
	0x3fef5818dcfba487, 0x3fef7c97337b9b5f, 0x3fefa4afa2a490da, 0x3fefd0765b6e4540,
}

func exp(x float32) float32 {
	bits := math.Float32bits(x)
	if bits == 0xff800000 {
		return 0
	}
	if bits&0x7fffffff >= 0x7f800000 {
		return x + x
	}
	if x > 0x1.62e42ep6 {
		return float32(math.Inf(1))
	}
	if x < -0x1.9fe368p6 {
		return 0
	}
	if x < -0x1.9d1d9ep6 {
		return math.SmallestNonzeroFloat32
	}
	const (
		inverse = 0x1.71547652b82fep0 * 32
		c0      = 0x1.c6af84b912394p-5 / (32 * 32 * 32)
		c1      = 0x1.ebfce50fac4f3p-3 / (32 * 32)
		c2      = 0x1.62e42ff0c52d6p-1 / 32
	)
	z := float64(inverse * float64(x))
	kd := math.Round(z)
	k := uint64(int64(kd))
	r := float64(z - kd)
	s := math.Float64frombits(expTable[k&31] + (k << 47))
	z = math.FMA(c0, r, c1)
	r2 := float64(r * r)
	y := math.FMA(c2, r, 1)
	y = math.FMA(z, r2, y)
	return float32(float64(y * s))
}
