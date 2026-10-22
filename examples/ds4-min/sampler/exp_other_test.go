//go:build !linux || !arm64

package sampler

import (
	"math"
	"testing"
)

func TestExpUnchangedPlatform(t *testing.T) {
	for _, bits := range []uint32{0, 0x80000000, 0xb8ca127e, 0xc2d00000, 0xff800000, 0x7f800000} {
		x := math.Float32frombits(bits)
		got := math.Float32bits(Exp(x))
		want := math.Float32bits(float32(math.Exp(float64(x))))
		if got != want {
			t.Fatalf("input=%08x got=%08x original=%08x", bits, got, want)
		}
	}
}
