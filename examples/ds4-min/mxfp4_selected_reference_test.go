package main

import (
	"encoding/binary"
	"math"
	"testing"
)

// Independent float64 arithmetic over serialized operands; the production
// kernel's lookup table and half conversion are deliberately not reused.
func mxfp4Q81DotReference(weights, input []byte) float64 {
	var result float64
	for b := 0; b < len(weights)/17; b++ {
		p := weights[b*17:]
		if p[0] == 255 {
			return math.NaN()
		}
		e := int(p[0]) - 127
		if p[0] == 0 {
			e = -127
		}
		scale := math.Ldexp(1, e)
		h := binary.LittleEndian.Uint16(input[b*36:])
		exponent := int((h >> 10) & 31)
		fraction := int(h & 1023)
		d := math.Ldexp(float64(1024+fraction), exponent-25)
		if exponent == 0 {
			d = math.Ldexp(float64(fraction), -24)
		}
		if h&0x8000 != 0 {
			d = -d
		}
		for i := 0; i < 32; i++ {
			q := int(p[1+i%16])
			if i >= 16 {
				q >>= 4
			}
			q &= 15
			magnitude := q & 7
			var value float64
			if magnitude < 4 {
				value = float64(magnitude) / 2
			} else {
				value = math.Ldexp(float64(2+(magnitude&1)), (magnitude>>1)-2)
			}
			if q&8 != 0 {
				value = -value
			}
			result += scale * value * d * float64(int8(input[b*36+4+i]))
		}
	}
	return result
}

func TestMXFP4Q81Reference(t *testing.T) {
	weights := make([]byte, 17)
	weights[0] = 127
	input := make([]byte, 36)
	binary.LittleEndian.PutUint16(input, 0x3800)
	for i := 0; i < 16; i++ {
		weights[i+1] = 0xf1
		input[4+i] = 2
		input[20+i] = 3
	}
	if got := mxfp4Q81DotReference(weights, input); got != -136 {
		t.Fatalf("signed halves dot=%g want -136", got)
	}
	weights[0] = 0
	if got, want := mxfp4Q81DotReference(weights, input), math.Ldexp(-136, -127); got != want {
		t.Fatalf("E8M0 zero dot=%g want %g", got, want)
	}
	weights[0] = 127
	mutated := append([]byte(nil), weights...)
	mutated[0]++
	if mxfp4Q81DotReference(weights, input) == mxfp4Q81DotReference(mutated, input) {
		t.Fatal("scale mutation survived")
	}
	for i := 1; i < 17; i++ {
		mutated[i] = (weights[i] << 4) | (weights[i] >> 4)
	}
	mutated[0] = weights[0]
	if mxfp4Q81DotReference(weights, input) == mxfp4Q81DotReference(mutated, input) {
		t.Fatal("nibble mutation survived")
	}
	weights[0] = 255
	if !math.IsNaN(mxfp4Q81DotReference(weights, input)) {
		t.Fatal("E8M0 reserved exponent was not rejected as NaN")
	}
}
