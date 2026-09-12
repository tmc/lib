package nccltest

import (
	"math"
	"testing"
)

func TestFloatFormatVectors(t *testing.T) {
	for _, tt := range []struct {
		name  string
		f     floatFormat
		value float64
		bits  uint16
	}{
		{"half one", float16Format, 1, 0x3c00},
		{"half subnormal", float16Format, 0x1p-24, 1},
		{"half underflow tie", float16Format, 0x1p-25, 0},
		{"half even lower", float16Format, 1 + 0x1p-11, 0x3c00},
		{"half even upper", float16Format, 1 + 3*0x1p-11, 0x3c02},
		{"half max", float16Format, 65504, 0x7bff},
		{"half overflow", float16Format, 65520, 0x7c00},
		{"bfloat one", bfloat16Format, 1, 0x3f80},
		{"bfloat subnormal", bfloat16Format, 0x1p-133, 1},
		{"bfloat max", bfloat16Format, float64(math.Float32frombits(0x7f7f0000)), 0x7f7f},
		{"bfloat overflow", bfloat16Format, float64(math.Float32frombits(0x7f7f8000)), 0x7f80},
		{"e4m3 one", float8e4m3Format, 1, 0x38},
		{"e4m3 subnormal", float8e4m3Format, 0x1p-9, 1},
		{"e4m3 underflow tie", float8e4m3Format, 0x1p-10, 0},
		{"e4m3 even lower", float8e4m3Format, 1.0625, 0x38},
		{"e4m3 even upper", float8e4m3Format, 1.1875, 0x3a},
		{"e4m3 max", float8e4m3Format, 448, 0x7e},
		{"e4m3 saturation", float8e4m3Format, math.Inf(1), 0x7e},
		{"e5m2 one", float8e5m2Format, 1, 0x3c},
		{"e5m2 subnormal", float8e5m2Format, 0x1p-16, 1},
		{"e5m2 underflow tie", float8e5m2Format, 0x1p-17, 0},
		{"e5m2 even lower", float8e5m2Format, 1.125, 0x3c},
		{"e5m2 even upper", float8e5m2Format, 1.375, 0x3e},
		{"e5m2 max", float8e5m2Format, 57344, 0x7b},
		{"e5m2 saturation", float8e5m2Format, math.Inf(1), 0x7b},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.f.encode(tt.value); got != tt.bits {
				t.Fatalf("encode(%g)=%x, want %x", tt.value, got, tt.bits)
			}
			sign := uint16(1 << (tt.f.exponentBits + tt.f.fractionBits))
			if got := tt.f.encode(-tt.value); got != tt.bits|sign {
				t.Fatalf("encode(%g)=%x, want %x", -tt.value, got, tt.bits|sign)
			}
		})
	}
}

func TestFloatFormatRoundTrip(t *testing.T) {
	for name, f := range map[string]floatFormat{"half": float16Format, "bfloat16": bfloat16Format, "e4m3": float8e4m3Format, "e5m2": float8e5m2Format} {
		t.Run(name, func(t *testing.T) {
			for bits := 0; bits < 1<<(1+f.exponentBits+f.fractionBits); bits++ {
				value := f.decode(uint16(bits))
				if math.IsNaN(value) || math.IsInf(value, 0) {
					continue
				}
				if got := f.encode(value); got != uint16(bits) {
					t.Fatalf("bits=%x value=%g encoded=%x", bits, value, got)
				}
			}
			if !math.IsNaN(f.decode(f.encode(math.NaN()))) {
				t.Fatal("NaN was lost")
			}
			if !math.Signbit(f.decode(f.encode(math.Copysign(0, -1)))) {
				t.Fatal("negative zero was lost")
			}
		})
	}
}
