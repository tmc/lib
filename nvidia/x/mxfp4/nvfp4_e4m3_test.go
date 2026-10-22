// SPDX-License-Identifier: Apache-2.0

package mxfp4

import (
	"math"
	"testing"
)

// decodeE4M3Spec decodes an unsigned FP8 E4M3 code directly from the bit layout
// (4 exponent bits with bias 7, 3 mantissa bits) without using nvfp4E4M3PosTable.
func decodeE4M3Spec(code uint8) float64 {
	exp := int((code >> 3) & 0x0F)
	mant := float64(code & 0x07)
	if exp == 0 {
		return mant * math.Pow(2, -9)
	}
	return (1.0 + mant/8.0) * math.Pow(2, float64(exp-7))
}

func TestPackUnpackNVFP4E4M3(t *testing.T) {
	// Verify every positive E4M3 table entry against the bit-layout specification.
	for code := uint8(0); code < 127; code++ {
		got, err := decodeE4M3Scale(code)
		if err != nil {
			t.Fatalf("decodeE4M3Scale(%#02x) failed: %v", code, err)
		}
		want := float32(decodeE4M3Spec(code))
		if got != want {
			t.Fatalf("code %#02x: got %v, want %v", code, got, want)
		}
	}

	// Choose a non-power-of-two scale (1.5, which is exact in E4M3: exp=7, mant=4 -> code (7<<3)|4 = 0x3C)
	// so every scaled E2M1 grid value (e.g. 6.0 * 1.5 = 9.0) is reconstructed losslessly.
	scale := float32(1.5)
	src := make([]float32, 64) // 4 blocks of 16 elements (matches FP4_INDEX_KV_FORMAT 64 elements = 32B + 4B)
	for b := 0; b < 4; b++ {
		s := scale * float32(b+1)
		// b=0 -> s=1.5, b=1 -> s=3.0, b=2 -> s=4.5, b=3 -> s=6.0 (all exact in E4M3: 1.5, 3.0, 4.5=4*(1+1/8), 6.0=4*1.5)
		for i := 0; i < 16; i++ {
			src[b*16+i] = E2M1Table[i] * s
		}
	}

	packedData, e4m3Scales, err := PackE2M1WithE4M3Scale(src)
	if err != nil {
		t.Fatalf("PackE2M1WithE4M3Scale failed: %v", err)
	}
	if len(packedData) != 4*NVFP4PackedBytes {
		t.Fatalf("len(packedData) = %d, want %d", len(packedData), 4*NVFP4PackedBytes)
	}
	if len(e4m3Scales) != 4*NVFP4ScaleBytes {
		t.Fatalf("len(e4m3Scales) = %d, want %d", len(e4m3Scales), 4*NVFP4ScaleBytes)
	}
	// Block 0 scale should be 0x3C (1.5 in E4M3).
	if e4m3Scales[0] != 0x3C {
		t.Fatalf("e4m3Scales[0] = %#02x, want 0x3c (1.5)", e4m3Scales[0])
	}

	unpacked, err := UnpackE2M1WithE4M3Scale(packedData, e4m3Scales, 64)
	if err != nil {
		t.Fatalf("UnpackE2M1WithE4M3Scale failed: %v", err)
	}
	for i, want := range src {
		if unpacked[i] != want {
			t.Fatalf("unpacked[%d] = %v, want exact %v", i, unpacked[i], want)
		}
	}
}

func TestNVFP4E4M3RejectsMalformed(t *testing.T) {
	if _, _, err := PackE2M1WithE4M3Scale(nil); err == nil {
		t.Fatal("PackE2M1WithE4M3Scale(nil) succeeded")
	}
	if _, _, err := PackE2M1WithE4M3Scale(make([]float32, 15)); err == nil {
		t.Fatal("PackE2M1WithE4M3Scale(len=15) succeeded")
	}
	bad := make([]float32, 16)
	bad[3] = float32(math.Inf(-1))
	if _, _, err := PackE2M1WithE4M3Scale(bad); err == nil {
		t.Fatal("PackE2M1WithE4M3Scale(-Inf) succeeded")
	}

	if _, err := UnpackE2M1WithE4M3Scale(make([]byte, 8), []uint8{0x38}, 15); err == nil {
		t.Fatal("UnpackE2M1WithE4M3Scale(count=15) succeeded")
	}
	if _, err := UnpackE2M1WithE4M3Scale(make([]byte, 7), []uint8{0x38}, 16); err == nil {
		t.Fatal("UnpackE2M1WithE4M3Scale(short packedData) succeeded")
	}
	if _, err := UnpackE2M1WithE4M3Scale(make([]byte, 8), []uint8{0x38, 0x38}, 16); err == nil {
		t.Fatal("UnpackE2M1WithE4M3Scale(mismatched e4m3Scales) succeeded")
	}
	if _, err := UnpackE2M1WithE4M3Scale(make([]byte, 8), []uint8{0x7F}, 16); err == nil {
		t.Fatal("UnpackE2M1WithE4M3Scale(NaN scale 0x7f) succeeded")
	}
	if _, err := UnpackE2M1WithE4M3Scale(make([]byte, 8), []uint8{0xBC}, 16); err == nil {
		t.Fatal("UnpackE2M1WithE4M3Scale(negative scale 0xbc) succeeded")
	}
}
