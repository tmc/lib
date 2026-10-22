// SPDX-License-Identifier: Apache-2.0

package mxfp4

import (
	"math"
	"testing"
)

func TestUE8M0Conversion(t *testing.T) {
	if !UE8M0(0xFF).IsNaN() {
		t.Fatal("UE8M0(0xFF).IsNaN() = false, want true")
	}
	if !math.IsNaN(float64(UE8M0(0xFF).Float32())) {
		t.Fatal("UE8M0(0xFF).Float32() did not return NaN")
	}
	if !Float32ToUE8M0(float32(math.NaN())).IsNaN() {
		t.Fatal("Float32ToUE8M0(NaN) did not return 0xFF")
	}
	if !Float32ToUE8M0(float32(math.Inf(1))).IsNaN() {
		t.Fatal("Float32ToUE8M0(+Inf) did not return 0xFF")
	}
	if got := Float32ToUE8M0(0); got != 0 {
		t.Fatalf("Float32ToUE8M0(0) = %d, want 0", got)
	}
	if got := Float32ToUE8M0(-1.0); got != 0 {
		t.Fatalf("Float32ToUE8M0(-1.0) = %d, want 0", got)
	}

	cases := []struct {
		in      float32
		wantExp UE8M0
		wantVal float32
	}{
		{0.5, 126, 0.5},
		{1.0, 127, 1.0},
		{1.25, 128, 2.0},
		{2.0, 128, 2.0},
		{4.0, 129, 4.0},
	}
	for _, tc := range cases {
		got := Float32ToUE8M0(tc.in)
		if got != tc.wantExp {
			t.Fatalf("Float32ToUE8M0(%v) = %d, want %d", tc.in, got, tc.wantExp)
		}
		if got.IsNaN() {
			t.Fatalf("UE8M0(%d).IsNaN() = true", got)
		}
		if got.Float32() != tc.wantVal {
			t.Fatalf("UE8M0(%d).Float32() = %v, want %v", got, got.Float32(), tc.wantVal)
		}
	}
}

func TestPackUnpackOCPBlocks(t *testing.T) {
	// Grid values scaled by 2.0 -> maxAbs = 12.0, amax/6.0 = 2.0 -> UE8M0(128).
	grid := []float32{0, 1, 2, 3, 4, 6, 8, 12, -1, -2, -3, -4, -6, -8, -12, 0}
	src := make([]float32, 64)
	copy(src[0:16], grid)
	copy(src[16:32], grid)
	for i := 0; i < 32; i++ {
		src[32+i] = src[i] * 0.25 // maxAbs = 3.0 -> amax/6.0 = 0.5 -> UE8M0(126)
	}

	packed, err := PackOCPBlocks(src)
	if err != nil {
		t.Fatalf("PackOCPBlocks failed: %v", err)
	}
	if len(packed) != 2*OCPBlockBytes {
		t.Fatalf("len(packed) = %d, want %d", len(packed), 2*OCPBlockBytes)
	}
	if packed[16] != 128 {
		t.Fatalf("block 0 scale byte = %d, want 128", packed[16])
	}
	if packed[OCPBlockBytes+16] != 126 {
		t.Fatalf("block 1 scale byte = %d, want 126", packed[OCPBlockBytes+16])
	}

	unpacked, err := UnpackOCPBlocks(packed, 64)
	if err != nil {
		t.Fatalf("UnpackOCPBlocks failed: %v", err)
	}
	for i, want := range src {
		if unpacked[i] != want {
			t.Fatalf("unpacked[%d] = %v, want exact %v", i, unpacked[i], want)
		}
	}

	// All-zero block test.
	zeros := make([]float32, 32)
	zPacked, err := PackOCPBlocks(zeros)
	if err != nil {
		t.Fatalf("PackOCPBlocks(zeros) failed: %v", err)
	}
	zUnpacked, err := UnpackOCPBlocks(zPacked, 32)
	if err != nil {
		t.Fatalf("UnpackOCPBlocks(zeros) failed: %v", err)
	}
	for i, v := range zUnpacked {
		if v != 0 {
			t.Fatalf("zUnpacked[%d] = %v, want 0", i, v)
		}
	}

	// Malformed input tests.
	if _, err := PackOCPBlocks(nil); err == nil {
		t.Fatal("PackOCPBlocks(nil) succeeded")
	}
	if _, err := PackOCPBlocks(make([]float32, 31)); err == nil {
		t.Fatal("PackOCPBlocks(len=31) succeeded")
	}
	badSrc := make([]float32, 32)
	badSrc[7] = float32(math.NaN())
	if _, err := PackOCPBlocks(badSrc); err == nil {
		t.Fatal("PackOCPBlocks(NaN) succeeded")
	}
	if _, err := UnpackOCPBlocks(packed, 31); err == nil {
		t.Fatal("UnpackOCPBlocks(count=31) succeeded")
	}
	if _, err := UnpackOCPBlocks(packed[:16], 32); err == nil {
		t.Fatal("UnpackOCPBlocks(short data) succeeded")
	}
	nanBlock := make([]byte, OCPBlockBytes)
	nanBlock[16] = 0xFF
	if _, err := UnpackOCPBlocks(nanBlock, 32); err == nil {
		t.Fatal("UnpackOCPBlocks(NaN scale 0xff) succeeded")
	}
}
