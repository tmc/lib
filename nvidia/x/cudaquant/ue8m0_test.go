// SPDX-License-Identifier: Apache-2.0

package cudaquant

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestScaleRecipeString(t *testing.T) {
	if got := ScaleRecipeFloat32.String(); got != "float32" {
		t.Fatalf("ScaleRecipeFloat32.String() = %q, want %q", got, "float32")
	}
	if got := ScaleRecipeUE8M0.String(); got != "ue8m0" {
		t.Fatalf("ScaleRecipeUE8M0.String() = %q, want %q", got, "ue8m0")
	}
	if got := ScaleRecipe(99).String(); got != "ScaleRecipe(99)" {
		t.Fatalf("ScaleRecipe(99).String() = %q, want %q", got, "ScaleRecipe(99)")
	}
}

func TestUE8M0ScaleQuantDequant(t *testing.T) {
	tests := []struct {
		name    string
		amax    float32
		fp8Max  float32
		wantExp uint8
		wantScl float32
	}{
		{"exact_448_gives_exp_127", 448.0, 448.0, 127, 1.0},
		{"exact_896_gives_exp_128", 896.0, 448.0, 128, 2.0},
		{"exact_224_gives_exp_126", 224.0, 448.0, 126, 0.5},
		{"just_above_448_rounds_up_to_128", 449.0, 448.0, 128, 2.0},
		{"zero_amax_clamps_to_1e_12", 0.0, 448.0, 79, float32(math.Ldexp(1.0, 79-127))},
		{"default_fp8max_when_zero", 448.0, 0.0, 127, 1.0},
		{"inf_amax_clamps_to_254", float32(math.Inf(1)), 448.0, 254, float32(math.Ldexp(1.0, 254-127))},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			exp := QuantizeScaleUE8M0(tc.amax, tc.fp8Max)
			if exp != tc.wantExp {
				t.Fatalf("QuantizeScaleUE8M0(%v, %v) = %d, want %d", tc.amax, tc.fp8Max, exp, tc.wantExp)
			}
			gotScl := DequantizeScaleUE8M0(exp)
			if gotScl != tc.wantScl {
				t.Fatalf("DequantizeScaleUE8M0(%d) = %v, want %v", exp, gotScl, tc.wantScl)
			}
		})
	}

	// Independent oracle across >1000 synthetic amax values.
	compared := 0
	for i := 1; i <= 1200; i++ {
		amax := float32(i) * 0.375
		gotExp := QuantizeScaleUE8M0(amax, 448.0)
		gotScale := DequantizeScaleUE8M0(gotExp)
		wantExp := int(math.Ceil(math.Log2(math.Max(float64(amax), 1e-12)/448.0))) + 127
		if wantExp < 0 {
			wantExp = 0
		}
		if wantExp > 254 {
			wantExp = 254
		}
		if int(gotExp) != wantExp {
			t.Fatalf("i=%d amax=%v: gotExp=%d, wantExp=%d", i, amax, gotExp, wantExp)
		}
		wantScale := float32(math.Ldexp(1.0, wantExp-127))
		if gotScale != wantScale {
			t.Fatalf("i=%d exp=%d: gotScale=%v, wantScale=%v", i, gotExp, gotScale, wantScale)
		}
		compared++
	}
	if compared < 1000 {
		t.Fatalf("oracle compared %d cases, want >= 1000", compared)
	}
}

func TestPackUnpackUE8M0(t *testing.T) {
	if _, err := PackUE8M0(nil); err == nil {
		t.Fatal("PackUE8M0(nil) succeeded, want error")
	}
	if _, err := UnpackUE8M0([]uint32{1}, 0); err == nil {
		t.Fatal("UnpackUE8M0(count=0) succeeded, want error")
	}
	if _, err := UnpackUE8M0([]uint32{1}, 5); err == nil {
		t.Fatal("UnpackUE8M0(mismatched word count) succeeded, want error")
	}

	for count := 1; count <= 17; count++ {
		exps := make([]uint8, count)
		for i := range exps {
			exps[i] = uint8((i*17 + 120) % 255)
		}
		packed, err := PackUE8M0(exps)
		if err != nil {
			t.Fatalf("count=%d: PackUE8M0 failed: %v", count, err)
		}
		wantWords := (count + 3) / 4
		if len(packed) != wantWords {
			t.Fatalf("count=%d: len(packed)=%d, want %d", count, len(packed), wantWords)
		}

		// Independent byte-level oracle via encoding/binary.LittleEndian.
		rawBytes := make([]byte, wantWords*4)
		copy(rawBytes, exps)
		for w := 0; w < wantWords; w++ {
			wantWord := binary.LittleEndian.Uint32(rawBytes[w*4 : (w+1)*4])
			if packed[w] != wantWord {
				t.Fatalf("count=%d word=%d: got %#08x, want %#08x", count, w, packed[w], wantWord)
			}
		}

		unpacked, err := UnpackUE8M0(packed, count)
		if err != nil {
			t.Fatalf("count=%d: UnpackUE8M0 failed: %v", count, err)
		}
		for i := range exps {
			if unpacked[i] != exps[i] {
				t.Fatalf("count=%d idx=%d: unpacked=%d, want %d", count, i, unpacked[i], exps[i])
			}
		}
	}
}

func TestUE8M0BlockFP8RoundTrip(t *testing.T) {
	rows, cols, blockSize := 4, 256, 128
	input := make([]float32, rows*cols)
	for i := range input {
		// Values chosen as exact multiples of power-of-two * E4M3 grid points.
		sign := float32(1.0)
		if i%3 == 1 {
			sign = -1.0
		}
		input[i] = sign * float32((i%7)+1) * 16.0
	}

	q, packedScales, err := QuantizeBlockFP8UE8M0CPU(input, rows, cols, blockSize)
	if err != nil {
		t.Fatalf("QuantizeBlockFP8UE8M0CPU failed: %v", err)
	}
	if len(q) != rows*cols {
		t.Fatalf("len(q) = %d, want %d", len(q), rows*cols)
	}
	if len(packedScales) != (rows*(cols/blockSize)+3)/4 {
		t.Fatalf("len(packedScales) = %d, want %d", len(packedScales), (rows*(cols/blockSize)+3)/4)
	}

	deq, err := DequantizeBlockFP8UE8M0CPU(q, packedScales, rows, cols, blockSize)
	if err != nil {
		t.Fatalf("DequantizeBlockFP8UE8M0CPU failed: %v", err)
	}
	for i, want := range input {
		if deq[i] != want {
			t.Fatalf("index %d: deq=%v, want exact %v", i, deq[i], want)
		}
	}

	// Reject malformed inputs.
	if _, _, err := QuantizeBlockFP8UE8M0CPU(input, 0, cols, blockSize); err == nil {
		t.Fatal("expected error for rows=0")
	}
	if _, _, err := QuantizeBlockFP8UE8M0CPU(input, rows, 130, 128); err == nil {
		t.Fatal("expected error for cols not divisible by blockSize")
	}
	badInput := append([]float32(nil), input...)
	badInput[10] = float32(math.NaN())
	if _, _, err := QuantizeBlockFP8UE8M0CPU(badInput, rows, cols, blockSize); err == nil {
		t.Fatal("expected error for NaN input")
	}
	badQ := append([]uint8(nil), q...)
	badQ[0] = 0x7F // E4M3 NaN
	if _, err := DequantizeBlockFP8UE8M0CPU(badQ, packedScales, rows, cols, blockSize); err == nil {
		t.Fatal("expected error for E4M3 NaN code 0x7f")
	}
	badScales := []uint32{0x000000FF}
	if _, err := DequantizeBlockFP8UE8M0CPU(make([]uint8, 128), badScales, 1, 128, 128); err == nil {
		t.Fatal("expected error for UE8M0 NaN scale 0xff")
	}
}
