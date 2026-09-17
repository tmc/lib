// SPDX-License-Identifier: Apache-2.0

package cudaquant

import (
	"encoding/binary"
	"testing"
)

func TestNormQuantReference(t *testing.T) {
	in := make([]float32, 32)
	for i := range in {
		in[i] = float32(i)
	}
	got, e := RMSNormQ8_1(in, nil, 1, 32, 1e-5, false)
	if e != nil || len(got) != 1 {
		t.Fatalf("quant=%v err=%v", got, e)
	}
}
func TestWeightedNormSemantics(t *testing.T) {
	_, e := RMSNormQ8_1(make([]float32, 32), make([]float32, 31), 1, 32, 1e-5, true)
	if e == nil {
		t.Fatal("accepted short weight")
	}
}
func TestQuantRoundingAndSaturation(t *testing.T) {
	in := make([]float32, 32)
	in[0] = 100
	got, e := RMSNormQ8_1(in, nil, 1, 32, 0, false)
	if e != nil || got[0].Quants[0] != 127 {
		t.Fatalf("quant=%v err=%v", got, e)
	}
}

func TestDequantizeQ4_0(t *testing.T) {
	data := make([]byte, 18)
	binary.LittleEndian.PutUint16(data[0:2], 0x3C00) // 1.0
	data[2] = 0x98                                   // low: 8 -> 0, high: 9 -> 1

	out, err := DequantizeQ4_0(data, 32)
	if err != nil {
		t.Fatalf("DequantizeQ4_0 failed: %v", err)
	}

	if out[0] != 0.0 {
		t.Errorf("out[0] = %f, want 0.0", out[0])
	}
	if out[16] != 1.0 {
		t.Errorf("out[16] = %f, want 1.0", out[16])
	}
}

func TestDequantizeQ8_0(t *testing.T) {
	data := make([]byte, 34)
	binary.LittleEndian.PutUint16(data[0:2], 0x3C00) // 1.0
	data[2] = 42                                     // int8(42)

	out, err := DequantizeQ8_0(data, 32)
	if err != nil {
		t.Fatalf("DequantizeQ8_0 failed: %v", err)
	}

	if out[0] != 42.0 {
		t.Errorf("out[0] = %f, want 42.0", out[0])
	}
}
