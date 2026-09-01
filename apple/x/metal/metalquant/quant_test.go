// SPDX-License-Identifier: Apache-2.0

package metalquant

import (
	"encoding/binary"
	"testing"
)

func TestDequantizeQ4_0(t *testing.T) {
	// 1 block of 32 elements (18 bytes): scale = 1.0 (0x3C00 in FP16)
	data := make([]byte, 18)
	binary.LittleEndian.PutUint16(data[0:2], 0x3C00) // 1.0

	// Set first byte nibbles to 8 (so q - 8 = 0) and 9 (q - 8 = 1)
	data[2] = 0x98 // low: 8 -> 0, high: 9 -> 1

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
	// 1 block of 32 elements (34 bytes): scale = 1.0 (0x3C00 in FP16)
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
