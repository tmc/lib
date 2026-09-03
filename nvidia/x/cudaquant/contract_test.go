// SPDX-License-Identifier: Apache-2.0

package cudaquant

import (
	"encoding/binary"
	"testing"
)

func TestDequantizeRejectsMalformedInput(t *testing.T) {
	tests := []struct {
		name string
		data []byte
		n    int
	}{
		{"negative count", nil, -32},
		{"partial block", nil, 31},
		{"short buffer", make([]byte, 17), 32},
		{"trailing buffer", make([]byte, 19), 32},
	}
	for _, tt := range tests {
		if _, err := DequantizeQ4_0(tt.data, tt.n); err == nil {
			t.Errorf("%s: DequantizeQ4_0 succeeded", tt.name)
		}
	}

	tests = []struct {
		name string
		data []byte
		n    int
	}{
		{"negative count", nil, -32},
		{"partial block", nil, 31},
		{"short buffer", make([]byte, 33), 32},
		{"trailing buffer", make([]byte, 35), 32},
	}
	for _, tt := range tests {
		if _, err := DequantizeQ8_0(tt.data, tt.n); err == nil {
			t.Errorf("%s: DequantizeQ8_0 succeeded", tt.name)
		}
	}
}

func TestDequantizeScaleBoundaries(t *testing.T) {
	for _, bits := range []uint16{0x7c00, 0xfc00, 0x7e00} {
		q4 := make([]byte, 18)
		binary.LittleEndian.PutUint16(q4, bits)
		if _, err := DequantizeQ4_0(q4, 32); err == nil {
			t.Errorf("DequantizeQ4_0(scale %#x) succeeded", bits)
		}
		q8 := make([]byte, 34)
		binary.LittleEndian.PutUint16(q8, bits)
		if _, err := DequantizeQ8_0(q8, 32); err == nil {
			t.Errorf("DequantizeQ8_0(scale %#x) succeeded", bits)
		}
	}

	q8 := make([]byte, 34)
	binary.LittleEndian.PutUint16(q8, 0x7bff)
	q8[2] = 1
	got, err := DequantizeQ8_0(q8, 32)
	if err != nil {
		t.Fatal(err)
	}
	if got[0] != 65504 {
		t.Errorf("maximum finite scale decoded as %v", got[0])
	}
}

func FuzzDequantizeNeverPanics(f *testing.F) {
	f.Add([]byte{}, uint8(0))
	f.Add(make([]byte, 34), uint8(1))
	f.Fuzz(func(t *testing.T, data []byte, blocks uint8) {
		n := int(blocks%5) * 32
		_, _ = DequantizeQ4_0(data, n)
		_, _ = DequantizeQ8_0(data, n)
	})
}

func BenchmarkDequantize(b *testing.B) {
	for _, bb := range []struct {
		name string
		data []byte
		fn   func([]byte, int) ([]float32, error)
	}{
		{"Q4_0", make([]byte, 18*128), DequantizeQ4_0},
		{"Q8_0", make([]byte, 34*128), DequantizeQ8_0},
	} {
		b.Run(bb.name, func(b *testing.B) {
			b.ReportAllocs()
			for range b.N {
				if _, err := bb.fn(bb.data, 4096); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
