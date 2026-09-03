// SPDX-License-Identifier: Apache-2.0

package metalquant

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestEmpty(t *testing.T) {
	for name, fn := range map[string]func() ([]byte, error){
		"Q4_0": func() ([]byte, error) { return QuantizeQ4_0(nil) },
		"Q8_0": func() ([]byte, error) { return QuantizeQ8_0(nil) },
	} {
		got, err := fn()
		if err != nil || len(got) != 0 {
			t.Errorf("Quantize%s(nil) = %v, %v, want empty, nil", name, got, err)
		}
	}
	if got, err := DequantizeQ4_0(nil, 0); err != nil || len(got) != 0 {
		t.Errorf("DequantizeQ4_0(nil, 0) = %v, %v, want empty, nil", got, err)
	}
	if got, err := DequantizeQ8_0(nil, 0); err != nil || len(got) != 0 {
		t.Errorf("DequantizeQ8_0(nil, 0) = %v, %v, want empty, nil", got, err)
	}
}

func TestQuantizeRejectsInvalidInput(t *testing.T) {
	tests := []struct {
		name string
		v    float32
	}{
		{"NaN", float32(math.NaN())},
		{"positive infinity", float32(math.Inf(1))},
		{"negative infinity", float32(math.Inf(-1))},
		{"scale overflow", math.MaxFloat32},
	}
	for _, tt := range tests {
		values := make([]float32, BlockSize)
		values[7] = tt.v
		if _, err := QuantizeQ4_0(values); err == nil {
			t.Errorf("QuantizeQ4_0(%s) succeeded", tt.name)
		}
		if _, err := QuantizeQ8_0(values); err == nil {
			t.Errorf("QuantizeQ8_0(%s) succeeded", tt.name)
		}
	}
}

func TestDequantizeRejectsMalformedInput(t *testing.T) {
	tests := []struct {
		name string
		data []byte
		n    int
	}{
		{"negative count", nil, -BlockSize},
		{"partial block", nil, BlockSize - 1},
		{"short buffer", make([]byte, blockBytesQ4_0-1), BlockSize},
		{"trailing buffer", make([]byte, blockBytesQ4_0+1), BlockSize},
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
		{"negative count", nil, -BlockSize},
		{"partial block", nil, BlockSize - 1},
		{"short buffer", make([]byte, blockBytesQ8_0-1), BlockSize},
		{"trailing buffer", make([]byte, blockBytesQ8_0+1), BlockSize},
	}
	for _, tt := range tests {
		if _, err := DequantizeQ8_0(tt.data, tt.n); err == nil {
			t.Errorf("%s: DequantizeQ8_0 succeeded", tt.name)
		}
	}
}

func TestDequantizeRejectsNonFiniteScale(t *testing.T) {
	for _, bits := range []uint16{0x7c00, 0xfc00, 0x7e00} {
		q4 := make([]byte, blockBytesQ4_0)
		binary.LittleEndian.PutUint16(q4, bits)
		if _, err := DequantizeQ4_0(q4, BlockSize); err == nil {
			t.Errorf("DequantizeQ4_0(scale %#x) succeeded", bits)
		}
		q8 := make([]byte, blockBytesQ8_0)
		binary.LittleEndian.PutUint16(q8, bits)
		if _, err := DequantizeQ8_0(q8, BlockSize); err == nil {
			t.Errorf("DequantizeQ8_0(scale %#x) succeeded", bits)
		}
	}
}

func TestMaximumFiniteHalfScale(t *testing.T) {
	data := make([]byte, blockBytesQ8_0)
	binary.LittleEndian.PutUint16(data, 0x7bff)
	data[2] = 1
	got, err := DequantizeQ8_0(data, BlockSize)
	if err != nil {
		t.Fatal(err)
	}
	if got[0] != 65504 {
		t.Errorf("got %v, want 65504", got[0])
	}
}

func TestZeroScale(t *testing.T) {
	for name, data := range map[string][]byte{
		"Q4_0": make([]byte, blockBytesQ4_0),
		"Q8_0": make([]byte, blockBytesQ8_0),
	} {
		var got []float32
		var err error
		if name == "Q4_0" {
			got, err = DequantizeQ4_0(data, BlockSize)
		} else {
			got, err = DequantizeQ8_0(data, BlockSize)
		}
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		for i, v := range got {
			if v != 0 {
				t.Errorf("%s: element %d = %v, want 0", name, i, v)
			}
		}
	}
}

func FuzzDequantizeNeverPanics(f *testing.F) {
	f.Add([]byte{}, uint8(0))
	f.Add(make([]byte, blockBytesQ8_0), uint8(1))
	f.Fuzz(func(t *testing.T, data []byte, blocks uint8) {
		n := int(blocks%5) * BlockSize
		_, _ = DequantizeQ4_0(data, n)
		_, _ = DequantizeQ8_0(data, n)
	})
}

func BenchmarkConversion(b *testing.B) {
	values := testValues(4096)
	q4, _ := QuantizeQ4_0(values)
	q8, _ := QuantizeQ8_0(values)
	benchmarks := []struct {
		name string
		fn   func()
	}{
		{"QuantizeQ4_0", func() { _, _ = QuantizeQ4_0(values) }},
		{"DequantizeQ4_0", func() { _, _ = DequantizeQ4_0(q4, len(values)) }},
		{"QuantizeQ8_0", func() { _, _ = QuantizeQ8_0(values) }},
		{"DequantizeQ8_0", func() { _, _ = DequantizeQ8_0(q8, len(values)) }},
	}
	for _, bb := range benchmarks {
		b.Run(bb.name, func(b *testing.B) {
			b.ReportAllocs()
			for range b.N {
				bb.fn()
			}
		})
	}
}
