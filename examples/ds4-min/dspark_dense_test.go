package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/tmc/lib/gguf"
)

func TestDSparkDenseActualRows(t *testing.T) {
	for _, typ := range []uint32{0, 1, 8} {
		rowBytes := 128
		if typ == 1 {
			rowBytes = 64
		}
		if typ == 8 {
			rowBytes = 34
		}
		data := make([]byte, 8+rowBytes*2)
		for row := 0; row < 2; row++ {
			b := data[8+row*rowBytes:]
			for i := 0; i < 32; i++ {
				switch typ {
				case 0:
					binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(float32(row+1)))
				case 1:
					binary.LittleEndian.PutUint16(b[i*2:], uint16(0x3c00+row*0x400))
				case 8:
					binary.LittleEndian.PutUint16(b, 0x3800)
					b[2+i] = byte(2 * (row + 1))
				}
			}
		}
		m := &dsparkModel{reader: bytes.NewReader(data), dataOffset: 8, tensors: map[string]gguf.Tensor{"weight": {Name: "weight", TypeID: typ, Dimensions: []uint64{32, 2}}}}
		for row := 0; row < 2; row++ {
			got, err := m.denseRow("weight", row)
			if err != nil {
				t.Fatal(err)
			}
			for _, x := range got {
				if x != float32(row+1) {
					t.Fatalf("type%d row%d got%g", typ, row, x)
				}
			}
		}
		if _, err := m.denseRow("weight", 2); err == nil {
			t.Fatal("out-of-range row admitted")
		}
	}
}

func TestDSparkHalfValues(t *testing.T) {
	for _, tc := range []struct {
		bits uint16
		want float32
	}{{0, 0}, {0x8000, float32(math.Copysign(0, -1))}, {1, float32(math.Ldexp(1, -24))}, {0x3c00, 1}, {0xc000, -2}, {0x7bff, 65504}} {
		got := dsparkHalf(tc.bits)
		if math.Float32bits(got) != math.Float32bits(tc.want) {
			t.Fatalf("%04x: %g want%g", tc.bits, got, tc.want)
		}
	}
}

func TestDSparkDenseRejectsNonfiniteAndTruncated(t *testing.T) {
	tensor := gguf.Tensor{Name: "norm", TypeID: 0, Dimensions: []uint64{1}}
	for _, data := range [][]byte{{0, 0, 0}, {0, 0, 128, 127}, {0, 0, 192, 127}} {
		m := &dsparkModel{reader: bytes.NewReader(data), tensors: map[string]gguf.Tensor{"norm": tensor}}
		if _, err := m.denseRow("norm", 0); err == nil {
			t.Fatal("malformed weight admitted")
		}
	}
}
