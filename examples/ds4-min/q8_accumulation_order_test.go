package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"
)

// TestQ8AccumulationOrder documents the bit-exact reduction order used by the
// Q8 kernels. It is a CPU diagnostic, not CUDA qualification.
func TestQ8AccumulationOrder(t *testing.T) {
	for _, cols := range []int{1024, 4096, 8192} {
		for _, q81 := range []bool{false, true} {
			for _, cancellation := range []bool{false, true} {
				t.Run(fmt.Sprintf("cols=%d/q81=%t/cancellation=%t", cols, q81, cancellation), func(t *testing.T) {
					weights, input := q8OrderFixture(cols/32, q81, cancellation)
					terms := make([]float32, cols/32)
					for b := range terms {
						w, x := weights[b*34:], input[b*36:]
						var dot int32
						for i := range 32 {
							dot += int32(int8(w[2+i])) * int32(int8(x[4+i]))
						}
						scale := math.Float32frombits(binary.LittleEndian.Uint32(x))
						if q81 {
							scale = q8OrderHalf(binary.LittleEndian.Uint16(x))
						}
						terms[b] = q8OrderHalf(binary.LittleEndian.Uint16(w)) * scale * float32(dot)
					}
					var sequential float32
					var partial [32]float32
					for b, term := range terms {
						sequential = float32(math.FMA(float64(term), 1, float64(sequential)))
						partial[b%32] = float32(math.FMA(float64(term), 1, float64(partial[b%32])))
					}
					for stride := 16; stride > 0; stride >>= 1 {
						old := partial
						for lane := 0; lane+stride < 32; lane++ {
							partial[lane] = old[lane] + old[lane+stride]
						}
					}
					wantSequential := float32(1)
					if cancellation {
						wantSequential = 0
					}
					if sequential != wantSequential || partial[0] != 1 {
						t.Fatalf("sequential=%g lane-tree=%g, want %g and 1", sequential, partial[0], wantSequential)
					}
				})
			}
		}
	}
}

func q8OrderFixture(blocks int, q81, cancellation bool) ([]byte, []byte) {
	w, x := make([]byte, blocks*34), make([]byte, blocks*36)
	for b := range blocks {
		wb, xb := w[b*34:], x[b*36:]
		binary.LittleEndian.PutUint16(wb, 0x3c00)
		if q81 {
			binary.LittleEndian.PutUint16(xb, 0x3c00)
		} else {
			binary.LittleEndian.PutUint32(xb, math.Float32bits(1))
		}
		wb[4], xb[5] = 127, 127
		if b < 3 {
			code := int8(64)
			if b == 1 {
				code = 1
			} else if b == 2 {
				code = -64
			}
			wb[2], xb[4] = 64, byte(code)
			if b == 1 {
				wb[2] = 1
			}
			if cancellation && b != 1 {
				binary.LittleEndian.PutUint16(wb, 0x6c00)
			}
		}
		if q81 {
			sum := float32(int8(xb[4])) + 127
			bits := math.Float32bits(sum)
			half := uint16(((bits>>23)-127+15)<<10 | (bits&0x7fffff)>>13)
			binary.LittleEndian.PutUint16(xb[2:], half)
		}
	}
	return w, x
}

func q8OrderHalf(h uint16) float32 {
	return float32(math.Ldexp(float64(1024+int(h&1023)), int(h>>10)-25))
}
