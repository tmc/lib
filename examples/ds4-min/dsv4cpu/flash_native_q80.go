package dsv4cpu

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

func scalarQ80Dot(row []byte, x []Q80Activation) float32 {
	var acc float32
	for index := range x {
		block := row[index*34:]
		activation := &x[index]
		var dot int32
		for i, value := range activation.Values {
			dot += int32(int8(block[2+i])) * int32(value)
		}
		coefficient := float32(half(binary.LittleEndian.Uint16(block)) * activation.Scale)
		acc = fused32(coefficient, float32(dot), acc)
	}
	return acc
}

// dotprodQ80Dot matches the macOS clang ARM DOTPROD block-pair schedule.
func dotprodQ80Dot(row []byte, x []Q80Activation) float32 {
	var a, b [4]float32
	accumulate := func(acc *[4]float32, index int) {
		block := row[index*34:]
		activation := &x[index]
		coefficient := float32(half(binary.LittleEndian.Uint16(block)) * activation.Scale)
		for lane := 0; lane < 4; lane++ {
			var dot int32
			for j := 0; j < 4; j++ {
				i := lane*4 + j
				dot += int32(int8(block[2+i])) * int32(activation.Values[i])
				dot += int32(int8(block[18+i])) * int32(activation.Values[16+i])
			}
			acc[lane] = fused32(float32(dot), coefficient, acc[lane])
		}
	}
	index := 0
	for ; index+1 < len(x); index += 2 {
		accumulate(&a, index)
		accumulate(&b, index+1)
	}
	if index < len(x) {
		accumulate(&a, index)
	}
	for lane := range a {
		a[lane] += b[lane]
	}
	return float32(a[0]+a[1]) + float32(a[2]+a[3])
}

func nativeQ80Mul(dst []float32, x []Q80Activation, m *Matrix) error {
	if m == nil || m.source == nil || m.encoding != Q80 || m.rows <= 0 || m.cols <= 0 || m.cols%32 != 0 || len(dst) != m.rows || len(x) != m.cols/32 {
		return errors.New("invalid native Q8_0 projection")
	}
	row := make([]byte, m.rowBytes)
	for i := range dst {
		n, err := m.source.ReadAt(row, m.offset+int64(i)*int64(m.rowBytes))
		if err == nil && n != len(row) {
			err = io.ErrUnexpectedEOF
		}
		if err != nil {
			return fmt.Errorf("read native Q8_0 row %d: %w", i, err)
		}
		dst[i] = nativeQ80Dot(row, x)
	}
	return nil
}
