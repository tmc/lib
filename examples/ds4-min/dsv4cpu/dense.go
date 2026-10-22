package dsv4cpu

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// Q80Activation is a 32-value signed activation block with a float scale.
// The scale is not rounded to half: that would change the CPU dense path.
type Q80Activation struct {
	Scale  float32
	Values [32]int8
}

// QuantizeQ80 quantizes whole activation blocks with a symmetric absolute
// maximum and ties-to-even rounding. It rejects nonfinite inputs and blocks
// whose derived scale or reciprocal is not representable as float32.
func QuantizeQ80(dst []Q80Activation, x []float32) error {
	if len(x) == 0 || len(x)%32 != 0 || len(dst) != len(x)/32 {
		return errors.New("invalid Q8_0 activation shape")
	}
	for _, v := range x {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return errors.New("nonfinite Q8_0 activation")
		}
	}
	for b := range dst {
		row := x[b*32 : (b+1)*32]
		var max float32
		for _, v := range row {
			a := float32(math.Abs(float64(v)))
			if a > max {
				max = a
			}
		}
		d := max / 127
		inv := float32(0)
		if max != 0 && d == 0 {
			return errors.New("non-representable Q8_0 activation scale")
		}
		if d != 0 {
			inv = 1 / d
		}
		if math.IsInf(float64(inv), 0) {
			return errors.New("non-representable Q8_0 activation reciprocal")
		}
		dst[b] = Q80Activation{Scale: d}
		for j, v := range row {
			q := int(math.RoundToEven(float64(float32(v * inv))))
			if q > 127 {
				q = 127
			}
			if q < -128 {
				q = -128
			}
			dst[b].Values[j] = int8(q)
		}
	}
	return nil
}

// MulQ80 runs the native quantized-activation dense/shared-expert dot product.
func (m *Matrix) MulQ80(dst []float32, x []Q80Activation) error {
	if m == nil || m.source == nil || m.encoding != Q80 || len(dst) != m.rows || len(x) != m.cols/32 {
		return errors.New("invalid Q8_0 multiplication shape")
	}
	row := make([]byte, m.rowBytes)
	for r := range dst {
		n, err := m.source.ReadAt(row, m.offset+int64(r)*int64(m.rowBytes))
		if err == nil && n != len(row) {
			err = io.ErrUnexpectedEOF
		}
		if err != nil {
			return fmt.Errorf("read dense row %d: %w", r, err)
		}
		var acc float32
		for b, a := range x {
			q := row[b*34:]
			var dot int32
			for j, v := range a.Values {
				dot += int32(int8(q[2+j])) * int32(v)
			}
			acc += float32(float32(half(binary.LittleEndian.Uint16(q))*a.Scale) * float32(dot))
		}
		dst[r] = acc
	}
	return nil
}
