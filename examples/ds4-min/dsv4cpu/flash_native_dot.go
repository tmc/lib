package dsv4cpu

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// fused32 rounds the exact product and addition once to float32. The float32
// product is exact in float64. TwoSum retains the low part of the addition;
// at a float32 halfway point it resolves the otherwise possible double round.
func fused32(a, b, c float32) float32 {
	p := float64(a) * float64(b)
	q := float64(c)
	s := math.FMA(float64(a), float64(b), q)
	if math.IsNaN(s) || math.IsInf(s, 0) || s == 0 {
		return float32(s)
	}
	v := s - p
	e := (p - (s - v)) + (q - v)
	r := float32(s)
	if e == 0 {
		return r
	}
	var midpoint float64
	if math.IsInf(float64(r), 0) {
		midpoint = math.Copysign(math.Ldexp(1, 128)-math.Ldexp(1, 103), s)
	} else {
		n := math.Nextafter32(r, float32(math.Copysign(math.Inf(1), s-float64(r))))
		midpoint = (float64(r) + float64(n)) / 2
	}
	if s == midpoint {
		s = math.Nextafter(s, math.Copysign(math.Inf(1), e))
	}
	return float32(s)
}

// nativeF16Dot matches pinned dot_f16_row's ARM NEON eight-value schedule.
func nativeF16Dot(row []byte, x []float32) float32 {
	var a, b [4]float32
	i := 0
	for ; i+8 <= len(x); i += 8 {
		for lane := 0; lane < 4; lane++ {
			a[lane] = fused32(half(binary.LittleEndian.Uint16(row[(i+lane)*2:])), x[i+lane], a[lane])
			b[lane] = fused32(half(binary.LittleEndian.Uint16(row[(i+lane+4)*2:])), x[i+lane+4], b[lane])
		}
	}
	for lane := range a {
		a[lane] += b[lane]
	}
	acc := float32(a[0]+a[1]) + float32(a[2]+a[3])
	for ; i < len(x); i++ {
		acc = fused32(half(binary.LittleEndian.Uint16(row[i*2:])), x[i], acc)
	}
	return acc
}

func nativeF16Mul(dst, x []float32, m *Matrix) error {
	if m == nil || m.source == nil || m.encoding != F16 || m.rows <= 0 || m.cols <= 0 || len(dst) != m.rows || len(x) != m.cols {
		return errors.New("invalid native F16 projection")
	}
	row := make([]byte, m.rowBytes)
	for i := range dst {
		n, err := m.source.ReadAt(row, m.offset+int64(i*m.rowBytes))
		if err != nil {
			return fmt.Errorf("read native F16 row %d: %w", i, err)
		}
		if n != len(row) {
			return fmt.Errorf("read native F16 row %d: %w", i, io.ErrUnexpectedEOF)
		}
		dst[i] = nativeF16Dot(row, x)
	}
	return nil
}
