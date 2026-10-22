package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// denseRow decodes one row of actual immutable support weights. The caller
// keeps the admitted ReaderAt open for the support model's lifetime. This
// reads F32/F16/Q8_0 only; routed expert tensors use their own forward path.
func (m *dsparkModel) denseRow(name string, row int) ([]float32, error) {
	if m == nil || m.reader == nil {
		return nil, fmt.Errorf("dspark dense: no admitted reader")
	}
	t, ok := m.tensor(name)
	if !ok || len(t.Dimensions) < 1 || len(t.Dimensions) > 2 || row < 0 || (len(t.Dimensions) == 1 && row != 0) || (len(t.Dimensions) == 2 && uint64(row) >= t.Dimensions[1]) {
		return nil, fmt.Errorf("dspark dense: invalid row for %s", name)
	}
	width := int(t.Dimensions[0])
	if width < 1 || width > Flash.Vocab {
		return nil, fmt.Errorf("dspark dense: unsupported row width")
	}
	bytes := width * 4
	switch t.TypeID {
	case 0:
	case 1:
		bytes = width * 2
	case 8:
		if width%32 != 0 {
			return nil, errShape
		}
		bytes = width / 32 * 34
	default:
		return nil, fmt.Errorf("dspark dense: unsupported type %d", t.TypeID)
	}
	b := make([]byte, bytes)
	offset := m.dataOffset + t.Offset + uint64(row)*uint64(bytes)
	if offset > math.MaxInt64 {
		return nil, fmt.Errorf("dspark dense: offset overflow")
	}
	if n, err := m.reader.ReadAt(b, int64(offset)); err != nil {
		return nil, fmt.Errorf("dspark dense %s row %d: %w", name, row, err)
	} else if n != len(b) {
		return nil, fmt.Errorf("dspark dense %s row %d: %w", name, row, io.ErrUnexpectedEOF)
	}
	out := make([]float32, width)
	for i := range out {
		switch t.TypeID {
		case 0:
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
		case 1:
			out[i] = dsparkHalf(binary.LittleEndian.Uint16(b[i*2:]))
		case 8:
			block := i / 32
			scale := dsparkHalf(binary.LittleEndian.Uint16(b[block*34:]))
			out[i] = scale * float32(int8(b[block*34+2+i%32]))
		}
		if math.IsNaN(float64(out[i])) || math.IsInf(float64(out[i]), 0) {
			return nil, fmt.Errorf("dspark dense: nonfinite weight")
		}
	}
	return out, nil
}

func dsparkHalf(bits uint16) float32 {
	negative := bits&0x8000 != 0
	e, m := int(bits>>10&31), int(bits&1023)
	var x float64
	switch e {
	case 0:
		x = math.Ldexp(float64(m), -24)
	case 31:
		if m != 0 {
			return float32(math.NaN())
		}
		x = math.Inf(1)
	default:
		x = math.Ldexp(float64(1024+m), e-25)
	}
	if negative {
		x = -x
	}
	return float32(x)
}

func (m *dsparkModel) denseMatvec(name string, input []float32) ([]float32, error) {
	t, ok := m.tensor(name)
	if !ok || len(t.Dimensions) != 2 || len(input) != int(t.Dimensions[0]) || !allFinite(input) {
		return nil, errShape
	}
	out := make([]float32, int(t.Dimensions[1]))
	for row := range out {
		weights, err := m.denseRow(name, row)
		if err != nil {
			return nil, err
		}
		for i, x := range input {
			out[row] += weights[i] * x
		}
	}
	if !allFinite(out) {
		return nil, fmt.Errorf("dspark dense: nonfinite projection")
	}
	return out, nil
}
