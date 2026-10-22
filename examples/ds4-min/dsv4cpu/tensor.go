package dsv4cpu

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// Encoding identifies a GGUF tensor encoding supported by Matrix.
type Encoding uint32

const (
	F32    Encoding = 0
	F16    Encoding = 1
	Q80    Encoding = 8
	Q2K    Encoding = 10
	Q4K    Encoding = 12
	IQ2XXS Encoding = 16
)

// Matrix is an immutable view of row-major weights. The ReaderAt must remain
// immutable and open while the matrix is in use. Operations allocate only one
// serialized row, not an expanded copy of the model.
// The zero value is not usable; construct a Matrix with NewMatrix.
type Matrix struct {
	source               io.ReaderAt
	offset               int64
	rows, cols, rowBytes int
	encoding             Encoding
}

// NewMatrix validates a bounded serialized matrix view.
func NewMatrix(source io.ReaderAt, offset, size int64, rows, cols int, encoding Encoding) (*Matrix, error) {
	if source == nil || offset < 0 || size < 0 || rows <= 0 || cols <= 0 {
		return nil, errors.New("invalid matrix shape")
	}
	var n int64
	if int64(cols) > math.MaxInt64/4 {
		return nil, errors.New("matrix row byte extent overflows")
	}
	switch encoding {
	case F32:
		n = int64(cols) * 4
	case F16:
		n = int64(cols) * 2
	case Q80:
		if cols%32 != 0 {
			return nil, errors.New("Q8_0 columns are not whole blocks")
		}
		n = int64(cols/32) * 34
	case Q2K:
		if cols%256 != 0 {
			return nil, errors.New("Q2_K columns are not whole blocks")
		}
		n = int64(cols/256) * 84
	case Q4K:
		if cols%256 != 0 {
			return nil, errors.New("Q4_K columns are not whole blocks")
		}
		n = int64(cols/256) * 144
	case IQ2XXS:
		if cols%256 != 0 {
			return nil, errors.New("IQ2_XXS columns are not whole blocks")
		}
		n = int64(cols/256) * 66
	default:
		return nil, fmt.Errorf("unsupported encoding %d", encoding)
	}
	if n <= 0 || n > int64(int(^uint(0)>>1)) || int64(rows) > math.MaxInt64/n || n*int64(rows) != size || offset > math.MaxInt64-size {
		return nil, errors.New("invalid matrix byte extent")
	}
	return &Matrix{source, offset, rows, cols, int(n), encoding}, nil
}

// Row decodes one row into dst. It rejects a short read without publishing a
// partial decoded row.
func (m *Matrix) Row(dst []float32, row int) error {
	if m == nil || m.source == nil || m.rows <= 0 || m.cols <= 0 {
		return errors.New("invalid matrix")
	}
	if row < 0 || row >= m.rows || len(dst) != m.cols {
		return errors.New("invalid matrix row")
	}
	b := make([]byte, m.rowBytes)
	n, err := m.source.ReadAt(b, m.offset+int64(row)*int64(m.rowBytes))
	if err == nil && n != len(b) {
		err = io.ErrUnexpectedEOF
	}
	if err != nil {
		return fmt.Errorf("read matrix row %d: %w", row, err)
	}
	m.decode(dst, b)
	return nil
}

func (m *Matrix) decode(dst []float32, b []byte) {
	for i := range dst {
		switch m.encoding {
		case Q4K:
			dst[i] = q4KValue(b[(i/256)*144:], i%256)
		case F32:
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
		case F16:
			dst[i] = half(binary.LittleEndian.Uint16(b[i*2:]))
		case Q80:
			block := b[(i/32)*34:]
			dst[i] = half(binary.LittleEndian.Uint16(block)) * float32(int8(block[2+i%32]))
		case Q2K:
			block := b[i/256*84:]
			idx := i % 256
			group, l := idx/16, idx%16
			base := 32*(group/8) + 16*(group&1)
			q := (block[16+base+l] >> uint(((group/2)&3)*2)) & 3
			scale := block[group]
			d := half(binary.LittleEndian.Uint16(block[80:]))
			dmin := half(binary.LittleEndian.Uint16(block[82:]))
			dst[i] = float32(float32(d*float32(scale&15))*float32(q)) - float32(dmin*float32(scale>>4))
		case IQ2XXS:
			block := b[i/256*66:]
			j := i % 256
			group, lane, element := j/32, (j%32)/8, j%8
			codes := block[2+group*8:]
			aux := binary.LittleEndian.Uint32(codes[4:])
			scale := float32(.125*half(binary.LittleEndian.Uint16(block))) * float32(2*(aux>>28)+1)
			grid := byte(iq2xxs_grid[codes[lane]] >> uint(element*8))
			sign := ksigns_iq2xs[(aux>>uint(7*lane))&127]
			v := float32(grid)
			if sign&(1<<uint(element)) != 0 {
				v = -v
			}
			dst[i] = scale * v
		}
	}
}

// Mul computes dst = m*x in serialized row order. dst and x must not overlap.
func (m *Matrix) Mul(dst, x []float32) error {
	if m == nil || m.source == nil || m.rows <= 0 || m.cols <= 0 {
		return errors.New("invalid matrix")
	}
	if len(dst) != m.rows || len(x) != m.cols {
		return errors.New("invalid matrix multiplication shape")
	}
	row := make([]float32, m.cols)
	for i := range dst {
		if err := m.Row(row, i); err != nil {
			return err
		}
		var sum float32
		var wide float64
		for j, v := range row {
			if m.encoding == F32 {
				wide += float64(v) * float64(x[j])
			} else {
				sum += float32(v * x[j])
			}
		}
		if m.encoding == F32 {
			sum = float32(wide)
		}
		dst[i] = sum
	}
	return nil
}

// RMSNorm implements the upstream scalar weighted RMS normalization order.
// dst may alias x. A nil weight selects normalization without learned scales.
func RMSNorm(dst, x, weight []float32, eps float32) error {
	if len(x) == 0 || len(dst) != len(x) || (weight != nil && len(weight) != len(x)) || !(eps > 0) || math.IsInf(float64(eps), 0) {
		return errors.New("invalid RMS normalization shape")
	}
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	scale := float32(1) / float32(math.Sqrt(float64(float32(ss/float64(len(x)))+eps)))
	for i, v := range x {
		dst[i] = v * scale
		if weight != nil {
			dst[i] *= weight[i]
		}
	}
	return nil
}

func half(v uint16) float32 {
	s := uint32(v&0x8000) << 16
	e, f := uint32(v>>10)&31, uint32(v&1023)
	if e == 0 {
		if f == 0 {
			return math.Float32frombits(s)
		}
		e = 113
		for f&1024 == 0 {
			f <<= 1
			e--
		}
		return math.Float32frombits(s | e<<23 | (f&1023)<<13)
	}
	if e == 31 {
		return math.Float32frombits(s | 0x7f800000 | f<<13)
	}
	return math.Float32frombits(s | (e+112)<<23 | f<<13)
}
