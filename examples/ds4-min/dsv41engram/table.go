package dsv41engram

import (
	"fmt"
	"io"
	"math"
	"sort"
)

// Table is a bounded view over immutable serialized rows. The reader and its
// contents must remain usable and unchanged for the lifetime of the Table.
// Table does not own or close the reader. Its zero value is not usable.
type Table struct {
	reader io.ReaderAt
	offset int64
	rows   uint32
}

// NewTable checks that all rows fit in size. Neither construction nor reads
// allocate storage proportional to the complete table.
func NewTable(reader io.ReaderAt, size, offset int64, rows uint32) (*Table, error) {
	if reader == nil || rows == 0 || size < 0 || offset < 0 || offset > size || int64(rows)*RowBytes > size-offset {
		return nil, fmt.Errorf("invalid table bounds")
	}
	return &Table{reader: reader, offset: offset, rows: rows}, nil
}

func (t *Table) check(ids []uint32, length, width int) error {
	if t == nil || t.reader == nil {
		return fmt.Errorf("invalid table")
	}
	if len(ids) > math.MaxInt/width || length != len(ids)*width {
		return fmt.Errorf("invalid output length")
	}
	for i, id := range ids {
		if id >= t.rows {
			return fmt.Errorf("row %d out of range", i)
		}
	}
	return nil
}

func (t *Table) read(id uint32, dst []byte) error {
	n, err := t.reader.ReadAt(dst, t.offset+int64(id)*RowBytes)
	if n != len(dst) {
		if err == nil {
			err = io.ErrUnexpectedEOF
		}
		return fmt.Errorf("read row %d: %w", id, err)
	}
	if err != nil {
		return fmt.Errorf("read row %d: %w", id, err)
	}
	return nil
}

// ReadEncodedRows writes count*RowBytes bytes in the requested order. The raw
// E4M3/E8M0 encoding is intended for a backend's own row decoder. On failure
// the entire output must be considered invalid.
func (t *Table) ReadEncodedRows(ids []uint32, dst []byte) error {
	if err := t.check(ids, len(dst), RowBytes); err != nil {
		return err
	}
	for i, id := range ids {
		if err := t.read(id, dst[i*RowBytes:(i+1)*RowBytes]); err != nil {
			return err
		}
	}
	return nil
}

// DecodeRow decodes a serialized row into 256 BF16-rounded float32 values.
// Invalid E4M3 codes, scales and non-finite results are rejected. On failure
// the output must be considered invalid.
func DecodeRow(raw []byte, dst []float32) error {
	if len(raw) != RowBytes || len(dst) != Dim {
		return fmt.Errorf("invalid row length")
	}
	for j := range Dim {
		code, scale := raw[j], raw[Dim+j/32]
		if code&127 == 127 || scale == 255 {
			return fmt.Errorf("invalid encoding at column %d", j)
		}
		e, m := int(code>>3&15), int(code&7)
		var v float64
		if e == 0 {
			v = math.Ldexp(float64(m), -9)
		} else {
			v = math.Ldexp(float64(8+m), e-10)
		}
		if code&128 != 0 {
			v = -v
		}
		value := float32(math.Ldexp(v, int(scale)-127))
		bits := math.Float32bits(value)
		bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
		value = math.Float32frombits(bits)
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("non-finite value at column %d", j)
		}
		dst[j] = value
	}
	return nil
}

// ReadRows writes BF16-rounded float32 rows in the requested order. On failure
// the entire output must be considered invalid.
func (t *Table) ReadRows(ids []uint32, dst []float32) error {
	if err := t.check(ids, len(dst), Dim); err != nil {
		return err
	}
	var raw [RowBytes]byte
	for i, id := range ids {
		if err := t.read(id, raw[:]); err != nil {
			return err
		}
		if err := DecodeRow(raw[:], dst[i*Dim:(i+1)*Dim]); err != nil {
			return fmt.Errorf("decode row %d: %w", id, err)
		}
	}
	return nil
}

type request struct {
	row    uint32
	output uint32
}

// ReadBatch reads 24 IDs per token from rows, using stride row IDs between
// tokens. It deduplicates reads within chunks of at most 2048 tokens, restores
// token order and uses at most 384 KiB of request storage plus one serialized
// row and one decoded row.
// Output is packed [token][column][dimension]. On failure it is invalid.
func (t *Table) ReadBatch(rows []uint32, tokens, stride int, dst []float32) error {
	if t == nil || t.reader == nil || tokens < 0 || tokens > math.MaxInt/(Columns*Dim) || len(dst) != tokens*Columns*Dim || tokens > 0 && (stride < Columns || tokens-1 > (math.MaxInt-Columns)/stride || len(rows) < (tokens-1)*stride+Columns) {
		return fmt.Errorf("invalid batch shape")
	}
	for i := 0; i < tokens; i++ {
		for _, id := range rows[i*stride : i*stride+Columns] {
			if id >= t.rows {
				return fmt.Errorf("batch row out of range")
			}
		}
	}
	capTokens := min(tokens, 2048)
	requests := make([]request, capTokens*Columns)
	var raw [RowBytes]byte
	var decoded [Dim]float32
	for start := 0; start < tokens; start += capTokens {
		n := min(tokens-start, capTokens) * Columns
		for i := range n {
			requests[i] = request{rows[(start+i/Columns)*stride+i%Columns], uint32(i)}
		}
		sort.Slice(requests[:n], func(i, j int) bool { return requests[i].row < requests[j].row })
		var previous uint32
		for i, r := range requests[:n] {
			if i == 0 || r.row != previous {
				if err := t.read(r.row, raw[:]); err != nil {
					return err
				}
				if err := DecodeRow(raw[:], decoded[:]); err != nil {
					return fmt.Errorf("decode row %d: %w", r.row, err)
				}
				previous = r.row
			}
			out := (start*Columns + int(r.output)) * Dim
			copy(dst[out:out+Dim], decoded[:])
		}
	}
	return nil
}
