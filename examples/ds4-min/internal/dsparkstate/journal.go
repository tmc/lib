package dsparkstate

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
)

// Journal describes trusted target captures only. Draft KV is
// not a substitute for a row missing from this journal. Device bytes have the
// same row/layer/feature order; snapshots must retain both bytes and metadata.
type Journal struct {
	Rows       []Row
	Frontier   int
	Prefix     [32]byte
	Epoch      uint64
	BasePrefix [32]byte
}

type Row struct {
	Position int
	Token    int
	Prefix   [32]byte
	Epoch    uint64
}

func Shape(context, block, layers, width int) (rows, values int, err error) {
	if context < 1 || block < 1 || layers < 1 || width < 1 || block > context {
		return 0, 0, fmt.Errorf("dspark feature Journal: invalid shape")
	}
	maxInt := int(^uint(0) >> 1)
	if block > maxInt-128-1 {
		return 0, 0, fmt.Errorf("dspark feature Journal: row count overflow")
	}
	rows = min(context, 128+block+1)
	if rows > maxInt/layers || rows*layers > maxInt/width || rows*layers*width > maxInt/4 {
		return 0, 0, fmt.Errorf("dspark feature Journal: byte count overflow")
	}
	return rows, rows * layers * width, nil
}

func NewJournal(rows int, epoch uint64) (*Journal, error) {
	if rows < 1 || epoch == 0 {
		return nil, fmt.Errorf("dspark feature Journal: invalid capacity or epoch")
	}
	j := &Journal{Rows: make([]Row, rows), Epoch: epoch}
	for i := range j.Rows {
		j.Rows[i].Position = -1
	}
	return j, nil
}

// append follows a successful target Eval, after every selected layer's
// device copy completed. A failed pending capture must not call append.
func (j *Journal) Append(position, token int, epoch uint64) error {
	if j == nil || len(j.Rows) == 0 || epoch != j.Epoch || position != j.Frontier || token < 0 || token >= 129280 {
		return fmt.Errorf("dspark feature Journal: noncontiguous or foreign capture")
	}
	if position == int(^uint(0)>>1) {
		return fmt.Errorf("dspark feature Journal: frontier overflow")
	}
	var input [36]byte
	copy(input[:32], j.Prefix[:])
	binary.LittleEndian.PutUint32(input[32:], uint32(token))
	prefix := sha256.Sum256(input[:])
	if position >= len(j.Rows) {
		j.BasePrefix = j.Rows[position%len(j.Rows)].Prefix
	}
	j.Rows[position%len(j.Rows)] = Row{Position: position, Token: token, Prefix: prefix, Epoch: epoch}
	j.Prefix, j.Frontier = prefix, position+1
	return nil
}

func (j *Journal) Row(position int, epoch uint64) (Row, error) {
	if j == nil || len(j.Rows) == 0 || epoch != j.Epoch || position < 0 || position >= j.Frontier {
		return Row{}, fmt.Errorf("dspark feature Journal: invalid row frontier or epoch")
	}
	v := j.Rows[position%len(j.Rows)]
	if v.Position != position || v.Epoch != epoch {
		return Row{}, fmt.Errorf("dspark feature Journal: row missing or overwritten")
	}
	return v, nil
}

func (j *Journal) Clone() *Journal {
	if j == nil {
		return nil
	}
	v := *j
	v.Rows = append([]Row(nil), j.Rows...)
	return &v
}

func (j *Journal) Validate(frontier int) error {
	if j == nil || len(j.Rows) == 0 || j.Epoch == 0 || j.Frontier != frontier || frontier < 0 {
		return fmt.Errorf("dspark feature Journal: invalid snapshot frontier")
	}
	prefix := j.BasePrefix
	for position := max(0, frontier-len(j.Rows)); position < frontier; position++ {
		v, err := j.Row(position, j.Epoch)
		if err != nil {
			return err
		}
		if v.Token < 0 || v.Token >= 129280 {
			return fmt.Errorf("dspark feature Journal: invalid snapshot token")
		}
		var input [36]byte
		copy(input[:32], prefix[:])
		binary.LittleEndian.PutUint32(input[32:], uint32(v.Token))
		prefix = sha256.Sum256(input[:])
		if prefix != v.Prefix {
			return fmt.Errorf("dspark feature Journal: snapshot history mismatch")
		}
	}
	if prefix != j.Prefix {
		return fmt.Errorf("dspark feature Journal: snapshot final prefix mismatch")
	}
	return nil
}

// groupedRow maps a target batch row, not a draft index, to the trusted
// capture's absolute position and physical journal slot.
func (j *Journal) GroupedRow(start, row, count int, epoch uint64) (position, slot int, err error) {
	if j == nil || len(j.Rows) == 0 || epoch != j.Epoch || start != j.Frontier || count < 1 || count > len(j.Rows) || row < 0 || row >= count || start > int(^uint(0)>>1)-row {
		return 0, 0, fmt.Errorf("dspark feature Journal: invalid grouped row")
	}
	position = start + row
	return position, position % len(j.Rows), nil
}
