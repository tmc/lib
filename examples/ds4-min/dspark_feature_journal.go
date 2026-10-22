package main

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
)

// dsparkFeatureJournal describes trusted target captures only. Draft KV is
// not a substitute for a row missing from this journal. Device bytes have the
// same row/layer/feature order; snapshots must retain both bytes and metadata.
type dsparkFeatureJournal struct {
	rows       []dsparkFeatureRow
	frontier   int
	prefix     [32]byte
	epoch      uint64
	basePrefix [32]byte
}

type dsparkFeatureRow struct {
	position int
	token    int
	prefix   [32]byte
	epoch    uint64
}

func dsparkFeatureJournalShape(context, block, layers, width int) (rows, values int, err error) {
	if context < 1 || block < 1 || layers < 1 || width < 1 || block > context {
		return 0, 0, fmt.Errorf("dspark feature journal: invalid shape")
	}
	maxInt := int(^uint(0) >> 1)
	if block > maxInt-Flash.SlidingWindow-1 {
		return 0, 0, fmt.Errorf("dspark feature journal: row count overflow")
	}
	rows = min(context, Flash.SlidingWindow+block+1)
	if rows > maxInt/layers || rows*layers > maxInt/width || rows*layers*width > maxInt/4 {
		return 0, 0, fmt.Errorf("dspark feature journal: byte count overflow")
	}
	return rows, rows * layers * width, nil
}

func newDSparkFeatureJournal(rows int, epoch uint64) (*dsparkFeatureJournal, error) {
	if rows < 1 || epoch == 0 {
		return nil, fmt.Errorf("dspark feature journal: invalid capacity or epoch")
	}
	j := &dsparkFeatureJournal{rows: make([]dsparkFeatureRow, rows), epoch: epoch}
	for i := range j.rows {
		j.rows[i].position = -1
	}
	return j, nil
}

// append follows a successful target Eval, after every selected layer's
// device copy completed. A failed pending capture must not call append.
func (j *dsparkFeatureJournal) append(position, token int, epoch uint64) error {
	if j == nil || len(j.rows) == 0 || epoch != j.epoch || position != j.frontier || token < 0 || token >= Flash.Vocab {
		return fmt.Errorf("dspark feature journal: noncontiguous or foreign capture")
	}
	if position == int(^uint(0)>>1) {
		return fmt.Errorf("dspark feature journal: frontier overflow")
	}
	var input [36]byte
	copy(input[:32], j.prefix[:])
	binary.LittleEndian.PutUint32(input[32:], uint32(token))
	prefix := sha256.Sum256(input[:])
	if position >= len(j.rows) {
		j.basePrefix = j.rows[position%len(j.rows)].prefix
	}
	j.rows[position%len(j.rows)] = dsparkFeatureRow{position: position, token: token, prefix: prefix, epoch: epoch}
	j.prefix, j.frontier = prefix, position+1
	return nil
}

func (j *dsparkFeatureJournal) row(position int, epoch uint64) (dsparkFeatureRow, error) {
	if j == nil || len(j.rows) == 0 || epoch != j.epoch || position < 0 || position >= j.frontier {
		return dsparkFeatureRow{}, fmt.Errorf("dspark feature journal: invalid row frontier or epoch")
	}
	v := j.rows[position%len(j.rows)]
	if v.position != position || v.epoch != epoch {
		return dsparkFeatureRow{}, fmt.Errorf("dspark feature journal: row missing or overwritten")
	}
	return v, nil
}

func (j *dsparkFeatureJournal) clone() *dsparkFeatureJournal {
	if j == nil {
		return nil
	}
	v := *j
	v.rows = append([]dsparkFeatureRow(nil), j.rows...)
	return &v
}

func (j *dsparkFeatureJournal) validate(frontier int) error {
	if j == nil || len(j.rows) == 0 || j.epoch == 0 || j.frontier != frontier || frontier < 0 {
		return fmt.Errorf("dspark feature journal: invalid snapshot frontier")
	}
	prefix := j.basePrefix
	for position := max(0, frontier-len(j.rows)); position < frontier; position++ {
		v, err := j.row(position, j.epoch)
		if err != nil {
			return err
		}
		if v.token < 0 || v.token >= Flash.Vocab {
			return fmt.Errorf("dspark feature journal: invalid snapshot token")
		}
		var input [36]byte
		copy(input[:32], prefix[:])
		binary.LittleEndian.PutUint32(input[32:], uint32(v.token))
		prefix = sha256.Sum256(input[:])
		if prefix != v.prefix {
			return fmt.Errorf("dspark feature journal: snapshot history mismatch")
		}
	}
	if prefix != j.prefix {
		return fmt.Errorf("dspark feature journal: snapshot final prefix mismatch")
	}
	return nil
}

// groupedRow maps a target batch row, not a draft index, to the trusted
// capture's absolute position and physical journal slot.
func (j *dsparkFeatureJournal) groupedRow(start, row, count int, epoch uint64) (position, slot int, err error) {
	if j == nil || len(j.rows) == 0 || epoch != j.epoch || start != j.frontier || count < 1 || count > len(j.rows) || row < 0 || row >= count || start > int(^uint(0)>>1)-row {
		return 0, 0, fmt.Errorf("dspark feature journal: invalid grouped row")
	}
	position = start + row
	return position, position % len(j.rows), nil
}
