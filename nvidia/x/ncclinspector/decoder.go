package ncclinspector

import (
	"bufio"
	"errors"
	"fmt"
	"io"
)

// Decoder reads an incremental JSON Lines epoch. It is not safe for concurrent
// use. Limits apply cumulatively to the epoch, including pending bytes.
// The zero value is not usable. The caller owns cancellation and polling.
type Decoder struct {
	ended          bool
	reader         *bufio.Reader
	limits         Limits
	pending        []byte
	bytes, records int
	offset         int64
	err            error
}

// NewDecoder creates a decoder. A reader that returns temporary EOF must allow
// subsequent reads after more data arrives; Decoder does not reset the reader.
func NewDecoder(r io.Reader, limits Limits) (*Decoder, error) {
	l, err := limits.checked()
	if err != nil {
		return nil, err
	}
	if r == nil {
		return nil, errors.New("inspector: nil reader")
	}
	return &Decoder{reader: bufio.NewReaderSize(r, 4096), limits: l}, nil
}

// Offset is the byte offset following the last successfully decoded record.
// Reading an incomplete or invalid record does not advance it.
func (d *Decoder) Offset() int64 { return d.offset }

// Next returns one record. io.EOF means temporary end of input: pending bytes
// are retained and the caller may call Next after an append. No empty record is
// returned successfully. Other errors are terminal for this decoder.
func (d *Decoder) Next() (Record, error) {
	if d == nil || d.reader == nil {
		return Record{}, errors.New("inspector: uninitialized decoder")
	}
	if d.err != nil {
		return Record{}, d.err
	}
	for {
		b, err := d.reader.ReadByte()
		if err != nil {
			if err != io.EOF {
				d.err = fmt.Errorf("inspector read: %w", err)
				return Record{}, d.err
			}
			d.ended = true
			return Record{}, io.EOF
		}
		d.ended = false
		d.bytes++
		if d.bytes > d.limits.Bytes {
			d.err = errors.New("inspector: byte limit exceeded")
			return Record{}, d.err
		}
		if b != '\n' && len(d.pending) >= d.limits.Line {
			d.err = errors.New("inspector: line limit exceeded")
			return Record{}, d.err
		}
		d.pending = append(d.pending, b)
		if b != '\n' {
			continue
		}
		if d.records >= d.limits.Records {
			d.err = errors.New("inspector: record limit exceeded")
			return Record{}, d.err
		}
		r, err := readRecord(d.pending)
		if err != nil {
			d.err = fmt.Errorf("inspector record %d: %w", d.records, err)
			return Record{}, d.err
		}
		d.offset += int64(len(d.pending))
		d.records++
		d.pending = d.pending[:0]
		return r, nil
	}
}

// Finish declares the epoch complete. It refuses pending bytes, an empty epoch
// or a prior failure. Call Next until io.EOF before Finish; Finish does not read.
func (d *Decoder) Finish() error {
	if d == nil || d.reader == nil {
		return errors.New("inspector: uninitialized decoder")
	}
	if d.err != nil {
		return d.err
	}
	if len(d.pending) != 0 || d.records == 0 {
		return errors.New("inspector: empty or incomplete JSON Lines capture")
	}
	if !d.ended || d.reader.Buffered() != 0 {
		return errors.New("inspector: unread buffered input")
	}
	return nil
}
