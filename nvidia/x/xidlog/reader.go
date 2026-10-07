package xidlog

import (
	"bufio"
	"errors"
	"io"
	"syscall"
)

// Format selects supplied line input or a one-record-per-Read kmsg reader.
type Format uint8

const (
	Text Format = iota + 1
	KernelRecords
)

// Reader parses bounded caller-owned input. It is not safe for concurrent Next
// calls. EOF means this supplied input ended, not that a boot is complete.
type Reader struct {
	input          io.Reader
	lines          *bufio.Reader
	format         Format
	options        Options
	bytes, records int
	done           bool
	pendingGap     bool
	buffer         []byte
}

// NewReader validates provenance, limits and input framing. It never opens files.
func NewReader(input io.Reader, format Format, o Options) (*Reader, error) {
	o, err := options(o)
	if err != nil {
		return nil, err
	}
	if input == nil {
		return nil, errors.New("xidlog: nil reader")
	}
	if format != Text && format != KernelRecords {
		return nil, errors.New("xidlog: unsupported format")
	}
	r := &Reader{input: input, format: format, options: o}
	if format == Text {
		r.lines = bufio.NewReaderSize(input, o.MaxRecordBytes+1)
	} else {
		r.buffer = make([]byte, o.MaxRecordBytes+1)
	}
	return r, nil
}

// Next returns facts and explicit malformed/fragment/gap records. Limits stop
// reading with an error. Kernel EPIPE produces Gap and permits subsequent reads;
// other I/O errors are returned unchanged. No lost count is inferred.
func (r *Reader) Next() (Record, error) {
	if r == nil {
		return Record{}, errors.New("xidlog: nil reader")
	}
	if r.done {
		return Record{}, io.EOF
	}
	if r.records >= r.options.MaxRecords {
		r.done = true
		return Record{}, errors.New("xidlog: record limit reached")
	}
	if r.pendingGap {
		r.pendingGap = false
		r.records++
		v := record("", r.options)
		v.Kind = Gap
		v.Problem = "kmsg EPIPE: records overwritten"
		return v, nil
	}
	var raw []byte
	var err error
	if r.format == Text {
		raw, err = r.lines.ReadSlice('\n')
	} else {
		var n int
		n, err = r.input.Read(r.buffer)
		if n < 0 || n > len(r.buffer) {
			r.done = true
			return Record{}, errors.New("xidlog: invalid read extent")
		}
		raw = r.buffer[:n]
	}
	if len(raw) > r.options.MaxRecordBytes || errors.Is(err, bufio.ErrBufferFull) || len(raw) > r.options.MaxBytes-r.bytes {
		r.done = true
		return Record{}, errors.New("xidlog: input byte limit reached")
	}
	r.bytes += len(raw)
	if r.format == KernelRecords && errors.Is(err, syscall.EPIPE) {
		if len(raw) == 0 {
			r.records++
			v := record("", r.options)
			v.Kind = Gap
			v.Problem = "kmsg EPIPE: records overwritten"
			return v, nil
		}
		r.pendingGap = true
		r.records++
		return malformed(record(string(raw), r.options), "data returned with kmsg EPIPE"), nil
	}
	if len(raw) == 0 {
		if err == io.EOF {
			r.done = true
		}
		if err == nil {
			err = io.ErrNoProgress
		}
		return Record{}, err
	}
	r.records++
	if err != nil && err != io.EOF {
		r.done = true
		return malformed(record(string(raw), r.options), "data returned with read error"), err
	}
	if err == io.EOF {
		r.done = true
	}
	if r.format == KernelRecords {
		return ParseKmsg(string(raw), r.options)
	}
	if raw[len(raw)-1] != '\n' {
		v := record(string(raw), r.options)
		v.Partial = true
		return malformed(v, "unterminated text line"), nil
	}
	return ParseText(string(raw), r.options)
}
