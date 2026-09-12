// Package wire defines the bounded, versioned records exchanged by a CUDA
// capture agent and its controller.
package wire

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"sync"
)

const (
	Version           uint16 = 1
	HeaderSize               = 80
	DefaultMaxPayload        = 16 << 20
	DefaultMaxRecords        = 4096
	DefaultQueueBytes        = 64 << 20
)

var (
	ErrBadMagic    = errors.New("wire: bad magic")
	ErrBadVersion  = errors.New("wire: unsupported version")
	ErrMalformed   = errors.New("wire: malformed frame")
	ErrOversize    = errors.New("wire: frame exceeds limit")
	ErrSession     = errors.New("wire: session identity mismatch")
	ErrDuplicate   = errors.New("wire: duplicate sequence")
	ErrRecordLimit = errors.New("wire: record count exceeds limit")
	ErrQueueFull   = errors.New("wire: queue full")
)

// Limits bounds allocations and queue growth. Zero fields use package defaults.
type Limits struct {
	MaxPayload, MaxRecords, MaxFrame uint32
	QueueRecords, QueueBytes         uint32
}

func (l Limits) normalized() Limits {
	if l.MaxPayload == 0 {
		l.MaxPayload = DefaultMaxPayload
	}
	if l.MaxRecords == 0 {
		l.MaxRecords = DefaultMaxRecords
	}
	if l.MaxFrame == 0 {
		l.MaxFrame = HeaderSize + l.MaxPayload
	}
	if l.QueueRecords == 0 {
		l.QueueRecords = DefaultMaxRecords
	}
	if l.QueueBytes == 0 {
		l.QueueBytes = DefaultQueueBytes
	}
	return l
}

// Identity distinguishes a capture session and a process generation. Process
// is normally a PID; Generation must change across exec or PID reuse.
type Identity struct {
	Session    [16]byte
	Process    uint64
	Generation uint64
}

// Record is one independently framed observation. Sequence numbers start at
// one within each process generation. Dropped is cumulative producer loss.
type Record struct {
	Kind uint16
	Identity
	Sequence  uint64
	Dropped   uint64
	Timestamp uint64
	Payload   []byte
}

var magic = [4]byte{'C', 'D', 'W', '1'}

func (r Record) MarshalBinary(l Limits) ([]byte, error) {
	l = l.normalized()
	if len(r.Payload) > int(l.MaxPayload) || uint64(HeaderSize)+uint64(len(r.Payload)) > uint64(l.MaxFrame) {
		return nil, ErrOversize
	}
	if r.Sequence == 0 {
		return nil, fmt.Errorf("%w: zero sequence", ErrMalformed)
	}
	b := make([]byte, HeaderSize+len(r.Payload))
	copy(b[:4], magic[:])
	binary.BigEndian.PutUint16(b[4:6], Version)
	binary.BigEndian.PutUint16(b[6:8], r.Kind)
	binary.BigEndian.PutUint32(b[8:12], uint32(len(r.Payload)))
	copy(b[12:28], r.Session[:])
	binary.BigEndian.PutUint64(b[28:36], r.Process)
	binary.BigEndian.PutUint64(b[36:44], r.Generation)
	binary.BigEndian.PutUint64(b[44:52], r.Sequence)
	binary.BigEndian.PutUint64(b[52:60], r.Dropped)
	binary.BigEndian.PutUint64(b[60:68], r.Timestamp)
	copy(b[HeaderSize:], r.Payload)
	binary.BigEndian.PutUint32(b[68:72], crc32.ChecksumIEEE(b[:68]))
	binary.BigEndian.PutUint32(b[72:76], crc32.ChecksumIEEE(b[HeaderSize:]))
	// 76:80 is reserved and must remain zero for forward compatibility.
	return b, nil
}

// Write writes one complete frame. It does not retry a short write that
// returns no error; io.ErrShortWrite is returned in that case.
func Write(w io.Writer, r Record, l Limits) error {
	b, err := r.MarshalBinary(l)
	if err != nil {
		return err
	}
	for len(b) > 0 {
		n, e := w.Write(b)
		if n > 0 {
			b = b[n:]
		}
		if e != nil {
			return e
		}
		if n == 0 {
			return io.ErrShortWrite
		}
	}
	return nil
}

// Decoder validates framing and maintains sequence/drop accounting.
type Decoder struct {
	r        io.Reader
	limits   Limits
	expected *[16]byte
	mu       sync.Mutex
	last     map[stream]streamState
	stats    Stats
}
type stream struct{ id Identity }
type streamState struct{ sequence, dropped uint64 }

// Stats describes validated records and observed sequence gaps.
type Stats struct{ Records, Bytes, Gaps, Dropped uint64 }

func NewDecoder(r io.Reader, l Limits) *Decoder {
	return &Decoder{r: r, limits: l.normalized(), last: make(map[stream]streamState)}
}
func (d *Decoder) ExpectSession(id [16]byte) { d.expected = &id }
func (d *Decoder) Stats() Stats              { d.mu.Lock(); defer d.mu.Unlock(); return d.stats }

// Next returns the next validated record. A truncated terminal frame returns
// io.ErrUnexpectedEOF, allowing journal recovery to retain the valid prefix.
func (d *Decoder) Next() (Record, error) {
	var h [HeaderSize]byte
	if _, err := io.ReadFull(d.r, h[:]); err != nil {
		return Record{}, err
	}
	if !bytes.Equal(h[:4], magic[:]) {
		return Record{}, ErrBadMagic
	}
	if binary.BigEndian.Uint16(h[4:6]) != Version {
		return Record{}, ErrBadVersion
	}
	if binary.BigEndian.Uint32(h[76:80]) != 0 {
		return Record{}, ErrMalformed
	}
	n := binary.BigEndian.Uint32(h[8:12])
	if n > d.limits.MaxPayload || uint64(HeaderSize)+uint64(n) > uint64(d.limits.MaxFrame) {
		return Record{}, ErrOversize
	}
	p := make([]byte, n)
	if _, err := io.ReadFull(d.r, p); err != nil {
		return Record{}, err
	}
	if binary.BigEndian.Uint32(h[68:72]) != crc32.ChecksumIEEE(h[:68]) || binary.BigEndian.Uint32(h[72:76]) != crc32.ChecksumIEEE(p) {
		return Record{}, ErrMalformed
	}
	var id Identity
	copy(id.Session[:], h[12:28])
	id.Process = binary.BigEndian.Uint64(h[28:36])
	id.Generation = binary.BigEndian.Uint64(h[36:44])
	if d.expected != nil && id.Session != *d.expected {
		return Record{}, ErrSession
	}
	r := Record{Kind: binary.BigEndian.Uint16(h[6:8]), Identity: id, Sequence: binary.BigEndian.Uint64(h[44:52]), Dropped: binary.BigEndian.Uint64(h[52:60]), Timestamp: binary.BigEndian.Uint64(h[60:68]), Payload: p}
	if r.Sequence == 0 {
		return Record{}, fmt.Errorf("%w: zero sequence", ErrMalformed)
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.stats.Records >= uint64(d.limits.MaxRecords) {
		return Record{}, ErrRecordLimit
	}
	k := stream{id}
	state := d.last[k]
	if r.Sequence <= state.sequence {
		return Record{}, ErrDuplicate
	}
	if state.sequence != 0 && r.Sequence-state.sequence > 1 {
		d.stats.Gaps += r.Sequence - state.sequence - 1
	}
	if r.Dropped < state.dropped {
		return Record{}, fmt.Errorf("%w: decreasing drop count", ErrMalformed)
	}
	d.stats.Dropped += r.Dropped - state.dropped
	d.last[k] = streamState{sequence: r.Sequence, dropped: r.Dropped}
	d.stats.Records++
	d.stats.Bytes += uint64(len(p))
	return r, nil
}

// Queue is a bounded nonblocking producer queue. Push copies the payload.
type Queue struct {
	mu      sync.Mutex
	limits  Limits
	records []Record
	bytes   uint32
	dropped uint64
	closed  bool
}

func NewQueue(l Limits) *Queue { l = l.normalized(); return &Queue{limits: l} }
func (q *Queue) Push(r Record) error {
	if _, err := r.MarshalBinary(q.limits); err != nil {
		return err
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.closed {
		return io.ErrClosedPipe
	}
	if uint32(len(q.records)) >= q.limits.QueueRecords || uint64(q.bytes)+uint64(len(r.Payload)) > uint64(q.limits.QueueBytes) {
		q.dropped++
		return ErrQueueFull
	}
	r.Payload = append([]byte(nil), r.Payload...)
	q.records = append(q.records, r)
	q.bytes += uint32(len(r.Payload))
	return nil
}
func (q *Queue) Pop() (Record, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.records) == 0 {
		return Record{}, false
	}
	r := q.records[0]
	copy(q.records, q.records[1:])
	q.records[len(q.records)-1] = Record{}
	q.records = q.records[:len(q.records)-1]
	q.bytes -= uint32(len(r.Payload))
	return r, true
}
func (q *Queue) Dropped() uint64 { q.mu.Lock(); defer q.mu.Unlock(); return q.dropped }
func (q *Queue) Close()          { q.mu.Lock(); q.closed = true; q.mu.Unlock() }
