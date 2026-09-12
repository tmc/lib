package wire

import (
	"bytes"
	"errors"
	"io"
	"testing"
)

func testRecord(session byte, process, seq uint64) Record {
	var s [16]byte
	s[0] = session
	return Record{Kind: 3, Identity: Identity{Session: s, Process: process, Generation: 1}, Sequence: seq, Dropped: seq - 1, Timestamp: 99, Payload: []byte("cuda")}
}

func TestRoundTripAndAccounting(t *testing.T) {
	var b bytes.Buffer
	for i := uint64(1); i <= 3; i++ {
		if err := Write(&b, testRecord(1, 7, i), Limits{}); err != nil {
			t.Fatal(err)
		}
	}
	d := NewDecoder(&b, Limits{})
	var s [16]byte
	s[0] = 1
	d.ExpectSession(s)
	for i := 0; i < 3; i++ {
		r, err := d.Next()
		if err != nil {
			t.Fatal(err)
		}
		if r.Sequence != uint64(i+1) {
			t.Fatalf("seq=%d", r.Sequence)
		}
	}
	if got := d.Stats(); got.Records != 3 || got.Bytes != 12 || got.Gaps != 0 || got.Dropped != 2 {
		t.Fatalf("stats=%+v", got)
	}
}
func TestRejectsMalformedAndIdentity(t *testing.T) {
	b, _ := testRecord(1, 1, 1).MarshalBinary(Limits{})
	b[0] = 'x'
	if _, e := NewDecoder(bytes.NewReader(b), Limits{}).Next(); !errors.Is(e, ErrBadMagic) {
		t.Fatalf("magic: %v", e)
	}
	b, _ = testRecord(1, 1, 1).MarshalBinary(Limits{})
	b[12] = 2
	var s [16]byte
	s[0] = 1
	d := NewDecoder(bytes.NewReader(b), Limits{})
	d.ExpectSession(s)
	if _, e := d.Next(); !errors.Is(e, ErrMalformed) {
		t.Fatalf("checksum: %v", e)
	}
	b, _ = testRecord(2, 1, 1).MarshalBinary(Limits{})
	d = NewDecoder(bytes.NewReader(b), Limits{})
	d.ExpectSession(s)
	if _, e := d.Next(); !errors.Is(e, ErrSession) {
		t.Fatalf("session: %v", e)
	}
}
func TestDuplicateGapTruncationAndBounds(t *testing.T) {
	var b bytes.Buffer
	Write(&b, testRecord(1, 1, 1), Limits{})
	Write(&b, testRecord(1, 1, 3), Limits{})
	d := NewDecoder(&b, Limits{})
	d.Next()
	d.Next()
	if d.Stats().Gaps != 1 {
		t.Fatal(d.Stats())
	}
	var dup bytes.Buffer
	Write(&dup, testRecord(1, 1, 1), Limits{})
	Write(&dup, testRecord(1, 1, 1), Limits{})
	d = NewDecoder(&dup, Limits{})
	if _, e := d.Next(); e != nil {
		t.Fatal(e)
	}
	if _, e := d.Next(); !errors.Is(e, ErrDuplicate) {
		t.Fatalf("duplicate: %v", e)
	}
	b.Reset()
	Write(&b, testRecord(1, 1, 1), Limits{})
	raw := b.Bytes()
	d = NewDecoder(bytes.NewReader(raw[:len(raw)-1]), Limits{})
	d.Next()
	if _, e := d.Next(); !errors.Is(e, io.EOF) && !errors.Is(e, io.ErrUnexpectedEOF) {
		t.Fatalf("truncation: %v", e)
	}
	if _, e := testRecord(1, 1, 1).MarshalBinary(Limits{MaxPayload: 1}); !errors.Is(e, ErrOversize) {
		t.Fatalf("bound: %v", e)
	}
}
func TestQueueOverflowAndCopy(t *testing.T) {
	q := NewQueue(Limits{QueueRecords: 1, QueueBytes: 4})
	r := testRecord(1, 1, 1)
	if e := q.Push(r); e != nil {
		t.Fatal(e)
	}
	r.Payload[0] = 'x'
	if e := q.Push(testRecord(1, 1, 2)); !errors.Is(e, ErrQueueFull) {
		t.Fatal(e)
	}
	got, ok := q.Pop()
	if !ok || string(got.Payload) != "cuda" || q.Dropped() != 1 {
		t.Fatalf("%q %v %d", got.Payload, ok, q.Dropped())
	}
	q.Close()
	if e := q.Push(testRecord(1, 1, 3)); !errors.Is(e, io.ErrClosedPipe) {
		t.Fatalf("closed queue: %v", e)
	}
}

func TestPerStreamDropsAndRecordLimit(t *testing.T) {
	var b bytes.Buffer
	Write(&b, testRecord(1, 1, 1), Limits{})
	Write(&b, testRecord(2, 2, 1), Limits{})
	r := testRecord(1, 1, 2)
	r.Dropped = 3
	Write(&b, r, Limits{})
	d := NewDecoder(&b, Limits{MaxRecords: 3})
	for i := 0; i < 3; i++ {
		if _, e := d.Next(); e != nil {
			t.Fatal(e)
		}
	}
	if d.Stats().Dropped != 3 {
		t.Fatalf("drops=%d", d.Stats().Dropped)
	}
	var one bytes.Buffer
	Write(&one, testRecord(1, 1, 1), Limits{})
	Write(&one, testRecord(1, 1, 2), Limits{})
	d = NewDecoder(&one, Limits{MaxRecords: 1})
	if _, e := d.Next(); e != nil {
		t.Fatal(e)
	}
	if _, e := d.Next(); !errors.Is(e, ErrRecordLimit) {
		t.Fatalf("limit=%v", e)
	}
}

func FuzzDecoderNeverPanics(f *testing.F) {
	b, _ := testRecord(1, 1, 1).MarshalBinary(Limits{MaxPayload: 64})
	f.Add(b)
	f.Fuzz(func(t *testing.T, in []byte) {
		d := NewDecoder(bytes.NewReader(in), Limits{MaxPayload: 64, MaxFrame: HeaderSize + 64})
		_, _ = d.Next()
	})
}
