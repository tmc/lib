package dsv41engram

import (
	"bytes"
	"errors"
	"io"
	"math"
	"reflect"
	"testing"
)

func testSpec() LayoutSpec {
	s := LayoutSpec{TokenMap: make([]uint32, 257), CompressedVocabSize: 101, PadID: 100}
	for i := range s.TokenMap {
		s.TokenMap[i] = uint32(i * 37 % 101)
	}
	for l := range Layers {
		for j := range NGram {
			s.Multipliers[l][j] = uint64(1000000007+l*100+j*18) | 1
		}
		for j := range Columns {
			s.Primes[l][j] = uint32(101 + l*10 + j*2)
			s.Rows[l] += s.Primes[l][j]
		}
	}
	return s
}

func TestHash(t *testing.T) {
	s := testSpec()
	l, err := NewLayout(s)
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int32{1, 2, 3, 4, 5, 6, 7}
	mask := []byte{1, 1, 0, 1, 1, 1, 1}
	a, b := make([]uint32, len(tokens)*Layers*Columns), make([]uint32, len(tokens)*Layers*Columns)
	var whole, split History
	if err := l.Hash(&whole, tokens, mask, a); err != nil {
		t.Fatal(err)
	}
	for i := range tokens {
		if err := l.Hash(&split, tokens[i:i+1], mask[i:i+1], b[i*Layers*Columns:(i+1)*Layers*Columns]); err != nil {
			t.Fatal(err)
		}
	}
	if !reflect.DeepEqual(a, b) || whole != split {
		t.Fatal("split changed hashes or history")
	}
	s.TokenMap[1] = 1000
	var again History
	c := make([]uint32, len(a))
	if err := l.Hash(&again, tokens, mask, c); err != nil || !reflect.DeepEqual(a, c) {
		t.Fatal("layout retained caller map")
	}
	before := whole
	keep := append([]uint32(nil), a...)
	if err := l.Hash(&whole, []int32{1, -1}, nil, a[:2*Layers*Columns]); err == nil || whole != before || !reflect.DeepEqual(a, keep) {
		t.Fatal("invalid tokens mutated state")
	}
}

func TestLayoutInvalid(t *testing.T) {
	for _, mutate := range []func(*LayoutSpec){func(s *LayoutSpec) { s.PadID = 101 }, func(s *LayoutSpec) { s.TokenMap[0] = 101 }, func(s *LayoutSpec) { s.Multipliers[0][0] = 2 }, func(s *LayoutSpec) { s.Multipliers[0][0] = math.MaxUint64 }, func(s *LayoutSpec) { s.Primes[0][0] = 1 }, func(s *LayoutSpec) { s.Rows[0]++ }} {
		s := testSpec()
		mutate(&s)
		if _, err := NewLayout(s); err == nil {
			t.Fatal("invalid layout accepted")
		}
	}
}

type countingReader struct {
	*bytes.Reader
	calls int
}

func (r *countingReader) ReadAt(p []byte, off int64) (int, error) {
	r.calls++
	return r.Reader.ReadAt(p, off)
}

func TestRows(t *testing.T) {
	raw := make([]byte, 3*RowBytes)
	for r := range 3 {
		for j := range Dim {
			raw[r*RowBytes+j] = byte((j + r) % 126)
		}
		for j := range 8 {
			raw[r*RowBytes+Dim+j] = 127
		}
	}
	r := &countingReader{Reader: bytes.NewReader(raw)}
	table, err := NewTable(r, int64(len(raw)), 0, 3)
	if err != nil {
		t.Fatal(err)
	}
	ids := []uint32{2, 0, 2}
	got := make([]float32, 3*Dim)
	if err := table.ReadRows(ids, got); err != nil {
		t.Fatal(err)
	}
	encoded := make([]byte, 3*RowBytes)
	if err := table.ReadEncodedRows(ids, encoded); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(encoded[:RowBytes], raw[2*RowBytes:]) {
		t.Fatal("encoded rows reordered")
	}
	for j := range Dim {
		if math.Float32bits(got[j]) != math.Float32bits(got[2*Dim+j]) {
			t.Fatal("duplicate differs")
		}
	}
	batch := make([]uint32, 2*30)
	for i := range batch {
		batch[i] = uint32(i % 3)
	}
	out := make([]float32, 2*Columns*Dim)
	r.calls = 0
	if err := table.ReadBatch(batch, 2, 30, out); err != nil {
		t.Fatal(err)
	}
	if r.calls != 3 {
		t.Fatalf("read count %d, want 3", r.calls)
	}
	for i := range 2 {
		for j := range Columns {
			want := make([]float32, Dim)
			_ = DecodeRow(raw[int(batch[i*30+j])*RowBytes:][:RowBytes], want)
			if !reflect.DeepEqual(want, out[(i*Columns+j)*Dim:][:Dim]) {
				t.Fatal("batch order differs")
			}
		}
	}
	before := append([]float32(nil), out...)
	if err := table.ReadRows([]uint32{3}, out[:Dim]); err == nil || !reflect.DeepEqual(before, out) {
		t.Fatal("out of range touched output")
	}
}

type shortReader struct{}

func (shortReader) ReadAt(p []byte, off int64) (int, error) { return len(p) - 1, nil }
func TestReadErrors(t *testing.T) {
	for _, x := range []struct {
		size, offset int64
		rows         uint32
	}{{263, 0, 1}, {264, -1, 1}, {264, 1, 1}, {264, 0, 0}} {
		if _, err := NewTable(bytes.NewReader(nil), x.size, x.offset, x.rows); err == nil {
			t.Fatal("invalid bounds accepted")
		}
	}
	table, _ := NewTable(shortReader{}, RowBytes, 0, 1)
	if err := table.ReadRows([]uint32{0}, make([]float32, Dim)); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("short read: %v", err)
	}
	for _, pair := range [][2]byte{{127, 127}, {255, 127}, {1, 255}, {126, 254}} {
		raw := make([]byte, RowBytes)
		raw[0] = pair[0]
		for i := range 8 {
			raw[Dim+i] = pair[1]
		}
		if err := DecodeRow(raw, make([]float32, Dim)); err == nil {
			t.Fatal("invalid code accepted")
		}
	}
}

func TestBatchBoundedChunks(t *testing.T) {
	raw := make([]byte, RowBytes)
	r := &countingReader{Reader: bytes.NewReader(raw)}
	table, _ := NewTable(r, RowBytes, 0, 1)
	const tokens = 2049
	if err := table.ReadBatch(make([]uint32, tokens*Columns), tokens, Columns, make([]float32, tokens*Columns*Dim)); err != nil {
		t.Fatal(err)
	}
	if r.calls != 2 {
		t.Fatalf("read count %d, want one per bounded chunk", r.calls)
	}
	if err := table.ReadBatch(nil, 0, 0, nil); err != nil {
		t.Fatal(err)
	}
	for _, x := range []struct {
		tokens, stride int
		rows           []uint32
		dst            []float32
	}{
		{-1, 24, nil, nil}, {1, 23, make([]uint32, 24), make([]float32, 24*Dim)},
		{2, math.MaxInt, nil, make([]float32, 48*Dim)}, {1, 24, nil, make([]float32, 24*Dim)},
	} {
		if err := table.ReadBatch(x.rows, x.tokens, x.stride, x.dst); err == nil {
			t.Fatal("invalid batch accepted")
		}
	}
}

func BenchmarkHash(b *testing.B) {
	l, _ := NewLayout(testSpec())
	tokens := make([]int32, 2048)
	rows := make([]uint32, len(tokens)*Layers*Columns)
	var h History
	b.ReportAllocs()
	for b.Loop() {
		h.Reset()
		if err := l.Hash(&h, tokens, nil, rows); err != nil {
			b.Fatal(err)
		}
	}
}
