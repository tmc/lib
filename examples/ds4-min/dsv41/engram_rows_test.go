package dsv41

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"reflect"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv41engram"
)

type engramRowsReader struct {
	raw   []byte
	reads int
	fail  bool
}

func (r *engramRowsReader) ReadAt(dst []byte, off int64) (int, error) {
	r.reads++
	if r.fail {
		return 0, io.ErrUnexpectedEOF
	}
	return bytes.NewReader(r.raw).ReadAt(dst, off)
}

func engramRowsFixture(t *testing.T) (*EngramRows, [2]*engramRowsReader) {
	t.Helper()
	spec := dsv41engram.LayoutSpec{TokenMap: []uint32{0, 1, 2, 3}, CompressedVocabSize: 4, PadID: 0, Rows: [2]uint32{120, 120}}
	for i := range spec.Primes {
		for j := range spec.Primes[i] {
			spec.Primes[i][j] = 5
		}
		spec.Multipliers[i] = [4]uint64{1, 3, 5, 7}
	}
	layout, err := dsv41engram.NewLayout(spec)
	if err != nil {
		t.Fatal(err)
	}
	var readers [2]*engramRowsReader
	var tables [2]*dsv41engram.Table
	for i := range readers {
		raw := make([]byte, 120*dsv41engram.RowBytes)
		for row := 0; row < 120; row++ {
			for j := 0; j < 256; j++ {
				raw[row*264+j] = byte(0x20 + (row+i)%16)
			}
			for j := 256; j < 264; j++ {
				raw[row*264+j] = 127
			}
		}
		readers[i] = &engramRowsReader{raw: raw}
		tables[i], err = dsv41engram.NewTable(readers[i], int64(len(raw)), 0, 120)
		if err != nil {
			t.Fatal(err)
		}
	}
	s, err := NewEngramRows(layout, tables, 16)
	if err != nil {
		t.Fatal(err)
	}
	return s, readers
}

func TestEngramRowsTransaction(t *testing.T) {
	s, r := engramRowsFixture(t)
	defer s.Close()
	before := s.history
	if err := s.Prepare(1, true, 0); err != nil {
		t.Fatal(err)
	}
	if s.Position() != 0 || s.history != before {
		t.Fatal("prepared history was published before token commit")
	}
	first, text, err := s.Rows(1)
	if err != nil || !text || len(first) != 6144 {
		t.Fatal("prepared rows", err)
	}
	first[0] = 999
	again, _, _ := s.Rows(1)
	if again[0] == 999 {
		t.Fatal("rows alias mutable staging")
	}
	if err := s.Prepare(2, true, 0); err == nil {
		t.Fatal("overwrote pending token")
	}
	s.Abort()
	if s.Position() != 0 || s.history != before {
		t.Fatal("aborted device token published history")
	}
	if _, _, err := s.Rows(1); err == nil {
		t.Fatal("aborted row remained visible")
	}
	if err := s.Commit(); err == nil {
		t.Fatal("aborted token committed")
	}
	r[1].fail = true
	if err := s.Prepare(2, true, 0); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatal("failed sparse read cause lost", err)
	}
	if s.Position() != 0 || s.history != before || s.prepared {
		t.Fatal("failed second read changed frontier")
	}
	r[1].fail = false
	if err := s.Prepare(1, true, 0); err != nil {
		t.Fatal(err)
	}
	if err := s.Commit(); err != nil {
		t.Fatal(err)
	}
	committed := s.history
	if s.Position() != 1 || committed == before {
		t.Fatal("successful token not committed")
	}
	if err := s.Commit(); err == nil || s.Position() != 1 || s.history != committed {
		t.Fatal("double commit advanced history")
	}
	if err := s.Prepare(1, true, 0); err == nil {
		t.Fatal("stale position admitted")
	}
	if err := s.Reset(); err != nil || s.Position() != 0 {
		t.Fatal("reset", err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s.Reset(); err == nil {
		t.Fatal("closed reset admitted")
	}
	if err := s.Prepare(0, true, 0); err == nil {
		t.Fatal("closed prepare admitted")
	}
}

func TestEngramRowsMaskedHistory(t *testing.T) {
	s, readers := engramRowsFixture(t)
	defer s.Close()
	var want dsv41engram.History
	for position, token := range []int32{1, 2, 3, 1, 2} {
		text := position != 1
		mask := byte(0)
		if text {
			mask = 1
		}
		var ids [48]uint32
		if err := s.layout.Hash(&want, []int32{token}, []byte{mask}, ids[:]); err != nil {
			t.Fatal(err)
		}
		reads := [2]int{readers[0].reads, readers[1].reads}
		if err := s.Prepare(token, text, position); err != nil {
			t.Fatal(err)
		}
		for i, layer := range []int{1, 14} {
			got, enabled, err := s.Rows(layer)
			if err != nil || enabled != text {
				t.Fatal("masked rows", err)
			}
			if !text {
				if len(got) != 0 || readers[i].reads != reads[i] {
					t.Fatal("image position read sparse rows")
				}
				continue
			}
			ref := make([]float32, 6144)
			if err := s.tables[i].ReadRows(ids[i*24:(i+1)*24], ref); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, ref) {
				t.Fatalf("row ordering at position %d table%d", position, i)
			}
		}
		if err := s.Commit(); err != nil {
			t.Fatal(err)
		}
		if s.history != want || s.Position() != position+1 {
			t.Fatal("masked DEAD did not advance committed history")
		}
	}
}

func TestEngramRowsAdmission(t *testing.T) {
	s, _ := engramRowsFixture(t)
	defer s.Close()
	for _, layer := range []int{-1, 0, 2, 13, 15, 40} {
		if _, _, err := s.Rows(layer); err == nil {
			t.Fatal("unprepared rows admitted")
		}
	}
	if err := s.Prepare(-1, true, 0); err == nil {
		t.Fatal("invalid token admitted")
	}
	if err := s.Prepare(0, true, -1); err == nil {
		t.Fatal("invalid frontier admitted")
	}
	if err := s.Prepare(0, true, 0); err != nil {
		t.Fatal(err)
	}
	if _, _, err := s.Rows(2); err == nil {
		t.Fatal("noninsertion layer admitted")
	}
	if _, err := NewEngramRows(nil, s.tables, 1); err == nil {
		t.Fatal("nil layout admitted")
	}
	if _, err := (*Model)(nil).NewEngramRows(1); err == nil {
		t.Fatal("nil model admitted")
	}
	s.Abort()
	s.tables[0] = nil
	if err := s.Prepare(0, true, 0); err == nil {
		t.Fatal("unreadable table admitted")
	}
}

func ExampleEngramRows_Position() {
	var s *EngramRows
	fmt.Println(s.Position())
	// Output: 0
}
