package cuptiactivity

import (
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestSessionSnapshot(t *testing.T) {
	p, err := newBufferPool(64, 1)
	if err != nil {
		t.Fatal(err)
	}
	c := &collection{pool: p, records: []Record{{Name: "original"}}, nativeDropped: 2, incomplete: 3}
	o := &activityOwner{pool: p, disable: func(cupti.ActivityKind) error { return nil }, flush: func() error { return nil }}
	s := &Session{owner: o, collection: c}
	r := s.Result()
	if r.Closed || r.DroppedRecords != 2 || r.IncompleteRecords != 3 {
		t.Fatalf("snapshot = %+v", r)
	}
	r.Records[0].Name = "changed"
	if s.Result().Records[0].Name != "original" {
		t.Fatal("snapshot aliases retained records")
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if !s.Result().Closed {
		t.Fatal("snapshot did not record close")
	}
}

func TestSessionZero(t *testing.T) {
	var s Session
	if s.Close() == nil || s.Result().FirstError == nil {
		t.Fatal("zero session accepted")
	}
	var nilSession *Session
	if err := nilSession.Close(); err != nil {
		t.Fatal(err)
	}
	if nilSession.Result().FirstError == nil {
		t.Fatal("nil session yielded clean result")
	}
}
