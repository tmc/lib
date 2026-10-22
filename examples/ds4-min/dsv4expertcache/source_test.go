package dsv4expertcache

import (
	"bytes"
	"errors"
	"fmt"
	"reflect"
	"testing"
)

func source(t *testing.T) *Source {
	t.Helper()
	tr := Triplet{Tensor{0, 4, 2, 2}, Tensor{4, 4, 2, 2}, Tensor{8, 4, 2, 2}}
	s, e := NewSource(bytes.NewReader([]byte("abcdefghijkl")), 12, identity(), map[int]Triplet{0: tr})
	if e != nil {
		t.Fatal(e)
	}
	return s
}
func TestSource(t *testing.T) {
	s := source(t)
	buf := make([]byte, 2)
	if e := s.Read(key(1), 1, buf); e != nil || string(buf) != "gh" {
		t.Fatalf("read %q %v", buf, e)
	}
	bad := key(1)
	bad.Identity.Quant[0]++
	if s.Read(bad, 0, buf) == nil {
		t.Fatal("wrong identity")
	}
	s.reader = bytes.NewReader([]byte("a"))
	if s.Read(key(0), 0, buf) == nil {
		t.Fatal("short read")
	}
	tr := s.layers[0]
	tr.Gate.Stride = 3
	if _, e := NewSource(bytes.NewReader(nil), 12, identity(), map[int]Triplet{0: tr}); e == nil {
		t.Fatal("wrong stride")
	}
	tr.Gate = Tensor{^uint64(0), 4, 2, 2}
	if _, e := NewSource(bytes.NewReader(nil), 12, identity(), map[int]Triplet{0: tr}); e == nil {
		t.Fatal("overflow")
	}
}

type transfer struct {
	pending    bool
	calls      int
	fail       bool
	missing    bool
	failedWait bool
	outputs    []record
}
type record struct {
	slot, component int
	bytes           string
}
type pendingEvent struct {
	t         *transfer
	upload    Upload
	component int
	src       []byte
	complete  bool
}

func (e *pendingEvent) Wait() error {
	if e.t.failedWait {
		return errors.New("event failure")
	}
	if !e.complete {
		e.t.outputs = append(e.t.outputs, record{e.upload.Slot, e.component, string(e.src)})
		e.complete = true
		e.t.pending = false
	}
	return nil
}
func (t *transfer) Submit(u Upload, component int, src []byte) (Completion, error) {
	if t.pending {
		return nil, errors.New("stage reused while pending")
	}
	t.pending = true
	t.calls++
	if t.missing {
		return nil, nil
	}
	if t.fail {
		return &pendingEvent{t: t, upload: u, component: component, src: src}, errors.New("partial submit")
	}
	return &pendingEvent{t: t, upload: u, component: component, src: src}, nil
}
func TestStageBytes(t *testing.T) {
	s := source(t)
	c := cache(t)
	l, _ := c.Acquire([]Key{key(1), key(0), key(1)})
	tr := &transfer{}
	if e := s.Stage(l, make([]byte, 2), tr, false); e != nil {
		t.Fatal(e)
	}
	want := []record{{0, 0, "cd"}, {0, 1, "gh"}, {0, 2, "kl"}, {1, 0, "ab"}, {1, 1, "ef"}, {1, 2, "ij"}}
	if !reflect.DeepEqual(tr.outputs, want) {
		t.Fatalf("outputs %v want %v", tr.outputs, want)
	}
	_ = l.Release(&completion{})
}
func TestStagePreflight(t *testing.T) {
	s := source(t)
	c := cache(t)
	load(t, c, key(0))
	l, _ := c.Acquire([]Key{key(0)})
	var nilTransfer *transfer
	for _, args := range []struct {
		stage    []byte
		transfer Transfer
		graph    bool
	}{{make([]byte, 2), nilTransfer, false}, {make([]byte, 2), &transfer{}, true}, {nil, &transfer{}, false}} {
		if s.Stage(l, args.stage, args.transfer, args.graph) == nil {
			t.Fatal("preflight passed")
		}
		if c.active != l {
			t.Fatal("preflight lost lease")
		}
	}
	s.identity.Model[0]++
	if s.Stage(l, make([]byte, 2), &transfer{}, false) == nil {
		t.Fatal("all-hit wrong identity accepted")
	}
	_ = l.Abort()
	if _, e := c.Acquire([]Key{key(0)}); e != nil {
		t.Fatal(e)
	}
	var reader *bytes.Reader
	if _, e := NewSource(reader, 12, identity(), s.layers); e == nil {
		t.Fatal("typed nil reader")
	}
}
func TestStageFailedWait(t *testing.T) {
	s := source(t)
	c := cache(t)
	l, _ := c.Acquire([]Key{key(0)})
	if s.Stage(l, make([]byte, 2), &transfer{failedWait: true}, false) == nil {
		t.Fatal("ignored failed event")
	}
	if _, e := c.Acquire([]Key{key(1)}); e == nil {
		t.Fatal("failed event reused")
	}
}
func ExampleSource_Stage() {
	tr := Triplet{Tensor{0, 4, 2, 2}, Tensor{4, 4, 2, 2}, Tensor{8, 4, 2, 2}}
	s, _ := NewSource(bytes.NewReader([]byte("abcdefghijkl")), 12, identity(), map[int]Triplet{0: tr})
	c, _ := New(identity(), 2, 6, 12)
	l, _ := c.Acquire([]Key{key(0)})
	adapter := &transfer{}
	_ = s.Stage(l, make([]byte, 2), adapter, false)
	fmt.Println(adapter.outputs)
	_ = l.Release(&completion{})
	_ = c.Close()
	// Output: [{0 0 ab} {0 1 ef} {0 2 ij}]
}

func ExampleNewSource() {
	tr := Triplet{Tensor{0, 4, 2, 2}, Tensor{4, 4, 2, 2}, Tensor{8, 4, 2, 2}}
	s, _ := NewSource(bytes.NewReader([]byte("abcdefghijkl")), 12, identity(), map[int]Triplet{0: tr})
	buf := make([]byte, 2)
	_ = s.Read(key(1), 2, buf)
	fmt.Println(string(buf))
	// Output: kl
}
func TestStage(t *testing.T) {
	s := source(t)
	c := cache(t)
	l, _ := c.Acquire([]Key{key(0)})
	tr := &transfer{}
	if e := s.Stage(l, make([]byte, 2), tr, false); e != nil {
		t.Fatal(e)
	}
	if tr.calls != 3 || tr.pending {
		t.Fatal("incomplete triplet")
	}
	_ = l.Release(&completion{})
	l, _ = c.Acquire([]Key{key(1)})
	tr.fail = true
	if s.Stage(l, make([]byte, 2), tr, false) == nil || tr.pending {
		t.Fatal("partial submission not drained")
	}
	l, _ = c.Acquire([]Key{key(1)})
	tr.fail = false
	tr.missing = true
	if s.Stage(l, make([]byte, 2), tr, false) == nil {
		t.Fatal("missing event accepted")
	}
	if _, e := c.Acquire([]Key{key(0)}); e == nil {
		t.Fatal("uncertain writes reused")
	}
}
