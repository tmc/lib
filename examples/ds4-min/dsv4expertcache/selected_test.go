package dsv4expertcache

import (
	"bytes"
	"errors"
	"fmt"
	"reflect"
	"testing"
)

type selectedStager struct {
	stage    []byte
	transfer *transfer
}

func (u selectedStager) Stage(s *Source, l *Lease, graph bool) error {
	return s.Stage(l, u.stage, u.transfer, graph)
}

func TestSelectConsumerRows(t *testing.T) {
	s := source(t)
	c := cache(t)
	tr := &transfer{}
	for _, ids := range [][]int{{1, 0, 1}, {0, 1, 0}} {
		lease, err := s.Select(c, 0, ids, selectedStager{make([]byte, 2), tr}, false)
		if err != nil {
			t.Fatal(err)
		}
		// Fake consumer reads exactly the uploaded slot bytes, independently
		// compared with the immutable reader's original expert-axis order.
		slabs := make(map[[2]int]string)
		for _, r := range tr.outputs {
			slabs[[2]int{r.slot, r.component}] = r.bytes
		}
		for i, slot := range lease.Slots() {
			for component := 0; component < 3; component++ {
				want := make([]byte, 2)
				if err := s.Read(key(ids[i]), component, want); err != nil {
					t.Fatal(err)
				}
				if slabs[[2]int{slot, component}] != string(want) {
					t.Fatalf("consumer slot=%d component=%d got %q want %q", slot, component, slabs[[2]int{slot, component}], want)
				}
			}
		}
		if _, err := s.Select(c, 0, []int{0}, selectedStager{make([]byte, 2), tr}, false); err == nil {
			t.Fatal("consumer lease did not protect slots")
		}
		if err := lease.Release(&completion{err: errors.New("consumer pending")}); err == nil {
			t.Fatal("failed consumer wait released slots")
		}
		if err := lease.Release(&completion{}); err != nil {
			t.Fatal(err)
		}
	}
	if tr.calls != 6 {
		t.Fatalf("cache hits copied again: %d calls", tr.calls)
	}
}

func TestSelectFailures(t *testing.T) {
	for _, name := range []string{"graph", "bad-id", "short-stage", "short-read", "submit", "wait", "missing-completion"} {
		t.Run(name, func(t *testing.T) {
			s, c, tr := source(t), cache(t), &transfer{}
			ids, stage, graph := []int{0}, make([]byte, 2), false
			switch name {
			case "graph":
				graph = true
			case "bad-id":
				ids = []int{2}
			case "short-stage":
				stage = stage[:1]
			case "short-read":
				s.reader = bytes.NewReader(nil)
			case "submit":
				tr.fail = true
			case "wait":
				tr.failedWait = true
			case "missing-completion":
				tr.missing = true
			}
			lease, err := s.Select(c, 0, ids, selectedStager{stage, tr}, graph)
			if err == nil {
				t.Fatal("fault accepted")
			}
			poison := name == "wait" || name == "missing-completion"
			if (lease != nil) != poison || c.closed != poison {
				t.Fatalf("retained=%v closed=%v poison=%v", lease != nil, c.closed, poison)
			}
			if c.active != nil {
				t.Fatal("failed selection retained active planner lease")
			}
			if name == "graph" && tr.calls != 0 {
				t.Fatal("graph rejection submitted uploads")
			}
		})
	}
}

func TestSelectLayerIdentity(t *testing.T) {
	s, c := source(t), cache(t)
	s.layers[1] = s.layers[0]
	lease, err := s.Select(c, 1, []int{1, 1}, selectedStager{make([]byte, 2), &transfer{}}, false)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(lease.Uploads()[0].Key, Key{identity(), 1, 1}) {
		t.Fatal("selection lost layer identity")
	}
	if err := lease.Release(&completion{}); err != nil {
		t.Fatal(err)
	}
}

func ExampleSource_Select() {
	id := Identity{Model: [32]byte{1}, Quant: [32]byte{2}, Layout: [32]byte{3}}
	tr := Triplet{Tensor{0, 4, 2, 2}, Tensor{4, 4, 2, 2}, Tensor{8, 4, 2, 2}}
	s, _ := NewSource(bytes.NewReader([]byte("abcdefghijkl")), 12, id, map[int]Triplet{0: tr})
	c, _ := New(id, 2, 6, 12)
	lease, _ := s.Select(c, 0, []int{1, 0, 1}, selectedStager{make([]byte, 2), &transfer{}}, false)
	fmt.Println(lease.Slots())
	_ = lease.Release(&completion{})
	// Output: [0 1 0]
}

type uncommittedSelectedStager struct{ failAbort bool }

func (s *uncommittedSelectedStager) Stage(_ *Source, l *Lease, _ bool) error {
	if s.failAbort {
		return l.Started(l.Uploads()[0], &completion{err: errors.New("uncertain upload")})
	}
	return nil
}
func TestSelectUncommittedRetainsOwner(t *testing.T) {
	for _, failed := range []bool{false, true} {
		s, c := source(t), cache(t)
		l, err := s.Select(c, 0, []int{0}, &uncommittedSelectedStager{failed}, false)
		if err == nil || (l != nil) != failed || c.closed != failed {
			t.Fatalf("uncommitted cleanup lost owner: retained=%v poisoned=%v err=%v", l != nil, c.closed, err)
		}
	}
}
func TestSelectTypedNilStager(t *testing.T) {
	var stager *uncommittedSelectedStager
	if l, err := source(t).Select(cache(t), 0, []int{0}, stager, false); err == nil || l != nil {
		t.Fatal("typed nil stager accepted")
	}
}

type committedErrorStager struct{}

func (committedErrorStager) Stage(_ *Source, l *Lease, _ bool) error {
	for _, u := range l.Uploads() {
		if err := l.Started(u, &completion{}); err != nil {
			return err
		}
	}
	if err := l.Commit(); err != nil {
		return err
	}
	return errors.New("after commit")
}
func TestSelectCommittedErrorRetainsLease(t *testing.T) {
	s, c := source(t), cache(t)
	l, err := s.Select(c, 0, []int{0}, committedErrorStager{}, false)
	if err == nil || l == nil || !l.committed || c.active != l {
		t.Fatal("committed error lease lost")
	}
	if err := l.Release(&completion{}); err != nil {
		t.Fatal(err)
	}
}
