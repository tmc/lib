package dsv4expertcache

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"
)

type completion struct {
	err   error
	waits int
}

func (e *completion) Wait() error { e.waits++; return e.err }
func identity() Identity          { var i Identity; i.Model[0] = 1; i.Quant[0] = 2; i.Layout[0] = 3; return i }
func key(n int) Key               { return Key{identity(), 0, n} }
func cache(t *testing.T) *Cache {
	t.Helper()
	c, e := New(identity(), 2, 10, 20)
	if e != nil {
		t.Fatal(e)
	}
	return c
}
func load(t *testing.T, c *Cache, keys ...Key) []int {
	t.Helper()
	l, e := c.Acquire(keys)
	if e != nil {
		t.Fatal(e)
	}
	for _, u := range l.Uploads() {
		if e = l.Started(u, &completion{}); e != nil {
			t.Fatal(e)
		}
	}
	if e = l.Commit(); e != nil {
		t.Fatal(e)
	}
	slots := l.Slots()
	if e = l.Release(&completion{}); e != nil {
		t.Fatal(e)
	}
	return slots
}
func TestProtectedHitAndDuplicates(t *testing.T) {
	c := cache(t)
	old := load(t, c, key(0), key(1))
	got := load(t, c, key(2), key(0), key(2))
	if !reflect.DeepEqual(got, []int{old[1], old[0], old[1]}) {
		t.Fatalf("remap %v", got)
	}
}
func TestIncompleteAndAbort(t *testing.T) {
	c := cache(t)
	load(t, c, key(0), key(1))
	l, _ := c.Acquire([]Key{key(2), key(1)})
	if l.Commit() == nil {
		t.Fatal("published incomplete upload")
	}
	if e := l.Abort(); e != nil {
		t.Fatal(e)
	}
	l, _ = c.Acquire([]Key{key(0), key(1)})
	if len(l.Uploads()) != 0 {
		t.Fatal("untouched victims lost")
	}
	_ = l.Abort()
	l, _ = c.Acquire([]Key{key(2), key(1)})
	u := l.Uploads()[0]
	event := &completion{}
	if e := l.Started(u, event); e != nil {
		t.Fatal(e)
	}
	if e := l.Abort(); e != nil {
		t.Fatal(e)
	}
	l, _ = c.Acquire([]Key{key(0), key(1)})
	if len(l.Uploads()) != 1 {
		t.Fatal("overwritten victim retained")
	}
	_ = l.Abort()
}
func TestAdmissionAndStale(t *testing.T) {
	c := cache(t)
	bad := key(0)
	bad.Identity.Model[0]++
	for _, keys := range [][]Key{nil, {bad}, {key(0), key(1), key(2)}} {
		if _, e := c.Acquire(keys); e == nil {
			t.Fatal("accepted invalid request")
		}
	}
	l, _ := c.Acquire([]Key{key(0)})
	u := l.Uploads()[0]
	u.Generation++
	if l.Started(u, &completion{}) == nil {
		t.Fatal("accepted stale generation")
	}
	_ = l.Abort()
	if l.Commit() == nil {
		t.Fatal("accepted stale lease")
	}
	c.generation = math.MaxUint64
	if _, e := c.Acquire([]Key{key(0)}); e == nil {
		t.Fatal("generation wrapped")
	}
	c.generation = 0
	_ = c.Close()
	if _, e := c.Acquire([]Key{key(0)}); e == nil {
		t.Fatal("acquire after close")
	}
	if _, e := New(identity(), 2, math.MaxUint64, math.MaxUint64); e == nil {
		t.Fatal("overflow budget accepted")
	}
}
func TestCompletionFailure(t *testing.T) {
	c := cache(t)
	l, _ := c.Acquire([]Key{key(0)})
	event := &completion{err: errors.New("device failure")}
	_ = l.Started(l.Uploads()[0], event)
	if l.Commit() == nil {
		t.Fatal("ignored upload failure")
	}
	if l.Abort() == nil {
		t.Fatal("ignored abort wait failure")
	}
	if _, e := c.Acquire([]Key{key(0)}); e == nil {
		t.Fatal("reused uncertain device slot")
	}
}
func TestNilAndConsumerFailure(t *testing.T) {
	var nilLease *Lease
	if nilLease.Slots() != nil || nilLease.Uploads() != nil {
		t.Fatal("nil lease slices")
	}
	c := cache(t)
	l, _ := c.Acquire([]Key{key(0)})
	var nilEvent *completion
	if l.Started(l.Uploads()[0], nilEvent) == nil {
		t.Fatal("accepted typed nil")
	}
	_ = l.Started(l.Uploads()[0], &completion{})
	_ = l.Commit()
	if l.Release(&completion{err: errors.New("consumer busy")}) == nil {
		t.Fatal("ignored consumer failure")
	}
	if _, e := c.Acquire([]Key{key(1)}); e == nil {
		t.Fatal("reused active consumer slot")
	}
	if l.Release(nilEvent) == nil {
		t.Fatal("accepted typed nil consumer")
	}
	_ = l.Release(&completion{})
}
func ExampleNew() {
	c, _ := New(identity(), 2, 10, 20)
	l, _ := c.Acquire([]Key{key(7), key(7)})
	fmt.Println(l.Slots(), len(l.Uploads()))
	for _, u := range l.Uploads() {
		_ = l.Started(u, &completion{})
	}
	_ = l.Commit()
	_ = l.Release(&completion{})
	_ = c.Close()
	// Output: [0 0] 1
}
