package closecopy

import (
	"errors"
	"fmt"
	"syscall"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

func twice(s driver.Stream) {
	s2 := s
	s2.Close()
	s.Close() // want `s.Close releases a handle that s2.Close released on line 14; each release zeroes only its own copy`
}

func usedAfter(s driver.Stream) error {
	s2 := s
	if err := s2.Close(); err != nil {
		return err
	}
	return s.Synchronize() // want `s is used after s2.Close released it on line \d+; each release zeroes only its own copy`
}

func deferred() error {
	s, err := driver.NewStream(0)
	if err != nil {
		return err
	}
	defer s.Close() // want `s.Close releases a handle that c.Close released on line \d+`
	c := s
	return c.Close()
}

type pair struct {
	a, b driver.Stream
}

func structCopy(p pair) {
	q := p
	q.a.Close()
	p.a.Close() // want `p.a.Close releases a handle that q.a.Close released on line \d+`
}

func ranged(streams []driver.Stream) {
	for _, s := range streams {
		s.Close()
	}
	streams[0].Close() // want `streams\[0\].Close releases a handle that s.Close released on line \d+`
}

func destroy(s driver.Stream) {
	s.Close()
}

func passed(s driver.Stream) {
	destroy(s)
	s.Synchronize() // want `s is used after destroy\(s\) released it on line \d+`
}

func deferredLiteral(s driver.Stream) {
	c := s
	defer func() {
		c.Close()
		s.Close() // want `s.Close releases a handle that c.Close released on line \d+`
	}()
}

// Only the copy is released, and the original is not used again.
func onlyCopy(s driver.Stream) error {
	c := s
	return c.Close()
}

// The original is cleared once its copy owns the handle.
func handedOver(p *pair) error {
	old := p.a
	p.a = 0
	if err := old.Close(); err != nil {
		return err
	}
	return p.a.Close()
}

func exclusiveBranches(s driver.Stream, which bool) {
	c := s
	if which {
		c.Close()
	} else {
		s.Close()
	}
}

func switchCases(s driver.Stream, n int) {
	c := s
	switch n {
	case 0:
		c.Close()
	case 1:
		s.Close()
	}
}

var errFail = errors.New("fail")

func earlyReturn(s driver.Stream, fail bool) error {
	c := s
	if fail {
		c.Close()
		return errFail
	}
	return s.Close()
}

// Comparing or printing the original reads the variable, not the handle.
func printed(s driver.Stream) {
	c := s
	c.Close()
	if s != 0 && 0 != s {
		fmt.Printf("%#x\n", uintptr(s))
	}
}

// A pointer copy shares the handle, and Close zeroes it for both.
func pointerCopy(s *driver.Stream) {
	p := s
	p.Close()
	s.Close()
}

// A wrapper held by pointer records its release in the shared value.
func wrapper(s *cuda.Stream) {
	t := s
	t.Close()
	s.Close()
}

// Released before the copy is made: a use after release, not a copy.
func releasedFirst(s driver.Stream) {
	s.Close()
	c := s
	c.Close()
}

// A released loop variable whose slice is then cleared.
func rangedCleared(streams []driver.Stream) {
	for _, s := range streams {
		s.Close()
	}
	streams = nil
	for _, s := range streams {
		s.Close()
	}
}

// The index form releases the element itself.
func rangedIndex(streams []driver.Stream) {
	for i := range streams {
		streams[i].Close()
	}
}

// A package's Close has no receiver to zero.
func packageClose(fd int) {
	syscall.Close(fd)
	syscall.Close(fd)
}

// Destroy of a Graph has a value receiver and zeroes nothing; the copy
// changes nothing.
func valueReceiver(g driver.Graph) {
	c := g
	c.Destroy()
	g.Destroy()
}

// A function literal that is not deferred may never run.
func literal(s driver.Stream) func() {
	c := s
	f := func() { c.Close() }
	s.Close()
	return f
}

func swap(p *driver.Stream) {
	*p = 0
}

// Once its address is taken the original may hold another handle.
func addressed(s driver.Stream) {
	c := s
	c.Close()
	swap(&s)
	s.Close()
}

// A copy of something that is not a handle.
func notHandle(streams []driver.Stream) {
	n := len(streams)
	m := n
	_ = m
	for _, s := range streams {
		s.Close()
	}
}

// Deferred calls run last deferred first: s is released before c here, and
// the later release is reported.
func deferOrder(s driver.Stream) {
	c := s
	defer c.Close() // want `c.Close releases a handle that s.Close released on line \d+`
	defer s.Close()
}

// A deferred release registered before an early return runs on it too.
func deferThenReturn(s driver.Stream, fail bool) error {
	defer s.Close() // want `s.Close releases a handle that c.Close released on line \d+`
	c := s
	if fail {
		c.Close()
		return errFail
	}
	return nil
}

// A release in a loop that breaks out before the original is released.
func loopBreak(streams []driver.Stream) {
	for _, s := range streams {
		if s != 0 {
			s.Close()
			break
		}
	}
	streams[0].Close()
}

func twoFields(p pair) {
	q := p
	q.a.Close()
	p.b.Close()
}

// A release that does not return ends the path.
func panics(s driver.Stream) {
	c := s
	if s == 0 {
		c.Close()
		panic("zero")
	}
	s.Close()
}

// A method value binds the handle it is taken from.
func methodValue(s driver.Stream) func() error {
	c := s
	c.Close()
	return s.Synchronize // want `s is used after c.Close released it`
}

// A copy into a struct field is still a copy.
func intoField(s driver.Stream) {
	var p pair
	p.a = s
	p.a.Close()
	s.Close() // want `s.Close releases a handle that p.a.Close released on line \d+`
}

// Two copies of one handle, both released, and the original too: each later
// release is reported once.
func twoCopies(s driver.Stream) {
	a := s
	b := s
	a.Close()
	b.Close() // want `b.Close releases a handle that a.Close released on line \d+`
	s.Close() // want `s.Close releases a handle that a.Close released on line \d+`
}

// The copy that was released is zeroed; releasing it again does nothing.
func sameCopy(s driver.Stream) {
	c := s
	c.Close()
	c.Close()
}

// Assigning to the original is not a use of it.
func clearedAfter(s driver.Stream) {
	c := s
	c.Close()
	s = 0
	_ = s
}

func rangedTwice(streams []driver.Stream) {
	for _, s := range streams {
		s.Close()
	}
	for _, t := range streams {
		t.Close() // want `t.Close releases a handle that s.Close released on line \d+`
	}
}

// A variable that joins the handle's copies late was not one before.
func laterJoin(s driver.Stream) {
	c := s
	c.Close()
	var x driver.Stream
	x.Synchronize()
	x = s // want `s is used after c.Close released it`
	_ = x
}

type triple struct {
	a, ab driver.Stream
}

// A field whose name begins with another's is a different place.
func prefixField(p triple) {
	c := p.a
	p.ab = 0
	c.Close()
	p.a.Close() // want `p.a.Close releases a handle that c.Close released on line \d+`
}

func destroyFirst(a, b driver.Stream) {
	a.Close()
}

// Only the parameter the function releases is released.
func otherParam(t, s driver.Stream) {
	destroyFirst(t, s)
	s.Synchronize()
}

// A pointer copy shares the struct it points to.
func pointerStruct(p *pair) {
	q := p
	q.a.Close()
	p.a.Close()
}

type holder struct {
	p *pair
	s driver.Stream
}

// A struct copy shares what its pointer fields point to.
func sharedThroughPointer(h holder) {
	g := h
	g.p.a.Close()
	h.p.a.Close()
}

func arrayCopy(a [2]driver.Stream) {
	b := a
	b[0].Close()
	a[0].Close() // want `a\[0\].Close releases a handle that b\[0\].Close released on line \d+`
}

func arrayRange(a [2]driver.Stream) {
	for _, s := range a {
		s.Close()
	}
	a[0].Close() // want `a\[0\].Close releases a handle that s.Close released on line \d+`
}

type list struct {
	streams []driver.Stream
}

// A struct copy shares the elements of its slice fields.
func sliceField(l list) {
	m := l
	m.streams[0].Close()
	l.streams[0].Close()
}

func declared(s driver.Stream) {
	var c = s
	c.Close()
	s.Close() // want `s.Close releases a handle that c.Close released on line \d+`
}

// Taking the address after the second release changes nothing before it.
func addressedLater(s driver.Stream) {
	c := s
	c.Close()
	s.Close() // want `s.Close releases a handle that c.Close released on line \d+`
	swap(&s)
}

// Arithmetic on a released handle reads it.
func arithmetic(s driver.Stream) driver.Stream {
	c := s
	c.Close()
	return s + 1 // want `s is used after c.Close released it`
}

func inCondition(s driver.Stream) {
	c := s
	if c.Close() == nil {
		s.Close() // want `s.Close releases a handle that c.Close released on line \d+`
	}
}

// A release deferred after a path that returns does not run on that path.
func deferAfterReturn(s driver.Stream, fail bool) error {
	c := s
	if fail {
		c.Close()
		return errFail
	}
	defer s.Close()
	return nil
}

func inInit(s driver.Stream) {
	c := s
	if err := c.Close(); err == nil {
		s.Close() // want `s.Close releases a handle that c.Close released on line \d+`
	}
}
