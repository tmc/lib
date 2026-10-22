package dsv4expertcache

import (
	"bytes"
	"errors"
	"sync"
	"testing"
)

type selectionTestUpload struct {
	fail            bool
	closed          bool
	postCommitError bool
}

func TestCUDASelectionStreamAdmission(t *testing.T) {
	for _, name := range []string{"live", "wrong_device", "closed", "closing", "stale", "failed", "uncommitted", "recorded"} {
		t.Run(name, func(t *testing.T) {
			o, _ := selectionFixture(t)
			s, err := o.Select(0, []int{3, 1}, false)
			if err != nil {
				t.Fatal(err)
			}
			switch name {
			case "closed":
				o.closed = true
			case "closing":
				o.closing = true
			case "stale":
				o.active = nil
			case "failed":
				s.failed = true
			case "uncommitted":
				s.lease.committed = false
			case "recorded":
				s.consumer = &selectionTestConsumer{}
			}
			checks, launches, writes := 0, 0, 0
			err = s.checkStream(func() error {
				checks++
				if name == "wrong_device" {
					return errors.New("device differs")
				}
				return nil
			})
			if err == nil {
				launches++
				writes++
			}
			if name == "live" {
				if err != nil || checks != 1 || launches != 1 || writes != 1 {
					t.Fatal("live admission refused")
				}
			} else {
				if err == nil || launches != 0 || writes != 0 {
					t.Fatal("failed admission reached consumer")
				}
				if name != "wrong_device" && checks != 0 {
					t.Fatal("invalid selection reached device check")
				}
			}
			if s.lease.done || (name != "recorded" && s.consumer != nil) {
				t.Fatal("admission changed completion")
			}
		})
	}
}

func (u *selectionTestUpload) Stage(_ *Source, l *Lease, _ bool) error {
	for _, v := range l.Uploads() {
		if err := l.Started(v, &selectionTestConsumer{}); err != nil {
			return err
		}
	}
	if err := l.Commit(); err != nil {
		return err
	}
	if u.postCommitError {
		return errors.New("failure after commit")
	}
	return nil
}
func (u *selectionTestUpload) Close() error {
	if u.fail {
		return errors.New("free failed")
	}
	u.closed = true
	return nil
}

type selectionTestConsumer struct {
	failWait, failClose bool
	waits, closes       int
}

func (c *selectionTestConsumer) Wait() error {
	c.waits++
	if c.failWait {
		return errors.New("wait failed")
	}
	return nil
}
func (c *selectionTestConsumer) close() error {
	c.closes++
	if c.failClose {
		return errors.New("event destroy failed")
	}
	return nil
}

func selectionFixture(t *testing.T) (*CUDASelections, *selectionTestUpload) {
	t.Helper()
	id := Identity{Model: [32]byte{1}, Quant: [32]byte{2}, Layout: [32]byte{3}}
	s, err := NewSource(bytes.NewReader(make([]byte, 24)), 24, id, map[int]Triplet{0: {Gate: Tensor{0, 8, 2, 4}, Up: Tensor{8, 8, 2, 4}, Down: Tensor{16, 8, 2, 4}}})
	if err != nil {
		t.Fatal(err)
	}
	c, err := New(id, 2, 6, 12)
	if err != nil {
		t.Fatal(err)
	}
	u := &selectionTestUpload{}
	o := &CUDASelections{source: s, cache: c, upload: u, view: func(slot, component int) (uint64, uint64, error) { return uint64(100 + slot*10 + component*2), 2, nil }}
	return o, u
}

func TestCUDASelectionDuplicateAndStale(t *testing.T) {
	o, _ := selectionFixture(t)
	s, err := o.Select(0, []int{3, 1, 3}, false)
	if err != nil {
		t.Fatal(err)
	}
	a, _ := s.Views(0)
	b, _ := s.Views(2)
	x, _, _ := a[0].Address()
	y, _, _ := b[0].Address()
	if x != y {
		t.Fatal("duplicate route order lost")
	}
	if _, err := s.Views(3); err == nil {
		t.Fatal("out of range accepted")
	}
	if err := s.finish(func() (cudaConsumer, error) { return &selectionTestConsumer{}, nil }); err != nil {
		t.Fatal(err)
	}
	if _, _, err := a[0].Address(); err == nil {
		t.Fatal("stale view accepted")
	}
	if _, err := s.Views(0); err == nil {
		t.Fatal("stale selection accepted")
	}
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestCUDASelectionFailedWaitRetainsReaders(t *testing.T) {
	o, u := selectionFixture(t)
	s, err := o.Select(0, []int{0}, false)
	if err != nil {
		t.Fatal(err)
	}
	c := &selectionTestConsumer{failWait: true}
	if err := s.finish(func() (cudaConsumer, error) { return c, nil }); err == nil {
		t.Fatal("wait failure hidden")
	}
	if _, err := o.Select(0, []int{1}, false); err == nil {
		t.Fatal("live reader evicted")
	}
	if err := o.Close(); err == nil || u.closed {
		t.Fatal("active reader freed")
	}
	c.failWait = false
	c.failClose = true
	if err := o.Close(); err == nil || u.closed {
		t.Fatal("event owner discarded")
	}
	c.failClose = false
	u.fail = true
	if err := o.Close(); err == nil || u.closed {
		t.Fatal("free failure hidden")
	}
	u.fail = false
	if err := o.Close(); err != nil || !u.closed {
		t.Fatal("close retry failed", err)
	}
}

func TestCUDASelectionRecordFailureAndConcurrentRequests(t *testing.T) {
	o, u := selectionFixture(t)
	s, err := o.Select(0, []int{0}, false)
	if err != nil {
		t.Fatal(err)
	}
	c := &selectionTestConsumer{failWait: true}
	if err := s.finish(func() (cudaConsumer, error) { return c, errors.New("record failed") }); err == nil {
		t.Fatal("record failure hidden")
	}
	var wg sync.WaitGroup
	for range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := o.Select(0, []int{1}, false); err == nil {
				t.Error("concurrent reader evicted")
			}
		}()
	}
	wg.Wait()
	if err := o.Close(); err == nil || u.closed {
		t.Fatal("fallback drain failure freed storage")
	}
	c.failWait = false
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestCUDASelectionUnrecordedCloseRefuses(t *testing.T) {
	o, u := selectionFixture(t)
	s, err := o.Select(0, []int{0}, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := o.Close(); err == nil || u.closed {
		t.Fatal("unrecorded consumer freed")
	}
	if err := s.finish(func() (cudaConsumer, error) { return &selectionTestConsumer{}, nil }); err != nil {
		t.Fatal(err)
	}
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestCUDASelectionViewsRaceCompletion(t *testing.T) {
	o, _ := selectionFixture(t)
	s, err := o.Select(0, []int{0}, false)
	if err != nil {
		t.Fatal(err)
	}
	v, err := s.Views(0)
	if err != nil {
		t.Fatal(err)
	}
	start := make(chan struct{})
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			for range 1000 {
				_, _, _ = v[0].Address()
				_, _ = s.Views(0)
			}
		}()
	}
	close(start)
	if err := s.finish(func() (cudaConsumer, error) { return &selectionTestConsumer{}, nil }); err != nil {
		t.Fatal(err)
	}
	wg.Wait()
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}
}

type selectionConsumedEvent struct {
	waits, rawDestroy int
	consumed          bool
}

func (c *selectionConsumedEvent) Wait() error { c.waits++; return nil }
func (c *selectionConsumedEvent) close() error {
	if c.consumed {
		return nil
	}
	c.rawDestroy++
	c.consumed = true
	return errors.New("restore context after successful destroy")
}

func TestCUDASelectionFinishRetryConsumedEvent(t *testing.T) {
	o, _ := selectionFixture(t)
	s, err := o.Select(0, []int{0}, false)
	if err != nil {
		t.Fatal(err)
	}
	c := &selectionConsumedEvent{}
	if err := s.finish(func() (cudaConsumer, error) { return c, nil }); err == nil {
		t.Fatal("restore error hidden")
	}
	if _, err := o.Select(0, []int{1}, false); err == nil {
		t.Fatal("unfinished event owner lost")
	}
	if err := s.finish(func() (cudaConsumer, error) { t.Fatal("consumer recreated"); return nil, nil }); err != nil {
		t.Fatal(err)
	}
	if c.waits != 1 || c.rawDestroy != 1 {
		t.Fatalf("completion or destruction repeated: %+v", c)
	}
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestCUDASelectionShapeAdmission(t *testing.T) {
	o, _ := selectionFixture(t)
	for _, strides := range [][3]int{{0, 0, 0}, {2, 0, 2}, {-1, 2, 2}, {3, 2, 2}} {
		if _, err := cudaSelectionCache(o.source, 2, strides); err == nil {
			t.Fatalf("invalid strides accepted: %v", strides)
		}
	}
	if _, err := cudaSelectionCache(o.source, 2, [3]int{2, 2, 2}); err != nil {
		t.Fatal(err)
	}
}

type selectionUncommitted struct{}

func (*selectionUncommitted) Stage(*Source, *Lease, bool) error { return nil }
func TestCUDASelectionSourcePrerequisites(t *testing.T) {
	o, _ := selectionFixture(t)
	var missing *selectionUncommitted
	if _, err := o.source.Select(o.cache, 0, []int{0}, missing, false); err == nil {
		t.Fatal("typed nil stager accepted")
	}
	if l, err := o.source.Select(o.cache, 0, []int{0}, &selectionUncommitted{}, false); err == nil || l != nil || o.cache.active != nil {
		t.Fatal("uncommitted stager accepted")
	}
}

func TestCUDASelectionPostCommitFailure(t *testing.T) {
	o, u := selectionFixture(t)
	u.postCommitError = true
	s, err := o.Select(0, []int{0}, false)
	if err == nil || s == nil || !s.lease.committed {
		t.Fatal("committed error owner discarded")
	}
	if _, err := s.Views(0); err == nil {
		t.Fatal("failed selection exposed views")
	}
	if err := s.finish(func() (cudaConsumer, error) { t.Fatal("failed selection accepted consumer"); return nil, nil }); err == nil {
		t.Fatal("failed selection accepted Finish")
	}
	if err := o.Close(); err != nil || !u.closed {
		t.Fatal("unexposed committed lease orphaned", err)
	}
}
