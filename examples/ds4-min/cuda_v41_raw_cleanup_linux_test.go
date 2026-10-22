//go:build linux

package main

import (
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"io"
	"sync"
	"testing"
)

// Test resources have one release transaction. A failed drain stops every
// subsequent release, including module unload and context destruction.
type v41RawCleanup struct {
	drain      func() error
	ownerClose func() error
	releases   []func() error
	owner      any
	err        error
}

func (c *v41RawCleanup) close() error {
	if c.err != nil {
		return c.err
	}
	if c.ownerClose != nil {
		if err := c.drain(); err != nil {
			c.err = err
			return err
		}
		if err := c.ownerClose(); err != nil {
			c.err = err
			return err
		}
		c.ownerClose = nil
	}
	for len(c.releases) != 0 {
		if err := c.drain(); err != nil {
			c.err = fmt.Errorf("attention test drain: %w", err)
			return c.err
		}
		i := len(c.releases) - 1
		if err := c.releases[i](); err != nil {
			c.err = fmt.Errorf("attention test release: %w", err)
			return c.err
		}
		c.releases = c.releases[:i]
	}
	c.owner = nil
	return nil
}

var v41RawRetained struct {
	sync.Mutex
	transactions []*v41RawCleanup
}

// Failed compilation can return a live retry owner even without a program.
// Keep that owner in the same drain-before-release transaction as device work.
func (c *v41RawCleanup) retainError(err error) {
	var owner io.Closer
	if errors.As(err, &owner) {
		c.releases = append(c.releases, owner.Close)
	}
}

type v41RawRetryError struct {
	closed *int
	fail   bool
}

func (e *v41RawRetryError) Error() string { return "planted compiler retry owner" }
func (e *v41RawRetryError) Close() error {
	*e.closed++
	if e.fail {
		return errors.New("planted compiler cleanup failure")
	}
	return nil
}

func TestCUDAV41RawCompilerRetryOwner(t *testing.T) {
	for _, fail := range []bool{false, true} {
		closed, drained := 0, 0
		c := &v41RawCleanup{owner: new(int), drain: func() error { drained++; return nil }}
		c.retainError(fmt.Errorf("compile: %w", &v41RawRetryError{closed: &closed, fail: fail}))
		if len(c.releases) != 1 {
			t.Fatal("compiler retry owner dropped")
		}
		err := c.close()
		if closed != 1 || drained != 1 || (err != nil) != fail {
			t.Fatalf("closed%d drained%d err%v", closed, drained, err)
		}
		if fail && (c.owner == nil || len(c.releases) != 1 || c.close() == nil || closed != 1) {
			t.Fatal("failed retry owner not retained sticky")
		}
	}
}

func newV41RawTestResources(t *testing.T) (*cudaFoundation, *v41RawCleanup) {
	t.Helper()
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatalf("requested native gate has no CUDA device: %v", err)
	}
	r := &cudaFoundation{device: dev}
	c := &v41RawCleanup{owner: r, drain: func() error { return dev.WithContext(func() error { return syncStream(r.stream()) }) }}
	c.releases = append(c.releases, r.Close)
	registerV41RawCleanup(t, c)
	return r, c
}

func registerV41RawCleanup(t *testing.T, c *v41RawCleanup) {
	t.Helper()
	t.Cleanup(func() {
		if err := c.close(); err != nil {
			v41RawRetained.Lock()
			v41RawRetained.transactions = append(v41RawRetained.transactions, c)
			v41RawRetained.Unlock()
			t.Errorf("retaining CUDA inputs/module/context until process exit: %v", err)
		}
	})
}

func v41RawUpload(t *testing.T, r *cudaFoundation, c *v41RawCleanup, host []float32) cudamem.Slice[float32] {
	t.Helper()
	d, err := cudamem.Alloc[float32](r.device, len(host))
	if d.Ptr() != 0 {
		c.releases = append(c.releases, d.Free)
	}
	if err != nil {
		t.Fatal(err)
	}
	if err = d.CopyFromHost(host); err != nil {
		t.Fatal(err)
	}
	return d
}

func TestCUDAV41RawCleanupFaults(t *testing.T) {
	t.Run("owner-first", func(t *testing.T) {
		freed, closed := 0, 0
		c := &v41RawCleanup{owner: new(int), drain: func() error { return nil }, ownerClose: func() error { closed++; return errors.New("planted owner close") }, releases: []func() error{func() error { freed++; return nil }}}
		if c.close() == nil || c.owner == nil || freed != 0 || closed != 1 || len(c.releases) != 1 {
			t.Fatal("owner failure released operand or context")
		}
		if c.close() == nil || closed != 1 || freed != 0 {
			t.Fatal("sticky owner failure retried release")
		}
	})
	for _, failDrain := range []int{0, 1, 2, 3} {
		calls, released := 0, 0
		c := &v41RawCleanup{owner: new(int)}
		c.drain = func() error {
			calls++
			if calls == failDrain {
				return fmt.Errorf("planted sync failure")
			}
			return nil
		}
		for i := 0; i < 3; i++ {
			c.releases = append(c.releases, func() error { released++; return nil })
		}
		err := c.close()
		want := 3
		if failDrain != 0 {
			want = failDrain - 1
		}
		if released != want || (err != nil) != (failDrain != 0) {
			t.Fatalf("drain%d releases%d want%d err%v", failDrain, released, want, err)
		}
		if failDrain != 0 && (c.owner == nil || len(c.releases) != 3-want) {
			t.Fatal("failed drain dropped live owners")
		}
		if failDrain != 0 {
			if c.close() == nil || released != want || calls != failDrain {
				t.Fatal("sticky failure retried release")
			}
		}
	}
	c := &v41RawCleanup{owner: new(int), drain: func() error { return nil }}
	c.releases = []func() error{func() error { t.Fatal("released older resource after failure"); return nil }, func() error { return fmt.Errorf("planted free failure") }}
	if c.close() == nil || c.owner == nil || len(c.releases) != 2 {
		t.Fatal("release failure lost owners")
	}
}
