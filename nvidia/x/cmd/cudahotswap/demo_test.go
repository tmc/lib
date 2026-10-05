package main

import (
	"context"
	"errors"
	"math"
	"testing"
)

func TestSaxpyCPUReference(t *testing.T) {
	for _, tc := range []struct {
		name string
		out  []float32
		fail bool
	}{
		{"correct", []float32{3, 7}, false},
		{"wrong", []float32{1, 1}, true},
		{"missing_store", []float32{3, math.Float32frombits(0x7fc00001)}, true},
		{"extent", []float32{3}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := compareSaxpy([]float32{1, 2}, []float32{1, 3}, tc.out)
			if (err != nil) != tc.fail {
				t.Fatalf("got %v fail %v", err, tc.fail)
			}
		})
	}
}

type demoReadCloser struct {
	calls int
	fail  bool
}

func (c *demoReadCloser) Close() error {
	c.calls++
	if c.fail {
		return errors.New("readback lease release")
	}
	return nil
}

func TestDemoRetainsFailedReadLease(t *testing.T) {
	c := &demoReadCloser{fail: true}
	a := &demoApplication{owners: new(sessionOwners), readLease: c}
	if err := a.close(context.Background()); err == nil || a.readLease != c {
		t.Fatal("readback owner lost")
	}
	c.fail = false
	if err := a.close(context.Background()); err != nil || a.readLease != nil || c.calls != 2 {
		t.Fatalf("retry err %v owner %v calls %d", err, a.readLease, c.calls)
	}
}
