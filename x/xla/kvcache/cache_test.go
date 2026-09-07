package kvcache

import (
	"errors"
	"math"
	"testing"

	"github.com/tmc/lib/x/xla"
)

func TestConfig(t *testing.T) {
	for _, cfg := range []Config{
		{}, {Capacity: -1, KeySize: 1, ValueSize: 1, ElementType: xla.F32},
		{BatchShape: []int64{0}, Capacity: 1, KeySize: 1, ValueSize: 1, ElementType: xla.F32},
		{Capacity: math.MaxInt64, KeySize: 2, ValueSize: 1, ElementType: xla.F32},
	} {
		if _, err := New(nil, cfg); err == nil {
			t.Fatalf("accepted %v", cfg)
		}
	}
}

func TestCommitCleanupFailureClosesCache(t *testing.T) {
	oldK, oldV := new(xla.Buffer), new(xla.Buffer)
	c := &Cache{keys: oldK, values: oldV, length: 3, plans: map[int64]*xla.Executable{1: new(xla.Executable)}}
	outputs := []*xla.Buffer{new(xla.Buffer), new(xla.Buffer), new(xla.Buffer)}
	sentinel := errors.New("cleanup failure")
	released := map[*xla.Buffer]int{}
	out, err := c.commit(outputs, 1, func(b *xla.Buffer) error {
		released[b]++
		if b == oldK {
			return sentinel
		}
		return nil
	})
	if !errors.Is(err, sentinel) || out != nil {
		t.Fatalf("result %v, %v", out, err)
	}
	if !c.closed || c.keys != nil || c.values != nil || c.Len() != 0 || len(c.plans) != 0 {
		t.Fatal("cleanup failure did not close cache")
	}
	if released[oldK] != 1 || released[oldV] != 1 {
		t.Fatalf("releases: %v", released)
	}
	if err := c.Reset(); err == nil {
		t.Fatal("reset reopened closed cache")
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
}
