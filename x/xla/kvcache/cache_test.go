package kvcache

import (
	"errors"
	"math"
	"testing"

	"github.com/tmc/lib/x/xla"
)

func TestConfig(t *testing.T) {
	for _, cfg := range []Config{
		{Capacity: 4, KeySize: 2, ValueSize: 3, ElementType: xla.F32, QueryHeads: 3, KVHeads: 2},
		{Capacity: 4, KeySize: 2, ValueSize: 3, ElementType: xla.F32, QueryHeads: 4},
		{Capacity: 4, KeySize: 2, ValueSize: 3, ElementType: xla.F32, KVHeads: 2},
		{Capacity: 4, KeySize: 2, ValueSize: 3, ElementType: xla.F32, QueryHeads: -4, KVHeads: 2},
		{Capacity: 4, KeySize: 2, ValueSize: 3, ElementType: xla.F32, QueryHeads: math.MaxInt64, KVHeads: 1},
		{}, {Capacity: -1, KeySize: 1, ValueSize: 1, ElementType: xla.F32},
		{BatchShape: []int64{0}, Capacity: 1, KeySize: 1, ValueSize: 1, ElementType: xla.F32},
		{Capacity: math.MaxInt64, KeySize: 2, ValueSize: 1, ElementType: xla.F32},
		{Capacity: math.MaxInt64 / 4, KeySize: 1, ValueSize: 1, ElementType: xla.F16},
	} {
		if _, err := New(nil, cfg); err == nil {
			t.Fatalf("accepted %v", cfg)
		}
	}
}

func TestCommitCleanupFailureClosesCache(t *testing.T) {
	oldK, oldV := new(xla.Buffer), new(xla.Buffer)
	c := &Cache{keys: oldK, values: oldV, lengths: []int64{3}, plans: map[int64]*xla.Executable{1: new(xla.Executable)}}
	outputs := []*xla.Buffer{new(xla.Buffer), new(xla.Buffer), new(xla.Buffer)}
	sentinel := errors.New("cleanup failure")
	released := map[*xla.Buffer]int{}
	out, err := c.commit(outputs, []int64{1}, func(b *xla.Buffer) error {
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

func TestCounts(t *testing.T) {
	c := &Cache{config: Config{Capacity: 4}, lengths: []int64{4, 3, 1, 0}}
	for _, counts := range [][]int64{nil, {0}, {0, -1, 0, 0}, {0, 0, 4, 0}, {1, 0, 0, 0}, {0, 2, 0, 0}} {
		if err := c.validateCounts(3, counts); err == nil {
			t.Fatalf("accepted %v", counts)
		}
	}
	if err := c.validateCounts(3, []int64{0, 1, 3, 0}); err != nil {
		t.Fatal(err)
	}
}
