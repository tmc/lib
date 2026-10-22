package dsv4

import (
	"math"
	"testing"
)

func TestContextMemoryCheckedPartition(t *testing.T) {
	for _, c := range []struct {
		name string
		m    ContextMemory
		want int64
		bad  bool
	}{
		{"live", ContextMemory{Raw: 32, Compressed: 64, Scratch: 16}, 112, false},
		{"released", ContextMemory{}, 0, false},
		{"negative", ContextMemory{Raw: -1}, 0, true},
		{"overflow", ContextMemory{Raw: math.MaxInt64, Scratch: 1}, 0, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, err := c.m.Total()
			if (err != nil) != c.bad || !c.bad && got != c.want {
				t.Fatalf("total=%d err=%v", got, err)
			}
		})
	}
}
