package cudakv

import (
	"math"
	"testing"
)

func TestRollingCapacityOverflow(t *testing.T) {
	for _, tt := range []struct {
		name         string
		sink, window int
		valid        bool
	}{
		{"overflow", math.MaxInt, 1, false},
		{"large overflow", math.MaxInt/2 + 1, math.MaxInt/2 + 1, false},
		{"boundary", math.MaxInt - 1, 1, true},
		{"window only", 0, math.MaxInt, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c, err := NewRollingKVCache(SlidingWindowConfig{SinkTokens: tt.sink, WindowSize: tt.window})
			if !tt.valid {
				if err == nil {
					t.Fatal("accepted overflowing capacity")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if err := c.Advance(math.MaxInt); err != nil {
				t.Fatal(err)
			}
			if got := c.ActiveTokens(); got != math.MaxInt {
				t.Fatalf("active = %d", got)
			}
			if err := c.Advance(1); err == nil {
				t.Fatal("accepted overflowing advance")
			}
			if c.TotalSeen() != math.MaxInt {
				t.Fatal("failed advance mutated state")
			}
		})
	}
}
