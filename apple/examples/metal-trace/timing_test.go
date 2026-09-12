package main

import (
	"math"
	"testing"
)

func TestCommandBufferNanos(t *testing.T) {
	for _, tt := range []struct {
		name       string
		start, end float64
		ok         bool
	}{
		{"valid", 1, 1.001, true}, {"zero start", 0, 1, false}, {"zero", 0, 0, false}, {"reversed", 2, 1, false}, {"nan", math.NaN(), 2, false}, {"infinite", 1, math.Inf(1), false}, {"overflow", 1, 1e20, false}, {"subnanosecond", 1, 1 + 1e-12, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			n, e := commandBufferNanos(tt.start, tt.end)
			if (e == nil) != tt.ok {
				t.Fatalf("n=%d err=%v", n, e)
			}
			if tt.ok && n <= 0 {
				t.Fatal("nonpositive duration")
			}
		})
	}
}

func TestInvalidOutput(t *testing.T) {
	for _, tt := range []struct {
		name        string
		got         float32
		wantInvalid bool
	}{
		{"valid", 1, false}, {"wrong", 2, true}, {"nan", float32(math.NaN()), true}, {"infinite", float32(math.Inf(1)), true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := invalidOutput(tt.got, 1, 1e-5); got != tt.wantInvalid {
				t.Fatalf("invalid=%v", got)
			}
		})
	}
}
