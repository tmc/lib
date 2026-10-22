package main

import (
	"math"
	"testing"
)

func TestShapeStateReserve(t *testing.T) {
	for _, tt := range []struct {
		name                     string
		reserve, consumers, want uint64
		bad                      bool
	}{
		{"empty", 0, 0, 192, false},
		{"normal", 100, 200, 492, false},
		{"limit", math.MaxUint64 - 192, 0, math.MaxUint64, false},
		{"reserve overflow", math.MaxUint64 - 191, 0, 0, true},
		{"consumer overflow", 0, math.MaxUint64, 0, true},
		{"combined overflow", 1, math.MaxUint64 - 192, 0, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := shapeStateReserve(tt.reserve, tt.consumers)
			if (err != nil) != tt.bad || got != tt.want {
				t.Fatalf("reserve=%d consumers=%d got=%d error=%v", tt.reserve, tt.consumers, got, err)
			}
		})
	}
}
