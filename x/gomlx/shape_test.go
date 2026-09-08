package gomlx

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"testing"
)

func TestSliceBounds(t *testing.T) {
	b := &Backend{}
	f := b.Builder("slice").Main()
	x, err := f.Parameter("x", shapes.Make(dtypes.Float32, 5), nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name                 string
		start, limit, stride []int
		want                 int
		invalid              bool
	}{
		{"stride", []int{0}, []int{5}, []int{2}, 3, false},
		{"default", []int{1}, []int{-1}, nil, 4, false},
		{"empty", []int{2}, []int{2}, nil, 0, false},
		{"negative", []int{-1}, []int{2}, nil, 0, true},
		{"past end", []int{0}, []int{6}, nil, 0, true},
		{"zero stride", []int{0}, []int{5}, []int{0}, 0, true},
		{"rank", nil, []int{5}, nil, 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			v, err := f.Slice(x, tc.start, tc.limit, tc.stride)
			if tc.invalid {
				if err == nil {
					t.Fatal("accepted invalid slice")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			s, err := f.Shape(v)
			if err != nil {
				t.Fatal(err)
			}
			if s.Dimensions[0] != tc.want {
				t.Fatalf("shape %v, want %d", s, tc.want)
			}
		})
	}
	if _, err := f.Concatenate(1, x, x); err == nil {
		t.Fatal("accepted invalid axis")
	}
	if _, err := f.Concatenate(0); err == nil {
		t.Fatal("accepted no operands")
	}
	y, err := f.Parameter("y", shapes.Make(dtypes.Float32, 1, 5), nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Concatenate(0, x, y); err == nil {
		t.Fatal("accepted mismatched rank")
	}
}
