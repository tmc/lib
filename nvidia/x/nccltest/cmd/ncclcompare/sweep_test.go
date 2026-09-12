package main

import (
	"slices"
	"testing"
)

func TestReferenceSweepArgs(t *testing.T) {
	for _, tt := range []struct {
		sizes []int
		want  []string
	}{
		{[]int{16}, []string{"-b", "16", "-e", "16"}},
		{[]int{16, 32, 48}, []string{"-b", "16", "-e", "48", "-i", "16", "-f", "1"}},
		{[]int{16, 128, 1024}, []string{"-b", "16", "-e", "1024", "-f", "8"}},
		{nil, nil}, {[]int{0}, nil}, {[]int{64, 16}, nil}, {[]int{16, 16}, nil}, {[]int{16, 32, 80}, nil},
	} {
		got, err := referenceSweepArgs(tt.sizes)
		if (err == nil) != (tt.want != nil) || !slices.Equal(got, tt.want) {
			t.Fatalf("sizes=%v: got %v %v", tt.sizes, got, err)
		}
	}
}
