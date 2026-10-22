//go:build linux

package main

import "testing"

func TestResidentSet(t *testing.T) {
	const page = 4096
	tensors := map[string]cudaFileExtent{
		"a.weight":                 {8192, 100},
		"b.weight":                 {8192 + 100 + 32, 200}, // merges with a
		"c.weight":                 {1 << 20, 64},          // gap > 64 KiB: own span
		"blk.0.ffn_up_exps.weight": {2 << 20, 1 << 20},     // replaced
		"empty.weight":             {3 << 20, 0},
	}
	s := planResidentSet(tensors, cudaExpertReplaced, page, 1<<20+100)
	if len(s.spans) != 2 {
		t.Fatalf("spans = %v, want 2", s.spans)
	}
	if want := uint64(page + 100); s.bytes != want {
		t.Errorf("bytes = %d, want %d", s.bytes, want)
	}
	tests := []struct {
		name     string
		off, n   uint64
		dev      uint64
		ok       bool
		replaced bool
	}{
		{"a", 8192, 100, 0, true, false},
		{"b", 8192 + 132, 200, 132, true, false},
		{"c", 1 << 20, 64, page, true, false},
		{"past end of file", 1 << 20, 101, 0, false, false},
		{"expert", 2 << 20, 1 << 20, 0, false, true},
		{"expert prefix", 2 << 20, 64, 0, false, false},
		{"before first span", 0, 16, 0, false, false},
		{"past span end", 8192, page + 1, 0, false, false},
		{"between spans", 1 << 19, 16, 0, false, false},
	}
	for _, tt := range tests {
		dev, ok := s.lookup(tt.off, tt.n)
		if ok != tt.ok || dev != tt.dev {
			t.Errorf("%s: lookup = %d, %v; want %d, %v", tt.name, dev, ok, tt.dev, tt.ok)
		}
		if got := s.isReplaced(tt.off, tt.n); got != tt.replaced {
			t.Errorf("%s: isReplaced = %v, want %v", tt.name, got, tt.replaced)
		}
	}
}
