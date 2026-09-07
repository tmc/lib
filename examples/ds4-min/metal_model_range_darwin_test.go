//go:build darwin && arm64

package main

import (
	"math"
	"testing"
)

func TestPlanMetalModelViews(t *testing.T) {
	const page = uint64(16 << 10)
	ranges, err := planMetalModelViews(page+32, 20*page, 10*page, 2*page, page)
	if err != nil {
		t.Fatal(err)
	}
	if len(ranges) < 2 {
		t.Fatalf("got %d ranges, want overlapping views", len(ranges))
	}
	wantEnd := page + 32 + 20*page
	if ranges[0].offset != page+32 {
		t.Fatalf("first offset = %d, want %d", ranges[0].offset, page+32)
	}
	last := ranges[len(ranges)-1]
	if last.offset+last.length != wantEnd {
		t.Fatalf("last end = %d, want %d", last.offset+last.length, wantEnd)
	}
	for i, view := range ranges {
		prefix := view.offset & (page - 1)
		backing, err := checkedRoundPage(prefix+view.length, page)
		if err != nil {
			t.Fatal(err)
		}
		if backing > 10*page {
			t.Fatalf("range %d backing length = %d, maximum %d", i, backing, 10*page)
		}
		if i == 0 {
			continue
		}
		previous := ranges[i-1]
		overlap := previous.offset + previous.length - view.offset
		if overlap < 3*page {
			t.Fatalf("range %d overlap = %d, want at least %d", i, overlap, 3*page)
		}
	}
	for offset := ranges[0].offset; offset+2*page <= wantEnd; offset += 32 {
		covered := false
		for _, view := range ranges {
			if offset >= view.offset && offset+2*page <= view.offset+view.length {
				covered = true
				break
			}
		}
		if !covered {
			t.Fatalf("tensor range [%d,%d) crosses every mapped view", offset, offset+2*page)
		}
	}
}

func TestPlanMetalModelViewsErrors(t *testing.T) {
	const page = uint64(16 << 10)
	tests := []struct {
		name                        string
		offset, length, max, tensor uint64
		page                        uint64
	}{
		{"empty", 0, 0, 10 * page, page, page},
		{"file range overflow", math.MaxUint64 - 1, 2, 10 * page, page, page},
		{"zero page", 0, page, 10 * page, page, 0},
		{"non power of two page", 0, page, 10 * page, page, page - 1},
		{"tensor overflow", 0, page, math.MaxUint64, math.MaxUint64, page},
		{"buffer too small", 0, 4 * page, 3 * page, 2 * page, page},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := planMetalModelViews(test.offset, test.length, test.max, test.tensor, test.page); err == nil {
				t.Fatal("planMetalModelViews succeeded")
			}
		})
	}
}
