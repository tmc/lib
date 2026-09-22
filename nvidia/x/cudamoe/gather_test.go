package cudamoe_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamoe"
)

func TestGatherGeometry(t *testing.T) {
	for _, tc := range []struct {
		tokens, width, tile int
		blocks              uint32
	}{
		{5, 1, 16, 1}, {5, 1, 32, 2}, {5, 33, 16, 2}, {5, 33, 32, 3}, {8, 16, 16, 1}, {9, 16, 16, 2},
	} {
		grid, block, err := cudamoe.GatherGeometry(tc.tokens, tc.width, tc.tile)
		if err != nil || grid != [3]uint32{tc.blocks, 1, 1} || block != [3]uint32{128, 1, 1} {
			t.Fatalf("%+v: grid=%v block=%v err=%v", tc, grid, block, err)
		}
	}
	for _, tc := range [][3]int{{0, 32, 16}, {1, 0, 16}, {-1, 32, 32}, {1, 32, 4}, {math.MaxInt32, math.MaxInt32, 32}} {
		if _, _, err := cudamoe.GatherGeometry(tc[0], tc[1], tc[2]); err == nil {
			t.Fatalf("accepted %v", tc)
		}
	}
}

func ExampleGatherGeometry() {
	grid, block, err := cudamoe.GatherGeometry(5, 33, 16)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(grid, block)
	// Output: [2 1 1] [128 1 1]
}

func TestGatherCandidates(t *testing.T) {
	candidates, err := cudamoe.GatherCandidates(5, 33, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(candidates) != 3 {
		t.Fatalf("candidates=%v", candidates)
	}
	// Descriptions allocate no tensor storage. Only tile32 exceeds the grid
	// limit here; it must not hide the still-legal scalar and tile16 schedules.
	large, err := cudamoe.GatherCandidates(1_000_000_000, 257, 1)
	if err != nil || len(large) != 2 || large[0].Tile != 0 || large[1].Tile != 16 {
		t.Fatalf("lost supported schedules: %+v %v", large, err)
	}
	for i, c := range candidates {
		if c.Tile != []int{0, 16, 32}[i] || c.Grid[0] != []uint32{2, 2, 3}[i] {
			t.Fatalf("candidate=%+v", c)
		}
		for j, want := range []int{396, 660, 20} {
			got, err := c.Layouts[j].FootprintBytes()
			if err != nil || got != want || c.Layouts[j].AlignmentBytes() != 4 {
				t.Fatalf("layout %d: got=%d err=%v", j, got, err)
			}
		}
	}
	for _, dims := range [][3]int{{0, 33, 3}, {5, 0, 3}, {5, 33, 0}, {math.MaxInt32, math.MaxInt32, 3}, {5, math.MaxInt32, math.MaxInt32}} {
		if _, err := cudamoe.GatherCandidates(dims[0], dims[1], dims[2]); err == nil {
			t.Fatalf("accepted %v", dims)
		}
	}
}

func ExampleGatherCandidates() {
	candidates, err := cudamoe.GatherCandidates(5, 33, 3)
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, c := range candidates {
		fmt.Println(c.Name, c.Grid)
	}
	// Output:
	// gather_scalar [2 1 1]
	// gather16 [2 1 1]
	// gather32 [3 1 1]
}
