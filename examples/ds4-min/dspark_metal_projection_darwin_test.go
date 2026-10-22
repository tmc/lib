//go:build darwin && arm64

package main

import "testing"

func TestMetalDSparkOutputGeometry(t *testing.T) {
	for _, drafts := range []int{1, 3, 7, 16} {
		end := 0
		for row := 0; row < drafts; row++ {
			for group := 0; group < 8; group++ {
				weight, start, next, err := metalDSparkOutputRow(drafts, row, group, drafts*8192)
				if err != nil || weight != group*1024 || start != end || next-start != 1024 {
					t.Fatalf("row %d group %d geometry %d/%d/%d: %v", row, group, weight, start, next, err)
				}
				end = next
			}
		}
		if end != drafts*8192 {
			t.Fatal("output A rows do not fill the admitted output B input")
		}
		if _, _, _, err := metalDSparkOutputRow(drafts, 0, 0, drafts*1024); err == nil {
			t.Fatal("admitted former group128 scratch geometry")
		}
	}
}
