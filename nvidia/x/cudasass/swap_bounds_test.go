package cudasass_test

import (
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

func swapBoundsBlock() cudasass.BasicBlock {
	// These constructed instructions exercise only the host legality model.
	// Equal synthetic control fields have no waits or differing stalls.
	return cudasass.BasicBlock{Instructions: []cudasass.Instruction{
		{Opcode: "NOP", Encoding: cudasass.Encoding{Lo: 1, Hi: 0x000fe20000007000}},
		{Opcode: "NOP", Encoding: cudasass.Encoding{Lo: 1, Hi: 0x000fe20000007000}},
		{Opcode: "NOP", Encoding: cudasass.Encoding{Lo: 1, Hi: 0x000fe20000007000}},
	}}
}

func TestCanSwapIndexBounds(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	minInt := -maxInt - 1
	for _, block := range []struct {
		name string
		b    cudasass.BasicBlock
	}{{"empty", cudasass.BasicBlock{}}, {"nonempty", swapBoundsBlock()}} {
		for _, pair := range [][2]int{{maxInt, minInt}, {minInt, maxInt}, {-1, 0}, {0, -1}, {3, 2}, {2, 3}} {
			t.Run(fmt.Sprintf("%s/%d/%d", block.name, pair[0], pair[1]), func(t *testing.T) {
				defer func() {
					if p := recover(); p != nil {
						t.Errorf("CanSwap panicked for invalid positions: %v", p)
					}
				}()
				v := cudasass.CanSwap(block.b, pair[0], pair[1], cudasass.SM121a, cudasass.NumericalPolicy{})
				if v.Legal || v.Undecided || len(v.Reasons) != 1 || v.Reasons[0] != "position out of range" {
					t.Errorf("CanSwap = %+v, want position out of range refusal", v)
				}
			})
		}
	}
}

func TestCanSwapValidPositions(t *testing.T) {
	b := swapBoundsBlock()
	for _, pair := range [][2]int{{0, 1}, {1, 0}, {1, 2}, {2, 1}} {
		if v := cudasass.CanSwap(b, pair[0], pair[1], cudasass.SM121a, cudasass.NumericalPolicy{}); !v.Legal || v.Undecided || len(v.Reasons) != 0 {
			t.Errorf("CanSwap(%d, %d) = %+v, want legal", pair[0], pair[1], v)
		}
	}
	for _, pair := range [][2]int{{0, 0}, {0, 2}, {2, 0}} {
		if v := cudasass.CanSwap(b, pair[0], pair[1], cudasass.SM121a, cudasass.NumericalPolicy{}); v.Legal || v.Undecided || len(v.Reasons) == 0 {
			t.Errorf("CanSwap(%d, %d) = %+v, want nonadjacent refusal", pair[0], pair[1], v)
		}
	}
	if v := cudasass.CanSwap(b, 0, 1, cudasass.Arch("sm_90a"), cudasass.NumericalPolicy{}); v.Legal || !v.Undecided {
		t.Errorf("unsupported architecture = %+v, want undecided refusal", v)
	}
}
