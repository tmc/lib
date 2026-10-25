package cudasass_test

import (
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// loopsListing has an outer loop holding an inner one, a cold block placed
// after EXIT that is entered by a forward branch and branches back to the
// instruction after it (a backward branch in address, but no cycle), a
// loop inside a subroutine reached by CALL, and the padding self-branch.
const loopsListing = "\t.section\t.text._Z1kv,\"ax\",@progbits\n" +
	"_Z1kv:\n" +
	".text._Z1kv:\n" +
	"        /*0000*/                   MOV R0, RZ ;\n" + // B0
	".L_x_1:\n" +
	"        /*0010*/                   IADD3 R0, R0, 0x1, RZ ;\n" + // B1 outer header
	".L_x_2:\n" +
	"        /*0020*/                   FADD R1, R1, R1 ;\n" + // B2 inner header
	"        /*0030*/              @P1 BRA `(.L_x_2) ;\n" +
	"        /*0040*/              @P2 BRA `(.L_x_1) ;\n" + // B3
	"        /*0050*/              @P3 BRA `(.L_x_9) ;\n" + // B4: to the cold block
	".L_x_3:\n" +
	"        /*0060*/                   CALL.REL.NOINC `(.L_x_5) ;\n" + // B5
	"        /*0070*/                   EXIT ;\n" + // B6
	".L_x_9:\n" +
	"        /*0080*/                   FMUL R2, R2, R2 ;\n" + // B7: cold
	"        /*0090*/                   BRA `(.L_x_3) ;\n" +
	".L_x_5:\n" +
	"        /*00a0*/                   FADD R3, R3, R3 ;\n" + // B8: subroutine loop
	"        /*00b0*/              @P4 BRA `(.L_x_5) ;\n" +
	"        /*00c0*/                   RET.REL.NODEC R20 `(_Z1kv) ;\n" + // B9
	".L_x_4:\n" +
	"        /*00d0*/                   BRA `(.L_x_4);\n" + // B10: padding
	".L_x_10:\n"

func TestLoops(t *testing.T) {
	d, err := cudasass.ParseString(loopsListing)
	if err != nil {
		t.Fatal(err)
	}
	g, err := cudasass.NewCFG(d.Functions[0])
	if err != nil {
		t.Fatal(err)
	}
	var starts []uint64
	for _, b := range g.Blocks {
		starts = append(starts, b.Start)
	}
	if want := []uint64{0, 0x10, 0x20, 0x40, 0x50, 0x60, 0x70, 0x80, 0xa0, 0xc0, 0xd0}; !reflect.DeepEqual(starts, want) {
		t.Fatalf("block starts %#x, want %#x", starts, want)
	}
	idom := g.Dominators()
	if want := []int{0, 0, 1, 2, 3, 4, 5, 4, 5, 8, -1}; !reflect.DeepEqual(idom, want) {
		t.Errorf("Dominators = %v, want %v", idom, want)
	}
	want := []cudasass.Loop{
		{Header: 1, Blocks: []int{1, 2, 3}},
		{Header: 2, Blocks: []int{2}},
		{Header: 8, Blocks: []int{8}},
	}
	if got := g.Loops(); !reflect.DeepEqual(got, want) {
		t.Errorf("Loops = %+v, want %+v", got, want)
	}
}

func TestReducible(t *testing.T) {
	// Two entries into the cycle B1 <-> B2: from B0 into each.
	const irreducible = "\t.section\t.text._Z1kv,\"ax\",@progbits\n" +
		"_Z1kv:\n" +
		".text._Z1kv:\n" +
		"        /*0000*/              @P0 BRA `(.L_x_2) ;\n" + // B0
		".L_x_1:\n" +
		"        /*0010*/                   FADD R1, R1, R1 ;\n" + // B1
		".L_x_2:\n" +
		"        /*0020*/                   FADD R2, R2, R2 ;\n" + // B2
		"        /*0030*/              @P1 BRA `(.L_x_1) ;\n" +
		"        /*0040*/                   EXIT ;\n" +
		".L_x_9:\n"
	for _, tt := range []struct {
		name, listing string
		want          bool
	}{
		{"natural", loopsListing, true},
		{"irreducible", irreducible, false},
	} {
		d, err := cudasass.ParseString(tt.listing)
		if err != nil {
			t.Fatal(err)
		}
		g, err := cudasass.NewCFG(d.Functions[0])
		if err != nil {
			t.Fatal(err)
		}
		if got := g.Reducible(); got != tt.want {
			t.Errorf("%s: Reducible = %v, want %v", tt.name, got, tt.want)
		}
	}
}
