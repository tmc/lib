package main

import (
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// loopListing is nvdisasm -g output, cut down from a kernel compiled with
// -maxrregcount 32: an outer loop holding an inner one, a forward branch,
// a stack access that is not a spill, and the trailing self-branch every
// kernel ends with.
const loopListing = `	.target	sm_90a
	.section	.text._Z1kPf,"ax",@progbits
        .global         _Z1kPf
        .type           _Z1kPf,@function
        .size           _Z1kPf,(.L_x_9 - _Z1kPf)
_Z1kPf:
.text._Z1kPf:
	//## File "/src/k.cu", line 3
        /*0000*/                   STL [R1+0x1c], R4                                 (*"SpillRefill"*);
        /*0010*/              @!P0 BRA ` + "`" + `(.L_x_0) ;
.L_x_1:
	//## File "/src/k.cu", line 6
        /*0020*/                   LDL.LU R21, [R1+0x1c]                             (*"SpillRefill"*);
.L_x_2:
	//## File "/src/k.cu", line 8
        /*0030*/                   STL [R1+0x20], R5                                 (*"SpillRefill"*);
        /*0040*/                   LDL R6, [R1+0x40] ;
        /*0050*/              @P1 BRA ` + "`" + `(.L_x_2) ;
	//## File "/src/k.cu", line 6
        /*0060*/              @P2 BRA ` + "`" + `(.L_x_1) ;
.L_x_0:
	//## File "/src/k.cu", line 11
        /*0070*/                   LDL.LU R25, [R1+0x20]                             (*"SpillRefill"*);
        /*0080*/                   EXIT ;
.L_x_3:
        /*0090*/                   BRA ` + "`" + `(.L_x_3);
.L_x_9:
`

func TestLocalAccesses(t *testing.T) {
	d, err := cudasass.Parse(strings.NewReader(loopListing))
	if err != nil {
		t.Fatal(err)
	}
	got, indirect, err := localAccesses(d.Functions[0])
	if err != nil || indirect {
		t.Fatalf("localAccesses: indirect %v, err %v", indirect, err)
	}
	want := []localAccess{
		{PC: 0x00, Opcode: "STL", Operands: "[R1+0x1c], R4", Spill: true, File: "/src/k.cu", Line: 3, Loops: 0},
		{PC: 0x20, Opcode: "LDL.LU", Operands: "R21, [R1+0x1c]", Spill: true, File: "/src/k.cu", Line: 6, Loops: 1},
		{PC: 0x30, Opcode: "STL", Operands: "[R1+0x20], R5", Spill: true, File: "/src/k.cu", Line: 8, Loops: 2},
		{PC: 0x40, Opcode: "LDL", Operands: "R6, [R1+0x40]", Spill: false, File: "/src/k.cu", Line: 8, Loops: 2},
		{PC: 0x70, Opcode: "LDL.LU", Operands: "R25, [R1+0x20]", Spill: true, File: "/src/k.cu", Line: 11, Loops: 0},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("localAccesses =\n%+v\nwant\n%+v", got, want)
	}
}

func TestPrintLocals(t *testing.T) {
	d, err := cudasass.Parse(strings.NewReader(loopListing))
	if err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	acc, _, err := localAccesses(d.Functions[0])
	if err != nil {
		t.Fatal(err)
	}
	printLocals(&b, "sm_90a", "_Z1kPf", acc)
	want := `  local sm_90a _Z1kPf: 5 accesses, 3 inside loops
    k.cu:8 spill, loop depth 2: 1 STL, 0 LDL (first at 0x0030)
    k.cu:8 local, loop depth 2: 0 STL, 1 LDL (first at 0x0040)
    k.cu:6 spill, loop depth 1: 0 STL, 1 LDL (first at 0x0020)
    k.cu:3 spill, outside loops: 1 STL, 0 LDL (first at 0x0000)
    k.cu:11 spill, outside loops: 0 STL, 1 LDL (first at 0x0070)
`
	if b.String() != want {
		t.Errorf("printLocals =\n%s\nwant\n%s", b.String(), want)
	}
}
