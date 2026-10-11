//go:build linux && (amd64 || arm64)

package bpfprobe

import (
	"encoding/binary"
	"runtime"
	"testing"
)

func TestArchitectureProgram(t *testing.T) {
	regs := []int16{112, 104, 96, 88, 72, 64}
	sp, pc, ret, skip, trap := int16(152), int16(128), int16(80), int32(8), 298
	if runtime.GOARCH == "arm64" {
		regs = []int16{0, 8, 16, 24, 32, 40, 48, 56}
		sp, pc, ret, skip, trap = 248, 256, 0, 0, 241
	}
	if sysPerfEventOpen != trap || stackArgumentOffset != skip {
		t.Fatal("architecture syscall/stack contract")
	}
	for _, returning := range []bool{false, true} {
		code := program(10, 11, 1, returning, true, 12, 13, 1, 1, 8, 16, 8, false, -1)
		loads := map[int16]bool{}
		var foundMask, foundRead bool
		for i := 0; i < len(code); i += 8 {
			op, dstsrc := code[i], code[i+1]
			off := int16(binary.LittleEndian.Uint16(code[i+2:]))
			imm := int32(binary.LittleEndian.Uint32(code[i+4:]))
			if op == 0x79 && dstsrc>>4 == 6 {
				loads[off] = true
			}
			if op == 0xb7 && dstsrc == 1 && imm == int32((1<<len(regs))-1) {
				foundMask = true
			}
			if op == 0xb7 && dstsrc == 2 && imm == int32((11-len(regs))*8) {
				foundRead = true
			}
		}
		for _, off := range []int16{sp, pc, ret} {
			if !loads[off] {
				t.Fatalf("missing register offset%d", off)
			}
		}
		if !returning {
			for _, off := range regs {
				if !loads[off] {
					t.Fatalf("missing arg offset%d", off)
				}
			}
			if !foundMask || !foundRead {
				t.Fatal("stack read/register availability mask")
			}
		}
	}
}
