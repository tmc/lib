package cudacapture_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// headerSASS attributes the first instruction to an inlined header and the
// second to the main source.
const headerSASS = `.section .text.compute
//## File "helpers.cuh", line 7
/* 0x0000 */   MOV R1, c[0x0][0x28] ;
//## File "kernels.cu", line 3
/* 0x0010 */   RET ;
`

const headerPTX = `.version 8.0
.target sm_90
.address_size 64

.file 1 "kernels.cu"
.file 2 "helpers.cuh"

.visible .entry compute(
	.param .u64 a
)
{
	.loc 2 7 0;
	mov.u64 %rd1, %rd2;
	.loc 1 3 0;
	ret;
}
`

func TestHeaderOriginatedLineIsKept(t *testing.T) {
	code, err := cudacapture.LoadCode(loadFixtureCapture(t, headerPTX, headerSASS))
	if err != nil {
		t.Fatalf("LoadCode: %v", err)
	}
	inst := code.Kernels[0].Instruction(0)
	if inst == nil {
		t.Fatal("no instruction at offset 0")
	}
	// The instruction was inlined from a header. Zeroing its line because the
	// file is not the main source discards a location the disassembler
	// measured, and renders as "no source information" rather than as the
	// header line it is.
	if inst.SourceFile != "helpers.cuh" {
		t.Errorf("SourceFile = %q, want helpers.cuh", inst.SourceFile)
	}
	if inst.SourceLine != 7 {
		t.Errorf("SourceLine = %d, want 7", inst.SourceLine)
	}
	main := code.Kernels[0].Instruction(0x10)
	if main.SourceFile != "kernels.cu" || main.SourceLine != 3 {
		t.Errorf("main instruction = %s:%d, want kernels.cu:3", main.SourceFile, main.SourceLine)
	}
}

func TestKernelPTXLineIsTheEntryNotAnOpcodeGuess(t *testing.T) {
	code, err := cudacapture.LoadCode(loadFixtureCapture(t, "", ""))
	if err != nil {
		t.Fatalf("LoadCode: %v", err)
	}
	k := code.Kernels[0]
	// kernels.ptx declares .visible .entry compute on line 7. Selecting
	// instead the line of the first instruction whose opcode looks
	// "arithmetic" manufactures a line from opcode text.
	if k.PTXLine != 7 {
		t.Errorf("Kernel.PTXLine = %d, want 7 (the .entry line)", k.PTXLine)
	}
	_ = strings.TrimSpace("")
}
