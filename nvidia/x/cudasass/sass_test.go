package cudasass_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

const sampleSASS = `
--- Kernel: LinearGEMV ---
/* 0x7f001200 */   MOV R1, c[0x0][0x28] ;              // Initialize block stack pointer
/* 0x7f001210 */   ISETP.GE.AND P0, PT, R1, R8, PT ;   // Check bound
/* 0x7f001220 */   @P0 RET.REL.NODEC ;                 // Early return
/* 0x7f001230 */   LDG.E.CONSTANT.F32 R4, [R2.64] ;    // Global load
/* 0x7f001250 */   FFMA R0, R4, R5, R0 ;               // Fused multiply-add accumulation

--- Kernel: RMSNorm ---
/* 0x7f001100 */   LDG.E.F32 R4, [R2.64] ;             // Global load x[i]
/* 0x7f001110 */   FFMA R0, R4, R4, R0 ;               // Accumulate sum of squares
/* 0x7f001140 */   BAR.SYNC 0x0 ;                      // Sync warps
`

func TestParse(t *testing.T) {
	d, err := cudasass.ParseString(sampleSASS)
	if err != nil {
		t.Fatalf("ParseString failed: %v", err)
	}

	if len(d.Functions) != 2 {
		t.Fatalf("expected 2 functions, got %d", len(d.Functions))
	}

	gemv := d.Function("LinearGEMV")
	if gemv == nil {
		t.Fatal("expected LinearGEMV function")
	}
	if gemv.BaseAddress != 0x7f001200 {
		t.Errorf("expected base address 0x7f001200, got 0x%x", gemv.BaseAddress)
	}
	if len(gemv.Instructions) != 5 {
		t.Fatalf("expected 5 instructions in LinearGEMV, got %d", len(gemv.Instructions))
	}

	retInst := gemv.Instructions[2]
	if retInst.Address != 0x7f001220 {
		t.Errorf("expected address 0x7f001220, got 0x%x", retInst.Address)
	}
	if retInst.Predicate != "@P0" {
		t.Errorf("expected predicate @P0, got %q", retInst.Predicate)
	}
	if retInst.Opcode != "RET.REL.NODEC" {
		t.Errorf("expected opcode RET.REL.NODEC, got %q", retInst.Opcode)
	}

	ffmaInst := gemv.Instructions[4]
	if ffmaInst.Address != 0x7f001250 {
		t.Errorf("expected address 0x7f001250, got 0x%x", ffmaInst.Address)
	}
	if ffmaInst.Opcode != "FFMA" {
		t.Errorf("expected opcode FFMA, got %q", ffmaInst.Opcode)
	}
	if ffmaInst.Operands != "R0, R4, R5, R0" {
		t.Errorf("expected operands 'R0, R4, R5, R0', got %q", ffmaInst.Operands)
	}
}

func TestParseReader(t *testing.T) {
	r := strings.NewReader(sampleSASS)
	d, err := cudasass.Parse(r)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}
	if len(d.Functions) != 2 {
		t.Fatalf("expected 2 functions, got %d", len(d.Functions))
	}
}

func TestDataDirectives(t *testing.T) {
	for _, tt := range []struct{ name, line string }{
		{"byte", "/*0000*/ .byte 0x01, 0x02"},
		{"word", "/*0004*/ .word 0x00000000"},
		{"symbol", "/*0044*/ .dword compute"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, ok := cudasass.ParseInstructionLine(tt.line); ok {
				t.Fatal("data directive parsed as an instruction")
			}
			d, err := cudasass.ParseString(tt.line + "\n.section .text.compute,\"ax\",@progbits\n/*0000*/ FFMA R0, R1, R2, R0;\n" + tt.line)
			if err != nil {
				t.Fatal(err)
			}
			if len(d.Functions) != 1 || d.Functions[0].Name != "compute" || len(d.Functions[0].Instructions) != 1 {
				t.Fatalf("unexpected functions: %+v", d.Functions)
			}
		})
	}
}

func TestParseLineInfo(t *testing.T) {
	input := `
--- Kernel: compute ---
//## File "kernels.cu", line 10
/* 0000 */   MOV R1, c[0x0][0x28] ;
/* 0010 */   ISETP.GE.AND P0, PT, R1, R8, PT ;
//## File "kernels.cu", line 15
/* 0020 */   @P0 RET.REL.NODEC ;

--- Kernel: other ---
//## File "other.cu", line 99
/* 0000 */   EXIT ;
`
	d, err := cudasass.ParseString(input)
	if err != nil {
		t.Fatalf("ParseString: %v", err)
	}

	fn := d.Function("compute")
	if fn == nil || len(fn.Instructions) != 3 {
		t.Fatalf("unexpected compute fn: %+v", fn)
	}

	if fn.Instructions[0].SourceFile != "kernels.cu" || fn.Instructions[0].SourceLine != 10 {
		t.Errorf("inst 0 source = %s:%d, want kernels.cu:10", fn.Instructions[0].SourceFile, fn.Instructions[0].SourceLine)
	}
	if fn.Instructions[1].SourceFile != "kernels.cu" || fn.Instructions[1].SourceLine != 10 {
		t.Errorf("inst 1 source = %s:%d, want kernels.cu:10", fn.Instructions[1].SourceFile, fn.Instructions[1].SourceLine)
	}
	if fn.Instructions[2].SourceFile != "kernels.cu" || fn.Instructions[2].SourceLine != 15 {
		t.Errorf("inst 2 source = %s:%d, want kernels.cu:15", fn.Instructions[2].SourceFile, fn.Instructions[2].SourceLine)
	}

	other := d.Function("other")
	if other == nil || len(other.Instructions) != 1 {
		t.Fatalf("unexpected other fn: %+v", other)
	}
	if other.Instructions[0].SourceFile != "other.cu" || other.Instructions[0].SourceLine != 99 {
		t.Errorf("other inst source = %s:%d, want other.cu:99", other.Instructions[0].SourceFile, other.Instructions[0].SourceLine)
	}
}
