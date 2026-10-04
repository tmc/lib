package cudalowering

import (
	"strings"
	"testing"
)

// Exact CUDA13.0 nvdisasm output from the retained offline lane72 attempt.
// This is textual mechanism evidence, not binary ISA attestation.
const observedIADD3Listing = `	.target	sm_121

	.elftype	@"ET_EXEC"


//--------------------- .text.kernel              --------------------------
	.section	.text.kernel,"ax",@progbits
	.align	128
        .global         kernel
        .type           kernel,@function
        .size           kernel,(.L_x_1 - kernel)
        .other          kernel,@"STO_CUDA_ENTRY STV_DEFAULT"
kernel:
.text.kernel:
        /*0000*/                   LDC R1, c[0x0][0x37c] ;
        /*0010*/                   S2R R3, SR_CgaCtaId ;
        /*0020*/                   LDC R9, c[0x0][0x388] ;
        /*0030*/                   LDCU.64 UR4, c[0x0][0x358] ;
        /*0040*/                   MOV R0, 0x400 ;
        /*0050*/                   S2R R5, SR_SWINHI ;
        /*0060*/                   LEA R0, R3, R0, 0x18 ;
        /*0070*/                   MOV R2, R0 ;
        /*0080*/                   MOV R3, R5 ;
        /*0090*/                   STS [R0], R9 ;
        /*00a0*/                   LD.E R2, desc[UR4][R2.64] ;
        /*00b0*/                   LDC.64 R4, c[0x0][0x380] ;
        /*00c0*/                   IADD3 R7, PT, PT, R2, 0x7, RZ ;
        /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;
        /*00e0*/                   EXIT ;
.L_x_0:
        /*00f0*/                   BRA ` + "`" + `(.L_x_0);
        /*0100*/                   NOP;
        /*0110*/                   NOP;
        /*0120*/                   NOP;
        /*0130*/                   NOP;
        /*0140*/                   NOP;
        /*0150*/                   NOP;
        /*0160*/                   NOP;
        /*0170*/                   NOP;
.L_x_1:


//--------------------- SYMBOLS --------------------------

	.type		.nv.reservedSmem.offset0,@object
	.size		.nv.reservedSmem.offset0,0x4
	.type		.nv.reservedSmem.cap,@object
	.size		.nv.reservedSmem.cap,0x4
`

func TestSharedConversionIADD3Observed(t *testing.T) {
	scan := scanSharedConversions(parseConversionFixture(t, observedIADD3Listing))
	want := sharedConversionChain{window: 0x50, lowCopy: 0x70, highCopy: 0x80, addressDefinition: 0x60, sharedAccess: 0x90, genericAccess: 0xa0}
	if len(scan.refusals) != 0 || len(scan.chains) != 1 || scan.chains[0] != want {
		t.Fatalf("scan=%+v want=%+v", scan, want)
	}
}

func TestSharedConversionIADD3ReachingIdentity(t *testing.T) {
	tests := []struct {
		name, instruction string
		wantChain         bool
	}{
		{"unrelated destination", "IADD3 R7, PT, PT, R2, 0x7, RZ", true},
		{"low destination", "IADD3 R2, PT, PT, R0, 0x7, RZ", false},
		{"in place low", "IADD3 R2, PT, PT, R2, 0x0, RZ", false},
		{"high destination", "IADD3 R3, PT, PT, R5, 0x0, RZ", false},
		{"shared destination", "IADD3 R0, PT, PT, R0, 0x0, RZ", false},
		{"original high source", "IADD3 R5, PT, PT, R5, 0x7, RZ", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			source := strings.Replace(observedSharedConversion, "/*0090*/ STS", "/*0088*/ "+tt.instruction+";\n/*0090*/ STS", 1)
			scan := scanSharedConversions(parseConversionFixture(t, renumberConversionFixture(source)))
			if tt.wantChain {
				if len(scan.chains) != 1 || len(scan.refusals) != 0 {
					t.Fatalf("supported instruction lost chain: %+v", scan)
				}
			} else if len(scan.chains) != 0 || len(scan.refusals) == 0 {
				t.Fatalf("overwritten provenance survived: %+v", scan)
			}
		})
	}
}

func TestSharedConversionIADD3RefusesOtherForms(t *testing.T) {
	tests := []struct{ name, instruction string }{
		{"missing operand", "IADD3 R7, PT, PT, R2, 0x7"},
		{"extra operand", "IADD3 R7, PT, PT, R2, 0x7, RZ, RZ"},
		{"first predicate output", "IADD3 R7, P0, PT, R2, 0x7, RZ"},
		{"second predicate output", "IADD3 R7, PT, P1, R2, 0x7, RZ"},
		{"predicate modifier", "IADD3 R7, !PT, PT, R2, 0x7, RZ"},
		{"predicated instruction", "@P0 IADD3 R7, PT, PT, R2, 0x7, RZ"},
		{"register addend", "IADD3 R7, PT, PT, R2, R8, RZ"},
		{"large immediate", "IADD3 R7, PT, PT, R2, 0x100000000, RZ"},
		{"negative immediate", "IADD3 R7, PT, PT, R2, -1, RZ"},
		{"bad immediate", "IADD3 R7, PT, PT, R2, BAD, RZ"},
		{"third input", "IADD3 R7, PT, PT, R2, 0x7, R9"},
		{"zero source", "IADD3 R7, PT, PT, RZ, 0x7, RZ"},
		{"negated source", "IADD3 R7, PT, PT, -R2, 0x7, RZ"},
		{"uniform source", "IADD3 R7, PT, PT, UR2, 0x7, RZ"},
		{"out of range source", "IADD3 R7, PT, PT, R256, 0x7, RZ"},
		{"reserved source", "IADD3 R7, PT, PT, R255, 0x7, RZ"},
		{"zero destination", "IADD3 RZ, PT, PT, R2, 0x7, RZ"},
		{"uniform destination", "IADD3 UR7, PT, PT, R2, 0x7, RZ"},
		{"reserved destination", "IADD3 R255, PT, PT, R2, 0x7, RZ"},
		{"modified opcode", "IADD3.X R7, PT, PT, R2, 0x7, RZ"},
		{"unknown opcode", "FOO R7, PT, PT, R2, 0x7, RZ"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			source := strings.Replace(observedSharedConversion, "/*0090*/ STS", "/*0088*/ "+tt.instruction+";\n/*0090*/ STS", 1)
			scan := scanSharedConversions(parseConversionFixture(t, renumberConversionFixture(source)))
			if len(scan.chains) != 0 || len(scan.refusals) == 0 {
				t.Fatalf("unsupported form accepted: %+v", scan)
			}
		})
	}
}
