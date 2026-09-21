package cudasass

import "testing"

// TestParseSymbols checks that a subroutine the compiler places after the
// kernel in the same section is recorded as a symbol at its first
// instruction, and that the kernel's own .type directive is not.
func TestParseSymbols(t *testing.T) {
	const sass = `
//--------------------- .text.k          --------------------------
	.section	.text.k,"ax",@progbits
        .global         k
        .type           k,@function
k:
.text.k:
        /*0000*/                   LDC R1, c[0x0][0x37c] ;
        /*0010*/                   CALL.REL.NOINC ` + "`" + `($__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath) ;
        /*0020*/                   EXIT ;
        .weak           $__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath
        .type           $__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath,@function
        .size           $__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath,(.L_x_15 - $__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath)
$__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath:
        /*0030*/                   SHF.R.U32.HI R12, RZ, 0x17, R15 ;
        /*0040*/                   RET.REL.NODEC R20 0x0 ;
`
	d, err := ParseString(sass)
	if err != nil {
		t.Fatal(err)
	}
	fn := d.Function("k")
	if fn == nil {
		t.Fatal("function k not parsed")
	}
	if len(fn.Instructions) != 5 {
		t.Fatalf("got %d instructions, want 5", len(fn.Instructions))
	}
	const slowpath = "$__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath"
	if len(fn.Symbols) != 1 || fn.Symbols[0] != (Symbol{Name: slowpath, Address: 0x30}) {
		t.Fatalf("Symbols = %+v, want one %s at 0x30", fn.Symbols, slowpath)
	}
	for _, tt := range []struct {
		addr uint64
		want string
	}{
		{0x00, "k"},
		{0x20, "k"},
		{0x30, slowpath},
		{0x40, slowpath},
	} {
		if got := fn.SymbolAt(tt.addr); got != tt.want {
			t.Errorf("SymbolAt(%#x) = %q, want %q", tt.addr, got, tt.want)
		}
	}
}
