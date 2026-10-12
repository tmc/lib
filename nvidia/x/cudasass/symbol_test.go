package cudasass_test

import (
	"github.com/tmc/lib/nvidia/x/cudasass"
	"slices"
	"testing"
)

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
	d, err := cudasass.ParseString(sass)
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
	if len(fn.Symbols) != 1 || fn.Symbols[0] != (cudasass.Symbol{Name: slowpath, Address: 0x30}) {
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

// These listings exercise the public symbol contract with constructed internal
// declarations; they are not captures of demangled vendor slowpaths.
func TestParseWholeFunctionSymbols(t *testing.T) {
	for _, tt := range []struct{ name, declaration, want string }{
		{"ordinary", ".type slow,@function", "slow"},
		{"demangled", ".type ns::slow(float *, int),@function", "ns::slow(float *, int)"},
		{"quoted", `.type "ns::slow(float *, int)",@function`, "ns::slow(float *, int)"},
		{"line comment", ".type ns::slow(float *, int),@function // |       |", "ns::slow(float *, int)"},
		{"object", ".type value,@object", "k"},
		{"function attribute in comment", ".type slow,@function // note ,@function", "slow"},
		{"demangled attribute in comment", ".type ns::slow(float *, int),@function // annotation ,@function", "ns::slow(float *, int)"},
		{"quoted attribute in comment", `.type "ns::slow(float *, int)",@function // annotation ,@function`, "ns::slow(float *, int)"},
		{"object attribute in comment", ".type slow,@object // note ,@function", "k"},
		{"demangled object in comment", ".type ns::slow(float *, int),@object // note ,@function", "k"},
		{"quoted object in comment", `.type "ns::slow(float *, int)",@object // note ,@function`, "k"},
		{"quoted comment marker", `.type "ns::slow//path(float *, int)",@function // note ,@function`, "ns::slow//path(float *, int)"},
		{"quoted attribute literal", `.type "slow,@function",@function // note ,@function`, "slow,@function"},

		{"directive boundary", ".typeinfo slow,@function", "k"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			d, err := cudasass.ParseString(".text.k:\n /*0000*/ EXIT ;\n" + tt.declaration + "\n /*0010*/ RET ;\n")
			if err != nil {
				t.Fatal(err)
			}
			fn := d.Function("k")
			if fn == nil {
				t.Fatal("function k not parsed")
			}
			if got := fn.SymbolAt(0x10); got != tt.want {
				t.Fatalf("SymbolAt(0x10) = %q, want %q", got, tt.want)
			}
			var want []cudasass.Symbol
			if tt.want != "k" {
				want = []cudasass.Symbol{{Name: tt.want, Address: 0x10}}
			}
			if !slices.Equal(fn.Symbols, want) {
				t.Fatalf("Symbols = %+v, want %+v", fn.Symbols, want)
			}
		})
	}
}

func TestParseFunctionSymbolClosure(t *testing.T) {
	const name = "ns::kernel(float *, int)"
	const first = "ns::slow(float *, int)"
	const second = "ns::slow(float *, double)"
	listing := ".text." + name + ":\n.type " + name + ",@function\n /*0000*/ EXIT ;\n.type " + first + ",@function\n /*0010*/ RET ;\n.type " + second + ",@function\n /*0020*/ RET ;\n.type pending(int),@function\n.text.next:\n /*0000*/ EXIT ;\n"
	d, err := cudasass.ParseString(listing)
	if err != nil {
		t.Fatal(err)
	}
	fn := d.Function(name)
	if fn == nil {
		t.Fatal("demangled kernel not parsed")
	}
	want := []cudasass.Symbol{{Name: first, Address: 0x10}, {Name: second, Address: 0x20}}
	if !slices.Equal(fn.Symbols, want) {
		t.Fatalf("Symbols = %+v, want %+v", fn.Symbols, want)
	}
	if got := fn.SymbolAt(0); got != name {
		t.Fatalf("self declaration changed SymbolAt(0) to %q", got)
	}
	next := d.Function("next")
	if next == nil || len(next.Symbols) != 0 || next.SymbolAt(0) != "next" {
		t.Fatalf("pending declaration crossed kernel header: %+v", next)
	}
}
