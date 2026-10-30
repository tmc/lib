package cudasass

import (
	"slices"
	"strings"
	"testing"
)

// symbolRegressionCall is a constructed demangled kernel and subroutine listing.
const symbolRegressionCall = `
//--------------------- .text.scale(float *, int) --------------------------
	.section	.text.scale(float *, int),"ax",@progbits
        .global         scale(float *, int)
        .type           scale(float *, int),@function
        .size           scale(float *, int),(.L_x_1 - scale(float *, int))
        .other          scale(float *, int),@"STO_CUDA_ENTRY STV_DEFAULT"
scale(float *, int):
.text.scale(float *, int):
        /*0000*/                   LDC R1, c[0x0][0x28] ;
        /*0010*/                   CALL.REL.NOINC ` + "`" + `(helper(float, float)) ;
        /*0020*/                   EXIT ;
        .type           helper(float, float),@function
        .size           helper(float, float),(.L_x_1 - helper(float, float))
helper(float, float):
        /*0030*/                   FADD R0, R4, R5 ;
        /*0040*/                   RET.REL.NODEC R20 0x0 ;
.L_x_1:
`

func TestFunctionSymbolRegression(t *testing.T) {
	tests := []struct {
		line   string
		want   string
		wantOK bool
	}{
		{".type           _Z10atomicDropPfPKfi,@function", "_Z10atomicDropPfPKfi", true},
		{".type\t\t_Z10atomicDropPfPKfi,@function", "_Z10atomicDropPfPKfi", true},
		{".type           atomicDrop(float *, const float *, int),@function", "atomicDrop(float *, const float *, int)", true},
		{".type           fmaChain(double *, const double *, const double *, int),@function", "fmaChain(double *, const double *, const double *, int)", true},
		{".type           _Z10atomicDropPfPKfi,@function                                                   // |       |        |        |", "_Z10atomicDropPfPKfi", true},
		{".type           atomicDrop(float *, const float *, int),@function       // |       |        |        |", "atomicDrop(float *, const float *, int)", true},
		{".type           $__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath,@function", "$__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath", true},
		{`.type "f(int)",@function`, "f(int)", true},
		{`.type f<"a//b">(),@function`, `f<"a//b">()`, true},
		{`.type "f<\"a//b\">()",@function // c`, `f<\"a//b\">()`, true},
		{`.type "scale(float *, int)",@function`, "scale(float *, int)", true},
		{`.type "scale(float *, int)" , @function // c`, "scale(float *, int)", true},
		{"\t.type\t\t.nv.reservedSmem.offset0,@object", "", false},
		{".type           atomicDrop(float *, const float *, int),@object", "", false},
		{".typed k,@function", "", false},
		{".type", "", false},
		{".type ,@function", "", false},
		{`.type "",@function`, "", false},
		{".type k", "", false},
		{".type k,@function junk", "", false},
		{`.type "f(int),@function`, "", false},
		{`.type "a"b",@function`, "", false},
		{`.type "" k,@function`, "", false},
		{`.type f"x,@function`, "", false},
	}
	for _, tt := range tests {
		got, ok := functionSymbol(strings.TrimSpace(tt.line))
		if got != tt.want || ok != tt.wantOK {
			t.Errorf("functionSymbol(%q) = %q, %v, want %q, %v", tt.line, got, ok, tt.want, tt.wantOK)
		}
	}
}

func TestParseDemangledSymbolsRegression(t *testing.T) {
	const (
		kernel = "scale(float *, int)"
		helper = "helper(float, float)"
	)
	life := strings.ReplaceAll(symbolRegressionCall, ",@function\n", ",@function        // |       |        |\n")
	quoted := strings.ReplaceAll(symbolRegressionCall, "type           "+helper+",", `type           "`+helper+`",`)
	noLabel := strings.Replace(symbolRegressionCall, ".text."+kernel+":\n", "", 1)
	noLabel = strings.Replace(noLabel, kernel+":\n", "", 1)
	tests := []struct {
		name    string
		listing string
	}{
		{"demangled", symbolRegressionCall},
		{"life-range columns", life},
		{"quoted subroutine", quoted},
		{"directive without label", noLabel},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d, err := ParseString(tt.listing)
			if err != nil {
				t.Fatal(err)
			}
			if len(d.Functions) != 1 {
				t.Fatalf("got %d functions, want 1", len(d.Functions))
			}
			fn := d.Function(kernel)
			if fn == nil || len(fn.Instructions) != 5 {
				t.Fatalf("function %q not parsed with 5 instructions", kernel)
			}
			if want := []Symbol{{Name: helper, Address: 0x30}}; !slices.Equal(fn.Symbols, want) {
				t.Fatalf("Symbols = %+v, want %+v", fn.Symbols, want)
			}
			for _, at := range []struct {
				addr uint64
				want string
			}{
				{0x00, kernel},
				{0x20, kernel},
				{0x30, helper},
				{0x40, helper},
			} {
				if got := fn.SymbolAt(at.addr); got != at.want {
					t.Errorf("SymbolAt(%#x) = %q, want %q", at.addr, got, at.want)
				}
			}
		})
	}
}
