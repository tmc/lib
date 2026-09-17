package cudaptx

import "testing"

// A .nv_intermediate_source_section is a brace-delimited module-scope block
// whose .source_begin/.source_end text is arbitrary and "may contain arbitrary
// text including newlines" (docs/ptx/11-directives.md:1878-1879). Nothing in it
// is PTX, so none of it may reach the statement scanner.
const sectionSrc = `.version 9.4
.target sm_121
.address_size 64
.file 1 "kernel.cu"
.file 2 "tileIR_source.ir"
.nv_intermediate_source_section {
    .code_block {
        .ir_name:        "TileIR"
        .sourceFileName: 2
        .source_begin
            // a brace that never closes: {
            fn tile(x) { if x > 0 /* and a block comment opener
            ret;
        .source_end
    }
}
.visible .entry after()
{
	.loc 1 5 1
	mul.lo.s32 %r1, %r2, %r3;
	ret;
}
`

func TestIntermediateSourceSectionIsOpaque(t *testing.T) {
	m, err := ParseString(sectionSrc)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Kernels) != 1 {
		t.Fatalf("got %d kernels, want 1: %+v", len(m.Kernels), m.Kernels)
	}
	k := m.Kernels[0]
	if k.Name != "after" {
		t.Errorf("kernel name = %q, want after", k.Name)
	}
	if len(k.Instructions) != 2 {
		t.Fatalf("got %d instructions, want 2: %+v", len(k.Instructions), k.Instructions)
	}
	if got := k.Instructions[0].Loc; got.File != "kernel.cu" || got.Line != 5 {
		t.Errorf("Loc = %v, want kernel.cu:5", got)
	}
}

func TestIntermediateSourceSectionBraceOnNextLine(t *testing.T) {
	src := `.version 9.4
.target sm_121
.file 1 "kernel.cu"
.nv_intermediate_source_section
{
    .code_block {
        .source_begin
            fn tile(x) { unbalanced
        .source_end
    }
}
.visible .entry after()
{
	.loc 1 5 1
	ret;
}
`
	m, err := ParseString(src)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Kernels) != 1 || m.Kernels[0].Name != "after" {
		t.Fatalf("kernels = %+v, want one named after", m.Kernels)
	}
}
