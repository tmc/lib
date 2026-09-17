package cudaptx

import "testing"

// A nested inline is written as successive .loc directives, each carrying one
// inlined_at that names the location of the .loc before it
// (docs/ptx/11-directives.md:1739-1748). Reading only the last directive gives
// one frame of a chain that is two deep.
const nestedInlineSrc = `.version 7.2
.target sm_121
.file 1 "kernel.cu"
.visible .entry k()
{
	.loc 1 27 3
	.loc 1 10 5, function_name info_string1, inlined_at 1 27 3
	.loc 1 15 3, function_name .debug_str+16, inlined_at 1 10 5
	setp.ne.s32 %p2, %r1, 18;
	.loc 1 40 1
	ret;
}
`

func TestInlinedAtChainsThroughSuccessiveLocs(t *testing.T) {
	m, err := ParseString(nestedInlineSrc)
	if err != nil {
		t.Fatal(err)
	}
	inst := m.Kernels[0].Instructions[0]
	if inst.Loc.Line != 15 {
		t.Fatalf("Loc.Line = %d, want 15", inst.Loc.Line)
	}
	want := []int{10, 27}
	if len(inst.InlinedAt) != len(want) {
		t.Fatalf("InlinedAt = %v, want %d frames (innermost first: 10 then 27)", inst.InlinedAt, len(want))
	}
	for i, w := range want {
		if inst.InlinedAt[i].Line != w {
			t.Errorf("InlinedAt[%d].Line = %d, want %d", i, inst.InlinedAt[i].Line, w)
		}
	}
}

func TestLocWithoutInlinedAtEndsTheChain(t *testing.T) {
	m, err := ParseString(nestedInlineSrc)
	if err != nil {
		t.Fatal(err)
	}
	// The .loc before ret has no inlined_at, so ret is not inlined at all.
	// Carrying the chain across it would report a call site that does not
	// exist.
	ret := m.Kernels[0].Instructions[1]
	if ret.Opcode != "ret" {
		t.Fatalf("second instruction = %q, want ret", ret.Opcode)
	}
	if ret.InlinedAt != nil {
		t.Errorf("ret InlinedAt = %v, want none", ret.InlinedAt)
	}
}

func TestUnlinkedInlinedAtDoesNotChain(t *testing.T) {
	// The inlined_at here names a location that is not the preceding .loc, so
	// there is no evidence the two directives describe one chain and only the
	// frame actually written is reported.
	src := `.version 7.2
.target sm_121
.file 1 "kernel.cu"
.visible .entry k()
{
	.loc 1 27 3
	.loc 1 10 5, inlined_at 1 99 9
	ret;
}
`
	m, err := ParseString(src)
	if err != nil {
		t.Fatal(err)
	}
	got := m.Kernels[0].Instructions[0].InlinedAt
	if len(got) != 1 || got[0].Line != 99 {
		t.Errorf("InlinedAt = %v, want exactly the written frame at line 99", got)
	}
}

func TestLocRecordsFunctionNameReference(t *testing.T) {
	m, err := ParseString(nestedInlineSrc)
	if err != nil {
		t.Fatal(err)
	}
	// function_name is a .debug_str label expression, and it is the only thing
	// distinguishing two functions inlined at the same line.
	if got := m.Kernels[0].Instructions[0].Loc.FunctionName; got != ".debug_str+16" {
		t.Errorf("Loc.FunctionName = %q, want %q", got, ".debug_str+16")
	}
	if got := m.Kernels[0].Instructions[1].Loc.FunctionName; got != "" {
		t.Errorf("ret Loc.FunctionName = %q, want empty", got)
	}
}
