package cudaptx

import "testing"

func TestMalformedLocIsUnknownNotLineZero(t *testing.T) {
	// A .loc whose line does not parse used to yield FileIndex=1, Line=0: a
	// location that reads as known, naming a file and a line that does not
	// exist. The zero Location is what every consumer already reads as
	// unknown.
	for _, bad := range []string{
		".loc 1 abc 3",
		".loc x 5 3",
		".loc 1",
		".loc",
	} {
		src := ".version 8.8\n.target sm_121\n.file 1 \"kernel.cu\"\n.visible .entry k()\n{\n\t" + bad + "\n\tret;\n}\n"
		m, err := ParseString(src)
		if err != nil {
			t.Fatalf("%s: %v", bad, err)
		}
		if got := m.Kernels[0].Instructions[0].Loc; got != (Location{}) {
			t.Errorf("%q gave Loc %+v, want the zero Location", bad, got)
		}
	}
}

func TestMalformedInlinedAtIsDropped(t *testing.T) {
	src := `.version 8.8
.target sm_121
.file 1 "kernel.cu"
.visible .entry k()
{
	.loc 1 9 3, inlined_at 1 zz 3
	ret;
}
`
	m, err := ParseString(src)
	if err != nil {
		t.Fatal(err)
	}
	inst := m.Kernels[0].Instructions[0]
	if inst.Loc.Line != 9 {
		t.Errorf("Loc.Line = %d, want 9; a bad inlined_at does not invalidate the .loc", inst.Loc.Line)
	}
	if inst.InlinedAt != nil {
		t.Errorf("InlinedAt = %v, want none", inst.InlinedAt)
	}
}

func TestLocColumnZeroIsKept(t *testing.T) {
	// Column 0 is written by real toolchains (docs/ptx/11-directives.md:1735)
	// and is a value, not a parse failure.
	m, err := ParseString(".version 8.8\n.target sm_121\n.file 2 \"a.cu\"\n.visible .entry k()\n{\n\t.loc 2 4237 0\n\tret;\n}\n")
	if err != nil {
		t.Fatal(err)
	}
	if got := m.Kernels[0].Instructions[0].Loc; got.FileIndex != 2 || got.Line != 4237 || got.Column != 0 {
		t.Errorf("Loc = %+v, want file 2 line 4237 col 0", got)
	}
}
