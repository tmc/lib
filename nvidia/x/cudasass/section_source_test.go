package cudasass_test

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// textThenMirror is a kernel's .text region followed by a header for the
// .nv.capmerc.text copy of the same kernel, spelled as nvdisasm spells a
// section: a .section directive, then the section's label. The listing is
// constructed: no nvdisasm output on hand prints a capmerc section. Its flags
// are "" because the section in testdata/roundtrip's cubin is neither
// allocated nor executable (sh_flags 0x10000000).
const textThenMirror = `	.target	sm_121a
	.section	.text.k,"ax",@progbits
.text.k:
        /*0000*/                   MOV R1, c[0x0][0x37c] ;
	.section	.nv.capmerc.text.k,"",@progbits
.nv.capmerc.text.k:
        /*0010*/                   EXIT ;
`

// nonTextCode holds every spelling of a code-section header that is not
// .text.<kernel>. Each must be refused: instructions under it are not the
// bytes measured to execute, and merging them into a kernel is silent.
var nonTextCode = []struct {
	name    string
	listing string
}{
	{"section directive and label", textThenMirror},
	{"label only", `	.section	.text.k,"ax",@progbits
.text.k:
        /*0000*/                   MOV R1, c[0x0][0x37c] ;
.nv.capmerc.text.k:
        /*0010*/                   EXIT ;
`},
	{"banner", `//--------------------- .text.k                   --------------------------
	.section	.text.k,"ax",@progbits
        /*0000*/                   MOV R1, c[0x0][0x37c] ;
//--------------------- .nv.capmerc.text.k        --------------------------
.nv.capmerc.text.k:
        /*0010*/                   EXIT ;
`},
	{"leading mirror", `	.section	.nv.capmerc.text.k,"",@progbits
        /*0000*/                   EXIT ;
`},
	{"banner only", `	.section	.text.k,"ax",@progbits
        /*0000*/                   MOV R1, c[0x0][0x37c] ;
//--------------------- .nv.capmerc.text.k        --------------------------
        /*0010*/                   EXIT ;
`},
	{"other section carrying a text name", `	.section	.text.k,"ax",@progbits
        /*0000*/                   MOV R1, c[0x0][0x37c] ;
	.section	.nv.merc.text.k,"",@progbits
        /*0010*/                   EXIT ;
`},
	{"section flags under a mirror", `	.section	.nv.capmerc.text.k,"",@progbits
	.sectionflags	@"SHF_BARRIERS=1"
        /*0000*/                   EXIT ;
`},
	{"flags after a space", `	.section	.text.k,"ax",@progbits
        /*0000*/                   MOV R1, c[0x0][0x37c] ;
	.section	.nv.other, "ax", @progbits
        /*0010*/                   EXIT ;
`},
	{"flags after a tab", "\t.section\t.text.k,\"ax\",@progbits\n" +
		"        /*0000*/                   MOV R1, c[0x0][0x37c] ;\n" +
		"\t.section\t.nv.other,\t\"ax\",@progbits\n" +
		"        /*0010*/                   EXIT ;\n"},
	{"text label under a mirror", `	.section	.text.k,"ax",@progbits
        /*0000*/                   MOV R1, c[0x0][0x37c] ;
	.section	.nv.capmerc.text.k,"",@progbits
.text.k:
        /*0010*/                   EXIT ;
`},
	{"code label under a mirror", `	.section	.text.k,"ax",@progbits
        /*0000*/                   MOV R1, c[0x0][0x37c] ;
	.section	.nv.capmerc.text.k,"",@progbits
.L_x_0:
        /*0010*/                   EXIT ;
`},
	{"cuobjdump header under a mirror", `	.section	.nv.capmerc.text.k,"",@progbits
Function : k
        /*0000*/                   EXIT ;
`},
	{"executable section not named .text", `	.section	.text.k,"ax",@progbits
        /*0000*/                   MOV R1, c[0x0][0x37c] ;
	.section	.nv.other,"ax",@progbits
        /*0010*/                   EXIT ;
`},
}

func TestParseRefusesNonTextCodeSection(t *testing.T) {
	for _, tt := range nonTextCode {
		t.Run(tt.name, func(t *testing.T) {
			d, err := cudasass.ParseString(tt.listing)
			if err == nil {
				for _, fn := range d.Functions {
					t.Logf("merged: function %q, %d instructions, labels %v", fn.Name, len(fn.Instructions), fn.Labels)
				}
				t.Fatal("ParseString accepted instructions under a code section that is not .text.<kernel>")
			}
			if !errors.Is(err, cudasass.ErrNotTextSection) {
				t.Fatalf("ParseString error = %v, want one wrapping ErrNotTextSection", err)
			}
			if !strings.Contains(err.Error(), ".nv.") {
				t.Errorf("error %q does not name the section", err)
			}
		})
	}
}

// TestParseAcceptsDataSections is the control for the refusal: nvdisasm
// without -c also prints the non-executable sections of a kernel, and those
// carry no instructions to misattribute. A capmerc header followed only by
// data is one of them; only an instruction under it is refused.
func TestParseAcceptsDataSections(t *testing.T) {
	tests := []struct {
		name    string
		listing string
	}{
		{"info, constant and relocation", `	.section	.nv.info.k,"",@"SHT_CUDA_INFO"
	.section	.text.k,"ax",@progbits
.text.k:
        /*0000*/                   EXIT ;
	.section	.nv.constant0.k,"a",@progbits
	.zero		896
	.section	.rela.text.k,"",@"SHT_RELA"
`},
		{"demangled data section", `//--------------------- .nv.constant0.k(float *, index_t) --------------------------
	.section	.nv.constant0.k(float *, index_t),"a",@progbits
	.align	4
.nv.constant0.k(float *, index_t):
	.zero		908
	.section	.text.k,"ax",@progbits
        /*0000*/                   EXIT ;
`},
		{"demangled info section", `	.section	.nv.info.k(float *, Matrix),"",@"SHT_CUDA_INFO"
	.section	.text.k,"ax",@progbits
        /*0000*/                   EXIT ;
`},
		{"capmerc header over data", `//--------------------- .text.k                   --------------------------
	.section	.text.k,"ax",@progbits
.text.k:
        /*0000*/                   EXIT ;
//--------------------- .nv.capmerc.text.k        --------------------------
	.section	.nv.capmerc.text.k,"",@"SHT_CUDA_CAPMERC"
.nv.capmerc.text.k:
        /*0000*/ 	.byte	0x01, 0x02, 0x03, 0x04
`},
		{"capmerc header over data, then the next kernel", `	.section	.nv.capmerc.text.j,"",@"SHT_CUDA_CAPMERC"
.nv.capmerc.text.j:
        /*0000*/ 	.byte	0x01, 0x02, 0x03, 0x04
	.section	.text.k,"ax",@progbits
        /*0000*/                   EXIT ;
`},
		{"quoted name and type", `	.section	".nv.x",@progbits
	.section	.text.k,"ax",@progbits
        /*0000*/                   EXIT ;
`},
		{"section flags and info", `	.section	.text.k,"ax",@progbits
	.sectionflags	@"SHF_BARRIERS=1"
	.sectioninfo	@"SHI_REGISTERS=24"
.text.k:
        /*0000*/                   EXIT ;
`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d, err := cudasass.ParseString(tt.listing)
			if err != nil {
				t.Fatalf("ParseString: %v", err)
			}
			if len(d.Functions) != 1 || len(d.Functions[0].Instructions) != 1 {
				t.Fatalf("ParseString: %d functions, want 1 with 1 instruction", len(d.Functions))
			}
			if got := d.Functions[0].ELFSection; !strings.HasPrefix(got, ".text.k") {
				t.Errorf("ELFSection = %q, want the .text section", got)
			}
		})
	}
}

// TestParseNamesSection checks that the refusal names the section and only
// the section: a demangled name holds commas and spaces, and the flags are
// the quoted field after it, with or without space before the quote.
func TestParseNamesSection(t *testing.T) {
	const demangled = ".nv.capmerc.text.k(float *, index_t)"
	tests := []struct {
		listing string
		name    string
	}{
		{"\t.section\t" + demangled + ",\"\",@progbits\n /*0000*/ EXIT ;\n", demangled},
		{"//--------------------- " + demangled + " --------------------------\n /*0000*/ EXIT ;\n", demangled},
		{"\t.section\t.nv.capmerc.text.k, \"\", @progbits\n /*0000*/ EXIT ;\n", ".nv.capmerc.text.k"},
		{"\t.section\t" + demangled + ",\t\"\",@progbits\n /*0000*/ EXIT ;\n", demangled},
	}
	for _, tt := range tests {
		_, err := cudasass.ParseString(tt.listing)
		if !errors.Is(err, cudasass.ErrNotTextSection) || !strings.Contains(err.Error(), "section "+tt.name+": ") {
			t.Errorf("ParseString(%q) = %v, want ErrNotTextSection naming %q", tt.listing, err, tt.name)
		}
	}
}

// TestParseOtherErrorsAreNotTextSectionErrors keeps the refusal
// distinguishable: a listing rejected for another reason must not satisfy
// errors.Is(err, ErrNotTextSection).
func TestParseOtherErrorsAreNotTextSectionErrors(t *testing.T) {
	for _, listing := range []string{
		".text.k:\n /*0001*/ NOP ;\n",
		".text.k:\n /*0010*/ NOP ;\n /*0000*/ EXIT ;\n",
		" /*0000*/ NOP ;\n",
	} {
		_, err := cudasass.ParseString(listing)
		if err == nil {
			t.Fatalf("ParseString(%q) succeeded, want an error", listing)
		}
		if errors.Is(err, cudasass.ErrNotTextSection) {
			t.Errorf("ParseString(%q) = %v, which wraps ErrNotTextSection", listing, err)
		}
	}
}

func TestELFSection(t *testing.T) {
	tests := []struct {
		name    string
		listing string
		want    string
	}{
		{"section directive", "\t.section\t.text.k,\"ax\",@progbits\n /*0000*/ EXIT ;\n", ".text.k"},
		{"label", ".text.k:\n /*0000*/ EXIT ;\n", ".text.k"},
		{"cuobjdump", "Function : k\n /*0000*/ EXIT ;\n", ""},
		{"project-local", "--- Kernel: k ---\n /*0000*/ EXIT ;\n", ""},
		{"cuobjdump then label", "Function : k\n.text.k:\n /*0000*/ EXIT ;\n", ".text.k"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d, err := cudasass.ParseString(tt.listing)
			if err != nil {
				t.Fatalf("ParseString: %v", err)
			}
			fn := d.Function("k")
			if fn == nil {
				t.Fatal("no function k")
			}
			if fn.ELFSection != tt.want {
				t.Errorf("Function(k).ELFSection = %q, want %q", fn.ELFSection, tt.want)
			}
			secs := d.Sections()
			if len(secs) != 1 || secs[0].ELFSection != tt.want {
				t.Errorf("Sections() = %+v, want one with ELFSection %q", secs, tt.want)
			}
		})
	}
}

// TestFixtureReadText ties the committed listing to its cubin: the cubin
// carries a .nv.capmerc.text mirror of each kernel, and the listing nvdisasm
// printed from it was read from .text.
func TestFixtureReadText(t *testing.T) {
	im, err := cudasass.OpenCubin(readRoundTripCubin(t))
	if err != nil {
		t.Fatal(err)
	}
	listing, err := os.ReadFile(roundTripFixture)
	if err != nil {
		t.Fatal(err)
	}
	d, err := cudasass.ParseString(string(listing))
	if err != nil {
		t.Fatalf("ParseString(%s): %v", roundTripFixture, err)
	}
	for _, kernel := range []string{"addk", "intmix"} {
		if got, want := im.Mirrors(kernel), []string{".nv.capmerc.text." + kernel}; !slices.Equal(got, want) {
			t.Errorf("Mirrors(%s) = %q, want %q", kernel, got, want)
		}
		fn := d.Function(kernel)
		if fn == nil {
			t.Fatalf("listing has no function %s", kernel)
		}
		if want := cudasass.TextSectionName(kernel); fn.ELFSection != want {
			t.Errorf("Function(%s).ELFSection = %q, want %q", kernel, fn.ELFSection, want)
		}
	}
}

func ExampleParse_textSection() {
	d, err := cudasass.ParseString(`	.section	.text.k,"ax",@progbits
        /*0000*/                   EXIT ;
`)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(d.Function("k").ELFSection)

	_, err = cudasass.ParseString(`	.section	.nv.capmerc.text.k,"",@progbits
        /*0000*/                   EXIT ;
`)
	fmt.Println(errors.Is(err, cudasass.ErrNotTextSection))
	// Output:
	// .text.k
	// true
}

// demangledFixture is nvdisasm -c output passed through cu++filt, as
// ../cudalowering/testdata/gen.sh produces it from one cubin, beside
// mangledFixture, the same disassembly before cu++filt.
const (
	demangledFixture = "../cudalowering/testdata/lowering.sm90.demangled.sass"
	mangledFixture   = "../cudalowering/testdata/lowering.sm90.sass"
)

// TestParseDemangledFixture checks that a kernel's name is the whole of its
// demangled header, spaces and commas included, and that its ELF section is
// the one the listing names. The oracle for the instructions is the mangled
// disassembly of the same cubin: each kernel must hold as many there.
func TestParseDemangledFixture(t *testing.T) {
	want := []string{
		"atomicDrop(float *, const float *, int)",
		"atomicKeep(float *, const float *, int)",
		"shuffleReduce(float *, const float *, int)",
		"spiller(float *, const float *, int)",
		"fmaChain(double *, const double *, const double *, int)",
	}
	demangled := parseFile(t, demangledFixture)
	mangled := parseFile(t, mangledFixture)
	if len(demangled.Functions) != len(want) || len(mangled.Functions) != len(want) {
		t.Fatalf("got %d demangled and %d mangled functions, want %d of each",
			len(demangled.Functions), len(mangled.Functions), len(want))
	}
	secs := demangled.Sections()
	if len(secs) != len(want) {
		t.Fatalf("got %d sections, want %d", len(secs), len(want))
	}
	for i, name := range want {
		fn := demangled.Functions[i]
		if fn.Name != name {
			t.Errorf("function %d is %q, want %q", i, fn.Name, name)
		}
		if got, want := fn.ELFSection, ".text."+name; got != want {
			t.Errorf("function %d ELFSection = %q, want %q", i, got, want)
		}
		if secs[i].Name != name || secs[i].ELFSection != ".text."+name {
			t.Errorf("section %d = %q in %q, want %q in %q", i, secs[i].Name, secs[i].ELFSection, name, ".text."+name)
		}
		if got, want := len(fn.Instructions), len(mangled.Functions[i].Instructions); got != want || got == 0 {
			t.Errorf("%s: %d instructions, want %d as in %s", name, got, want, mangled.Functions[i].Name)
		}
	}
}

func parseFile(t *testing.T, name string) *cudasass.Disassembly {
	t.Helper()
	src, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	d, err := cudasass.ParseString(string(src))
	if err != nil {
		t.Fatalf("ParseString(%s): %v", name, err)
	}
	return d
}

// Two overloads whose demangled names share their first word, spelled as
// cu++filt spells nvdisasm's headers. The listings are constructed: no
// committed disassembly has overloads or a qualified demangled name.
const (
	overloadF = "scale(float *, int)"
	overloadD = "scale(float *, double)"
	qualified = "ns::scale(float *, int)"
)

// overloadListing returns a two-kernel listing, with the live-register table
// nvdisasm --life-range-mode count appends to each line when life is set.
func overloadListing(life bool) string {
	col := func(gpr string) string {
		if !life {
			return ""
		}
		return "        // |" + gpr + "|        |        |"
	}
	var b strings.Builder
	for _, k := range []struct{ name, gpr1, gpr2 string }{
		{overloadF, "   1   ", "   2   "},
		{overloadD, "   5   ", "   6   "},
	} {
		fmt.Fprintf(&b, "//--------------------- .text.%s --------------------------\n", k.name)
		fmt.Fprintf(&b, "\t.section\t.text.%s,\"ax\",@progbits\n", k.name)
		fmt.Fprintf(&b, "%s:%s\n", k.name, col("       "))
		fmt.Fprintf(&b, ".text.%s:%s\n", k.name, col("       "))
		fmt.Fprintf(&b, "        /*0000*/                   LDC R1, c[0x0][0x28] ;%s\n", col(k.gpr1))
		fmt.Fprintf(&b, "        /*0010*/                   EXIT ;%s\n", col(k.gpr2))
	}
	return b.String()
}

// TestParseDemangledHeaders checks the header spellings one at a time: a
// kernel is found under its whole demangled name whether a directive, a label
// or both introduce it, and two kernels that share a first word stay two.
func TestParseDemangledHeaders(t *testing.T) {
	tests := []struct {
		name    string
		listing string
		want    []string
	}{
		{"overloads", overloadListing(false), []string{overloadF, overloadD}},
		{"directive only", "\t.section\t.text." + qualified + ",\"ax\",@progbits\n /*0000*/ EXIT ;\n", []string{qualified}},
		{"label only", ".text." + qualified + ":\n /*0000*/ EXIT ;\n", []string{qualified}},
		{"directive then label", "\t.section\t.text." + qualified + ",\"ax\",@progbits\n.text." + qualified + ":\n /*0000*/ EXIT ;\n", []string{qualified}},
		{"label with life columns", ".text." + qualified + ":        // |       |        |\n /*0000*/ EXIT ;\n", []string{qualified}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d, err := cudasass.ParseString(tt.listing)
			if err != nil {
				t.Fatalf("ParseString: %v", err)
			}
			var got []string
			for _, fn := range d.Functions {
				got = append(got, fn.Name)
				if want := ".text." + fn.Name; fn.ELFSection != want {
					t.Errorf("%q: ELFSection = %q, want %q", fn.Name, fn.ELFSection, want)
				}
			}
			if !slices.Equal(got, tt.want) {
				t.Fatalf("functions = %q, want %q", got, tt.want)
			}
			if n := len(d.Sections()); n != len(tt.want) {
				t.Errorf("%d sections, want %d", n, len(tt.want))
			}
			for _, fn := range d.Functions {
				if fn.Instructions[len(fn.Instructions)-1].Opcode != "EXIT" || len(fn.Instructions) > 2 {
					t.Errorf("%q holds %d instructions, want its own", fn.Name, len(fn.Instructions))
				}
			}
		})
	}
}

// TestParseLabelComment checks that a comment after a ".text.NAME:" label is
// not part of the name, whatever its form and whether or not it holds a
// colon, while a demangled name keeps its own colons and slashes. A label
// that read its comment as part of the name would open a second kernel and
// leave the kernel its directive opened with no instructions.
func TestParseLabelComment(t *testing.T) {
	const life = "        // |   3   |        |        |"
	tests := []struct {
		name  string
		label string // the kernel's label line, comment included
		want  string
	}{
		{"block comment", ".text.k:\t/* c */", "k"},
		{"block comment no space", ".text.k:/* c */", "k"},
		{"line comment", ".text.k: // c", "k"},
		{"block then line comment", ".text.k: /* a */ // b", "k"},
		{"two block comments", ".text.k: /* a */ /* b */", "k"},
		{"block comment with colon", ".text.k: /* a: b */", "k"},
		{"line comment with colon", ".text.k: // a: b", "k"},
		{"trailing blanks", ".text.k:  \t", "k"},
		{"qualified with block comment", ".text." + qualified + ":\t/* c */", qualified},
		{"qualified with line comment colon", ".text." + qualified + ": // a: b", qualified},
		{"slash in name", ".text.operator/(a, b):", "operator/(a, b)"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			directive := "\t.section\t.text." + tt.want + ",\"ax\",@progbits\n"
			body := tt.label + "\n        /*0000*/                   EXIT ;" + life + "\n"
			for _, listing := range []string{directive + body, body} {
				d, err := cudasass.ParseString(listing)
				if err != nil {
					t.Fatalf("ParseString(%q): %v", listing, err)
				}
				var got []string
				for _, fn := range d.Functions {
					got = append(got, fmt.Sprintf("%s/%d/%s", fn.Name, len(fn.Instructions), fn.ELFSection))
				}
				want := []string{tt.want + "/1/.text." + tt.want}
				if !slices.Equal(got, want) {
					t.Errorf("ParseString(%q): name/instructions/section = %q, want %q", listing, got, want)
				}
				if n := len(d.Sections()); n != 1 {
					t.Errorf("ParseString(%q): %d sections, want 1", listing, n)
				}

				press, err := cudasass.ParseLifeRangeCount(strings.NewReader(listing))
				if err != nil {
					t.Fatalf("ParseLifeRangeCount(%q): %v", listing, err)
				}
				wantPress := map[string]map[uint64]cudasass.RegisterPressure{tt.want: {0x0: {GPR: 3}}}
				if fmt.Sprint(press) != fmt.Sprint(wantPress) {
					t.Errorf("ParseLifeRangeCount(%q) = %v, want %v", listing, press, wantPress)
				}
			}
		})
	}
}

// TestParseLabelWithoutColon checks that a ".text." line that is not a
// well-formed label still begins a kernel, rather than letting the kernel's
// instructions merge into the kernel before it.
func TestParseLabelWithoutColon(t *testing.T) {
	tests := []struct {
		name  string
		label string
		want  string
	}{
		{"no colon", ".text.k", "k"},
		{"no colon with line comment", ".text.k // c", "k"},
		{"qualified no colon", ".text." + qualified, qualified},
		{"no colon, line comment with colon", ".text.k // a: b", "k"},
		{"no colon, line comment with block comment", ".text.k // a /* b", "k"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			listing := ".text.j:\n /*0000*/ EXIT ;\n" + tt.label + "\n /*0010*/ EXIT ;\n"
			d, err := cudasass.ParseString(listing)
			if err != nil {
				t.Fatalf("ParseString: %v", err)
			}
			var got []string
			for _, fn := range d.Functions {
				got = append(got, fmt.Sprintf("%s/%d", fn.Name, len(fn.Instructions)))
			}
			if want := []string{"j/1", tt.want + "/1"}; !slices.Equal(got, want) {
				t.Fatalf("name/instructions = %q, want %q", got, want)
			}
		})
	}
}

// TestParseRefusesMalformedLabel checks that Parse refuses a ".text." line
// whose name breaks one of the five clauses of the label rule (see Parse),
// rather than read the rest of the line into a kernel name and move the
// instruction after it there. The listings are constructed: every ".text."
// line in the committed disassemblies is ".text.NAME:", alone or followed by
// a "//" comment. Constructed lines the rule accepts are in acceptedByRule.
func TestParseRefusesMalformedLabel(t *testing.T) {
	tests := []struct {
		name  string
		label string
	}{
		{"nested block comment", ".text.k: /* a /* b */ */"},
		{"unterminated block comment", ".text.k: /* c"},
		{"text after colon", ".text.k: junk"},
		{"text after colon and tab", ".text.k:\tjunk"},
		{"text after block comment", ".text.k: /* c */ junk"},
		{"unterminated comment without colon", ".text.k /* c"},
		{"block comment without colon", ".text.k /* c */"},
		{"stray comment close", ".text.k */"},
		{"text right after colon", ".text.k:junk"},
		{"text after colon and NBSP", ".text.k:\u00a0junk"},
		{"text after colon and VT", ".text.k:\vjunk"},
		{"text after blank and colon", ".text.k : junk"},
		{"block comment before terminal colon", ".text.k /* c */:"},
		{"block comment before terminal colon, no blank", ".text.k/* c */:"},
		{"text after colon, then terminal colon", ".text.k: junk:"},
		{"colon outside closed parentheses", ".text.k(int): junk"},
		{"colon after stray close", ".text.k): junk"},
		{"colon in unclosed parenthesis", ".text.k(: junk"},
		{"colon in unclosed angle bracket", ".text.operator<(A, B): junk"},
		{"single colon in qualifier", ".text.ns:k(int)"},
		{"triple colon in qualifier", ".text.ns:::k(int)"},
		{"colon in parentheses", ".text.k (a: b)"},
		{"colon in angle brackets", ".text.k<x: junk>"},
		{"colon first in parentheses", ".text.k(: junk)"},
		{"colon last in parentheses", ".text.k (junk:)"},
		{"colon in closed angle brackets before an open parenthesis", ".text.k<a:b>("},
		{"conditional colon outside its depth", ".text.k(a ? b [c) : d"},
		{"colon after mismatched closer", ".text.k(a]: junk)"},
		{"colon after conditional closed", ".text.k(a ? 1 : 2): junk"},
		{"second colon of one conditional", ".text.f<a ? 1 : 2 : 3>()"},
		{"abi tag after blank", ".text.h[ abi:cxx11]()"},
		{"auto without digit", ".text.f(auto:x)"},
		{"colon in braces", ".text.{a: b}"},
		{"conditional colon after a less-than and its depth closed", ".text.k(a ? b < c) : d"},
		{"digit after colon", ".text.k:1junk"},
		{"conditional with no middle operand", ".text.k ? : junk"},
		{"conditional with no middle operand in parentheses", ".text.(k ? : junk)"},
		{"operator?:", ".text.operator?: junk"},
		{"blank in ABI tag", ".text.k[abi: junk]"},
		{"unclosed ABI tag", ".text.k[abi:junk"},
		{"auto placeholder after an identifier", ".text.kauto:1junk"},
		{"auto placeholder after an identifier in parentheses", ".text.f(kauto:1)"},
		{"auto placeholder then identifier", ".text.f(auto:1junk)"},
		{"auto placeholder without digits", ".text.f(auto:)"},
		{"empty ABI tag", ".text.k[abi:]"},
		{"conditional colon after an unspaced less-than and its depth closed", ".text.k(a ? b<c) : d"},
		{"question in an unspaced less-than closed by a parenthesis", ".text.decltype(a<b ? 1) : 2 f()"},
		{"latest question has no middle operand", ".text.f(a ? b ? : c)"},
		{"auto placeholder then dollar", ".text.f(auto:1$x)"},
		{"question in an angle bracket closed by a square bracket", ".text.k[a<b ? c) : d]"},
		{"conditional in a parenthesis left open", ".text.k(a ? b : c"},
		{"conditional in a parenthesis closed by a square bracket", ".text.f(a ? b : c]"},
		{"ABI tag in a parenthesis", ".text.k(abi:x])"},
		{"unterminated string literal", `.text.f<L8{"a`},
		{"unterminated character literal", `.text.f<'a>()`},
		{"single quote left open", `.text.k'x`},
		{"escaped backslash ends the literal", `.text.f<L8{"a\\"b"}>()`},
		{"square bracket closes a parenthesis", ".text.f(a]"},
		{"parenthesis left open", ".text.f(a"},
		{"closing parenthesis outside every bracket", ".text.f)"},
		{"parenthesis closes a brace", ".text.f{a)"},
		{"brace closes a square bracket", ".text.f[a}"},
		{"triple colon after a conditional", ".text.f(a ? b :::c)"},
		{"ABI tag holding a blank", ".text.f[abi:x y]"},
		{"ABI tag in parentheses", ".text.f(abi:x)"},
		{"ABI tag after an identifier byte", ".text.f[xabi:x]"},
		{"auto placeholder at the end", ".text.auto:1"},
		{"auto placeholder after an underscore", ".text.f(_auto:1)"},
		{"auto placeholder after a dollar", ".text.f($auto:1)"},
		{"auto placeholder after a digit", ".text.f(1auto:1)"},
		{"conditional with only NBSP between", ".text.k ?\u00a0: b"},
		{"question closed in parentheses before the colon", ".text.(a ? b) : c"},
		{"colon in parentheses opened after the question", ".text.a ? (b : c)"},
		{"third operand colon", ".text.f(a ? b : c : d)"},
		{"comment open sharing a slash", ".text.k/*/"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			directive := "\t.section\t.text.k,\"ax\",@progbits\n"
			body := tt.label + "\n        /*0000*/                   EXIT ;\n"
			for _, listing := range []string{directive + body, body} {
				d, err := cudasass.ParseString(listing)
				if err == nil {
					var got []string
					for _, fn := range d.Functions {
						got = append(got, fmt.Sprintf("%q/%d/%q", fn.Name, len(fn.Instructions), fn.ELFSection))
					}
					t.Fatalf("ParseString(%q) accepted a malformed label: name/instructions/section = %s", listing, got)
				}
				if d != nil {
					t.Errorf("ParseString(%q) = %v and a non-nil Disassembly holding %d functions", listing, err, len(d.Functions))
				}
				if errors.Is(err, cudasass.ErrNotTextSection) {
					t.Errorf("ParseString(%q) = %v, which wraps ErrNotTextSection", listing, err)
				}
				if !strings.Contains(err.Error(), fmt.Sprintf("%q", tt.label)) {
					t.Errorf("ParseString(%q) = %v, want an error naming the label", listing, err)
				}
			}
		})
	}
}

// acceptedByRule holds constructed ".text." lines, no C++ name among them,
// that the label rule (see Parse) accepts, each with the clause that accepts
// it. Several were refused by earlier heuristics; the rule accepts them
// because '<' and '>' are not brackets, every "::" is accepted, a
// conditional pairs at any ()[]{} depth, and "[abi:" and "auto:N" carry no
// condition on what precedes or encloses them. Each names a kernel by its
// whole label name, which under a .section directive naming another kernel
// opens a second kernel holding the instruction.
var acceptedByRule = []struct {
	label, want, reason string
}{
	{".text.k:: junk", "k:: junk", "3: every \"::\" is accepted"},
	{".text.k::\tjunk", "k::\tjunk", "3: every \"::\" is accepted"},
	{".text.k::\u00a0junk", "k::\u00a0junk", "3: every \"::\" is accepted"},
	{".text.k:::", "k::", "3: the last colon ends the label; the name's \"::\" is accepted"},
	{".text.k ::junk", "k ::junk", "3: every \"::\" is accepted"},
	{".text.k\t::junk", "k\t::junk", "3: every \"::\" is accepted"},
	{".text.k\u00a0::junk", "k\u00a0::junk", "3: every \"::\" is accepted"},
	{".text.k, ::junk", "k, ::junk", "3: every \"::\" is accepted"},
	{".text.f(a::)", "f(a::)", "3: every \"::\" is accepted"},
	{".text.f(a::1)", "f(a::1)", "3: every \"::\" is accepted"},
	{".text.f(a::, b)", "f(a::, b)", "3: every \"::\" is accepted"},
	{".text.f(a ? b ::c)", "f(a ? b ::c)", "3: every \"::\" is accepted; the '?' stays unpaired"},
	{".text.k ? f<a : b>", "k ? f<a : b>", "2, 4c: '<' opens nothing, so the colon pairs with the '?' at depth 0"},
	{".text.k ? f<a : b>()", "k ? f<a : b>()", "2, 4c: '<' opens nothing, so the colon pairs with the '?' at depth 0"},
	{".text.(k ? f<a : b>)", "(k ? f<a : b>)", "2, 4c: '<' opens nothing, so the colon pairs with the '?' in the parentheses"},
	{".text.(a ? k<b : c)", "(a ? k<b : c)", "2, 4c: '<' opens nothing, so the colon pairs with the '?' in the parentheses"},
	{".text.f(a ? g<b : c> d)", "f(a ? g<b : c> d)", "2, 4c: '<' opens nothing, so the colon pairs with the '?' in the parentheses"},
	{".text.k<x ? a : b", "k<x ? a : b", "2, 4c: '<' opens nothing, so nothing is left open"},
	{".text.k<y<x ? a : b", "k<y<x ? a : b", "2, 4c: '<' opens nothing, so nothing is left open"},
	{".text.k < x ? a : b", "k < x ? a : b", "4c: a conditional pairs at depth 0"},
	{".text.k ? junk : junk", "k ? junk : junk", "4c: a conditional pairs at depth 0"},
	{".text.a?b:c", "a?b:c", "4c: a conditional pairs at depth 0; no blank is required"},
	{".text.f<a ? b>: junk", "f<a ? b>: junk", "2, 4c: '>' closes nothing, so the colon pairs with the '?'"},
	{".text.f<a ? xoperator>: b", "f<a ? xoperator>: b", "2, 4c: '>' closes nothing, so the colon pairs with the '?'"},
	{".text.f(g<a ? xoperator> : b)", "f(g<a ? xoperator> : b)", "2, 4c: '>' closes nothing, so the colon pairs with the '?'"},
	{".text.f<a>= b ? 1 : 2>()", "f<a>= b ? 1 : 2>()", "2, 4c: '>' closes nothing, so the colon pairs with the '?'"},
	{".text.void f<1 >> 2, b ? 1 : 2>()", "void f<1 >> 2, b ? 1 : 2>()", "2, 4c: \">>\" closes nothing, so the colon pairs with the '?'"},
	{".text.f(a ? b ? c : d : e)", "f(a ? b ? c : d : e)", "4c: each colon pairs with the latest unpaired '?'"},
	{".text.f(a ?\tb : c)", "f(a ?\tb : c)", "4c: text other than white space lies between '?' and ':'"},
	{".text.k [abi:cxx11]", "k [abi:cxx11]", "4a: nothing is required before \"[abi:\""},
	{".text.[abi:cxx11]", "[abi:cxx11]", "4a: nothing is required before \"[abi:\""},
	{".text.f[abi:x$]", "f[abi:x$]", "4a: '$' is an identifier byte"},
	{".text.f<auto:1>()", "f<auto:1>()", "2, 4b: \"auto:N\" needs no enclosing bracket"},
	{".text.f<int, auto:1>()", "f<int, auto:1>()", "2, 4b: \"auto:N\" needs no enclosing bracket"},
	{".text.auto:1 k", "auto:1 k", "4b: \"auto\" at the start of the name begins a token"},
	{".text.f(auto:12)", "f(auto:12)", "4b: N is one or more digits"},
	{`.text.f<L8{"a:b"}>()`, `f<L8{"a:b"}>()`, "1: nothing inside a literal counts"},
	{`.text.f(L8{"(:"})`, `f(L8{"(:"})`, "1: nothing inside a literal counts"},
	{`.text.f<L8{"a\"b:c"}>()`, `f<L8{"a\"b:c"}>()`, "1: a backslash escapes the quote after it"},
	{`.text.f('a\'b:')`, `f('a\'b:')`, "1: a backslash escapes the quote after it"},
	{`.text.k"x"`, `k"x"`, "1: a literal may follow any byte"},
	{`.text.f<L8{"a//b"}>():`, `f<L8{"a//b"}>()`, "1, 5: \"//\" inside a literal before a terminal colon counts for nothing"},
}

// TestParseAcceptedByRule checks that each line in acceptedByRule names one
// kernel by its whole label name, owning the instruction after it, with and
// without a .section directive naming that kernel. A directive's name drops
// one trailing colon, so a name ending in a colon is checked without one.
func TestParseAcceptedByRule(t *testing.T) {
	for _, tt := range acceptedByRule {
		directive := "\t.section\t.text." + tt.want + ",\"ax\",@progbits\n"
		body := tt.label + "\n        /*0000*/                   EXIT ;\n"
		listings := []string{directive + body, body}
		if strings.HasSuffix(tt.want, ":") {
			listings = listings[1:]
		}
		for _, listing := range listings {
			d, err := cudasass.ParseString(listing)
			if err != nil {
				t.Errorf("ParseString(%q) (%s): %v", listing, tt.reason, err)
				continue
			}
			var got []string
			for _, fn := range d.Functions {
				got = append(got, fmt.Sprintf("%s/%d/%s", fn.Name, len(fn.Instructions), fn.ELFSection))
			}
			if want := []string{tt.want + "/1/.text." + tt.want}; !slices.Equal(got, want) {
				t.Errorf("ParseString(%q) (%s): name/instructions/section = %q, want %q", listing, tt.reason, got, want)
			}
		}
	}
}

// Demangled names holding colons. The first twelve are spelled as the
// c++filt on a review host (Apple LLVM 21) printed the demangled symbol,
// including a conditional's colon at the top of a template argument list
// after an operator name whose symbol holds '>'. The rest are constructed in the same style: scope operators,
// conditionals in decltype, template arguments, array bounds and nested
// conditionals, ABI tags, generic lambda placeholders, operator names,
// clone suffixes and mangled names. No committed disassembly holds one.
var colonExpressionNames = []string{
	"void f<&operator>>(A&, A), true ? 1 : 2>()",
	"void f<&A::operator->(), true ? 1 : 2>()",
	"void f<&operator>=(A const&, A const&), true ? 1 : 2>()",
	"void f<&operator>(A const&, A const&), true ? 1 : 2>()",
	"void f<(1 >> 2), true ? 1 : 2>()",
	"void f<(1 > 2), true ? 1 : 2>()",
	"decltype(true ? 1 : 2) f<&operator>(A const&, A const&)>()",
	"void f<&operator>(A const&, A const&)>(int (*) [true ? 1 : 2])",
	"void f<&operator<(A const&, A const&), true ? 1 : 2>()",
	"void f<true ? 1 : 2, &operator>(A const&, A const&)>()",
	"void f<int>(int (*) [int ? 1 : 2])",
	"decltype((1 > 1) ? 2 : 3) f<1>()",
	"void f<1>(int (*) [(1 > 0) ? 1 : 2])",
	"decltype(fp ? 1 : 2) f<int>(int)",
	"decltype(fp ? 1:2) f<int>(int)",
	"decltype(fp ? 1 > 2 : 3) f()",
	"decltype(fp ? 1 < 2 : 3) f()",
	"X<b ? &operator> : &operator<>()",
	"X<b ? &A::operator-> : 0>()",
	"X<b ? &operator>> : 0>()",
	"X<b ? &operator>= : 0>()",
	"X<b ? &operator<=> : 0>()",
	"X<b ? &operator>>= : 0>()",
	"X<b ? &operator->* : 0>()",
	"tag<(a ? 1 : 2)>()",
	"A<(1)?(2) : (3)> f<1>()",
	"ns::g(A<(1)?(2) : (3)>, int)",
	"decltype(((a)<(b)) ? 1 : 2) cmp<int>(int)",
	"bool operator< <(a ? 1 : 2)>(A, A)",
	"bool operator< <true ? 1 : 2>(A, A)",
	"bool operator<  <true ? 1 : 2>(A, A)",
	"bool operator<\u00a0<true ? 1 : 2>(A, A)",
	"decltype(a\u00a0< b ? 1 : 2) f()",
	"X<sizeof (int (*) [a<b]) ? 1 : 2> f()",
	"S<a >>= b ? 1 : 2> f()",
	"S<a <<= b ? 1 : 2> f()",
	"S<a <=> b ? 1 : 2> f()",
	"k<a ::b",
	"S<my_operator<int>, a ? 1 : 2> f()",
	"ns::(anonymous namespace)::f()",
	"void f<1 < 2 ? 3 : 4>()",
	"void f<(1 > 2) ? 3 : 4>()",
	"decltype(int ? 1 : 2) f<int>(decltype(int ? 1 : 2))",
	"decltype(a ? b ? 1 : 2 : 3) f()",
	"f<(a ? (b ? 1 : 2) : 3)>()",
	"void f<a ? b ? 1 : 2 : 3>()",
	"decltype(fp ? 1 : 2) f<int>(int) [clone .constprop.0]",
	"X<&operator<, true ? 1 : 2>()",
	"X<&operator>>, true ? 1 : 2>()",
	"X<&operator->, b ? 1 : 2>()",
	"X<&operator>=, b ? 1 : 2>()",
	"X<&operator->*, b ? 1 : 2>()",
	"X<&operator<=>, b ? 1 : 2>()",
	"X<&operator<<, b ? 1 : 2>()",
	"X<&operator<=, b ? 1 : 2>()",
	"X<&operator(), b ? 1 : 2>()",
	"X<&operator[], b ? 1 : 2>()",
	"h[abi:cxx11](int)",
	"ns::h[abi:cxx11][abi:foo]()",
	"operator<[abi:cxx11](A, B)",
	"operator>[abi:cxx11](A, B)",
	"f<int>[abi:cxx11]()",
	"A<(1 > 2)>::f[abi:x]()",
	"main::{lambda(auto:1)#1}::operator()<int>(int) const",
	"f(auto:1, auto:2)",
	"f<int, ::ns::T>()",
	"auto main::'lambda'(auto)::operator()<int>(auto) const",
	"operator<<(A&, A&)",
	"operator>>(A&, A&)",
	"operator>(A&, A&)",
	"A::operator->()",
	"A::operator()()",
	"operator<=>(A const&, A const&)",
	"A::operator->*(int)",
	"A::operator bool() const",
	"bool operator< <int>(X<int> const&, X<int> const&)",
	"{lambda(int)#1}::operator()(int) const",
	"main::{lambda(int)#1}::operator()(int) const",
	"main::'lambda'(int)::operator()(int) const",
	"k(int) const::{lambda()#1}::operator()() const",
	"(anonymous namespace)::k(float*)",
	"k(float*) [clone .constprop.0]",
	"k(float*) (.constprop.0)",
	"k(float*) [clone .isra.0] [clone .cold]",
	"std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::size() const",
	"std::_Bind<void (*(int))(int)>::operator()()",
	"void f<ns::A, ns::B<int> >(ns::A)",
	"void f<&ns::g>()",
	"f<::ns::T>()",
	"::k()",
	"int A::* f(int A::*)",
	"_Z6kernelPf",
	"_ZN2ns1kEi",
}

// cxxfiltNames and llvmNames are demangled names, one per line, as the
// LLVM c++filt on a review host (Apple LLVM 21, with Apple clang 21
// compiling constructed declarations) printed them: operator names,
// conditionals in decltype, template arguments and array bounds, "->",
// "->*" and " >= " in template arguments, scope operators after blanks
// inside brackets, and ABI tags. cxxfiltNames holds 307 names; llvmNames
// holds 139 lines, 130 distinct names, of which 11 were refused by an
// earlier version of this parser. GNU c++filt spellings are untested. Each
// SHA-256 pins its list, so an edit to one is deliberate.
const (
	cxxfiltNames = `__ZZ3usevENK3$_0clIiiEEDaT_T0_
B::operator int*<int>() const
bool B::operator<<int>(int) const
bool B::operator><int>(int) const
bool operator<<int>(S<1>, int)
decltype(::delete fp, 0) dt4<int>(int*)
decltype(::new int, 0) dt3<int>(int*)
decltype(((fp...) + ...) ? 1 : 2) dt26<int, int>(int, int)
decltype((fp > 1) ? fp : 2) dt9<int>(int)
decltype((fp >> 1) ? 1 : 2) dt16<int>(int)
decltype((int)(1) < fp ? 1 : 2) dt29<int>(int)
decltype((static_cast<int>(fp) > 1) ? 1 : 2) dt27<int>(int)
decltype(3 ? (3 > 2) ? 1 : 2 : 3) tern<3>(int (*) [(3 > 1) ? 3 < 5 ? 1 : 2 : 3])
decltype(auto) da<int>(int)
decltype(delete[] fp, 0) e17<int>(int*)
decltype(fp ? ::new int : nullptr) e2<int>(int)
decltype(fp ? 1 : ns::T::v) e6<int>(int)
decltype(fp ? fp < 2 : (fp > 3)) dt20<int>(int)
decltype(fp ? ns::g(int)(1) : 2) dt22<int>(int)
decltype(fp ? ns::T{} : ns::T{}) dt7<int>(int)
decltype(fp < 1 ? fp : 2) dt8<int>(int)
decltype(fp < 1 ? fp < 2 ? 1 : 2 : 3) dt28<int>(int)
decltype(fp->*&A::m ? 1 : 2) dt15<A*>(A*)
decltype(fp, ::new int) e4<int>(int)
decltype(fp.*&A::m ? 1 : 2) dt14<A>(A)
decltype(fp.A::operator->() ? 1 : 2) dt30<A>(A)
decltype(fp.get<int>() ? 1 : 2) e7<A>(A)
decltype(fp.operator bool() ? 1 : 2) dt12<A>(A)
decltype(fp.operator int() ? 1 : 2) dt11<A>(A)
decltype(fp.operator int*() ? 1 : 2) g6<B>(B)
decltype(fp.operator ns::T().v ? 1 : 2) dt17<A>(A)
decltype(fp.operator()<int>(1) ? 1 : 2) g7<B>(B)
decltype(fp.operator<<int>(1) ? (fp > 1) : fp < 2) g8<B>(B)
decltype(fp.operator><int>(1) ? 1 : 2) g10<B>(B)
decltype(fp.operator>><int>(1) ? 1 : 2) g9<B>(B)
decltype(fp.T::v) dt2<ns::T>(ns::T)
decltype(new(operator new(unsigned long)(1)) int) e16<int>(int)
decltype(new[] int ? 1 : 2) dt18<int>(int)
decltype(noexcept (::new int) ? 1 : 2) g4<int>(int)
decltype(ns::g(int)(fp)) dt<int>(int)
decltype(ns::T{}, fp ? 1 : ns::g(int)(1)) dt21<int>(int)
decltype(operator new(unsigned long)(sizeof (int)) ? 1 : 2) e15<int>(int)
decltype(operator"" _x(unsigned long long)(1) ? fp : fp) dt13<int>(int)
decltype(operator<(fp, fp) ? 1 : 2) dt10<A>(A)
decltype(operator<(S<1>{}, fp) ? 1 : 2) e14<int>(int)
decltype(sizeof (::new int)) g2<int>(int)
decltype(sizeof (int) < 4 ? 1 : 2) dt19<int>(int)
decltype(throw ::new int(fp), 0) g1<int>(int)
decltype(throw fp, 0) dt5<int>(int)
decltype(true ? 1 : 2) f<&A::operator bool()>()
decltype(true ? 1 : 2) f<&A::operator co_await()>()
decltype(true ? 1 : 2) f<&A::operator int()>()
decltype(true ? 1 : 2) f<&operator delete(void*)>()
decltype(true ? 1 : 2) f<&operator delete[](void*)>()
decltype(true ? 1 : 2) f<&operator new(unsigned long)>()
decltype(true ? 1 : 2) f<&operator new[](unsigned long)>()
decltype(true ? 1 : 2) f<&operator--(int, int)>()
decltype(true ? 1 : 2) f<&operator-(int, int)>()
decltype(true ? 1 : 2) f<&operator-=(int, int)>()
decltype(true ? 1 : 2) f<&operator->(int, int)>()
decltype(true ? 1 : 2) f<&operator->*(int, int)>()
decltype(true ? 1 : 2) f<&operator,(int, int)>()
decltype(true ? 1 : 2) f<&operator!(int, int)>()
decltype(true ? 1 : 2) f<&operator!=(int, int)>()
decltype(true ? 1 : 2) f<&operator"" _x(int, int)>()
decltype(true ? 1 : 2) f<&operator()(int, int)>()
decltype(true ? 1 : 2) f<&operator[](int, int)>()
decltype(true ? 1 : 2) f<&operator*(int, int)>()
decltype(true ? 1 : 2) f<&operator*=(int, int)>()
decltype(true ? 1 : 2) f<&operator/(int, int)>()
decltype(true ? 1 : 2) f<&operator/=(int, int)>()
decltype(true ? 1 : 2) f<&operator&(int, int)>()
decltype(true ? 1 : 2) f<&operator&&(int, int)>()
decltype(true ? 1 : 2) f<&operator&=(int, int)>()
decltype(true ? 1 : 2) f<&operator%(int, int)>()
decltype(true ? 1 : 2) f<&operator%=(int, int)>()
decltype(true ? 1 : 2) f<&operator^(int, int)>()
decltype(true ? 1 : 2) f<&operator^=(int, int)>()
decltype(true ? 1 : 2) f<&operator+(int, int)>()
decltype(true ? 1 : 2) f<&operator++(int, int)>()
decltype(true ? 1 : 2) f<&operator+=(int, int)>()
decltype(true ? 1 : 2) f<&operator<(int, int)>()
decltype(true ? 1 : 2) f<&operator<<(int, int)>()
decltype(true ? 1 : 2) f<&operator<<=(int, int)>()
decltype(true ? 1 : 2) f<&operator<=(int, int)>()
decltype(true ? 1 : 2) f<&operator<=>(int, int)>()
decltype(true ? 1 : 2) f<&operator=(int, int)>()
decltype(true ? 1 : 2) f<&operator==(int, int)>()
decltype(true ? 1 : 2) f<&operator>(int, int)>()
decltype(true ? 1 : 2) f<&operator>=(int, int)>()
decltype(true ? 1 : 2) f<&operator>>(int, int)>()
decltype(true ? 1 : 2) f<&operator>>=(int, int)>()
decltype(true ? 1 : 2) f<&operator|(int, int)>()
decltype(true ? 1 : 2) f<&operator|=(int, int)>()
decltype(true ? 1 : 2) f<&operator||(int, int)>()
decltype(true ? 1 : 2) f<&operator~(int, int)>()
decltype(true ? fp : fp) dt6<int>(int)
inl()
inl()::'lambda'(int)::operator()(int) const
int B::operator()<int>(int) const
int B::operator()<long>(long) const
int B::operator>><int>(int) const
int dt23<int>(int)
int dt24<int>(int)
int dt25<int>(int)
int e1<int>(int)
Tag[abi:cxx11] tagged2<int>(int)
tagged[abi:cxx11](int)
unsigned long e3<int>(int)
use()
void e10<4>(S<(4 >> 1) ? 1 : 2>)
void e11<1>(S<1 < 2 ? 1 : (1 > 5) ? 2 : 3>)
void e12<int>(int)
void e8<2>(S<(2 > 1) ? 2 < 3 ? 1 : 2 : 3>)
void e9<4>(S<4 ? (4 >> 1) : 2>)
void f<&A::m, true ? 1 : 2>()
void f<&A::operator bool() const, 5>()
void f<&A::operator bool(), true ? 1 : 2>()
void f<&A::operator bool()>(int (*) [true ? 1 : 2])
void f<&A::operator co_await(), true ? 1 : 2>()
void f<&A::operator co_await()>(int (*) [true ? 1 : 2])
void f<&A::operator delete[](void*), 8>()
void f<&A::operator int A::*() const, 16>()
void f<&A::operator int() const, 4>()
void f<&A::operator int(), true ? 1 : 2>()
void f<&A::operator int()>(int (*) [true ? 1 : 2])
void f<&A::operator new[](unsigned long), 7>()
void f<&A::operator ns::T() const, 15>()
void f<&A::operator->(), 6>()
void f<&A::operator->*(int) const, 10>()
void f<&A::operator()(int), 9>()
void f<&A::operator[](int), 12>()
void f<&A::operator<<(int) const, 20>()
void f<&A::operator<=(A const&) const, 19>()
void f<&A::operator>>(int) const, 11>()
void f<&operator delete(void*), true ? 1 : 2>()
void f<&operator delete(void*)>(int (*) [true ? 1 : 2])
void f<&operator delete[](void*), true ? 1 : 2>()
void f<&operator delete[](void*)>(int (*) [true ? 1 : 2])
void f<&operator new(unsigned long), true ? 1 : 2>()
void f<&operator new(unsigned long)>(int (*) [true ? 1 : 2])
void f<&operator new[](unsigned long), true ? 1 : 2>()
void f<&operator new[](unsigned long)>(int (*) [true ? 1 : 2])
void f<&operator--(int, int), true ? 1 : 2>()
void f<&operator--(int, int)>(int (*) [true ? 1 : 2])
void f<&operator-(int, int), true ? 1 : 2>()
void f<&operator-(int, int)>(int (*) [true ? 1 : 2])
void f<&operator-=(int, int), true ? 1 : 2>()
void f<&operator-=(int, int)>(int (*) [true ? 1 : 2])
void f<&operator->(int, int), true ? 1 : 2>()
void f<&operator->(int, int)>(int (*) [true ? 1 : 2])
void f<&operator->*(int, int), true ? 1 : 2>()
void f<&operator->*(int, int)>(int (*) [true ? 1 : 2])
void f<&operator,(int, int), true ? 1 : 2>()
void f<&operator,(int, int)>(int (*) [true ? 1 : 2])
void f<&operator!(int, int), true ? 1 : 2>()
void f<&operator!(int, int)>(int (*) [true ? 1 : 2])
void f<&operator!=(int, int), true ? 1 : 2>()
void f<&operator!=(int, int)>(int (*) [true ? 1 : 2])
void f<&operator"" _s(char const*, unsigned long), 14>()
void f<&operator"" _x(int, int), true ? 1 : 2>()
void f<&operator"" _x(int, int)>(int (*) [true ? 1 : 2])
void f<&operator"" _x(unsigned long long), 13>()
void f<&operator()(int, int), true ? 1 : 2>()
void f<&operator()(int, int)>(int (*) [true ? 1 : 2])
void f<&operator[](int, int), true ? 1 : 2>()
void f<&operator[](int, int)>(int (*) [true ? 1 : 2])
void f<&operator*(int, int), true ? 1 : 2>()
void f<&operator*(int, int)>(int (*) [true ? 1 : 2])
void f<&operator*=(int, int), true ? 1 : 2>()
void f<&operator*=(int, int)>(int (*) [true ? 1 : 2])
void f<&operator/(int, int), true ? 1 : 2>()
void f<&operator/(int, int)>(int (*) [true ? 1 : 2])
void f<&operator/=(int, int), true ? 1 : 2>()
void f<&operator/=(int, int)>(int (*) [true ? 1 : 2])
void f<&operator&(int, int), true ? 1 : 2>()
void f<&operator&(int, int)>(int (*) [true ? 1 : 2])
void f<&operator&&(int, int), true ? 1 : 2>()
void f<&operator&&(int, int)>(int (*) [true ? 1 : 2])
void f<&operator&=(int, int), true ? 1 : 2>()
void f<&operator&=(int, int)>(int (*) [true ? 1 : 2])
void f<&operator%(int, int), true ? 1 : 2>()
void f<&operator%(int, int)>(int (*) [true ? 1 : 2])
void f<&operator%=(int, int), true ? 1 : 2>()
void f<&operator%=(int, int)>(int (*) [true ? 1 : 2])
void f<&operator^(int, int), true ? 1 : 2>()
void f<&operator^(int, int)>(int (*) [true ? 1 : 2])
void f<&operator^=(int, int), true ? 1 : 2>()
void f<&operator^=(int, int)>(int (*) [true ? 1 : 2])
void f<&operator+(int, int), true ? 1 : 2>()
void f<&operator+(int, int)>(int (*) [true ? 1 : 2])
void f<&operator++(int, int), true ? 1 : 2>()
void f<&operator++(int, int)>(int (*) [true ? 1 : 2])
void f<&operator+=(int, int), true ? 1 : 2>()
void f<&operator+=(int, int)>(int (*) [true ? 1 : 2])
void f<&operator<(A const&, A const&), 1>()
void f<&operator<(int, int), true ? 1 : 2>()
void f<&operator<(int, int)>(int (*) [true ? 1 : 2])
void f<&operator<<(int, int), true ? 1 : 2>()
void f<&operator<<(int, int)>(int (*) [true ? 1 : 2])
void f<&operator<<=(int, int), true ? 1 : 2>()
void f<&operator<<=(int, int)>(int (*) [true ? 1 : 2])
void f<&operator<=(int, int), true ? 1 : 2>()
void f<&operator<=(int, int)>(int (*) [true ? 1 : 2])
void f<&operator<=>(int, int), true ? 1 : 2>()
void f<&operator<=>(int, int)>(int (*) [true ? 1 : 2])
void f<&operator=(int, int), true ? 1 : 2>()
void f<&operator=(int, int)>(int (*) [true ? 1 : 2])
void f<&operator==(int, int), true ? 1 : 2>()
void f<&operator==(int, int)>(int (*) [true ? 1 : 2])
void f<&operator>(A const&, A const&), 2>()
void f<&operator>(int, int), true ? 1 : 2>()
void f<&operator>(int, int)>(int (*) [true ? 1 : 2])
void f<&operator>=(A const&, A const&), 3>()
void f<&operator>=(int, int), true ? 1 : 2>()
void f<&operator>=(int, int)>(int (*) [true ? 1 : 2])
void f<&operator>>(int, int), true ? 1 : 2>()
void f<&operator>>(int, int)>(int (*) [true ? 1 : 2])
void f<&operator>>=(int, int), true ? 1 : 2>()
void f<&operator>>=(int, int)>(int (*) [true ? 1 : 2])
void f<&operator|(int, int), true ? 1 : 2>()
void f<&operator|(int, int)>(int (*) [true ? 1 : 2])
void f<&operator|=(int, int), true ? 1 : 2>()
void f<&operator|=(int, int)>(int (*) [true ? 1 : 2])
void f<&operator||(int, int), true ? 1 : 2>()
void f<&operator||(int, int)>(int (*) [true ? 1 : 2])
void f<&operator~(int, int), true ? 1 : 2>()
void f<&operator~(int, int)>(int (*) [true ? 1 : 2])
void f<1 < 2 ? (3 > 4) ? 1 : 2 : 3>()
void f<2, 17>()
void f<2, 18>()
void f<int (A::*)(int), true ? 1 : 2>()
void f<int A::*, true ? 1 : 2>()
void f<true ? (1 > 2) ? 1 : 2 : 3>()
void f<true ? 1 : 1 < 2 ? 2 : 3>()
void f<true ? 1 : 2, &A::operator bool()>()
void f<true ? 1 : 2, &A::operator co_await()>()
void f<true ? 1 : 2, &A::operator int()>()
void f<true ? 1 : 2, &operator delete(void*)>()
void f<true ? 1 : 2, &operator delete[](void*)>()
void f<true ? 1 : 2, &operator new(unsigned long)>()
void f<true ? 1 : 2, &operator new[](unsigned long)>()
void f<true ? 1 : 2, &operator--(int, int)>()
void f<true ? 1 : 2, &operator-(int, int)>()
void f<true ? 1 : 2, &operator-=(int, int)>()
void f<true ? 1 : 2, &operator->(int, int)>()
void f<true ? 1 : 2, &operator->*(int, int)>()
void f<true ? 1 : 2, &operator,(int, int)>()
void f<true ? 1 : 2, &operator!(int, int)>()
void f<true ? 1 : 2, &operator!=(int, int)>()
void f<true ? 1 : 2, &operator"" _x(int, int)>()
void f<true ? 1 : 2, &operator()(int, int)>()
void f<true ? 1 : 2, &operator[](int, int)>()
void f<true ? 1 : 2, &operator*(int, int)>()
void f<true ? 1 : 2, &operator*=(int, int)>()
void f<true ? 1 : 2, &operator/(int, int)>()
void f<true ? 1 : 2, &operator/=(int, int)>()
void f<true ? 1 : 2, &operator&(int, int)>()
void f<true ? 1 : 2, &operator&&(int, int)>()
void f<true ? 1 : 2, &operator&=(int, int)>()
void f<true ? 1 : 2, &operator%(int, int)>()
void f<true ? 1 : 2, &operator%=(int, int)>()
void f<true ? 1 : 2, &operator^(int, int)>()
void f<true ? 1 : 2, &operator^=(int, int)>()
void f<true ? 1 : 2, &operator+(int, int)>()
void f<true ? 1 : 2, &operator++(int, int)>()
void f<true ? 1 : 2, &operator+=(int, int)>()
void f<true ? 1 : 2, &operator<(int, int)>()
void f<true ? 1 : 2, &operator<<(int, int)>()
void f<true ? 1 : 2, &operator<<=(int, int)>()
void f<true ? 1 : 2, &operator<=(int, int)>()
void f<true ? 1 : 2, &operator<=>(int, int)>()
void f<true ? 1 : 2, &operator=(int, int)>()
void f<true ? 1 : 2, &operator==(int, int)>()
void f<true ? 1 : 2, &operator>(int, int)>()
void f<true ? 1 : 2, &operator>=(int, int)>()
void f<true ? 1 : 2, &operator>>(int, int)>()
void f<true ? 1 : 2, &operator>>=(int, int)>()
void f<true ? 1 : 2, &operator|(int, int)>()
void f<true ? 1 : 2, &operator|=(int, int)>()
void f<true ? 1 : 2, &operator||(int, int)>()
void f<true ? 1 : 2, &operator~(int, int)>()
void f<true ? false ? 1 : 2 : 3, true ? 1 : 2>()
void f<true ? false ? 1 < 2 ? 7 : 8 : 2 : 3>()
void g11<1, 1>(S<1 ? 1 ? 1 : 2 : 1 ? 3 : 4>)
void g12<1>(S<1 < 2 ? (1 > 0) ? 1 : 2 : 1 < 9 ? 3 : 4>)
void g13<1>(S<1 == 1 ? 1 : 1 != 2 ? 2 : 3>)
void pm<&A::m>()
void pmf<&A::fn(int)>()
void rq<int>(int)
void rq2<int>(int)
void tf<inl()::'lambda'(auto, auto...)>(auto)
void tf<inl()::'lambda'(auto), true ? 1 : 2>(auto)
void tf<inl()::'lambda'(auto)>(auto)
void tf<inl()::'lambda'(int)::operator()(int) const::'lambda'(auto)>(auto)
void tf<inl()::'lambda'<typename $T>($T)>($T)
void tf<int (A::*)(int)>(int (A::*)(int))
void tf<int A::* A::*>(int A::* A::*)
void tf<int A::*>(int A::*)
void tf<use()::$_0>(use()::$_0)
void tf<use()::$_1>(use()::$_1)
void tf<use()::$_2>(use()::$_2)
void tf<use()::$_3>(use()::$_3)
void tf<void tmpl<int>()::'lambda'(auto)>(auto)
void tf<void tmpl<int>()::'lambda'<int $N>(S<$N>)>(void tmpl<int>()::'lambda'<int $N>(S<$N>))
void tmpl<int>()
void ts<1>(S<1 ? (1 > 1) ? 1 : 2 : 3>)
`
	cxxfiltNamesSHA256 = "d28509f826de70ee5101ffcbc41664db3d17fcaab1b0f8669781c5f99fdeff00"

	llvmNames = `S<1 ? 1 : 2> r1<1>()
S<1 < 2 ? 1 : 2> r2<1>()
S<(1 > 2) ? 1 : 2> r3<1>()
S<(&gp)->x ? 1 : 2> r4<&gp>()
S<(&gp)->x> r5<&gp>()
S<((&gp)->x > 0) ? 1 : 2> r6<&gp>()
std::__1::vector<S<1 ? 1 : 2>, std::__1::allocator<S<1 ? 1 : 2>>> r7<1>()
S<1 ? 1 : 2> const r8<1>()
S<1 ? 1 : 2> volatile* r9<1>()
void tf<lam()::'lambda'(std::__1::vector<int, std::__1::allocator<int>> const&, auto&&)>(auto)
void tf<lam()::'lambda'(auto, auto...)>(auto)
void tf<lam()::'lambda'(auto)>(auto)
void tf<lam()::'lambda'<int $N>(S<$N ? 1 : 2>)>(lam()::'lambda'<int $N>(S<$N ? 1 : 2>))
void tf<lam()::'lambda'<template<typename $T> typename $TT>($TT<int>)>(lam()::'lambda'<template<typename $T> typename $TT>($TT<int>))
void tf<lam()::'lambda'<typename $T>($T)>($T)
void tf<lam()::'lambda0'<typename $T>($T)>($T)
void tf<lam()::'lambda1'<typename $T>($T)>($T)
void tf<use()::$_0>(use()::$_0)
void tf<st()::$_0>(st()::$_0)
void tf<st()::$_1>(st()::$_1)
void tf<auto lam()::'lambda0'(auto)::operator()<int>(auto) const::'lambda'(auto)>(auto)
vs<int>
vt<int>
lam()
S<1 ? 1 : 2> const* r10<1>()
decltype(std::declval<int>() < 1) r11<int>(int)
decltype(std::declval<A>().get<int>()) r12<A>(A)
S<sizeof (std::declval<A>().get<int>()) ? 1 : 2> r13<A>(A)
std::__1::enable_if<sizeof (int) ? true : false, int>::type r14<int>(int)
std::__1::enable_if<(sizeof (int) > 1) ? true : false, S<1>>::type r15<int>(int)
ns::W<S<sizeof (int) ? 1 : 2>>::type r16<int>(int)
S<ns::W<int>::v ? 1 : 2> r17<int>(int)
S<1 < sizeof (int) ? 1 : 2> r18<int>(int)
S<sizeof (int) < 2 ? 1 : 2> r19<int>(int)
S<(sizeof (int) >> 1) ? 1 : 2> r20<int>(int)
S<sizeof (int) << 1 ? 1 : 2> r21<int>(int)
S<sizeof (int) <= 2 ? 1 : 2> r22<int>(int)
S<sizeof (int) >= 2 ? 1 : 2> r23<int>(int)
S<&A::m ? 1 : 2> r24<&A::m>()
S<(&gp)->self ? 1 : 2> r25<&gp>()
decltype((&gp)->x ? 1 : 2) r27<&gp>()
S<&(&gp)->x ? 1 : 2> r28<&gp>()
S<sizeof (int) ? sizeof (int) : 2> (*r29<int>(int))()
S<sizeof (int) ? 1 : 2> (A::*r30<int>(int))()
decltype(fp.operator<<int>(1) ? 1 : 2) r31<A>(A)
S<sizeof (fp.operator<<int>(1)) ? 1 : 2> r32<A>(A)
decltype(operator<<int>(A{}, fp)) r33<int>(int)
S<sizeof (operator<<(A{}, fp)) ? 1 : 2> r34<int>(int)
S<sizeof (fp->*&A::m) ? 1 : 2> r35<A*>(A*)
S<sizeof (fp->m) ? 1 : 2> r36<A*>(A*)
S<sizeof (int)> r38<int>(int)
S<1> r39<int>(int)
S<sizeof (int) ? 1 : 2> r40<int>(int)
S<1> r41<int>(int)
S<1> r42<int>(int)
use()
bool operator><int>(A, int)
st()
bool operator<<<int>(A, int)
S<(3 > 1) ? 1 : 2> operator<<<3>(S<3>, long)
bool operator<<int>(A, int)
S<1 ? 1 : 2> operator<<1>(S<1>, int)
void* G::operator new<int>(unsigned long, int)
V<int> A::get<int>() const
S<sizeof (int) ? 1 : 2> A::getc<int>() const
A::operator S<1><1>() const
A::operator V<int><int>() const
A::operator std::__1::vector<int, std::__1::allocator<int>><int>() const
bool A::operator><int>(int) const
bool A::operator<=<int>(int) const
bool A::operator<<<int>(int) const
bool A::operator<<int>(int) const
bool A::operator->*<int>(int) const
std::__1::vector<S<1>, std::__1::allocator<S<1>>>::__annotate_delete[abi:nqe210106]() const
std::__1::vector<S<1>, std::__1::allocator<S<1>>>::__annotate_shrink[abi:nqe210106](unsigned long) const
std::__1::vector<S<1>, std::__1::allocator<S<1>>>::__annotate_contiguous_container[abi:nqe210106](void const*, void const*) const
std::__1::vector<S<1>, std::__1::allocator<S<1>>>::data[abi:nqe210106]() const
std::__1::vector<S<1>, std::__1::allocator<S<1>>>::size[abi:nqe210106]() const
std::__1::vector<S<1>, std::__1::allocator<S<1>>>::capacity[abi:nqe210106]() const
std::__1::vector<int, std::__1::allocator<int>>::__annotate_delete[abi:nqe210106]() const
std::__1::vector<int, std::__1::allocator<int>>::__annotate_shrink[abi:nqe210106](unsigned long) const
std::__1::vector<int, std::__1::allocator<int>>::__annotate_contiguous_container[abi:nqe210106](void const*, void const*) const
std::__1::vector<int, std::__1::allocator<int>>::data[abi:nqe210106]() const
std::__1::vector<int, std::__1::allocator<int>>::size[abi:nqe210106]() const
std::__1::vector<int, std::__1::allocator<int>>::capacity[abi:nqe210106]() const
void std::__1::__destroy_at[abi:nqe210106]<S<1>, 0>(S<1>*)
void std::__1::__destroy_at[abi:nqe210106]<int, 0>(int*)
S<1>* std::__1::__to_address[abi:nqe210106]<S<1>>(S<1>*)
int* std::__1::__to_address[abi:nqe210106]<int>(int*)
std::__1::__non_trivial_if<true, std::__1::allocator<S<1>>>::__non_trivial_if[abi:nqe210106]()
std::__1::__non_trivial_if<true, std::__1::allocator<int>>::__non_trivial_if[abi:nqe210106]()
std::__1::allocator_traits<std::__1::allocator<S<1>>>::deallocate[abi:nqe210106](std::__1::allocator<S<1>>&, S<1>*, unsigned long)
void std::__1::allocator_traits<std::__1::allocator<S<1>>>::destroy[abi:nqe210106]<S<1>, 0>(std::__1::allocator<S<1>>&, S<1>*)
std::__1::allocator_traits<std::__1::allocator<int>>::deallocate[abi:nqe210106](std::__1::allocator<int>&, int*, unsigned long)
void std::__1::allocator_traits<std::__1::allocator<int>>::destroy[abi:nqe210106]<int, 0>(std::__1::allocator<int>&, int*)
void std::__1::__libcpp_deallocate[abi:nqe210106]<S<1>>(std::__1::__type_identity<S<1>>::type*, std::__1::__element_count, unsigned long)
void std::__1::__libcpp_deallocate[abi:nqe210106]<int>(std::__1::__type_identity<int>::type*, std::__1::__element_count, unsigned long)
std::__1::__is_overaligned_for_new[abi:nqe210106](unsigned long)
void std::__1::__annotate_contiguous_container[abi:nqe210106]<std::__1::allocator<S<1>>>(void const*, void const*, void const*, void const*)
void std::__1::__annotate_contiguous_container[abi:nqe210106]<std::__1::allocator<int>>(void const*, void const*, void const*, void const*)
std::__1::vector<S<1>, std::__1::allocator<S<1>>>::__destroy_vector::__destroy_vector[abi:nqe210106](std::__1::vector<S<1>, std::__1::allocator<S<1>>>&)
std::__1::vector<S<1>, std::__1::allocator<S<1>>>::__destroy_vector::__destroy_vector[abi:nqe210106](std::__1::vector<S<1>, std::__1::allocator<S<1>>>&)
std::__1::vector<S<1>, std::__1::allocator<S<1>>>::__destroy_vector::operator()[abi:nqe210106]()
std::__1::vector<S<1>, std::__1::allocator<S<1>>>::__base_destruct_at_end[abi:nqe210106](S<1>*)
std::__1::vector<S<1>, std::__1::allocator<S<1>>>::clear[abi:nqe210106]()
std::__1::vector<S<1>, std::__1::allocator<S<1>>>::vector[abi:nqe210106]()
std::__1::vector<S<1>, std::__1::allocator<S<1>>>::vector[abi:nqe210106]()
std::__1::vector<S<1>, std::__1::allocator<S<1>>>::~vector[abi:nqe210106]()
std::__1::vector<S<1>, std::__1::allocator<S<1>>>::~vector[abi:nqe210106]()
std::__1::vector<int, std::__1::allocator<int>>::__destroy_vector::__destroy_vector[abi:nqe210106](std::__1::vector<int, std::__1::allocator<int>>&)
std::__1::vector<int, std::__1::allocator<int>>::__destroy_vector::__destroy_vector[abi:nqe210106](std::__1::vector<int, std::__1::allocator<int>>&)
std::__1::vector<int, std::__1::allocator<int>>::__destroy_vector::operator()[abi:nqe210106]()
std::__1::vector<int, std::__1::allocator<int>>::__base_destruct_at_end[abi:nqe210106](int*)
std::__1::vector<int, std::__1::allocator<int>>::clear[abi:nqe210106]()
std::__1::vector<int, std::__1::allocator<int>>::vector[abi:nqe210106]()
std::__1::vector<int, std::__1::allocator<int>>::vector[abi:nqe210106]()
std::__1::vector<int, std::__1::allocator<int>>::~vector[abi:nqe210106]()
std::__1::vector<int, std::__1::allocator<int>>::~vector[abi:nqe210106]()
std::__1::allocator<S<1>>::deallocate[abi:nqe210106](S<1>*, unsigned long)
std::__1::allocator<S<1>>::allocator[abi:nqe210106]()
std::__1::allocator<S<1>>::allocator[abi:nqe210106]()
std::__1::allocator<int>::deallocate[abi:nqe210106](int*, unsigned long)
std::__1::allocator<int>::allocator[abi:nqe210106]()
std::__1::allocator<int>::allocator[abi:nqe210106]()
S<sizeof (int) ? 1 : 2> A::mf<int>(int) const volatile
bool operator>><int>(A, int)
auto lam()::'lambda0'(auto)::operator()<int>(auto) const
S<&gp->*&P::y ? 1 : 2> q1<&gp, &P::y>()
S<sizeof (int) >= 2 ? 1 : 2> q2<int>(int)
V<S<sizeof (int) >= 2 ? 1 : 2>> q3<int>(int)
S<sizeof (int) >= 2> q4<int>(int)
void q5<int>(S<sizeof (int) >= 2 ? 1 : 2>)
S<sizeof (int) >= 2 && sizeof (int) ? 1 : 2> q6<int>(int)
S<3 >= 2 ? 3 : 2> q7<3>()
S<(&gp)->x >= 1 ? 1 : 2> q8<&gp>()
S<(&gp)->x> q9<&gp>()
S<3 >= 2 ? 1 : 2>* q10<3>()
S<3 >= 2 ? 1 : 2> (*q11<3>())()
use()
`
	llvmNamesSHA256 = "d1f9128b6bddfa5e083228963c6587aff52921a773e3e2160ad5a1992ea79868"

	// r4Names holds 727 demangled names, one per line, printed by the same
	// c++filt from symbols the same clang compiled for a later review: its
	// 563-line list (562 distinct) of templates whose arguments hold a
	// conditional beside "->", ">>", "<=>", ">=" or an operator name, then its
	// 165 names whose C++20 string-literal arguments, such as L8{"a:b"},
	// L8{"::1"} and L8{"a/*b"}, hold colons, brackets and comment delimiters.
	r4Names = `S<3 + 2> g1<3>()
S<3 + 2 ? 1 : 2> g2<3>()
S<3 + 2 ? 1 : 2> g3<3>()
S<3 ? 3 + 2 : 2> g4<3>()
S<3 ? 1 : 3 + 2> g5<3>()
void g6<3>(S<3 + 2>, S<3 + 2 ? 1 : 2>)
SA<3 + 2, 3 ? 1 : 2> g7<3>()
SA<3 ? 1 : 2, 3 + 2> g8<3>()
S<W<3 + 2>::value ? 1 : 2> g9<3>()
V<S<3 + 2 ? 1 : 2>, S<3>> g10<3>()
S<3 + 2 ? 1 : 2>* g11<3>(S<3 + 2>*)
decltype(fp + fp) g12<A>(A)
decltype(sizeof (A) ? fp + fp : fp + fp) g13<A>(A)
V<decltype(fp + fp), S<sizeof (A) ? 1 : 2>> g14<A>(A)
void g15<A>(A, decltype((A)() + (A)())*, S<sizeof (A) ? 1 : 2>)
SA<sizeof (A) ? 1 : 2, &A::operator+> g16<A>()
void g17<A>(SA<sizeof (A) ? 1 : 2, &A::operator+>)
SA<&A::operator+, sizeof (A) ? 1 : 2> g18<A>()
S<3 - 2> g19<3>()
S<3 - 2 ? 1 : 2> g20<3>()
S<3 - 2 ? 1 : 2> g21<3>()
S<3 ? 3 - 2 : 2> g22<3>()
S<3 ? 1 : 3 - 2> g23<3>()
void g24<3>(S<3 - 2>, S<3 - 2 ? 1 : 2>)
SA<3 - 2, 3 ? 1 : 2> g25<3>()
SA<3 ? 1 : 2, 3 - 2> g26<3>()
S<W<3 - 2>::value ? 1 : 2> g27<3>()
V<S<3 - 2 ? 1 : 2>, S<3>> g28<3>()
S<3 - 2 ? 1 : 2>* g29<3>(S<3 - 2>*)
decltype(fp - fp) g30<A>(A)
decltype(sizeof (A) ? fp - fp : fp - fp) g31<A>(A)
V<decltype(fp - fp), S<sizeof (A) ? 1 : 2>> g32<A>(A)
void g33<A>(A, decltype((A)() - (A)())*, S<sizeof (A) ? 1 : 2>)
SA<sizeof (A) ? 1 : 2, &A::operator-> g34<A>()
void g35<A>(SA<sizeof (A) ? 1 : 2, &A::operator->)
SA<&A::operator-, sizeof (A) ? 1 : 2> g36<A>()
S<3 * 2> g37<3>()
S<3 * 2 ? 1 : 2> g38<3>()
S<3 * 2 ? 1 : 2> g39<3>()
S<3 ? 3 * 2 : 2> g40<3>()
S<3 ? 1 : 3 * 2> g41<3>()
void g42<3>(S<3 * 2>, S<3 * 2 ? 1 : 2>)
SA<3 * 2, 3 ? 1 : 2> g43<3>()
SA<3 ? 1 : 2, 3 * 2> g44<3>()
S<W<3 * 2>::value ? 1 : 2> g45<3>()
V<S<3 * 2 ? 1 : 2>, S<3>> g46<3>()
S<3 * 2 ? 1 : 2>* g47<3>(S<3 * 2>*)
decltype(fp * fp) g48<A>(A)
decltype(sizeof (A) ? fp * fp : fp * fp) g49<A>(A)
V<decltype(fp * fp), S<sizeof (A) ? 1 : 2>> g50<A>(A)
void g51<A>(A, decltype((A)() * (A)())*, S<sizeof (A) ? 1 : 2>)
SA<sizeof (A) ? 1 : 2, &A::operator*> g52<A>()
void g53<A>(SA<sizeof (A) ? 1 : 2, &A::operator*>)
SA<&A::operator*, sizeof (A) ? 1 : 2> g54<A>()
S<3 / 2> g55<3>()
S<(3 / 2) ? 1 : 2> g56<3>()
S<(3 / 2) ? 1 : 2> g57<3>()
S<3 ? 3 / 2 : 2> g58<3>()
S<3 ? 1 : 3 / 2> g59<3>()
void g60<3>(S<3 / 2>, S<(3 / 2) ? 1 : 2>)
SA<3 / 2, 3 ? 1 : 2> g61<3>()
SA<3 ? 1 : 2, 3 / 2> g62<3>()
S<W<3 / 2>::value ? 1 : 2> g63<3>()
V<S<(3 / 2) ? 1 : 2>, S<3>> g64<3>()
S<(3 / 2) ? 1 : 2>* g65<3>(S<3 / 2>*)
decltype(fp / fp) g66<A>(A)
decltype(sizeof (A) ? fp / fp : fp / fp) g67<A>(A)
V<decltype(fp / fp), S<sizeof (A) ? 1 : 2>> g68<A>(A)
void g69<A>(A, decltype((A)() / (A)())*, S<sizeof (A) ? 1 : 2>)
SA<sizeof (A) ? 1 : 2, &A::operator/> g70<A>()
void g71<A>(SA<sizeof (A) ? 1 : 2, &A::operator/>)
SA<&A::operator/, sizeof (A) ? 1 : 2> g72<A>()
S<3 % 2> g73<3>()
S<3 % 2 ? 1 : 2> g74<3>()
S<3 % 2 ? 1 : 2> g75<3>()
S<3 ? 3 % 2 : 2> g76<3>()
S<3 ? 1 : 3 % 2> g77<3>()
void g78<3>(S<3 % 2>, S<3 % 2 ? 1 : 2>)
SA<3 % 2, 3 ? 1 : 2> g79<3>()
SA<3 ? 1 : 2, 3 % 2> g80<3>()
S<W<3 % 2>::value ? 1 : 2> g81<3>()
V<S<3 % 2 ? 1 : 2>, S<3>> g82<3>()
S<3 % 2 ? 1 : 2>* g83<3>(S<3 % 2>*)
decltype(fp % fp) g84<A>(A)
decltype(sizeof (A) ? fp % fp : fp % fp) g85<A>(A)
V<decltype(fp % fp), S<sizeof (A) ? 1 : 2>> g86<A>(A)
void g87<A>(A, decltype((A)() % (A)())*, S<sizeof (A) ? 1 : 2>)
SA<sizeof (A) ? 1 : 2, &A::operator%> g88<A>()
void g89<A>(SA<sizeof (A) ? 1 : 2, &A::operator%>)
SA<&A::operator%, sizeof (A) ? 1 : 2> g90<A>()
S<3 ^ 2> g91<3>()
S<3 ^ 2 ? 1 : 2> g92<3>()
S<3 ^ 2 ? 1 : 2> g93<3>()
S<3 ? 3 ^ 2 : 2> g94<3>()
S<3 ? 1 : 3 ^ 2> g95<3>()
void g96<3>(S<3 ^ 2>, S<3 ^ 2 ? 1 : 2>)
SA<3 ^ 2, 3 ? 1 : 2> g97<3>()
SA<3 ? 1 : 2, 3 ^ 2> g98<3>()
S<W<3 ^ 2>::value ? 1 : 2> g99<3>()
V<S<3 ^ 2 ? 1 : 2>, S<3>> g100<3>()
S<3 ^ 2 ? 1 : 2>* g101<3>(S<3 ^ 2>*)
decltype(fp ^ fp) g102<A>(A)
decltype(sizeof (A) ? fp ^ fp : fp ^ fp) g103<A>(A)
V<decltype(fp ^ fp), S<sizeof (A) ? 1 : 2>> g104<A>(A)
void g105<A>(A, decltype((A)() ^ (A)())*, S<sizeof (A) ? 1 : 2>)
SA<sizeof (A) ? 1 : 2, &A::operator^> g106<A>()
void g107<A>(SA<sizeof (A) ? 1 : 2, &A::operator^>)
SA<&A::operator^, sizeof (A) ? 1 : 2> g108<A>()
S<3 & 2> g109<3>()
S<3 & 2 ? 1 : 2> g110<3>()
S<3 & 2 ? 1 : 2> g111<3>()
S<3 ? 3 & 2 : 2> g112<3>()
S<3 ? 1 : 3 & 2> g113<3>()
void g114<3>(S<3 & 2>, S<3 & 2 ? 1 : 2>)
SA<3 & 2, 3 ? 1 : 2> g115<3>()
SA<3 ? 1 : 2, 3 & 2> g116<3>()
S<W<3 & 2>::value ? 1 : 2> g117<3>()
V<S<3 & 2 ? 1 : 2>, S<3>> g118<3>()
S<3 & 2 ? 1 : 2>* g119<3>(S<3 & 2>*)
decltype(fp & fp) g120<A>(A)
decltype(sizeof (A) ? fp & fp : fp & fp) g121<A>(A)
V<decltype(fp & fp), S<sizeof (A) ? 1 : 2>> g122<A>(A)
void g123<A>(A, decltype((A)() & (A)())*, S<sizeof (A) ? 1 : 2>)
SA<sizeof (A) ? 1 : 2, &A::operator&> g124<A>()
void g125<A>(SA<sizeof (A) ? 1 : 2, &A::operator&>)
SA<&A::operator&, sizeof (A) ? 1 : 2> g126<A>()
S<3 | 2> g127<3>()
S<3 | 2 ? 1 : 2> g128<3>()
S<3 | 2 ? 1 : 2> g129<3>()
S<3 ? 3 | 2 : 2> g130<3>()
S<3 ? 1 : 3 | 2> g131<3>()
void g132<3>(S<3 | 2>, S<3 | 2 ? 1 : 2>)
SA<3 | 2, 3 ? 1 : 2> g133<3>()
SA<3 ? 1 : 2, 3 | 2> g134<3>()
S<W<3 | 2>::value ? 1 : 2> g135<3>()
V<S<3 | 2 ? 1 : 2>, S<3>> g136<3>()
S<3 | 2 ? 1 : 2>* g137<3>(S<3 | 2>*)
decltype(fp | fp) g138<A>(A)
decltype(sizeof (A) ? fp | fp : fp | fp) g139<A>(A)
V<decltype(fp | fp), S<sizeof (A) ? 1 : 2>> g140<A>(A)
void g141<A>(A, decltype((A)() | (A)())*, S<sizeof (A) ? 1 : 2>)
SA<sizeof (A) ? 1 : 2, &A::operator|> g142<A>()
void g143<A>(SA<sizeof (A) ? 1 : 2, &A::operator|>)
SA<&A::operator|, sizeof (A) ? 1 : 2> g144<A>()
S<3 << 2> g145<3>()
S<3 << 2 ? 1 : 2> g146<3>()
S<3 << 2 ? 1 : 2> g147<3>()
S<3 ? 3 << 2 : 2> g148<3>()
S<3 ? 1 : 3 << 2> g149<3>()
void g150<3>(S<3 << 2>, S<3 << 2 ? 1 : 2>)
SA<3 << 2, 3 ? 1 : 2> g151<3>()
SA<3 ? 1 : 2, 3 << 2> g152<3>()
S<W<3 << 2>::value ? 1 : 2> g153<3>()
V<S<3 << 2 ? 1 : 2>, S<3>> g154<3>()
S<3 << 2 ? 1 : 2>* g155<3>(S<3 << 2>*)
SA<sizeof (A) ? 1 : 2, &A::operator<<> g160<A>()
void g161<A>(SA<sizeof (A) ? 1 : 2, &A::operator<<>)
SA<&A::operator<<, sizeof (A) ? 1 : 2> g162<A>()
S<(3 >> 2)> g163<3>()
S<(3 >> 2) ? 1 : 2> g164<3>()
S<(3 >> 2) ? 1 : 2> g165<3>()
S<3 ? (3 >> 2) : 2> g166<3>()
S<3 ? 1 : (3 >> 2)> g167<3>()
void g168<3>(S<(3 >> 2)>, S<(3 >> 2) ? 1 : 2>)
SA<(3 >> 2), 3 ? 1 : 2> g169<3>()
SA<3 ? 1 : 2, (3 >> 2)> g170<3>()
S<W<(3 >> 2)>::value ? 1 : 2> g171<3>()
V<S<(3 >> 2) ? 1 : 2>, S<3>> g172<3>()
S<(3 >> 2) ? 1 : 2>* g173<3>(S<(3 >> 2)>*)
SA<sizeof (A) ? 1 : 2, &A::operator>>> g178<A>()
void g179<A>(SA<sizeof (A) ? 1 : 2, &A::operator>>>)
SA<&A::operator>>, sizeof (A) ? 1 : 2> g180<A>()
S<3 < 2> g181<3>()
S<3 < 2 ? 1 : 2> g182<3>()
S<3 < 2 ? 1 : 2> g183<3>()
S<3 ? 3 < 2 : 2> g184<3>()
S<3 ? 1 : 3 < 2> g185<3>()
void g186<3>(S<3 < 2>, S<3 < 2 ? 1 : 2>)
SA<3 < 2, 3 ? 1 : 2> g187<3>()
SA<3 ? 1 : 2, 3 < 2> g188<3>()
S<W<3 < 2>::value ? 1 : 2> g189<3>()
V<S<3 < 2 ? 1 : 2>, S<3>> g190<3>()
S<3 < 2 ? 1 : 2>* g191<3>(S<3 < 2>*)
decltype(fp < fp) g192<A>(A)
decltype(sizeof (A) ? fp < fp : fp < fp) g193<A>(A)
V<decltype(fp < fp), S<sizeof (A) ? 1 : 2>> g194<A>(A)
void g195<A>(A, decltype((A)() < (A)())*, S<sizeof (A) ? 1 : 2>)
SA<sizeof (A) ? 1 : 2, &A::operator<> g196<A>()
void g197<A>(SA<sizeof (A) ? 1 : 2, &A::operator<>)
SA<&A::operator<, sizeof (A) ? 1 : 2> g198<A>()
S<(3 > 2)> g199<3>()
S<(3 > 2) ? 1 : 2> g200<3>()
S<(3 > 2) ? 1 : 2> g201<3>()
S<3 ? (3 > 2) : 2> g202<3>()
S<3 ? 1 : (3 > 2)> g203<3>()
void g204<3>(S<(3 > 2)>, S<(3 > 2) ? 1 : 2>)
SA<(3 > 2), 3 ? 1 : 2> g205<3>()
SA<3 ? 1 : 2, (3 > 2)> g206<3>()
S<W<(3 > 2)>::value ? 1 : 2> g207<3>()
V<S<(3 > 2) ? 1 : 2>, S<3>> g208<3>()
S<(3 > 2) ? 1 : 2>* g209<3>(S<(3 > 2)>*)
decltype((fp > fp)) g210<A>(A)
decltype(sizeof (A) ? (fp > fp) : (fp > fp)) g211<A>(A)
V<decltype(fp > fp), S<sizeof (A) ? 1 : 2>> g212<A>(A)
void g213<A>(A, decltype(((A)() > (A)()))*, S<sizeof (A) ? 1 : 2>)
SA<sizeof (A) ? 1 : 2, &A::operator>> g214<A>()
void g215<A>(SA<sizeof (A) ? 1 : 2, &A::operator>>)
SA<&A::operator>, sizeof (A) ? 1 : 2> g216<A>()
S<3 <= 2> g217<3>()
S<3 <= 2 ? 1 : 2> g218<3>()
S<3 <= 2 ? 1 : 2> g219<3>()
S<3 ? 3 <= 2 : 2> g220<3>()
S<3 ? 1 : 3 <= 2> g221<3>()
void g222<3>(S<3 <= 2>, S<3 <= 2 ? 1 : 2>)
SA<3 <= 2, 3 ? 1 : 2> g223<3>()
SA<3 ? 1 : 2, 3 <= 2> g224<3>()
S<W<3 <= 2>::value ? 1 : 2> g225<3>()
V<S<3 <= 2 ? 1 : 2>, S<3>> g226<3>()
S<3 <= 2 ? 1 : 2>* g227<3>(S<3 <= 2>*)
decltype(fp <= fp) g228<A>(A)
decltype(sizeof (A) ? fp <= fp : fp <= fp) g229<A>(A)
V<decltype(fp <= fp), S<sizeof (A) ? 1 : 2>> g230<A>(A)
void g231<A>(A, decltype((A)() <= (A)())*, S<sizeof (A) ? 1 : 2>)
SA<sizeof (A) ? 1 : 2, &A::operator<=> g232<A>()
void g233<A>(SA<sizeof (A) ? 1 : 2, &A::operator<=>)
SA<&A::operator<=, sizeof (A) ? 1 : 2> g234<A>()
S<3 >= 2> g235<3>()
S<3 >= 2 ? 1 : 2> g236<3>()
S<3 >= 2 ? 1 : 2> g237<3>()
S<3 ? 3 >= 2 : 2> g238<3>()
S<3 ? 1 : 3 >= 2> g239<3>()
void g240<3>(S<3 >= 2>, S<3 >= 2 ? 1 : 2>)
SA<3 >= 2, 3 ? 1 : 2> g241<3>()
SA<3 ? 1 : 2, 3 >= 2> g242<3>()
S<W<3 >= 2>::value ? 1 : 2> g243<3>()
V<S<3 >= 2 ? 1 : 2>, S<3>> g244<3>()
S<3 >= 2 ? 1 : 2>* g245<3>(S<3 >= 2>*)
decltype(fp >= fp) g246<A>(A)
decltype(sizeof (A) ? fp >= fp : fp >= fp) g247<A>(A)
V<decltype(fp >= fp), S<sizeof (A) ? 1 : 2>> g248<A>(A)
void g249<A>(A, decltype((A)() >= (A)())*, S<sizeof (A) ? 1 : 2>)
SA<sizeof (A) ? 1 : 2, &A::operator>=> g250<A>()
void g251<A>(SA<sizeof (A) ? 1 : 2, &A::operator>=>)
SA<&A::operator>=, sizeof (A) ? 1 : 2> g252<A>()
S<3 == 2> g253<3>()
S<3 == 2 ? 1 : 2> g254<3>()
S<3 == 2 ? 1 : 2> g255<3>()
S<3 ? 3 == 2 : 2> g256<3>()
S<3 ? 1 : 3 == 2> g257<3>()
void g258<3>(S<3 == 2>, S<3 == 2 ? 1 : 2>)
SA<3 == 2, 3 ? 1 : 2> g259<3>()
SA<3 ? 1 : 2, 3 == 2> g260<3>()
S<W<3 == 2>::value ? 1 : 2> g261<3>()
V<S<3 == 2 ? 1 : 2>, S<3>> g262<3>()
S<3 == 2 ? 1 : 2>* g263<3>(S<3 == 2>*)
decltype(fp == fp) g264<A>(A)
decltype(sizeof (A) ? fp == fp : fp == fp) g265<A>(A)
V<decltype(fp == fp), S<sizeof (A) ? 1 : 2>> g266<A>(A)
void g267<A>(A, decltype((A)() == (A)())*, S<sizeof (A) ? 1 : 2>)
SA<sizeof (A) ? 1 : 2, &A::operator==> g268<A>()
void g269<A>(SA<sizeof (A) ? 1 : 2, &A::operator==>)
SA<&A::operator==, sizeof (A) ? 1 : 2> g270<A>()
S<3 != 2> g271<3>()
S<3 != 2 ? 1 : 2> g272<3>()
S<3 != 2 ? 1 : 2> g273<3>()
S<3 ? 3 != 2 : 2> g274<3>()
S<3 ? 1 : 3 != 2> g275<3>()
void g276<3>(S<3 != 2>, S<3 != 2 ? 1 : 2>)
SA<3 != 2, 3 ? 1 : 2> g277<3>()
SA<3 ? 1 : 2, 3 != 2> g278<3>()
S<W<3 != 2>::value ? 1 : 2> g279<3>()
V<S<3 != 2 ? 1 : 2>, S<3>> g280<3>()
S<3 != 2 ? 1 : 2>* g281<3>(S<3 != 2>*)
decltype(fp != fp) g282<A>(A)
decltype(sizeof (A) ? fp != fp : fp != fp) g283<A>(A)
V<decltype(fp != fp), S<sizeof (A) ? 1 : 2>> g284<A>(A)
void g285<A>(A, decltype((A)() != (A)())*, S<sizeof (A) ? 1 : 2>)
SA<sizeof (A) ? 1 : 2, &A::operator!=> g286<A>()
void g287<A>(SA<sizeof (A) ? 1 : 2, &A::operator!=>)
SA<&A::operator!=, sizeof (A) ? 1 : 2> g288<A>()
S<3 && 2> g289<3>()
S<3 && 2 ? 1 : 2> g290<3>()
S<3 && 2 ? 1 : 2> g291<3>()
S<3 ? 3 && 2 : 2> g292<3>()
S<3 ? 1 : 3 && 2> g293<3>()
void g294<3>(S<3 && 2>, S<3 && 2 ? 1 : 2>)
SA<3 && 2, 3 ? 1 : 2> g295<3>()
SA<3 ? 1 : 2, 3 && 2> g296<3>()
S<W<3 && 2>::value ? 1 : 2> g297<3>()
V<S<3 && 2 ? 1 : 2>, S<3>> g298<3>()
S<3 && 2 ? 1 : 2>* g299<3>(S<3 && 2>*)
decltype(fp && fp) g300<A>(A)
decltype(sizeof (A) ? fp && fp : fp && fp) g301<A>(A)
V<decltype(fp && fp), S<sizeof (A) ? 1 : 2>> g302<A>(A)
void g303<A>(A, decltype((A)() && (A)())*, S<sizeof (A) ? 1 : 2>)
SA<sizeof (A) ? 1 : 2, &A::operator&&> g304<A>()
void g305<A>(SA<sizeof (A) ? 1 : 2, &A::operator&&>)
SA<&A::operator&&, sizeof (A) ? 1 : 2> g306<A>()
S<3 || 2> g307<3>()
S<3 || 2 ? 1 : 2> g308<3>()
S<3 || 2 ? 1 : 2> g309<3>()
S<3 ? 3 || 2 : 2> g310<3>()
S<3 ? 1 : 3 || 2> g311<3>()
void g312<3>(S<3 || 2>, S<3 || 2 ? 1 : 2>)
SA<3 || 2, 3 ? 1 : 2> g313<3>()
SA<3 ? 1 : 2, 3 || 2> g314<3>()
S<W<3 || 2>::value ? 1 : 2> g315<3>()
V<S<3 || 2 ? 1 : 2>, S<3>> g316<3>()
S<3 || 2 ? 1 : 2>* g317<3>(S<3 || 2>*)
decltype(fp || fp) g318<A>(A)
decltype(sizeof (A) ? fp || fp : fp || fp) g319<A>(A)
V<decltype(fp || fp), S<sizeof (A) ? 1 : 2>> g320<A>(A)
void g321<A>(A, decltype((A)() || (A)())*, S<sizeof (A) ? 1 : 2>)
SA<sizeof (A) ? 1 : 2, &A::operator||> g322<A>()
void g323<A>(SA<sizeof (A) ? 1 : 2, &A::operator||>)
SA<&A::operator||, sizeof (A) ? 1 : 2> g324<A>()
S<(3, 2)> g325<3>()
S<(3, 2) ? 1 : 2> g326<3>()
S<(3, 2 ? 1 : 2)> g327<3>()
S<3 ? 3, 2 : 2> g328<3>()
S<3 ? 1 : (3, 2)> g329<3>()
void g330<3>(S<(3, 2)>, S<(3, 2 ? 1 : 2)>)
SA<(3, 2), 3 ? 1 : 2> g331<3>()
SA<3 ? 1 : 2, (3, 2)> g332<3>()
S<W<(3, 2)>::value ? 1 : 2> g333<3>()
V<S<(3, 2 ? 1 : 2)>, S<3>> g334<3>()
S<(3, 2 ? 1 : 2)>* g335<3>(S<(3, 2)>*)
decltype(fp, fp) g336<A>(A)
decltype(sizeof (A) ? fp, fp : fp, fp) g337<A>(A)
V<decltype(fp, fp), S<sizeof (A) ? 1 : 2>> g338<A>(A)
void g339<A>(A, decltype((A)(), (A)())*, S<sizeof (A) ? 1 : 2>)
SA<sizeof (A) ? 1 : 2, &A::operator,> g340<A>()
void g341<A>(SA<sizeof (A) ? 1 : 2, &A::operator,>)
SA<&A::operator,, sizeof (A) ? 1 : 2> g342<A>()
decltype(fp <=> fp) g343<A>(A)
decltype(sizeof (A) ? fp <=> fp : fp <=> fp) g344<A>(A)
V<decltype(fp <=> fp), S<sizeof (A) ? 1 : 2>> g345<A>(A)
void g346<A>(A, decltype((A)() <=> (A)())*, S<sizeof (A) ? 1 : 2>)
SA<sizeof (A) ? 1 : 2, &A::operator<=>> g347<A>()
void g348<A>(SA<sizeof (A) ? 1 : 2, &A::operator<=>>)
SA<&A::operator<=>, sizeof (A) ? 1 : 2> g349<A>()
SA<sizeof (A) ? 1 : 2, &A::operator=> g350<A>()
SA<sizeof (A) ? 1 : 2, &A::operator-=> g352<A>()
SA<sizeof (A) ? 1 : 2, &A::operator<<=> g359<A>()
SA<sizeof (A) ? 1 : 2, &A::operator>>=> g360<A>()
SA<sizeof (A) ? 1 : 2, &A::operator->*> g361<A>()
SA<sizeof (A) ? 1 : 2, &A::operator()> g362<A>()
SA<sizeof (A) ? 1 : 2, &A::operator[]> g363<A>()
SA<sizeof (A) ? 1 : 2, &A::operator++> g364<A>()
SA<sizeof (A) ? 1 : 2, &A::operator!> g366<A>()
SA<sizeof (A) ? 1 : 2, &A::operator~> g367<A>()
SA<sizeof (A) ? 1 : 2, &A::operator->> g368<A>()
decltype(fp = fp) g369<int>(int)
V<decltype(fp = fp), S<sizeof (int) ? 1 : 2>> g370<int>(int)
decltype(sizeof (int) ? fp = fp : fp) g371<int>(int)
decltype(fp += fp) g372<int>(int)
V<decltype(fp += fp), S<sizeof (int) ? 1 : 2>> g373<int>(int)
decltype(sizeof (int) ? fp += fp : fp) g374<int>(int)
decltype(fp -= fp) g375<int>(int)
V<decltype(fp -= fp), S<sizeof (int) ? 1 : 2>> g376<int>(int)
decltype(sizeof (int) ? fp -= fp : fp) g377<int>(int)
decltype(fp *= fp) g378<int>(int)
V<decltype(fp *= fp), S<sizeof (int) ? 1 : 2>> g379<int>(int)
decltype(sizeof (int) ? fp *= fp : fp) g380<int>(int)
decltype(fp /= fp) g381<int>(int)
V<decltype(fp /= fp), S<sizeof (int) ? 1 : 2>> g382<int>(int)
decltype(sizeof (int) ? fp /= fp : fp) g383<int>(int)
decltype(fp %= fp) g384<int>(int)
V<decltype(fp %= fp), S<sizeof (int) ? 1 : 2>> g385<int>(int)
decltype(sizeof (int) ? fp %= fp : fp) g386<int>(int)
decltype(fp ^= fp) g387<int>(int)
V<decltype(fp ^= fp), S<sizeof (int) ? 1 : 2>> g388<int>(int)
decltype(sizeof (int) ? fp ^= fp : fp) g389<int>(int)
decltype(fp &= fp) g390<int>(int)
V<decltype(fp &= fp), S<sizeof (int) ? 1 : 2>> g391<int>(int)
decltype(sizeof (int) ? fp &= fp : fp) g392<int>(int)
decltype(fp |= fp) g393<int>(int)
V<decltype(fp |= fp), S<sizeof (int) ? 1 : 2>> g394<int>(int)
decltype(sizeof (int) ? fp |= fp : fp) g395<int>(int)
decltype(fp <<= fp) g396<int>(int)
V<decltype(fp <<= fp), S<sizeof (int) ? 1 : 2>> g397<int>(int)
decltype(sizeof (int) ? fp <<= fp : fp) g398<int>(int)
decltype(fp >>= fp) g399<int>(int)
V<decltype(fp >>= fp), S<sizeof (int) ? 1 : 2>> g400<int>(int)
decltype(sizeof (int) ? fp >>= fp : fp) g401<int>(int)
S<-3> g402<3>()
S<-3 ? 1 : 2> g403<3>()
void g404<3>(S<-3 ? -3 : 2>)
S<+3> g405<3>()
S<+3 ? 1 : 2> g406<3>()
void g407<3>(S<+3 ? +3 : 2>)
S<!3> g408<3>()
S<!3 ? 1 : 2> g409<3>()
void g410<3>(S<!3 ? !3 : 2>)
S<~3> g411<3>()
S<~3 ? 1 : 2> g412<3>()
void g413<3>(S<~3 ? ~3 : 2>)
decltype(fp++) g414<int>(int)
decltype(sizeof (int) ? fp++ : fp++) g415<int>(int)
decltype(fp--) g416<int>(int)
decltype(sizeof (int) ? fp-- : fp--) g417<int>(int)
decltype(*(&fp)) g418<int>(int)
decltype(sizeof (int) ? *(&fp) : *(&fp)) g419<int>(int)
decltype(&fp) g420<int>(int)
decltype(sizeof (int) ? &fp : &fp) g421<int>(int)
decltype(fp++) g422<int>(int)
V<decltype(fp++), S<sizeof (int) ? 1 : 2>> g423<int>(int)
decltype(sizeof (int) ? fp++ : fp) g424<int>(int)
decltype(fp--) g425<int>(int)
V<decltype(fp--), S<sizeof (int) ? 1 : 2>> g426<int>(int)
decltype(sizeof (int) ? fp-- : fp) g427<int>(int)
S<((1, 2) + ...)> g428<1, 2>()
S<(... + (1, 2))> g429<1, 2>()
S<(0 + ... + (1, 2))> g430<1, 2>()
S<((1 < 2, 2 < 2) && ...) ? 1 : 2> g431<1, 2>()
S<((1 > 2, 2 > 2) || ...) ? 1 : 2> g432<1, 2>()
S<((1 >= 2, 2 >= 2) , ...) ? 1 : 2> g433<1, 2>()
S<(... << (1, 2)) ? 1 : 2> g434<1, 2>()
S<((1, 2) >> ...) ? 1 : 2> g435<1, 2>()
S<((1, 2) - ...) ? 1 : 2> g437<1, 2>()
S<sizeof...(int, char)> g439<int, char>()
S<(sizeof...(int, char) > 1) ? 1 : 2> g440<int, char>()
S<sizeof...(int, char) >= 1 ? 1 : 2> g441<int, char>()
void g442<int, char>(S<sizeof...(int, char) < 1 ? 1 : 2>)
S<alignof (int)> g443<int>()
S<alignof (int) >= 4 ? 1 : 2> g444<int>()
S<alignof (int) < 4 ? alignof (int) : 2> g445<int>()
S<noexcept ((int)())> g446<int>()
S<noexcept ((int)()) ? 1 : 2> g447<int>()
void g448<int>(S<noexcept ((int)() < (int)()) ? 1 : 2>)
S<(sizeof (int) > 2) ? 1 : 2> g449<int>()
S<sizeof (int) < 2 ? 1 : 2> g450<int>()
S<sizeof (int) << 2 ? 1 : 2> g451<int>()
S<(sizeof (int) >> 2) ? 1 : 2> g452<int>()
S<static_cast<int>(3)> g454<3>()
S<static_cast<int>(3) ? 1 : 2> g455<3>()
S<(static_cast<int>(3) > 1) ? 1 : 2> g456<3>()
S<static_cast<int>(3) < 1 ? 1 : 2> g457<3>()
S<(int)(3) ? 1 : 2> g458<3>()
S<(int)(3) ? 1 : 2> g459<3>()
S<int{3} ? 1 : 2> g460<3>()
decltype(reinterpret_cast<char*>(fp) ? fp : fp) g461<int>(int*)
decltype(const_cast<int const*>(fp)) g462<int>(int*)
decltype(static_cast<int>(fp) < fp ? 1 : 2) g464<int>(int)
decltype(new int) g466<int>(int)
decltype(new[] int) g467<int>(int)
decltype(::new int(fp)) g468<int>(int)
__Z4g469IiEDTnw_T_ilfp_EES0_
decltype(delete fp) g470<int>(int*)
decltype(::delete fp) g471<int>(int*)
decltype(delete[] fp) g472<int>(int*)
decltype(::delete[] fp) g473<int>(int*)
decltype(sizeof (int) ? new int : nullptr) g474<int>(int)
decltype(operator new(unsigned long)(sizeof (int))) g475<int>(int)
V<decltype(::new int), S<sizeof (int) ? 1 : 2>> g476<int>(int)
decltype(fp.next->next->a) g477<Q>(Q)
decltype((fp.next->arr)[1] ? fp.a : 2) g478<Q>(Q)
decltype((fp->next->a > 1) ? 1 : 2) g479<Q>(Q*)
decltype(fp->self().next->a >= 1 ? 1 : 2) g480<Q>(Q*)
decltype(fp.f() < 1 ? fp.f() : 2) g481<Q>(Q)
decltype(fp.*&Q::a ? 1 : 2) g482<Q, &Q::a>(Q)
decltype(fp->*&Q::a ? 1 : 2) g483<Q, &Q::a>(Q*)
decltype(fp->*&Q::f() const() ? 1 : 2) g484<Q, &Q::f() const>(Q*)
decltype((fp.*&Q::f() const() > 1) ? 1 : 2) g485<Q, &Q::f() const>(Q)
S<((&gq)->arr)[1] ? 1 : 2> g486<&gq>()
S<((&gq)->a > 0) ? 1 : 2> g487<&gq>()
S<&(&gq)->a != nullptr ? 1 : 2> g488<&gq>()
S<garr[1] ? 1 : 2> g489<1>()
S<(garr[1] > 1) ? garr[1] : 2> g490<1>()
S<1[garr] ? 1 : 2> g491<1>()
S<cf(int)(1) ? 1 : 2> g492<1>()
S<cf(int)(1) < cf(int)(2) ? 1 : 2> g493<1>()
S<cf(int)(1 > 2 ? 1 : 2) ? 1 : 2> g494<1>()
void g495<1>(S<(cf(int)(1) >> 1) ? 1 : 2>)
decltype(fp.operator->()) g498<A>(A)
decltype(fp.operator>(fp) ? 1 : 2) g500<A>(A)
decltype(fp.operator<(fp) ? 1 : 2) g501<A>(A)
decltype(fp.operator>>(1)) g502<A>(A)
decltype(fp.operator--()) g503<A>(A)
decltype(operator"" _x(unsigned long long)(sizeof (int)) ? 1 : 2) g504<int>(int)
S<W<(3 > 1)>::value ? 1 : 2> g505<3>()
S<W<3 < 1>::value < 2 ? 1 : 2> g506<3>()
V<S<(3 > 1)>, S<3 < 1 ? 1 : 2>, S<3 >= 1>> g507<3>()
S<W<(W<3 >> 1>::value > 1)>::value ? 1 : 2> g508<3>()
void g509<3>(V<S<3 >= 2>>, S<3 <= 2 ? 1 : 2>)
S<(V<int>::value > 0) ? 1 : 2> g510<int>()
S<V<int, V<int>>::value >= 0 ? 1 : 2> g511<int>()
S<W<1>::value < 2 ? 1 : 2> g512<W<1>>()
g514()
g515()
g516()
g517()
g518()
auto g519<int>()
__Z4g520IUlvE_Evv
__Z4g521I8lam_g521MUliE_EvT_
g522(int (*)(int))
g524()
g525()
g526()
g527()
S<(sizeof (int) + operator"" _x(unsigned long long)(1ull) > 2) ? 1 : 2> g528<int>()
SL<Lit{"::c"}> g532<Lit{"::c"}>()
SL<Lit{"a:b"}> g532<Lit{"a:b"}>()
SL<Lit{"a?b"}> g532<Lit{"a?b"}>()
S<(char)58 == (char)58 ? 1 : 2> g533<(char)58>()
S<1> g534<(char)58>()
S<1> g534<(char)60>()
S<1> g534<(char)62>()
S<1> g534<(char)63>()
void g537<int>(int)
void g538<int>(int)
V<decltype(fp ? 1 : 2)...> g539<int, int>(int, int)
S<(3, 3 > 1 ? 1 : 2)> g540<3>()
S<(3 > 1) ? 3 < 2 ? 1 : 2 : 3> g542<3>()
S<(3 > 1) ? 3 < 2 ? 1 : 2 : 3 >= 3 ? 3 : 4> g543<3>()
S<&gq ? 1 : 2> g544<&gq>()
S<&Q::a ? 1 : 2> g544<&Q::a>()
S<&Q::f() const ? 1 : 2> g544<&Q::f() const>()
decltype(fp.~Q()) g553<Q>(Q)
decltype(fp->~Q(), 1 ? 1 : 2) g554<Q>(Q*)
decltype((sizeof (Q) > 1) ? &Q::a : &Q::a) g556<Q>(Q)
decltype(&A::operator() ? 1 : 2) g557<A>(A)
decltype(fp.T::f() ? 1 : 2) g558<Q>(Q)
decltype(fp.::Q::f() ? 1 : 2) g559<Q>(Q)
decltype(cf(int)(fp) ? garr[fp] : garr[0]) g560<int>(int)
call_g546()
operator"" _g529(char const*)
int operator"" _g530<(char)49, (char)58>()
int operator"" _g531<Lit{"a:b"}>()
S<(3 > 1) ? 1 : 2> (anonymous namespace)::g546<3>()
$_0::__invoke(int)
C_g523::m(int)
D_g551<S<1>>::~D_g551()
S<(sizeof (char) > 4ul) ? 1 : 2> M_g545<int>::f<char>()
S<(3 > 1) ? 1 : 2> n_g547::v1::f<3>()
X_g552::g()
A::operator<=>(A const&) const
$_0::operator()(int) const
$_0::operator int (*)(int)() const
bool O_g549::operator<<int>(int) const
bool O_g550::operator><int>(int) const
bool O_g550::operator<<<int>(int) const
bool O_g550::operator-<int>(int) const
bool O_g550::operator<=><int>(int) const
std::__1::strong_ordering::equal
non-virtual thunk to X_g552::g()
typeinfo for X_g552
typeinfo name for X_g552
vtable for X_g552
g514()::$_0::operator()(int) const
__ZZ4g515vENK3$_0clIiJiEEEDaT_DpT0_
__ZZ4g516vENK3$_0clIiEEDaT_
__ZZ4g517vENK3$_0clILi3EEEDa1SIXqugtT_Li1ELi1ELi2EEE
__ZZ4g518vENK3$_0clIiEEDaT_
auto g519<int>()::'lambda'(int)::operator()(int) const
g524()::L::f()
g525()::$_0::f()
g526()::$_0::operator()(int) const
__ZZ4g527vEN3$_0clIiEEDaT_
C_g523::m(int)::'lambda'()::operator()() const
__ZZZ4g518vENK3$_0clIiEEDaT_ENKUlS2_E_clIiEES1_S2_
SL8<L8{"a:b"}> h0<L8{"a:b"}>()
SL8<L8{"a/*b"}> h1<L8{"a/*b"}>()
SL8<L8{"a*/b"}> h2<L8{"a*/b"}>()
SL8<L8{"a//b"}> h3<L8{"a//b"}>()
SL8<L8{"a<b"}> h4<L8{"a<b"}>()
SL8<L8{"a>b"}> h5<L8{"a>b"}>()
SL8<L8{"a(b"}> h6<L8{"a(b"}>()
SL8<L8{"a)b"}> h7<L8{"a)b"}>()
SL8<L8{"a]b"}> h8<L8{"a]b"}>()
SL8<L8{"a?b:c"}> h9<L8{"a?b:c"}>()
SLN<L8{"a:b"}, (3 > 1) ? 1 : 2> k0<3>()
SLN<L8{"a/*b"}, (3 > 1) ? 1 : 2> k1<3>()
SLN<L8{"a*/b"}, (3 > 1) ? 1 : 2> k2<3>()
SLN<L8{"a//b"}, (3 > 1) ? 1 : 2> k3<3>()
SLN<L8{"a<b"}, (3 > 1) ? 1 : 2> k4<3>()
SLN<L8{"a>b"}, (3 > 1) ? 1 : 2> k5<3>()
SLN<L8{"a(b"}, (3 > 1) ? 1 : 2> k6<3>()
SLN<L8{"a)b"}, (3 > 1) ? 1 : 2> k7<3>()
SLN<L8{"a]b"}, (3 > 1) ? 1 : 2> k8<3>()
SLN<L8{"a?b:c"}, (3 > 1) ? 1 : 2> k9<3>()
void m0<3>(SLN<L8{"a:b"}, (3 > 1) ? 1 : 2>)
void m1<3>(SLN<L8{"a/*b"}, (3 > 1) ? 1 : 2>)
void m2<3>(SLN<L8{"a*/b"}, (3 > 1) ? 1 : 2>)
void m3<3>(SLN<L8{"a//b"}, (3 > 1) ? 1 : 2>)
void m4<3>(SLN<L8{"a<b"}, (3 > 1) ? 1 : 2>)
void m5<3>(SLN<L8{"a>b"}, (3 > 1) ? 1 : 2>)
void m6<3>(SLN<L8{"a(b"}, (3 > 1) ? 1 : 2>)
void m7<3>(SLN<L8{"a)b"}, (3 > 1) ? 1 : 2>)
void m8<3>(SLN<L8{"a]b"}, (3 > 1) ? 1 : 2>)
void m9<3>(SLN<L8{"a?b:c"}, (3 > 1) ? 1 : 2>)
SA<sizeof (B0) ? 1 : 2, &B0::operator-> o0<B0>()
SA<sizeof (B1) ? 1 : 2, &B1::operator>> o1<B1>()
SA<sizeof (B2) ? 1 : 2, &B2::operator<=> o2<B2>()
SA<sizeof (B3) ? 1 : 2, &B3::operator--> o3<B3>()
SA<sizeof (B4) ? 1 : 2, &B4::operator<> o4<B4>()
SA<sizeof (B5) ? 1 : 2, &B5::operator>>> o5<B5>()
SA<sizeof (B6) ? 1 : 2, &B6::operator->> o6<B6>()
SA<sizeof (B7) ? 1 : 2, &B7::operator>=> o7<B7>()
SA<sizeof (B8) ? 1 : 2, &B8::operator<<> o8<B8>()
SA<&B0::operator-, (sizeof (B0) > 1) ? 1 : 2> p0<B0>()
SA<&B1::operator>, (sizeof (B1) > 1) ? 1 : 2> p1<B1>()
SA<&B2::operator<=, (sizeof (B2) > 1) ? 1 : 2> p2<B2>()
SA<&B3::operator--, (sizeof (B3) > 1) ? 1 : 2> p3<B3>()
SA<&B4::operator<, (sizeof (B4) > 1) ? 1 : 2> p4<B4>()
SA<&B5::operator>>, (sizeof (B5) > 1) ? 1 : 2> p5<B5>()
SA<&B6::operator->, (sizeof (B6) > 1) ? 1 : 2> p6<B6>()
SA<&B7::operator>=, (sizeof (B7) > 1) ? 1 : 2> p7<B7>()
SA<&B8::operator<<, (sizeof (B8) > 1) ? 1 : 2> p8<B8>()
V<SA<&B0::operator->, S<(sizeof (B0) > 1) ? 1 : 2>> q0<B0>()
V<SA<&B1::operator>>, S<(sizeof (B1) > 1) ? 1 : 2>> q1<B1>()
V<SA<&B2::operator<=>, S<(sizeof (B2) > 1) ? 1 : 2>> q2<B2>()
V<SA<&B3::operator-->, S<(sizeof (B3) > 1) ? 1 : 2>> q3<B3>()
V<SA<&B4::operator<>, S<(sizeof (B4) > 1) ? 1 : 2>> q4<B4>()
V<SA<&B5::operator>>>, S<(sizeof (B5) > 1) ? 1 : 2>> q5<B5>()
V<SA<&B6::operator->>, S<(sizeof (B6) > 1) ? 1 : 2>> q6<B6>()
V<SA<&B7::operator>=>, S<(sizeof (B7) > 1) ? 1 : 2>> q7<B7>()
V<SA<&B8::operator<<>, S<(sizeof (B8) > 1) ? 1 : 2>> q8<B8>()
decltype(sizeof (B0) ? &B0::operator- : &B0::operator-) r0<B0>(B0)
decltype(sizeof (B1) ? &B1::operator> : &B1::operator>) r1<B1>(B1)
decltype(sizeof (B2) ? &B2::operator<= : &B2::operator<=) r2<B2>(B2)
decltype(sizeof (B3) ? &B3::operator-- : &B3::operator--) r3<B3>(B3)
decltype(sizeof (B4) ? &B4::operator< : &B4::operator<) r4<B4>(B4)
decltype(sizeof (B5) ? &B5::operator>> : &B5::operator>>) r5<B5>(B5)
decltype(sizeof (B6) ? &B6::operator-> : &B6::operator->) r6<B6>(B6)
decltype(sizeof (B7) ? &B7::operator>= : &B7::operator>=) r7<B7>(B7)
decltype(sizeof (B8) ? &B8::operator<< : &B8::operator<<) r8<B8>(B8)
SL8<L8{":"}> h10<L8{":"}>()
SL8<L8{"x::"}> h11<L8{"x::"}>()
SL8<L8{"::1"}> h12<L8{"::1"}>()
SL8<L8{"a ::b"}> h13<L8{"a ::b"}>()
SL8<L8{"<"}> h14<L8{"<"}>()
SL8<L8{"?:"}> h15<L8{"?:"}>()
SL8<L8{"a'b"}> h16<L8{"a'b"}>()
SL8<L8{"a\"b"}> h17<L8{"a\"b"}>()
SL8<L8{"{:}"}> h18<L8{"{:}"}>()
SLN<L8{":"}, (3 > 1) ? 1 : 2> k10<3>()
SLN<L8{"x::"}, (3 > 1) ? 1 : 2> k11<3>()
SLN<L8{"::1"}, (3 > 1) ? 1 : 2> k12<3>()
SLN<L8{"a ::b"}, (3 > 1) ? 1 : 2> k13<3>()
SLN<L8{"<"}, (3 > 1) ? 1 : 2> k14<3>()
SLN<L8{"?:"}, (3 > 1) ? 1 : 2> k15<3>()
SLN<L8{"a'b"}, (3 > 1) ? 1 : 2> k16<3>()
SLN<L8{"a\"b"}, (3 > 1) ? 1 : 2> k17<3>()
SLN<L8{"{:}"}, (3 > 1) ? 1 : 2> k18<3>()
void m10<3>(SLN<L8{":"}, (3 > 1) ? 1 : 2>)
void m11<3>(SLN<L8{"x::"}, (3 > 1) ? 1 : 2>)
void m12<3>(SLN<L8{"::1"}, (3 > 1) ? 1 : 2>)
void m13<3>(SLN<L8{"a ::b"}, (3 > 1) ? 1 : 2>)
void m14<3>(SLN<L8{"<"}, (3 > 1) ? 1 : 2>)
void m15<3>(SLN<L8{"?:"}, (3 > 1) ? 1 : 2>)
void m16<3>(SLN<L8{"a'b"}, (3 > 1) ? 1 : 2>)
void m17<3>(SLN<L8{"a\"b"}, (3 > 1) ? 1 : 2>)
void m18<3>(SLN<L8{"{:}"}, (3 > 1) ? 1 : 2>)
SA<sizeof (B10) ? 1 : 2, &B10::operator<=>> o10<B10>()
SA<sizeof (B11) ? 1 : 2, &B11::operator->*> o11<B11>()
SA<sizeof (B12) ? 1 : 2, &B12::operator()> o12<B12>()
SA<sizeof (B13) ? 1 : 2, &B13::operator[]> o13<B13>()
SA<sizeof (B14) ? 1 : 2, &B14::operator!> o14<B14>()
SA<sizeof (B15) ? 1 : 2, &B15::operator~> o15<B15>()
SA<sizeof (B16) ? 1 : 2, &B16::operator&&> o16<B16>()
SA<sizeof (B17) ? 1 : 2, &B17::operator||> o17<B17>()
SA<sizeof (B18) ? 1 : 2, &B18::operator,> o18<B18>()
SA<sizeof (B19) ? 1 : 2, &B19::operator+> o19<B19>()
SA<sizeof (B20) ? 1 : 2, &B20::operator*> o20<B20>()
SA<sizeof (B21) ? 1 : 2, &B21::operator&> o21<B21>()
SA<sizeof (B22) ? 1 : 2, &B22::operator|> o22<B22>()
SA<sizeof (B23) ? 1 : 2, &B23::operator^> o23<B23>()
SA<sizeof (B24) ? 1 : 2, &B24::operator%> o24<B24>()
SA<sizeof (B25) ? 1 : 2, &B25::operator/> o25<B25>()
SA<sizeof (B26) ? 1 : 2, &B26::operator==> o26<B26>()
SA<sizeof (B27) ? 1 : 2, &B27::operator!=> o27<B27>()
SA<&B10::operator<=>, (sizeof (B10) > 1) ? 1 : 2> p10<B10>()
SA<&B11::operator->*, (sizeof (B11) > 1) ? 1 : 2> p11<B11>()
SA<&B12::operator(), (sizeof (B12) > 1) ? 1 : 2> p12<B12>()
SA<&B13::operator[], (sizeof (B13) > 1) ? 1 : 2> p13<B13>()
SA<&B14::operator!, (sizeof (B14) > 1) ? 1 : 2> p14<B14>()
SA<&B15::operator~, (sizeof (B15) > 1) ? 1 : 2> p15<B15>()
SA<&B16::operator&&, (sizeof (B16) > 1) ? 1 : 2> p16<B16>()
SA<&B17::operator||, (sizeof (B17) > 1) ? 1 : 2> p17<B17>()
SA<&B18::operator,, (sizeof (B18) > 1) ? 1 : 2> p18<B18>()
SA<&B19::operator+, (sizeof (B19) > 1) ? 1 : 2> p19<B19>()
SA<&B20::operator*, (sizeof (B20) > 1) ? 1 : 2> p20<B20>()
SA<&B21::operator&, (sizeof (B21) > 1) ? 1 : 2> p21<B21>()
SA<&B22::operator|, (sizeof (B22) > 1) ? 1 : 2> p22<B22>()
SA<&B23::operator^, (sizeof (B23) > 1) ? 1 : 2> p23<B23>()
SA<&B24::operator%, (sizeof (B24) > 1) ? 1 : 2> p24<B24>()
SA<&B25::operator/, (sizeof (B25) > 1) ? 1 : 2> p25<B25>()
SA<&B26::operator==, (sizeof (B26) > 1) ? 1 : 2> p26<B26>()
SA<&B27::operator!=, (sizeof (B27) > 1) ? 1 : 2> p27<B27>()
V<SA<&B10::operator<=>>, S<(sizeof (B10) > 1) ? 1 : 2>> q10<B10>()
V<SA<&B11::operator->*>, S<(sizeof (B11) > 1) ? 1 : 2>> q11<B11>()
V<SA<&B12::operator()>, S<(sizeof (B12) > 1) ? 1 : 2>> q12<B12>()
V<SA<&B13::operator[]>, S<(sizeof (B13) > 1) ? 1 : 2>> q13<B13>()
V<SA<&B14::operator!>, S<(sizeof (B14) > 1) ? 1 : 2>> q14<B14>()
V<SA<&B15::operator~>, S<(sizeof (B15) > 1) ? 1 : 2>> q15<B15>()
V<SA<&B16::operator&&>, S<(sizeof (B16) > 1) ? 1 : 2>> q16<B16>()
V<SA<&B17::operator||>, S<(sizeof (B17) > 1) ? 1 : 2>> q17<B17>()
V<SA<&B18::operator,>, S<(sizeof (B18) > 1) ? 1 : 2>> q18<B18>()
V<SA<&B19::operator+>, S<(sizeof (B19) > 1) ? 1 : 2>> q19<B19>()
V<SA<&B20::operator*>, S<(sizeof (B20) > 1) ? 1 : 2>> q20<B20>()
V<SA<&B21::operator&>, S<(sizeof (B21) > 1) ? 1 : 2>> q21<B21>()
V<SA<&B22::operator|>, S<(sizeof (B22) > 1) ? 1 : 2>> q22<B22>()
V<SA<&B23::operator^>, S<(sizeof (B23) > 1) ? 1 : 2>> q23<B23>()
V<SA<&B24::operator%>, S<(sizeof (B24) > 1) ? 1 : 2>> q24<B24>()
V<SA<&B25::operator/>, S<(sizeof (B25) > 1) ? 1 : 2>> q25<B25>()
V<SA<&B26::operator==>, S<(sizeof (B26) > 1) ? 1 : 2>> q26<B26>()
V<SA<&B27::operator!=>, S<(sizeof (B27) > 1) ? 1 : 2>> q27<B27>()
decltype(sizeof (B10) ? &B10::operator<=> : &B10::operator<=>) r10<B10>(B10)
decltype(sizeof (B11) ? &B11::operator->* : &B11::operator->*) r11<B11>(B11)
decltype(sizeof (B12) ? &B12::operator() : &B12::operator()) r12<B12>(B12)
decltype(sizeof (B13) ? &B13::operator[] : &B13::operator[]) r13<B13>(B13)
decltype(sizeof (B14) ? &B14::operator! : &B14::operator!) r14<B14>(B14)
decltype(sizeof (B15) ? &B15::operator~ : &B15::operator~) r15<B15>(B15)
decltype(sizeof (B16) ? &B16::operator&& : &B16::operator&&) r16<B16>(B16)
decltype(sizeof (B17) ? &B17::operator|| : &B17::operator||) r17<B17>(B17)
decltype(sizeof (B18) ? &B18::operator, : &B18::operator,) r18<B18>(B18)
decltype(sizeof (B19) ? &B19::operator+ : &B19::operator+) r19<B19>(B19)
decltype(sizeof (B20) ? &B20::operator* : &B20::operator*) r20<B20>(B20)
decltype(sizeof (B21) ? &B21::operator& : &B21::operator&) r21<B21>(B21)
decltype(sizeof (B22) ? &B22::operator| : &B22::operator|) r22<B22>(B22)
decltype(sizeof (B23) ? &B23::operator^ : &B23::operator^) r23<B23>(B23)
decltype(sizeof (B24) ? &B24::operator% : &B24::operator%) r24<B24>(B24)
decltype(sizeof (B25) ? &B25::operator/ : &B25::operator/) r25<B25>(B25)
decltype(sizeof (B26) ? &B26::operator== : &B26::operator==) r26<B26>(B26)
decltype(sizeof (B27) ? &B27::operator!= : &B27::operator!=) r27<B27>(B27)
`
	r4NamesSHA256 = "a904cb70da9fe34be02fdf65c0b5401e36183da01b64f010b5c8122c440e756a"
)

// TestParseCxxfiltNames checks that each name in cxxfiltNames, llvmNames
// and r4Names parses whole as one kernel owning the instruction after its
// label, with no terminal colon, with one, and with one followed by a "//"
// or "/* */" comment, with and without a .section directive naming the
// kernel.
func TestParseCxxfiltNames(t *testing.T) {
	for _, list := range []struct {
		name, text, sha256 string
		lines              int
	}{
		{"cxxfiltNames", cxxfiltNames, cxxfiltNamesSHA256, 307},
		{"llvmNames", llvmNames, llvmNamesSHA256, 139},
		{"r4Names", r4Names, r4NamesSHA256, 727},
	} {
		if got := fmt.Sprintf("%x", sha256.Sum256([]byte(list.text))); got != list.sha256 {
			t.Fatalf("%s SHA-256 = %s, want %s", list.name, got, list.sha256)
		}
		names := strings.Split(strings.TrimSuffix(list.text, "\n"), "\n")
		if len(names) != list.lines {
			t.Fatalf("%s: %d names, want %d", list.name, len(names), list.lines)
		}
		for _, name := range names {
			for _, term := range []string{"", ":", ": // c", ": /* c */"} {
				directive := "\t.section\t.text." + name + ",\"ax\",@progbits\n"
				body := ".text." + name + term + "\n        /*0000*/                   EXIT ;\n"
				for _, listing := range []string{directive + body, body} {
					d, err := cudasass.ParseString(listing)
					if err != nil {
						t.Errorf("ParseString(%q): %v", listing, err)
						continue
					}
					var got []string
					for _, fn := range d.Functions {
						got = append(got, fmt.Sprintf("%s/%d/%s", fn.Name, len(fn.Instructions), fn.ELFSection))
					}
					if want := []string{name + "/1/.text." + name}; !slices.Equal(got, want) {
						t.Errorf("ParseString(%q): name/instructions/section = %q, want %q", listing, got, want)
					}
				}
			}
		}
	}
}

// literalCommentNames are the names in r4Names that hold "//" inside a
// string literal.
var literalCommentNames = []string{
	`SL8<L8{"a//b"}> h3<L8{"a//b"}>()`,
	`SLN<L8{"a//b"}, (3 > 1) ? 1 : 2> k3<3>()`,
	`void m3<3>(SLN<L8{"a//b"}, (3 > 1) ? 1 : 2>)`,
}

// TestParseLiteralLineComment checks that a "//" inside a string literal
// does not begin a comment: each name in literalCommentNames, as a bare
// label, with a terminal colon, under a matching directive, and followed by
// an unquoted "// c" comment, opens one kernel and one section under its
// whole name, owning the EXIT after it.
func TestParseLiteralLineComment(t *testing.T) {
	const exit = "        /*0000*/                   EXIT ;\n"
	for _, name := range literalCommentNames {
		if !strings.Contains(r4Names, name+"\n") {
			t.Fatalf("%q is not in r4Names", name)
		}
		directive := "\t.section\t.text." + name + ",\"ax\",@progbits\n"
		for _, tt := range []struct{ variant, listing string }{
			{"bare label", ".text." + name + "\n" + exit},
			{"terminal colon", ".text." + name + ":\n" + exit},
			{"directive", directive + exit},
			{"directive and bare label", directive + ".text." + name + "\n" + exit},
			{"directive and terminal colon", directive + ".text." + name + ":\n" + exit},
			{"bare label and comment", ".text." + name + " // c\n" + exit},
			{"terminal colon and comment", ".text." + name + ": // c\n" + exit},
			{"bare label and comment holding a quote", ".text." + name + " // \"c\n" + exit},
		} {
			d, err := cudasass.ParseString(tt.listing)
			if err != nil {
				t.Errorf("%s: ParseString(%q): %v", tt.variant, tt.listing, err)
				continue
			}
			var got, secs []string
			for _, fn := range d.Functions {
				got = append(got, fmt.Sprintf("%s/%d/%s", fn.Name, len(fn.Instructions), fn.ELFSection))
			}
			for _, sec := range d.Sections() {
				secs = append(secs, fmt.Sprintf("%s/%s/%t", sec.Name, sec.ELFSection, strings.Contains(string(sec.Text), "EXIT")))
			}
			if want := []string{name + "/1/.text." + name}; !slices.Equal(got, want) {
				t.Errorf("%s: ParseString(%q): name/instructions/section = %q, want %q", tt.variant, tt.listing, got, want)
			}
			if want := []string{name + "/.text." + name + "/true"}; !slices.Equal(secs, want) {
				t.Errorf("%s: ParseString(%q): sections name/ELF section/holds EXIT = %q, want %q", tt.variant, tt.listing, secs, want)
			}
		}
	}
}

// TestParseLiteralLineCommentEdges checks the literal-aware "//" search at
// its edges: escaped quotes and character literals keep their "//", an
// unquoted "//" before a quote is a comment, and a literal left open is
// refused whether or not a "//" follows it.
func TestParseLiteralLineCommentEdges(t *testing.T) {
	for _, tt := range []struct {
		label, want string // want "" means refused
	}{
		{`.text.f<L8{"a\"//"}>() // c`, `f<L8{"a\"//"}>()`},
		{`.text.f('\'//') // c`, `f('\'//')`},
		{`.text.f<'//'>()`, `f<'//'>()`},
		{`.text.k // "c`, "k"},
		{`.text.k // 'c`, "k"},
		{`.text.f<"a // c`, ""},
		{`.text.f<'a // c`, ""},
		{`.text.f<"a\" // c`, ""},
		{`.text.k: junk // "c"`, ""},
		{`.text.k: "x" // c`, ""},
	} {
		listing := tt.label + "\n /*0000*/ EXIT ;\n"
		d, err := cudasass.ParseString(listing)
		if tt.want == "" {
			if err == nil {
				t.Errorf("ParseString(%q) accepted %d functions, want an error", listing, len(d.Functions))
			}
			continue
		}
		if err != nil {
			t.Errorf("ParseString(%q): %v", listing, err)
			continue
		}
		var got []string
		for _, fn := range d.Functions {
			got = append(got, fmt.Sprintf("%s/%d/%s", fn.Name, len(fn.Instructions), fn.ELFSection))
		}
		if want := []string{tt.want + "/1/.text." + tt.want}; !slices.Equal(got, want) {
			t.Errorf("ParseString(%q): name/instructions/section = %q, want %q", listing, got, want)
		}
	}
}

// TestParseDemangledColonExpression checks that a demangled name keeps its
// colons, with or without the label's terminal colon and with or without a
// .section directive naming the kernel: such a colon belongs to the name,
// not to a label terminator.
func TestParseDemangledColonExpression(t *testing.T) {
	for _, name := range colonExpressionNames {
		for _, term := range []string{"", ":", ": // c"} {
			label := ".text." + name + term
			directive := "\t.section\t.text." + name + ",\"ax\",@progbits\n"
			body := label + "\n        /*0000*/                   EXIT ;\n"
			for _, listing := range []string{directive + body, body} {
				d, err := cudasass.ParseString(listing)
				if err != nil {
					t.Errorf("ParseString(%q): %v", listing, err)
					continue
				}
				var got []string
				for _, fn := range d.Functions {
					got = append(got, fmt.Sprintf("%s/%d/%s", fn.Name, len(fn.Instructions), fn.ELFSection))
				}
				if want := []string{name + "/1/.text." + name}; !slices.Equal(got, want) {
					t.Errorf("ParseString(%q): name/instructions/section = %q, want %q", listing, got, want)
				}
				if secs := d.Sections(); len(secs) != 1 || secs[0].Name != name || secs[0].ELFSection != ".text."+name {
					t.Errorf("ParseString(%q): sections = %+v, want one named %q", listing, secs, name)
				}
			}
		}
	}
}

// TestParseLabelTerminalBlank checks that a blank after a label's terminal
// colon is any Unicode white space, as strings.TrimSpace defines it: a
// no-break space or a vertical tab there, alone or before a "//" comment,
// ends the label like a space does.
func TestParseLabelTerminalBlank(t *testing.T) {
	for _, label := range []string{
		".text.k:\u00a0",
		".text.k:\v",
		".text.k:\u00a0// c",
		".text.k:\v// c",
		".text.k:\u00a0/* c */\v",
	} {
		directive := "\t.section\t.text.k,\"ax\",@progbits\n"
		body := label + "\n        /*0000*/                   EXIT ;\n"
		for _, listing := range []string{directive + body, body} {
			d, err := cudasass.ParseString(listing)
			if err != nil {
				t.Errorf("ParseString(%q): %v", listing, err)
				continue
			}
			var got []string
			for _, fn := range d.Functions {
				got = append(got, fmt.Sprintf("%s/%d/%s", fn.Name, len(fn.Instructions), fn.ELFSection))
			}
			if want := []string{"k/1/.text.k"}; !slices.Equal(got, want) {
				t.Errorf("ParseString(%q): name/instructions/section = %q, want %q", listing, got, want)
			}
		}
	}
}

// TestParseHeaderSurvey runs a survey of ".text." header forms through Parse,
// each alone and under a .section directive naming the kernel the form
// should open, and checks which function, ELF section and region own the
// instruction that follows. A refused form must fail the parse rather than
// move the instruction to a kernel named by a guess. The forms are
// constructed.
func TestParseHeaderSurvey(t *testing.T) {
	const refuse = ""
	tests := []struct {
		label string
		want  string // the kernel owning the instruction, or refuse
	}{
		{".text.k: junk", refuse},
		{".text.k:\tjunk", refuse},
		{".text.k: /* a /* b */ */", refuse},
		{".text.k: /* c", refuse},
		{".text.k: /* c */ junk", refuse},
		{".text.k /* c", refuse},
		{".text.k /* c */", refuse},
		{".text.k */", refuse},
		{".text.k:junk", refuse},
		{".text.k:\u00a0junk", refuse},
		{".text.k:\vjunk", refuse},
		{".text.k /* c */:", refuse},
		{".text.decltype(fp ? 1 : 2) f<int>(int)", "decltype(fp ? 1 : 2) f<int>(int)"},
		{".text.decltype(fp ? 1 : 2) f<int>(int):", "decltype(fp ? 1 : 2) f<int>(int)"},
		{".text.decltype(fp ? 1:2) f<int>(int)", "decltype(fp ? 1:2) f<int>(int)"},
		{".text.ns::k(int)", "ns::k(int)"},
		{".text.ns::k(int):", "ns::k(int)"},
		{".text.k:", "k"},
		{".text.k: /* c */ // tail", "k"},
		{".text.k // a: b", "k"},
	}
	for _, tt := range tests {
		kernel := tt.want
		if kernel == refuse {
			kernel = "k"
		}
		directive := ".section .text." + kernel + ",\"ax\",@progbits\n"
		body := tt.label + "\n/*0000*/ EXIT ; // | 3 | | | |\n"
		for _, listing := range []string{body, directive + body} {
			d, err := cudasass.ParseString(listing)
			if tt.want == refuse {
				if err == nil {
					var got []string
					for _, fn := range d.Functions {
						got = append(got, fmt.Sprintf("%q/%d/%q", fn.Name, len(fn.Instructions), fn.ELFSection))
					}
					t.Errorf("ParseString(%q) accepted: name/instructions/section = %s, want an error", listing, got)
				} else if d != nil {
					t.Errorf("ParseString(%q) = %v and a non-nil Disassembly", listing, err)
				}
				continue
			}
			if err != nil {
				t.Errorf("ParseString(%q): %v", listing, err)
				continue
			}
			var got, secs []string
			for _, fn := range d.Functions {
				got = append(got, fmt.Sprintf("%s/%d/%s", fn.Name, len(fn.Instructions), fn.ELFSection))
			}
			for _, sec := range d.Sections() {
				secs = append(secs, sec.Name+"/"+sec.ELFSection)
			}
			if want := []string{tt.want + "/1/.text." + tt.want}; !slices.Equal(got, want) {
				t.Errorf("ParseString(%q): name/instructions/section = %q, want %q", listing, got, want)
			}
			if want := []string{tt.want + "/.text." + tt.want}; !slices.Equal(secs, want) {
				t.Errorf("ParseString(%q): sections = %q, want %q", listing, secs, want)
			}
		}
	}
}

// TestLifeRangeCountDemangled checks the register-pressure pass under
// demangled names: ParseLifeRangeCount keys each kernel by the name Parse
// gives it, so overloads that share a first word keep their own counts
// through JoinPressure.
func TestLifeRangeCountDemangled(t *testing.T) {
	press, err := cudasass.ParseLifeRangeCount(strings.NewReader(overloadListing(true)))
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]map[uint64]cudasass.RegisterPressure{
		overloadF: {0x0: {GPR: 1}, 0x10: {GPR: 2}},
		overloadD: {0x0: {GPR: 5}, 0x10: {GPR: 6}},
	}
	if fmt.Sprint(press) != fmt.Sprint(want) {
		t.Fatalf("ParseLifeRangeCount = %v, want %v", press, want)
	}
	d, err := cudasass.ParseString(overloadListing(false))
	if err != nil {
		t.Fatal(err)
	}
	if err := cudasass.JoinPressure(d, press); err != nil {
		t.Fatalf("JoinPressure: %v", err)
	}
	for name, kmap := range want {
		fn := d.Function(name)
		if fn == nil {
			t.Fatalf("no function %q", name)
		}
		for _, inst := range fn.Instructions {
			if got, want := inst.LiveGPR, kmap[inst.Address].GPR; got != want {
				t.Errorf("%q at %#x: LiveGPR = %d, want %d", name, inst.Address, got, want)
			}
		}
	}
}

// TestLifeRangeCountFixture is the mangled control: nvdisasm prints the
// live-register table after each header label, and the kernel keys must
// still be the names Parse reads from the plain disassembly of the cubin.
func TestLifeRangeCountFixture(t *testing.T) {
	const life = "../cudalowering/testdata/lowering.sm90.rr24.life.sass"
	src, err := os.ReadFile(life)
	if err != nil {
		t.Fatal(err)
	}
	press, err := cudasass.ParseLifeRangeCount(strings.NewReader(string(src)))
	if err != nil {
		t.Fatal(err)
	}
	d := parseFile(t, "../cudalowering/testdata/lowering.sm90.rr24.sass")
	var names, keys []string
	for _, fn := range d.Functions {
		names = append(names, fn.Name)
	}
	for k := range press {
		keys = append(keys, k)
	}
	slices.Sort(names)
	slices.Sort(keys)
	if len(names) == 0 || !slices.Equal(keys, names) {
		t.Fatalf("ParseLifeRangeCount kernels = %q, want %q", keys, names)
	}
	if err := cudasass.JoinPressure(d, press); err != nil {
		t.Fatalf("JoinPressure: %v", err)
	}
}

// No-flags demangled directives are constructed inputs, not retained vendor
// captures. They must agree with their labels and preserve the whole section.
func TestParseNoFlagsWholeSectionName(t *testing.T) {
	for _, name := range []string{"k", "ns::slow(float *, int)", "ns::slow(float *, double)", `"ns::slow(float *, int)"`} {
		for _, label := range []bool{false, true} {
			listing := ".section .text." + name + "\n"
			if label {
				listing += ".text." + name + ": /* complete: comment */ // tail\n"
			}
			listing += " /*0000*/ EXIT ;\n"
			d, err := cudasass.ParseString(listing)
			if err != nil {
				t.Fatal(err)
			}
			if len(d.Functions) != 1 || d.Functions[0].Name != name || d.Functions[0].ELFSection != ".text."+name || len(d.Functions[0].Instructions) != 1 {
				t.Fatalf("ParseString(%q) functions = %+v", listing, d.Functions)
			}
			sections := d.Sections()
			if len(sections) != 1 || sections[0].Name != name || sections[0].ELFSection != ".text."+name || string(sections[0].Text) != listing {
				t.Fatalf("ParseString(%q) sections = %+v", listing, sections)
			}
		}
	}
	listing := ".section .text." + overloadF + "\n /*0000*/ EXIT ;\n.section .text." + overloadD + "\n /*0000*/ EXIT ;\n"
	d, err := cudasass.ParseString(listing)
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Functions) != 2 || d.Functions[0].Name != overloadF || d.Functions[1].Name != overloadD || len(d.Sections()) != 2 {
		t.Fatalf("no-flags overloads merged: %+v", d.Functions)
	}

}

// The TestColonFlags tests are the controls for reading a ".text." label's
// terminal colon, and a ".section" directive's flags field, outside string
// and character literals. There is one test for each of the five items of
// the root control plan for this change: TestColonFlagsRequiredNames,
// TestColonFlagsLiteralColon, TestColonFlagsSectionFlags,
// TestColonFlagsRootControls and TestColonFlagsListingDigests. Each checks
// the public results of ParseString, Functions and Sections, not a helper's
// return value.

// colonFlagsColonNames and colonFlagsCommaNames are demangled names that the
// LLVM c++filt on a review host (Apple LLVM 21) printed from symbols Apple
// clang 21 compiled from generated C++20 declarations. They are every name of
// that 648-name corpus that Parse at e8fc019c0 refused under all five label
// endings of colonFlagsSuffixes (the 17 colonFlagsColonNames, whose literal
// holds a colon that blanks and "//" follow), or read as a truncated name under
// a .section directive naming it (the 25 colonFlagsCommaNames, which hold a
// comma before a double quote). Each list's SHA-256 is of its names, one per
// line, so an edit to either is deliberate.
var colonFlagsColonNames = []string{
	`AT[abi:tg] kt<S<5>{": //"}>()`,
	`AT[abi:tg] kt<S<7>{"a: //b"}>()`,
	`Q<S<6>{"a://b"}>::g(int)`,
	`decltype(fp ? 1 : 2) ns::Q<S<5>{"\"://"}, long>::h<bool>(bool)`,
	`decltype(fp ? 1 : 2) ns::Q<S<7>{"a: //b"}, long>::h<bool>(bool)`,
	`ns::Q<S<5>{"\"://"}, int>::g(int)`,
	`ns::Q<S<5>{"\"://"}, int>::operator()(int) const`,
	`ns::Q<S<7>{"a: //b"}, int>::g(int)`,
	`ns::Q<S<7>{"a: //b"}, int>::operator()(int) const`,
	`void k<S<4>{"://"}>()`,
	`void k<S<6>{"a://b"}>()`,
	`void k<S<7>{"a: //b"}>()`,
	`void ns::k<S<15>{"a: /* c */ //x"}>()`,
	`void ns::k<S<5>{": //"}>()`,
	`void ns::k<S<5>{"\"://"}>()`,
	`void ns::k<S<6>{"a://b"}>()`,
	`void ns::k<S<7>{"a: //b"}>()`,
}

const colonFlagsColonNamesSHA256 = "a67f008e2939eb410f5767a65fbeb4da0493c32c6e12069b7f3647f85491429d"

var colonFlagsCommaNames = []string{
	"AT[abi:tg] kt<S<10>{\"@{`<)-`>,\"}>()",
	`AT[abi:tg] kt<S<4>{">~,"}>()`,
	`AT[abi:tg] kt<S<4>{"^{,"}>()`,
	"AT[abi:tg] kt<S<6>{\"\\\"$`\\\\,\"}>()",
	`decltype(fp ? 1 : 2) ns::Q<S<6>{"|bb/,"}, long>::h<bool>(bool)`,
	`decltype(fp ? 1 : 2) ns::Q<S<9>{"!\t/=!b=,"}, long>::h<bool>(bool)`,
	`decltype(fp, "<char const [5]>") s1<int>(int)`,
	`decltype(fp, "<char const [5]>") s6<int>(int)`,
	`decltype(fp, "<char16_t const [2]>") s3<int>(int)`,
	`decltype(fp, "<char32_t const [3]>") s4<int>(int)`,
	`decltype(fp, "<char8_t const [4]>") s2<int>(int)`,
	`decltype(fp, "<wchar_t const [3]>") s5<int>(int)`,
	`ns::Q<S<6>{"|bb/,"}, int>::g(int)`,
	`ns::Q<S<6>{"|bb/,"}, int>::operator()(int) const`,
	`ns::Q<S<9>{"!\t/=!b=,"}, int>::g(int)`,
	`ns::Q<S<9>{"!\t/=!b=,"}, int>::operator()(int) const`,
	"void ns::k<S<10>{\"@{`<)-`>,\"}>()",
	`void ns::k<S<4>{">~,"}>()`,
	`void ns::k<S<4>{"^{,"}>()`,
	`void ns::k<S<5>{"$|!,"}>()`,
	"void ns::k<S<6>{\"\\\"$`\\\\,\"}>()",
	`void ns::k<S<6>{"\n\":-,"}>()`,
	`void ns::k<S<6>{"|bb/,"}>()`,
	`void ns::k<S<8>{"*-(]/~,"}>()`,
	`void ns::k<S<9>{"!\t/=!b=,"}>()`,
}

const colonFlagsCommaNamesSHA256 = "398b78ba6bd5f7d5d4f4c10bd470f48cfa22096902df7e6f86cf0294bf0472c9"

// colonFlagsExit is the one instruction of each kernel in the listings the
// TestColonFlags tests build.
const colonFlagsExit = "        /*0000*/                   EXIT ;\n"

// colonFlagsDirective returns the .section directive nvdisasm prints for the
// .text section of the kernel name.
func colonFlagsDirective(name string) string {
	return "\t.section\t.text." + name + ",\"ax\",@progbits\n"
}

// colonFlagsCheckList checks that names still hold the bytes sha pins.
func colonFlagsCheckList(t *testing.T, names []string, sha string) {
	t.Helper()
	if got := fmt.Sprintf("%x", sha256.Sum256([]byte(strings.Join(names, "\n")+"\n"))); got != sha {
		t.Fatalf("name list SHA-256 = %s, want %s", got, sha)
	}
}

// colonFlagsKernels checks that d, parsed from listing, holds exactly the
// kernels names, in order: for each, a Function of that name read from
// .text.<name> holding one instruction, EXIT at address 0, and a Section of
// the same name and ELF section whose Text holds that one EXIT. A kernel
// with no instruction, such as one a truncated name would open, fails.
func colonFlagsKernels(t *testing.T, listing string, d *cudasass.Disassembly, names ...string) {
	t.Helper()
	var got, want []string
	for _, fn := range d.Functions {
		s := fmt.Sprintf("function %q in %q", fn.Name, fn.ELFSection)
		for _, inst := range fn.Instructions {
			s += fmt.Sprintf(" %#x:%s", inst.Address, inst.Opcode)
		}
		got = append(got, s)
	}
	for _, sec := range d.Sections() {
		got = append(got, fmt.Sprintf("section %q in %q holding %d EXIT", sec.Name, sec.ELFSection, strings.Count(string(sec.Text), "EXIT ;")))
	}
	for _, name := range names {
		want = append(want, fmt.Sprintf("function %q in %q 0x0:EXIT", name, ".text."+name))
	}
	for _, name := range names {
		want = append(want, fmt.Sprintf("section %q in %q holding 1 EXIT", name, ".text."+name))
	}
	if !slices.Equal(got, want) {
		t.Errorf("ParseString(%q):\ngot  %q\nwant %q", listing, got, want)
	}
}

// colonFlagsWant checks that listing, which spells the header of one kernel
// once or twice and then holds one EXIT, is the one kernel want (see
// colonFlagsKernels). It then checks the same listing between two other
// kernels, so that its header neither merges want into the kernel before it
// nor lets the kernel after it into want, and that each region's Text parses
// again to its own kernel alone.
func colonFlagsWant(t *testing.T, listing, want string) {
	t.Helper()
	d, err := cudasass.ParseString(listing)
	if err != nil {
		t.Fatalf("ParseString(%q): %v", listing, err)
	}
	colonFlagsKernels(t, listing, d, want)

	const before, after = "colonflags_before", "colonflags_after"
	three := ".text." + before + ":\n" + colonFlagsExit + listing + ".text." + after + ":\n" + colonFlagsExit
	d, err = cudasass.ParseString(three)
	if err != nil {
		t.Fatalf("ParseString(%q): %v", three, err)
	}
	colonFlagsKernels(t, three, d, before, want, after)
	for _, sec := range d.Sections() {
		region, err := cudasass.ParseString(string(sec.Text))
		if err != nil {
			t.Errorf("ParseString(%q): %v", sec.Text, err)
			continue
		}
		colonFlagsKernels(t, string(sec.Text), region, sec.Name)
	}
}

// colonFlagsRefused checks that Parse refuses listing at its label line
// label, with an error naming the line that does not wrap
// ErrNotTextSection, and returns no Disassembly, so that no Function or
// Section named by a guess escapes.
func colonFlagsRefused(t *testing.T, listing, label string) {
	t.Helper()
	d, err := cudasass.ParseString(listing)
	if err == nil {
		var got []string
		for _, fn := range d.Functions {
			got = append(got, fmt.Sprintf("%q/%d/%q", fn.Name, len(fn.Instructions), fn.ELFSection))
		}
		t.Fatalf("ParseString(%q) accepted: name/instructions/section = %s, want an error", listing, got)
	}
	if d != nil {
		t.Errorf("ParseString(%q) = %v and a non-nil Disassembly", listing, err)
	}
	if errors.Is(err, cudasass.ErrNotTextSection) {
		t.Errorf("ParseString(%q) = %v, which wraps ErrNotTextSection", listing, err)
	}
	if want := fmt.Sprintf("cudasass: malformed label %q", strings.TrimSpace(label)); err.Error() != want {
		t.Errorf("ParseString(%q) = %q, want %q", listing, err, want)
	}
}

// TestColonFlagsRequiredNames is root plan item 1: each name in
// literalCommentNames, in the eight variants TestParseLiteralLineComment
// reads, is one kernel under its whole name, alone and between two others
// (see colonFlagsWant).
func TestColonFlagsRequiredNames(t *testing.T) {
	for i, name := range literalCommentNames {
		directive := colonFlagsDirective(name)
		label := ".text." + name
		for _, tt := range []struct{ variant, listing string }{
			{"bare", label + "\n" + colonFlagsExit},
			{"colon", label + ":\n" + colonFlagsExit},
			{"dir", directive + colonFlagsExit},
			{"dir-bare", directive + label + "\n" + colonFlagsExit},
			{"dir-colon", directive + label + ":\n" + colonFlagsExit},
			{"bare-comment", label + " // c\n" + colonFlagsExit},
			{"colon-comment", label + ": // c\n" + colonFlagsExit},
			{"bare-comment-quote", label + " // \"c\n" + colonFlagsExit},
		} {
			t.Run(fmt.Sprintf("%d/%s", i, tt.variant), func(t *testing.T) {
				colonFlagsWant(t, tt.listing, name)
			})
		}
	}
}

// colonFlagsSuffixes are the five endings the review corpus put after each
// label: none, a terminal colon, a terminal colon and a comment, a comment,
// and a terminal colon, a block comment and nvdisasm's live-register
// columns.
var colonFlagsSuffixes = []struct{ slug, text string }{
	{"bare", ""},
	{"colon", ":"},
	{"colon-comment", ": // c"},
	{"comment", " // c"},
	{"colon-block-comment", ":\t/* x */ //  |  1 |"},
}

// colonFlagsEscapedColonNames are constructed names in the style of
// colonFlagsColonNames whose literal holds ": //" after an escaped double
// quote, single quote or backslash.
var colonFlagsEscapedColonNames = []string{
	`void k<S<6>{"\": //"}>()`,
	`void k<S<6>{"\': //"}>()`,
	`void k<S<7>{"\\: //"}>()`,
	`void k<C{'\': //'}>()`,
	`void k<C{'\": //'}>()`,
	`void k<C{'\\: //'}>()`,
}

// colonFlagsCommentColonLabels are constructed labels with a colon inside or
// after an unquoted "//" comment, each with the kernel it names. Parse at
// e8fc019c0 refused the first two, which TestParseRefusesMalformedLabel then
// listed as malformed.
var colonFlagsCommentColonLabels = []struct{ label, want string }{
	{".text.k//:", "k"},
	{".text.k // x:", "k"},
	{".text.k // note:", "k"},
	{".text.k: // a: b", "k"},
	{".text.k: // note:", "k"},
	{`.text.k // "note:`, "k"},
	{`.text.k // 'a': b:`, "k"},
	{".text.k: /* a: */ // b:", "k"},
	{`.text.f<S{": //"}>() // note:`, `f<S{": //"}>()`},
	{`.text.f<S{": //"}>(): // note:`, `f<S{": //"}>()`},
}

// TestColonFlagsLiteralColon is root plan item 2. A colon inside a literal
// does not end a label, even where blanks and "//" follow it inside the
// literal, and a colon inside an unquoted comment ends nothing. Each name in
// colonFlagsColonNames and colonFlagsEscapedColonNames, after each of
// colonFlagsSuffixes, alone and after its .section directive, is one kernel
// under its whole name; so is each label in colonFlagsCommentColonLabels.
func TestColonFlagsLiteralColon(t *testing.T) {
	colonFlagsCheckList(t, colonFlagsColonNames, colonFlagsColonNamesSHA256)
	for _, group := range []struct {
		slug  string
		names []string
	}{
		{"real", colonFlagsColonNames},
		{"escaped", colonFlagsEscapedColonNames},
	} {
		for i, name := range group.names {
			for _, sfx := range colonFlagsSuffixes {
				body := ".text." + name + sfx.text + "\n" + colonFlagsExit
				t.Run(fmt.Sprintf("%s/%02d/%s/plain", group.slug, i, sfx.slug), func(t *testing.T) {
					colonFlagsWant(t, body, name)
				})
				t.Run(fmt.Sprintf("%s/%02d/%s/dir", group.slug, i, sfx.slug), func(t *testing.T) {
					colonFlagsWant(t, colonFlagsDirective(name)+body, name)
				})
			}
		}
	}
	for i, tt := range colonFlagsCommentColonLabels {
		body := tt.label + "\n" + colonFlagsExit
		t.Run(fmt.Sprintf("comment/%02d/plain", i), func(t *testing.T) {
			colonFlagsWant(t, body, tt.want)
		})
		t.Run(fmt.Sprintf("comment/%02d/dir", i), func(t *testing.T) {
			colonFlagsWant(t, colonFlagsDirective(tt.want)+body, tt.want)
		})
	}
}

// colonFlagsTails are directive endings after a kernel's name, each of which
// sectionFlags must read as the directive's metadata or, for "", as no
// metadata at all: flags with and without a type, the blanks nvdisasm and
// LLVM allow around them, a trailing comment, each spelling of a type, and
// further fields after the type.
var colonFlagsTails = []string{
	"",
	",\"ax\",@progbits",
	", \"ax\", @progbits",
	",\t\"ax\",\t@progbits",
	",\"ax\"",
	",\"ax\" // c",
	",\"ax\",@progbits // c",
	",\"ax\",%progbits",
	",\"ax\",\"progbits\"",
	",\"ax\",@\"SHT_PROGBITS\"",
	",\"axG\",@progbits,grp,comdat",
	",\"axG\",@progbits,\"grp\",comdat",
	",\"aM\",@progbits,4",
}

// colonFlagsLiteralCommaNames are constructed names whose comma and double
// quote lie inside a character literal, or beside an escaped quote or
// backslash in a string literal.
var colonFlagsLiteralCommaNames = []string{
	`f<',"'>()`,
	`f<'\',"'>()`,
	`f<"\",">()`,
	`f<"\\,">()`,
	`f<"'", ',"'>()`,
}

// colonFlagsComponentNames are constructed names holding, outside every
// literal, a comma and a quoted literal that more of the name follows. Only
// the fourth, whose literal is single-quoted, never resembled flags.
var colonFlagsComponentNames = []string{
	`f(a, "x", b)`,
	`f(a, "x", "y")`,
	`f(a, "x") const`,
	`f(a, 'x', b)`,
	`f(a, "x",@y)`,
}

// colonFlagsForms returns the four listings root plan item 3 reads for the
// kernel name under directive: the directive alone, and the directive
// followed by the kernel's label bare, with a terminal colon, and with a
// terminal colon and a comment.
func colonFlagsForms(name, directive string) []struct{ slug, listing string } {
	label := ".text." + name
	return []struct{ slug, listing string }{
		{"only", directive + colonFlagsExit},
		{"bare", directive + label + "\n" + colonFlagsExit},
		{"colon", directive + label + ":\n" + colonFlagsExit},
		{"colon-comment", directive + label + ": // c\n" + colonFlagsExit},
	}
}

// TestColonFlagsSectionFlags is root plan item 3. A directive's name ends at
// its flags field, not at a comma and quote inside a literal or at a comma
// and quoted literal that more of the name follows. Each name in
// colonFlagsCommaNames, under its directive alone and followed by its
// label, is one kernel under its whole name, with no kernel holding no
// instruction; so is each name under each of colonFlagsTails, and each name
// in colonFlagsLiteralCommaNames and colonFlagsComponentNames with and
// without flags. A directive naming such a name in a code section other
// than .text names the whole section in its refusal.
func TestColonFlagsSectionFlags(t *testing.T) {
	colonFlagsCheckList(t, colonFlagsCommaNames, colonFlagsCommaNamesSHA256)
	for i, name := range colonFlagsCommaNames {
		for _, c := range colonFlagsForms(name, colonFlagsDirective(name)) {
			t.Run(fmt.Sprintf("real/%02d/%s", i, c.slug), func(t *testing.T) {
				colonFlagsWant(t, c.listing, name)
			})
		}
		for j, tail := range colonFlagsTails {
			for _, c := range colonFlagsForms(name, "\t.section\t.text."+name+tail+"\n")[:2] {
				t.Run(fmt.Sprintf("tail/%02d/%02d/%s", i, j, c.slug), func(t *testing.T) {
					colonFlagsWant(t, c.listing, name)
				})
			}
		}
		for _, tt := range []struct{ slug, section, flags string }{
			{"capmerc", ".nv.capmerc.text." + name, `""`},
			{"other", ".nv.other." + name, `"ax"`},
		} {
			listing := "\t.section\t" + tt.section + "," + tt.flags + ",@progbits\n" + colonFlagsExit
			t.Run(fmt.Sprintf("nontext/%02d/%s", i, tt.slug), func(t *testing.T) {
				_, err := cudasass.ParseString(listing)
				want := fmt.Sprintf("cudasass: section %s: instruction at 0x0000: %v", tt.section, cudasass.ErrNotTextSection)
				if err == nil || !errors.Is(err, cudasass.ErrNotTextSection) || err.Error() != want {
					t.Errorf("ParseString(%q) = %v, want %q", listing, err, want)
				}
			})
		}
	}
	for _, group := range []struct {
		slug  string
		names []string
	}{
		{"constructed", colonFlagsLiteralCommaNames},
		{"component", colonFlagsComponentNames},
	} {
		for i, name := range group.names {
			for _, c := range colonFlagsForms(name, colonFlagsDirective(name)) {
				t.Run(fmt.Sprintf("%s/%02d/flags-%s", group.slug, i, c.slug), func(t *testing.T) {
					colonFlagsWant(t, c.listing, name)
				})
			}
			for _, c := range colonFlagsForms(name, ".section .text."+name+"\n")[:2] {
				t.Run(fmt.Sprintf("%s/%02d/noflags-%s", group.slug, i, c.slug), func(t *testing.T) {
					colonFlagsWant(t, c.listing, name)
				})
			}
		}
	}
}

// colonFlagsRootLabels are the twenty ".text." lines of the root control
// replay, each with the kernel it names, or "" where Parse must refuse it:
// the first twelve, the four binding refusals ".text.k:junk",
// ".text.k:\u00a0junk", ".text.k:\vjunk" and ".text.k /* c */:" among them.
var colonFlagsRootLabels = []struct{ label, want string }{
	{".text.k: junk", ""},
	{".text.k:\tjunk", ""},
	{".text.k: /* a /* b */ */", ""},
	{".text.k: /* c", ""},
	{".text.k: /* c */ junk", ""},
	{".text.k /* c", ""},
	{".text.k /* c */", ""},
	{".text.k */", ""},
	{".text.k:junk", ""},
	{".text.k:\u00a0junk", ""},
	{".text.k:\vjunk", ""},
	{".text.k /* c */:", ""},
	{".text.decltype(fp ? 1 : 2) f<int>(int)", "decltype(fp ? 1 : 2) f<int>(int)"},
	{".text.decltype(fp ? 1 : 2) f<int>(int):", "decltype(fp ? 1 : 2) f<int>(int)"},
	{".text.decltype(fp ? 1:2) f<int>(int)", "decltype(fp ? 1:2) f<int>(int)"},
	{".text.ns::k(int)", "ns::k(int)"},
	{".text.ns::k(int):", "ns::k(int)"},
	{".text.k:", "k"},
	{".text.k: /* c */ // tail", "k"},
	{".text.k // a: b", "k"},
}

// colonFlagsEdges are constructed ".text." lines at the edges of reading
// literals and comments: literals left open, a final backslash, escaped
// closing quotes, comment tokens inside and outside literals, a colon whose
// tail is a literal, and Unicode blanks after a terminal colon. Each has the
// kernel it names, or "" where Parse must refuse it.
var colonFlagsEdges = []struct{ label, want string }{
	{`.text.f<"a:`, ""},
	{`.text.f<"a: // c`, ""},
	{`.text.f<'a:`, ""},
	{`.text.f<'a: // c`, ""},
	{`.text.f<"a\`, ""},
	{`.text.f<"a\":`, ""},
	{`.text.f<'a\':`, ""},
	{`.text.f<"a\\"b":`, ""},
	{`.text.k /* "x */:`, ""},
	{`.text.f<S{"/*"}> /* c */:`, ""},
	{".text.f<S{\": //\"}>():\u00a0junk", ""},
	{`.text.f<S{"a"}>(): "b: //"`, ""},
	{`.text.f<S{"a\": //"}>()`, `f<S{"a\": //"}>()`},
	{`.text.f<S{"a\\"}>(): // c`, `f<S{"a\\"}>()`},
	{`.text.f<S{"/*"}>():`, `f<S{"/*"}>()`},
	{`.text.f<S{"*/"}>() // c`, `f<S{"*/"}>()`},
	{`.text.k: /* "x */`, "k"},
	{`.text.k: /* ': */ // c`, "k"},
	{".text.f<S{\": //\"}>():\u00a0", `f<S{": //"}>()`},
	{".text.f<S{\": //\"}>():\v// c", `f<S{": //"}>()`},
	{".text.f<S{\": //\"}>()\u00a0// c", `f<S{": //"}>()`},
	{".text.f<S{\": //\"}>():\u3000/* c */\u00a0", `f<S{": //"}>()`},
	{`.text.f<'"'>():`, `f<'"'>()`},
	{`.text.f<"'">() // c`, `f<"'">()`},
}

// TestColonFlagsRootControls is root plan item 4. The twenty lines of
// colonFlagsRootLabels, alone and after a directive naming k as the root
// replay spelled them, keep the outcomes the replay recorded at e8fc019c0:
// a refusal returning no Disassembly, or the named kernel owning the
// instruction, with a kernel k holding none where the directive named k and
// the label another kernel. Each constructed line of acceptedByRule names
// its kernel alone and between two others (see colonFlagsWant), and each
// line of colonFlagsEdges is refused or names its kernel, alone and after a
// directive naming that kernel, or k where the line is refused.
func TestColonFlagsRootControls(t *testing.T) {
	const body = "\n/*0000*/ EXIT ; // | 3 | | | |\n"
	const directive = ".section .text.k,\"ax\",@progbits\n"
	for i, tt := range colonFlagsRootLabels {
		for _, ctx := range []struct{ slug, prefix string }{{"plain", ""}, {"dir", directive}} {
			listing := ctx.prefix + tt.label + body
			t.Run(fmt.Sprintf("root/%02d/%s", i, ctx.slug), func(t *testing.T) {
				if tt.want == "" {
					colonFlagsRefused(t, listing, tt.label)
					return
				}
				d, err := cudasass.ParseString(listing)
				if err != nil {
					t.Fatalf("ParseString(%q): %v", listing, err)
				}
				wantFns := []string{tt.want + "|.text." + tt.want + "|1"}
				wantSecs := []string{tt.want}
				if ctx.prefix != "" && tt.want != "k" {
					wantFns = append([]string{"k|.text.k|0"}, wantFns...)
					wantSecs = append([]string{"k"}, wantSecs...)
				}
				var fns, secs []string
				for _, fn := range d.Functions {
					fns = append(fns, fmt.Sprintf("%s|%s|%d", fn.Name, fn.ELFSection, len(fn.Instructions)))
				}
				for _, sec := range d.Sections() {
					secs = append(secs, sec.Name)
				}
				if !slices.Equal(fns, wantFns) || !slices.Equal(secs, wantSecs) {
					t.Errorf("ParseString(%q): functions %q, sections %q; want %q, %q", listing, fns, secs, wantFns, wantSecs)
				}
			})
		}
	}
	for i, tt := range acceptedByRule {
		body := tt.label + "\n" + colonFlagsExit
		t.Run(fmt.Sprintf("accepted/%02d/plain", i), func(t *testing.T) {
			colonFlagsWant(t, body, tt.want)
		})
		if strings.HasSuffix(tt.want, ":") {
			continue
		}
		t.Run(fmt.Sprintf("accepted/%02d/dir", i), func(t *testing.T) {
			colonFlagsWant(t, colonFlagsDirective(tt.want)+body, tt.want)
		})
	}
	for i, tt := range colonFlagsEdges {
		body := tt.label + "\n" + colonFlagsExit
		t.Run(fmt.Sprintf("edge/%02d/plain", i), func(t *testing.T) {
			if tt.want == "" {
				colonFlagsRefused(t, body, tt.label)
				return
			}
			colonFlagsWant(t, body, tt.want)
		})
		t.Run(fmt.Sprintf("edge/%02d/dir", i), func(t *testing.T) {
			if tt.want == "" {
				colonFlagsRefused(t, colonFlagsDirective("k")+body, tt.label)
				return
			}
			colonFlagsWant(t, colonFlagsDirective(tt.want)+body, tt.want)
		})
	}
}

// colonFlagsListings are the 28 committed listings the root control replay
// parsed, as paths from the repository root, each with the SHA-256 of the
// JSON encoding of its Disassembly and Sections that the replay recorded at
// e8fc019c0, the same for LF and CRLF line endings.
var colonFlagsListings = []struct{ path, sha256 string }{
	{"examples/nvidia/llama-infer/forward.sass", "63989a4af877cc28e3d226f0015de93facd8755a0f0cb5f4ac1cc8972b1dde33"},
	{"nvidia/internal/cudaview/testdata/kernel.sass", "da5cb3050be7aa9b9c362f0e45072547de5d000d646686cefbefccb8e3475921"},
	{"nvidia/x/cmd/cudahotswap/testdata/sidecar/recorded-bar/bar.sass", "f2ac77b5211efdff7316d54cf36d39cbb164424aec6457257eb9cd48bb187841"},
	{"nvidia/x/cmd/cudaprof/testdata/kernels.sass", "c93e3edb4aebf3a9a7b824352a6b8b90549fb5eadfceab3b705ae12b2e6dfdf5"},
	{"nvidia/x/cudacapture/testdata/inline.sass", "5e75b067d156338b4a24144273e66591242825c93c0478f95d522e9188880db2"},
	{"nvidia/x/cudacapture/testdata/kernels.sass", "fc7cdd1c891284d1b938c25dfbe172b076d2a2ddf41e0ec484fdbd375e10da28"},
	{"nvidia/x/cudacapture/testdata/main.sass", "e23b1be81c4d6b12485a9be0eb4239aec25e25f1ac45895cf6bd088ff3279942"},
	{"nvidia/x/cudalineinfo/testdata/forward.sass", "bc8e4e9be2e0adfa897f02a8241b078d82b4c276e59cff67520c735d06a44a18"},
	{"nvidia/x/cudalineinfo/testdata/fused_gate-fast.sass", "c8686a069c34b5121ee1eb454234f7f8e729fb62d4e28d335eff89826ae90c09"},
	{"nvidia/x/cudalineinfo/testdata/fused_gate.sass", "9b38d64160617ca38dd3dce276f31ad8bea767f40b88d57707920e64c8c86f05"},
	{"nvidia/x/cudalineinfo/testdata/fused_layered.sass", "2d32876b2b508d2f0d03c45604d424dc9741fc31a16bfcc019c81799dddaaf8c"},
	{"nvidia/x/cudalowering/testdata/lowering.sm121a.sass", "586a4d20d568e797d8eb0f2eb1599e02bc748a86b26352a27543a97ee4c0d49a"},
	{"nvidia/x/cudalowering/testdata/lowering.sm90.demangled.sass", "6323a2c015469aeadcebebe5aacd0dca82fac5675a966515afd43cf35c619096"},
	{"nvidia/x/cudalowering/testdata/lowering.sm90.rr24.life.sass", "7c81ae519925ecd3e6504173cbb2eddb417a90e2668bfeea23e6d7b6809fd172"},
	{"nvidia/x/cudalowering/testdata/lowering.sm90.rr24.sass", "ee00d046f978229958833f41a1ce439bdcc95cb179c6465968a1bc9f328a74f0"},
	{"nvidia/x/cudalowering/testdata/lowering.sm90.sass", "71412094125fcd388815ddc3f914a6b20f202a09a0c8e9190ade53f9463b6362"},
	{"nvidia/x/cudalowering/testdata/other.sm90.sass", "f3e4e188c9d4fffd6fb3c06bf905a23dda54eda68c13c518480f2ca903a86985"},
	{"nvidia/x/cudamega/testdata/golden/cross-2cta.sm_121a.sass", "c774c6db418b0040c6fead154d8fb522297a4c3d2c246c383c4c5999b1fe38fb"},
	{"nvidia/x/cudamega/testdata/golden/cta-1cta.sm_121a.sass", "a2762e013d71e0add84ab9dc7fa654279338848fa1d998b4e8f06108325d9276"},
	{"nvidia/x/cudamega/testdata/golden/device-2cta.sm_121a.sass", "de6c21b584c55af34cf9ff7386c31c191ce4da1b7baa8d1c1a53f16819586f0d"},
	{"nvidia/x/cudasass/testdata/roundtrip/sm121a-roundtrip.sass", "d8c5f6a76fb84f792e54267fd495516c40d46c601a5fe2369075f49524e81fbc"},
	{"nvidia/x/cudasass/testdata/sm121a-encoding.sass", "a0bed2c6787f26a9c3b7aece5f6844d9634da151bcf35b194e44f5e5532ad579"},
	{"nvidia/x/cudasass/testdata/sm121a-loop-13.3.sass", "16a15d92c85fa2a921360caf3ee89f106c1d76e1f9676ca5f901b8b2980c15b4"},
	{"nvidia/x/cudasass/testdata/sm121a-loop-13.4.sass", "443687df0a5d19a4f4655268a9df0cdb08ab287c890fc3326711114d48a2e448"},
	{"nvidia/x/cudasass/testdata/sm121a-reconvergence.sass", "f619b8de06d797db161d9b2d68b59f736fe8adb73f55ebcec919ac0f23f3e32c"},
	{"nvidia/x/examples/cuptibench-capture/testdata/kernels.sass", "c93e3edb4aebf3a9a7b824352a6b8b90549fb5eadfceab3b705ae12b2e6dfdf5"},
	{"nvidia/x/examples/ptx-sass-trace/testdata/kernels.sass", "6725fc696ef681d1e4e688b5fa0fc67aff6c601c0d90abc99de4a5df98013ef3"},
	{"nvidia/x/examples/ptx-sass-trace/testdata/padding.sass", "aba7c87b6d7e4fe425e140a885b519fe18bd0e8712c8f1a03d0b50c5d47945d5"},
}

// TestColonFlagsListingDigests is root plan item 5: each listing in
// colonFlagsListings, with LF and with CRLF line endings, parses to the
// Disassembly and Sections the root replay recorded, encoded as the replay
// encoded them.
func TestColonFlagsListingDigests(t *testing.T) {
	for i, tt := range colonFlagsListings {
		b, err := os.ReadFile("../../../" + tt.path)
		if err != nil {
			t.Fatal(err)
		}
		for _, encoding := range []string{"LF", "CRLF"} {
			data := string(b)
			if encoding == "CRLF" {
				data = strings.ReplaceAll(strings.ReplaceAll(data, "\r\n", "\n"), "\n", "\r\n")
			}
			t.Run(fmt.Sprintf("%02d/%s", i, encoding), func(t *testing.T) {
				d, err := cudasass.Parse(strings.NewReader(data))
				if err != nil {
					t.Fatalf("Parse(%s): %v", tt.path, err)
				}
				raw, err := json.Marshal(struct {
					D        *cudasass.Disassembly
					Sections []cudasass.Section
				}{d, d.Sections()})
				if err != nil {
					t.Fatal(err)
				}
				if got := fmt.Sprintf("%x", sha256.Sum256(raw)); got != tt.sha256 {
					t.Errorf("%s (%s): SHA-256 = %s, want %s", tt.path, encoding, got, tt.sha256)
				}
			})
		}
	}
}

// directiveRefused checks that Parse refuses listing at its .section
// directive line directive, with an error naming the line that does not
// wrap ErrNotTextSection, and returns no Disassembly, so that no Function
// or Section named by a guess escapes.
func directiveRefused(t *testing.T, listing, directive string) {
	t.Helper()
	d, err := cudasass.ParseString(listing)
	if err == nil {
		var got []string
		for _, fn := range d.Functions {
			got = append(got, fmt.Sprintf("%q/%d/%q", fn.Name, len(fn.Instructions), fn.ELFSection))
		}
		t.Fatalf("ParseString(%q) accepted: name/instructions/section = %s, want an error", listing, got)
	}
	if d != nil {
		t.Errorf("ParseString(%q) = %v and a non-nil Disassembly", listing, err)
	}
	if errors.Is(err, cudasass.ErrNotTextSection) {
		t.Errorf("ParseString(%q) = %v, which wraps ErrNotTextSection", listing, err)
	}
	if want := fmt.Sprintf("cudasass: malformed section directive %q", strings.TrimSpace(directive)); err.Error() != want {
		t.Errorf("ParseString(%q) = %q, want %q", listing, err, want)
	}
}

// directiveNames are constructed .section directives, without ".section"
// and the blank after it, at the edges of reading a directive's name: string
// and character literals left open, a final backslash, escaped closing
// quotes, a flags field left open, and names ending in a comma and a quoted
// literal. Each has the kernel it names, or "" where Parse must refuse it.
var directiveNames = []struct{ directive, want string }{
	// A literal left open in the name, with and without flags.
	{`.text.f<"a>()`, ""},
	{`.text.f<"a>(),"ax",@progbits`, ""},
	{`.text.f<'a>()`, ""},
	{`.text.f<'a>(),"ax",@progbits`, ""},
	{`.text.f<'a>(), "ax", @progbits`, ""},
	{`.nv.info.f<'a>(),"",@"SHT_CUDA_INFO"`, ""},
	// An escaped closing quote leaves the literal open.
	{`.text.f<"a\">(),"ax",@progbits`, ""},
	{`.text.f<'a\'>(),"ax",@progbits`, ""},
	{`.text.f<"a\">()`, ""},
	// A final backslash inside a literal escapes nothing and closes nothing.
	{`.text.f<"a\`, ""},
	{`.text.f<'a\`, ""},
	{`.text.f<"a\,"ax",@progbits`, ""},
	// A flags field left open is no flags field, and opens a literal.
	{`.text.k,"ax`, ""},
	{`.text.k, "ax`, ""},
	// Two commas that each begin a flags field, where the name before the
	// second could equally be the kernel's.
	{`.text.k, "a","ax",@progbits`, ""},
	{`.text.k, "", "ax", @progbits`, ""},
	{`.text.k, "x", "a"`, ""},
	// A quoted literal of flag bytes after the type is a group or other
	// field, not a second flags field: the name before it holds a type.
	{`.text.k,"ax",@progbits,"a"`, "k"},
	{`.text.k,"axG",@progbits,"x"`, "k"},
	{`.text.k,"axG",@progbits,"ax"`, "k"},
	{`.text.k,"ax",@progbits, "a"`, "k"},
	// A quoted literal holding a byte outside the flag bytes, followed by a
	// type, is a malformed flags field.
	{`.text.k,"axq",@progbits`, ""},
	{`.text.k, "lit", @progbits`, ""},
	{`.text.k,"b",@progbits`, ""},
	{`.text.k,"axq","progbits"`, ""},
	{`.text.k,"axq",@progbits,"grp",comdat`, ""},
	{`.text.k,"axq",@progbits,"ax",@progbits`, ""},
	{`.text.k,"axq",@progbits // c`, ""},
	{`.text.k,"axq",@progbits,"grp" // c`, ""},
	{`.text.k,"axq",@progbits // c, x(y)`, ""},
	// A later flags field whose name, before it, is well-formed although
	// the name before an earlier one is not.
	{`.text.k,"ax",@progbits,"a",@b,x(y),"ax",@progbits`, ""},
	// A field after the type that holds a bracket is no metadata field, so
	// the literal is no flags field: a demangled name's literal arguments
	// are followed by the bracket that closes them. clang++ -std=c++17
	// demangles _Z2s1IiEDTcl1gfp_LA2_KcELA3_S0_ELi0EEET_ to the first.
	{`.text.decltype(g(fp, "<char const [2]>", "<char const [3]>", 0)) s1<int>(int),"ax",@progbits`, `decltype(g(fp, "<char const [2]>", "<char const [3]>", 0)) s1<int>(int)`},
	{`.text.decltype(g(fp, "<char const [2]>", "<char const [3]>", 0)) s1<int>(int), "ax", @progbits`, `decltype(g(fp, "<char const [2]>", "<char const [3]>", 0)) s1<int>(int)`},
	{`.text.decltype(g(fp, "<char const [2]>", "<char const [3]>", 0)) s1<int>(int)`, `decltype(g(fp, "<char const [2]>", "<char const [3]>", 0)) s1<int>(int)`},
	{`.text.decltype(g(fp, "<char const [2]>", "<char const [3]>", fp)) s1<int>(int),"ax",@progbits`, `decltype(g(fp, "<char const [2]>", "<char const [3]>", fp)) s1<int>(int)`},
	{`.text.decltype(g(fp, "<char const [2]>", "<char const [3]>", "<char const [4]>")) s1<int>(int),"ax",@progbits`, `decltype(g(fp, "<char const [2]>", "<char const [3]>", "<char const [4]>")) s1<int>(int)`},
	{`.text.f<decltype(h("<char const [2]>", "<char const [3]>", 1))>(int),"ax",@progbits`, `f<decltype(h("<char const [2]>", "<char const [3]>", 1))>(int)`},
	{`.text.k,"axq",@progbits,f(x)`, `k,"axq",@progbits,f(x)`},
	// Nor is a field holding a blank, a single quote or a brace, and a
	// single-quoted literal is no flags field.
	{`.text.k,"axq",@progbits,1 2`, `k,"axq",@progbits,1 2`},
	{".text.k,\"axq\",@progbits,1\t2", "k,\"axq\",@progbits,1\t2"},
	{`.text.k,"axq",@progbits,'a'`, `k,"axq",@progbits,'a'`},
	{`.text.k,"axq",@progbits,x{y}`, `k,"axq",@progbits,x{y}`},
	{`.text.k,"axq",@progbits,{y}`, `k,"axq",@progbits,{y}`},
	{`.text.k, 'q', @p`, `k, 'q', @p`},
	{`.text.k,'a',@progbits`, `k,'a',@progbits`},
	// A comma, literal and type inside a literal belong to the name.
	{`.text.k '?,"q",@p,',"ax",@progbits`, `k '?,"q",@p,'`},
	{`.text.k '?,"q",@p,"x'\"","ax",@progbits`, `k '?,"q",@p,"x'\""`},
	// Terminated literals, escapes and a backslash outside every literal.
	{`.text.f<"a\"b">(),"ax",@progbits`, `f<"a\"b">()`},
	{`.text.f<"a\\">(),"ax",@progbits`, `f<"a\\">()`},
	{`.text.f<'\''>(),"ax",@progbits`, `f<'\''>()`},
	{`.text.f<'\\'>()`, `f<'\\'>()`},
	{`.text.k\`, `k\`},
	{`.text.k\,"ax",@progbits`, `k\`},
	// A comma and a quoted literal of flag bytes is the flags field; one
	// holding any other byte belongs to the name.
	{`.text.k, "a"`, "k"},
	{`.text.k, ""`, "k"},
	{`.text.k,"ax" // c`, "k"},
	{`.text.k, "lit"`, `k, "lit"`},
	{`.text.k, "lit","ax",@progbits`, `k, "lit"`},
	{`.text.k, "lit", "ax", @progbits`, `k, "lit"`},
	{`.text.k,"axG",@progbits,"grp",comdat`, "k"},
	{`.text.k,"ax","progbits"`, "k"},
}

// TestParseRefusesMalformedSectionDirective checks each line of
// directiveNames, alone, before a label, and after another kernel: a
// refusal returns no Disassembly, and an accepted directive is the one
// kernel it names, alone and between two others (see colonFlagsWant).
func TestParseRefusesMalformedSectionDirective(t *testing.T) {
	const before = ".text.colonflags_before:\n" + colonFlagsExit
	for i, tt := range directiveNames {
		directive := "\t.section\t" + tt.directive + "\n"
		if tt.want == "" {
			for _, c := range []struct{ slug, listing string }{
				{"only", directive + colonFlagsExit},
				{"label", directive + ".text.k:\n" + colonFlagsExit},
				{"after", before + directive + colonFlagsExit},
			} {
				t.Run(fmt.Sprintf("%02d/%s", i, c.slug), func(t *testing.T) {
					directiveRefused(t, c.listing, directive)
				})
			}
			continue
		}
		for _, c := range colonFlagsForms(tt.want, directive)[:3] {
			t.Run(fmt.Sprintf("%02d/%s", i, c.slug), func(t *testing.T) {
				colonFlagsWant(t, c.listing, tt.want)
			})
		}
	}
}

// llvmSectionFlagBytes are the 16 bytes LLVM's ELF assembler reads in a
// .section directive's flags field, one per row so that a reader that drops
// any one of them fails its row.
var llvmSectionFlagBytes = []string{
	"a", "e", "o", "w", "x", "M", "S", "T", "G", "R", "?",
	"c", "d", "l", "s", "y",
}

// TestParseSectionFlagBytes checks that each byte of llvmSectionFlagBytes,
// and all 16 together, is a flags field that ends the kernel's name, with
// and without a type after it, while a byte outside them, "b", belongs to
// the name when no type follows it. (With a type, as in `k,"b",@progbits`,
// the directive is refused; see directiveNames.)
func TestParseSectionFlagBytes(t *testing.T) {
	if len(llvmSectionFlagBytes) != 16 {
		t.Fatalf("len(llvmSectionFlagBytes) = %d, want 16", len(llvmSectionFlagBytes))
	}
	flags := append(slices.Clone(llvmSectionFlagBytes), strings.Join(llvmSectionFlagBytes, ""))
	for _, f := range flags {
		for _, tt := range []struct{ directive, want string }{
			{"k,\"" + f + "\",@progbits", "k"},
			{"k, \"" + f + "\"", "k"},
		} {
			t.Run(fmt.Sprintf("%q", tt.directive), func(t *testing.T) {
				colonFlagsWant(t, "\t.section\t.text."+tt.directive+"\n"+colonFlagsExit, tt.want)
			})
		}
	}
	for _, tt := range []struct{ directive, want string }{
		{`k, "b"`, `k, "b"`},
	} {
		t.Run(fmt.Sprintf("%q", tt.directive), func(t *testing.T) {
			colonFlagsWant(t, "\t.section\t.text."+tt.directive+"\n"+colonFlagsExit, tt.want)
		})
	}
}

// TestParseSectionDirectiveControls is the positive control for
// TestParseRefusesMalformedSectionDirective: each name in
// colonFlagsCommaNames, whose literal holds a comma before a double quote,
// is one kernel under its whole name under a directive with flags, with no
// flags, and with tabs around its flags and type, alone and followed by its
// label.
func TestParseSectionDirectiveControls(t *testing.T) {
	colonFlagsCheckList(t, colonFlagsCommaNames, colonFlagsCommaNamesSHA256)
	for i, name := range colonFlagsCommaNames {
		for _, form := range []struct{ slug, directive string }{
			{"flags", "\t.section\t.text." + name + ",\"ax\",@progbits\n"},
			{"noflags", ".section .text." + name + "\n"},
			{"tabs", "\t.section\t.text." + name + ",\t\"ax\",\t@progbits\n"},
		} {
			for _, c := range colonFlagsForms(name, form.directive)[:3] {
				t.Run(fmt.Sprintf("%02d/%s/%s", i, form.slug, c.slug), func(t *testing.T) {
					colonFlagsWant(t, c.listing, name)
				})
			}
		}
	}
}

// TestColonFlagsLiteralSkipInName pins sectionFlags' literal skipping: a
// comma inside a name's string literal is followed, after the literal
// closes, by a literal of flag bytes and an LLVM-looking tail. Only the
// directive's real flags field ends the name.
func TestColonFlagsLiteralSkipInName(t *testing.T) {
	const name = `k"x,"a",@t,"`
	for _, listing := range []string{
		".section .text." + name + ",\"ax\",@progbits\n/*0000*/ EXIT ;\n",
		".section .text." + name + ",\"ax\",@progbits\n.text." + name + ":\n/*0000*/ EXIT ;\n",
	} {
		d, err := cudasass.ParseString(listing)
		if err != nil {
			t.Errorf("ParseString(%q): %v", listing, err)
			continue
		}
		if len(d.Functions) != 1 || d.Functions[0].Name != name || d.Functions[0].ELFSection != ".text."+name || len(d.Functions[0].Instructions) != 1 {
			t.Errorf("ParseString(%q) functions = %+v", listing, d.Functions)
		}
	}
}
