package cudasass_test

import (
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
