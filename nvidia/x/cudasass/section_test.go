package cudasass

import (
	"os"
	"strings"
	"testing"
)

// twoKernelSASS is the shape nvdisasm prints for a module with more than one
// kernel: a banner per kernel, and a trailing SYMBOLS banner that belongs to
// no kernel at all.
const twoKernelSASS = `//--------------------- .text.alpha                --------------------------
	.section	.text.alpha,"ax",@progbits
        .global         alpha
        .type           alpha,@function
        //## File "a.cu", line 10
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   EXIT ;

//--------------------- .text.beta                 --------------------------
	.section	.text.beta,"ax",@progbits
        .global         beta
        .type           beta,@function
        //## File "b.cu", line 20
        /*0000*/                   IADD3 R2, R2, 0x1, RZ ;
        /*0010*/                   EXIT ;

//--------------------- SYMBOLS --------------------------
.type alpha,@function
.type beta,@function
`

func TestSections(t *testing.T) {
	d, err := ParseString(twoKernelSASS)
	if err != nil {
		t.Fatal(err)
	}
	secs := d.Sections()
	var names []string
	for _, s := range secs {
		names = append(names, s.Name)
	}
	want := []string{"alpha", "beta"}
	if len(names) != len(want) || names[0] != want[0] || names[1] != want[1] {
		t.Fatalf("Sections() names = %v, want %v", names, want)
	}
	// The SYMBOLS banner must end beta rather than extend it. Without that,
	// beta's Text would carry the symbol table and re-parsing it would report
	// declarations as part of the kernel.
	if strings.Contains(string(secs[1].Text), "SYMBOLS") {
		t.Error("beta section absorbed the trailing SYMBOLS banner")
	}
	if !strings.HasPrefix(string(secs[0].Text), "//--------------------- .text.alpha") {
		t.Errorf("section does not start at its banner: %q", firstLine(secs[0].Text))
	}
	// Each section must carry only its own kernel's source directive; a section
	// that leaked into its neighbour would name both files.
	if strings.Contains(string(secs[0].Text), "b.cu") || strings.Contains(string(secs[1].Text), "a.cu") {
		t.Error("section text crosses a kernel boundary")
	}
}

// TestSectionsRoundTrip is the property callers actually depend on: a section's
// Text, parsed on its own, yields that kernel and nothing else, with the same
// instructions the whole listing gave.
func TestSectionsRoundTrip(t *testing.T) {
	whole, err := ParseString(twoKernelSASS)
	if err != nil {
		t.Fatal(err)
	}
	for _, sec := range whole.Sections() {
		part, err := ParseString(string(sec.Text))
		if err != nil {
			t.Fatalf("%s: re-parse: %v", sec.Name, err)
		}
		if len(part.Functions) != 1 || part.Functions[0].Name != sec.Name {
			t.Fatalf("%s: re-parse gave %d functions, want just %s", sec.Name, len(part.Functions), sec.Name)
		}
		want := whole.Function(sec.Name)
		got := part.Functions[0]
		if len(got.Instructions) != len(want.Instructions) {
			t.Fatalf("%s: %d instructions after re-parse, want %d", sec.Name, len(got.Instructions), len(want.Instructions))
		}
		for i := range got.Instructions {
			if got.Instructions[i].Raw != want.Instructions[i].Raw {
				t.Errorf("%s: instruction %d differs after re-parse:\n got %q\nwant %q",
					sec.Name, i, got.Instructions[i].Raw, want.Instructions[i].Raw)
			}
			if got.Instructions[i].SourceLine != want.Instructions[i].SourceLine {
				t.Errorf("%s: instruction %d source line %d after re-parse, want %d",
					sec.Name, i, got.Instructions[i].SourceLine, want.Instructions[i].SourceLine)
			}
		}
	}
}

// TestSectionsRealListing runs the same check against a listing nvdisasm
// actually produced, so the synthetic fixture above cannot be the only witness.
func TestSectionsRealListing(t *testing.T) {
	b, err := os.ReadFile("testdata/sm121a-loop-13.4.sass")
	if err != nil {
		t.Fatal(err)
	}
	d, err := ParseString(string(b))
	if err != nil {
		t.Fatal(err)
	}
	secs := d.Sections()
	if len(secs) != len(d.Functions) {
		t.Fatalf("%d sections for %d functions", len(secs), len(d.Functions))
	}
	for i, sec := range secs {
		if sec.Name != d.Functions[i].Name {
			t.Errorf("section %d is %q, function %d is %q", i, sec.Name, i, d.Functions[i].Name)
		}
		if strings.Contains(string(sec.Text), "SYMBOLS") {
			t.Errorf("%s absorbed the SYMBOLS banner", sec.Name)
		}
		part, err := ParseString(string(sec.Text))
		if err != nil {
			t.Fatalf("%s: re-parse: %v", sec.Name, err)
		}
		if n := len(part.Functions[0].Instructions); n != len(d.Functions[i].Instructions) {
			t.Errorf("%s: %d instructions after re-parse, want %d", sec.Name, n, len(d.Functions[i].Instructions))
		}
	}
}

func firstLine(b []byte) string {
	s := string(b)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}
