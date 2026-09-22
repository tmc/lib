package cudalineinfo

import (
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// testdata/forward.sass is nvdisasm's disassembly of testdata/forward.cu,
// the same file handWrittenSource embeds, compiled for sm_121 with line
// information. It is written by testdata/gen.go and is the tool's output
// unedited, but for the working directory that gen.go strips from the name
// NVRTC records, so the fixture names forward.cu on every machine.
//
// It is here because handWrittenSASS is hand-authored, and a hand-authored
// fixture can hold a shape no compiler emits. That is not hypothetical: this
// file's two kernels arrive mangled, and while Unit.Kernel matched names
// literally it attributed none of the 80 instructions.
func compiledDisassembly(t *testing.T) *cudasass.Disassembly {
	t.Helper()
	b, err := os.ReadFile("testdata/forward.sass")
	if err != nil {
		t.Fatal(err)
	}
	dis, err := cudasass.ParseString(string(b))
	if err != nil {
		t.Fatal(err)
	}
	return dis
}

// TestCompiledFunctions records what the compiler actually emitted: two
// kernels under their mangled symbols, the names every other test in this
// file must match against.
func TestCompiledFunctions(t *testing.T) {
	dis := compiledDisassembly(t)
	want := []struct {
		name  string
		insts int
	}{
		{"_Z7rmsnormPKfPfi", 32},
		{"_Z5gegluPKfPfi", 48},
	}
	if len(dis.Functions) != len(want) {
		t.Fatalf("functions = %d, want %d", len(dis.Functions), len(want))
	}
	for i, w := range want {
		fn := dis.Functions[i]
		if fn.Name != w.name {
			t.Errorf("function %d = %q, want %q", i, fn.Name, w.name)
		}
		if len(fn.Instructions) != w.insts {
			t.Errorf("%s = %d instructions, want %d", fn.Name, len(fn.Instructions), w.insts)
		}
	}
}

// TestAttributeCompiled attributes real compiler output. The unit names its
// kernels as the source writes them, so every case here fails to the
// subroutine bucket if the mangled symbols stop matching.
func TestAttributeCompiled(t *testing.T) {
	dis := compiledDisassembly(t)
	for _, tt := range []struct {
		name       string
		kernels    []string
		attributed int
		subs       map[string]int
	}{
		{
			name:       "both kernels",
			kernels:    []string{"geglu", "rmsnorm"},
			attributed: 80,
		},
		{
			// The other kernel's instructions are a subroutine of this
			// unit, which is what says the match is by name and not by
			// counting everything the module holds.
			name:       "one of the two",
			kernels:    []string{"geglu"},
			attributed: 48,
			subs:       map[string]int{"_Z7rmsnormPKfPfi": 32},
		},
		{
			name:       "named by symbol",
			kernels:    []string{"_Z5gegluPKfPfi", "_Z7rmsnormPKfPfi"},
			attributed: 80,
		},
		{
			name:       "neither",
			kernels:    []string{"other"},
			attributed: 0,
			subs:       map[string]int{"_Z5gegluPKfPfi": 48, "_Z7rmsnormPKfPfi": 32},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			a := Attribute(Unit{Name: "forward", Source: handWrittenSource, Kernels: tt.kernels}, "", dis)
			if a.Instructions != 80 {
				t.Fatalf("instructions = %d, want 80", a.Instructions)
			}
			if a.Attributed != tt.attributed {
				t.Errorf("attributed %d of %d, want %d", a.Attributed, a.Instructions, tt.attributed)
			}
			if a.Remainder.Total() != a.Instructions-a.Attributed {
				t.Errorf("remainder totals %d, want %d", a.Remainder.Total(), a.Instructions-a.Attributed)
			}
			if len(a.Remainder.Subroutines) != len(tt.subs) {
				t.Errorf("subroutines = %v, want %v", a.Remainder.Subroutines, tt.subs)
			}
			for name, n := range tt.subs {
				if a.Remainder.Subroutines[name] != n {
					t.Errorf("subroutine %s = %d, want %d", name, a.Remainder.Subroutines[name], n)
				}
			}
		})
	}
}

// TestAttributeCompiledSites checks that the instructions land on the lines
// of the source, not only that they were counted. Both kernels are covered
// end to end: geglu spans forward.cu:3-8 and rmsnorm forward.cu:10-15, and
// nothing is left over.
func TestAttributeCompiledSites(t *testing.T) {
	a := Attribute(Unit{Name: "forward", Source: handWrittenSource, Kernels: []string{"geglu", "rmsnorm"}}, "", compiledDisassembly(t))
	want := []struct {
		line int
		ops  int
	}{
		{3, 1}, {4, 4}, {5, 3}, {6, 5}, {7, 22}, {8, 13},
		{10, 1}, {11, 4}, {12, 3}, {13, 5}, {14, 8}, {15, 11},
	}
	got := a.Sites()
	if len(got) != len(want) {
		t.Fatalf("sites = %v, want %d sites", got, len(want))
	}
	for i, w := range want {
		site := Site{"forward.cu", w.line}
		if got[i] != site {
			t.Errorf("site %d = %v, want %v", i, got[i], site)
		}
		if n := len(a.SASS[site]); n != w.ops {
			t.Errorf("%v = %d instructions, want %d", site, n, w.ops)
		}
	}
	// Every instruction of both kernels reached a site: the sites are the
	// whole disassembly, not a part of it that happened to carry a line.
	total := 0
	for _, s := range got {
		total += len(a.SASS[s])
	}
	if total != a.Instructions {
		t.Errorf("sites hold %d instructions, want all %d", total, a.Instructions)
	}
}
