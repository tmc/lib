package cudalowering

import (
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

func TestMatchKey(t *testing.T) {
	tests := []struct {
		name string
		want string
	}{
		// The symbols nvcc writes, as they appear in testdata/lowering.sm90.ptx
		// and in the .section directives of testdata/lowering.sm90.sass.
		{"_Z10atomicDropPfPKfi", "atomicDrop"},
		{"_Z8fmaChainPdPKdS1_i", "fmaChain"},
		{"_Z13shuffleReducePfPKfi", "shuffleReduce"},

		// The same SASS after cu++filt, as the section parser hands it over:
		// the demangled signature contains spaces, so only its first word
		// survives. The name is still ahead of the parenthesis.
		{"atomicDrop(float", "atomicDrop"},
		{"fmaChain(double", "fmaChain"},
		{"shuffleReduce(float *, const float *, int)", "shuffleReduce"},

		// Namespaces and templates.
		{"_ZN5cudnn6kernel6matmulEPfS1_", "cudnn::kernel::matmul"},
		{"cudnn::kernel::matmul(float*, float*)", "cudnn::kernel::matmul"},
		{"_Z3fooIiEvPT_", "foo"},
		{"foo<int>(int*)", "foo"},

		// extern "C" and hand-written PTX carry no encoding, and are left alone.
		{"compute", "compute"},
		{"_Z", "_Z"},
		{"_Z0x", "_Z0x"},
		{"_Z99short", "_Z99short"},
		{"", ""},
		{"  compute  ", "compute"},
	}
	for _, tt := range tests {
		if got := MatchKey(tt.name); got != tt.want {
			t.Errorf("MatchKey(%q) = %q, want %q", tt.name, got, tt.want)
		}
	}
}

// TestMatchMangling pairs one PTX module with two disassemblies of the very
// same cubin, one of which has been through cu++filt. Both must pair every
// kernel: a comparison that silently drops the demangled half would report no
// divergence for any of them.
func TestMatchMangling(t *testing.T) {
	m := readPTX(t, "lowering.sm90.ptx")
	for _, name := range []string{"lowering.sm90.sass", "lowering.sm90.demangled.sass"} {
		match := Match(m, readSASS(t, name))
		if len(match.Pairs) != 5 {
			t.Errorf("%s: got %d pairs, want 5", name, len(match.Pairs))
		}
		if len(match.UnmatchedPTX) != 0 || len(match.UnmatchedSASS) != 0 {
			t.Errorf("%s: unmatched %d ptx, %d sass, want none",
				name, len(match.UnmatchedPTX), len(match.UnmatchedSASS))
		}
		for _, p := range match.Pairs {
			if MatchKey(p.PTX.Name) != MatchKey(p.SASS.Name) {
				t.Errorf("%s: paired %q with %q", name, p.PTX.Name, p.SASS.Name)
			}
		}
	}
}

func TestMatchUnrelated(t *testing.T) {
	match := Match(readPTX(t, "lowering.sm90.ptx"), readSASS(t, "other.sm90.sass"))
	if len(match.Pairs) != 0 {
		t.Errorf("got %d pairs between unrelated artifacts, want 0", len(match.Pairs))
	}
	if len(match.UnmatchedPTX) != 5 || len(match.UnmatchedSASS) != 1 {
		t.Errorf("unmatched %d ptx, %d sass, want 5 and 1",
			len(match.UnmatchedPTX), len(match.UnmatchedSASS))
	}
}

// TestMatchAmbiguous checks that two kernels sharing a match key are left
// unmatched rather than paired by guess. Overloads mangle to different symbols
// and demangle to the same first word, which is exactly when this arises.
func TestMatchAmbiguous(t *testing.T) {
	m := &cudaptx.Module{Kernels: []*cudaptx.Kernel{
		{Name: "_Z6narrowPf", Entry: true, Body: true},
		{Name: "_Z6narrowPd", Entry: true, Body: true},
		{Name: "_Z4wardPf", Entry: true, Body: true},
	}}
	d := &cudasass.Disassembly{Functions: []*cudasass.Function{
		{Name: "narrow(float"},
		{Name: "narrow(double"},
		{Name: "ward(float"},
	}}
	match := Match(m, d)
	if len(match.Pairs) != 1 {
		t.Fatalf("got %d pairs, want only the unambiguous one:\n%+v", len(match.Pairs), match.Pairs)
	}
	if match.Pairs[0].PTX.Name != "_Z4wardPf" || match.Pairs[0].SASS.Name != "ward(float" {
		t.Errorf("paired %q with %q, want the ward kernels",
			match.Pairs[0].PTX.Name, match.Pairs[0].SASS.Name)
	}
	if len(match.UnmatchedPTX) != 2 || len(match.UnmatchedSASS) != 2 {
		t.Errorf("unmatched %d ptx, %d sass, want 2 and 2",
			len(match.UnmatchedPTX), len(match.UnmatchedSASS))
	}
}

// TestMatchDeviceFunctions checks that a .func device function is not reported
// as a kernel the SASS is missing. Device functions are inlined into their
// callers and have no function of their own to pair with.
func TestMatchDeviceFunctions(t *testing.T) {
	m := &cudaptx.Module{Kernels: []*cudaptx.Kernel{
		{Name: "_Z6kernelPf", Entry: true, Body: true},
		{Name: "_Z6helperPf", Entry: false, Body: true},
		{Name: "_Z7declaredPf", Entry: true, Body: false},
	}}
	d := &cudasass.Disassembly{Functions: []*cudasass.Function{{Name: "_Z6kernelPf"}}}
	match := Match(m, d)
	if len(match.Pairs) != 1 || len(match.UnmatchedPTX) != 0 || len(match.UnmatchedSASS) != 0 {
		t.Errorf("got %d pairs, %d unmatched ptx, %d unmatched sass, want 1, 0, 0",
			len(match.Pairs), len(match.UnmatchedPTX), len(match.UnmatchedSASS))
	}
}

func TestMatchNil(t *testing.T) {
	if match := Match(nil, nil); len(match.Pairs) != 0 {
		t.Errorf("Match(nil, nil) paired %d", len(match.Pairs))
	}
	match := Match(readPTX(t, "lowering.sm90.ptx"), nil)
	if len(match.UnmatchedPTX) != 5 {
		t.Errorf("Match against nil disassembly: %d unmatched, want 5", len(match.UnmatchedPTX))
	}
}
