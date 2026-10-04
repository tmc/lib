package cudareg_test

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
	"github.com/tmc/lib/nvidia/x/cudareg"
)

// TestSM121aIsAdmitted pins the correction this package's target list needed.
// The hand-written list named sm_90a, sm_100a, sm_110a and sm_120a, so the
// package that models setmaxnreg rejected the one architecture this project
// has attached -- an answer nothing contradicted because nothing checked it
// against an assembler.
func TestSM121aIsAdmitted(t *testing.T) {
	for _, name := range []string{"sm_90a", "sm_100a", "sm_120a", "sm_121a"} {
		if _, err := cudareg.TargetFor(name); err != nil {
			t.Errorf("TargetFor(%q) = %v, want admitted", name, err)
		}
	}
	// The unsuffixed targets stay rejected: ptxas rejects setmaxnreg there.
	for _, name := range []string{"sm_90", "sm_100", "sm_121"} {
		if _, err := cudareg.TargetFor(name); err == nil {
			t.Errorf("TargetFor(%q) succeeded; ptxas rejects setmaxnreg on unsuffixed targets", name)
		}
	}
}

// findPTXAS locates a ptxas, or "" when none is installed.
//
// It defers to cudadevicetable.FindPTXAS rather than searching $PATH itself,
// so this package probes the same assembler the device table measures with.
// A local search would drift from it, and on a host whose distribution ships
// an older ptxas ahead of the toolkit's the two would disagree about which
// targets exist.
func findPTXAS() string {
	p, err := cudadevicetable.FindPTXAS()
	if err != nil {
		return ""
	}
	return p
}

// TestLegalCountBoundaries checks the range and granularity this package
// asserts against the assembler that decides it. It skips where no ptxas is
// installed, which makes it a real check on a CUDA host and silent elsewhere
// rather than a claim that is never tested anywhere.
func TestLegalCountBoundaries(t *testing.T) {
	ptxas := findPTXAS()
	if ptxas == "" {
		t.Skip("no ptxas found; set PTXAS to run this check")
	}
	tests := []struct {
		count uint32
		want  bool
	}{
		{16, false}, {20, false}, {23, false},
		{24, true}, {25, false}, {28, false},
		{32, true}, {96, true}, {248, true}, {256, true},
		{264, false}, {512, false},
	}
	// Each target is assembled at the oldest PTX ISA that has it, so an older
	// ptxas checks every target it knows.
	for _, tg := range []struct{ target, isa string }{
		{"sm_90a", "8.0"},
		{"sm_121a", "8.8"},
	} {
		t.Run(tg.target, func(t *testing.T) {
			if out, err := assemble(t, ptxas, tg.target, tg.isa, ""); err != nil {
				t.Skipf("ptxas %s does not assemble %s at PTX %s: %s", ptxas, tg.target, tg.isa, out)
			}
			for _, action := range []string{"dec", "inc"} {
				for _, tt := range tests {
					insn := fmt.Sprintf("setmaxnreg.%s.sync.aligned.u32 %d;", action, tt.count)
					_, err := assemble(t, ptxas, tg.target, tg.isa, insn)
					if got := err == nil; got != tt.want {
						t.Errorf("%s setmaxnreg.%s %d: ptxas accepted=%v, package says legal=%v",
							tg.target, action, tt.count, got, tt.want)
					}
				}
			}
		})
	}
}

// assemble runs ptxas on an entry holding insn, returning its output.
func assemble(t *testing.T, ptxas, target, isa, insn string) ([]byte, error) {
	t.Helper()
	dir := t.TempDir()
	src := filepath.Join(dir, "probe.ptx")
	body := fmt.Sprintf(".version %s\n.target %s\n.address_size 64\n"+
		".visible .entry probe()\n{\n  %s\n  ret;\n}\n",
		isa, target, insn)
	if err := os.WriteFile(src, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return exec.Command(ptxas, "-arch="+target, src, "-o", filepath.Join(dir, "out.cubin")).CombinedOutput()
}

// TestUnprobedTargetIsAdmitted covers the failure this package had in the
// opposite direction from a hallucinated fact: refusing on silence. Reading a
// table miss as "unsupported" made each new GPU less capable than the one
// before it, with no way to fix that but a table regeneration. setmaxnreg is
// an instruction, so a target that really lacks it fails at assembly time
// with the assembler's own message.
func TestUnprobedTargetIsAdmitted(t *testing.T) {
	const future = "sm_130a"
	if _, ok := cudadevicetable.LookupTarget(future); ok {
		t.Skipf("%s is now probed; pick a target that is not", future)
	}
	tgt, err := cudareg.TargetFor(future)
	if err != nil {
		t.Fatalf("cudareg.TargetFor(%s): %v", future, err)
	}
	if len(tgt.LegalCounts) == 0 {
		t.Error("admitted target has no legal counts")
	}

	// The contrast: a measured rejection must still be refused, or the
	// admission above is not a policy but an absence of one.
	if _, err := cudareg.TargetFor("sm_90"); err == nil {
		t.Error("sm_90 was admitted; ptxas rejects setmaxnreg there")
	}
	// And a non-target must not reach the probe at all.
	for _, bad := range []string{"hopper", "sm_", "compute_90a", ""} {
		if _, err := cudareg.TargetFor(bad); err == nil {
			t.Errorf("cudareg.TargetFor(%q) succeeded", bad)
		}
	}
}
