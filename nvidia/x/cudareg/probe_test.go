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
func findPTXAS() string {
	if p := os.Getenv("PTXAS"); p != "" {
		return p
	}
	if p, err := exec.LookPath("ptxas"); err == nil {
		return p
	}
	for _, p := range []string{"/usr/local/cuda/bin/ptxas", "/usr/local/cuda-13.0/bin/ptxas"} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
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
	for _, target := range []string{"sm_90a", "sm_121a"} {
		for _, action := range []string{"dec", "inc"} {
			for _, tt := range tests {
				got := assembles(t, ptxas, target, action, tt.count)
				if got != tt.want {
					t.Errorf("%s setmaxnreg.%s %d: ptxas accepted=%v, package says legal=%v",
						target, action, tt.count, got, tt.want)
				}
			}
		}
	}
}

func assembles(t *testing.T, ptxas, target, action string, count uint32) bool {
	t.Helper()
	dir := t.TempDir()
	src := filepath.Join(dir, "probe.ptx")
	body := fmt.Sprintf(".version 9.0\n.target %s\n.address_size 64\n"+
		".visible .entry probe()\n{\n  setmaxnreg.%s.sync.aligned.u32 %d;\n  ret;\n}\n",
		target, action, count)
	if err := os.WriteFile(src, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return exec.Command(ptxas, "-arch="+target, src, "-o", filepath.Join(dir, "out.cubin")).Run() == nil
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
