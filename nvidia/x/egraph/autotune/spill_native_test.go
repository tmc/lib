package autotune_test

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/internal/cudatool"
)

// ptxasSpills matches ptxas -v's "N bytes spill stores, M bytes spill loads".
var ptxasSpills = regexp.MustCompile(`(\d+) bytes spill stores, (\d+) bytes spill loads`)

// TestAutotuneSpillCandidateSASS compiles the spill candidate and a
// register-unconstrained candidate without a device: NVRTC builds each
// cubin and nvdisasm counts STL and LDL, as the daemon does, and nvcc with
// -Xptxas -v reports ptxas's own spill bytes for the same source when nvcc
// is installed. It needs CUDA_EGRAPH_AUTOTUNE_SASS=1, libnvrtc and
// nvdisasm, and no GPU lock.
func TestAutotuneSpillCandidateSASS(t *testing.T) {
	if os.Getenv("CUDA_EGRAPH_AUTOTUNE_SASS") != "1" {
		t.Skip("CUDA_EGRAPH_AUTOTUNE_SASS=1 required")
	}
	target := os.Getenv("CUDA_EGRAPH_AUTOTUNE_NATIVE_ARCH")
	if target == "" {
		target = "sm_121a"
	}
	twin, err := autotune.TwinForTarget(target)
	if err != nil {
		t.Fatal(err)
	}
	nvdisasm, err := cudatool.LookTool("nvdisasm", "")
	if err != nil {
		t.Fatalf("nvdisasm: %v", err)
	}
	nvcc, nvccErr := cudatool.LookTool("nvcc", "")
	b := autotune.NVRTCBuilder{
		Twin:         twin,
		Target:       target,
		Entry:        gemmEntry,
		Source:       gemmSourceFor,
		Disassembler: cudasass.Disassembler{Path: nvdisasm},
	}
	for _, c := range gemmCandidates() {
		if c.Name != spillCandidate && c.Name != "t32_k8_s2" {
			continue
		}
		v, err := b.Build(context.Background(), c)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s %s: SASS STL=%d LDL=%d", c.Name, target, v.SpillStores, v.SpillLoads)
		sassSpills := v.SpillStores+v.SpillLoads > 0
		if want := c.Name == spillCandidate; sassSpills != want {
			t.Errorf("%s: STL/LDL present = %v, want %v", c.Name, sassSpills, want)
		}
		if nvccErr != nil {
			t.Logf("nvcc not found, ptxas -v skipped: %v", nvccErr)
			continue
		}
		dir := t.TempDir()
		src := filepath.Join(dir, c.Name+".cu")
		if err := os.WriteFile(src, []byte(v.Source.Code), 0o644); err != nil {
			t.Fatal(err)
		}
		out, err := exec.Command(nvcc, "-cubin", "-arch="+target, "-Xptxas", "-v", "-o", filepath.Join(dir, c.Name+".cubin"), src).CombinedOutput()
		t.Logf("%s ptxas -v:\n%s", c.Name, out)
		if err != nil {
			t.Fatalf("%s: nvcc: %v", c.Name, err)
		}
		m := ptxasSpills.FindSubmatch(out)
		if m == nil {
			t.Fatalf("%s: no spill line in ptxas -v output", c.Name)
		}
		stores, _ := strconv.Atoi(string(m[1]))
		loads, _ := strconv.Atoi(string(m[2]))
		if ptxasSpill := stores+loads > 0; ptxasSpill != sassSpills {
			t.Errorf("%s: ptxas reports %d/%d spill bytes but SASS has STL=%d LDL=%d", c.Name, stores, loads, v.SpillStores, v.SpillLoads)
		}
	}
}
