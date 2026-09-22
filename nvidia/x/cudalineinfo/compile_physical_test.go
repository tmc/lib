package cudalineinfo

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
)

// TestCompilePhysical compiles the fixture unit the way the example does and
// checks the attribution against the artifacts checked in beside it. The
// fixtures were produced by one compiler on one machine, so instruction
// counts are the compiler's to change; what must hold is that the same Go
// statements own instructions, and that withholding the #line directives
// leaves none of them owning any.
func TestCompilePhysical(t *testing.T) {
	if os.Getenv("CUDA_COMPILER_TEST") != "1" {
		t.Skip("CUDA_COMPILER_TEST=1 requires NVRTC, a CUDA device, and nvdisasm")
	}
	unit := fixtureUnit(t, "fused_gate")
	opts := Options{Arch: arch(t), Nvdisasm: os.Getenv("NVDISASM")}
	ctx := context.Background()
	a, err := Compile(ctx, unit, opts)
	if err != nil {
		t.Fatal(err)
	}
	// The two sides name the same statements by different paths: the SASS
	// a live compile emits resolves the #line file against the working
	// directory, while the checked-in disassembly carries the relative path
	// the example recorded. Comparing sites as a diagnostic prints them
	// compares what the reader is shown.
	want := sassSites(loadFixture(t, "fused_gate", ""))
	got := sassSites(a)
	if len(got) != len(want) {
		t.Errorf("attributed %d program sites, fixture has %d", len(got), len(want))
	}
	for site := range want {
		if !got[site] {
			t.Errorf("%s owns no instruction in this build", site)
		}
	}
	for site := range got {
		if !want[site] {
			t.Errorf("%s owns instructions the fixture does not attribute", site)
		}
	}

	opts.NoDirectives = true
	control, err := Compile(ctx, unit, opts)
	if err != nil {
		t.Fatal(err)
	}
	if n := len(sassSites(control)); n != 0 {
		t.Errorf("without #line directives %d program sites still own instructions", n)
	}
	if control.Remainder.Total() == 0 {
		t.Error("control attributed nothing at all, so it proves nothing")
	}
}

// sassSites returns, by name and line, the program sites that own at least
// one instruction.
func sassSites(a *Attribution) map[string]bool {
	sites := make(map[string]bool)
	for site, insts := range a.SASS {
		if len(insts) > 0 {
			sites[site.String()] = true
		}
	}
	return sites
}

// arch reports the compute capability of device 0 as an NVRTC target, so the
// test compiles for the GPU it is running on rather than for the one the
// fixtures were built with.
func arch(t *testing.T) string {
	t.Helper()
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	defer dev.Close()
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	return fmt.Sprintf("sm_%d%d", major, minor)
}
