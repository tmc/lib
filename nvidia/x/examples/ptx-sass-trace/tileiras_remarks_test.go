package main

import (
	"fmt"
	"strings"
	"testing"
)

// TestShippedRemarksMatchTileIR holds the two bundled artifacts to each other.
// The remarks YAML is embedded but only the summary reads it, so without this
// the file could be malformed, or could describe kernels the bundled tile IR
// no longer defines, and nothing would say so until a capture ran on a GPU
// host.
func TestShippedRemarksMatchTileIR(t *testing.T) {
	remarks, err := sources.ReadFile(remarksArtifact)
	if err != nil {
		t.Fatalf("read embedded %s: %v", remarksArtifact, err)
	}
	tileIR, err := sources.ReadFile(tileIRArtifact)
	if err != nil {
		t.Fatalf("read embedded %s: %v", tileIRArtifact, err)
	}
	summary, err := tileIRRemarkSummary(string(remarks), string(tileIR))
	if err != nil {
		t.Fatalf("tileIRRemarkSummary: %v", err)
	}

	lines := kernelTileIRLines(string(tileIR))
	// Every remark resolves to the kernel's own tile IR line; a summary that
	// cited some other line would still contain the kernel name, so the line
	// number is what is checked.
	for _, want := range []struct{ name, remark string }{
		{"ds4_q8_0_matvec", "TMALoadOptimized"},
		{"ds4_q8_0_matvec", "TensorCoreMMASelected"},
		{"ds4_iq2_simd", "IDP4ASelected"},
		{"ds4_smem_stage_matvec", "SharedMemoryStagingOptimized"},
		{"ds4_moe_expert_dispatch", "MoEIndirectAddressing"},
	} {
		line, ok := lines[want.name]
		if !ok {
			t.Errorf("%s defines no entry %s", tileIRArtifact, want.name)
			continue
		}
		loc := fmt.Sprintf("%s at %s:%d", want.remark, tileIRArtifact, line)
		if !strings.Contains(summary, loc) {
			t.Errorf("summary missing %q", loc)
		}
	}

	// ds4_iq2_scalar is the unoptimized twin of ds4_iq2_simd and draws no
	// remark; it must appear as a kernel the compiler was silent about, not
	// vanish from the roster.
	if !strings.Contains(summary, "ds4_iq2_scalar") {
		t.Errorf("summary does not account for ds4_iq2_scalar:\n%s", summary)
	}
	if want := fmt.Sprintf("5 remarks over 4 of %d tile IR kernels", len(lines)); !strings.Contains(summary, want) {
		t.Errorf("summary missing %q:\n%s", want, summary)
	}
}

func TestRemarkSummaryRejectsMismatch(t *testing.T) {
	tileIR, err := sources.ReadFile(tileIRArtifact)
	if err != nil {
		t.Fatalf("read embedded %s: %v", tileIRArtifact, err)
	}
	const orphan = `--- !Passed
Pass:            tileiras-tma-lowering
Name:            TMALoadOptimized
Function:        ds4_kernel_that_left
Args:
  - String:          'Lowered load_view_tko to TMA'
`
	if _, err := tileIRRemarkSummary(orphan, string(tileIR)); err == nil {
		t.Error("tileIRRemarkSummary accepted a remark naming an absent kernel")
	}
	if _, err := tileIRRemarkSummary("Pass: Memory\n--- !Passed\n", string(tileIR)); err == nil {
		t.Error("tileIRRemarkSummary accepted malformed yaml")
	}
}
