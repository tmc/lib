package main

import (
	"bytes"
	"encoding/hex"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/internal/capturefiles"
)

// fusionFixture returns the committed kernels.* capture fixture. When shift is
// nonzero the PTX line table is moved by that many lines, which is what a
// second compilation of a file of the same name produces: the tiers still
// parse, still name the same kernels and still agree on every PC, so the only
// thing that can fail is the binding between them. This is the technique
// x/cudacapture's own tiering test uses.
func fusionFixture(t *testing.T, shift int) *cudacapture.Capture {
	t.Helper()
	dir := "testdata"
	read := func(name string) []byte {
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		return b
	}
	cubin, err := hex.DecodeString(strings.Join(strings.Fields(string(read("kernels.cubin.hex"))), ""))
	if err != nil {
		t.Fatalf("decode cubin hex: %v", err)
	}
	ptx := read("kernels.ptx")
	if shift != 0 {
		ptx = shiftLocLines(t, ptx, shift)
	}
	return &cudacapture.Capture{
		Code: cudacapture.CodeFiles{Source: "kernels.cu", PTX: "kernels.ptx", CUBIN: "kernels.cubin", SASS: "kernels.sass"},
		Files: map[string][]byte{
			"kernels.cu":    read("kernels.cu"),
			"kernels.ptx":   ptx,
			"kernels.cubin": cubin,
			"kernels.sass":  read("kernels.sass"),
			// Register life ranges for the same kernels, so the pressure table
			// builds from capture files alone. Without them the loader shells
			// out to nvdisasm and the test would only ever run on a CUDA host.
			"kernels.lrm": []byte(fixtureLifeRanges),
		},
	}
}

// fixtureLifeRanges is a --life-range-mode=count disassembly of the fixture's
// one kernel, carrying the same two PCs its committed SASS does.
const fixtureLifeRanges = `
	.section	.text.compute,"ax",@progbits
compute:
.text.compute:
        /*0000*/                   MOV R1, c[0x0][0x28] ;   // |   1   |        |        |         |
        /*0010*/                   RET ;                    // |   1   |        |        |         |
`

// shiftLocLines rewrites every ".loc <file> <line> <col>" directive, and fails
// the test if it rewrites none: a probe that silently patches nothing reports
// the near-miss half as a pass. PTX separates the directive from its operands
// with a tab, so the fields are split rather than prefix-matched.
func shiftLocLines(t *testing.T, ptx []byte, shift int) []byte {
	t.Helper()
	lines := strings.Split(string(ptx), "\n")
	patched := 0
	for i, ln := range lines {
		f := strings.Fields(strings.TrimSpace(ln))
		if len(f) < 3 || f[0] != ".loc" {
			continue
		}
		n, err := strconv.Atoi(f[2])
		if err != nil {
			continue
		}
		f[2] = strconv.Itoa(n + shift)
		lines[i] = "\t" + strings.Join(f, " ")
		patched++
	}
	if patched == 0 {
		t.Fatal("shiftLocLines patched nothing; the fixture PTX carries no .loc directives")
	}
	return []byte(strings.Join(lines, "\n"))
}

// TestFusionFixtureDiscriminates is the precondition for the three tests
// below: the unshifted fixture must load cleanly and the shifted one must not.
// Without this, a caveat that never appears and a caveat that always appears
// are indistinguishable from correct behaviour.
func TestFusionFixtureDiscriminates(t *testing.T) {
	if _, fusion, err := capturefiles.LoadCode(fusionFixture(t, 0)); err != nil || fusion != nil {
		t.Fatalf("clean fixture: fusion=%v err=%v", fusion, err)
	}
	code, fusion, err := capturefiles.LoadCode(fusionFixture(t, 2))
	if err != nil {
		t.Fatalf("shifted fixture: %v", err)
	}
	if fusion == nil {
		t.Fatal("shifted fixture reported no source-fusion failure")
	}
	if code == nil {
		t.Fatal("LoadCode withheld the kernel tier on a source-fusion failure")
	}
}

// TestPressureStatesSourceFusion covers cmd/cudaprof/pressure.go, which
// discarded the load error entirely.
func TestPressureStatesSourceFusion(t *testing.T) {
	const marker = "source attribution unavailable"
	var fused, clean bytes.Buffer
	if err := printCapturePressure(&fused, fusionFixture(t, 2), ""); err != nil {
		t.Fatalf("fused capture: %v", err)
	}
	if !strings.Contains(fused.String(), marker) {
		t.Fatalf("pressure table rendered a fused capture with no caveat:\n%s", fused.String())
	}
	if err := printCapturePressure(&clean, fusionFixture(t, 0), ""); err != nil {
		t.Fatalf("clean capture: %v", err)
	}
	if strings.Contains(clean.String(), marker) {
		t.Fatalf("caveat appears on a capture whose tiers fuse:\n%s", clean.String())
	}
}

// TestAdviseStatesSourceFusion covers cmd/cudaprof/advise.go. The finding
// bodies cite SourceFile and SourceLine, which are exactly the fields the
// failure zeroes, so the report has to say so above them.
func TestAdviseStatesSourceFusion(t *testing.T) {
	fused := &AdviceReport{}
	auditUncoalescedMemory(fusionFixture(t, 2), fused, nil)
	if fused.SourceFusion == "" {
		t.Fatal("advise report carries no source-fusion failure for a fused capture")
	}
	if md := fused.MarkdownWithLimit(0); !strings.Contains(md, "Source attribution unavailable") {
		t.Fatalf("markdown report withheld the caveat:\n%s", md)
	}
	clean := &AdviceReport{}
	auditUncoalescedMemory(fusionFixture(t, 0), clean, nil)
	if clean.SourceFusion != "" {
		t.Fatalf("clean capture reported a source-fusion failure: %s", clean.SourceFusion)
	}
	if md := clean.MarkdownWithLimit(0); strings.Contains(md, "Source attribution unavailable") {
		t.Fatalf("caveat appears on a capture whose tiers fuse:\n%s", md)
	}
}

// TestViewStatesSourceFusion covers cmd/cudaprof/view.go. The pipeline view
// renders the CUDA C++, Tile IR and PTX panes unannotated on a fused capture,
// which is indistinguishable from a compilation that carried no line table.
func TestViewStatesSourceFusion(t *testing.T) {
	diagnostics := func(shift int) string {
		t.Helper()
		data, err := buildSinglePipelineViewData(fusionFixture(t, shift), "capture", "")
		if err != nil {
			t.Fatalf("buildSinglePipelineViewData(shift=%d): %v", shift, err)
		}
		return strings.Join(data.Diagnostics, "\n")
	}
	if got := diagnostics(2); !strings.Contains(got, "source attribution unavailable") {
		t.Fatalf("view rendered a fused capture with no diagnostic: %q", got)
	}
	if got := diagnostics(0); strings.Contains(got, "source attribution unavailable") {
		t.Fatalf("diagnostic appears on a capture whose tiers fuse: %q", got)
	}
}
