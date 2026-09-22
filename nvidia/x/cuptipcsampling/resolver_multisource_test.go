package cuptipcsampling_test

import (
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

// The main.* fixture in ../cudacapture/testdata is the shape that exposed the
// defect: the unit's primary file is an include list that defines nothing, and
// the kernel lives in another file of the same translation unit, at a line
// number the two-file main.cu does not have. That is the real case in
// miniature -- ds4_math.cu is a thirteen-line include list and the attribution
// landed on line 516 of it.
//
// A single-file capture makes Code.SourceFile, Kernel.SourceFile and
// Instruction.SourceFile the same string, so every assertion about "the file"
// passes whichever one the resolver actually read. Only a unit whose files
// disagree can tell them apart, and only a kernel past main.cu's end can tell
// a wrong file from a plausible one.
const multiKernel = "_Z10accumulatePfPKfi"

func loadMultiSourceCode(t *testing.T) (*cudacapture.Code, *cudacapture.Capture) {
	t.Helper()
	dir := filepath.Join("..", "cudacapture", "testdata")
	read := func(name string) []byte {
		t.Helper()
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		return b
	}
	cubinBytes, err := hex.DecodeString(strings.TrimSpace(string(read("main.cubin.hex"))))
	if err != nil {
		t.Fatalf("decode cubin hex: %v", err)
	}
	cap := &cudacapture.Capture{
		Code: cudacapture.CodeFiles{
			Source:  "main.cu",
			Sources: []string{"main.cu", "impl.cu"},
			PTX:     "main.ptx",
			CUBIN:   "main.cubin",
			SASS:    "main.sass",
		},
		Files: map[string][]byte{
			"main.cu":    read("main.cu"),
			"impl.cu":    read("impl.cu"),
			"main.ptx":   read("main.ptx"),
			"main.cubin": cubinBytes,
			"main.sass":  read("main.sass"),
		},
	}
	code, err := cudacapture.LoadCode(cap)
	if err != nil {
		t.Fatalf("LoadCode: %v", err)
	}
	return code, cap
}

// TestResolvePCReportsTheDefiningFile is the gate the single-file tests could
// not be. A resolved location must name the file its line number belongs to.
// Reporting the unit's file with the instruction's line produces a location no
// tier measured, and it is worse than an unresolved PC: an unresolved PC
// announces itself, while "main.cu:3" resolves against a source tree and reads
// as an attribution.
func TestResolvePCReportsTheDefiningFile(t *testing.T) {
	code, capture := loadMultiSourceCode(t)
	r := cuptipcsampling.NewCaptureResolverFromCapture(code, capture)

	// The fixture is only a gate if the three tiers really do disagree; if the
	// unit's file were also impl.cu this test would pass on the broken code.
	if code.SourceFile != "main.cu" {
		t.Fatalf("fixture no longer disagrees: Code.SourceFile = %q", code.SourceFile)
	}
	k := code.Kernel(multiKernel)
	if k == nil {
		t.Fatalf("%s not decoded", multiKernel)
	}
	if k.SourceFile != "impl.cu" {
		t.Fatalf("fixture no longer disagrees: Kernel.SourceFile = %q", k.SourceFile)
	}

	for _, pc := range []uint64{0x0, 0x10} {
		loc := r.ResolvePC(cuptipcsampling.ModuleID{}, multiKernel, pc)
		if !loc.Resolved {
			t.Fatalf("pc %#x: not resolved", pc)
		}
		if loc.SourceFile != "impl.cu" {
			t.Errorf("pc %#x: SourceFile = %q, want impl.cu (the file the line belongs to)", pc, loc.SourceFile)
		}
		if loc.SourceLine == 0 {
			t.Errorf("pc %#x: SourceLine = 0", pc)
		}
	}
}

// TestResolvePCFileAndLineComeFromOneTier is the property behind the fix, and
// it is the one that stays true as the tiers change. Whatever line the resolver
// reports, that line must exist in the file it reports alongside it. Checking
// the pair against the capture's own bytes catches a future refinement of one
// half that forgets the other.
func TestResolvePCFileAndLineComeFromOneTier(t *testing.T) {
	code, capture := loadMultiSourceCode(t)
	r := cuptipcsampling.NewCaptureResolverFromCapture(code, capture)

	for _, pc := range []uint64{0x0, 0x10} {
		loc := r.ResolvePC(cuptipcsampling.ModuleID{}, multiKernel, pc)
		body, ok := capture.Files[loc.SourceFile]
		if !ok {
			t.Fatalf("pc %#x: resolved to %q, which the capture does not bundle", pc, loc.SourceFile)
		}
		lines := strings.Split(string(body), "\n")
		if int(loc.SourceLine) > len(lines) {
			t.Errorf("pc %#x: resolved to %s:%d, but that file has only %d lines",
				pc, loc.SourceFile, loc.SourceLine, len(lines))
		}
	}
}

// TestResolvePCSingleSourceUnchanged pins the compatibility half: the fix must
// not move a single-file capture's answer, where the unit file and the defining
// file are the same string.
func TestResolvePCSingleSourceUnchanged(t *testing.T) {
	code, capture := loadTestCode(t)
	r := cuptipcsampling.NewCaptureResolverFromCapture(code, capture)
	loc := r.ResolvePC(cuptipcsampling.ModuleID{}, "_Z5scalePfPKff", 0)
	if loc.SourceFile != "kernels.cu" {
		t.Errorf("SourceFile = %q, want kernels.cu", loc.SourceFile)
	}
	if loc.SourceLine == 0 {
		t.Error("SourceLine = 0")
	}
}
