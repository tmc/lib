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

// The shape below is the one that exposed the defect: the unit's primary file
// is an include list that defines nothing, and the kernel lives in another file
// of the same translation unit. A single-file capture makes Code.SourceFile,
// Kernel.SourceFile and Instruction.SourceFile the same string, so every
// assertion about "the file" passes whichever one the resolver actually read.
// Only a unit whose files disagree can tell them apart.
const (
	resolverMainCU = `// Translation unit; defines nothing itself.
#include "impl.cu"
`
	// impl.cu is padded so that its kernel sits at a line number the two-line
	// main.cu does not have. That is the real case in miniature: ds4_math.cu is
	// a thirteen-line include list and the attribution landed on line 516 of
	// it. A short impl.cu would put every line in range for both files, and the
	// file-and-line consistency check below would pass on the broken code.
	resolverImplCU = `// Implementation.
// pad
// pad
// pad
// pad
// pad
void compute(int *a) {
    *a = 1;
}
`
	resolverPTX = `.version 8.0
.target sm_90
.address_size 64

.file 1 "main.cu"
.file 2 "impl.cu"

.visible .entry compute(
	.param .u64 a
)
{
	.loc 2 8 0;
	mov.u64 %rd1, %rd2;
	.loc 2 9 0;
	ret;
}
`
	resolverSASS = `.section .text.compute
//## File "impl.cu", line 8
/* 0x0000 */   MOV R1, c[0x0][0x28] ;
//## File "impl.cu", line 9
/* 0x0010 */   RET ;
`
)

func loadMultiSourceCode(t *testing.T) (*cudacapture.Code, *cudacapture.Capture) {
	t.Helper()
	dir := filepath.Join("..", "cudacapture", "testdata")
	cubinHex, err := os.ReadFile(filepath.Join(dir, "kernels.cubin.hex"))
	if err != nil {
		t.Fatalf("read kernels.cubin.hex: %v", err)
	}
	cubinBytes, err := hex.DecodeString(strings.TrimSpace(string(cubinHex)))
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
			"main.cu":    []byte(resolverMainCU),
			"impl.cu":    []byte(resolverImplCU),
			"main.ptx":   []byte(resolverPTX),
			"main.cubin": cubinBytes,
			"main.sass":  []byte(resolverSASS),
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
	k := code.Kernel("compute")
	if k == nil {
		t.Fatal("compute not decoded")
	}
	if k.SourceFile != "impl.cu" {
		t.Fatalf("fixture no longer disagrees: Kernel.SourceFile = %q", k.SourceFile)
	}

	for _, pc := range []uint64{0x0, 0x10} {
		loc := r.ResolvePC(cuptipcsampling.ModuleID{}, "compute", pc)
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
		loc := r.ResolvePC(cuptipcsampling.ModuleID{}, "compute", pc)
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
	loc := r.ResolvePC(cuptipcsampling.ModuleID{}, "compute", 0)
	if loc.SourceFile != "kernels.cu" {
		t.Errorf("SourceFile = %q, want kernels.cu", loc.SourceFile)
	}
	if loc.SourceLine == 0 {
		t.Error("SourceLine = 0")
	}
}
