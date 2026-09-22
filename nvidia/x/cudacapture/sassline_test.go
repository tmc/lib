package cudacapture_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// TestSASSLineSurvivesCRLF pins that a SASS instruction is located in the
// disassembly text by its address rather than by its text.
//
// cudasass parses with a bufio.Scanner, which drops a trailing "\r"; the
// scan here splits on "\n", which keeps it. Keying the line table on the
// instruction text therefore misses every lookup on a CRLF disassembly and
// the whole capture is refused. The address is the same number on both
// sides, and is already the identity the ordering and bounds checks trust.
func TestSASSLineSurvivesCRLF(t *testing.T) {
	sass, err := os.ReadFile(filepath.Join("testdata", "inline.sass"))
	if err != nil {
		t.Fatalf("read inline.sass: %v", err)
	}
	crlf := strings.ReplaceAll(string(sass), "\n", "\r\n")
	code, err := cudacapture.LoadCode(loadInlineCapture(t, crlf))
	if err != nil {
		t.Fatalf("LoadCode: %v", err)
	}
	k := code.Kernel(inlineKernel)
	if k == nil {
		t.Fatalf("%s not decoded", inlineKernel)
	}
	// The same fixture read with LF, so the comparison is against what the
	// loader itself reports rather than against a transcribed line number.
	lf, err := cudacapture.LoadCode(loadInlineCapture(t, ""))
	if err != nil {
		t.Fatalf("LoadCode on the LF disassembly: %v", err)
	}
	want := lf.Kernel(inlineKernel)
	if len(k.Instructions) != len(want.Instructions) {
		t.Fatalf("got %d instructions, want %d", len(k.Instructions), len(want.Instructions))
	}
	for i, inst := range k.Instructions {
		if inst.SASSLine != want.Instructions[i].SASSLine {
			t.Errorf("instruction %d SASSLine = %d, want %d", i, inst.SASSLine, want.Instructions[i].SASSLine)
		}
	}
	if got := k.Instruction(inlinedOffset).SourceFile; got != "helpers.cuh" {
		t.Errorf("SourceFile = %q, want helpers.cuh", got)
	}
}
