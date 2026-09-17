package cudacapture_test

import (
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
	crlf := func(s string) string {
		return strings.ReplaceAll(s, "\n", "\r\n")
	}
	code, err := cudacapture.LoadCode(loadFixtureCapture(t, headerPTX, crlf(headerSASS)))
	if err != nil {
		t.Fatalf("LoadCode: %v", err)
	}
	k := code.Kernels[0]
	if len(k.Instructions) != 2 {
		t.Fatalf("got %d instructions, want 2", len(k.Instructions))
	}
	// The //## comment precedes each instruction, so the recorded SASS line
	// is the instruction's own line in the disassembly text.
	for i, want := range []int{3, 5} {
		if got := k.Instructions[i].SASSLine; got != want {
			t.Errorf("instruction %d SASSLine = %d, want %d", i, got, want)
		}
	}
	if got := k.Instructions[0].SourceFile; got != "helpers.cuh" {
		t.Errorf("SourceFile = %q, want helpers.cuh", got)
	}
}
