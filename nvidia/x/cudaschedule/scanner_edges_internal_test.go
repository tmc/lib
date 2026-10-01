package cudaschedule

import (
	"testing"
)

func TestFindPTXToken_SkipsOpcodeLookalikes(t *testing.T) {
	// These are scanner inputs, not complete PTX modules. In particular, the
	// opcode-shaped label pins the colon guard without claiming that periods
	// are legal in PTX user labels.
	for _, tc := range []struct{ name, prefix string }{
		{"block comment", "/* membar.gl; */ "},
		{"line comment", "// membar.gl;\n"},
		{"quoted pragma", ".pragma \"membar.gl;\";\n"},
		{"left word boundary", ".reg .b32 %xmembar.gl;\n"},
		{"right word boundary", ".reg .b32 %membar.glSuffix;\n"},
		{"opcode label", "membar.gl:\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			source := tc.prefix + "membar.gl;"
			if got := findPTXToken(source, 0, "membar.gl"); got != len(tc.prefix) {
				t.Fatalf("opcode offset = %d, want real statement at %d", got, len(tc.prefix))
			}
		})
	}
}
