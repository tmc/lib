package cudalowering

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

func TestSharedConversionRequiresExecutableMatch(t *testing.T) {
	for _, source := range []string{"", ".version 8.0\n.target sm_121\n.address_size 64\n"} {
		m, err := cudaptx.Parse(strings.NewReader(source))
		if err != nil {
			t.Fatal(err)
		}
		d := &cudasass.Disassembly{Target: "sm_121"}
		for _, opts := range []*Options{{CheckSharedConversions: true}, {InferredSharedSlots: []string{"slot"}}} {
			if _, err := Compare(m, d, opts); err == nil {
				t.Fatal("empty executable comparison accepted")
			}
		}
		if _, err := Compare(m, d, nil); err != nil {
			t.Fatalf("default comparison changed: %v", err)
		}
	}
}

func TestSharedConversionRequiresCompleteEntryText(t *testing.T) {
	m, err := cudaptx.Parse(strings.NewReader(".version 8.0\n.target sm_121\n.address_size 64\n.entry kernel() { ret; }\n"))
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct{ name, text string }{
		{"missing entry", "/*0010*/ MOV R0, 0x400;\n/*0020*/ EXIT;"},
		{"missing instruction", "/*0000*/ MOV R0, 0x400;\n/*0020*/ EXIT;"},
		{"discarded data", "/*0000*/ MOV R0, 0x400;\n/*0010*/ .word 0;\n/*0020*/ EXIT;"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d, err := cudasass.Parse(strings.NewReader("code for sm_121\nFunction : kernel\n" + tt.text + "\n"))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := VerifyNoSharedConversions(m, d, "%slot"); err == nil || !strings.Contains(err.Error(), "entry instruction") {
				t.Fatal("incomplete instruction text established absence")
			}
		})
	}
	d, err := cudasass.Parse(strings.NewReader("code for sm_121\nFunction : kernel\n/*0000*/ MOV R0, 0x400;\n/*0010*/ EXIT;\n"))
	if err != nil {
		t.Fatal(err)
	}
	if r, err := VerifyNoSharedConversions(m, d, "%slot"); err != nil || r.Err() != nil {
		t.Fatalf("complete supported entry refused: report=%v err=%v", r, err)
	}
}
