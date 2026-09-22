package cudacapture_test

import (
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// The inline.* fixture is a unit whose kernel force-inlines a device function
// from helpers.cuh, so the disassembler attributes some of its instructions to
// that header and the rest to inline.cu. A capture bundles the .cu files of
// its unit and not the headers, which is the shape the attribution has to
// survive.
const (
	inlineKernel = "_Z5blendPfPKfi"

	// inlinedOffset is the instruction the compiler took from helpers.cuh,
	// and headerInlinedOffset the one that also carries the call site it was
	// inlined at.
	inlinedOffset       = 0xe0
	headerInlinedOffset = 0xf0
	// mainOffset is the store the kernel's own line 10 emits, after the
	// inlined body has ended.
	mainOffset = 0x110
)

// loadInlineCapture returns the committed inline.* fixture with the
// disassembly optionally replaced, so a control can change that one tier.
func loadInlineCapture(t *testing.T, sass string) *cudacapture.Capture {
	t.Helper()
	read := func(name string) []byte {
		b, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		return b
	}
	cubin, err := hex.DecodeString(strings.TrimSpace(string(read("inline.cubin.hex"))))
	if err != nil {
		t.Fatalf("decode cubin hex: %v", err)
	}
	sassBytes := read("inline.sass")
	if sass != "" {
		sassBytes = []byte(sass)
	}
	return &cudacapture.Capture{
		Code: cudacapture.CodeFiles{Source: "inline.cu", PTX: "inline.ptx", CUBIN: "inline.cubin", SASS: "inline.sass"},
		Files: map[string][]byte{
			"inline.cu":    read("inline.cu"),
			"inline.ptx":   read("inline.ptx"),
			"inline.cubin": cubin,
			"inline.sass":  sassBytes,
		},
	}
}

func TestHeaderOriginatedLineIsKept(t *testing.T) {
	code, err := cudacapture.LoadCode(loadInlineCapture(t, ""))
	if err != nil {
		t.Fatalf("LoadCode: %v", err)
	}
	k := code.Kernel(inlineKernel)
	if k == nil {
		t.Fatalf("%s not decoded", inlineKernel)
	}
	inst := k.Instruction(inlinedOffset)
	if inst == nil {
		t.Fatalf("no instruction at offset %#x", inlinedOffset)
	}
	// The instruction was inlined from a header. Zeroing its line because the
	// file is not the main source discards a location the disassembler
	// measured, and renders as "no source information" rather than as the
	// header line it is.
	if inst.SourceFile != "helpers.cuh" {
		t.Errorf("SourceFile = %q, want helpers.cuh", inst.SourceFile)
	}
	if inst.SourceLine != 6 {
		t.Errorf("SourceLine = %d, want 6", inst.SourceLine)
	}
	main := k.Instruction(mainOffset)
	if main == nil {
		t.Fatalf("no instruction at offset %#x", mainOffset)
	}
	if main.SourceFile != "inline.cu" || main.SourceLine != 10 {
		t.Errorf("main instruction = %s:%d, want inline.cu:10", main.SourceFile, main.SourceLine)
	}
}

// TestInlineChainNamesTheCallSite is the other half of the same attribution:
// the header line says where the code was written and the chain says where it
// was placed, and a reader that keeps only one of them cannot get back to the
// kernel.
func TestInlineChainNamesTheCallSite(t *testing.T) {
	code, err := cudacapture.LoadCode(loadInlineCapture(t, ""))
	if err != nil {
		t.Fatalf("LoadCode: %v", err)
	}
	inst := code.Kernel(inlineKernel).Instruction(headerInlinedOffset)
	if inst == nil {
		t.Fatalf("no instruction at offset %#x", headerInlinedOffset)
	}
	if inst.SourceFile != "helpers.cuh" {
		t.Errorf("SourceFile = %q, want helpers.cuh", inst.SourceFile)
	}
	if len(inst.InlinedAt) != 1 {
		t.Fatalf("InlinedAt = %v, want one call site", inst.InlinedAt)
	}
	if got := filepath.Base(inst.InlinedAt[0].File); got != "inline.cu" {
		t.Errorf("call site file = %q, want inline.cu", got)
	}
	if inst.InlinedAt[0].Line != 10 {
		t.Errorf("call site line = %d, want 10", inst.InlinedAt[0].Line)
	}
}

func TestKernelPTXLineIsTheEntryNotAnOpcodeGuess(t *testing.T) {
	code, err := cudacapture.LoadCode(loadFixtureCapture(t, "", ""))
	if err != nil {
		t.Fatalf("LoadCode: %v", err)
	}
	// kernels.ptx declares .visible .entry on those lines. Selecting instead
	// the line of the first instruction whose opcode looks "arithmetic"
	// manufactures a line from opcode text.
	for name, want := range map[string]int{"fill": 49, "_Z5scalePfPKff": 15} {
		k := code.Kernel(name)
		if k == nil {
			t.Fatalf("%s not decoded", name)
		}
		if k.PTXLine != want {
			t.Errorf("%s: Kernel.PTXLine = %d, want %d (the .entry line)", name, k.PTXLine, want)
		}
	}
}
