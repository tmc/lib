package cudacapture_test

import (
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

// Two lines of ten bytes each, so line 1 starts at 0 and line 2 at 10.
const testPTX = "// aaaaaa\n// bbbbbb\n"

var testKernel = &cudacapture.Kernel{
	Name:       "rmsnorm",
	Offset:     0xd900,
	PTXLine:    2,
	SourceLine: 40,
	Instructions: []cudacapture.Instruction{
		{Offset: 0x00, SASSLine: 10, PTXLine: 2, SourceLine: 41, Text: "LDG.E R2, [R0];"},
		{Offset: 0x10, SASSLine: 11, PTXLine: 2, SourceLine: 42, Text: "FFMA R3, R2, R1, R3;"},
	},
}

func TestPTXAddresses(t *testing.T) {
	a := cudacapture.NewPTXAddresses(testPTX)
	tests := []struct {
		name string
		line int
		want uint64
		ok   bool
	}{
		{name: "first line is the image base", line: 1, want: cudaprof.PTXMappingStart, ok: true},
		{name: "second line", line: 2, want: cudaprof.PTXMappingStart + 10, ok: true},
		{name: "zero line is unknown", line: 0},
		{name: "negative line is unknown", line: -1},
		{name: "past the text is unknown", line: len(a.Offsets())},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := a.Address(tt.line)
			if ok != tt.ok {
				t.Fatalf("Address(%d) ok = %v, want %v", tt.line, ok, tt.ok)
			}
			if ok && got != tt.want {
				t.Errorf("Address(%d) = %#x, want %#x", tt.line, got, tt.want)
			}
		})
	}
}

// A capture that retained no PTX text knows no PTX address. Reporting one would
// be indistinguishable from a resolved line, and zero is not a usable sentinel
// because a real address is a base plus an offset the caller cannot check.
func TestPTXAddressesNilResolvesNothing(t *testing.T) {
	var a *cudacapture.PTXAddresses
	if got, ok := a.Address(1); ok {
		t.Errorf("Address(1) on a nil map = %#x, ok; want unknown", got)
	}
	if got := a.Offsets(); got != nil {
		t.Errorf("Offsets() on a nil map = %v, want nil", got)
	}
}

func TestFrameBuilderAddresses(t *testing.T) {
	b := cudacapture.NewFrameBuilder(cudacapture.Artifacts{
		Source: "kernels.cu",
		PTX:    "kernels.ptx",
		SASS:   "kernels.sass",
	}, cudacapture.NewPTXAddresses(testPTX))

	f, ok := b.SASSKernelFrame(testKernel)
	if !ok {
		t.Fatal("SASSKernelFrame: no frame for a bundled disassembly")
	}
	// The kernel term has to be in the address: cudacapture enforces that the
	// first instruction's offset is zero, so dropping it is invisible on the
	// first kernel in the CUBIN and wrong on every other.
	if want := cudaprof.DefaultMappingStart + 0xd900; f.Address != want {
		t.Errorf("SASSKernelFrame address = %#x, want %#x", f.Address, want)
	}
	if f.FileName != "kernels.sass" || f.FunctionName != "rmsnorm.SASS" || f.Line != 10 {
		t.Errorf("SASSKernelFrame = %+v", f)
	}

	inst := testKernel.Instructions[1]
	f, ok = b.SASSInstructionFrame(testKernel, inst)
	if !ok {
		t.Fatal("SASSInstructionFrame: no frame for a bundled disassembly")
	}
	if want := cudaprof.DefaultMappingStart + 0xd900 + 0x10; f.Address != want {
		t.Errorf("SASSInstructionFrame address = %#x, want %#x", f.Address, want)
	}
	if f.Line != 11 {
		t.Errorf("SASSInstructionFrame line = %d, want 11", f.Line)
	}

	f, ok = b.PTXFrame(testKernel, 2)
	if !ok {
		t.Fatal("PTXFrame: no frame for a bundled PTX text")
	}
	if want := cudaprof.PTXMappingStart + 10; f.Address != want {
		t.Errorf("PTXFrame address = %#x, want %#x", f.Address, want)
	}

	f, ok = b.SourceFrame(testKernel, 40)
	if !ok {
		t.Fatal("SourceFrame: no frame for a bundled source")
	}
	if f.Address != 0 {
		t.Errorf("SourceFrame address = %#x, want none: source lines are not in the CUBIN mapping", f.Address)
	}
	if f.FunctionName != "rmsnorm" || f.FileName != "kernels.cu" || f.Line != 40 {
		t.Errorf("SourceFrame = %+v", f)
	}
}

// A line the retained text does not cover has no address. The frame still names
// the line, so a reader sees the statement; what it must not carry is an address
// pointing at some other statement.
func TestFrameBuilderPTXFrameWithoutAddress(t *testing.T) {
	b := cudacapture.NewFrameBuilder(cudacapture.Artifacts{PTX: "kernels.ptx"}, cudacapture.NewPTXAddresses(testPTX))
	f, ok := b.PTXFrame(testKernel, 9999)
	if !ok {
		t.Fatal("PTXFrame: no frame for a bundled PTX text")
	}
	if f.Address != 0 {
		t.Errorf("PTXFrame(9999) address = %#x, want none", f.Address)
	}
	if f.Line != 9999 {
		t.Errorf("PTXFrame(9999) line = %d, want the line it was asked for", f.Line)
	}

	nob := cudacapture.NewFrameBuilder(cudacapture.Artifacts{PTX: "kernels.ptx"}, nil)
	if f, _ := nob.PTXFrame(testKernel, 2); f.Address != 0 {
		t.Errorf("PTXFrame with no address map = %#x, want none", f.Address)
	}
}

// A capture must not name a file it does not contain: a frame naming one sends
// every source view for that level to nothing, silently and for every reader.
func TestFrameBuilderOmitsUnbundledArtifacts(t *testing.T) {
	b := cudacapture.NewFrameBuilder(cudacapture.Artifacts{}, cudacapture.NewPTXAddresses(testPTX))
	if f, ok := b.SASSKernelFrame(testKernel); ok {
		t.Errorf("SASSKernelFrame = %+v with no SASS artifact, want none", f)
	}
	if f, ok := b.SASSInstructionFrame(testKernel, testKernel.Instructions[0]); ok {
		t.Errorf("SASSInstructionFrame = %+v with no SASS artifact, want none", f)
	}
	if f, ok := b.PTXFrame(testKernel, 2); ok {
		t.Errorf("PTXFrame = %+v with no PTX artifact, want none", f)
	}
	if f, ok := b.SourceFrame(testKernel, 40); ok {
		t.Errorf("SourceFrame = %+v with no source artifact, want none", f)
	}
}

// Two kernels in one CUBIN must not share an entry address, which is what a
// kernel-relative address gives them.
func TestFrameBuilderKernelsDoNotAlias(t *testing.T) {
	b := cudacapture.NewFrameBuilder(cudacapture.Artifacts{SASS: "kernels.sass"}, nil)
	first, _ := b.SASSKernelFrame(&cudacapture.Kernel{Name: "a", Offset: 0})
	second, _ := b.SASSKernelFrame(&cudacapture.Kernel{Name: "b", Offset: 0x400})
	if first.Address == second.Address {
		t.Errorf("both kernels entered at %#x", first.Address)
	}
}

// ExampleFrameBuilder shows the composition a consumer outside this package
// writes: the leaf is the machine instruction, and each outer frame is the
// representation it was lowered from.
func ExampleFrameBuilder() {
	k := &cudacapture.Kernel{Name: "rmsnorm", Offset: 0x100, PTXLine: 2, SourceLine: 40}
	b := cudacapture.NewFrameBuilder(cudacapture.Artifacts{
		Source: "kernels.cu",
		PTX:    "kernels.ptx",
		SASS:   "kernels.sass",
	}, cudacapture.NewPTXAddresses("// aaaaaa\n// bbbbbb\n"))

	var frames []cudaprof.Frame
	if f, ok := b.SASSKernelFrame(k); ok {
		frames = append(frames, f)
	}
	if f, ok := b.PTXFrame(k, k.PTXLine); ok {
		frames = append(frames, f)
	}
	if f, ok := b.SourceFrame(k, k.SourceLine); ok {
		frames = append(frames, f)
	}
	for _, f := range frames {
		fmt.Printf("%s %s:%d %#x\n", f.FunctionName, f.FileName, f.Line, f.Address)
	}
	// Output:
	// rmsnorm.SASS kernels.sass:0 0x7f000100
	// rmsnorm.PTX kernels.ptx:2 0x7e00000a
	// rmsnorm kernels.cu:40 0x0
}
