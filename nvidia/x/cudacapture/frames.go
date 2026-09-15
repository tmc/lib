package cudacapture

import "github.com/tmc/lib/nvidia/x/cudaprof"

// Artifacts names the files a capture bundles its retained code under. A name
// left empty means the capture does not carry that artifact, and the frames for
// it are omitted rather than emitted against a file no reader can open.
type Artifacts struct {
	Source string // CUDA source, as named by CodeFiles.Source.
	PTX    string // PTX text, as named by CodeFiles.PTX.
	SASS   string // Disassembly, as named by CodeFiles.SASS.
}

// PTXAddresses resolves a one-based PTX line to the logical address the profile
// gives it. The addresses are derived from the retained PTX text itself, so a
// line that is not in that text has no address rather than a plausible one.
//
// A nil *PTXAddresses resolves nothing, which is what a capture that retained no
// PTX knows.
type PTXAddresses struct {
	offsets []uint64
}

// NewPTXAddresses returns the address map for the retained PTX text.
func NewPTXAddresses(ptx string) *PTXAddresses {
	return &PTXAddresses{offsets: cudaprof.PTXLineOffsets(ptx)}
}

// Offsets returns the byte offset at which each one-based line of the PTX text
// begins, in the form cudaprof.PTXLineOffsets documents. Consumers that address
// PTX themselves take the offsets from here so the capture and they cannot
// derive two different addresses for one statement.
func (a *PTXAddresses) Offsets() []uint64 {
	if a == nil {
		return nil
	}
	return a.offsets
}

// Address returns the logical address of a one-based PTX line. ok reports
// whether the capture establishes one; when it is false there is no address,
// which is distinct from an address that happens to be zero.
func (a *PTXAddresses) Address(line int) (addr uint64, ok bool) {
	if a == nil || line < 1 || line >= len(a.offsets) {
		return 0, false
	}
	return cudaprof.PTXMappingStart + a.offsets[line], true
}

// A FrameBuilder builds pprof frames for the code a capture retains. Each frame
// names an artifact the caller says the capture bundles and carries a line and,
// where the capture establishes one, an address derived from the Kernel and
// Instruction identities rather than from a convention.
//
// Frames are returned one at a time, leaf first when composed: the SASS frame
// is the leaf, then PTX, then the CUDA source. A caller that records further
// levels of its own, such as a tile IR, composes them in between.
//
// The zero value is not usable; call NewFrameBuilder.
type FrameBuilder struct {
	artifacts Artifacts
	ptx       *PTXAddresses
}

// NewFrameBuilder returns a FrameBuilder naming frames after the artifacts a
// capture bundles and addressing PTX through ptx, which may be nil when the
// capture retains no PTX text.
func NewFrameBuilder(artifacts Artifacts, ptx *PTXAddresses) *FrameBuilder {
	return &FrameBuilder{artifacts: artifacts, ptx: ptx}
}

// SASSKernelFrame returns the frame for the entry of k. Its address is the
// kernel's own offset in the CUBIN, so kernels do not share an entry address.
// ok is false when the capture bundles no disassembly.
func (b *FrameBuilder) SASSKernelFrame(k *Kernel) (cudaprof.Frame, bool) {
	if b == nil || k == nil || b.artifacts.SASS == "" {
		return cudaprof.Frame{}, false
	}
	line := 0
	if len(k.Instructions) > 0 {
		line = k.Instructions[0].SASSLine
	}
	return cudaprof.Frame{
		FunctionName: k.Name + ".SASS",
		FileName:     b.artifacts.SASS,
		Line:         int64(line),
		Address:      cudaprof.DefaultMappingStart + k.Offset,
	}, true
}

// SASSInstructionFrame returns the frame for one instruction of k. ok is false
// when the capture bundles no disassembly.
//
// Instruction.Offset is relative to the kernel's own .text section, and the
// first instruction's is zero, so the kernel's offset has to be added to reach
// the CUBIN file offset the profile's addresses are defined as. Without it every
// kernel's instructions alias the first kernel's range, which no line-oriented
// view shows and every address-keyed one gets wrong.
func (b *FrameBuilder) SASSInstructionFrame(k *Kernel, inst Instruction) (cudaprof.Frame, bool) {
	if b == nil || k == nil || b.artifacts.SASS == "" {
		return cudaprof.Frame{}, false
	}
	return cudaprof.Frame{
		FunctionName: k.Name + ".SASS",
		FileName:     b.artifacts.SASS,
		Line:         int64(inst.SASSLine),
		Address:      cudaprof.DefaultMappingStart + k.Offset + inst.Offset,
	}, true
}

// PTXFrame returns the frame for a one-based PTX line of k. ok is false when the
// capture bundles no PTX text.
//
// A line the capture cannot address, because it retained no PTX or because the
// line is outside the text it retained, yields a frame with no address. Pprof
// then reads it as a source-only frame, which is what the capture knows; the
// alternative is an address that resolves to some other statement.
func (b *FrameBuilder) PTXFrame(k *Kernel, line int) (cudaprof.Frame, bool) {
	if b == nil || k == nil || b.artifacts.PTX == "" {
		return cudaprof.Frame{}, false
	}
	f := cudaprof.Frame{
		FunctionName: k.Name + ".PTX",
		FileName:     b.artifacts.PTX,
		Line:         int64(line),
	}
	if addr, ok := b.ptx.Address(line); ok {
		f.Address = addr
	}
	return f, true
}

// SourceFrame returns the CUDA source frame for k at a one-based source line.
// ok is false when the capture bundles no source. The frame carries no address:
// source lines are not addressed by the CUBIN mapping.
func (b *FrameBuilder) SourceFrame(k *Kernel, line int) (cudaprof.Frame, bool) {
	if b == nil || k == nil || b.artifacts.Source == "" {
		return cudaprof.Frame{}, false
	}
	return cudaprof.Frame{
		FunctionName: k.Name,
		FileName:     b.artifacts.Source,
		Line:         int64(line),
	}, true
}
