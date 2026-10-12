// Package cudasass parses and inspects NVIDIA Streaming Multiprocessor (SASS) machine assembly.
//
// It provides structured access to disassembled GPU machine instructions, PC offsets,
// instruction predicates, opcodes, memory operands, and control codes.
//
// [Diff] and [DiffFunctions] perform normalized differential inspection between
// kernel versions, aligning instructions across loop unrolls and memory offset drift,
// and computing architectural summaries including register spills and opcode substitutions.
//
// # Control flow
//
// [BasicBlocks] partitions a function at branches and branch targets, and
// [NewCFG] adds the edges between the blocks, as NVBit's CFG_t does. A
// function containing an indirect branch (BRX, JMX) is [CFG.IsDegenerate]: the
// targets are in a register at run time and the graph lacks their edges.
// [CFG.DOT] renders the graph with instruction offsets; [DisassembleCFG] is
// nvdisasm's rendering of the same function, and the independent check.
//
// [EdgeIndex] and [EdgeCoverage] are the AFL edge hash over block IDs as
// cuFuzz computes it on the device, for reading or reproducing its coverage
// maps on the host.
//
// # Toolkit versions
//
// [Disassembler] runs nvdisasm, whose options differ between toolkits. Of the
// options this package emits, only --emit-json moved: it is absent in CUDA 12.2
// and 12.6 and present from 12.8 on, measured against the eight toolkits
// installed on the GB10 host and the CUDA 12.2 floor, so [JSON] mode is
// unavailable before 12.8.
// [Disassembler.Disassemble] checks
// the command line it is about to run against what the resolved nvdisasm
// accepts and reports an error naming the tool and its version, rather than
// letting the tool answer with "Unknown option". The check is enforced only
// when the tool's capabilities can be determined: an executable that reports no
// version and whose --help cannot be read is run, and diagnoses itself.
//
// # Kernel header spellings
//
// The parser starts a new kernel at four header spellings. Three come from
// vendor tools: ".section .text.NAME" and a bare ".text.NAME:" label from
// nvdisasm, and "Function : NAME" from cuobjdump --dump-sass. Those name a
// kernel by its mangled symbol (_Z5scalePff); use [github.com/tmc/lib/nvidia/x/cudalowering.MatchKey]
// to reduce one to the source spelling.
//
// The fourth, "--- Kernel: NAME ---", is project-local. No vendor tool emits
// it; it exists for hand-written listings and carries a friendly name. Code
// that handles only this spelling has not been exercised against real tool
// output.
//
// [Function.ELFSection] and [Section.ELFSection] record the ELF section a
// kernel's instructions were read from, ".text.NAME", when a header names it,
// and are empty for "Function : NAME" and "--- Kernel: NAME ---". ptxas also
// emits a ".nv.capmerc.text.NAME" section beside each kernel's text; it is not
// the code that was measured to execute (see [CubinImage.Mirrors]). Parse
// refuses a listing with an instruction under that section, or under any
// other code section not named ".text.NAME", whether the section was opened
// by a ".section" directive, a banner or its label, with an error wrapping
// [ErrNotTextSection], rather than merge the instruction into the kernel. Only
// the next ".section" directive or banner ends such a section; a label or a
// kernel header under it does not. Such a section holding only data is
// accepted, as are data sections such as ".nv.constant0.NAME".
//
// # Instruction layout and alignment
//
// SASS instruction addresses must be aligned to hardware instruction boundaries.
// On modern architectures (sm_70 and later, covering Volta, Turing, Ampere, Ada,
// Hopper, and Blackwell), instructions are 16 bytes, requiring addr%16 == 0.
// On older architectures (sm_30 through sm_62), instructions are 8 bytes, requiring
// addr%8 == 0. When no .target directive is present in the listing, the parser
// defaults to sm_70+ (16-byte alignment). Misaligned instruction addresses return
// an error.
//
// Instruction addresses within each section must be strictly increasing.
// Decreasing or duplicate addresses within a section return an error naming the
// section and the offending addresses.
//
// Every instruction in a disassembly listing must belong to a kernel section.
// An instruction appearing before any kernel header, or after the trailing SYMBOLS
// banner, is outside any section and is rejected with an error.
//
// # Control codes and dependency barriers
//
// [EncodeControl] encodes control fields into instruction encodings. Control codes
// are currently calibrated for sm_121a only; attempting to encode control codes for
// an uncalibrated architecture returns an error that wraps [github.com/tmc/lib/nvidia/x/cudaerrors.ErrUnsupported],
// allowing callers to test it with [errors.Is].
//
// # Cubin formats
//
// A cubin carries its producing toolkit's version in the ELF header: nvcc
// writes e_version = 126 through CUDA 12 and e_version = 1 from 13.0 on, so
// this package validates the header itself rather than through [debug/elf],
// which accepts only EV_CURRENT. Reading a cubin from an older toolkit
// generation than nvdisasm still costs something the check cannot repair:
// nvdisasm 13.x disassembles a 12.x cubin in text and JSON modes but warns
// "Disassembling Std Elf to Old format ... not supported yet" and disables
// dataflow analysis, which makes [ControlFlow], [BasicBlockFlow] and
// [RegisterLife] fail in the tool. Matching the disassembler to the toolkit
// that produced the cubin is what makes every mode work.
//
// # Toolchain provenance
//
// e_version separates CUDA 12 from 13 and no more. The exact compiler is
// stamped in the .note.nv.tkinfo ELF note ptxas writes into every cubin, which
// [ReadToolchains] reads with [debug/elf] alone, loading no CUDA driver and
// running no external tool:
//
//	stamps, err := cudasass.ReadToolchains(bytes.NewReader(cubin))
//
// Recording that stamp beside a checked-in cubin, disassembly or measurement
// says which compiler produced the code, so artifacts captured months apart
// are comparable and an image assembled from several toolkits is visible as
// more than one stamp. ptxas codegen differs between CUDA minor releases;
// nothing here claims a performance consequence for that.
//
// Only tki_toolName and tki_toolVersion are reported. tki_toolBranch is not:
// in the cubin under testdata it holds the unexpanded placeholder "Build
// system must define TOOLS_VERSION_EXTENDED" while nvdisasm-13.0-sm121.json
// records a real branch string for another image of the same release, so the
// field does not mean the same thing in every cubin.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudasass
