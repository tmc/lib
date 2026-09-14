// Package cudasass parses and inspects NVIDIA Streaming Multiprocessor (SASS) machine assembly.
//
// It provides structured access to disassembled GPU machine instructions, PC offsets,
// instruction predicates, opcodes, memory operands, and control codes.
//
// [Diff] and [DiffFunctions] perform normalized differential inspection between
// kernel versions, aligning instructions across loop unrolls and memory offset drift,
// and computing architectural summaries including register spills and opcode substitutions.
//
// # Toolkit versions
//
// [Disassembler] runs nvdisasm, whose options differ between toolkits. Of the
// options this package emits, only --emit-json moved: it is absent in CUDA 12.6
// and present from 12.8 on, measured against the eight toolkits installed on
// the GB10 host, so [JSON] mode is unavailable under 12.6.
// [Disassembler.Disassemble] checks
// the command line it is about to run against what the resolved nvdisasm
// accepts and reports an error naming the tool and its version, rather than
// letting the tool answer with "Unknown option". The check is enforced only
// when the tool's capabilities can be determined: an executable that reports no
// version and whose --help cannot be read is run, and diagnoses itself.
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
package cudasass
