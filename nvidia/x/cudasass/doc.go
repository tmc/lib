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
package cudasass
