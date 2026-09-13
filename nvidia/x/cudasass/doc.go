// Package cudasass parses and inspects NVIDIA Streaming Multiprocessor (SASS) machine assembly.
//
// It provides structured access to disassembled GPU machine instructions, PC offsets,
// instruction predicates, opcodes, memory operands, and control codes.
//
// # Toolkit versions
//
// [Disassembler] runs nvdisasm, whose options differ between toolkits. Of the
// options this package emits, only --emit-json moved: it arrived in CUDA 13.0,
// so [JSON] mode is unavailable under 12.6. [Disassembler.Disassemble] checks
// the command line it is about to run against what the resolved nvdisasm
// accepts and reports an error naming the tool and its version, rather than
// letting the tool answer with "Unknown option". The check is enforced only
// when the tool's capabilities can be determined: an executable that reports no
// version and whose --help cannot be read is run, and diagnoses itself.
package cudasass
