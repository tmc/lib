// Package cudaptx parses NVIDIA PTX virtual assembly.
//
// Parse reports a module's version and target, its kernels and device
// functions, their parameters and launch-bound tuning directives, and for each
// instruction its opcode, guard predicate, operand text, and the source
// location attached by .file and .loc. Module.AssertOpcodes audits a kernel's
// instruction mix, mirroring the same method on cudasass.Disassembly so that a
// lowering can be checked at both levels.
//
// Operands are kept as unparsed text. PTX operand syntax is large and the
// consumers in this repository attribute samples to source lines rather than
// interpret addressing modes.
//
// Instruction keyword, category, directive, data type, and special register
// tables and reference documentation are generated from the PTX ISA
// documentation; see IsInstructionKeyword, InstructionCategory, IsDataType,
// IsSpecialRegister, LookupDoc, Explain, HoverInfo, and PTXISAVersion.
//
// Reference: PTX ISA 9.4. The keyword table is deliberately newer than the
// fixtures in testdata, which nvcc from CUDA 13.0 emits as PTX ISA 9.0: the
// table is a superset used to audit parser output, so generating it from newer
// documentation is safe, and neither version number should be changed to match
// the other.
package cudaptx

//go:generate go run ./internal/genptxtables -docs ../../../../libgen/docs/ptx -out tables.gen.go -docout ptxdoc.json.gz
