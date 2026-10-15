// Package cudalowering compares the PTX for a kernel with the SASS it lowered to.
//
// A PTX assertion says what the compiler was asked for. A SASS assertion says
// what the hardware was given. Running both proves nothing on its own; the
// finding is where the two disagree, because that is where ptxas made a
// decision the PTX does not predict. [Compare] pairs the two levels and
// reports the disagreements:
//
//   - [Erased]: the PTX asked for an opcode and the SASS contains no
//     instruction it lowers to. ptxas substituted something cheaper, or
//     deleted the work.
//   - [Unsourced]: the SASS contains an opcode that ptxas emits only for a PTX
//     opcode the kernel never used. The instruction was introduced, not
//     translated.
//   - [Spill]: an unsourced local memory access. Every value lived in a virtual
//     register at the PTX level, so the spill is the register allocator's.
//   - [Padding]: the fraction of the kernel body that is NOP. Padding displaces
//     the instruction mix a throughput prediction was made from.
//   - [MissingSASS], [MissingPTX]: a kernel that exists at one level only. These
//     are reported rather than skipped, so that a comparison run against the
//     wrong pair of artifacts fails instead of finding nothing.
//   - [SharedConversion]: a shared conversion in PTX or the bounded observed
//     SM121 SASS reaching-value mechanism. [VerifyNoSharedConversions] checks
//     matched kernels in aggregate; its caller labels are not slot attribution.
//
// [VerifyInferredSharedConversions] derives per-slot textual attribution from
// actual nvvmx inference provenance, physical PTX lines and direct compiler
// source locations. It validates original IR identity; texts remain caller-paired
// observations, not proof that a compiler emitted them. The production
// [VerifyCompiledSharedConversions] path consumes one actual compiled input's
// retained origins and verifies its copied PTX digest. Neither verifies native
// artifact association or debug-metadata correctness on a device.
//
// A clean slot requires a supported direct shared witness for every terminal
// use, unique source origins and no attributed conversion. Missing/ambiguous
// locations, unsupported mechanisms, dependent slots or eliminated uses return
// an error and partial report. They are never counted as conversion absence.
// The checker reuses the existing scanner and parsers, supports exact sm_121
// textual observations only, and does not infer register/source/name mappings.
// Inference remains opt-in; native LLVM/PTX/SASS A/B, tool/version identities,
// debug-location preservation and bit-identical output qualification stay open.
//
// The mapping from PTX opcodes to the SASS opcodes they lower to is
// [DefaultLowerings], which callers replace or extend through
// [Options.Lowerings]. It records what ptxas emits, not which emissions are
// acceptable: which divergences matter is the caller's policy, expressed by
// choosing the table and by reading [Report.Divergences].
//
// Spills here are counted from LDL and STL opcodes. Live register counts, which
// [github.com/tmc/lib/nvidia/x/cudasass.JoinPressure] attaches to instructions, are a different
// measurement: high pressure explains a spill but does not count one, so this
// package reports pressure only as supporting detail on a [Spill] divergence.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudalowering
