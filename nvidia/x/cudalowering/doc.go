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
package cudalowering
