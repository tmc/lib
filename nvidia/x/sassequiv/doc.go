// Package sassequiv compares two compilations of a kernel, from their
// SASS, to find a change in what they compute.
//
// It is for changes that should not change results: a __launch_bounds__
// or -maxrregcount change, a rewrite believed to be exact, a toolkit
// upgrade. Instruction counts and order cannot tell a reschedule or a
// different register allocation from a reassociation; comparing values
// can.
//
// [Compare] builds a value graph for each function: every register,
// predicate and stack slot written is an expression over the function's
// inputs (constant-bank words, special registers, loads), memory is a
// chain of stores, barriers and atomics in program order, and a value
// live around a loop is a phi at the loop header. Register names, spill
// slots, and the instructions ptxas uses to materialize a constant (MOV,
// IMAD.MOV, HFMA2.MMA, UMOV) do not appear in the graph. Integer sums and
// products by constants are linear forms, so IADD3, IMAD, LEA and shifts
// computing one sum agree; a 64-bit address formed with LEA and LEA.HI.X
// is the IMAD.WIDE it computes. Operands of a predicated instruction are
// read assuming its predicate holds. Each call runs its own copy of the
// subroutine, returning to the instruction after that call, and blocks
// joined by an edge no other block shares are one block, so register
// moves ptxas places after a return do not change the graph.
//
// Two functions are [Equal] when their control flow graphs have the same
// shape and, block by block, their exits see the same memory chain and
// their branches the same conditions, under the greatest bisimulation of
// the two graphs, which identifies loops computing the same recurrence.
//
// Otherwise Compare finds the first pair of operations that differ and
// reports [Different] only for differences that cannot be another form of
// the same computation: a different floating-point operation, operand or
// constant; a constant added into an address or stored value with
// nothing else on its path differing; or two values each side computes,
// used in exchanged places, as when two loads swap addresses. That is a
// difference in the computation, not proof that some input produces
// different bits: (a+b)+c and a+(b+c) differ here and agree on many
// inputs.
//
// The answer is [Unknown] when the model cannot decide: integer
// expressions that differ in form (ptxas computes one index with masks,
// shifts, induction variables or a different split between a register
// and an address offset), memory operations in a different order (they
// commute when their addresses differ, which the model does not decide),
// code moved across a block boundary, control flow graphs of different
// shapes (one loop unrolled further, say), indirect branches, or an
// instruction the model does not describe. Unknown never means equal.
//
// Register limits are a hard case. Of egraph's generated decode, MoE,
// prefill and fragment kernels, the 16 functions whose SASS changes
// between __launch_bounds__(n) and __launch_bounds__(n, 1) are all
// Equal, though ptxas allocates, spills and schedules them differently;
// so are the 41 whose SASS does not change. Of 62 functions
// in hopper's golden kernels whose SASS changes between -maxrregcount
// 255 and 128, 96 or 64, 2 are Equal, none Different and 60 Unknown:
// ptxas rewrites their index arithmetic and store order along with their
// allocation.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package sassequiv
