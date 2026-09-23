// Package main is a candidate-kernel harness: it takes one CUDA C++ source
// file, compiles it with NVRTC, checks its output against a CPU reference the
// candidate never sees, measures its device time, and prints a classified
// report that a generate-and-refine loop can read.
//
// The report is the point. A loop that learns only "pass" or "fail" cannot
// tell a syntax error from a race, so every failure is classified and every
// number says how it was obtained. Four rules shape the output:
//
//   - The oracle is not the producer. Each task owns a CPU reference written
//     from the task's definition, in this repository, compiled into the
//     harness. A candidate cannot supply, influence, or observe it.
//   - Time is device time. Kernel timing comes from CUDA events spanning the
//     submitted work ([cuda.EventTimer]). Wall clock is reported only where it
//     is the only thing available -- compilation -- and is named WallNanos.
//   - A failure is classified. Compile error, module-load rejection, missing
//     entry point, rejected launch, device fault, timeout and wrong answer are
//     separate outcomes carrying separate evidence.
//   - What was not measured is listed. The report carries an "unmeasured"
//     array naming every quantity the harness deliberately does not report,
//     so an absent number is visible rather than inferred as a zero.
//
// Usage:
//
//	kernelharness -task axpby -kernel candidate.cu -json
//
// With no CUDA device the harness exits 3 and reports outcome "no_device".
// That is distinct from both success (0) and a failed candidate (1): a loop
// must not read an absent GPU as a passing kernel.
package main
