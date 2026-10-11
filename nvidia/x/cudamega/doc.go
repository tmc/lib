// Package cudamega schedules finite task graphs and generates CUDA workers.
//
// [NewProgram] binds operations to a validated graph, a target, and the
// caller's measured worker residency. [ProgramSource] emits a persistent
// worker and an ordinary-launch reference for the same tasks. Callers retain
// the worker's buffers until stream completion.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudamega
