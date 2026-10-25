// Package ptxasinfo parses the resource report that ptxas prints with
// -v (--verbose), as in nvcc -Xptxas -v.
//
// For every function compiled for every target architecture, the report
// gives registers per thread, the stack frame, spill stores and loads,
// barriers, static shared memory and constant banks. These are the numbers
// a kernel author checks after each change, and they need only a compiler,
// not a GPU.
//
// Parse is strict about the lines it owns: a resource field it does not
// know is an error rather than a silently missing number. Lines from other
// tools in the same stream (nvcc, cudafe, the host compiler) are skipped.
package ptxasinfo
