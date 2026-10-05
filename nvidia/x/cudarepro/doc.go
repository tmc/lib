// Package cudarepro reduces recorded GPU failures without editing kernel source.
//
// Reduction considers a closed set of dimensions, in order: input values and
// lengths; each tensor shape extent; grid, block, and dynamic shared memory;
// and subsets of the campaign's recorded injection decisions. Unknown record
// fields are errors. Kernel source reduction is outside this package's scope.
//
// A candidate must preserve the recorded failure identity: the campaign verdict
// and its kind-specific signature. A mismatch identifies the logical output
// element and numerical comparison policy. An instrumented access violation
// identifies its violation kind and access site. Missing identity evidence is
// an error; another failure does not substitute for the recorded one.
package cudarepro
