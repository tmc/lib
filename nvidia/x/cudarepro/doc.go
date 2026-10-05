// Package cudarepro reduces recorded GPU failures without editing kernel source.
//
// Reduction considers a closed set of dimensions, in order: input values and
// lengths; each tensor shape extent; grid, block, and dynamic shared memory;
// and subsets of the campaign's recorded injection decisions. Unknown record
// fields are errors. Kernel source reduction is outside this package's scope.
// Recorded schedules replay against pristine source through cudaschedule.Replay;
// every measured run must report the digest of that exact replayed artifact.
// Empty decision subsets execute pristine source. The caller supplies the
// existing campaign runner and its explicit workload input contract.
//
// A candidate must preserve the recorded failure identity: the campaign verdict
// and its kind-specific signature. A mismatch identifies the logical output
// element and numerical comparison policy. An instrumented access violation
// identifies its violation kind and access site. Missing identity evidence is
// an error; another failure does not substitute for the recorded one.
//
// Reproduction measurements require an explicit positive replay count R. A
// candidate uses the same R and must reproduce at least as often as the fresh
// baseline. This conservative empirical rule has no statistical confidence
// claim. Each measurement makes at most R calls to the existing runner; it
// never retries invalid execution evidence. One reproduction in R attempts is
// a one-off, which callers must explicitly opt into reducing.
package cudarepro
