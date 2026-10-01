// Package shape compares kernel launches using explicitly observed application
// metadata and launch geometry. It preserves raw symbols, including templates,
// and refuses missing or ambiguous identities instead of pairing by occurrence.
//
// Compare requires token, operation, dtype, cache length, grid, and block.
// Phase is required by the marker schema. Optional layer, shared memory,
// lookahead, and materialization participate in matching when supplied;
// absence never matches an explicit value. An identity describes instrumentation
// claims, not an independent correctness check of the application.
//
// CompareComponents uses separate version 3 component contracts, explicit
// request mappings and observed root submissions. It compares repeated launch
// configurations as per-root multisets, including dynamic shared memory.
// Declared banks or construction scopes do not establish root execution.
//
// Duration distributions describe launches within captures. They are not
// independent-run confidence intervals. Completeness and stationarity gates
// belong to the caller and cannot be replaced by shape qualification.
// This package is outside the compatibility promise of github.com/tmc/lib/nvidia.
package shape
