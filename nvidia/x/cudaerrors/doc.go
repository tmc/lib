// Package cudaerrors holds the sentinels that more than one package must
// agree on.
//
// It exists because several capability gates in unrelated areas -- launch
// validation, fusion boundaries, race conformance, cluster distributed shared
// memory -- have to give the same answer to the same question, and a sentinel
// defined in whichever package happened to need it first would make every
// other one import that package for an error value.
//
// It holds sentinels and nothing else: no helpers, no error types with fields,
// no constructors. A package with behaviour in it attracts dependencies, and
// this one is imported by capability gates that must stay cheap to import.
// Wrap these at the call site with fmt.Errorf so the message carries the
// caller's context; match them with errors.Is.
package cudaerrors
