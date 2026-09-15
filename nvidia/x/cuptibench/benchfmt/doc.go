// Package benchfmt converts explicit external measurements to Go benchmark
// text for benchstat. It does not run workloads or establish equivalence to a
// Go benchmark. Names, scopes, operation definitions, timing boundaries, and
// file configuration must match the Go comparison arm.
//
// One Sample emits one benchmark row, regardless of Operations. Operations is
// a normalization denominator, not a count of independent statistical samples.
// Source identities and provenance remain in JSON comments; they do not alter
// the benchmark names or split benchstat tables. Config holds matching Go
// headers such as goos, goarch, pkg, and cpu.
package benchfmt
