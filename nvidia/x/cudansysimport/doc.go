// Package cudansysimport imports a bounded subset of documented Nsight Systems
// SQLite exports through a small read-only database boundary.
//
// It supports release 2024.2/schema 3.11.0 and release 2025.3/schema 3.20.2.
// Generic SQLite access belongs to the Reader supplied by the caller; no Nsight
// capture command, binary report parser or database dependency is included.
// A Reader must enforce read-only access, cancellation and its own process and
// output limits before returning rows. Import also bounds returned rows/bytes.
//
// Imported timing has basis external:nsys. Raw rows, nanosecond timestamps,
// explicit export clock settings and correlation domains remain available.
// Missing providers are unavailable, not empty measurements. Import does not
// certify capture completeness, align different reports or create local CUPTI
// evidence. NCCL annotations remain NVTX rows: undocumented payload semantics
// are not inferred from a range name.
package cudansysimport
