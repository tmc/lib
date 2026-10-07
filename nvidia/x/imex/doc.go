// Package imex reads finite IMEX domain JSON snapshots supplied by a caller.
//
// Read requires an explicit pinned profile. It retains directional connections,
// reported state, unparsed timestamps, unavailable versions/hostnames, unknown
// fields and raw input. It does not query a daemon, determine health, certify
// domain membership or report channel counts. NVIDIA/DCGM specification
// provenance is recorded in testdata/README.md; fixtures are not measurements.
package imex
