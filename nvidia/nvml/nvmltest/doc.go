// Package nvmltest scripts and records the small NVML query interface used by
// telemetry consumers. Replay makes no NVML calls. Recorded opaque handles
// are evidence values, not live handles usable with the driver.
//
// Recordings distinguish measured hardware responses from synthetic fixtures.
// The package does not simulate GPU execution or qualify another GPU model.
// JSON field names are exact and may not be duplicated or aliased. Arguments
// include every parameter, preserving a nil ID list separately from an empty
// list. The 16 MiB limit includes provenance, framing and all calls. Error
// records exceeding 32 nested causes or 4096 cause nodes are refused.
package nvmltest
