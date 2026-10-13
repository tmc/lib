// Package nsightcompute reads a bounded project envelope around an offline
// NVIDIA Nsight Compute details CSV export. It never invokes Nsight Compute.
//
// The supported profile is the exact details header exported by version
// 2026.3.0.0 with base units and mangled kernel names. SchemaVersion versions
// this project's envelope, not an NVIDIA JSON or binary report schema.
// Unknown profiles and headers are rejected. Raw rows, units, unsupported
// metrics, and exact unsigned integers are retained.
//
// A CSV kernel name or launch shape cannot establish an image or workload
// identity. Available evidence requires an explicit external launch identity
// and a unique compatible capture backed by retained module bytes. It is
// classified as external:ncu, never as a local CUPTI observation. Joins verify
// producer records against retained bytes; they are not driver attestations.
package nsightcompute
