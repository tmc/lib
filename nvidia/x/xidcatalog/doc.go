// Package xidcatalog looks up NVIDIA XID descriptions and resolution buckets
// from the pinned NVIDIA XID Catalog. It does not diagnose faults or perform
// recovery actions.
//
// Applicability is limited to the four products named by the source table:
// A100, H100, B100 and GB200. It must not be extrapolated to every GPU in the
// same architecture family. Unknown IDs or products return no match.
package xidcatalog
