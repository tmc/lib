// Package cudagroup describes bounded CUDA execution-group contracts.
//
// Group membership, collective participation, launch dimensions, and memory
// visibility scopes are separate values. The package models host-side
// validation; it does not emit warpgroup uniformity or TMEM instructions.
// The supported CTA subset is at most 1024 threads and the supported cluster
// subset is at most eight CTAs.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudagroup
