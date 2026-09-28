// Package cudashared plans checked per-block shared-memory storage.
//
// A Plan assigns aligned offsets to regions whose stage lifetimes do not
// overlap.  Limits are supplied by a target record; the package has no
// architectural defaults.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
//
// HazardMatrix compares MemoryAccess pairs for byte conflicts, word overlap,
// and missing synchronization across execution scopes.
package cudashared
