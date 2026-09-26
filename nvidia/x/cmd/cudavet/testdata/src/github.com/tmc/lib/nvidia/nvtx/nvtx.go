// Package nvtx is a declaration-only stand-in for a binding whose GetVersion
// returns only a number, 0 when the library is not loaded.
package nvtx

func GetVersion() int { return 0 }
