//go:build !linux

package nvrtcx

// No loaded-file attestation is available here. Compilation may proceed, but
// callers must not use the resulting output as a qualified compiler-cache hit.
func checkedMappedCompilerPath(name string) (string, bool, error) { return "", false, nil }
