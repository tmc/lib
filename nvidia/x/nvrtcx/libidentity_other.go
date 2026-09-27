//go:build !linux

package nvrtcx

// mappedLibrary has no mapping table to read outside Linux.
func mappedLibrary(soname string) string { return "" }
