//go:build !linux

package cudaworkload

func currentTID() uint64 {
	return 0
}
