package main

import "fmt"

// dynamicSharedLimit returns the most dynamic shared memory one block of a
// kernel may request, given the device's opt-in ceiling, its default per-block
// ceiling, and the kernel's static allocation. A device that reports no opt-in
// ceiling keeps the default one. The result is never negative.
func dynamicSharedLimit(optin, perBlock, static int) int {
	limit := max(perBlock, optin)
	if static > limit {
		return 0
	}
	return limit - static
}

// sharedMemFits reports whether a launch requesting bytes of dynamic shared
// memory is within limit. A request the device cannot satisfy fails at launch
// with an error that does not say why, and a decode kernel whose request grows
// with the sequence position would fail mid-generation; checking here turns
// that into an error that names the number.
func sharedMemFits(bytes, limit int) error {
	if bytes < 0 {
		return fmt.Errorf("dynamic shared memory request %d bytes is negative", bytes)
	}
	if bytes > limit {
		return fmt.Errorf("dynamic shared memory request %d bytes exceeds the %d bytes one block may use on this device", bytes, limit)
	}
	return nil
}
