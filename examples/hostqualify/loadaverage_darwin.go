package main

import (
	"encoding/binary"
	"fmt"

	"golang.org/x/sys/unix"
)

// loadAverage reads the 1-minute load average. Darwin reports it through
// vm.loadavg as a struct loadavg: three fixed-point averages scaled by a
// long fscale that follows them.
func loadAverage() (float64, error) {
	raw, err := unix.SysctlRaw("vm.loadavg")
	if err != nil {
		return 0, fmt.Errorf("read vm.loadavg: %w", err)
	}
	const header = 3 * 4
	if len(raw) < header+8 {
		return 0, fmt.Errorf("read vm.loadavg: %d bytes, want at least %d", len(raw), header+8)
	}
	scale := binary.LittleEndian.Uint64(raw[len(raw)-8:])
	if scale == 0 {
		return 0, fmt.Errorf("read vm.loadavg: zero scale")
	}
	return float64(binary.LittleEndian.Uint32(raw[:4])) / float64(scale), nil
}
