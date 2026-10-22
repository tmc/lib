//go:build linux || darwin

package dsv4bench

import (
	"fmt"
	"runtime"
	"syscall"
)

func peakRSSBytes() (int64, error) {
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		return 0, fmt.Errorf("peak RSS: %w", err)
	}
	return normalizePeakRSS(usage.Maxrss, runtime.GOOS == "linux")
}
