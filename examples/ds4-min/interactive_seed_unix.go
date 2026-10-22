//go:build darwin || linux

package main

import (
	"fmt"
	"os"
	"syscall"
	"time"
)

func interactiveClockSeed() (uint64, error) {
	var usage syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &usage); err != nil {
		return 0, fmt.Errorf("read process CPU clock: %w", err)
	}
	// POSIX clock() and getrusage describe process user+system CPU time.
	// Linux and Darwin CLOCKS_PER_SEC is 1,000,000.
	clock := uint64(usage.Utime.Sec+usage.Stime.Sec)*1000000 + uint64(usage.Utime.Usec+usage.Stime.Usec)
	return uint64(time.Now().Unix()) ^ uint64(os.Getpid())<<32 ^ clock, nil
}
