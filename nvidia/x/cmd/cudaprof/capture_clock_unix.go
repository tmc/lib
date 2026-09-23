//go:build unix

package main

import "golang.org/x/sys/unix"

// monotonicNanos reads CLOCK_MONOTONIC, the clock the agent stamps its
// records with. It is system-wide, so a reading here and a record's
// timestamp from the target are on one timeline.
func monotonicNanos() (uint64, bool) {
	var ts unix.Timespec
	if err := unix.ClockGettime(unix.CLOCK_MONOTONIC, &ts); err != nil {
		return 0, false
	}
	return uint64(ts.Nano()), true
}
