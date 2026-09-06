package main

import (
	"time"

	"golang.org/x/sys/unix"
)

// uptimeNS reads the clock gputrace calls cpu_uptime_ns. Linux has no
// CLOCK_UPTIME_RAW; CLOCK_MONOTONIC is the clock that counts from boot
// and excludes suspend, which is what the name describes.
func uptimeNS() uint64 {
	var ts unix.Timespec
	if err := unix.ClockGettime(unix.CLOCK_MONOTONIC, &ts); err != nil {
		return 0
	}
	return uint64(ts.Sec)*1e9 + uint64(ts.Nsec)
}

// unixNS reads CLOCK_REALTIME, the domain gpuevent spans declare.
func unixNS() uint64 { return uint64(time.Now().UnixNano()) }
